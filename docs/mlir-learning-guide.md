# MLIR 学习文档：核心机制 + Dialect Conversion / Linalg 深入

> **本文档的定位**
> - 基于 **MLIR 官方文档主干**（LangRef、DialectConversion、Interfaces、PatternRewriter、Bufferization、PassManagement、DefiningDialects/Operations、Linalg Rationale）蒸馏，讲的是**今天的 MLIR 实际长什么样**。
> - 与 [`paper-notes/03-mlir.md`](./paper-notes/03-mlir.md) 分工明确：那篇是 **2020 论文**笔记，讲"为什么需要多层 IR、为什么不能一步跳到 LLVM"。本文讲**结构与机制**，尤其是论文只点到、官方文档才写透的 **Dialect Conversion**、**OpInterface**、**One-Shot Bufferize**。
> - 服务目标：读完能独立扩展 [`mlir-toy-dialect/`](../mlir-toy-dialect/)，并读懂 IREE 的 `linalg → flow → stream → hal` 流水线在 MLIR 层发生了什么。下游机器码见 [`llvm-learning-guide.md`](./llvm-learning-guide.md)；工业运行时见 [`iree-learning-guide.md`](./iree-learning-guide.md)。
> - **先修**：[`ai-compiler-foundations.md`](./ai-compiler-foundations.md) §5.1（tensor vs buffer）、§6（SSA / 多层 IR / Pass）。
> - **邻接动手**：图级划分/委托见 [`../onnx-delegate-lab/`](../onnx-delegate-lab/)；调度搜索见 [`../tvm-fatbin-lab/`](../tvm-fatbin-lab/)。
>
> **主要信息源**
> - LangRef：https://mlir.llvm.org/docs/LangRef/
> - Dialect Conversion：https://mlir.llvm.org/docs/DialectConversion/
> - Interfaces / Traits：https://mlir.llvm.org/docs/Interfaces/ · https://mlir.llvm.org/docs/Traits/
> - Pattern Rewriter：https://mlir.llvm.org/docs/PatternRewriter/
> - Bufferization：https://mlir.llvm.org/docs/Bufferization/
> - Pass Management：https://mlir.llvm.org/docs/PassManagement/
> - Defining Dialects / Operations：https://mlir.llvm.org/docs/DefiningDialects/
> - Linalg Rationale：https://mlir.llvm.org/docs/Rationale/RationaleLinalgDialect/
>
> **一句话读法**：如果只有两小时，读[第 1 章总图](#第-1-章-mlir-是什么渐进式-lowering-的坐标系)、[第 4 章 Interfaces](#第-4-章-traits-vs-interfaces重点)、[第 6 章 Dialect Conversion](#第-6-章-dialect-conversion重点)、[第 8 章 Linalg + Bufferization](#第-8-章-linalg--bufferization重点)。

---

## 目录

- [第 1 章 MLIR 是什么：渐进式 lowering 的坐标系](#第-1-章-mlir-是什么渐进式-lowering-的坐标系)
- [第 2 章 IR 数据结构：Operation ⊃ Region ⊃ Block](#第-2-章-ir-数据结构operation--region--block)
- [第 3 章 Dialect + ODS/TableGen：怎么定义 Op](#第-3-章-dialect--odstablegen怎么定义-op)
- [**第 4 章 Traits vs Interfaces**](#第-4-章-traits-vs-interfaces重点)
- [第 5 章 Pattern Rewriting + Pass](#第-5-章-pattern-rewriting--pass)
- [**第 6 章 Dialect Conversion**](#第-6-章-dialect-conversion重点)
- [第 7 章 内置 Dialect 地图](#第-7-章-内置-dialect-地图)
- [**第 8 章 Linalg + Bufferization**](#第-8-章-linalg--bufferization重点)
- [第 9 章 工具链与调试](#第-9-章-工具链与调试)
- [第 10 章 与 AI-Infra / IREE / toy 项目的接缝](#第-10-章-与-ai-infra--iree--toy-项目的接缝)
- [第 11 章 学习路径：最小必要集与动手清单](#第-11-章-学习路径最小必要集与动手清单)
- [附录：一页速查](#附录一页速查)

---

## 第 1 章 MLIR 是什么：渐进式 lowering 的坐标系

**MLIR = Multi-Level Intermediate Representation**，一套**可扩展的编译器基础设施**：同一套宿主机制（Operation / Region / Type / Attribute）承载多层方言（dialect），用渐进式 lowering 把高层领域语义一步步降到可执行表示。

一句话记住它和 LLVM 的关系：

> LLVM 给了你**一层**固定抽象（"带向量的 C"）+ 世界级后端；  
> **MLIR 给了你在到达那一层之前，还能保留 N 层领域语义的能力**——并且每一层都能做优化、验证、调试。

### 1.1 为什么不能一步跳到 LLVM IR

把模型或 DSL 直接降成 LLVM IR，会立刻丢掉后续优化最需要的信息：

| 高层信息 | 在 LLVM IR 里变成什么 | 后果 |
|---------|---------------------|------|
| 张量形状 / 布局 | 扁平 `ptr` + 手动 GEP | 融合、tiling 要重新做脆弱的模式匹配 |
| 循环嵌套结构 | preheader/header/latch 的 CFG | 分块、向量化前要先"raise"结构 |
| 算子语义（matmul / conv） | 三层嵌套 for + fmuladd | 无法直接调库或做算子级变换 |
| 设备亲和 / 异步时间线 | 没有对应物 | 调度决策必须另起炉灶 |

论文动机见 [`paper-notes/03-mlir.md`](./paper-notes/03-mlir.md) §1；本文只记结论：**过早 lowering = 语义丢失 = 后续要么猜（模式匹配），要么放弃（不做高层优化）**。

### 1.2 渐进式 lowering 的典型流水线

```
前端图 IR：  torch.* / onnx.* / stablehlo.* / tosa.*
                    │  保留框架语义 → 转结构化计算
                    ▼
结构化计算： linalg.*（+ tensor.*）··········· tile / fuse / 仍是 linalg.matmul
                    │  bufferization（tensor → memref）
                    ▼
循环与控制流： scf.* / affine.* + memref.* ····· 显式循环，仍是 Region 嵌套
                    │  向量化 / 拍扁 CFG
                    ▼
向量与标量：  vector.* / arith.* / func.* / cf.*
                    │  Dialect Conversion
                    ▼
LLVM Dialect： llvm.*  ······················ MLIR 内对 LLVM IR 的忠实映射
                    │  mlir-translate --mlir-to-llvmir
                    ▼
LLVM IR  ──▶  后端七阶段（见 llvm-learning-guide）──▶ 机器码
```

每一层只跨越一小步抽象距离；**多个 dialect 的 op 可以在同一份 IR 里共存混合**——不必整模块统一降完才能进下一层。这就是 Partial Conversion 存在的理由（第 6 章）。

### 1.3 与 LLVM 内部多层的对照

[`llvm-learning-guide.md`](./llvm-learning-guide.md) 第 1 章讲 LLVM 内部也有四层（LLVM IR → SelectionDAG → MachineIR → MC）。思想一致，差别在：

| | LLVM 内部多层 | MLIR 多层 dialect |
|--|--------------|-------------------|
| 层数 | 固定、不可扩展 | 用户可任意增加 |
| 扩展方式 | 改 LLVM 核心 / 加 target | 定义新 dialect + conversion |
| 混合 | 同一时刻通常只在一层 | 高层与低层 op 可共存 |
| 典型用途 | 目标相关 CodeGen | 领域语义保留 + 渐进下降 |

> **关键认知**：MLIR 不是"又一个 IR"，而是**让你廉价地发明下一层 IR，并把它接到已有层上**的基础设施。`mlir-toy-dialect` 里的 `toy` / `low` 就是这个思想的最小演示。

---

## 第 2 章 IR 数据结构：Operation ⊃ Region ⊃ Block

### 2.1 核心嵌套关系（必须能默画）

```
Operation（唯一的语义单元）
├── OpName            —— "dialect.opname"，如 "scf.for"、"toy.add"
├── Operands[]        —— 输入的 SSA Value
├── Results[]         —— 输出的 SSA Value（同样遵守 SSA）
├── Attributes        —— 编译期常量 key-value（字典）
├── Regions[]         —— ★嵌套结构的来源：0..N 个 Region
│     └── Region
│           └── Block[]
│                 ├── BlockArguments   —— 替代传统 φ 节点
│                 ├── Operation[]      —— 可再嵌套 Region（递归）
│                 └── Terminator Op    —— 块必须以终结符结尾
├── Successors        —— 控制流后继 Block（分支/循环用）
└── Location          —— 源码/变换可追溯位置
```

**要点列表**（先记结论，下面展开）：

1. **一切都是 Operation**——包括 `builtin.module`、`func.func`；没有"特殊的内建指令集"。
2. **Region 的语义由所属 Op 定义**——`scf.for` 的 Region 是循环体；`scf.if` 的两个 Region 是 then/else。
3. **Block argument 替代 phi**——终结符把值放到后继 Block 的参数上（functional SSA）。
4. **Value 只有两种定义点**：Op 的 Result，或 Block 的 Argument。
5. **Type / Attribute 本身也可扩展**——自定义类型（如 `!toy.num`）是可扩展性的第二根支柱。

> **速记**：[notes/mlir-op-region-block.md](./notes/mlir-op-region-block.md) —— 语句→Op、作用域→Region、直线段→Block；顶层多为 `module`；Block≈Region 内基本块；自定义 dialect 只补领域层。

### 2.2 Value / Type / Attribute

| 概念 | 是什么 | 关键约束 |
|------|--------|---------|
| **Value** | SSA 值，定义一次、使用多次 | 类型固定；use-def 链是分析的基础 |
| **Type** | Value 的类型；可自定义 | 非依赖类型；挂在 Value / Op 结果上 |
| **Attribute** | 编译期常量（整数、字符串、AffineMap、数组…） | 不可变、可 uniqued；也可作为属性别名复用（`#map = ...`） |

Attribute 与 Type 的分工：Type 描述"运行时值长什么样"，Attribute 描述"编译期已知的静态信息"。例如 `arith.constant {value = 42 : i32}` 里，结果类型是 `i32`，常量本身是 Attribute。

#### 示例精讲：把一段 IR 逐字段拆成 Value / Type / Attribute

> 目的：拿到任何一行 MLIR，都能立刻指出「哪个是值、哪个是类型、哪个是编译期常量」。

```mlir
#map = affine_map<(d0, d1) -> (d0, d1)>        // ① 属性别名：一个 AffineMapAttr

func.func @demo(%arg0: i32) -> i32 {
  %c = arith.constant 42 : i32                 // ② 结果 %c；类型 i32；值 42 是 Attribute
  %s = arith.addi %arg0, %c : i32              // ③ 两个 Operand，一个 Result
  %t = arith.muli %s, %c : i32                 // ④ %c 被第二次使用（use-def 是图）
  return %t : i32
}
```

逐字段对号入座：

| 文本片段 | 是什么 | 说明 |
|----------|--------|------|
| `%arg0` | **Value**（BlockArgument） | 定义点是入口 Block 的参数，不是某个 op 的结果 |
| `%c` / `%s` / `%t` | **Value**（OpResult） | 定义点是各自那条 op |
| `i32` | **Type** | 挂在 Value 上，描述运行时长什么样 |
| `42` | **Attribute**（`IntegerAttr`） | 编译期就知道，存在 op 的属性字典里，**不是** Value |
| `#map` | **Attribute**（`AffineMapAttr`） | 属性可以起别名复用，本身不可变、uniqued |
| `arith.addi` | **Operation** | OpName = `dialect.opname` |

关键区别：`42` 写在 op 里是 Attribute，但 `%c` 是它**物化出来的 Value**。所以下面两句含义不同：

```mlir
%c = arith.constant 42 : i32     // 有一个 SSA 值 %c，可被多处使用
// 而属性 42 本身没有 SSA 名字，不能被别的 op 当操作数直接引用
```

`%c` 的 use-def 链（图，不是树）：

```text
%c ──use──▶ arith.addi (%arg0, %c)
   └─use──▶ arith.muli (%s,   %c)
```

> **自测**：把 `%c` 删掉但保留 `arith.addi %arg0, %c`，verifier 会报什么类别的错——类型错还是定义缺失？

> **速记**：[notes/mlir-type-attr-interface.md](./notes/mlir-type-attr-interface.md) —— Type≠必须内存布局；Attribute 值种类内置、key 由 Op 约定；Interface≠fold，是跨 dialect 能力契约。

### 2.3 Block arguments 为何替代 phi

LLVM IR：

```llvm
%y = phi i32 [ %a, %bb1 ], [ %b, %bb2 ]
```

MLIR 等价形态：

```mlir
^bb3(%y: i32):          // 后继块用参数接收
  ...
// 前驱块的终结符把值"传进去"：
cf.br ^bb3(%a : i32)     // 来自 bb1
cf.br ^bb3(%b : i32)     // 来自 bb2
```

**为什么更好**：

- phi 节点在概念上"不属于任何一个前驱"，对变换不友好；block argument 把汇合点固定在块入口。
- 嵌套 Region 时，外层值通过 Region 参数或捕获进入内层，模型一致。
- Verifier 更容易检查：每个前驱传给后继的参数个数/类型必须匹配。

#### 示例精讲：`max0` —— 同一个汇合点，phi 版与 block argument 版

> 目的：把「汇合点接收不同前驱的值」这件事，在两套 IR 里各看一遍，并能默画对象树。

源语义：`int max0(int x) { return x > 0 ? x : 0; }`

**LLVM IR（phi 版）**

```llvm
define i32 @max0(i32 %x) {
entry:
  %cmp = icmp sgt i32 %x, 0
  br i1 %cmp, label %pos, label %neg
pos:
  br label %exit
neg:
  br label %exit
exit:
  %r = phi i32 [ %x, %pos ], [ 0, %neg ]   ; ← 汇合：按"从哪来"选值
  ret i32 %r
}
```

**MLIR（block argument 版）**

```mlir
func.func @max0(%x: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %cmp = arith.cmpi sgt, %x, %c0 : i32
  cf.cond_br %cmp, ^exit(%x : i32), ^exit(%c0 : i32)   // ← 由前驱"把值传进去"
^exit(%r: i32):
  return %r : i32
}
```

两者语义相同，差别只在**谁负责表达这次汇合**：

| | LLVM phi | MLIR block argument |
|--|----------|---------------------|
| 汇合写在哪 | 后继块开头的 `phi`，列出 `(值, 来自哪个前驱)` | 后继块的**参数列表** `^exit(%r: i32)` |
| 谁提供值 | phi 自己去"回看"前驱 | **前驱的终结符**在跳转时传参 `^exit(%x : i32)` |
| 加一个前驱要改哪 | 改 phi 的 incoming 列表 | 改新前驱的终结符即可 |
| verifier 检查 | incoming 数与前驱数一致 | 每个前驱传的参数个数/类型与块参数匹配 |

MLIR 侧的对象树（Pass 拿到 `func.func` 时看到的就是这棵树）：

```text
func.func @max0                  ← Operation
└─ Region
   ├─ Block ^entry  args: %x
   │    ├─ arith.constant
   │    ├─ arith.cmpi
   │    └─ cf.cond_br            ← Terminator，带两个后继 + 各自实参
   └─ Block ^exit   args: %r     ← 汇合点：值从这里"进入"
        └─ func.return
```

> **自测**：若把 `cf.cond_br` 的一侧实参从 `%c0` 改成一个 `f32` 值，报错会发生在 verifier 的哪一条规则上？

读 [`llvm-learning-guide.md`](./llvm-learning-guide.md) §2.3 时，记住这句话：**MLIR 用 block argument 改进了 LLVM 的 phi 表示，语义等价、结构更干净**。

> **速记**：[notes/mlir-block-arg-ssa.md](./notes/mlir-block-arg-ssa.md) —— phi/block-arg 是数据流汇合（到达定值），不是分支；≠ 活跃变量；SSA 值在构造时对应源变量/`alloca`。

### 2.4 两种 Region 语义：SSA CFG vs Graph

官方 LangRef 区分 Region 的**种类**：

| 种类 | 特征 | 典型用途 |
|------|------|---------|
| **SSACFG Region** | Block 形成 CFG；必须有 terminator；Value 遵守 dominance | `func.func`、`scf.for`、`cf.br` 所在的控制流 |
| **Graph Region** | 允许任意 use-def（可有环、可前向引用）；用于数据流图 | 部分图 IR / 硬件描述场景 |

对 AI 编译主线，你几乎整天都在 SSACFG Region 里工作。Graph Region 知道存在即可——遇到 CIRCT 或某些图 dialect 时再深挖。

### 2.5 与 toy 项目的对照

在 [`mlir-toy-dialect/`](../mlir-toy-dialect/) 里：

```
builtin.module                    ← 最外层 Operation + SymbolTable
  └── func.func                   ← 带 1 个 Region（函数体）
        └── ^bb0(...):            ← 入口 Block（参数 = 函数形参）
              toy.repeat {        ← Operation，带 1 个 Region
                ^bb0:
                  ...
                  toy.yield ...   ← Terminator
              }
              func.return
```

跑 `test/region.mlir`，确认 Pass 会**自动递归进入 Region**——这是嵌套 IR 的工程后果：你写的 function pass 默认会看到循环体内的 op。

---

## 第 3 章 Dialect + ODS/TableGen：怎么定义 Op

### 3.1 Dialect 是什么

**Dialect = Op + Type + Attribute + Interface 的命名空间与注册单元**。它本身不引入新语义，只提供：

- 文本前缀（`toy.`、`linalg.`、`func.`）
- dialect 级钩子（如常量物化 `materializeConstant`）
- 与 Pass / Conversion 注册的挂载点

关键事实：

- **MLIR 几乎没有"特权内建 op"**——`module`、`func.func` 也是普通 op。
- 一个 Op 只属于一个 dialect，但**不同 dialect 的 op 可在同一 IR 中混合**。
- 扩展三支柱：**自定义 Op、自定义 Type、自定义 Attribute**（Interface 是第四根"复用支柱"，见第 4 章）。

> **速记**：[notes/mlir-op-region-block.md](./notes/mlir-op-region-block.md) —— 用排除法看 dialect 范围：标准 dialect 管通用脚手架，自定义只补领域 op/type。

### 3.2 ODS 定义 Op 的最小字段

用 TableGen（与 LLVM 同一套语言）写 `.td`，由 `mlir-tblgen` 生成 C++ 样板。一份典型 Op 定义包含：

```tablegen
def Toy_AddOp : Toy_Op<"add", [Pure, Commutative]> {
  let summary = "element-wise add";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasVerifier = 1;           // 生成 verify() 声明，你实现约束
  let hasFolder = 1;             // 生成 fold() 声明
  let hasCanonicalizer = 1;      // 生成 getCanonicalizationPatterns()
}
```

| 字段 / 机制 | 作用 |
|------------|------|
| `arguments` / `results` | 声明操作数与结果（含约束：`AnyType`、`I32`、`TensorOf<...>`） |
| `assemblyFormat` | 声明式打印/解析；覆盖 95% 场景，不必手写 parser |
| `traits` 列表 | 如 `Pure`、`Commutative`、`Terminator`、`SingleBlock` |
| `hasVerifier` | 结构之外的语义检查（如 `count > 0`） |
| `hasFolder` / `hasCanonicalizer` | 接入通用 canonicalize 流水线 |
| `regions` | 声明嵌套 Region（个数、是否 SingleBlock 等） |

### 3.3 从 `.td` 到可用 C++ 的链路

```
ToyOps.td  ──mlir-tblgen──▶  ToyOps.h.inc / ToyOps.cpp.inc
                                 │
ToyOps.h  #include 生成头  ◀─────┘
ToyOps.cpp  实现 fold / verify / interface 方法
ToyDialect.cpp  addOperations / addTypes / materializeConstant
```

在 toy 项目中对照：

| 文件 | 职责 |
|------|------|
| `include/Toy/ToyOps.td` | 声明 `toy.add/mul/repeat/...` |
| `include/Toy/ToyTypes.td` | 声明 `!toy.num` |
| `include/Toy/ToyInterfaces.td` | 声明 `ToyCostOpInterface` |
| `lib/ToyOps.cpp` | `fold`、`verify`、`getCost` |
| `lib/ConvertToyToLow.cpp` | 用 Dialect Conversion 降低 |

**掌握标准**（对应根 README §3.2）：能独立写出带 `arguments/results/assemblyFormat/traits/hasVerifier` 的 `.td`，并接上 verifier / folder。

### 3.4 Trait 在 ODS 里的第一印象

Traits 在 `.td` 的方括号里列出，编译期附着到 Op 类上。常见内置：

| Trait | 含义 | 谁消费 |
|-------|------|--------|
| `Pure` | 无内存副作用、无控制流副作用 | DCE、CSE、移动代码 |
| `Commutative` | 操作数可交换 | canonicalize |
| `ConstantLike` | 像常量一样可物化 | folding / materializeConstant |
| `Terminator` | 必须是 Block 最后一条 | Verifier |
| `IsolatedFromAbove` | 内层不捕获外层 SSA（除显式参数） | 并行 Pass、符号隔离 |
| `SingleBlock` | Region 恰好一个 Block | Region 结构检查 |

Traits 是"是/否"或轻量行为标记；需要**方法契约**时用 Interface（下一章）。

> **速记**：[notes/mlir-trait-vs-interface.md](./notes/mlir-trait-vs-interface.md) —— Trait 可自定义但是标签；Interface 才是打分/重写；候选是遍历窗口而非全体一次处理；正确性靠逐步保义而非看完全局。

---

## 第 4 章 Traits vs Interfaces（重点）

### 4.1 核心问题：开放 Op 集合上如何写通用 Pass？

当任何人都能加新 op 时，通用 Pass（DCE、Inliner、Bufferize、代价模型）不可能 `switch` 遍所有 op 名。MLIR 的答案分两层：

```
Trait     = 编译期附着的"标签 / 轻量行为"（有没有某性质）
Interface = 运行时可查询的"能力契约"（能不能做某事、怎么做）
```

> **速记**：[notes/mlir-type-attr-interface.md](./notes/mlir-type-attr-interface.md) —— Interface 让 Pass 只认契约不认 op 名；与 `fold`/canonicalize 不是一回事。  
> **速记**：[notes/mlir-trait-vs-interface.md](./notes/mlir-trait-vs-interface.md) —— 融合例子看 Trait/Interface 能力边界；候选窗口与「逐步保义」如何保证正确。

| | Trait | Interface |
|--|-------|-----------|
| 表达力 | 主要是布尔/静态属性 | 虚方法：查询 + 变换钩子 |
| 绑定时机 | ODS 声明时钉死在 Op 类上 | Op 类实现，或 **External Model** 外挂 |
| 典型消费者 | CSE/DCE（看 `Pure`） | Inliner / Bufferize / 循环变换 |
| 类比 | Java 的 marker annotation | Java 的 interface |

### 4.2 为什么 Interfaces 能让 Pass 跨 dialect 工作

```
        ┌─────────────────────────────────────┐
        │  通用 Pass：ToyPrintCost            │
        │  只认识 ToyCostOpInterface          │
        └──────────────┬──────────────────────┘
                       │ dyn_cast<ToyCostOpInterface>(op)
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     toy.add        low.mul      low.shl
   (实现接口)     (实现接口)   (实现接口)
```

Pass **不 `#include` 任何具体 dialect 头文件**也能工作——只要 op 实现了接口。这是 [`mlir-toy-dialect`](../mlir-toy-dialect/) 里 `--toy-print-cost` 的全部意义：`ToyCostPass.cpp` 是教学版的"跨 dialect 通用 Pass"。

工业界同构例子：

- `BufferizableOpInterface` → One-Shot Bufferize 不认识 `linalg` / `tensor` 的具体类名，只问接口。
- `LoopLikeOpInterface` → 循环变换 Pass 同时服务 `scf.for`、`affine.for`、自定义循环 op。
- `MemoryEffectsOpInterface` → 副作用分析驱动 DCE / 调度。

### 4.3 DialectInterface vs OpInterface

| 种类 | 挂在谁身上 | 典型用途 |
|------|-----------|---------|
| **OpInterface** | 单个 Operation | 该 op 如何 bufferize、是否像循环、内存效应 |
| **DialectInterface** | 整个 Dialect | 该 dialect 如何参与 inlining、常量物化、解码 |

常见 **DialectInterface**：`DialectInlinerInterface`（方言级决定"能不能内联进/出"）、dialect 级的常量物化钩子。  
常见 **OpInterface**：下面 4.4 列出的四个，几乎是 AI 编译每天都会碰到的。

> **速记**：[notes/mlir-inlining.md](./notes/mlir-inlining.md) —— 内联=调用点展开函数体（≠`#include`）；`DialectInlinerInterface` 决定本方言如何参与通用 Inliner。

### 4.4 四个必知 OpInterface

#### MemoryEffectsOpInterface

描述 op 对内存的读/写/分配/释放效应。通用副作用分析、DCE、某些调度依赖它。`Pure` trait 可视为"无效应"的快捷标记；更细的效应用接口列出。

#### LoopLikeOpInterface

统一"像循环的东西"：归纳变量、上下界、步长、循环体 Region。这样 tiling / 向量化前置分析不必为每种循环 dialect 写一份。

#### DestinationStyleOpInterface（DPS）

声明 op 的哪些操作数是 **destination（`outs`）**——与结果一一对应的"被更新的初始值"。这是 One-Shot Bufferize 的锚点（第 8 章）。典型：`linalg.generic` / `linalg.matmul` 的 `outs` 操作数。

#### BufferizableOpInterface

One-Shot Bufferize 的扩展点：每个可 bufferize 的 op 实现：

- 分析钩子：别名关系、是否可 in-place、读写哪些 tensor 操作数；
- 重写钩子：如何把 tensor 版 op 改成 memref 版。

实现常常放在 **External Model**（另一编译单元），避免 dialect 定义与 transforms 循环依赖。

#### 示例精讲：四个接口各看一眼「它让通用 Pass 免于认 op 名」

> 目的：每个接口配一段最小 IR + 一句「通用 Pass 因此能做什么」。

**① MemoryEffectsOpInterface —— 能不能删**

```mlir
%s = arith.addi %a, %b : i32          // Pure：无副作用
memref.store %s, %buf[%i] : memref<8xi32>   // 有 Write 效应
```

通用 DCE 的判断逻辑（伪代码）——它不认识 `arith.addi`，只问效应：

```cpp
if (isMemoryEffectFree(op) && op->use_empty())
  rewriter.eraseOp(op);      // %s 若没人用可删；store 有 Write 效应，不能删
```

**② LoopLikeOpInterface —— 是不是循环**

```mlir
scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (i32) {
  %n = arith.addi %acc, %i : i32
  scf.yield %n : i32
}
```

接口把「归纳变量 / 上下界 / 步长 / 循环体 Region」统一暴露出来，于是 LICM、tiling 前置分析对 `scf.for`、`affine.for`、你自己的循环 op **同一套代码**即可。

**③ DestinationStyleOpInterface（DPS）—— 结果写到哪份初值上**

```mlir
%0 = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x4xf32>)
                   outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>
```

`outs(%c)` 就是 destination：结果 `%0` 与 `%c` 同形状。bufferize 时若无冲突，`%0` 可直接复用 `%c` 的 buffer（in-place），否则插 copy——见 §8.6。

**④ BufferizableOpInterface —— 怎么从 tensor 变 memref**

它是一组问答钩子，One-Shot Bufferize 逐个 OpOperand 去问：

| 钩子回答什么 | 例子（对 `linalg.matmul` 的 `outs`） |
|--------------|--------------------------------------|
| 这个操作数会被读吗 | 是（累加初值要读） |
| 这个操作数会被写吗 | 是 |
| 结果与哪个操作数等价/别名 | 结果 ↔ `outs` |
| 重写成什么 memref 形态 | `linalg.matmul` on memref |

**共同点**：以上四个 Pass 都写成「walk + `dyn_cast<接口>`」，没有一处 `if (op->getName() == "linalg.matmul")`：

```cpp
getOperation()->walk([&](Operation *op) {
  if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) { /* 只认契约 */ }
});
```

toy 项目里的 `ToyCostOpInterface` + `--toy-print-cost` 就是这套模式的教学版（见 §4.2）。

> **自测**：把一个自定义循环 op 接上 `LoopLikeOpInterface` 之后，你**不需要**改动哪一类代码就能享受已有的循环变换？

### 4.5 论文四种策略 → 工程落点

[`paper-notes/03-mlir.md`](./paper-notes/03-mlir.md) §3.5 的四种策略，对照今天的代码：

| 策略 | 工程落点 |
|------|---------|
| ① 基础 Trait | `Pure` / `Commutative` → CSE/DCE |
| ② 特权钩子 | `fold()` / `getCanonicalizationPatterns()` |
| ③ Optimization Interface | OpInterface / DialectInterface |
| ④ dialect 专属 Pass | 直接写死语义的 Pass（最后手段） |

**学习建议**：先把 toy 的 `ToyCostOpInterface` 吃透，再读 `BufferizableOpInterface` 的头文件——两者是同一设计模式的教学版与生产版。

---

## 第 5 章 Pattern Rewriting + Pass

### 5.1 RewritePattern 是什么

把"匹配一个子图 → 替换成另一个子图"抽象为可复用对象：

```cpp
struct SimplifyMulOne : public OpRewritePattern<toy::MulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(toy::MulOp op,
                                PatternRewriter &rewriter) const override {
    // x * 1 → x
    ...
    rewriter.replaceOp(op, {x});
    return success();
  }
};
```

多个 Pattern 放进 `RewritePatternSet`，交给驱动器执行。

#### 示例精讲：`x*1 → x` 与 `x+0 → x` 的一次完整重写

> 目的：看清「根 op → 沿 use-def 查条件 → replaceOp → 驱动器再来一轮」这条链，并理解不动点。
> 可直接跑：`toy-opt test/simplify.mlir --toy-simplify`（见 [`mlir-toy-dialect/`](../mlir-toy-dialect/)）。

**输入 IR**（把两条规则串起来，便于观察多轮迭代）

```mlir
func.func @chain(%arg0: i32) -> i32 {
  %one  = toy.constant 1 : i32
  %zero = toy.constant 0 : i32
  %m = toy.mul %arg0, %one : i32     // 可被规则 1 命中
  %a = toy.add %m, %zero : i32       // 规则 2 要等 %m 被替换后才"看得清"
  return %a : i32
}
```

**规则 1 在干什么**（`lib/ToyPasses.cpp` 的 `SimplifyMulByOne`，节选）

```cpp
LogicalResult matchAndRewrite(MulOp op, PatternRewriter &rewriter) const override {
  // 根 = toy.mul；沿 use-def 向上看操作数的定义 op
  if (isConstantWithValue(op.getRhs(), 1)) {     // getDefiningOp<ConstantOp>() + 值==1
    rewriter.replaceOp(op, op.getLhs());         // 把 %m 的所有使用者改成 %arg0
    return success();
  }
  return failure();                              // 不匹配就交给别的规则
}
```

**驱动器的两轮迭代**

```text
第 1 轮：访问 toy.mul → 规则 1 命中 → %m 的用户全部改指向 %arg0
         此时 IR 变成： %a = toy.add %arg0, %zero
第 2 轮：访问 toy.add → 规则 2 命中（rhs 是常量 0）→ %a 的用户改指向 %arg0
         %one / %zero 失去所有使用者 → 驱动器顺带做 DCE 清掉
第 3 轮：没有规则再命中 → 到达不动点，结束
```

**输出 IR**

```mlir
func.func @chain(%arg0: i32) -> i32 {
  return %arg0 : i32
}
```

三个容易记混的点：

| 直觉 | 实际 |
|------|------|
| pattern 会扫全图找子图 | 只以**当前 op 为根**，再沿 use-def 查有限几步 |
| 一条规则要一次改到位 | 多条规则**互相喂**，靠驱动器反复迭代收敛 |
| 常量要自己删 | `applyPatternsAndFoldGreedily` 会顺带 fold + 清死代码 |

> **自测**：若把输入里的 `%zero` 换成函数参数（不再是常量），第 2 轮会发生什么？最终 IR 里还剩几条 op？

> **速记**：[notes/mlir-pattern-rewriting.md](./notes/mlir-pattern-rewriting.md) —— 机制而非专属优化；子图=根+use-def 约束；宏观靠流水线/高层 op；Conversion 也用 pattern 外加 Target/TypeConverter。

### 5.2 贪心驱动器（Greedy Pattern Rewrite Driver）

入口：`applyPatternsAndFoldGreedily`（以及相关变体）。

行为直觉：

1. 在工作列表上迭代 op；
2. 命中 pattern 就重写；
3. 同时尝试 `fold`；
4. 直到没有 pattern 能再应用（不动点）或达到迭代上限。

| 特性 | 含义 |
|------|------|
| 收敛条件 | "没有规则能再命中" |
| 漏写规则 | **悄悄留下**非法/未降的 op——可能不报错 |
| 类型变化 | 困难：没有 TypeConverter，签名/block arg 要自己对齐 |
| 典型用途 | canonicalize、代数化简、同类型 peephole |

toy 对照：`--toy-simplify`、`--toy-to-low`（贪心版）走这条路。

### 5.3 fold vs canonicalize

| 机制 | 触发点 | 典型行为 |
|------|--------|---------|
| **`fold()`** | 构建时 / 贪心驱动中 | 返回 Attribute（算出新常量）或 Value（复用已有 SSA） |
| **Canonicalization patterns** | `--canonicalize` Pass | 代数化简、规范化形态（如 `x-x→0`、交换律排序） |
| **Constant materializer** | dialect 钩子 | 把 fold 出的 Attribute 再变成 `dialect.constant` op |

toy 里两种 fold 都有：`add/mul` 返回 Attribute；`unbox` 返回 Value（见 `ToyOps.cpp`）。

#### 示例精讲：同一段 IR，`fold` 与 pattern 各消掉了什么

> 目的：分清「算值」（fold）与「换结构」（pattern），并看到 materializer 在中间补的那一步。
> 可直接跑：`toy-opt test/canonicalize.mlir --canonicalize`。

**输入**

```mlir
func.func @mix(%arg0: i32) -> i32 {
  %0 = toy.constant 10 : i32
  %1 = toy.constant 20 : i32
  %2 = toy.add %0, %1 : i32        // ① 两个操作数都是常量 → fold 能算出 30
  %3 = toy.constant 1 : i32
  %4 = toy.mul %arg0, %3 : i32     // ② 含未知值 %arg0 → fold 算不动，靠 pattern
  %5 = toy.add %2, %4 : i32
  return %5 : i32
}
```

**三种机制各自负责哪一句**

| 机制 | 命中哪句 | 产出 | 为什么是它 |
|------|----------|------|-----------|
| `fold()` | ① `toy.add 10, 20` | **Attribute** `30` | 操作数都是编译期常量，可直接算 |
| Constant materializer | 承接上一步 | `toy.constant 30 : i32` | fold 只给 Attribute，要有人把它变回一条 op |
| Canonicalization pattern | ② `toy.mul %arg0, 1` | 直接复用 `%arg0` | 结构化简，不涉及算值 |

**`--canonicalize` 之后**

```mlir
func.func @mix(%arg0: i32) -> i32 {
  %c30 = toy.constant 30 : i32          // ← fold 出 Attribute 30，再由 materializer 物化
  %r = toy.add %c30, %arg0 : i32        // ← mul 已被换成 %arg0
  return %r : i32
}
```

一句话分工：

> **fold 回答「这条 op 的结果是不是已经能算/已经存在」**（返回 Attribute 或已有 Value）；  
> **pattern 回答「这块结构能不能换成更好的结构」**；  
> **materializer 只在 fold 给出 Attribute 时补一条常量 op**。

> **自测**：若只实现了 `fold()` 却忘了给 dialect 实现 `materializeConstant`，①那句会发生什么？



### 5.4 PassManager

MLIR 的 Pass **挂在任意 Op 类型上**，不是写死 module/function/loop 三层：

```
builtin.module(
  func.func(
    canonicalize,
    cse,
    my-pass
  )
)
```

关键点：

- **嵌套 pipeline 字符串**：`mlir-opt --pass-pipeline='...'`
- **IsolatedFromAbove**：可并行跑多个此类 op 的 Pass（use-def 不跨界）
- **分析失效**：变换 Pass 必须声明保留了哪些分析（与 LLVM New PM 的 `PreservedAnalyses` 同构，见 [`llvm-learning-guide.md`](./llvm-learning-guide.md) 第 3 章）

> **速记**：[notes/llvm-mlir-pass-ir-unit.md](./notes/llvm-mlir-pass-ir-unit.md) —— Pass 传入的是解析后的 IR 根对象；LLVM 四层 vs MLIR 按 Op 锚定；细粒度在回调内用 API/walk，Function/Op 不是黑盒。

`toy-opt` ≈ LLVM 的 `opt`：注册 dialect + pass，读 `.mlir`，跑 pipeline。

### 5.5 贪心 lowering vs Dialect Conversion（预告）

| | 贪心 `--toy-to-low` | Dialect Conversion `--toy-to-low-convert` |
|--|---------------------|------------------------------------------|
| 入口 | `applyPatternsAndFoldGreedily` | `applyPartial/FullConversion` |
| "做完"的定义 | 没有 pattern 再命中 | 所有 **illegal** op 都合法化 |
| 漏规则 | 可能静默残留 | **失败并指出**哪个 op 未能合法化 |
| 换类型 | 基本不支持 | `TypeConverter` + materialization |
| 生产用法 | canonicalize / peephole | **几乎所有跨 dialect lowering** |

下一章把右列拆开讲透。

---

## 第 6 章 Dialect Conversion（重点★）

> 官方文档：https://mlir.llvm.org/docs/DialectConversion/  
> 这是跨 dialect lowering 的主武器，也是根 README §3.2 标 ★ 的必补项。

> **速记**：[notes/conversion-llvm-vs-mlir.md](./notes/conversion-llvm-vs-mlir.md) —— 与 LLVM IR「Conversion」指令类不是一回事：那边是值级 cast，这里是跨 dialect lowering。  
> **速记**：[notes/mlir-pattern-rewriting.md](./notes/mlir-pattern-rewriting.md) —— Conversion 的重写手段仍是 pattern（ConversionPattern）；Target/TypeConverter 管合法与类型。

### 6.1 框架三件套

```
┌──────────────────┐   ┌─────────────────────┐   ┌──────────────────┐
│ ConversionTarget │ + │ ConversionPattern[] │ + │ TypeConverter?   │
│ 什么算合法        │   │ 非法 → 合法的重写    │   │ 类型怎么映射      │
└──────────────────┘   └─────────────────────┘   └──────────────────┘
                              │
                              ▼
              applyPartial / Full / Analysis Conversion
```

### 6.2 三种模式

| 模式 | API | 行为 |
|------|-----|------|
| **Partial** | `applyPartialConversion` | 尽量合法化；**未标 illegal** 的预存 op 可保留 |
| **Full** | `applyFullConversion` | **全部**输入 op 必须合法化，否则失败 |
| **Analysis** | `applyAnalysisConversion` | 做一遍"假如转换"的分析，**不改 IR**；记录哪些能合法化 |

渐进式 lowering 的日常选择是 **Partial**：这一趟只消灭某一层 illegal op，其余 dialect 原样留下。

### 6.3 遍历顺序与自动链式合法化

官方保证：

> **框架按 preorder（前序）walk**：先看一个 op，再看它 Region 里的嵌套 op。

更关键的能力——**pattern 可以自动链式组合**：

```
目标只合法：foo.add
你提供 pattern： bar.add → baz.add
                 baz.add → foo.add
框架能自动推出： bar.add → … → foo.add
```

你不必手写每对 dialect 的直接边。这是"专家混合编译器"（Mixture of Experts）能拼 pipeline 的基础设施前提。

#### 示例精讲：两条规则拼出一条没人写过的降低路径

> 目的：看清「链式合法化」不是玄学，而是「合法化目标 + 逐步重写」自然导出的结果。

设有三层同语义的加法 op，你只写了**相邻两跳**的 pattern：

```text
已有 pattern：  hi.add  →  mid.add
已有 pattern：  mid.add →  low.add
Target 声明：   只有 low.* 合法
```

**输入**

```mlir
func.func @chain(%a: i32, %b: i32) -> i32 {
  %0 = hi.add %a, %b : i32
  return %0 : i32
}
```

**框架的推进过程**（preorder walk，每轮只要求"离合法更近一步"）

```text
步骤 1：访问 hi.add → 非法 → 命中 pattern → 变成 mid.add
步骤 2：新产生的 mid.add 重新入队 → 仍非法 → 命中第二条 pattern → 变成 low.add
步骤 3：low.add 合法 → 该分支结束
```

**输出**

```mlir
func.func @chain(%a: i32, %b: i32) -> i32 {
  %0 = low.add %a, %b : i32
  return %0 : i32
}
```

关键点：**你从没写过 `hi.add → low.add` 这条规则**，但因为「合法性」是终点声明、重写是逐步的，框架自己把两跳接了起来。

| 对比 | 贪心驱动器 | Dialect Conversion |
|------|-----------|-------------------|
| 何时停 | 没有规则再命中 | 所有 illegal op 都合法化 |
| 中途多出的 `mid.add` | 也会被继续改写（若有规则） | 同样继续，且**最终没降完会报错** |
| 少写一跳规则 | 可能静默留下 `mid.add` | 报 `failed to legalize operation 'mid.add'` |

> **自测**：若把 `mid.add → low.add` 那条规则删掉，Partial 模式下最终 IR 会变成什么？错误信息会指向哪个 op？

### 6.4 ConversionTarget：legal / illegal / dynamicallyLegal / recursive

| 标记 | 含义 |
|------|------|
| **Legal** | 该 op/dialect 的任何实例都合法 |
| **Illegal** | 必须被转换掉；残留则失败（Full）或必须处理（Partial 对 illegal） |
| **DynamicallyLegal** | 合法与否由回调决定（如"仅当操作数已是 memref"） |
| **Unknown**（未标记） | Partial 下可保留；Full 下通常不可接受 |
| **RecursivelyLegal** | op 本身合法 ⇒ 其嵌套 Region 内所有 op 也视为合法 |

```cpp
target.addIllegalDialect<toy::ToyDialect>();
target.addLegalDialect<low::LowDialect>();
target.addLegalOp<func::FuncOp, func::ReturnOp>();

// 动态合法：函数签名里不再出现 !toy.num 才合法
target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
  return converter.isSignatureLegal(op.getFunctionType());
});

// 递归合法：某容器 op 内部先别动
target.markOpRecursivelyLegal<MyKernelOp>();
```

`dynamicallyLegal` 是真实 pipeline 的日常工具：例如 `func.func` 在类型转换完成前是非法的，签名转完后变成合法——用动态条件表达"阶段性合法"。

#### 示例精讲：同一个 `func.func`，为什么先非法后合法

> 目的：理解「合法与否是一个**随转换进度变化的判断**」，而不是一次性打的静态标签。

**输入**（函数签名里还带着 `!toy.num`）

```mlir
func.func @sig(%n: !toy.num) -> !toy.num {
  %x = toy.unbox %n : !toy.num -> i32
  %c = toy.constant 4 : i32
  %m = toy.mul %x, %c : i32
  %r = toy.box %m : i32 -> !toy.num
  return %r : !toy.num
}
```

**Target 的三条标记，各自管什么**

```cpp
target.addIllegalDialect<toy::ToyDialect>();   // toy.* 一律要转掉
target.addLegalDialect<low::LowDialect>();     // low.* 随便用
target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
  return converter.isSignatureLegal(op.getFunctionType());   // 签名里没有 !toy.num 才算合法
});
```

**转换过程中的三个时刻**

| 时刻 | `func.func @sig` 是否合法 | 原因 |
|------|--------------------------|------|
| 开始时 | **非法** | 签名是 `(!toy.num) -> !toy.num`，动态回调返回 false |
| 签名转换 pattern 跑完后 | **合法** | 签名变成 `(i32) -> i32` |
| 结束时 | 合法 | 函数体内 `toy.*` 也已全部换成 `low.*` |

**结果**

```mlir
func.func @sig(%arg0: i32) -> i32 {
  %c = low.constant 4 : i32
  %m = low.mul %arg0, %c : i32
  return %m : i32
}
```

对照另外两种标记会怎样：

| 换成 | 结果 |
|------|------|
| `addLegalOp<func::FuncOp>()`（静态合法） | 函数体被转了，但**签名永远不改**，最终留下类型不一致的 IR |
| `markOpRecursivelyLegal<...>()` | 该 op **连同 Region 内部**都被跳过，内层 `toy.*` 不会被降低 |

> **自测**：如果忘了写 `addDynamicallyLegalOp<func::FuncOp>`，`applyPartialConversion` 大概率在哪一步报错，报的是「无法合法化」还是「类型不匹配」？

### 6.5 ConversionPattern 与 remapped operands

`ConversionPattern` 比普通 `RewritePattern` 多一个关键参数：**已经 remap 过的操作数**。

```cpp
LogicalResult matchAndRewrite(Operation *op,
                              ArrayRef<Value> operands,  // ★ remapped
                              ConversionPatternRewriter &rewriter) const;
```

| 你问 | 得到什么 |
|------|---------|
| `op->getOperand(i)` | **原始**类型的 SSA（可能仍是旧类型） |
| `operands[i]` / adaptor | **最近替换后**的值；若 pattern 绑了 TypeConverter，类型已是合法化类型 |

类型变化时，驱动器会（概念上）插入 `builtin.unrealized_conversion_cast` 把新旧 IR 接上，避免在转换中途破坏类型契约。转换成功结束后，多余的 cast 应被消掉；**若还残留，说明 lowering 没做干净**——这是极有用的错误信号。

#### 示例精讲：`operands[i]` 和 `op->getOperand(i)` 到底差在哪

> 目的：用一段真实会发生类型变化的 IR，看清「原始操作数」与「已 remap 操作数」的区别。
> 可直接跑：`toy-opt test/convert.mlir --toy-to-low-convert`。

**输入**（`!toy.num` 会被 TypeConverter 映射成 `i32`）

```mlir
func.func @type_lowering(%arg0: i32) -> i32 {
  %n = toy.box %arg0 : i32 -> !toy.num     // ① 装箱：i32 → !toy.num
  %x = toy.unbox %n : !toy.num -> i32      // ② 拆箱：!toy.num → i32
  %c = toy.constant 4 : i32
  %m = toy.mul %x, %c : i32
  return %m : i32
}
```

**转换到 ② 时，pattern 手上有两份不同的东西**

```cpp
LogicalResult matchAndRewrite(toy::UnboxOp op,
                              ArrayRef<Value> operands,        // ★ 已 remap
                              ConversionPatternRewriter &rewriter) const override {
  Value oldIn = op->getOperand(0);   // 原始：%n，类型仍是 !toy.num
  Value newIn = operands[0];         // remap 后：来自已转换的 toy.box，类型已是 i32
  // 因为 !toy.num 就映射成 i32，拆箱在低层退化成"什么都不做"
  rewriter.replaceOp(op, newIn);
  return success();
}
```

| 你问 | 拿到什么 | 什么时候用 |
|------|----------|-----------|
| `op->getOperand(0)` | `%n`，**旧类型** `!toy.num` | 想读原始语义/属性时 |
| `operands[0]`（或 adaptor） | 已替换的值，**新类型** `i32` | 构造新 op 时**必须**用这个 |

用错的后果很直接：拿 `op->getOperand(0)` 去构造 `low.*`，会造出一条操作数类型为 `!toy.num` 的低层 op，随后 verifier 或后续 pattern 报类型不匹配。

**中途的接缝长什么样**（若关闭 materialization 回调，框架会用 cast 顶上）

```mlir
// 转换进行中的示意快照：新旧类型用 cast 接住，避免中途破坏类型契约
%0 = builtin.unrealized_conversion_cast %arg0 : i32 to !toy.num
```

**成功结束后**，这类 cast 应当全部消失：

```mlir
func.func @type_lowering(%arg0: i32) -> i32 {
  %c = low.constant 4 : i32
  %m = low.mul %arg0, %c : i32     // box/unbox 因类型映射成恒等而消失
  return %m : i32
}
```

> 若最终 IR 里还留着 `builtin.unrealized_conversion_cast`，不要忽略它：**它精确指出哪一对类型没人负责转**。

> **自测**：如果只给 `toy.mul` 写了 pattern、忘了写 `toy.box`/`toy.unbox` 的，`applyPartialConversion` 会静默通过还是报错？报错信息里会出现哪个 op 名？

### 6.6 TypeConverter：conversion vs materialization

这是整章最容易混的一点。官方区分两条机制：

```
conversion（类型映射）          materialization（值层转接）
─────────────────────          ─────────────────────────
Type → Type(s)                 产生 IR（插入 cast/构造 op）
不产生 IR                      在转换过程中维护类型安全
addConversion(...)             addSource / Target Materialization(...)
```

#### Conversion

描述"源类型应变成哪些目标类型"。源类型映到自身 ⇒ 该类型视为 **legal**。支持 1→1 与 1→N；最近注册的规则优先尝试。

```cpp
converter.addConversion([](toy::NumType t) -> Type {
  return IntegerType::get(t.getContext(), 32);
});
converter.addConversion([](Type t) { return t; }); // 其余原样
```

#### Source Materialization

**何时**：值已被替换成新类型，但**仍有未转换的用户**期望旧的 source 类型。  
**做什么**：把新类型的值再"转回"旧类型（或插入等价转接 IR），供残留用户使用。

场景：`!toy.num` → `i32` 后，某个尚未转换的 `toy.print` 还要用原来的 `!toy.num`。

#### Target Materialization

**何时**：pattern（按其 TypeConverter）期望操作数已是**目标合法类型**，但当前手上的替换值类型还不对。  
**做什么**：把现有值物化成 pattern 期望的目标类型。

场景：pattern 要 `i32`，但 remapped 值暂时是别的合法中间类型，或尚未替换。

一句话记忆：

> **Source materialization**：已转换的值 → 还供给未转换用户（回到旧类型）。  
> **Target materialization**：未对齐的值 → 供给正在转换的 pattern（变成新类型）。

若关闭 `buildMaterializations`，框架改用 `unrealized_conversion_cast` 代替回调——toy 项目的 `ConvertToyToLow.cpp` 就演示了用 unrealized cast 做转接的路径。

### 6.7 Region / 签名转换

类型一变，函数类型与 block 参数签名必须跟着变。框架提供：

- `convertBlockSignature` / `applySignatureConversion`
- 对 `func.func` 等的类型转换 pattern（常与 dynamicallyLegal 联动）

这是贪心驱动器做不到、Conversion 必须上的核心原因之一。

### 6.8 与 toy 的 `--toy-to-low` 对照阅读

打开并并排阅读：

1. `mlir-toy-dialect/lib/LowPasses.cpp` —— 贪心版  
2. `mlir-toy-dialect/lib/ConvertToyToLow.cpp` —— Conversion 版  
3. `mlir-toy-dialect/test/convert.mlir` —— lit 期望

你应该能指出：

- `ConversionTarget` 把 `toy` 标 illegal、`low` 标 legal；
- `TypeConverter` 把 `!toy.num` → `i32`；
- `applyPartialConversion` 失败时如何暴露漏网 op；
- 为何生产 lowering（`convert-linalg-to-loops`、`convert-vector-to-llvm`…）都长成 Conversion 结构。

### 6.9 调试 Conversion

```bash
mlir-opt input.mlir \
  --pass-pipeline='builtin.module(my-conversion-pass)' \
  --debug-only=dialect-conversion \
  --mlir-print-ir-after-failure
```

| 手段 | 用途 |
|------|------|
| `--debug-only=dialect-conversion` | 打印合法化尝试、pattern 应用、失败原因 |
| 残留 `unrealized_conversion_cast` | 类型边界没消干净 |
| Analysis Conversion | 先问"哪些能降"再动手 |
| 对比 Partial vs Full | Full 一次暴露所有未覆盖 op |

---

## 第 7 章 内置 Dialect 地图

按**抽象层级从高到低**记忆（AI 编译主路径）：

```
[框架/领域]   torch / onnx / stablehlo / tosa / mhlo ...
      ↓
[结构化计算]  linalg  (+ tensor)
      ↓
[缓冲/循环]   memref  +  scf / affine
      ↓
[向量]        vector
      ↓
[标量/函数]   arith  +  func  +  math
      ↓
[CFG]         cf
      ↓
[设备]        gpu  (+ nvvm / rocdl / spirv ...)
      ↓
[LLVM 映射]   llvm
```

### 7.1 各层一句话 + 典型 op

| Dialect | 抽象层级 | 回答的问题 | 典型 op / 类型 |
|---------|---------|-----------|----------------|
| **func** | 函数与模块边界 | 调用约定、符号 | `func.func`、`func.call`、`func.return` |
| **arith** | 标量/向量算术 | 数值计算 | `addi`、`mulf`、`cmpi`、`constant` |
| **math** | 初等函数 | sin/exp/erf… | `math.exp`、`math.sqrt` |
| **scf** | 结构化控制流 | for/if/while（Region） | `scf.for`、`scf.if`、`scf.while`、`scf.yield` |
| **cf** | 扁平 CFG | 分支/开关 | `cf.br`、`cf.cond_br`、`cf.switch` |
| **tensor** | 值语义张量 | 不可变张量 SSA | `tensor.empty`、`extract`、`insert_slice` |
| **memref** | 缓冲语义 | 可副作用的内存视图 | `memref.alloc`、`load`、`store`、`subview` |
| **linalg** | 结构化计算 | 循环结构 + payload | `linalg.matmul`、`linalg.generic`、`linalg.yield` |
| **vector** | 多维向量 | 近硬件向量抽象 | `vector.transfer_read`、`contract`、`broadcast` |
| **gpu** | 主机/设备启动 | kernel launch、barrier | `gpu.launch`、`gpu.alloc`、`gpu.barrier` |
| **llvm** | LLVM IR 映射 | 出海到 LLVM | `llvm.func`、`llvm.add`、`llvm.load`、`!llvm.ptr` |

### 7.2 两对容易混的边界

**tensor vs memref**

| | tensor | memref |
|--|--------|--------|
| 语义 | 值（SSA，不可变） | 缓冲（可 load/store） |
| 别名 | 分析相对简单 | 必须处理别名与副作用 |
| 优化 | tile/fuse 更友好 | 靠近运行时与 ABI |
| 过渡 | —— | **bufferization**（第 8 章） |

**scf vs cf**

- `scf`：保留结构化（循环/条件还是 Region），利于循环变换。  
- `cf`：拍成 CFG，利于接到 LLVM。  
路径通常是 `scf → cf → llvm`（或部分直接 `scf → llvm` 相关 conversion）。

### 7.3 先跳过的外围 dialect

与根 README §3.2 一致：`spirv` / `emitc` / `async` / `pdl` / `acc` / `omp` 的细节——**知道名字，用到再查**。Transform dialect 知道"用 IR 描述调度"即可，做 tiling 策略时再学。

---

## 第 8 章 Linalg + Bufferization（重点★）

> Linalg Rationale：https://mlir.llvm.org/docs/Rationale/RationaleLinalgDialect/  
> Bufferization：https://mlir.llvm.org/docs/Bufferization/  
> IREE 整条编译栈建在"linalg on tensors → …"之上，见 [`iree-learning-guide.md`](./iree-learning-guide.md) 第 2 章。

### 8.1 Linalg 的设计取向

官方 rationale 的核心立场（压缩版）：

1. **变换与简洁性优先于单纯表达力**——op 集合应可变换、尽量正交，避免 ONNX 式算子爆炸。  
2. **信息保存在 IR 里，而不是靠分析猜回来**——op 语义声明传统上靠依赖分析才能得到的合法性信息。  
3. **变换保持高层形态**：例如 tiling 之后 **`linalg.matmul` 仍然是 `linalg.matmul`**（只是迭代空间变了），而不是立刻变成三层 `scf.for`。  
4. **合法性与收益性分离**——先保证变换合法，收益（何时 tile、tile 多大）留给搜索/启发式。

```
传统路径（过早降循环）:
  matmul → 三层 for → 再想 fuse？要做依赖分析 / 模式匹配

Linalg 路径（保结构）:
  linalg.matmul
       │ tile
       ▼
  linalg.matmul（小块）嵌在 scf.for 里
       │ fuse（仍看 iterator_types + indexing_maps）
       ▼
  更大的 linalg.generic / 融合后的 structured op
       │ 最后才
       ▼
  scf + memref/vector → llvm
```

### 8.2 结构化 op 携带的两份元信息

以 `linalg.generic` 为代表：

| 元信息 | 含义 | 用途 |
|--------|------|------|
| **`iterator_types`** | `parallel` / `reduction` / … | 哪些维可并行、哪些是归约 |
| **`indexing_maps`** | 仿射映射：迭代空间 → 操作数索引 | 数据访问模式、融合合法性 |

**融合不必理解"这是 softmax 还是 gelu"**——只看这两份元信息能否对齐。这正是相对 TOSA/StableHLO 上两两算子特殊融合规则的优势（IREE 文档同样强调这一点）。

#### 示例精讲：读懂一个 `linalg.generic`，并据此判断能否融合

> 目的：把 `indexing_maps` / `iterator_types` 从"术语"变成"能逐维读出来的东西"。

**最小的逐元素加法**

```mlir
#id = affine_map<(d0, d1) -> (d0, d1)>          // 迭代空间 (d0,d1) → 操作数下标 (d0,d1)

%out = linalg.generic {
    indexing_maps  = [#id, #id, #id],           // 顺序：ins..., outs...
    iterator_types = ["parallel", "parallel"]   // 两个维都可并行
  } ins(%a, %b : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%init : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %acc: f32):            // 每个"点"上的标量计算
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<4x8xf32>
```

逐项读法：

| 元信息 | 这里的取值 | 怎么读 |
|--------|-----------|--------|
| 迭代空间 | `(d0, d1)`，形状取自操作数 `4x8` | 一共要跑 4×8 个点 |
| `indexing_maps[0]` | `(d0,d1) -> (d0,d1)` | 第一个输入按 `a[d0][d1]` 取 |
| `indexing_maps[2]` | 同上 | 输出按 `out[d0][d1]` 写 |
| `iterator_types` | `parallel, parallel` | 没有归约维，两维都能并行/切分 |
| Region `^bb0` | 三个标量参数 | **点计算**：每个点做一次 `addf` |

**对照 matmul（含归约维）**

```mlir
#A = affine_map<(m, n, k) -> (m, k)>
#B = affine_map<(m, n, k) -> (k, n)>
#C = affine_map<(m, n, k) -> (m, n)>            // 输出不含 k → k 是归约维
// iterator_types = ["parallel", "parallel", "reduction"]
```

**融合合法性怎么判**：不看 op 名字，只看下游 op 读上游结果时的映射是否对得上。

| 情形 | 上游写出的映射 | 下游读入的映射 | 能否直接融合 |
|------|----------------|----------------|--------------|
| add → relu（逐元素） | `(d0,d1)->(d0,d1)` | `(d0,d1)->(d0,d1)` | **可以**：同一个点只需算一次 |
| matmul → relu | `(m,n)` | `(m,n)` | 可以（relu 作用在 matmul 的输出点上） |
| transpose → add | `(d0,d1)->(d1,d0)` | `(d0,d1)->(d0,d1)` | 需要先对齐映射，不能盲目合并 |
| 上游含 reduction，下游要整行 | 输出少一维 | 需要跨点访问 | 一般不能简单融合，要靠 tiling 配合 |

这就是「保结构」的价值：只要还是 `linalg.*`，这两份元信息就一直在 IR 里；一旦过早降成三层 `scf.for`，就得靠依赖分析把它们**猜回来**。

> **自测**：把上面 add 的 `iterator_types` 改成 `["parallel", "reduction"]`，输出类型应该变成什么？为什么？

### 8.3 Destination-Passing Style（DPS）

在 tensor 世界里，SSA 值不可变："更新张量"其实是**返回新张量**。DPS 要求：每个 tensor 结果对应一个 **`outs` / destination 操作数**，表示"在这份初始值上更新"。

```mlir
// outs(%c) 是 destination：结果与 %c 同 shape，语义上"写入 C"
%0 = linalg.matmul ins(%a, %b : tensor<MxKxf32>, tensor<KxNxf32>)
                   outs(%c : tensor<MxNxf32>) -> tensor<MxNxf32>
```

| 概念 | 说明 |
|------|------|
| Destination | 与结果配对的 tensor 操作数（quotes 重要：tensor 语义下并非原地写） |
| `DestinationStyleOpInterface` | 声明哪些操作数是 destination |
| 对 bufferization 的意义 | 若无冲突，结果可 **alias** 到 destination 的 buffer → in-place |

非 DPS 的 op（如部分 `tensor.generate`）往往**强制新分配**——这也是写 IR 时尽量用 `linalg.generic` + `outs` 的原因之一。

### 8.4 何时 Bufferize

官方建议非常明确：

> **多数变换（tile / fuse / …）先在 tensor 世界做完，再 bufferize**；  
> Bufferization 通常是 pipeline **较后**的步骤，紧挨着 memref→LLVM 之前。

原因：tensor SSA 的 use-def 链让融合与 in-place 分析更干净；过早进入 memref 会被迫做更重的别名分析。

### 8.5 One-Shot Bufferize：目标与性质

**目标**（与寄存器分配同构的难度）：

1. 尽量少占内存；  
2. 尽量少拷贝。

**One-Shot Bufferize 的官方定性**：

| 性质 | 含义 |
|------|------|
| **Monolithic** | 单 Pass 完成分析+重写（相对旧式碎片化 bufferize pass） |
| **Extensible** | 所有实现 `BufferizableOpInterface` 的 op 均可参与 |
| **Whole-function analysis** | 看全函数 tensor use-def，而不是局部 peephole |
| **2-Phase** | **先 analyze 再 rewrite**——分析阶段可看到精确 SSA 信息 |
| **Greedy** | 按启发式顺序逐个决定是否 in-place / 是否插入 copy |
| **Modular analysis** | 分析可替换；甚至可用 `AlwaysCopyAnalysisState` 关掉智能分析 |

**不做的事**：不负责释放 buffer——所有权释放由 **Ownership-based Buffer Deallocation** 流水线做。

### 8.6 两阶段在干什么

```
Phase 1  Analyze
  遍历（启发式顺序）带 tensor 语义的 op
  查询 BufferizableOpInterface
  构建 alias / equivalence 集合
  决定每个 OpOperand：in-place 还是必须 copy
        │
        ▼
Phase 2  Rewrite
  按决定把 tensor op → memref op
  需要时插入 alloc / copy
  （随后另跑 deallocation 管道）
```

RaW（Read-after-Write）冲突靠 SSA use-def 链检测：若同一 tensor 值在写入后仍被其他用户读取，就不能 in-place，必须 copy。

#### 示例精讲：同一段代码，一个能 in-place，一个必须 copy

> 目的：亲手判一次 RaW 冲突，并看到 Phase 2 因此插出来的 `memref.copy`。

**情形 A：写完就不再读旧值 → 可以 in-place**

```mlir
func.func @inplace(%A: tensor<8xf32>, %v: f32) -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %0 = tensor.insert %v into %A[%c0] : tensor<8xf32>   // 写
  return %0 : tensor<8xf32>                            // %A 之后没人再读
}
```

Phase 1 的判断：`%A` 在写之后**没有其他读用户** → 结果 `%0` 可与 `%A` 共用同一块 buffer。

Phase 2 产出（示意）：

```mlir
func.func @inplace(%A: memref<8xf32>, %v: f32) -> memref<8xf32> {
  %c0 = arith.constant 0 : index
  memref.store %v, %A[%c0] : memref<8xf32>   // 直接原地写，无 alloc / 无 copy
  return %A : memref<8xf32>
}
```

**情形 B：写完还要读旧值 → 必须 copy**

```mlir
func.func @conflict(%A: tensor<8xf32>, %v: f32) -> (tensor<8xf32>, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.insert %v into %A[%c0] : tensor<8xf32>   // 写：产生新值 %0
  %e = tensor.extract %A[%c1] : tensor<8xf32>          // ★ 仍读【旧】值 %A
  return %0, %e : tensor<8xf32>, f32
}
```

Phase 1 的判断：若让 `%0` 原地写在 `%A` 的 buffer 上，后面那句读到的就不再是旧值——**RaW 冲突**，因此 `%0` 必须落在新 buffer 上。

Phase 2 产出（示意）：

```mlir
%new = memref.alloc() : memref<8xf32>
memref.copy %A, %new : memref<8xf32> to memref<8xf32>   // ← 冲突的代价在这里显形
memref.store %v, %new[%c0] : memref<8xf32>
%e = memref.load %A[%c1] : memref<8xf32>                 // 旧值仍完好
```

两阶段各自的产物：

| 阶段 | 输入 | 输出 | 关键动作 |
|------|------|------|----------|
| Phase 1 Analyze | tensor 语义 IR | **决策表**（每个 OpOperand：in-place / copy） | 查 `BufferizableOpInterface`，建 alias 与等价集合，判 RaW |
| Phase 2 Rewrite | 决策表 | memref IR | 按决策换 op，必要处插 `alloc` / `copy` |

> 因此「多做几次 tile/fuse 再 bufferize」是有道理的：tensor 世界里 SSA use-def 干净，冲突一眼可判；进了 memref 之后要靠更重的别名分析才能得到同样结论。

> **自测**：把情形 B 里的 `tensor.extract %A` 改成 `tensor.extract %0`，还需要那次 copy 吗？为什么？

### 8.7 与 IREE / 算力网的关系

IREE 的主路径是 **linalg on tensors → Flow（dispatch）→ …**，bufferization 与 codegen 在设备侧深化。对你意味着：

- 在 MLIR 层把 **结构化计算 + DPS + One-Shot Bufferize** 学透，才能读懂 IREE 为何能在进入 HAL 之前做融合与切分；  
- 设备抽象、fence、多后端是下一阶段，见 [`iree-learning-guide.md`](./iree-learning-guide.md)。

---

## 第 9 章 工具链与调试

### 9.1 mlir-opt 高频 flag

```bash
# 跑指定 pass / pipeline
mlir-opt input.mlir --canonicalize
mlir-opt input.mlir --pass-pipeline='builtin.module(func.func(canonicalize,cse))'

# 打印 IR 演变
mlir-opt ... --mlir-print-ir-after-all
mlir-opt ... --mlir-print-ir-after-change
mlir-opt ... --mlir-print-ir-before=my-pass

# 泛型形式（不走美化 assemblyFormat，排歧义）
mlir-opt ... --mlir-print-op-generic

# 转换失败时留下线索
mlir-opt ... --mlir-print-ir-after-failure

# Dialect Conversion 专用调试
mlir-opt ... --debug-only=dialect-conversion

# 元素属性太大时折叠打印
mlir-opt ... --mlir-elide-elementsattrs-if-larger=8
```

toy 项目把同一套习惯用在 `toy-opt` 上：

```bash
cd mlir-toy-dialect
bash scripts/all.sh
# 或
./build/bin/toy-opt test/convert.mlir --toy-to-low-convert
./build/bin/toy-opt test/cost.mlir --toy-print-cost
```

### 9.2 lit + FileCheck

MLIR / LLVM 共用同一套测试文化：

```mlir
// RUN: toy-opt %s --toy-to-low-convert | FileCheck %s

func.func @add(%a: !toy.num, %b: !toy.num) -> !toy.num {
  %0 = toy.add %a, %b : !toy.num
  return %0 : !toy.num
}
// CHECK-LABEL: func @add
// CHECK: low.add
// CHECK-NOT: toy.add
```

| 技巧 | 用途 |
|------|------|
| `-split-input-file` | 一个文件多个独立用例 |
| `-verify-diagnostics` | 测 verifier / 诊断信息 |
| `--check-prefix=XXX` | 同一输入多组期望 |
| `CHECK-NOT` | 断言某层 op 已消失 |

### 9.3 从 MLIR 出海到 LLVM

```bash
mlir-opt input.mlir \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  | mlir-translate --mlir-to-llvmir \
  | opt -passes='default<O2>' -S \
  | llc -mtriple=x86_64-- -o -
```

这条链把本文、[`llvm-learning-guide.md`](./llvm-learning-guide.md)、以及将来 toy→linalg→llvm 验收钉在同一条命令上。

---

## 第 10 章 与 AI-Infra / IREE / toy 项目的接缝

### 10.1 仓库里的位置

```
paper-notes/03-mlir.md      ← 论文动机：为何多层、为何 Interfaces
docs/mlir-learning-guide.md ← 本文：官方机制蒸馏
mlir-toy-dialect/           ← 动手：双 dialect / Conversion / Interface
docs/llvm-learning-guide.md ← 出海之后：四层 IR + CodeGen
docs/iree-learning-guide.md ← 工业栈：linalg→flow→stream→hal
```

### 10.2 概念同构表

| LLVM（hello-compile） | MLIR（toy-dialect） | IREE |
|----------------------|---------------------|------|
| Module / Function / BB | Module / `func.func` / Block | 同左 + Flow/Stream/HAL 容器 op |
| Instruction | Operation | 各层 dialect op |
| `phi` | block argument | 同 MLIR |
| Pass + 分析保留 | Pass + DialectConversion | `iree-compile` 相位 |
| TableGen | ODS（同一语言） | 大量 ODS dialect |
| `opt` | `toy-opt` / `mlir-opt` | `iree-compile` |

### 10.3 为什么 IREE 需要你先懂 MLIR

[`iree-learning-guide.md`](./iree-learning-guide.md) 的相位图不是"黑盒阶段名"，每一跳都是 MLIR 机制的实例：

| IREE 相位 | 你在本文对应的概念 |
|----------|-------------------|
| 输入 → Linalg | Dialect Conversion + 结构化 op |
| Linalg 上 tile/fuse | 第 8 章信息保留；仍是 linalg.* |
| Flow dispatch | Region 边界 + 子图划分（业务层） |
| Stream / HAL | 更低层次 dialect；同步与设备对象 |
| Codegen → LLVM/SPIR-V | Conversion 到 `llvm` / `spirv` dialect |

**不会 Dialect Conversion / Interface，就只能把 IREE 当命令行工具；会了，才能改 lowering、加 op、读 `--compile-to` dump。**

### 10.4 toy 项目已经演示了什么

见 [`mlir-toy-dialect/README.md`](../mlir-toy-dialect/) 覆盖对照表。机制上已齐：

- 双 dialect + 自定义类型 + Region  
- 贪心 lowering **与** Dialect Conversion 对照  
- OpInterface 驱动的跨 dialect Pass  

**刻意留白**（正是你下一阶段要补的）：完整 bufferization、降到 linalg/llvm、Transform dialect——与根 README §3.2 动手验收对齐。

---

## 第 11 章 学习路径：最小必要集与动手清单

### 11.1 必须掌握

1. **渐进式 lowering**：为何不一步到 LLVM；每层应固化什么决定（第 1 章）。  
2. **默画** `Operation ⊃ Region ⊃ Block`，说清 Value/Type/Attribute（第 2 章）。  
3. **Block argument 替代 phi** 的理由与写法。  
4. **ODS 五件套**：arguments / results / assemblyFormat / traits / hasVerifier（第 3 章）。  
5. **Trait vs Interface**：为何 Interface 让一个 Pass 跨 dialect 工作；能举 `ToyCost` / `BufferizableOpInterface`（第 4 章）。  
6. **贪心 Rewrite vs Dialect Conversion** 的收敛条件与失败模式（第 5–6 章）。  
7. **Conversion 三模式** + Target 的 legal/illegal/dynamicallyLegal/recursive（第 6 章）。  
8. **TypeConverter**：conversion（不产 IR）vs source/target materialization（产 IR）；preorder walk；pattern 链式 `bar→baz→foo`（第 6 章）。  
9. **内置 dialect 地图**：func/arith/scf/cf/memref/tensor/linalg/vector/gpu/llvm 的层级（第 7 章）。  
10. **Linalg 信息保留** + DPS + One-Shot Bufferize 的 monolithic / 2-phase / Interface 扩展（第 8 章）。  
11. **`mlir-opt` 调试旗标**与 lit/FileCheck 基本姿势（第 9 章）。

### 11.2 可以先跳过

与根 README §3.2「先跳过」一致：

- PDL / PDLL 声明式重写（先写 C++ pattern）。  
- Transform dialect 完整用法（知道是"IR 描述调度"即可）。  
- Python bindings、手写 custom assembly parser（`assemblyFormat` 够用）。  
- 外围 dialect 细节：`spirv` / `emitc` / `async` / `pdl` / `acc` / `omp`。  
- MLIR C API、ExecutionEngine 内部实现。  
- Graph Region 的深水区（非 AI 主线）。  
- Ownership-based deallocation 的实现细节（先会用 pipeline）。

### 11.3 动手清单（对齐根 README §3.2 验收）

**第零步：跑通现有项目（硬门槛）**

```bash
cd mlir-toy-dialect
bash scripts/all.sh
```

验收：能讲解 `--toy-to-low-convert` 与 `--toy-print-cost`；能指着 `ConvertToyToLow.cpp` 说出 Target / TypeConverter / Pattern 三件套。

**第一步：对照读 Conversion**

1. 读 `lib/LowPasses.cpp`（贪心）与 `lib/ConvertToyToLow.cpp`（Conversion）。  
2. 改 `test/convert.mlir`，故意漏一条 pattern，观察 Partial/Full 行为差异。  
3. 加上 `--debug-only=dialect-conversion`（若工具链支持）看合法化轨迹。

**第二步：TypeConverter 验收项 ★**

> 根 README：给 Toy 加 TypeConverter，把自定义 tensor/类型转到 `memref`/`i32` 等，走完整 `applyPartialConversion`。

在现有 `!toy.num → i32` 基础上扩展（或新增 `!toy.tensor → memref`）：补 source/target materialization，写 lit 断言函数签名与 cast 清理。

**第三步：OpInterface 验收项 ★**

> 根 README：实现如 `ToyShapeInferenceOpInterface`，写通用 shape inference Pass，验证同一 Pass 作用于多 dialect op。

模仿 `ToyCostOpInterface`：接口定义在 `.td`，`toy`/`low`（或第二 dialect）各自实现，Pass 只依赖接口。

**第四步：端到端 Toy → linalg → scf → llvm ★**

> 根 README：把 Toy 降到 linalg，再用 `-linalg-tile` / `-convert-linalg-to-loops` 降到 `scf`+`memref`，再到 `llvm`，用 `mlir-cpu-runner`（或等价）跑出结果。

建议里程碑：

```
toy.*  ──conversion──▶  linalg.* + tensor.*
       ──tile/fuse──▶   仍为 linalg.*（确认没过早变循环）
       ──bufferize──▶   linalg.* + memref.*
       ──loops──▶       scf.* + memref.*
       ──▶ llvm dialect ──translate──▶ LLVM IR ──▶ 跑通
```

每一步加 lit（已有 `test/` 基础设施）。

**第五步：接到 IREE 文档**

用同一心智模型读 [`iree-learning-guide.md`](./iree-learning-guide.md) 第 2 章：把 `dispatch region` 理解成"又一层带 Region 的结构化边界"。

### 11.4 过关自检（五分钟口述）

- [ ] 为什么 `x*4` 在 toy 层没法变 `<<2`，降到 low 就可以？（多层分工）  
- [ ] 贪心 lowering 漏规则 vs Conversion 漏规则，现象有何不同？  
- [ ] Source materialization 和 Target materialization 各在什么边界触发？  
- [ ] 为什么 tiling 后还保持 `linalg.matmul` 很重要？  
- [ ] One-Shot Bufferize 为什么要先 analyze 再 rewrite？  
- [ ] Interface 如何让 `--toy-print-cost` 不 `#include` low dialect 也能给 `low.shl` 标价？

---

## 附录：一页速查

```
【定位】  多层 dialect + 渐进 lowering；论文动机 → paper-notes/03-mlir.md
         机制本文；动手 → mlir-toy-dialect/；出海 → llvm-learning-guide
         工业栈 → iree-learning-guide（linalg→flow→stream→hal）

【IR】    Operation ⊃ Region ⊃ Block ⊃ Operation（可嵌套）
         Value = Result | BlockArgument；Attribute = 编译期常量
         Block argument 替代 phi；SSACFG Region vs Graph Region

【Dialect】命名空间；扩展三支柱 Op / Type / Attribute
【ODS】    arguments / results / assemblyFormat / traits /
          hasVerifier / hasFolder / hasCanonicalizer

【Trait vs Interface】
  Trait     = 标签（Pure/Commutative/Terminator/IsolatedFromAbove）
  Interface = 能力契约 → 通用 Pass 跨 dialect
  必知      MemoryEffects / LoopLike / DestinationStyle / Bufferizable
  另记      DialectInterface（如 Inliner）vs OpInterface

【Pattern / Pass】
  RewritePattern + applyPatternsAndFoldGreedily → 不动点；漏规则可能静默
  fold() → Attribute|Value；canonicalize 吃 folder + patterns
  PassManager：任意 op 粒度嵌套 pipeline

【Dialect Conversion ★】
  模式    Partial | Full | Analysis
  遍历    preorder；pattern 可链式 bar→baz→foo
  Target  legal / illegal / dynamicallyLegal / recursivelyLegal
  Pattern ConversionPattern + remapped operands（adaptor）
  TypeConverter
    conversion      ：Type→Type(s)，不产 IR
    source material ：已转换值 → 供未转换用户（回源类型）
    target material ：现有值 → 供 pattern 期望的目标类型
  vs 贪心 toy-to-low：Conversion 以 Target 契约定义"做完"

【Dialect 地图】
  高 linalg/tensor → scf/affine+memref → vector → arith/func → cf
    → gpu → llvm →（translate）→ LLVM IR

【Linalg ★】
  变换与简洁性优先；iterator_types + indexing_maps 保信息
  tiling 后 linalg.matmul 仍是 linalg.matmul
  DPS outs ↔ DestinationStyleOpInterface

【One-Shot Bufferize ★】
  目标：少内存、少拷贝 | 时机：tensor 变换之后
  Monolithic + BufferizableOpInterface
  2-Phase：analyze（alias/equivalence/in-place）→ rewrite
  Greedy；不负责 free（ownership deallocation 另做）

【调试】
  mlir-opt --pass-pipeline=... --mlir-print-ir-after-all
           --mlir-print-op-generic --debug-only=dialect-conversion
  lit + FileCheck；残留 unrealized_conversion_cast = 类型边界未消净

【toy 验收（README §3.2）】
  ① TypeConverter + applyPartialConversion
  ② OpInterface + 通用 Pass
  ③ Toy→linalg→scf→llvm 端到端 + lit
  硬门槛：all.sh；能讲 --toy-to-low-convert 与 --toy-print-cost
```
