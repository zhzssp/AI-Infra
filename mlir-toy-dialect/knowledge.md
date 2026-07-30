# 知识沉淀：MLIR 的 IR 是一棵「嵌套的树」

> 本文档沉淀一个最核心、也最容易误解的 MLIR 概念：**IR 是由 `Operation` 递归嵌套而成的树**。
> 所有结论都 anchoring 在本仓库 `mlir-toy-dialect` 的真实代码与 `out/` 产物上，便于随时对照验证。
> 阅读顺序建议：先建立"万物皆 Operation"的心智模型，再看本项目的代码证据，最后理解嵌套带来的意义。

---

## 一、核心心智模型：万物皆 Operation

在 MLIR 里，**一切都是 `Operation`**（`mlir::Operation` 或其子类）。函数、模块、常量、
加法、return，甚至 `module` 关键字本身，在内存里都是同一种东西。

一棵 IR 树，就是由 `Operation` 一层套一层构成的。**嵌套（nesting）的来源，是 `Operation` 的一个字段：`regions`（区域）。**

一个 `Operation` 的内存结构（简化）：

```
Operation
├── name        : "toy.add" / "func.func" / "module"   ← 操作名（方言.操作）
├── operands    : [%0, %1]                ← 输入（别的 op 的结果，即 SSA value）
├── results     : [%2]                    ← 输出（自身产生的新 SSA value）
├── attributes  : {value = 42}            ← 编译期常量属性
├── regions     : [...]                   ← ★ 嵌套的"子操作集合"——树的分支
└── successors  : [...]                   ← 控制流后继（分支/循环用）
```

**`regions` 就是树的分支**：一个 op 可以"肚子"里装一个 region，region 里装若干 block，
block 里再装一串 op；这些 op 自己又可能有 region……如此递归，就成了树。

> 一句话：**MLIR 的 IR = 一棵由 `Operation` 递归嵌套的树；嵌套发生在 `regions` 字段上——结构型 op 开 region 当树枝，叶子 op 当树叶。**

---

## 二、本项目的代码证据（可逐条核对）

### 证据 1：我们的 `toy.add` 是「叶子」——`ZeroRegions`

由 `include/Toy/ToyOps.td` 经 `mlir-tblgen` 自动生成的 C++ 头文件里，`AddOp` 的类声明：

```98:98:build/include/Toy/ToyOps.h.inc
class AddOp : public ::mlir::Op<AddOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ...> {
```

`OpTrait::ZeroRegions` 宣告 **`toy.add` 没有 region，是叶子节点，不能嵌套子操作**。
本项目里 `toy.constant` / `toy.add` / `toy.mul` 全是这种叶子 op（ODS 里没给它们定义 region）。

### 证据 2：框架天生为「带 region 的 op」设计

同一个生成文件里，每个 op 的 Adaptor 都持有 `RegionRange`：

```49:49:build/include/Toy/ToyOps.h.inc
  ::mlir::RegionRange odsRegions;
```

说明 `mlir::Operation` 的通用底座支持"操作携带 regions"——只是我们的 toy op 选择 `ZeroRegions` 不装而已。
而 `func.func` 和 `module` 则是**带 region 的 op**：

| Op | 来自 | region 含义 |
|----|------|------------|
| `module` | 内置 `ModuleOp` | 带 1 个 region，装整个文件的内容 |
| `func.func` | 内置 `func` dialect 的 `FuncOp` | 带 1 个 region，装函数体 |

这正是 `module { func.func { ... } }` 嵌套结构的 C++ 根源。

### 证据 3：两个方言都被注册，故能互相嵌套

`tools/toy-opt/toy-opt.cpp` 里：

```39:41:tools/toy-opt/toy-opt.cpp
  mlir::registerAllDialects(registry);          // 注册内置（含 func 方言）
  registry.insert<mlir::toy::ToyDialect>();      // 注册我们的 toy
  registry.insert<mlir::low::LowDialect>();      // 注册 low
```

解析器同时认识 `func`（提供 `func.func`/`return`）和 `toy`（提供 `toy.add` 等），
所以才能让 `toy` 的叶子 op 挂在 `func.func` 的 region 内部——**不同方言的操作在同一棵树上共存**，
正是 MLIR 区别于 LLVM 的关键能力。

---

## 三、把 demo2 的输出画成真实的树

`out/demo2_canonicalize.mlir` 那段 IR，在内存里就是下面这棵树：

```
ModuleOp  (module)                          ← ① 根，带 1 个 region
│
└─ Region（module 的 { }）
   └─ Block（模块级唯一 block）
      │
      ├─ FuncOp  func.func @fold_add() -> i32   ← ② 带 1 个 region（函数体）
      │  │
      │  └─ Region（func 的 { }）
      │     └─ Block（函数体 block）
      │        ├─ toy.constant 30 : i32   ← ③ 叶子，ZeroRegions
      │        └─ func.return %0 : i32    ← ③ 叶子（func 方言的 return）
      │
      └─ FuncOp  func.func @fold_add_mul() -> i32
         └─ Region → Block → toy.constant 60 / func.return
```

要点：
- **三层嵌套**：`ModuleOp` ⊃ `FuncOp` ⊃ `toy.constant`/`return`。
- **越往下抽象层次越低**：module（容器）→ function（控制结构）→ 算术（toy 方言的具体计算）。
- **toy 的 op 全是叶子**（无 region），只有 func/module 这类"结构型 op"才展开新分支。

> 重要提醒（常见误解）：`module { func.func {...} }` 这个**外壳在输入文件里本来就写好了**，
> 优化（如 `--canonicalize`）只改写 `func` 的**函数体内部**，外壳一直不变。
> 见 `test/canonicalize.mlir` 第 16 行的输入与 `out/demo2_canonicalize.mlir` 第 40 行的输出对比。

---

## 四、树是怎么被「建起来」和「走下去」的

### 建树（Parser）

你写 `test/canonicalize.mlir` 时，文本是嵌套的：

```16:21:test/canonicalize.mlir
func.func @fold_add() -> i32 {
  %0 = toy.constant 10 : i32
  %1 = toy.constant 20 : i32
  %2 = toy.add %0, %1 : i32
  return %2 : i32
}
```

解析器看到 `func.func ... {` 就 new 一个 `FuncOp` 并开一个 region；进到 `{` 里，把每一行
（如 `toy.constant`）解析成 region 内 block 里的一个子 `Operation`；`}` 闭合 region。
**文本的缩进/花括号，就是树的形状。**

### 遍历（Pass / Printer）

- **Printer** 递归地"进 region 就打印 `{`，出 region 就打印 `}`"——所以你看到 `module { func.func { ... } }`。
- **Pass**（如 `--canonicalize`）也是递归进 region：先处理 module，再进每个 func，再进函数体 block，
  最终找到 `toy.add` 调用你的 `fold()`。因为大家都是 `Operation`，Pass 用**统一的 API**
  （`getRegions()` / `getBody()` / `walk<toy::AddOp>()` / `replaceOp()`）就能在任意层级操作——
  这正是 `RewritePattern` 能"透明地"在 `func.func` 内部匹配 `toy.add` 的原因。

---

## 五、嵌套树的几大意义（为什么这么设计）

对比 **LLVM IR**：它也是 函数→基本块→指令 的树，但**指令不能嵌套指令**，且只有一套固定的类型/操作抽象层。
MLIR 的树更"通用"：

1. **多层 IR（Multi-Level）**：同一棵树上可以在高层放 `func`/高层计算，低层放 `low.shl` 等贴近硬件的操作。
   Lowering 就是把高层 op 替换成低层 op（演示 4：`toy.add`→`low.add`；演示 5-b：`low.mul`→`low.shl`），
   树的结构不变，只是节点"降级"。**在信息最合适的那一层，做最合适的优化。**
2. **统一操作接口**：因为万物皆 `Operation`，无论在哪一层、哪个方言，都用同一套 API 增删改查、写 Pass。
   本项目 toy/low 两个方言能无缝协作，靠的就是这点。
3. **作用域与符号**：`func.func` 的 region 定义了一个作用域，`return` 属于它，SSA 值 `%0` 的作用域也限定在所在 block。
   嵌套天然表达了"谁属于谁"。
4. **Pass 粒度灵活**：可以对整个 module 跑 Pass，也可以只对一个 `func.func` 跑——因为 Pass 作用在树上的某个 `Operation` 节点，粒度由你定。

---

## 六、速查卡（FAQ）

| 问题 | 答案 |
|------|------|
| MLIR IR 里最小单位是什么？ | `Operation`（`mlir::Operation`），一切皆它 |
| 嵌套是怎么实现的？ | `Operation` 的 `regions` 字段装子操作，递归成树 |
| 为什么 `toy.add` 没有嵌套？ | 它带 `OpTrait::ZeroRegions`，是叶子 op |
| `module`/`func.func` 是谁定义的？ | `module`=`ModuleOp`（内置），`func.func`=`FuncOp`（内置 `func` dialect），**不在本项目 toy/low 方言里** |
| 嵌套和 LLVM 的本质区别？ | LLVM 指令不能嵌套指令、只有单层抽象；MLIR 操作可任意嵌套、支持多层方言 |
| Pass 怎么遍历嵌套树？ | 通过 `getRegions()`/`getBody()`/`walk<>()` 等统一 API 递归进入 |
| `module { func.func {...} }` 是优化产生的吗？ | 不是，输入 `.mlir` 文本本来就写了，优化只改函数体内部 |

---

## 七、延伸阅读（本仓库内）

- `MLIR-运行流程与关键组件.md` —— 编译期 vs 运行期两条时间线、fold vs RewritePattern 两条改写主线。
- `README.md` —— 项目总览、多层 dialect 分工（`toy` 高层 vs `low` 低层）、启动流程。
- `include/Toy/ToyOps.td` —— 声明式定义 Operation 的样板，看 `assemblyFormat` 如何决定文本语法。
- `lib/ToyPasses.cpp` / `lib/LowPasses.cpp` —— `RewritePattern` 与 lowering/强度削减的真实实现。
- `out/demo2_canonicalize.mlir` / `out/demo5b_strength.mlir` —— 嵌套树与多层 lowering 的实证产物。
