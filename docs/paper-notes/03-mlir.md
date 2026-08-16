# MLIR：面向异构硬件与多层抽象的可扩展编译器基础设施

> **导航**：[笔记索引](README.md) · [自学枢纽](../README.md)（阶段 2） · [横切概念](../learning-guides/ai-compiler-foundations-learning-guide.md) §6.2（渐进 lowering）  
> **配套阅读**：本篇讲**为什么需要多层 IR**；**机制怎么用**（Dialect Conversion / Interfaces / Linalg + Bufferize）见 [`mlir-learning-guide.md`](../learning-guides/mlir-learning-guide.md)；动手在 [`mlir-toy-dialect/`](../../mlir-toy-dialect/)；工业形态见 [`iree-learning-guide.md`](../learning-guides/iree-learning-guide.md)。

> **论文元信息**
> 标题：*MLIR: A Compiler Infrastructure for the End of Moore's Law*
> 作者/机构：Chris Lattner（Google，发表时已加入 SiFive）、Mehdi Amini、Uday Bondhugula（IISc）、Albert Cohen、Andy Davis、Jacques Pienaar、River Riddle、Tatiana Shpeisman、Nicolas Vasilache、Oleksandr Zinenko（均 Google）
> 发表信息：Preprint，2020；arXiv：[2002.11054](https://arxiv.org/abs/2002.11054)（v2, 2020-03-01）

---

## 1. 它解决什么问题

编译器基础设施长期存在一个"一刀切"（one size fits all）问题：LLVM IR 本质上是"带向量的 C"，JVM 是"带 GC 的面向对象类型系统"。这类单一抽象层级的设计对主流语言（C/C++、Java）非常成功，但对高层语言（Swift、Rust、Julia、Fortran）和机器学习系统而言，源码语义与 LLVM IR 之间的抽象鸿沟太大——在 LLVM IR 上做源码级分析（例如线性类型检查、张量形状推导）几乎不可能。

论文用两张图刻画了这个碎片化问题：

- **图 1（TensorFlow 生态）**：同一个模型要经过 TensorFlow Graph → XLA HLO / TensorRT / nGraph / CoreML / TFLite → LLVM IR / TPU IR / NNAPI 等一堆互不兼容的中间表示和编译器，每一段转换都是手工、脆弱、难调试的。
- **图 2（通用语言生态）**：Swift 有 SIL，Rust 有 MIR，Julia 有自己的 IR，各自独立发明、独立实现了一整套解析、打印、位置追踪、pass 调度、验证等基础设施，然后再各自下降到共享的 LLVM IR 后端。

核心痛点可以归纳为三条：

1. **重复造轮子**：每个领域都要重新实现"IR 基础设施"这一层（文本解析/打印、diagnostics、pass manager、多线程编译支持），工程成本高、质量参差不齐（编译慢、诊断差、优化后调试体验差是常见后果）。
2. **过早 lowering 导致高层语义丢失**：一旦把高层结构（循环嵌套、张量形状、虚函数表、稀疏度信息……）下降成扁平化的低级表示，就很难再"raise"回去做高层优化或语言特定分析——这种回溯是脆弱且侵入式的。
3. **异构硬件编译无统一范式**：面对加速器（TPU、GPU、DSA）越来越多样化的现实，缺少一套既能表达高层数据流图、又能逐步下降到具体硬件指令的统一基础设施。

MLIR（Multi-Level Intermediate Representation）的目标就是提供一套**可复用、可扩展**的编译器基础设施：标准化 SSA 形式的 IR 数据结构，提供声明式定义新 IR（dialect）的系统，并把常见的编译器工程问题（解析/打印、位置追踪、pass 管理、并行编译）一次性解决好，让各领域只需要"定义新的 op/type"而不必重新发明基础设施。

---

## 2. 整体运行框架

### 2.1 多层 Dialect + 渐进式 lowering 的整体形态

MLIR 不是一层 IR，而是"同一套宿主机制（Operation/Region/Type/Attribute）承载多层方言"。一个典型的深度学习编译流水线大致长这样（并非论文强制规定，而是社区常见实践，例如 IREE / TensorFlow 的做法）：

```
前端图 IR：tf.* / torch.* / onnx.*        （领域特定，携带算子语义、形状、设备信息）
        │  转换为结构化张量计算
        ▼
结构化计算层：linalg.* / tosa.*            （张量级别的通用线性代数抽象）
        │  tiling / bufferization（tensor -> memref）
        ▼
循环与控制流层：affine.* / scf.*           （显式循环嵌套、结构化控制流，仍是 Region 嵌套）
        │  向量化
        ▼
向量与内建标量层：vector.* / arith.* / func.*
        │  逐指令翻译（Dialect Conversion）
        ▼
LLVM Dialect：llvm.*                        （MLIR 内对 LLVM IR 的忠实映射）
        │  mlir-translate --mlir-to-llvmir
        ▼
LLVM IR  ──▶  LLVM 后端（SelectionDAG/GlobalISel） ──▶  目标机器码（CPU/GPU/TPU/加速器）
```

每一步都只跨越"一小步"抽象距离，这正是论文强调的**渐进式下降（progressive lowering）**：不是像传统前端一样"AST 直接跳到 LLVM IR"，而是经过若干个语义明确、可独立验证的中间层，每一层只处理它擅长的信息（比如 `affine` 层擅长做依赖分析和循环变换，`vector` 层擅长指令级并行）。多个 dialect 的操作可以在同一个 IR 里**共存混合**（第 6.2 节详述），而不是必须整体转换完才能进入下一层。

### 2.2 核心 IR 数据结构层次

MLIR 的 IR 本质上是一棵由 `Operation` 递归嵌套构成的树（论文 Figure 3）。从大到小、从外到内的层次关系如下：

```
Operation（操作，唯一的语义单元）
├── OpName            —— "dialect.opname"，如 "affine.for"、"toy.add"
├── Operands           —— 输入的 SSA Value 列表
├── Results            —— 输出的 SSA Value 列表（同样遵守 SSA）
├── Attributes         —— 编译期静态信息的 key-value 字典（Attribute 是其值类型）
├── Regions[]          —— ★嵌套结构的来源：一个 Op 可以带 0..N 个 Region
│     └── Region
│           └── Block[]         —— Region 由若干 Block 构成一个 CFG
│                 ├── BlockArguments   —— Block 的类型化入参（MLIR 用它替代 φ 节点）
│                 ├── Operation[]      —— Block 内是顺序排列的 Operation 列表（可递归嵌套）
│                 └── Terminator（Op）—— Block 必须以终结符 Op 结尾（branch/return/yield…）
├── Successors         —— 控制流后继 Block（用于分支/循环）
└── Location           —— 该 Op 的来源位置信息（可追溯到源码、AST 节点等）
```

逐个说明：

- **Value**：SSA 值，只能被定义一次，类型固定，可以是某个 Op 的 Result，也可以是 Block 的 Argument。
- **Type**：每个 Value 必须有类型；类型系统本身是可扩展的（可以直接引用外部类型系统，如 `llvm::Type`），并且是非依赖类型（non-dependent），支持 trivial/parametric/function/sum/product 等类型形式。
- **Attribute**：结构化的编译期常量信息（整数、字符串、浮点数组、仿射映射等），类型化，挂在 Op 的字典里，也可以作为"属性别名"（`#map1 = ...`）复用。
- **Block**：Op 列表 + 一个终结符；一个 Region 内多个 Block 之间通过终结符的后继关系构成标准的 CFG；Block 可以带类型化参数（block arguments），MLIR 用这种"functional SSA"形式（终结符把值传给后继 Block 的参数）取代了传统 φ 节点。
- **Region**：Op 挂"肚子"里的容器，装一组 Block；Region 的语义完全由它所属的 Op 定义（比如 `affine.for` 的单 Block Region 表示循环体，`scf.if` 的两个 Region 表示 then/else 分支）。这是 MLIR 相对 LLVM **扁平 CFG** 最大的结构差异：LLVM 的指令不能嵌套指令，只有一层"函数 → 基本块 → 指令"；而 MLIR 的 Operation 可以任意递归嵌套 Region，天然表达循环树、闲包、并行区域等结构化语义。
- **Symbol / SymbolTable**：一种不遵守 SSA 定义规则的命名机制（可以先使用后定义，用于表达递归引用，如函数自身调用自己）。带 SymbolTable 的 Op（如 `module`、`func.func`）内部的具名实体（函数、全局变量）注册在符号表里，可以嵌套。

#### 示例精讲：把一段日常写法的 IR 摊成上面这棵树

> 目的：术语树看得懂，但拿到真实 `.mlir` 时要能立刻指认每一层。

```mlir
module {
  func.func @add2(%a: i32, %b: i32) -> i32 {
    %s = arith.addi %a, %b : i32
    return %s : i32
  }
}
```

对应的对象树（注意 `module` 和 `func.func` 本身也是 Operation）：

```text
Operation "builtin.module"                 ← 带 SymbolTable
└─ Region
   └─ Block                                ← 无参数，装全局符号
      └─ Operation "func.func" @add2       ← Attributes 里存着函数名与函数类型
         └─ Region                          ← 函数体
            └─ Block ^entry                 ← BlockArguments: %a, %b
               ├─ Operation "arith.addi"    ← Operands: %a,%b；Results: %s
               └─ Operation "func.return"   ← Terminator
```

三处最容易被忽略的对应关系：

| 文本里看到的 | 树里的位置 | 说明 |
|--------------|-----------|------|
| `@add2` | `func.func` 的 **Attribute**（符号名） | 不是 Value，不能被当操作数使用 |
| `%a`, `%b` | 入口 Block 的 **BlockArguments** | 定义点是块，不是某条 op |
| `return` | Block 的 **Terminator** | 每个 Block 必须以终结符结尾 |

论文 Figure 3 的 generic 语法（`"dialect.op"(%operands) : (类型) -> 类型`）与上面的自定义语法描述的是**同一棵树**，只是打印形式不同——`mlir-opt --mlir-print-op-generic` 可以随时切过去看。

> **自测**：`@add2` 这个名字如果写错成不存在的符号被别处引用，报错来自 Verifier 的哪一类检查（SSA dominance 还是符号解析）？

`module { func.func { ... } }` 就是这棵树最外两层的具体实例：`ModuleOp` 带 1 个 Region 装全局符号表，`FuncOp` 带 1 个 Region 装函数体 Block，函数体 Block 里才是真正的计算 Op（如 `toy.add`）。这与仓库里 `mlir-toy-dialect/knowledge.md` 画的那棵"嵌套树"完全一致——理解那份笔记之后，本节内容就是把它推广到 MLIR 论文的正式术语上。

### 2.3 Dialect 是什么

**Dialect（方言）**是 Op、Type、Attribute 的**逻辑分组机制**，本身不引入任何新语义，只是提供命名空间（文本前缀，如 `affine.`、`toy.`）和"dialect 级默认行为"（例如整个 dialect 统一的常量折叠钩子 `hasConstantMaterializer`）。关键设计取舍：

- MLIR **没有任何内建操作集**（no builtin ops）——哪怕 `module`、`func.func`、`return` 这些"看起来像基础设施"的概念，也是 builtin/func dialect 里定义的普通 Op，和用户自定义的 `toy.add` 地位完全相同。
- 一个 Op/Type/Attribute 只属于一个 dialect，但**不同 dialect 的 Op 可以在同一个 IR 里自由混合**——这是支持渐进式 lowering 和跨领域复用的基础（第 3、6 节详述）。

### 2.4 Pass 与 Conversion 基础设施

除 IR 本身外，MLIR 提供了一整套"开箱即用"的编译器工程基础设施：

- **PassManager**：与传统编译器"固定粒度"（module pass / function pass / loop pass）不同，MLIR 的 module、function 都只是普通带 Region 的 Op，所以 PassManager 天然支持**任意层级、任意嵌套深度**的 Op 上运行 Pass（例如 `builtin.module(func.func(canonicalize))` 表示只在每个 `func.func` 内部跑 `canonicalize`）。这也是并行编译的基础：只要一个 Op 是 *isolated-from-above*（作用域隔离，如 `func.func`），它的子树内部的 use-def 链不会跨越边界，多个这样的子树可以被多线程并行处理。
- **RewritePattern**：把"匹配一个子图 → 替换成另一个子图"的转换抽象为可复用的 `OpRewritePattern`/`RewritePattern`。多个 Pattern 组成 `RewritePatternSet`，交给**贪心驱动器**（`applyPatternsAndFoldGreedily`）反复扫描 IR、命中就重写，直到不动点。这是 MLIR 里写优化 Pass 的最主要武器。
- **DialectConversion（方言转换）**：当需要把一个 dialect **整体或部分**翻译到另一个 dialect 时（比如 `affine`→`scf`→`llvm`），单纯的贪心重写不够用，因为需要精确控制"转换后的 IR 是否合法"。DialectConversion 框架由三部分组成：
  - `ConversionTarget`：声明目标状态下哪些 Op/dialect 是"合法"的（legal），哪些是非法必须被消除的；
  - `TypeConverter`：定义类型层面的转换规则（如 `tensor<...>` → `memref<...>`），并自动处理跨类型的 block argument、函数签名转换；
  - `ConversionPattern`：带类型转换上下文的 RewritePattern 变体，`matchAndRewrite` 拿到的是已经转换过的 operand。
  - **Partial conversion（部分转换）** vs **Full conversion（完全转换）**：部分转换允许转换后的 IR 里仍然残留一部分未转换的旧 dialect Op（只要它们被标记为 legal），这正是"渐进式下降"在工具链层面的落地——不必一次 pass 就把所有 op 都转换干净。
- **Trait & Interface**：Trait 是编译期通过 C++ 类型系统附着在 Op 上的"位标记"或行为（如 `Commutative`、`Pure`/无副作用、`IsolatedFromAbove`），供通用 pass（DCE、CSE）判断是否可以安全变换；Interface 是更重的"契约"，允许不同 dialect 的 Op 实现同一套接口方法（如 `DialectInlinerInterface`），从而让一个通用 pass（如 inliner）在完全不知道具体 dialect 语义的情况下也能正确工作。
- **Verifier**：每个 Op/Attribute 可以声明自己的结构与语义约束（操作数个数、类型匹配、必须携带某属性等），框架在结构层面统一检查 SSA dominance、terminator 完整性、符号唯一性等不变式；验证失败即中止编译，这既是正确性保障，也是最直接的调试工具。
- **Canonicalization（规范化）**：一个专门的、对所有 dialect 通用的 pass，驱动每个 Op 的 `fold()`（常量折叠）和 `getCanonicalizationPatterns()`（代数化简模式，如 `x - x → 0`），把 LLVM 生态里分散的 `InstCombine`/`DAGCombine`/`PeepholeOptimizer`/`SILCombine` 等专用 pass 统一成一套可扩展机制。
- **TableGen/ODS 代码生成**：借用 LLVM 已有的 TableGen 工具，用声明式的 `.td` 文件描述 Op 的结构（Operation Definition Specification，ODS），由 `mlir-tblgen` 生成 C++ 样板代码（accessor、builder、verifier、parser/printer骨架），既减少手写代码量，也保证文档与实现同步（第 4 节详述）。

---

## 3. 核心特性逐条拆解

### 3.1 可扩展的 Op/Type/Attribute 系统与"无内建操作集"

**是什么**：MLIR 的核心只有极少数根概念——Operation、Type、Attribute——其余一切（包括 module、function、return、int 类型）都是用户可定义、可替换的扩展。系统本身不假设任何具体领域。

**为什么这样设计**：论文的设计原则第一条即"little builtin, everything customizable"。作者观察到不同领域需要表达的抽象差异极大（ML 图、AST、多面体模型、CFG、LLVM 风格指令级 IR），若把任何具体假设写进核心，就会限制其中一部分场景；反之，用最少的通用抽象（Op + Type + Attribute）去表达一切,可以让系统同时服务所有这些场景。

**带来什么能力**：任何团队都可以"add new ops, new types"，甚至"收纳成一个新 dialect"来解决自己的领域问题，而不需要修改 MLIR 核心或说服上游接受自己的抽象——这是 MLIR 能同时支撑 TensorFlow、Flang、CIRCT（硬件描述）等差异巨大的项目的根本原因。

### 3.2 Region / 嵌套 IR

**是什么**：Operation 可以携带若干个 Region，Region 内是 Block 的 CFG，Block 内又是 Operation 列表——递归定义。

**为什么这样设计**：LLVM IR 的指令不能嵌套指令，循环、条件等结构化控制流在 LLVM 层面必须被"拍扁"成 pre-header/header/latch/body 这种标准形态的基本块图。这种扁平化虽然便于统一处理，但一旦拍扁，循环的"结构"信息就丢了，后续如果还想做循环级变换（分块、向量化、多面体分析）就要重新从 CFG 里"raise"结构，既昂贵又不可靠。MLIR 选择把嵌套 Region 作为**一等公民**，允许在需要结构化信息的层级（如 `affine.for`、`scf.for`）保留循环树，只在真正需要扁平 CFG 语义的层级（如 `cf.br`/`llvm` dialect）才拍扁。

#### 示例精讲：同一个循环，保结构 vs 拍扁

> 目的：亲眼看到「拍扁之后循环结构去哪了」，从而理解为什么 MLIR 要保留 Region。

**保结构（`scf.for`，循环体在 Region 里）**

```mlir
%sum = scf.for %i = %c0 to %c8 step %c1 iter_args(%acc = %init) -> (i32) {
  %v = memref.load %buf[%i] : memref<8xi32>
  %n = arith.addi %acc, %v : i32
  scf.yield %n : i32
}
```

这里「哪个是归纳变量、上下界是多少、循环体是哪一段」都是 IR 里**写着的**，tiling / 展开只需读这几个字段。

**拍扁（`cf` 分支，标准 CFG 形态）**

```mlir
  cf.br ^header(%c0, %init : index, i32)
^header(%i: index, %acc: i32):
  %cond = arith.cmpi slt, %i, %c8 : index
  cf.cond_br %cond, ^body, ^exit(%acc : i32)
^body:
  %v = memref.load %buf[%i] : memref<8xi32>
  %n = arith.addi %acc, %v : i32
  %i2 = arith.addi %i, %c1 : index
  cf.br ^header(%i2, %n : index, i32)
^exit(%r: i32):
```

语义一样，但「这是一个从 0 到 8 步长 1 的循环」这句话已经不再直接写在 IR 里——要靠循环识别与归纳变量分析**重新推回来**。

| | `scf.for`（保结构） | `cf.*`（拍扁） |
|--|--------------------|----------------|
| 循环边界 | op 的操作数，直接可读 | 藏在 `cmpi` + 分支模式里 |
| 循环体 | 一个 Region | 若干 Block，需靠支配关系判定 |
| 做 tiling | 读字段即可 | 先做循环识别，代价高且易失败 |
| 何时该用 | 高层变换阶段 | 接近 LLVM、准备最终 codegen 时 |

> **自测**：如果一条 pipeline 在很早期就把 `scf.for` 降成 `cf.*`，后面还想做分块，会多付出什么代价？

**带来什么能力**：编译器可以在同一 IR 内让"结构化的高层部分"与"已经拍扁的低层部分"共存——比如一个自定义加速器编译器可以复用 MLIR 提供的高层循环结构，同时混入自己特有的标量/向量指令。这直接服务于"maintain higher-level semantics"这一设计目标。

### 3.3 Progressive Lowering / 渐进式下降

**是什么**：从高层表示到最低层表示之间，通过多个中间抽象层级、多个小步骤完成下降，而不是一步到位。

**为什么这样设计**：论文指出编译 pass 大致可分四类角色——优化变换、使能变换（enabling transformation）、lowering、清理（cleanup）——传统编译器往往把它们按固定的、粗粒度的阶段顺序串联（如 Open64 WHIRL 的 5 级、Clang 的 AST→LLVM IR→SelectionDAG→MachineInstr→MCInst 固定管线），扩展新阶段很僵化。MLIR 允许在**单个 Op 的粒度**上混合这四种角色，而不是必须让整个编译单元统一走完一个阶段才能进入下一阶段。

**带来什么能力**：新增一个抽象层级的成本大幅降低（只需定义新 dialect + 写清楚它到邻近层的转换 pattern），并且允许一部分 IR 已经下降到低层、另一部分仍停留在高层（见 3.4 的部分转换）——这正是 `mlir-toy-dialect` 里 `toy → low` 两层 dialect 演示的核心思想。

### 3.4 Dialect Conversion 与部分转换（Partial Conversion）

**是什么**：以 `ConversionTarget` 声明"目标合法状态"，以 `TypeConverter` 处理类型级映射，以 `ConversionPattern` 执行 Op 级重写，框架自动检查转换后的 IR 是否全部落在合法状态内；"部分转换"允许合法状态里仍包含一部分未被转换的原 dialect Op。

**为什么这样设计**：单纯的贪心 `RewritePattern` 重写不保证"收敛到一个合法的目标状态"——它只是不断重写直到没有 pattern 能匹配为止，不检查最终结果是否满足下一阶段的假设。当转换涉及**类型变化**（如 `tensor` → `memref`，需要同步改写所有引用该类型的函数签名、block argument）时，必须有专门的机制去传播类型转换。Dialect Conversion 框架就是为这种"整层/半层"迁移设计的。

**带来什么能力**：可以安全地实现"只转换一部分 dialect，保留另一部分不动"的渐进式 pipeline（例如 `--convert-linalg-to-affine` 只处理 `linalg` op，不动 `arith`/`scf`），并且转换失败会被显式报告（而不是静默留下不合法 IR）。这是真实 MLIR pipeline（TF/XLA、IREE）里做多阶段 lowering 的标准做法；需要注意的是，本文档第 6 节会指出 `mlir-toy-dialect` 目前用的是普通 `RewritePattern` + 贪心驱动器做 lowering，**没有**使用完整的 Dialect Conversion 框架（因为两层都只涉及 `i32`，不需要类型转换）。

### 3.5 Traits / Interfaces 带来的"跨 dialect 复用通用 pass"能力

**是什么**：Trait 是编译期附着在 Op 类型上的属性标记（如 `Commutative`、`Pure` 无副作用、`IsolatedFromAbove`），Interface 是一组具名的虚方法契约（如 `DialectInlinerInterface` 要求实现 `isLegalToInline` / `handleTerminator`）。

**为什么这样设计**：论文提出一个尖锐的问题——"当操作和类型系统都开放可扩展时，如何写一个通用的编译器 pass？"完全保守（对未知 op 一律不动）能保证正确但收益有限。论文给出四种应对策略，由轻到重：① 基础 Trait（DCE/CSE 只需要"无副作用"、"可交换"这类简单位标记）；② 特权钩子（constant folding、`getCanonicalizationPatterns`，需要少量 C++ 代码但仍是通用机制）；③ Optimization Interface（如 inliner 不知道具体是 TensorFlow 图还是 Flang 函数，但只要 dialect 实现了 `DialectInlinerInterface`，通用 inliner 就能正确工作）；④ dialect 专属 pass（不追求通用，直接写死语义）。

#### 示例精讲：一个 Pass 同时服务两个互不相识的 dialect

> 目的：把「策略③ Optimization Interface」落到可读的代码与 IR 上。

假设两个 dialect 各自有一个"很贵"的算子，都实现了同一个接口 `ToyCostOpInterface`（仓库 `mlir-toy-dialect` 里的教学版）：

```mlir
func.func @mixed(%a: i32, %b: i32) -> i32 {
  %0 = toy.mul %a, %b : i32     // 实现了接口，getCost() = 3
  %1 = low.add %0, %b : i32     // 实现了接口，getCost() = 1
  %2 = arith.addi %1, %a : i32  // 【未】实现接口
  return %2 : i32
}
```

通用 Pass 完全不 `#include` 任何一个 dialect 的头文件：

```cpp
void runOnOperation() override {
  int64_t total = 0;
  getOperation()->walk([&](Operation *op) {
    if (auto costOp = dyn_cast<ToyCostOpInterface>(op))   // 只认契约
      total += costOp.getCost();
    // 没实现接口的 op（如 arith.addi）直接跳过，不影响正确性
  });
}
```

结果：`total = 3 + 1 = 4`；`arith.addi` 被安全忽略。

| 策略 | 本例对应物 | 代价 |
|------|-----------|------|
| ① Trait | 若只需判断"能否删"，`Pure` 就够 | 最轻，但表达力最弱 |
| ③ Interface | `ToyCostOpInterface` | 每个 dialect 实现一次，通用 Pass 写一次 |
| ④ 专属 Pass | 直接 `if (op名 == "toy.mul")` | 每加一个 dialect 就要改 Pass |

这就是"开放 op 集合上仍能写通用 pass"的工程答案：**Pass 依赖接口，不依赖 op 名单**。

> **自测**：给 `arith.addi` 也接上该接口，需要改动通用 Pass 的代码吗？

**带来什么能力**：一套通用 pass（inliner、canonicalizer、CSE、DCE）可以同时服务几十个语义完全不同的 dialect，而各 dialect 只需要按需实现相应的 Trait/Interface，不需要修改通用 pass 本身——这是"add new ops/types"这一开发范式能够成立的关键支撑。

### 3.6 声明式（ODS/TableGen、PDL/DRR 重写规则）

**是什么**：用 TableGen 语言写 `.td` 文件声明 Op 的名字、参数、结果、trait、自定义打印格式（ODS），以及用同样是 TableGen 语言写的 Declarative Rewrite Rule（DRR）表达"源 DAG 模式 → 目标 DAG 模式"的等价改写规则。两者最终都被 `mlir-tblgen` 编译成 C++ 代码。

**为什么这样设计**：论文提出"定义新变换应该和定义新抽象一样简单"，而且声明式规则更容易做机器分析（比如判断改写系统是否终止、是否保持某些不变式），比手写的命令式 C++ 重写代码更利于长期维护和形式化验证研究。

**带来什么能力**：常见的"peephole"级别代数化简（如论文图 6 的 `LeakyRelu → Compare + Select`）几行 TableGen 就能表达，且自动生成的匹配代码经过统一优化（论文提到甚至可以把 rewrite pattern 本身表示成一个 MLIR dialect，用 FSM 优化匹配过程，供硬件厂商在驱动里动态扩展 lowering 规则）。需要更复杂控制流的规则仍可以退回手写 C++ `RewritePattern`，二者可以混用。

### 3.7 位置信息（Location）与源码可追溯性、Verifier 带来的可调试性

**是什么**：每个 Op 携带一个可扩展的 `Location`（可以是文件:行:列、AST 节点引用、DWARF 调试信息，甚至是"由某次变换从另一个 Location 派生而来"的复合 Location），配合结构化的 Verifier 在每个 Op/Attribute 层面检查不变式。

**为什么这样设计**：论文指出复杂编译系统普遍存在"lack-of-transparency"问题（WYSINWYX——你看到的不是你得到的），这在安全敏感场景（密码学代码、需要软件认证的系统）尤其致命，因为优化可能悄悄破坏某些非功能性属性（如抗侧信道特性）却不留痕迹。可追溯的位置信息让"这段代码是怎么从源码一步步变成现在这样的"始终可查。

#### 示例精讲：一段违规 IR 与它触发的检查

> 目的：把「Verifier 失败即停」从口号变成能认出来的报错形态。

**违规一：Block 没有终结符**

```mlir
func.func @no_terminator(%a: i32) -> i32 {
  %s = arith.addi %a, %a : i32
  // 缺 return：Block 必须以 Terminator 结尾
}
```

**违规二：使用了不支配自己的值**

```mlir
func.func @dominance(%c: i1) -> i32 {
  cf.cond_br %c, ^a, ^b
^a:
  %x = arith.constant 1 : i32
  cf.br ^b
^b:
  return %x : i32        // ^b 也可能从入口块直接来，此时 %x 未必已定义
}
```

两类检查的分工：

| 检查层 | 谁定义的 | 例子 |
|--------|----------|------|
| 结构不变式（框架统一做） | MLIR 核心 | 终结符完整性、SSA 支配关系、符号唯一性 |
| Op 自定义约束 | 该 Op 的 `verify()` | 「`toy.repeat` 的次数必须 > 0」这类语义约束 |

**Location 的价值**：每个 op 带位置，重写后还能保留派生关系，于是诊断能指回源头：

```mlir
%0 = toy.add %a, %b : i32 loc("model.mlir":3:5)
```

一次 pattern 重写把两条 op 合成一条时，新 op 的 Location 可以是二者的融合位置，从而回答「现在这条指令是从原来哪两句来的」。

> **自测**：违规二如果把 `return %x` 换成 `return %c` 会不会仍然报错？为什么？

**带来什么能力**：诊断信息可以精确定位到源码；测试可以用"文本 IR 作为输入 + 文本 IR 作为期望输出"的方式对单个 pass 做隔离测试（因为 IR 无隐藏状态，pass 的输出只依赖输入 IR，不依赖运行历史）；Verifier 失败会立刻中止编译并指出具体是哪个 Op 违反了哪条约束，而不是让错误的 IR 静默流入下一个 pass 产生更难排查的连锁错误。

### 3.8 泛型 Pass 与 Pass Pipeline 的组织方式

**是什么**：PassManager 不绑定固定粒度，Pass 可以注册在任意 Op 类型上运行；多个 Pass 通过嵌套的 pipeline 字符串（如 `builtin.module(func.func(canonicalize,cse))`）组织执行顺序和作用范围。

**为什么这样设计**：因为 module/function 在 MLIR 里"不特殊"（只是带 Region 的普通 Op），如果 PassManager 只支持固定几种粒度，就无法适配用户自定义的、带 Region 的容器 Op（比如一个表示"kernel"或"设备子图"的自定义 Op）。

**带来什么能力**：pipeline 可以按需在恰当的嵌套层级运行恰当的 pass（例如只在每个函数体内部跑规范化，而不触碰模块级的全局符号），同时天然支持"isolated-from-above"子树的并行编译。

---

## 4. 使用示例

### 4.1 通用文本 IR（Generic Syntax）逐行解析

论文 Figure 3 给出了 MLIR 最原始的"通用语法"（不依赖任何自定义打印格式，任何 dialect 的任何 Op 都可以用这种形式写出来）：

```mlir
%results:2 = "d.operation"(%arg0, %arg1) ({
// Region 属于 Op，可以包含多个 Block。
^block(%argument: !d.type):
  // Op 的类型是函数类型（表达 operand -> result 的映射）。
  %value = "nested.operation"() ({
    // Op 可以嵌套 Region。
    "d.op"() : () -> ()
  }) : () -> (!d.other_type)
  "consume.value"(%value) : (!d.other_type) -> ()
^other_block:
  "d.terminator"() [^block(%argument : !d.type)] : () -> ()
}) {attribute = "value" : !d.type} : () -> (!d.type, !d.other_type)
```

逐行拆解语法元素：

- `%results:2 = "d.operation"(%arg0, %arg1) (...)`：`%results:2` 表示这个 Op 产生一个"打包"的、含 2 个值的结果组，用 `%results#0`/`%results#1` 可取出单个值；`"d.operation"` 是**用引号包裹的 opcode 字符串**，`d` 是 dialect 前缀；`(%arg0, %arg1)` 是操作数（operand）列表。
- `({ ... })`：紧跟在操作数之后、用 `({` `})` 包裹的部分是这个 Op 附带的**Region**。
- `^block(%argument: !d.type):`：Region 内第一个 Block 的标签是 `^block`，它带一个类型为 `!d.type` 的 **Block Argument**（`!` 前缀表示这是一个自定义/dialect 类型）；进入 Region 时，Op 的语义决定这个入口 Block 的参数从何而来（对 `affine.for` 而言就是循环归纳变量）。
- `%value = "nested.operation"() ({ "d.op"() : () -> () }) : () -> (!d.other_type)`：说明 Op 可以**递归嵌套**——这个内层 Op 自己又带了一个 Region，Region 里是叶子 Op `"d.op"`。
- `^other_block: "d.terminator"() [^block(%argument : !d.type)] : () -> ()`：Region 内第二个 Block `^other_block` 以 `d.terminator` 结尾，方括号 `[^block(%argument : !d.type)]` 是**后继 Block 及要传递的参数**——这正是 MLIR 用"给后继 Block 传参数"代替传统 φ 节点的具体写法。
- `{attribute = "value" : !d.type}`：紧跟在操作数与 Region 之后、类型标注之前的花括号是**属性字典**（Attribute dictionary），key 是 `attribute`，value 是一个带类型的字符串常量。
- `: () -> (!d.type, !d.other_type)`：整个 Op 末尾的函数式类型标注，`() -> (...)` 前半部分是（此处省略，因为操作数已在括号中给出）操作数类型，后半部分是结果类型列表。

真实场景很少写这种"通用语法"，因为每个 Op 通常会定义自己的自定义打印/解析格式（见 4.2 的 `assemblyFormat`），下面用论文 Figure 8 的 `affine`/`arith` 例子看更贴近日常的可读语法（多面体多项式乘法内核 `C(i+j) += A(i) * B(j)`）：

```mlir
// affine.for 是带 Region 的 Op，循环体是它的单 Block Region。
affine.for %arg0 = 0 to %N {
  // 嵌套的内层循环——Region 天然表达循环嵌套树，无需拍扁成 CFG。
  affine.for %arg1 = 0 to %N {
    // affine.load 的下标必须是循环归纳变量的仿射（affine）表达式；
    // 结果类型 f32 由 memref 的元素类型决定。
    %0 = affine.load %A[%arg0] : memref<?xf32>
    // memref 类型可以携带"仿射布局映射"，把逻辑下标空间和实际地址空间分离：
    // (d0)[s0] -> (d0 + s0) 表示第 d0 维下标加符号常量 s0 才是真实地址偏移。
    %1 = affine.load %B[%arg1] : memref<?xf32, (d0)[s0] -> (d0 + s0)>
    // mulf 来自 arith dialect（论文中写作 std dialect），与 affine dialect 混用。
    %2 = mulf %0, %1 : f32
    %3 = affine.load %C[%arg0 + %arg1] : memref<?xf32>
    %4 = addf %3, %2 : f32
    affine.store %4, %C[%arg0 + %arg1] : memref<?xf32>
  }
}
```

要点：`affine.for`/`affine.load`/`affine.store` 是 `affine` dialect 的 Op（负责结构化循环与仿射下标合法性），`mulf`/`addf` 是算术 dialect 的 Op（负责标量计算），二者在同一段 IR 里**混合共存**——这就是 3.2 节和第 6.2 节讨论的"dialect 混合"在真实代码里的样子。

### 4.2 ODS（.td）定义 Op 的最小示例

论文 Figure 5 给出的 `LeakyReluOp` 定义（结构与本仓库 `mlir-toy-dialect/include/Toy/ToyOps.td` 里的 `Toy_ConstantOp`/`Toy_AddOp` 完全同构）：

```tablegen
// Op 是一条 TableGen 定义，继承自 "Op" 类，用 mnemonic（操作名字符串）参数化，
// 并附带一组用于验证/优化的 Trait 列表。
def LeakyReluOp : Op<"leaky_relu",
    [NoSideEffect, SameOperandsAndResultType]> {
  // 一行摘要，用于生成文档。
  let summary = "Leaky Relu operator";
  // 完整描述，同样用于文档生成。
  let description = [{
    Element-wise Leaky ReLU operator
    x -> x >= 0 ? x : (alpha * x)
  }];
  // 具名参数：既包含类型化的操作数（AnyTensor:$input），也包含属性（F32Attr:$alpha）。
  let arguments = (ins AnyTensor:$input, F32Attr:$alpha);
  // 具名结果。
  let results = (outs AnyTensor:$output);
}
```

字段含义：

- `Op<"leaky_relu", [...]>`：第一个参数是操作名（配合所属 Dialect 会拼成 `mydialect.leaky_relu`），第二个参数是 Trait 列表。
- `arguments`：`ins` 里既可以放操作数（带类型约束，如 `AnyTensor`），也可以放属性（如 `F32Attr`），二者语法统一，用 `$name` 命名。
- `results`：`outs` 里放结果的类型约束和名字。
- `assemblyFormat`（本例未展示，`mlir-toy-dialect` 里大量使用，如 `"$value attr-dict \`:\` type($result)"`）：声明式定义自定义文本语法，替代 4.1 节又长又难读的通用语法。
- `hasCanonicalizer` / `hasFolder`：声明"我会提供 C++ 实现的规范化模式 / `fold()` 方法"，`mlir-tblgen` 会生成对应的虚函数声明，具体实现留给 `.cpp` 文件（对照仓库 `ToyOps.cpp` 里 `ConstantOp::fold`/`AddOp::fold`/`MulOp::fold` 的写法）。

### 4.3 C++ RewritePattern / ConversionPattern 最小骨架

**普通 `OpRewritePattern`**（用于同 dialect 内的代数化简，或不涉及类型变化的跨 dialect lowering，本仓库 `lib/ToyPasses.cpp`、`lib/LowPasses.cpp` 全部采用这种形式）：

```cpp
// 只匹配 MyMulOp 这一种 Op。
struct SimplifyMulByOne : public OpRewritePattern<MyMulOp> {
  using OpRewritePattern<MyMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MyMulOp op,
                                 PatternRewriter &rewriter) const override {
    // 1. 匹配：检查是否满足改写前提（这里检查右操作数是否是常量 1）。
    if (auto c = op.getRhs().getDefiningOp<MyConstantOp>();
        c && c.getValue() == 1) {
      // 2. 重写：用左操作数直接替换掉整个 op（及其所有使用者）。
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    // 不匹配时必须返回 failure()，交给其他 pattern 继续尝试。
    return failure();
  }
};

// 注册到 PatternSet，交给贪心驱动器反复应用直到不动点。
RewritePatternSet patterns(context);
patterns.add<SimplifyMulByOne>(context);
(void)applyPatternsAndFoldGreedily(op, std::move(patterns));
```

**`ConversionPattern`**（用于需要 `TypeConverter` 参与的方言转换，例如某类型要在两个 dialect 间变化）：

```cpp
// ConversionPattern 比 OpRewritePattern 多一个 TypeConverter 参数，
// matchAndRewrite 拿到的 operands 是"已经按目标类型转换过"的版本。
struct ConvertMyOp : public ConversionPattern {
  ConvertMyOp(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter, MyOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // 用转换后的 operands 和目标 dialect 的类型构造新 Op。
    auto resultType = getTypeConverter()->convertType(op->getResult(0).getType());
    rewriter.replaceOpWithNewOp<TargetOp>(op, resultType, operands);
    return success();
  }
};

// 驱动方式不同于贪心重写：需要显式声明 ConversionTarget（合法/非法状态）。
ConversionTarget target(*context);
target.addLegalDialect<TargetDialect>();
target.addIllegalOp<MyOp>(); // 转换后不允许再出现 MyOp
TypeConverter typeConverter;
typeConverter.addConversion([](MySourceType t) { return TargetType::get(t.getContext()); });
RewritePatternSet patterns(context);
patterns.add<ConvertMyOp>(typeConverter, context);
// applyPartialConversion 允许残留部分未转换的合法 Op；
// applyFullConversion 要求转换后不能有任何非法 Op。
if (failed(applyPartialConversion(module, target, std::move(patterns))))
  return failure();
```

`mlir-toy-dialect` 目前的 `toy-to-low` lowering 用的是第一种（普通 `OpRewritePattern` + 贪心驱动器），因为 `toy`/`low` 两层都只涉及 `i32`，不需要 `TypeConverter`；第二种 `ConversionPattern` 骨架是真实 MLIR pipeline（如 `--convert-linalg-to-loops`、`--convert-scf-to-cf`）里更常见的写法，是第 6 节要点出的"仓库尚未覆盖"的机制之一。

### 4.4 常用命令行示例

```bash
# 1. 用 mlir-opt 跑一条 pass pipeline：先做 canonicalize，再做 CSE，
#    嵌套写法 module(func.func(...)) 表示只在每个函数体内部执行。
mlir-opt input.mlir --pass-pipeline='builtin.module(func.func(canonicalize,cse))'

# 2. 把已经降到 llvm dialect 的 MLIR 翻译成真正的 LLVM IR 文本。
mlir-translate --mlir-to-llvmir lowered.mlir -o output.ll

# 3. 本仓库的 toy-opt：只解析、不做任何变换（验证 dialect 能正确解析/打印）。
toy-opt test/ops.mlir

# 4. 跑常量折叠（--canonicalize 驱动 ConstantOp/AddOp/MulOp 的 fold()）。
toy-opt test/canonicalize.mlir --canonicalize

# 5. 跑代数化简（自定义 Pass --toy-simplify，用 RewritePattern 实现 x*1=x, x+0=x）。
toy-opt test/simplify.mlir --toy-simplify

# 6. 渐进式下降：toy.* -> low.*，再在低层做强度削减 mul->shl。
toy-opt test/strength.mlir --toy-to-low --low-strength-reduce

# 7. 用 lit + FileCheck 跑全部回归测试（round-trippable 文本 IR 的自动化验证）。
cmake --build build --target check-toy
```

---

## 5. 关键评估结论

论文没有做微观 benchmark，而是用"被多少真实系统采用、解决了什么问题"作为核心评估证据：

- **社区规模**：截至发表时已有超过 26 个 dialect 在公开或私有开发中，7 个公司级项目正在用 MLIR 替换自研基础设施；HPC 相关的 MLIR workshop 吸引了 16 所高校、4 个国家实验室参与；LLVM Developer Meeting 上 100+ 工业界开发者参加了 MLIR 圆桌会议。
- **TensorFlow / XLA**：MLIR 被用来建模 TensorFlow 的数据流图（含设备放置、异步执行、`switch`/`merge` 控制结构），支撑从代数简化、面向数据中心加速器集群的并行化重定向，到移动端部署的下降，再到用 XLA 生成高效原生代码的完整链路。
- **多面体代码生成（affine dialect）**：相比传统多面体框架，`affine` dialect 用"仿射映射作为属性、循环仍是 Op"而非把整段代码抽成纯数学多面体，带来四点收益——① memref 的布局映射把数据变换和循环变换解耦；② 循环体内仍是普通 SSA 值，可与传统编译器分析/变换交织使用；③ 不需要"从变换后的多面体重新生成循环"这一计算复杂度很高的步骤（因为循环结构从未丢失）；④ 编译速度更快（不依赖整数线性规划求解器）。
- **Flang 的 FIR（Fortran IR）**：NVIDIA/PGI 主导的 LLVM Fortran 前端用 MLIR 构建专属 IR，第一类支持虚函数分发表（`fir.dispatch_table`），使得健壮的去虚化（devirtualization）优化可以直接在 IR 层面实现；同时可以复用与语言无关的 OpenMP/OpenACC dialect，在异构平台（如通过 GPU dialect）上与 C/Fortran 共享基础设施，而不必各自重新实现。
- **领域专用小型编译器**（如 Lattice Regression 编译器）：原实现基于 C++ 模板元编程，性能好但难以做端到端的图级优化；改用 MLIR 重写后，仅投入约 3 人月工程量，在生产模型上取得最高 8× 的性能提升，同时编译过程的可观察性也明显提升。
- **可动态扩展的 pattern rewriting**：把 rewrite pattern 本身表示成一个 MLIR dialect，允许硬件厂商在驱动层动态注入新的 lowering 规则，并借助有限状态机（FSM）做匹配优化——效果上对标 LLVM SelectionDAG/GlobalISel 里的指令选择技术。

论文也坦诚指出局限：MLIR 提供的是"能力"而非"最佳实践"（unopinionated design），如何设计一个好的 dialect/抽象仍是一门尚不成熟的"艺术"，需要社区持续积累经验。

---

## 6. 与本仓库 `mlir-toy-dialect` 实践的对应关系

`mlir-toy-dialect` 是一个树外（out-of-tree）最小 MLIR 项目，包含 `toy`（高层，数学语义）与 `low`（低层，贴近硬件，多了 `shl` 移位）两个 dialect。逐一对照论文机制：

| 论文机制 | 本仓库对应文件/代码 | 覆盖情况 |
|---|---|---|
| Dialect 定义（命名空间、`hasConstantMaterializer`） | `include/Toy/ToyDialect.td`、`include/Low/LowDialect.td` | 已覆盖：两个独立 dialect，各自命名空间 `toy.`/`low.` |
| ODS 定义 Op（arguments/results/traits/assemblyFormat/hasFolder） | `include/Toy/ToyOps.td`、`include/Low/LowOps.td` | 已覆盖：`Toy_ConstantOp`/`Toy_AddOp`/`Toy_MulOp` 用 `Pure`/`Commutative`/`ConstantLike` trait，自定义 `assemblyFormat`，声明 `hasFolder = 1` |
| Op 语义实现（`fold()` 常量折叠） | `lib/ToyOps.cpp`（`ConstantOp::fold`/`AddOp::fold`/`MulOp::fold`）、`lib/LowOps.cpp` | 已覆盖：对应论文 3.6/4.1 节的"privileged operation hooks"里的常量折叠部分 |
| RewritePattern + 贪心驱动器（3.6 声明式重写的手写 C++ 版本） | `lib/ToyPasses.cpp` 的 `SimplifyMulByOne`/`SimplifyAddZero` | 已覆盖：手写 `OpRewritePattern`，对应论文"declarative rewrite patterns"设计原则的命令式落地（未使用 DRR/PDL 的声明式 `.td` 写法） |
| 自定义 Pass（`PassWrapper`，对应论文的 dialect-specific pass / 4.3 PassManager） | `include/Toy/ToyPasses.h`、`lib/ToyPasses.cpp` 里的 `ToySimplifyPass` | 已覆盖：手写 `PassWrapper<..., OperationPass<>>`，命令行开关通过 `getArgument()` 注册 |
| Progressive Lowering（渐进式下降，3.3 节） | `lib/LowPasses.cpp` 的 `ToyToLowPass`（`toy.* → low.*`） | 已覆盖：用 `RewritePattern` 逐 Op 替换，`getDependentDialects` 声明依赖，体现"换层不换优化"的下降步骤 |
| "在恰当层级做恰当优化"（多层 dialect 分工，对应论文 5.2/6.2 混合 dialect 思想） | `lib/LowPasses.cpp` 的 `LowStrengthReducePass`（`low.mul x,2^k → low.shl x,k`） | 已覆盖：`README.md` 第五点五节的核心演示——`x*4` 在 `toy` 层因为没有 `shl` 概念而无法优化，必须先降到 `low` 层才"看得见"这个优化机会，直观复现了论文"maintain higher-level semantics + progressive lowering"的设计动机 |
| mlir-opt 风格 CLI、DialectRegistry 注册 | `tools/toy-opt/toy-opt.cpp` | 已覆盖：`registerAllDialects` + 自定义 dialect `insert`，复用 `MlirOptMain` |
| Round-trippable 文本 IR + 自动化验证（4.4 节） | `test/*.mlir` + `test/lit.cfg.py`（`check-toy` 目标） | 已覆盖：`RUN:`/`CHECK:` 形式的 lit 测试，验证文本 IR 输入输出的往返一致性 |
| Region / 嵌套 IR 树（3.2 节） | `knowledge.md` | 已覆盖（以笔记形式）：清楚画出 `ModuleOp ⊃ FuncOp ⊃ toy.constant` 的嵌套关系，并指出 `toy.*` 都是 `ZeroRegions` 叶子 Op |

**尚未覆盖、后续学习需要补的机制**：

1. **Interface（Optimization Interface / DialectInterface）**：`toy`/`low` 都没有定义任何 Interface（如 inliner interface），没有演示论文 3.5/6.1 节"跨 dialect 复用通用 pass"的完整能力——目前的两个 Pass 都是 dialect-specific pass。
2. **Dialect Conversion 框架（`TypeConverter`/`ConversionTarget`/legality/`ConversionPattern`）**：`toy-to-low` 用的是普通 `OpRewritePattern` + 贪心重写，不是标准的 `applyPartialConversion`/`applyFullConversion` 流程，因为两层都只用 `i32`，没有类型变化的场景。真实项目（linalg→loops、tensor→memref）几乎必然要用到这套机制。
3. **Bufferization（tensor ↔ memref 转换）**：本仓库没有张量/内存引用类型，未涉及这个在真实 DL 编译器（linalg/TOSA → memref）里非常核心的转换阶段。
4. **PDL / DRR 声明式重写规则**：所有 pattern 都是手写 C++，没有用 TableGen 的 `.td` 声明式写法（对应论文 4.2 节、图 6）。
5. **Symbol / SymbolTable 的显式使用**：目前只用到内建 `func.func` 自带的符号机制，没有自定义带符号表的 Op。
6. **多线程/isolated-from-above 的显式验证**：论文 4.3 节的并行编译能力在小项目里难以体现，仓库未涉及。
7. **自定义 Location 传播**：没有演示如何在 lowering 过程中保留/构造复合 Location 以追溯"这个 low.shl 是从哪个 toy.mul 变换而来"。

---

## 7. 学习这篇论文时的最小必要集

**必须掌握（做 AI 编译 / 多后端基础设施绕不过去的 6-8 个点）**：

1. **Operation 是唯一语义单元，Region 是嵌套的来源**——理解"MLIR IR 是一棵树"而不是一条扁平指令序列，这决定了后面理解 `affine`/`scf`/`linalg` 的循环、条件表达方式。
2. **Dialect 划分与多 dialect 混合共存**——理解"没有内建操作集"意味着每个新领域都是"加 dialect"而不是"改核心"，这是 MLIR 生态扩张（TF/CIRCT/Flang/IREE/Triton）的根本机制。
3. **ODS/TableGen 定义 Op 的基本字段**（`arguments`/`results`/traits/`assemblyFormat`/`hasFolder`/`hasCanonicalizer`）——后续读任何 dialect 源码（`linalg`、`tosa`、`stablehlo`）第一步都是看它的 `.td` 定义。
4. **渐进式下降 + Dialect Conversion 的机制**（`TypeConverter`/`ConversionTarget`/partial vs full conversion）——这是从图级 IR 走到 IREE/ONNX Runtime 多后端委托时天天打交道的核心工具，即使本仓库的 demo 还没用到，也要提前建立心智模型。
5. **RewritePattern + 贪心驱动器**，以及它与 Dialect Conversion driver 的适用场景区别（同层代数化简 vs 跨层类型变化）。
6. **Trait / Interface 实现跨 dialect 复用通用 pass 的思路**——理解 inliner、canonicalizer 这类"框架自带但靠 dialect 实现契约获得具体行为"的设计模式，为后续读 TVM/Glow 里类似的 pass 抽象打基础。
7. **PassManager 的嵌套 pipeline 组织方式**（`builtin.module(func.func(...))`）——写/调试任何真实 MLIR pipeline 都要会读、会写 pass pipeline 字符串。
8. **Location/Verifier 带来的调试习惯**——尤其是"用文本 IR 做 pass 级隔离测试"这一方法论，直接迁移到日后调试 TVM/IREE 的 pass 时同样适用。

**可以先跳过、以后遇到再深入的内容**：

- **具体各 dialect 内部细节**（`affine` 多面体表达式的完整数学定义、`linalg` 的具体算子集合）——用到哪个 dialect 再查对应文档，不必提前系统学习。
- **PDL（Pattern Descriptor Language）/ DRR 的底层实现细节**以及论文提到的 FSM matcher 优化——先会用手写 `RewritePattern` 即可，声明式重写框架属于"锦上添花"的效率工具。
- **Bufferization 的具体 pass 细节**——等进入"运行时与部署（IREE）"阶段，实际处理 tensor→memref 转换时再深入，不必现在啃。
- **MLIR 底层 C++ 模板机制**（`TypeID` 生成、Trait 用 CRTP 如何实现、ODS 生成代码的具体样板）——先当黑盒使用生成的 API（`getLhs()`/`getValue()`/`replaceOpWithNewOp`），不必读 `mlir-tblgen` 源码。
- **完整的 LLVM 后端代码生成细节**（`SelectionDAG`/`GlobalISel` 指令选择、寄存器分配）——除非要写具体硬件后端，否则只需知道"最终会走到 `llvm` dialect 再 `mlir-translate` 成 LLVM IR"这一层认知即可。
- **并行编译 / isolated-from-above 的工程实现**——这是 MLIR 内部工程优化，理解"存在这个机制、用于加速大规模编译"即可，不必深究调度细节。
