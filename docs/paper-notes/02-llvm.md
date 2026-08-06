# LLVM：面向"终身"程序分析与变换的编译框架，用一份低层但保留类型/数据流信息的 IR 统一编译时/链接时/运行时/空闲时优化

> **导航**：[笔记索引](README.md) · [自学枢纽](../README.md)（阶段 0） · [横切概念](../ai-compiler-foundations.md) §6  
> **配套阅读**：本篇是 2004 年论文，讲**当年为什么这样设计**；**今天 LLVM 的实际结构**（四层 IR、New Pass Manager、后端 CodeGen 七阶段）见 [`llvm-learning-guide.md`](../llvm-learning-guide.md)；动手在 [`llvm-hello-compile/`](../../llvm-hello-compile/)。

> 论文元信息
> - 标题：*LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation*
> - 作者/机构：Chris Lattner, Vikram Adve — University of Illinois at Urbana-Champaign
> - 发表：CGO 2004（原文档内标注为会议论文，机构主页 `http://llvm.cs.uiuc.edu/`）
> - 原文链接：论文 PDF 未包含 DOI，公开可查地址为 `https://llvm.org/pubs/2004-01-30-CGO-LLVM.pdf`

## 1. 它解决什么问题

2004 年前后，应用程序规模变大、行为随时间动态变化，且经常由多种语言混合实现，单纯依赖"编译时一次性优化"已经不够：很多优化机会只有在链接时（跨模块可见全部代码）、安装时（知道目标机器）、运行时（知道真实输入分布）甚至空闲时（可以离线做激进分析）才能被有效利用。论文把这种贯穿程序整个生命周期的分析与变换称为 lifelong program analysis and transformation（终身程序分析与变换）。当时已有的技术路线各有短板：传统源码级编译器只做静态编译时优化，不支持链接时/运行时优化；商业链接时优化器只覆盖链接阶段；JVM/CLI 等高层虚拟机能做运行时优化，但强绑定特定语言的运行时/对象模型，且字节码验证限制了编译时能做的激进变换；二进制运行时优化系统（如 Dynamo）只能在信息贫乏的机器码上做有限优化。论文提出 LLVM（Low Level Virtual Machine），核心问题是：**能否设计一套代码表示（IR）+ 编译框架，同时具备五种能力，而此前没有任何系统同时具备这五者**：

1. **终身编译模型（lifelong compilation model）**：只要保留 LLVM 表示，就能在应用生命周期的任意阶段（编译时/链接时/运行时/空闲时）继续做复杂的分析和优化。
2. **离线代码生成（offline code generation）**：即便支持终身优化，也依然能够离线用昂贵算法生成高质量的原生机器码——这对性能关键程序至关重要。
3. **用户侧 profile 驱动优化（user-based profiling and optimization）**：在真实终端用户的机器上采集运行时 profile，并把它用于运行时和空闲时的重优化，而不是依赖开发者提供的、可能不具代表性的训练输入。
4. **透明的运行时模型（transparent runtime model）**：不规定任何特定的对象模型、异常语义或运行时环境，因此可以承载任意语言（或多语言混合）编译到它。
5. **统一的全程序编译（uniform, whole-program compilation）**：语言无关性使得应用里所有代码（包括语言相关的运行时库、系统库）都能在链接后被统一地优化和编译。

传统源码级编译器只提供 2、4；商业链接时优化器额外提供 1、5（但仅限链接时）；JVM/CLI 一类高层虚拟机提供 3 和部分 1、5，却不提供 2、4；二进制运行时优化系统（如 Dynamo）提供 2、4、5，但不提供 1，对 3 的支持也很有限。

论文特别澄清了 LLVM 与 SmallTalk/Self/JVM/Microsoft CLI 这类高层虚拟机（High-Level Virtual Machine）的关系：二者是互补而非竞争，区别体现在三点。第一，LLVM 不表达类、继承、异常语义这类高层构造，即便正在编译的源语言本身有这些特性——它们在前端就被展开成低层原语（见第 3.2 节）。第二，LLVM 不规定任何运行时系统或对象模型，低层到可以直接**用 LLVM 本身去实现**某个语言的运行时系统（论文因此提出一个开放问题：能否把 JVM/CLI 这类高层虚拟机构建在 LLVM 之上）。第三，LLVM 不像类型安全语言的字节码那样提供类型安全、内存安全或语言互操作性的任何保证——它对这些性质的态度和物理处理器的汇编语言一致，安全性是上层（例如论文案例中的 SAFECode）自己在 LLVM 之上构建出来的能力，不是 LLVM 本身承诺的语义。理解这三点区别，有助于在后面读到"LLVM IR 保留类型信息"时不会误以为 LLVM 是一个类型安全的语言。

## 2. 整体运行框架

### 2.1 从源码到可执行文件的完整流水线

```
┌──────────────────────────── compile-time（每个 translation unit 独立编译） ────────────────────────┐
│                                                                                                    │
│   C/C++/Fortran/...  ──▶  Frontend（如 Clang）  ──▶  LLVM IR（.ll 文本 / .bc 位码）                 │
│   源码                     - 词法/语法/语义分析          - SSA 形式、无限虚拟寄存器                    │
│                            - 语言相关优化（可选）          - 携带类型信息、显式 CFG                    │
│                            - lower 到 LLVM 指令集         - 可选模块内优化 pass（如 -O1/-O2）         │
└────────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                              │ 每个模块各自的目标文件（内含 LLVM bytecode）
                                              ▼
┌──────────────────────────── link-time（whole-program 首次全部可见） ──────────────────────────────┐
│  Linker：合并所有模块的 IR  ──▶  IPO / IPA 跨模块优化器（LTO 的原型）                                 │
│                                  - 跨模块内联、死全局/死函数消除（DGE/DAE）                            │
│                                  - Data Structure Analysis（指针分析）、Automatic Pool Allocation    │
│                                  - 输出仍是 LLVM IR，可继续往下游任意阶段传递                          │
└────────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                              │ 优化后的整体 LLVM IR
                        ┌─────────────────────┴─────────────────────┐
                        ▼                                           ▼
┌──────────────────────────────────┐            ┌──────────────────────────────────┐
│ 离线 Native CodeGen（静态后端）    │            │ JIT 执行引擎                       │
│ - 可承受昂贵的指令选择/寄存器分配   │            │ - 运行时按需逐函数生成机器码         │
│ - 插入轻量运行时探测代码（热点循环）│            │ - 无 native codegen 时退化为解释器  │
└────────────────┬─────────────────┘            └──────────────────┬───────────────┘
                 │ 生成：目标机器码 + 可选内嵌 LLVM bytecode                            │
                 ▼                                                                  ▼
┌───────────────────────────── run-time（在终端用户的机器上执行） ───────────────────────────────────┐
│  exe 执行 → 运行时插桩探测热路径(path profiling) → 复制热 trace 并重做 LLVM 优化                     │
│           → 生成新机器码写入 software trace cache，缝合进原程序继续执行                              │
│           → 同时把 profile & trace info 落盘，供 idle-time 阶段使用                                 │
└────────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                              ▼
┌──────────────────────────── idle-time（Offline Reoptimizer，用户机器空闲时运行） ───────────────────┐
│  基于终端用户的真实 profile，对保留下来的整体 LLVM IR 重新做一轮更激进的跨模块 IPO + CodeGen，          │
│  替换可执行文件中的机器码部分；随使用模式变化可反复迭代                                                │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 逐组件说明

- **Frontend（前端，如后来的 Clang）**：输入是某种源语言（论文中为 C/C++），输出是 LLVM IR（`.ll` 文本或 `.bc` 位码）。职责有三项，其中第一和第三项是可选的：(1) 做语言相关优化（如高阶函数语言里的闭包优化）；(2) 把源程序翻译成 LLVM IR，尽量保留类型信息；(3) 调用 LLVM 的模块级/过程间优化 pass。关键设计：前端**不需要自己构造 SSA**——可以先把局部变量放在栈上（`alloca`，非 SSA），再交给 LLVM 自带的 stack promotion pass 提升为 SSA 寄存器，这大大降低了写前端的门槛。
- **Linker（链接器）+ IPO/IPA（Interprocedural Optimization/Analysis）**：链接时是程序生命周期中**第一个"几乎全部代码"都可见的阶段**（共享库、系统库可能仍不可见）。因为各编译单元产出的都是同一份 LLVM IR，链接器可以直接在 IR 层面合并模块，再跑跨模块的过程间优化（死全局消除 DGE、死参数消除 DAE、跨模块内联 inline、指针分析 DSA 等）。论文特别指出：编译期可以为每个函数预先计算"过程间摘要"并附着在 bytecode 里，链接时优化器直接消费摘要而不必从零分析，从而支持增量式的快速重编译。
- **离线 Native CodeGen（llc 一类的静态后端）**：把链接后优化过的整体 LLVM IR 降级（lower）为目标平台机器码。因为是离线运行，可以承受比运行时优化更昂贵的指令选择、寄存器分配、指令调度算法。生成机器码的同时，还会插入**轻量级探测指令**用于识别热点循环区域，并可选择把整份 LLVM bytecode 一并保存进最终可执行文件（保证后续运行时/离线优化器拿到的 IR 与实际运行的代码是同一份，不会有偏差）。
- **JIT 执行引擎**：另一条路径是不做离线 codegen，而是在运行时按函数粒度即时生成机器码（如果目标平台没有可用的原生 codegen，则退化为可移植的 LLVM 解释器）。
- **Runtime Optimizer（运行时优化器）**：程序运行时，通过离线插入的轻量探测 + 在线插桩库，先定位"热循环区域"，再定位区域内的"热路径"（hot path）。一旦确定热路径，就把对应的 LLVM IR 复制成一份 trace，对这份 trace 做 LLVM 级优化，重新生成机器码放入 software trace cache，再缝合回原程序执行流。这一步之所以可行，正是因为可执行文件里还保留着 LLVM IR——机器码天生缺乏做这种优化所需的类型/数据流信息。
- **Offline Reoptimizer（离线再优化器 / idle-time 优化器）**：运行时优化器为了不抢占用户 CPU 时间，只能做局部、轻量的优化；而离线再优化器在用户机器空闲时运行，利用运行时收集到的、来自**真实终端用户**（而不是开发者训练集）的 profile & trace 信息，对整份程序重新做一轮激进的跨模块优化和代码生成，直接替换可执行文件里的机器码。论文写作时这一阶段仍处于设计阶段，尚未完全实现。

### 2.3 "Lifelong" 是如何体现在分层上的

终身分析/变换的核心诉求是：**同一份程序表示要能在编译时、链接时、安装时、运行时、空闲时被反复分析和改写，而不产生语义损耗或表示转换代价**。LLVM 用两个关键设计支撑这一点：

1. **表示的可逆性/等价性**：LLVM IR 定义了完全等价的文本、位码（二进制）和内存三种表示，互相转换不丢信息（对比传统 JVM 需要把栈式字节码转换成另一种适合优化的内部表示，甚至再转成 SSA，转换本身有损耗且增加复杂度）。因此，无论在链接时、运行时还是空闲时拿到的都是"同一门语言"，可以复用同一套 pass 基础设施。
2. **贯穿各阶段但职责分层的组件划分**：前端只管"翻译到 LLVM"，链接器/IPO 只管"跨模块优化"，CodeGen 只管"IR→机器码"，运行时优化器只管"热路径局部重优化"，离线优化器只管"整体重优化"。各阶段共享同一份 IR、同一套优化 pass 库，新增一个优化 pass 就自动对所有阶段可用——这正是"框架"（而不是单一编译器）的含义。

## 3. 核心特性逐条拆解

### 3.1 LLVM IR 的设计：SSA、无限虚拟寄存器、类型系统、load/store 内存模型、getelementptr

**是什么**：LLVM IR 是一种抽象的、类似 RISC 的三地址指令集，只有 31 个 opcode，大致分成几类：

- **终结指令（terminator）**：`br`（条件/无条件分支）、`return`、`invoke`/`unwind`（异常控制流），每个基本块必须以且仅以一条终结指令结尾。
- **算术/逻辑指令**：`add`/`sub`/`mul`/`div`/`rem`、位运算 `and`/`or`/`xor`/`shl`/`shr`，均为三地址形式（一到两个操作数、产出一个结果）。
- **内存指令**：`load`/`store`（唯一的内存读写方式）、`malloc`/`free`（堆分配/释放）、`alloca`（栈帧内分配）、`getelementptr`（类型化地址计算）。
- **其他关键指令**：`phi`（SSA 汇合）、`cast`（类型转换，也是唯一的类型转换手段）、`call`（函数调用）。

这套指令集之所以能压缩到 31 个 opcode，一是避免为同一操作设多个 opcode（例如没有单独的一元 `not`/`neg`，而是用 `xor`/`sub` 表达），二是让大多数 opcode 支持类型重载（例如 `add` 同时处理整数和浮点操作数，具体语义由操作数类型在语义层面区分）。每个函数由若干基本块（basic block）组成，每个基本块以且仅以一条终结指令（`branch`/`return`/`invoke`/`unwind`）结束，终结指令显式列出后继基本块，因此**控制流图（CFG）是显式的**。所有值都存放在**无限个（infinite）带类型的虚拟寄存器**中，这些寄存器处于**静态单赋值（SSA, Static Single Assignment）**形式——每个寄存器只被定义一次，每次使用都被其定义"支配"（dominate）；分支汇合处用显式的 `phi` 指令对应标准的 φ 函数。内存访问方面，LLVM 是**load/store 架构**：寄存器与内存之间只能通过带类型指针的 `load`/`store` 指令传值，没有隐式内存访问，也不需要"取地址"运算符——全局变量、函数、栈上变量（`alloca`）本身都只是"提供地址的符号"。地址运算靠 `getelementptr` 指令完成：给定一个指向聚合类型（结构体/数组）对象的带类型指针，加上字段号或索引，它按类型规则计算出子元素的指针，相当于把 C 里的 `.` 和 `[]` 合并成一条保持类型信息的指令；`load`/`store` 本身只接受单个指针，不做任何下标运算。

**为什么这样设计**：如果直接用无类型的裸地址运算（像传统汇编或 GCC 的 RTL 中间表示那样），编译器在做别名分析、结构体重排、内存管理变换时几乎拿不到任何结构信息；但如果 IR 完全保留高层语言语义（像 AST 级 IR），又会丧失"与源语言无关"的通用性，且不适合运行时/链接时的重量场景。SSA + 无限虚拟寄存器则是数据流分析领域的成熟结论：SSA 让"寄存器不可能有别名"，很多传统需要昂贵流敏感（flow-sensitive）数据流分析才能做的优化，用流不敏感（flow-insensitive）算法就能达到接近的精度，同时大幅简化变换逻辑；无限虚拟寄存器则把"物理寄存器分配"这个高度机器相关的决策推迟到 CodeGen 阶段，让 IR 层完全不用关心目标机器的寄存器数量、调用约定等细节。

**带来什么能力**：类型信息 + `getelementptr` 让编译器可以在**不信任但可验证**的前提下使用类型（论文用 Data Structure Analysis 验证声明类型与实际访问是否一致，见第 5 节），从而安全地做结构体字段重排、自定义内存分配器插入（Automatic Pool Allocation）等激进变换——这些变换传统上只有在类型安全语言的源码级编译器里才敢做。显式 CFG + SSA 使得链接时/运行时这种对性能极度敏感的场景也能用便宜的算法拿到接近全局的分析精度。

### 3.2 低层但保留高层信息的取舍

**是什么**：LLVM 明确声明自己不是"通用编译器 IR"——它不表达高层语言构造（类、继承、异常语义本身），也不表达机器相关的特征（物理寄存器、流水线、调用约定），这些必须分别在前端做 lowering、在后端做 lowering。但同时它比传统汇编"富"：每个 SSA 寄存器和显式内存对象都带类型，控制流和数据流（SSA）都是显式的一等信息。

**为什么这样设计**：论文认为，如果 IR 试图表达所有源语言特性（如历史上 UNCOL、ANDF 这类"统一中间表示"的尝试），会因为要覆盖所有语言的语义而变得臃肿、难以推广，历史上这类方案基本都失败了；反过来，如果 IR 完全不带类型/结构信息（如传统汇编或早期基于 GCC RTL 的前端实践），链接时/运行时想做的深度分析（指针分析、内存变换）就没有可靠的信息源头。LLVM 选择的平衡点是：**只提供一小撮与语言无关但足以表达绝大多数高层语义"操作行为"的原语类型**（primitive 类型 + 4 种 derived 类型：pointer/array/struct/function），高层特性统一"翻译"成这些原语的组合——例如 C++ 的多重继承被展开成嵌套结构体类型，虚函数表被表示为一个只读的函数指针数组全局常量，异常处理被 lower 成 `invoke`/`unwind` 两条原语指令（同一机制也直接复用于实现 C 的 `setjmp`/`longjmp`）。

**带来什么能力**：一方面，这使 LLVM IR 可以对任意源语言保持中立（LLVM 本身不规定对象模型、异常语义、运行时系统，所以理论上可以在 LLVM 之上实现 JVM/CLI 这类高层虚拟机的运行时）；另一方面，因为高层语义都是"显式展开"而不是"隐式黑盒"，链接时的过程间分析可以直接看穿这些展开后的结构做优化——论文举例：把 C++ 虚函数调用解析、异常处理器裁剪都下放到通用的 LLVM 优化器里做，效果比大多数只能看到单个编译单元的源码级编译器更好，因为 LLVM 优化器天然工作在链接时的全程序视图上。代价是：语言相关的优化（例如闭包特化）必须在前端完成，LLVM 层面看不到任何"这是不是一个 lambda"这样的信息。

### 3.3 Pass 基础设施与 Pass Manager（analysis pass vs transform pass）

**是什么**：论文里的优化能力都是以"pass"为单位组织的（DGE 死全局消除、DAE 死参数消除、inline 内联、DSA 数据结构分析、stack promotion 提升到 SSA 等），并且"LLVM optimizations are built into libraries"——优化被实现成可复用的库，前端/链接器/运行时优化器都可以按需调用同一套 pass。概念上可以把 pass 分成两类：**analysis pass**（只读，计算某种程序性质，例如 DSA 计算的是指针指向关系和类型安全性，不修改 IR）和 **transform pass**（读取某些 analysis 的结果，修改 IR，例如内联、死代码消除）。现代 LLVM（论文发表后的工程演进）把这一思想固化为显式的 **PassManager** 基础设施：transform pass 在声明时列出自己依赖哪些 analysis，PassManager 负责按依赖顺序调度、缓存 analysis 结果；一旦某个 transform pass 改变了 IR 的某个方面（例如改变了 CFG），它必须显式声明"哪些之前算好的 analysis 结果失效了（invalidate）"，PassManager 据此决定要不要重新跑对应 analysis，而不是无脑地全部重算。

**为什么这样设计**：链接时/运行时场景对分析速度极度敏感（详见第 5 节的秒级/毫秒级计时），如果每个 pass 都各自独立重新计算一遍所需的所有分析，会有大量重复工作；反过来如果分析结果被跨 pass 长期缓存却不管失效，又会导致用过期信息做出错误变换的 bug。用"声明式依赖 + 显式失效"的方式，把"分析结果的生命周期管理"从每个 pass 的作者手里收归到统一的 PassManager，既保证正确性，又天然支持增量复用（论文提到的"编译期算好摘要、链接期直接复用摘要"就是这一思想在过程间分析上的具体体现）。

**带来什么能力**：pass 之间高度解耦、可插拔——你可以只启用 DGE+DAE 这种便宜的 pass 做增量构建时的快速反馈，也可以在链接时或空闲时开启 DSA 这种代价更高但更精确的分析；新增一个 pass 不需要改动其余 pass，只需要声明清楚它依赖/破坏哪些 analysis。这套基础设施后来演进成 LLVM 里的 Legacy PassManager 和更现代的 **New PassManager**（本笔记第 4 节会给出 New PassManager 风格的最小 pass 示例）。

### 3.4 Link-time / 跨模块（whole-program）优化

**是什么**：链接时把各编译单元产出的 LLVM IR 合并（论文里由链接器完成，现代工具链对应 `llvm-link` + LTO 流程），在整份程序都可见的前提下跑过程间优化：跨模块内联、死全局/死函数消除、指针分析（DSA）等。

**为什么这样设计**：分离编译（separate compilation）是构建大型软件的基本工程实践，但它天然把"全程序视角的优化机会"切碎在了各个编译单元里——一个函数只在别的 `.c` 文件里被调用一次，单文件编译器根本看不到这个信息。链接时优化器不需要用户改动 Makefile 意义上的构建流程（只需要产出的目标文件里带着 LLVM IR 而不是纯机器码），就能拿到与"把所有源码糊成一个文件再编译"等价的全局视野，同时保留分离编译在开发效率上的好处。

**带来什么能力**：论文的实验（第 5 节）显示，这类过程间优化在真实 SPEC 程序上比用 GCC 编译同一程序还快，即使 GCC 根本不做跨模块优化——这说明"链接时优化"在工程上是可以接受的开销，不是纸上谈兵。更重要的是，链接时优化器和运行时/离线优化器共享同一套 pass 库，"在链接时验证过的优化技术"可以直接原样搬到运行时用，不需要重新实现一套"运行时安全"版本。

### 3.5 运行时/Profile 驱动优化、JIT

**是什么**：LLVM 支持两条运行时路径。一是 JIT 执行引擎，按需逐函数把 LLVM IR 编译成机器码执行（没有可用的原生 codegen 时退化为解释器）。二是论文重点描述的**运行时路径 profiling + trace 级再优化**策略：离线阶段先插入轻量探测代码定位热点循环区域，运行时在线插桩库进一步定位区域内的热路径；一旦识别出热路径，就把对应的 LLVM IR 复制成一条 trace，对这条 trace（而不是整个函数/程序）重新跑 LLVM 优化并生成机器码，写入 software trace cache，缝合进原程序继续执行。同时，运行时收集的 profile 信息可以落盘，供 idle-time 的离线再优化器使用。

**为什么这样设计**：论文认为一个"最优"的运行时优化器需要同时具备三个特性：(a) 初始代码质量足够高——这要求初始 native code 生成本身可以离线、用昂贵算法完成，而不是依赖运行时即时生成的低质量代码；(b) 代码生成器和运行时优化器能够协作——两者共享同一套 LLVM 基础设施，运行时优化器可以复用代码生成器提供的插桩、简化变换等能力；(c) 运行时优化器能拿到比裸机器码丰富得多的信息（类型、SSA、显式 CFG）来做决策。传统二进制级运行时优化系统（如 Dynamo）只能在裸机器码上工作，天然缺失 (c)；传统 JIT 型虚拟机往往缺失 (a)，因为它们的"离线阶段"就是解释执行或简单编译。LLVM 把"离线生成高质量初始代码"和"运行时用同一套 IR 做局部重优化"结合起来，同时满足三者。

**带来什么能力**：既能获得静态编译器级别的基线性能，又能针对**真实运行时行为**（而不是开发者提供的训练输入）做针对性优化，且这种优化是"透明"的——不需要用户对程序做任何改动。

### 3.6 前后端解耦与多目标 CodeGen

**是什么**：前端只负责把源语言翻译成 LLVM IR，不关心目标机器；CodeGen（无论是离线的 `llc` 类静态后端还是运行时的 JIT）只负责把 LLVM IR 降级为特定目标（论文中实现了 SPARC V9 和 x86）的机器码，不关心源语言是什么。二者之间唯一的契约就是 LLVM IR 本身。

**为什么这样设计**：这是经典的"编译器中间表示分层"思想在 LLVM 场景下的落地——如果前端直接翻译到某个具体目标的机器码，N 种语言 × M 种目标就需要 N×M 套翻译逻辑；引入一层公共 IR 后只需要 N 个前端 + M 个后端。LLVM 在这基础上额外强调的是：这层公共 IR 不仅要"通用"，还要**同时**服务于编译时静态优化、链接时跨模块优化、运行时优化——因此它不能像传统教材里的三地址码那样只追求"易于生成目标码"，还要保留类型/数据流信息以支撑高层分析。

**带来什么能力**：一次编写的优化 pass（内联、DCE、指针分析等）对所有前端/所有后端自动生效；新增一个目标平台只需要写新的 CodeGen 后端，不需要改前端或已有的优化 pass；新增一种源语言只需要写新前端。这正是后来 Clang/LLVM 生态"多前端（C/C++/Rust/Swift/...）× 多后端（x86/ARM/RISC-V/GPU/...）"能够以线性而不是平方级代价扩展的根本原因。

## 4. 使用示例

### 4.1 C++ 异常处理示例：论文原文 IR 与现代 LLVM IR 对照

论文 Figure 1/2 给出的例子：一个栈上分配的、带析构函数的 `Object`，在调用可能抛异常的 `func()` 时，如果异常发生要保证析构函数被执行后再继续向上 unwind。

对应 C++ 源码（论文 Figure 1）：

```cpp
{
  Class Object; // Has a destructor
  func();       // Might throw
  ...
}
```

**论文原文写法**（2004 年的 LLVM 语法，已过时——用 `except label` 而非 `unwind label`，用裸 `unwind` 指令而非 `landingpad`/`resume`）：

```llvm
; Allocate stack space for object:
%Object = alloca %Class, uint 1
; Construct object:
call void %Class::Class(%Class* %Object)
; Call ``func()'':
invoke void %func() to label %OkLabel
            except label %ExceptionLabel
OkLabel:
; ... execution continues...
ExceptionLabel:
; If unwind occurs, excecution continues
; here. First, destroy the object:
call void %Class::~Class(%Class* %Object)
; Next, continue unwinding:
unwind
```

**现代 LLVM IR 等价写法**（LLVM 16+，使用 opaque pointer `ptr`、`landingpad`/`resume` 异常模型，函数名带 `@` 前缀，示意签名做了简化）：

```llvm
define void @scope_example() personality ptr @__gxx_personality_v0 {
entry:
  %Object = alloca %Class, align 8
  call void @Class_ctor(ptr %Object)
  invoke void @func()
      to label %OkLabel unwind label %ExceptionLabel

OkLabel:
  ; ... 正常执行继续 ...
  ret void

ExceptionLabel:
  %lp = landingpad { ptr, i32 }
          cleanup
  call void @Class_dtor(ptr %Object)
  resume { ptr, i32 } %lp
}
```

对照关系：论文的 `except label` 就是现代语法里 `invoke ... unwind label` 的旧写法；论文里"裸" `unwind` 指令在 2010 年前后被移除，改用 `landingpad` 显式描述异常着陆块要处理哪些异常类型（这里的 `cleanup` 表示"只做清理、继续向上抛"），并用 `resume` 显式表达"继续 unwind"。指针类型从 `%Class*` 变成不带具体指向类型的 `ptr`（opaque pointer，LLVM 16 起的默认设计，进一步简化了类型系统但把"指向什么类型"这一信息从指针类型上移除，交给访问指令自己携带）。

### 4.2 一个更基础的例子：结构体字段访问与 `getelementptr`

C 源码：

```c
struct Point { int x; int y; };

int sum_point(struct Point *p) {
    return p->x + p->y;
}
```

对应的现代 LLVM IR（`clang -O0 -S -emit-llvm`，opaque pointer 语法）：

```llvm
%struct.Point = type { i32, i32 }

define i32 @sum_point(ptr %p) {
entry:
  %x_ptr = getelementptr inbounds %struct.Point, ptr %p, i32 0, i32 0
  %x = load i32, ptr %x_ptr, align 4
  %y_ptr = getelementptr inbounds %struct.Point, ptr %p, i32 0, i32 1
  %y = load i32, ptr %y_ptr, align 4
  %sum = add nsw i32 %x, %y
  ret i32 %sum
}
```

可以直接对照第 3.1 节的机制：`%struct.Point` 是显式声明的聚合类型；`getelementptr` 用"类型 + 索引序列"计算出字段地址而不做任何隐式指针算术（第一个 `i32 0` 是"从 `%p` 本身跳 0 个 `Point`"，第二个索引才是字段号）；`load`/`store` 只接受单个已经算好的指针；`%x`、`%y`、`%sum` 都只被定义一次，天然满足 SSA。

### 4.3 SSA 与显式控制流：`phi` 指令示例

再看一个体现"显式 CFG + SSA"机制本身的例子，对应第 3.1 节的机制说明：

```c
int max2(int a, int b) {
    if (a > b) return a;
    else return b;
}
```

对应的现代 LLVM IR：

```llvm
define i32 @max2(i32 %a, i32 %b) {
entry:
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %then, label %else

then:
  br label %merge

else:
  br label %merge

merge:
  %result = phi i32 [ %a, %then ], [ %b, %else ]
  ret i32 %result
}
```

这里每个基本块（`entry`/`then`/`else`/`merge`）都以恰好一条终结指令（`br`/`ret`）结束，且终结指令显式列出了后继标签——这就是"CFG 是显式的"的字面含义，不需要单独的数据结构去猜控制流去向。`merge` 块里的 `%result = phi i32 [ %a, %then ], [ %b, %else ]` 精确对应论文里"显式 `phi` 指令对应标准 φ 函数"的说法：它表示"如果是从 `%then` 跳过来的，取 `%a` 的值；如果是从 `%else` 跳过来的，取 `%b` 的值"，从而让 `%result` 仍然只被定义一次，满足 SSA 的约束，而不需要给 `a`/`b`/`result` 引入内存别名。

### 4.4 三种等价表示之间的转换（对应论文 2.5 节）

论文 2.5 节强调 LLVM 定义了完全等价、可无损互转的文本、位码（binary）、内存三种表示，这一点直接体现在工具链上：

```bash
# 文本形式 (.ll) <-> 位码形式 (.bc)，二者语义完全等价、可无损互转
llvm-as sum_point.ll -o sum_point.bc     # 文本 -> 位码
llvm-dis sum_point.bc -o roundtrip.ll    # 位码 -> 文本（应与原始 .ll 语义一致）

# opt / llc / lli 等工具都可以直接读写 .bc，不需要先转换成文本再解析，
# 这正是链接时/运行时能够高效地反复传递、加载同一份 IR 的基础
opt -O2 sum_point.bc -o sum_point.opt.bc
```

这解释了为什么第 2.3 节说"IR 的可逆性/等价性"是终身编译模型能落地的前提：编译时产出的 `.bc`、链接时合并优化后的 `.bc`、离线 CodeGen 时嵌入可执行文件里的 bytecode、运行时优化器读取的 IR，全部是同一套表示，工具之间不需要做任何有损的格式转换。

### 4.5 常用命令行流水线

```bash
# 1. C 源码 -> LLVM IR 文本形式（前端只做到 IR 这一步，不做机器相关决策）
clang -S -emit-llvm sum_point.c -o sum_point.ll

# 2. 对 IR 独立运行一组优化 pass（opt 是脱离前端的通用 LLVM 优化器/pass 驱动工具）
opt -O2 sum_point.ll -S -o sum_point.opt.ll

# 3. 把优化后的 IR 降级为目标平台汇编（llc 是静态 CodeGen 后端，对应论文里的"离线 CodeGen"）
llc sum_point.opt.ll -o sum_point.s

# 4. 汇编 -> 目标文件 -> 链接为可执行文件（走系统汇编器/链接器）
clang sum_point.s -o sum_point

# 5. 也可以完全不落地机器码，直接用 LLVM 解释器/JIT 跑 IR（对应论文里的 JIT 路径）
lli sum_point.opt.ll
```

跨模块链接时优化（LTO，对应第 2/3 节的 Linker + IPO/IPA）：

```bash
# 每个编译单元都产出带 LLVM bytecode 的目标文件，而不是纯机器码
clang -flto -c a.c -o a.o
clang -flto -c b.c -o b.o

# 链接阶段触发 LTO：链接器先把 a.o/b.o 里的 LLVM IR 合并、跑跨模块 IPO，
# 再统一做一次离线 CodeGen，从而拿到只有在“全程序视角”下才能发现的优化机会
clang -flto a.o b.o -o app
```

### 4.6 一个最小的自定义 Pass 骨架（New PassManager 风格）

```cpp
// CountInstructionsPass.cpp —— 只读分析型 pass：统计每个函数的指令数
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
struct CountInstructionsPass : PassInfoMixin<CountInstructionsPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    unsigned Count = 0;
    for (auto &BB : F)
      Count += BB.size();
    errs() << F.getName() << ": " << Count << " instructions\n";
    // 只读分析，没有修改 IR，因此保留所有已有 analysis 结果
    return PreservedAnalyses::all();
  }
};
} // namespace

// 通过 PassPlugin 机制在运行时把这个 pass 注册进 opt 的 pipeline 解析器
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "CountInstructions", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "count-instructions") {
                    FPM.addPass(CountInstructionsPass());
                    return true;
                  }
                  return false;
                });
          }};
}
```

编译成插件并在 `opt` 中加载运行（不需要重新编译整个 LLVM）：

```bash
clang++ -shared -fPIC $(llvm-config --cxxflags) \
    CountInstructionsPass.cpp -o libCountInstructions.so

opt -load-pass-plugin=./libCountInstructions.so \
    -passes=count-instructions sum_point.ll -disable-output
```

这个骨架直接对应第 3.3 节的 pass 分类：这里写的是一个 analysis 性质的只读 pass（返回 `PreservedAnalyses::all()`）；如果要写 transform pass，只需在 `run` 里修改 `F`，并根据实际改动返回 `PreservedAnalyses::none()` 或更精确的失效集合，PassManager 会据此决定后续 pass 要不要重新计算相关 analysis。

## 5. 关键实验/评估结论

- **类型信息的可用性**：论文用 Data Structure Analysis（一种流不敏感、字段敏感、上下文敏感的指针分析）验证 LLVM 声明类型与实际内存访问是否一致，在 SPEC CPU2000 的 12 个 C benchmark 上，平均 74.3%（摘要里概述为约 74.6%）的静态 load/store 被证明是"类型安全"的访问，部分程序（gzip、vpr、art、mcf、equake、crafty、bzip2）甚至接近或达到 90%~100%。这说明：**即使是像 C 这样完全不保证类型安全的语言，只要前端在翻译到 LLVM 时认真保留类型信息，编译器后端也能在事后可靠地恢复出绝大部分类型事实**，而不需要语言本身强制类型安全——这是"低层但保留类型信息"这条设计取舍在实践中被验证有效的核心证据。剩余的"丢类型"案例主要来自自定义内存分配器（把内存当 `void*`/字节数组管理）和分析本身不够激进，而不是 IR 表达能力的问题；论文特别提到早期基于 GCC RTL 的前端因为丢失了太多类型信息，导致同样的分析效果差得多，反过来证明"前端翻译质量"和"IR 是否携带类型"同样重要。
- **IR 体积**：把 SPEC CPU2000 编译到 LLVM 位码后的文件大小，与用 GCC 3.3 `-O3` 生成的 SPARC（RISC）原生可执行文件基本相当，比 x86（更紧凑的变长指令集）大约大 25%。结论是：**携带丰富类型信息 + 显式 CFG + SSA 形式（本质上是携带了额外的数据流信息）并没有带来数量级的体积膨胀**，这对"把 LLVM bytecode 长期跟随可执行文件保留下来，供链接时/运行时/空闲时反复使用"这一核心设计目标至关重要——如果体积膨胀几倍，这个"终身保留 IR"的策略在工程上就不现实。
- **过程间优化的运行速度**：在同一批 SPEC 程序上，测量了 DGE（死全局消除）、DAE（死参数消除）、函数内联、DSA（指针分析）这几个跨模块优化的运行耗时，全部远小于用 GCC `-O3` 编译同一程序所需的时间（例如 255.vortex 上 DSA 耗时约 3 秒，而 GCC 编译该程序耗时约 24 秒；DGE/DAE 通常在几十毫秒以内）——即便 GCC 根本没有做任何跨模块优化。结论是：**精心设计的低层 IR（小指令集、显式 CFG、SSA、良好的数据结构实现）使得即便是"全程序范围"的过程间分析，其开销也完全可以被链接时甚至安装时的场景接受**，这正是"link-time whole-program optimization 在工程上可行"的关键论据，也是后来 LTO 成为主流工具链标配特性的技术起点。
- **案例研究的定性结论**：论文用 SAFECode（基于 LLVM 类型信息做静态内存安全检查，替代垃圾回收）、DSA/Automatic Pool Allocation（基于 LLVM 类型/指针分析做数据结构级内存布局优化）、以及"作为硬件 V-ISA"的探索作为三个案例，共同印证同一个论点：**这些技术之所以能在 LLVM 上以"语言无关、可在链接时全程序运行"的方式落地，根本原因都可以归结到 LLVM IR 同时具备类型信息和 whole-program 可分析性**，而不是任何单一某个特性单独起作用。

## 6. 与 MLIR / AI 编译器的衔接

- **被 MLIR 直接继承的概念**：`Module`/`Function`（MLIR 里是嵌套的 `Operation`，但顶层容器结构同源）、`BasicBlock`、`Value`、显式 CFG、SSA 形式、`phi`/block argument 式的数据流表达、以及"Pass 读取/失效 analysis"的 PassManager 思想，都被 MLIR 原样继承并进一步泛化。可以说 LLVM 定义的这套"IR = 类型化的 SSA 值 + 显式 CFG 的基本块 + 可插拔 Pass"范式，是包括 MLIR 在内几乎所有现代编译器基础设施的公共祖先。**LLVM 之前的世界**（例如传统汇编、GCC 的 RTL/Tree-SSA、各种 ad-hoc 编译器内部表示）普遍缺乏"跨阶段复用同一份 IR""IR 三态等价（文本/二进制/内存）""统一 pass 基础设施"这几项组合，这也是本论文反复强调的差异化贡献。
- **LLVM IR 在 AI 编译栈中的位置**：LLVM IR 几乎从不作为 AI 编译器的顶层表示（顶层通常是计算图/张量级 IR，如 TVM 的 Relay/TIR、IREE 的 `flow`/`stream`/`hal` dialect、Triton 的 Triton IR），而是作为**CPU（以及部分 GPU 后端如 NVPTX 依赖的中间层）代码生成的最后一段公共下游**：TVM 的 LLVM codegen backend 把 lower 后的 TIR 翻译成 LLVM IR，再交给 LLVM 的 `llc` 走完指令选择/寄存器分配生成机器码；IREE 的 `LLVMCPU` 编译目标把 MLIR 里较低层的 `llvm` dialect 直接对应到真正的 LLVM IR，复用 LLVM 现成的 CPU 后端而不必自己重新实现寄存器分配等极其成熟的底层能力；Triton 把 Triton IR 逐步下降到 `TTGIR`（GPU 相关的 tiling/layout 信息）之后，最终生成 LLVM IR，再交给 NVPTX/AMDGPU 后端产出 PTX/GCN。换句话说，**在 AI 编译栈里，LLVM 扮演的角色和这篇论文里描述的完全一致——一个语言无关、目标相对独立、专注把"已经足够低层"的程序表示高效映射到具体机器码的公共基础设施**，只是"喂给它的语言"从 C/C++ 换成了各种张量编译器的 lower 结果。
- **LLVM 的局限，为什么单层 IR 不够，催生了 MLIR**：LLVM IR 是**单一层级、面向标量/内存指令**的表示，天然缺少对高维张量、显式并行/向量化语义、领域相关操作（如卷积、矩阵乘）的一等（first-class）表达能力——想在 LLVM IR 上表达"这是一次 4D 张量卷积"，只能先展开成大量标量循环和 `getelementptr`，一旦展开就永久丢失了"这原本是一次卷积"的结构信息，后续所有基于这个结构信息的高层调度/融合优化都无法再进行（这与本论文第 3.2 节讨论的"高层特性展开成低层原语"是同一机制，但当展开粒度过细、发生得过早时，就会丢掉本该保留到更晚阶段的优化机会）。同时，不同抽象层次（图级张量运算、循环级仿射变换、硬件相关的向量/张量指令、最终的标量机器码）如果都硬塞进同一层 LLVM IR，会让每一层的 pass 都要处理不属于自己抽象层级的细节。MLIR 的核心动机正是**把"一层 IR、一个类型系统、一套指令集"这一 LLVM 假设，替换成"多层 dialect、每层保留恰好适合该抽象层的信息、可以逐层 lowering"的框架**——但其"SSA + 基本块 + Pass 基础设施"的骨架仍然直接继承自 LLVM。工程上这一衔接是显式建模的：MLIR 里存在一个几乎与 LLVM IR 一一对应的 `llvm` dialect，图级/张量级的高层 dialect（如 TVM 之外的 `linalg`、`affine`、`memref`、IREE 的 `hal`/`stream` 等）经过一系列 lowering pass 最终都会降到 `llvm` dialect，再用 `mlir-translate --mlir-to-llvmir` 之类的工具"跨语言"翻译成真正的 LLVM IR，交给 LLVM 后端走完最后一段——这一步之后，前面几节讨论的 LLVM Pass 基础设施、CodeGen、目标机器码生成就会原样接管，AI 编译器不需要（也不应该）重新发明这一段。

从工程投入产出比的角度看，这也解释了为什么 TVM/IREE/Triton 这些图级 AI 编译器几乎都选择"复用 LLVM 做最后一段 CodeGen"而不是自己重写指令选择和寄存器分配：指令选择、寄存器分配、指令调度是编译器领域已经被研究了几十年、高度依赖具体目标 ISA 细节的"体力活"，LLVM 已经为几十种主流 CPU/GPU 架构维护了成熟的后端实现；AI 编译器团队真正有差异化价值的部分是图级/张量级的融合、调度、内存规划这些"更高层"的优化，把最后一段机器码生成的工作交还给 LLVM，符合本论文"前后端解耦"这条设计原则的初衷——只是这里的"前端"变成了整套图级 AI 编译器。

## 7. 学习这篇论文时的最小必要集

对于要走"分布式训练 → LLVM/MLIR → 图级 DL 编译器 → 运行时多后端部署"这条路线的工程师，本论文里**必须掌握**的点：

1. **SSA + 无限虚拟寄存器 + 显式 CFG** 这套组合本身——这是 LLVM IR、MLIR、几乎所有现代编译器 IR 的共同底座，理解它为什么能"用便宜算法拿到接近流敏感分析的精度"是后续读 MLIR/TVM 源码的前提。
2. **类型系统 + `getelementptr` 的设计取舍**：为什么 IR 要"低层但保留类型/结构信息"，而不是走纯汇编或纯 AST 两个极端——这是理解一切"分层 IR"设计（包括 MLIR 的多 dialect 体系）的思想源头。
3. **load/store 内存模型的"无隐式访问"哲学**：所有内存操作都通过带类型指针的显式指令完成，这个约束大幅简化了别名分析和内存相关变换，是后续理解 TVM/IREE 里 buffer/memref 抽象的基础参照。
4. **Pass / PassManager 机制**：analysis pass 与 transform pass 的区分、依赖声明与失效（invalidate）机制——这是所有编译器基础设施（LLVM、MLIR、TVM 的 Pass Infra）组织优化逻辑的通用范式，必须能读懂、能写一个最小 pass。
5. **Link-time / whole-program 优化（LTO）的价值和代价**：为什么"全程序可见"是很多分析/优化的前提条件，以及"预先计算摘要、增量复用"这种加速手段——这对应到分布式/多后端场景里"全局视图 vs 局部视图"优化的通用矛盾，是很有迁移价值的直觉。
6. **前后端解耦、CodeGen 是独立可插拔阶段**这一架构决策——这是直接理解 IREE/TVM/ONNX Runtime 的"多后端委托（delegate）"模型的历史源头：本质上都是"公共 IR + 可插拔后端"的同一套思路在不同抽象层级的重复应用。
7. **"同一份 IR 贯穿编译时/链接时/运行时/空闲时"的终身编译思想**，尤其是它依赖的前提——IR 三态（文本/二进制/内存）等价、无损转换。理解这一点能帮助理解为什么现代 AI 编译框架也执着于"能不能把某种中间表示序列化后原样复用于 AOT 编译和运行时"（例如 ExecuTorch 的 `.pte` 格式、IREE 的 VM bytecode）。
8. **理解 LLVM 单层 IR 的局限**（结构信息一旦 lower 就不可逆、缺少张量/高层算子的一等表达），并知道这正是 MLIR 要解决的问题——这一点决定了你能不能把这篇 2004 年的论文和 2020 年代的 AI 编译器工作串起来看。

**可以先跳过、以后遇到再看的内容**：

- `invoke`/`unwind`（现代已演进为 `landingpad`/`resume`）异常处理指令的具体底层实现细节（零成本表驱动 vs `setjmp`/`longjmp`），除非要直接参与 LLVM 异常处理机制本身的开发。
- SAFECode、Data Structure Analysis、Automatic Pool Allocation 的具体算法内部细节——只需要知道"它们依赖 LLVM 的类型信息 + whole-program 视角"这个结论即可，具体的上下文敏感指针分析算法可以留到需要写静态分析工具时再深入。
- 论文里 SPARC/x86 实验的具体数字——只需要记住量级结论（"IR 体积与 native code 同量级""过程间优化远快于整体编译时间"），不需要背具体百分比。
- 与 UNCOL、ANDF、TAL/LTAL、SafeTSA、Slim Binaries 等历史系统的详细比较（第 5 节 Related Work）——这些是给论文审稿人看的历史脉络，对工程实践没有直接指导意义，了解"LLVM 不是第一个想做统一 IR 的项目，但吸取了此前失败的教训"这一结论即可。
- 论文里描述的、基于软件 trace cache 的运行时路径 profiling 具体实现机制——这套 2004 年的具体运行时优化方案在今天的 LLVM 生态里已基本不是主流实践（现代 PGO/BOLT 等工具走的是不同的具体路径），了解其设计动机（"运行时也能复用同一套 IR 做优化"）比记住实现细节更重要。

## 附：关键术语中英对照

- 终身程序分析与变换 —— lifelong program analysis and transformation
- 静态单赋值形式 —— Static Single Assignment (SSA) form
- 无限虚拟寄存器 —— infinite virtual register set
- 控制流图 —— Control Flow Graph (CFG)
- 类型化地址计算指令 —— `getelementptr`
- 过程间优化/分析 —— Interprocedural Optimization / Analysis (IPO / IPA)
- 链接时优化 —— Link-Time Optimization (LTO)
- 全程序优化 —— whole-program optimization
- 数据结构分析（一种指针分析） —— Data Structure Analysis (DSA)
- 自动池分配 —— Automatic Pool Allocation
- 剖析引导优化 —— Profile-Guided Optimization (PGO)
- 即时编译 —— Just-In-Time compilation (JIT)
- 软件跟踪缓存 —— software trace cache
- 分析型 pass / 变换型 pass —— analysis pass / transform pass
- 遍管理器 —— Pass Manager
- 不透明指针 —— opaque pointer
- 方言（MLIR 中分层抽象的单元） —— dialect
