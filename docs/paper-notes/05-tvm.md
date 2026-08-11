# TVM：面向异构硬件后端的自动化端到端深度学习编译器

> **导航**：[笔记索引](README.md) · [自学枢纽](../README.md)（阶段 4） · [横切概念](../ai-compiler-foundations.md) §3（融合）、§4（调度）

> **论文元信息**
> - 标题：*TVM: An Automated End-to-End Optimizing Compiler for Deep Learning*
> - 作者/机构：Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Meghan Cowan, Haichen Shen, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy —— University of Washington（联合 AWS、上海交通大学、UC Davis、Cornell）
> - 会议：OSDI 2018
> - arXiv：<https://arxiv.org/abs/1802.04799>
> - 开源项目：<https://github.com/apache/tvm>（现为 Apache TVM）
>
> **与工程学习文档的分工**：本文是 OSDI 2018 论文笔记（动机 / 历史框架）。今天的 Apache TVM 作为工程系统（流水线、融合细则、schedule 原语前后对比、MetaSchedule、动手清单）见 [`../tvm-learning-guide.md`](../tvm-learning-guide.md)。

## 1. 它解决什么问题

深度学习系统在 2017-2018 年前后面临一个典型的 **N×M 组合爆炸（combinatorial explosion）** 问题：上层有 N 种框架（TensorFlow、PyTorch、MXNet、Caffe、CNTK、Keras……），下层有 M 种硬件后端（server GPU、嵌入式 CPU/GPU、FPGA、TPU 类 ASIC……）。传统做法是每个框架针对每种硬件调用**厂商提供的算子库**（cuDNN、cuBLAS、ARM Compute Library、TFLite kernels……），这条路径存在三个结构性问题：

1. **算子库不可扩展**：库是为一小部分“规范算子形状 + 主流硬件”手工调优的，一旦出现新算子（depthwise conv、低比特量化算子）或新硬件（FPGA 加速器、自定义 ASIC），库跟不上，只能退化为未优化实现，或者直接不支持。
2. **图级优化与算子级优化脱节**：框架的计算图优化（如算子融合）只在“图重写”层面发生，但融合出来的新算子需要有对应的库实现——算子库不可能穷尽所有融合模式的组合，这迫使框架在“放弃融合”和“使用未优化的融合算子”之间做痛苦取舍。
3. **硬件多样性远超单一编程模型能覆盖的范围**：CPU 依赖隐式管理的多级缓存（implicitly managed memory hierarchy）和标量/向量计算单元；GPU 是混合式管理（部分显式的 shared memory + 隐式缓存）；TPU 类加速器则完全依赖显式管理的 on-chip buffer（explicitly managed）和张量计算单元（tensor compute primitive）。三者的内存子系统与计算原语差异巨大（论文 Figure 1），任何针对单一硬件模型设计的编译流程都难以直接迁移。

TVM 的核心主张是：**用一个端到端编译栈同时承担图级和算子级优化，并把“为每个硬件后端手工调优”的工作，替换为“自动生成 + 机器学习引导搜索”**，从而把 N×M 的人力问题变成一次性的编译器基础设施投入。

## 2. 整体运行框架

TVM 的编译 pipeline 可以概括为下图（括号内标出论文中的模块名，以及现代 TVM 中对应/演进的组件）：

```
前端模型 (TensorFlow / PyTorch / MXNet / Keras / ONNX / CoreML ...)
        │  frontend import
        ▼
计算图 IR  (论文: NNVM Graph IR；现代 TVM: Relay → 更新的 Relax/Relax-Dyn)
        │
        ▼  ── 图级优化 (Graph-level Optimization, 论文 Section 3) ──
        │     · 算子融合 (operator fusion)
        │     · 常量折叠 (constant folding)
        │     · 数据布局变换/传播 (data layout transformation)
        │     · 静态内存规划 (static memory planning)
        ▼
优化后的计算图 (Optimized Computational Graph)
        │  对图中每一个"融合后的算子" (fused op)
        ▼
张量表达式 Tensor Expression, TE  (lambda 描述 "算什么"，不描述 "怎么算")
        │
        ▼  ── 调度 Schedule (论文 Section 4) ──
        │     tile / split / reorder / fuse / vectorize / unroll / parallel /
        │     bind / cache_read / cache_write / compute_at / tensorize /
        │     virtual thread (cooperative fetching + latency hiding)
        ▼
Lowering → 低层循环程序 TIR (Halide 风格的 loop AST)
        │
        ▼  ── 自动化优化 (论文 Section 5) ──
        │     schedule template + search space
        │     → ML cost model (XGBoost, rank objective)
        │     → 模拟退火探索 (schedule explorer)
        │     → RPC 分布式设备集群实测反馈
        │     → tuning log 数据库
        ▼
CodeGen: LLVM IR (CPU) / CUDA·Metal·OpenCL (GPU) / Accelerator backend (VTA/FPGA)
        │
        ▼
可部署模块 (Deployable Module: lib + optimized graph + params)
        │
        ▼
Runtime: PackedFunc 统一 ABI + graph executor (按拓扑序调度算子)
         + RPC (支持异构/远程设备的编译-部署-测量闭环)
```

逐个说明各组件的输入/输出与职责：

- **前端（frontend）**：把各框架的模型格式解析成 TVM 自己的计算图表示。输入是框架特定的模型描述，输出是统一的图 IR。这一层只做格式转换，不做优化决策。
- **计算图 IR**：节点是算子（对张量的操作），边是数据依赖。与 LLVM IR 的关键区别是数据是大规模多维张量而非标量寄存器，这让图级 IR 可以利用 DL workload **shape 大多静态已知**这一先验做更激进的分析（比论文所述："take advantage of shape specificity"）。
- **图级优化**：输入是原始计算图，输出是**功能等价但执行代价更低、且已经确定了每个融合算子的粒度**的新计算图。这一步的决策直接决定了下游 operator-level 需要生成代码的“算子集合”是什么样子。
- **张量表达式 TE**：对图中每一个（可能是融合后的）算子，用声明式的 index formula 描述其计算规则，只规定"输出的每个元素怎么算"，完全不涉及循环顺序、并行策略、内存层次这些实现细节。
- **调度 Schedule**：在 TE 描述之上，通过一系列 schedule primitive 增量式地转换出具体的循环结构、并行策略与内存访问模式。同一份 TE 可以产生成千上万种功能等价但性能迥异的 schedule。
- **Lowering → TIR**：把 TE + schedule 的组合，转换成一棵具体的、Halide 风格的低层循环 AST（论文中称为 low-level loop program，现代 TVM 中正式命名为 TIR，Tensor IR）。
- **自动化优化（AutoTVM 及其后继者）**：在给定 schedule template 定义的搜索空间里，用 ML cost model 引导搜索，并通过 RPC 在真实硬件设备集群上实测校验，收敛出一组高性能的 schedule 参数。
- **CodeGen**：把最终确定的 TIR 编译成目标后端的可执行代码——CPU 走 LLVM IR，GPU 走 CUDA/Metal/OpenCL kernel 源码，自定义加速器走驱动 API 调用序列。
- **Runtime**：加载 CodeGen 产物和优化后的图结构，用统一的 **PackedFunc** ABI 调用底层生成函数，graph executor 按拓扑序依次派发算子；RPC 机制让这一整套编译-运行-测量流程可以跨主机、跨设备透明地工作（对嵌入式设备和加速器尤为关键）。

**图级优化与算子级优化的分工与协同**是理解 TVM 架构的关键：图级优化（fusion、layout）**先验地决定了**算子级要生成什么形状、什么维度顺序的代码——fusion 决定了“要不要把 conv+bn+relu 当成一个算子来生成循环”，layout 决定了“这个算子的 schedule 空间里，哪个维度应该被 tile、按什么粒度对齐”。反过来，算子级能否为某种融合/layout 组合生成高效代码，也约束了图级优化该不该做这个决策——这正是论文强调"图级优化只有算子库能跟上才有意义"的原因，也是 TVM 用**自动代码生成**取代**手工算子库**来打通这个闭环的根本动机。

## 3. 核心特性逐条拆解

### 3.1 算子融合（Operator Fusion）

- **是什么**：TVM 把计算图中的算子分为四类：(1) **injective**（一对一映射，如 `add`）；(2) **reduction**（如 `sum`）；(3) **complex-out-fusable**（计算复杂但输出可以融合element-wise 操作，典型如 `conv2d`）；(4) **opaque**（无法融合，如 `sort`）。基于分类给出通用融合规则：多个 injective 算子可以彼此融合成一个 injective 算子；reduction 算子可以融合其输入侧的 injective 算子（例如 `scale` + `sum`）；complex-out-fusable 算子（如 conv2d）可以把输出侧的 element-wise 算子融合进来（如 conv2d + bias + relu）。
- **为什么这样设计**：与其为“conv+bn+relu”“depthwise conv+bias”等每一种融合模式手写一个融合 kernel（组合数随算子和 layout/dtype 数量成倍增长，不可持续），不如用一套通用的、基于算子语义分类的规则，让编译器**自动判定哪些子图可以合并成一个 kernel**，再交给下游代码生成器统一处理。
- **带来什么能力**：避免了中间结果写回/读取内存的开销，在内存带宽敏感的 GPU 和加速器上收益显著；论文实测融合带来 1.2×–2× 的加速（Figure 4），且这一收益对手写库天然免费获得——因为省去了手工枚举融合 kernel 的工程成本。

#### 示例精讲：`conv2d → bias_add → relu` 融合前后，到底省了什么

> 目的：把「融合」从口号变成能数出来的访存次数。

**融合前**：三个算子 = 三个 kernel，每个 kernel 都要「从显存读 → 算 → 写回显存」。

```text
conv2d      读 X, W        →  写 T1（大小 = 输出张量）
bias_add    读 T1, b       →  写 T2（同样大小）
relu        读 T2          →  写 Y
```

对输出张量大小 `N` 计一笔账（只数中间结果的进出显存次数）：

| | 中间结果写回 | 中间结果再读入 | 合计额外访存 |
|--|-------------|---------------|--------------|
| 融合前 | T1, T2 各写一次 = 2N | T1, T2 各读一次 = 2N | **4N** |
| 融合后 | 无 | 无 | **0** |

**融合后**：一个 kernel，中间值停在寄存器/共享内存里。

```text
fused_conv2d_bias_relu   读 X, W, b  →  逐点算完 conv、加 bias、做 relu  →  写 Y
```

**为什么这三个能合**（对照 3.1 的四类分类）：

| 算子 | 类别 | 在这次融合中的角色 |
|------|------|--------------------|
| `conv2d` | complex-out-fusable | 融合的**起点**：允许把输出侧的逐元素算子吸进来 |
| `bias_add` | injective | 被吸收：只在同一个输出点上加一个数 |
| `relu` | injective | 被吸收：同上 |

反例：若把 `relu` 换成 `sort`（opaque），链条在那里断开——因为 `sort` 需要看到**整个张量**，无法在「算出一个点就立刻处理一个点」的循环里完成。

> 规则的实质是：**能不能在同一个输出点上把后续计算就地做完**。这与 MLIR/Linalg 用 `indexing_maps` 判融合是同一个问题的两种表述（见 [`../mlir-learning-guide.md`](../mlir-learning-guide.md) §8.2）。

> **自测**：`conv2d → conv2d` 为什么一般不能直接融合成一个 kernel？（提示：第二个 conv 的一个输出点需要第一个 conv 的多少个输出点？）

### 3.2 数据布局变换与传播（Data Layout Transformation）

- **是什么**：为图中每个算子，根据目标硬件的内存层次约束声明一个“偏好的”数据布局（不仅是行主序/列主序，也包括更复杂的 blocked layout，例如某加速器偏好把张量按 4×4 分块存储以优化访存局部性）。当一条边的 producer 偏好布局与 consumer 偏好布局不一致时，插入一个 layout transform 节点做转换。
- **为什么这样设计**：不同硬件对“什么布局访存最快”的答案完全不同（cache line、SIMD 宽度、tensor core 的 tile 形状都不一样），如果把布局硬编码在算子实现里，同一个算子在不同硬件上就需要不同版本；把布局作为图上的一等属性传播，使得布局决策与算子实现解耦。
- **带来什么能力**：布局传播的结果直接塑造了下游算子级 schedule 的搜索空间起点（例如决定了 tile 的对齐维度），是图级决策向算子级"传导"优化意图的主要通道之一；同时它也是本笔记第 6 节要讨论的"跨后端 layout 兼容性"问题在单硬件场景下的原型。

### 3.3 常量折叠与静态内存规划

- **是什么**：常量折叠在编译期预计算图中可以静态确定的部分（省去运行时重复计算）；静态内存规划则为每个中间张量预先分配、并尽可能复用内存 buffer（类似编译器的寄存器分配，但对象是张量而非标量）。
- **为什么这样设计**：DL workload 的 shape 在训练/推理配置固定后基本是静态已知的（这与通用程序不同），这为编译期做激进的、无需运行时开销的内存规划提供了条件。
- **带来什么能力**：减少运行时内存分配次数和显存/内存峰值占用，这对嵌入式设备和资源受限的加速器（SRAM 只有几十 KB）尤为重要。

### 3.4 张量表达式语言（Tensor Expression, TE）

- **是什么**：一种声明式的 index formula 语言，用 `compute((shape), lambda indices: expr)` 的形式描述“输出张量的每个元素怎么算”，例如矩阵乘：

```python
C = t.compute((m, n), lambda y, x: t.sum(A[k, y] * B[k, x], axis=k))
```

  该语言**不描述**循环顺序、tiling、并行策略、内存层次——这些全部交给独立的 **schedule** 来指定。这继承自 Halide 的 compute/schedule 解耦思想。

- **TVM 相对 Halide 到底改了什么、去掉了 Halide 的哪些假设**：这是理解 TVM 设计取舍的关键点。Halide 面向的是图像处理 pipeline，其调度原语隐含假定硬件是"隐式或半隐式管理内存层次"的（CPU 的多级缓存、GPU 的部分显式 shared memory），且并行模型是**shared-nothing 的 nested parallelism**（fork-join，同一并行阶段内的线程互不可见彼此数据）。TVM 保留了 Halide 的循环变换（loop transform）、线程绑定（thread binding）、计算局部性（compute locality）这三类原语（论文 Figure 6 的对照表），但为了覆盖 TPU 类加速器，**去掉/突破了以下假设**：
  1. **去掉了"内存层次由硬件隐式管理"的假设**：引入了通用的 **memory scope** 机制，让 schedule 可以显式声明某个 buffer 位于 GPU shared memory 或加速器的专属 scratchpad，从而支持跨线程协作数据复用和加速器的显式片上存储管理（Halide 当时只有针对 GPU shared memory 的特例支持，没有通用的 memory scope 抽象）。
  2. **去掉了"计算原语只能是标量/SIMD 向量"的假设**：引入 **tensorization**，把调度对象从单条指令扩展为一段可以映射到硬件张量指令（如 GEMM8×8 micro-kernel、TensorCore 指令）的子计算单元，且这个映射是**可扩展**的（新硬件只需声明新的 intrinsic，不用改编译器核心）。
  3. **去掉了"并行只能是 shared-nothing fork-join"的假设**：引入 **虚拟线程（virtual thread）+ 显式延迟隐藏（latency hiding）**机制，支持 decoupled access-execute（DAE）架构下需要软件插入细粒度同步指令才能重叠内存与计算的加速器场景（这是 CPU 靠 SMT/prefetch、GPU 靠 warp 快速切换隐式解决的问题，在瘦控制的加速器上必须由编译器显式处理）。
- **带来什么能力**：TE + Schedule 的解耦让"一份计算描述"可以在完全不改变计算语义的前提下，为 CPU、GPU、以及事先完全未知的定制加速器生成风格迥异的代码——这是 TVM 能以 ~2k LoC 的代价接入一款全新 FPGA 加速器（VDLA）的根本原因。

### 3.5 调度原语与调度空间

TVM 的 schedule 原语可以分成"继承自 Halide"和"TVM 新增"两组（对应论文 Figure 6 的表格，按 CPU / GPU / Accelerator 三类后端标出适用范围）：

| 原语 | 语义 | 主要作用后端 |
|---|---|---|
| `split` | 把一个轴拆成 outer × inner 两个轴 | 全部 |
| `tile` | 对多个轴同时做 `split`，构造块状循环结构 | 全部 |
| `reorder` | 调整循环轴的嵌套顺序，改善数据局部性 | 全部 |
| `fuse` | 把多个轴合并成一个轴（常用于映射到一维线程索引） | 全部 |
| `vectorize` | 把最内层轴标记为向量化，映射到 SIMD 指令 | CPU |
| `unroll` | 展开循环，减少分支/循环开销，暴露指令级并行 | CPU/GPU |
| `parallel` | 把轴标记为多核并行 | CPU |
| `bind` | 把轴绑定到 GPU 的 `blockIdx`/`threadIdx` | GPU |
| `cache_read` / `cache_write` | 为某个 buffer 插入一份缓存副本（可指定 memory scope），并生成对应的搬运语句 | GPU / Accelerator |
| `compute_at` | 把一个计算阶段内联到另一计算阶段的某层循环内部，控制计算发生的粒度（compute locality） | 全部 |
| **`tensorize`** | 把一段循环替换为对硬件张量指令的调用（见 3.4/3.6） | Accelerator（也可用于 CPU/GPU） |
| **虚拟线程 + cooperative fetching** | 显式管理内存层次、隐藏访存延迟（见 3.6） | Accelerator（GPU 部分适用） |

这些原语共同构成了一个**巨大的组合调度空间**——同一份 TE 描述经过不同的原语序列，可以产生几何级数量的合法 schedule，这正是第 3.7 节 AutoTVM 需要解决的搜索问题的来源。

### 3.6 内存作用域（Memory Scope）与显式共享内存管理

- **是什么**：给一个计算阶段（compute stage）打上 scope 标签（例如 `shared`），告知 lowering pass 这块 buffer 需要：(a) 由一组线程协作填充（cooperative fetching），(b) 在写完成后插入同步屏障（memory barrier）保证可见性，(c) 对加速器而言，映射到其专属的片上 buffer 并生成相应的搬运指令。若不显式指定，TVM 的默认 scope 推断会把中间 buffer 标记为 thread-local，无法表达跨线程复用。
- **为什么这样设计**：GPU 的 shared memory 和加速器的 scratchpad 都是"软件可控、硬件不自动管理"的存储层，如果没有一个显式的调度机制，就无法在 schedule 层面表达"这块数据应该被一组线程共享一次搬运、重复使用多次"这种关键优化。
- **带来什么能力**：论文以矩阵乘为例展示了效果——用 cooperative fetching 把 A、B 的 tile 协作搬入 shared memory 后复用，相较于不做这个优化的版本有明显加速（Figure 7：`cuBLAS` vs `TVM w/o coop.` vs `TVM`，在 Titan X 上随矩阵规模增大差距扩大）。对加速器而言，memory scope 更是**必需**而非可选——它是把数据放进有限 SRAM 里、避免频繁访问 DRAM 的唯一手段。

### 3.7 Tensorization（张量化）

- **是什么**：把调度的"最小硬件操作单元"从单条标量/SIMD 指令，扩展为一段可以映射到硬件张量指令（如加速器的 `gemm8x8`、CPU 的 bit-serial micro-kernel、TensorCore 类指令）的子计算。实现方式是**用同一套 TE 语言，既声明硬件 intrinsic 的行为（behavior），又声明它的 lowering 规则**（如何从抽象计算映射到具体的硬件调用），二者打包成一个 `tensor_intrin`；随后用 `tensorize` schedule 原语把 schedule 中匹配该计算模式的一段循环替换为对应 intrinsic 调用。
- **为什么这样设计**：新的加速器不断引入自己独有的张量指令变体，编译器不可能内置一份"全体硬件张量指令"的固定列表；把 intrinsic 的声明从调度逻辑中剥离出来、做成可插拔的声明式接口，使得**支持新硬件指令不需要修改编译器核心**，只需要新增一份 intrinsic 声明。
- **带来什么能力**：这是 tensorization 类比 SIMD 向量化（vectorization）但更进一步的地方——向量化面对的是定长、单一数据布局的标量指令，而 tensorization 要处理多维、可变长度、且有各自数据布局要求的张量指令。论文给出的实测例子是移动 CPU 上的超低精度（1~2 bit）矩阵向量乘微内核，通过 tensorize 接入后比非张量化版本快 1.5×；VDLA 加速器上的 GEMM 8×8 指令也是通过同一机制接入的。

### 3.8 显式内存延迟隐藏（Latency Hiding）与虚拟线程（Virtual Thread）

- **是什么**：CPU 靠同步多线程（SMT）/硬件预取隐式隐藏访存延迟，GPU 靠海量 warp 的快速上下文切换隐式隐藏，但 TPU 类加速器为了省控制逻辑，通常采用 **decoupled access-execute（DAE）架构**——把访存单元（load unit）和计算单元（execute unit）拆成独立流水线阶段，二者靠软件插入的细粒度依赖 token（push/pop dependence）显式同步，才能保证正确性同时让 load 和 execute 重叠执行。手写这种带显式 token 同步的指令流极其繁琐且容易出错。TVM 引入 **virtual thread** 调度原语：程序员按"多线程程序"的直觉写调度（就像给普通多核写并行代码一样），TVM 在 lowering 阶段自动把多个虚拟线程的指令交织（interleave）进一条单一指令流，并在其中插入正确的 RAW/WAR 依赖 push/pop 操作（论文 Figure 8）。
- **为什么这样设计**：如果直接把 DAE 同步细节暴露给用户手写调度，等于把硬件流水线编程的复杂性转嫁给了每一个算子开发者；虚拟线程把"表达并行意图"和"生成正确的低层同步指令"分离，让编译器承担同步正确性的证明与生成工作。
- **带来什么能力**：论文在自研 FPGA 加速器上实测，开启 latency hiding 后 ResNet 各层的计算利用率从 70% 提升到 88%（用 roofline 图衡量，Figure 10），效果直接体现为让实际吞吐更贴近硬件理论峰值（roofline 的 compute-bound 上界）。

### 3.9 AutoTVM：自动化调度优化

- **是什么**：一套完整的闭环自动调优系统，由四部分组成：
  1. **Schedule template + 搜索空间**：开发者可以显式声明调度中的"knob"（如 tile size、unroll factor），也可以用一套通用 master template 从 TE 描述里自动抽取可调 knob，覆盖数十亿量级的配置空间。
  2. **基于 XGBoost 的 ML cost model**：不构建精确的硬件性能模型（那需要为每种硬件重新建模，且现代硬件太复杂难以精确刻画），而是训练一个**统计模型**，输入是从 lowered 循环程序抽取的特征（每级循环下各 buffer 的访存次数/复用率，以及 `vectorize`/`unroll`/`parallel` 等标注的 one-hot 编码），输出是**相对排序**（而非绝对运行时间——因为 explorer 只需要知道"A 比 B 快"，不需要绝对数值，用 rank objective 训练即可）。论文对比发现树模型（XGBoost）比 TreeRNN 直接学习 AST 表征预测质量相近，但预测速度快一倍、训练开销小得多，因此被选为默认模型；预测一次仅需 0.67ms，比真实硬件测量快几千倍。
  3. **Schedule explorer（模拟退火）**：用 cost model 的预测引导一个并行模拟退火过程在配置空间中游走，优先探索预测代价更低的邻近配置，收敛后再挑选一批候选去真实硬件上实测。
  4. **迁移学习与分布式 RPC 反馈闭环**：cost model 用实测数据持续在线更新，这些数据可以在**相关 workload 之间迁移复用**（不用每个新算子都从零训练模型）；真实测量通过一个自建的 RPC-based 分布式设备池（device pool + tracker）完成，客户端可以在 host 上编译，透明地把二进制上传到 Raspberry Pi、Mali GPU 手机板、FPGA 板等远程设备上运行并取回结果，这套基础设施同时服务于单算子调优和端到端图推理部署。
- **为什么这样设计**：论文用一张对比表（Table 1）说明了三种自动化路线的取舍——纯 blackbox auto-tuning 无模型偏差但需要海量真实实验；predefined cost model 需要精确硬件知识且不可迁移；**ML-based cost model 用少量硬件先验换来低模型偏差 + 可从历史数据学习**，是二者的折中，且天然支持"越用越准"的在线迭代。
- **带来什么能力**：论文对比显示，同样的试验预算下，ML cost model 引导的搜索比 blackbox 遗传算法、随机搜索更快找到接近甚至超过 cuDNN 的配置（Figure 12）；这套系统是 TVM 能在完全不依赖厂商库的前提下，对**新算子**（depthwise conv、超低精度算子）也能自动找到有竞争力的 kernel 的核心原因。

### 3.10 运行时：PackedFunc、Graph Executor 与异构部署

- **是什么**：**PackedFunc** 是 TVM 定义的一套统一调用 ABI——不论一个函数是 LLVM 编译出的 CPU kernel、CUDA runtime 加载的 GPU kernel，还是加速器驱动 API 调用，只要包装成 PackedFunc，就可以用同一种方式跨语言（C++/Python/Java）注册、序列化、调用。**Graph executor** 加载编译产物中的三元组（优化后的图结构 `graph`、生成的算子库 `lib`、参数 `params`），按图的拓扑序依次把节点 dispatch 给对应的 PackedFunc 执行。**RPC** 机制让上述编译-部署-执行流程可以跨主机透明工作：可以在开发机上完成交叉编译，把模块动态上传到远程设备（嵌入式板、FPGA）执行，并在同一份脚本里拿到结果，这消除了嵌入式/加速器部署传统上需要的繁琐手工交叉编译和部署步骤。
- **为什么这样设计**：CPU/GPU/加速器的可执行产物格式天差地别（.so、PTX/cubin、加速器专属的指令流），如果 runtime 按后端类型写死分支逻辑，每接入一种新后端都要改运行时核心；统一 ABI + 模块化加载把"如何执行一个函数"这个问题封装成插件式的可扩展机制。
- **带来什么能力**：同一套 runtime 代码可以驱动 server GPU、嵌入式 CPU/GPU、FPGA 加速器；同一套 RPC 基础设施同时服务"单算子自动调优时的远程实测"和"端到端模型的远程部署推理"，无需为这两个场景各建一套设备管理系统。

### 3.11 对 FPGA/加速器（VTA）类硬件的可扩展性

论文用一个自研的通用 DL 加速器原型 **VDLA（Vanilla Deep Learning Accelerator，即后来开源社区中 VTA 的前身）**验证 TVM 的可扩展性：VDLA 是一个张量处理器，具备分层的片上存储（输入/权重/寄存器文件/微指令 SRAM）、GEMM 计算单元和向量 ALU、以及 load/compute/store 三条独立的指令队列，采用 decoupled access-execute 设计并提供显式的流水线同步控制。TVM 团队把它实现在一块低功耗 PYNQ FPGA 板（ARM Cortex A9 + Artix-7 FPGA）上，理论峰值约 102.4 GOPS/s，片上 buffer 总共不到 256KB（远不足以容纳 ResNet 单层的数据），因此天然构成了显存复用与延迟隐藏的极限测试场景。**接入这个全新加速器后端只花费约 2000 行 Python 代码**（对比接入 LLVM/CUDA 这类通用后端所需的工程量小得多），这直接得益于 3.4~3.8 节所述的 tensorize + memory scope + virtual thread 三件套——它们把"硬件特有的指令/内存/同步细节"都收敛成了声明式接口，而不需要改动编译器核心的图优化和调度框架。端到端 ResNet 推理中，可以 offload 到 FPGA 的卷积层获得 40× 加速，但受 Amdahl 定律限制，无法 offload 的 CPU-only 部分（首层浅层卷积、residual/activation 等 VDLA 不支持的算子）成为整体性能瓶颈——这也说明"可扩展性"和"覆盖率"是两个独立的问题：TVM 解决了前者，后者仍需要持续扩展加速器支持的算子集合。

## 4. 使用示例

> 以下示例基于 TVM 的 `te`（Tensor Expression）经典 API 风格（对应 Apache TVM 0.1x 系列版本），用于对照论文语义讲解调度原语。较新的 TVM（Unity 分支之后）正在把 Relay 逐步替换为 **Relax**，把手写 schedule 逐步替换/统一为 **MetaSchedule**（统一了 AutoTVM 的模板式搜索和 AutoScheduler/Ansor 的模板无关搜索），核心概念不变但顶层 API 名称有变化，示例中会标注对应关系。

### 4.1 用 TE 定义矩阵乘：naive schedule vs. tiled + cache_write + vectorize

```python
import tvm
from tvm import te

M, N, K = 1024, 1024, 1024
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
k = te.reduce_axis((0, K), name="k")
# compute 只描述"每个输出元素怎么算"，不涉及循环顺序
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

# ---- naive schedule：不做任何变换，等价于最朴素的三重循环 ----
s_naive = te.create_schedule(C.op)
print(tvm.lower(s_naive, [A, B, C], simple_mode=True))
```

`tvm.lower(..., simple_mode=True)` 打印出的伪 TIR 大致是：

```text
for (i, 0, 1024) {
  for (j, 0, 1024) {
    C[i, j] = 0f
    for (k, 0, 1024) {
      C[i, j] = C[i, j] + A[i, k] * B[k, j]
    }
  }
}
```

接下来对同一份 TE 施加 tiling + cache_write + vectorize：

```python
s = te.create_schedule(C.op)
# cache_write: 引入一个临时 buffer CL 承接累加结果，避免对 C 反复读改写
CL = s.cache_write(C, "local")

# tile: 同时对 i、j 两个轴做 split，构造出 (io, jo, ii, ji) 四层循环
io, jo, ii, ji = s[C].tile(C.op.axis[0], C.op.axis[1], x_factor=32, y_factor=32)

# 把累加 (CL 上的计算) 内联到 C 的 tile 内部，让每个 tile 独立累加后再写回
s[CL].compute_at(s[C], jo)

# 对 CL 内部的 reduce 轴做 split，拆出可以向量化的最内层
cki, ko = s[CL].op.reduce_axis[0], None
kio, kii = s[CL].split(CL.op.reduce_axis[0], factor=8)

# vectorize: 把最内层连续访存的轴标记为向量化，映射到 SIMD 指令
s[CL].vectorize(kii)

print(tvm.lower(s, [A, B, C], simple_mode=True))
```

优化后的伪 TIR 结构大致变为（省略部分细节以突出结构变化）：

```text
for (io, 0, 32) {
  for (jo, 0, 32) {
    // compute_at 把 CL 的计算下沉到这一层：每个 32x32 tile 独立处理
    local CL[32][32] = 0
    for (ko, 0, 128) {
      for (ii, 0, 32) {
        for (ji, 0, 32) {
          for (kii, 0, 8) {           // vectorize 作用在这一层
            CL[ii, ji] += A[io*32+ii, ko*8+kii] * B[ko*8+kii, jo*32+ji]  // vectorized
          }
        }
      }
    }
    for (ii, 0, 32) {
      for (ji, 0, 32) {
        C[io*32+ii, jo*32+ji] = CL[ii, ji]   // 一次性写回，减少对 C 的重复读改写
      }
    }
  }
}
```

每个原语对循环结构的作用可以总结为：`tile` 把两层循环拆成四层，制造出可以复用数据的"块"；`cache_write` + `compute_at` 让每个块的累加发生在一个局部临时 buffer 上，把对全局输出 `C` 的写操作从"每次累加都写"降到"每个块写一次"；`vectorize` 把最内层的连续访存标记为 SIMD 操作。这正是论文 Figure 5 中"从朴素三重循环 → tile → 映射到加速器 intrinsic"的同一条演进思路在 CPU 上的具体体现。

### 4.2 GPU schedule：bind blockIdx/threadIdx + shared memory cache_read + cooperative fetching

```python
import tvm
from tvm import te

M, N, K = 1024, 1024, 1024
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
k = te.reduce_axis((0, K), name="k")
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

s = te.create_schedule(C.op)
block_x = te.thread_axis("blockIdx.x")
block_y = te.thread_axis("blockIdx.y")
thread_x = te.thread_axis("threadIdx.x")
thread_y = te.thread_axis("threadIdx.y")

# bind: 把外层循环轴映射到 GPU 的 block/thread 网格
by, ty = s[C].split(C.op.axis[0], factor=64)
bx, tx = s[C].split(C.op.axis[1], factor=64)
s[C].bind(by, block_y)
s[C].bind(bx, block_x)
s[C].bind(ty, thread_y)
s[C].bind(tx, thread_x)

# cache_read: 为 A、B 各引入一份 shared memory 副本，供同一 block 内所有线程共享
AS = s.cache_read(A, "shared", [C])
BS = s.cache_read(B, "shared", [C])

ko, ki = s[C].split(k, factor=8)
# compute_at 到 ko 这一层：每次进入新的 k-tile 时，由 block 内线程协作搬运一次
# (cooperative fetching：多个线程分摊同一份共享数据的搬运工作，而非各自重复搬运)
s[AS].compute_at(s[C], ko)
s[BS].compute_at(s[C], ko)

print(tvm.lower(s, [A, B, C], simple_mode=True))
```

这里的关键设计对应第 3.6 节的 memory scope：`cache_read(..., "shared", ...)` 显式声明 `AS`/`BS` 位于 GPU 的 shared memory scope，lowering 阶段会自动为协作搬运插入 `memory_barrier`，确保所有线程写完共享数据后才能读取——这正是论文 Figure 7 展示性能差异的那个优化（`TVM` vs `TVM w/o coop.`）。

### 4.3 AutoTVM / auto_scheduler 调优最小脚本骨架

**AutoTVM（模板式搜索，论文中描述的原始机制）：**

```python
import tvm
from tvm import te, autotvm

@autotvm.template("tutorial/matmul")
def matmul_template(M, N, K, dtype):
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    s = te.create_schedule(C.op)

    # 声明可调 knob：tile 因子作为搜索空间维度（对应论文 5.1 的 schedule template）
    cfg = autotvm.get_config()
    cfg.define_split("tile_y", cfg.axis(M), num_outputs=2)
    cfg.define_split("tile_x", cfg.axis(N), num_outputs=2)
    yo, yi = cfg["tile_y"].apply(s, C, C.op.axis[0])
    xo, xi = cfg["tile_x"].apply(s, C, C.op.axis[1])
    s[C].reorder(yo, xo, yi, xi)
    return s, [A, B, C]

task = autotvm.task.create("tutorial/matmul", args=(M, N, K, "float32"), target="llvm")

# measure_option: 对应论文 5.4 的分布式实测（这里用 LocalRunner 简化为本机测量）
measure_option = autotvm.measure_option(
    builder=autotvm.LocalBuilder(),
    runner=autotvm.LocalRunner(number=10, repeat=3),
)

# XGBTuner: 对应论文 5.2 的 XGBoost cost model + 5.3 的模拟退火探索
tuner = autotvm.tuner.XGBTuner(task)
tuner.tune(
    n_trial=200,
    measure_option=measure_option,
    callbacks=[autotvm.callback.log_to_file("matmul.log")],  # tuning log 可持久化复用
)

with autotvm.apply_history_best("matmul.log"):
    with tvm.target.Target("llvm"):
        s, args = matmul_template(M, N, K, "float32")
        lib = tvm.build(s, args)
```

**auto_scheduler / Ansor（模板无关搜索，AutoTVM 的后继者之一）与 MetaSchedule（当前统一框架）**只需替换搜索空间生成方式，核心的"search space → cost model → 分布式实测 → 日志复用"闭环思想不变：

```python
from tvm import auto_scheduler

@auto_scheduler.register_workload
def matmul_add(M, N, K, dtype):
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    return [A, B, C]

task = auto_scheduler.SearchTask(
    func=matmul_add, args=(M, N, K, "float32"), target=tvm.target.Target("llvm")
)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=200,
    measure_callbacks=[auto_scheduler.RecordToFile("matmul.json")],  # 调优日志复用
)
task.tune(tune_option)
sch, args = task.apply_best("matmul.json")
```

在最新的 TVM 主干上，这套流程被统一进 `tvm.meta_schedule`（例如通过 `relax.transform.MetaScheduleTuneIRMod` / `MetaScheduleApplyDatabase` 直接作用在 Relax `IRModule` 上），概念上是 AutoTVM + AutoScheduler 的合并升级版，但"schedule 搜索空间 + ML cost model + 硬件实测 + 可持久化 tuning database"这套论文提出的方法论骨架保持不变。

### 4.4 端到端：导入 ONNX 模型 → build → graph executor 推理

```python
import onnx
import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_executor

# 1. 从 ONNX 导入模型，得到 Relay IRModule（对应论文中的"计算图 IR"层，
#    Relay 是论文发表后 TVM 社区推出的、比论文原始 NNVM Graph IR 更完备的图 IR）
onnx_model = onnx.load("resnet18.onnx")
input_name = "input"
shape_dict = {input_name: (1, 3, 224, 224)}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# 2. 图级优化 + 算子级 codegen 一并在 relay.build 中完成
#    opt_level=3 会启用算子融合、常量折叠、layout 变换等图级 pass（对应论文 Section 3）
target = tvm.target.Target("llvm -mcpu=core-avx2")
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# 3. Runtime 侧：graph executor 加载 lib，按拓扑序调度 PackedFunc（对应论文 Section 2 末尾示例）
dev = tvm.cpu(0)
module = graph_executor.GraphModule(lib["default"](dev))
module.set_input(input_name, np.random.rand(1, 3, 224, 224).astype("float32"))
module.run()
output = module.get_output(0).numpy()
print(output.shape)
```

在正在演进中的 Relax（TVM Unity）路线下，同样的流程用 `tvm.relax` 命名空间表达，核心结构一致，主要差异是 IR 表示从 Relay 换成支持符号 shape 的 Relax，编译产物从 `graph_executor` 换成更通用的 `relax.VirtualMachine`：

```python
from tvm import relax

mod = relax.frontend.onnx.from_onnx(onnx_model, shape_dict=shape_dict)
ex = relax.build(mod, target=target)
vm = relax.VirtualMachine(ex, dev)
output = vm["main"](tvm.nd.array(np.random.rand(1, 3, 224, 224).astype("float32")))
```

## 5. 关键实验结论

- **Server GPU（NVIDIA Titan X）端到端**：相较 MXNet(v1.1)、TensorFlow(v1.7)、TensorFlow XLA，TVM 取得 1.6×–3.8× 的端到端加速；其中 DQN 的 3.8× 主要来自其使用了 cuDNN 未充分优化的“非常规”卷积配置（4×4 kernel、stride=2），说明**手工库的优化覆盖面天然有限，越"非主流"的算子形状，自动生成的优势越明显**。
- **算子级 breakdown（ResNet-18 的全部 conv2d、MobileNet 的全部 depthwise conv2d）**：标准 conv2d 上 TVM 大多数层仍能超过 cuDNN（一个被高度手工优化的库），depthwise conv2d（当时是新算子，主流库支持不成熟）上 TVM 和另一个自动调优框架 Tensor Comprehensions 都明显优于 MXNet 的手写实现——说明**自动调优对"新兴算子"的追赶速度远快于人工优化库的迭代速度**。
- **嵌入式 CPU（ARM Cortex A53）**：相较 TensorFlow Lite 的手工优化算子，TVM 生成的算子全面更优，端到端推理也超过 TFLite；进一步在超低精度（2-bit 激活、1-bit 权重）量化算子上，TVM（借助 tensorize 接入 ARM 位运算 micro-kernel）单线程已优于 Caffe2 的手工超低精度库，多线程后优势更大——说明 **tensorize + AutoTVM 的组合能覆盖手工库经济上不值得覆盖的长尾精度/算子配置**。
- **嵌入式 GPU（ARM Mali-T860MP4）**：相较厂商库 ARM Compute Library，float32/float16 下端到端加速 1.2×–1.6×，验证了同一套编译栈无需为移动 GPU 单独开发即可复用。
- **FPGA 加速器（VDLA 原型）**：可 offload 的卷积层单独取得 40× 加速，但受 Amdahl 定律限制，端到端加速比被无法 offload 的 CPU-only 部分（首层卷积、残差/激活算子）拖累——说明**"能生成高效代码"和"能覆盖足够多算子"是加速器落地要同时解决的两个问题**；latency hiding（virtual thread）单独把该加速器的计算利用率从 70% 提到 88%（roofline 度量），验证了显式延迟隐藏对瘦控制加速器是必需而非锦上添花的优化。
- **优化贡献拆解**：算子融合单独贡献 1.2×–2× 的加速（Figure 4）；ML cost model 引导的搜索在同样试验预算下显著快于 blackbox 遗传算法/随机搜索收敛到好配置（Figure 12），且 cost model 单次预测（0.67ms）比真实硬件测量快几千倍，这是"可以在可行时间内搜索数十亿配置空间"的前提条件。

## 6. 与本项目（算力网多后端分布式基础设施）的关联

- **算子融合、cost model 与"子图划分"问题**：TVM 的融合决策是单机内、基于算子语义分类（injective/reduction/complex-out-fusable/opaque）的**静态规则**，本质上回答的是"这几个算子应不应该合并成一个 kernel"；算力网场景下的子图划分回答的是更粗粒度的"这部分模型该跑在哪个后端/设备上"，但两者面对的是**同一类结构性教训**：融合/切分的决策不能脱离"下游能否为切分结果生成高效代码"单独做出——TVM 用图级+算子级联合优化避免了"融合了却没有库实现"的尴尬，子图划分同样需要联动"目标后端编译器/runtime 对该子图结构的支持能力",否则会出现"划分合理但某后端跑不动或跑得很慢"的情况。TVM 的 cost model 思路（用学习到的统计模型替代精确硬件建模）也提示了子图划分收益评估的一种可行路径：与其为每个候选划分建精确性能模型，不如学习一个"划分方案 → 预期性能"的统计代理模型。
- **layout 变换传播与"跨后端 layout/量化兼容"问题**：TVM 的 layout 传播机制解决的是**单一目标硬件**下 producer/consumer 偏好布局不一致时插入 transform 的问题；算力网场景中"后端"本身是异构的（不同厂商 NPU/GPU 的原生 layout、量化格式、算子语义都可能不兼容），子图边界处的转换开销可能远高于单硬件场景内的 layout transform，甚至涉及精度损失（量化格式转换）。TVM 给出的是"如何在图上表达和传播布局偏好"的一般性框架，但**跨厂商后端之间没有统一的偏好协商机制**，这是算力网基础设施需要在 TVM 思路之上专门解决的增量问题。
- **AutoTVM 的调优日志复用与"配置组合爆炸"问题**：AutoTVM 的核心洞察——cost model 可以在相关 workload 之间迁移、tuning record 可以持久化复用——直接对应算力网"多设备型号 × 多算子 × 多 shape"导致的配置组合爆炸问题的一种应对范式：建立一个可共享的、按 (硬件型号, 算子签名, shape) 索引的调优知识库，运行时优先查库命中，只有未命中才触发新的搜索。这本质上是把"一次性调优成本"分摊到"整个算力网集群的所有请求"上，而不是每个设备/每次部署都重新搜索。
- **TVM 在动态 shape 上的短板与 Relax/Dyn 的演进**：论文中的 Relay（继承自更早的 NNVM Graph IR）以及论文本身描述的编译流程，都建立在"每层输入 shape 静态已知，为每个具体 shape 生成特化 kernel"的假设上——这对 CNN 类固定输入尺寸的模型效果很好，但对 NLP/LLM 等 batch size、序列长度频繁变化的场景不友好（每个新 shape 都要重新走一遍编译/调优流程，成本高）。这直接推动了 **Relax（TVM Unity 路线）** 引入符号 shape（symbolic shape）、shape 函数和运行时 shape 推导机制，让同一份编译产物可以处理一定范围内变化的动态 shape，减少了"为每个 shape 重新编译"的开销。这与用户关心的"大模型分布式运行"场景高度相关——请求的 batch/seq len 天然是动态的。
- **TVM 与 MLIR/IREE 路线的对比**：TVM 走的是"**自建专用 IR（Relay/Relax + TIR）+ 强力 ML 搜索**"路线，重心在于用自动化搜索逼近甚至超越手工库的单算子性能，图级 IR 和算子级 IR 都是 TVM 自成一体的设计；MLIR/IREE 走的是"**通用可扩展 dialect 基础设施 + 各硬件方言插件化 + 渐进式 lowering**"路线，重心在于让不同硬件厂商可以在统一的基础设施上定义自己的 dialect 和 lowering pass，生态的可组合性和可插拔性优先于单一路径的极致搜索能力。二者不是互斥关系：TVM 证明了"自动搜索代码生成可以打败手工调优库"这一方法论的可行性；MLIR 证明了"分层、可插拔的 IR 设计更适合支撑数量持续爆炸的硬件后端生态"。对于算力网这种要同时对接多种异构后端、且后端集合会持续增长的场景，MLIR/IREE 的分层可插拔思路更贴近"基础设施"定位，但 TVM 的自动调优/cost model 方法论仍然是各 dialect 内部做算子级优化时值得复用的核心技术资产——这也是为什么 IREE 自身的 codegen 后端也在吸收类似 auto-tuning 的思想。

## 7. 学习这篇论文时的最小必要集

**必须掌握（6-8 个点）：**

1. 计算图级优化四件套：算子融合（四类算子分类 + 融合规则）、常量折叠、静态内存规划、数据布局变换传播——以及它们如何共同决定"要为算子级生成什么样的代码"。
2. 张量表达式（TE）的 compute/schedule 解耦思想，以及 TVM 相对 Halide **去掉的三个假设**：隐式内存管理假设（→ memory scope）、固定标量/向量计算原语假设（→ tensorization）、shared-nothing fork-join 并行假设（→ virtual thread/latency hiding）。
3. 常用调度原语的语义：`tile`/`split`/`reorder`/`fuse`/`vectorize`/`unroll`/`parallel`/`bind`/`cache_read`/`cache_write`/`compute_at`，能看懂一段 schedule 代码对循环结构做了什么变换。
4. Tensorization 机制：如何用同一套 TE 语言声明硬件 intrinsic 的行为与 lowering 规则，为什么这是"支持新加速器只需 ~2k LoC"的关键。
5. GPU 的 cooperative fetching + memory scope：理解 shared memory 协作搬运为什么能提升 GPU 利用率（对应 Figure 7 的实验）。
6. Virtual thread + decoupled access-execute：理解加速器上"显式延迟隐藏"要解决什么问题、编译器如何自动插入依赖同步 token。
7. AutoTVM 的完整闭环：schedule template 定义搜索空间 → XGBoost cost model（rank objective，不预测绝对时间只预测相对顺序）→ 模拟退火探索 → RPC 分布式实测反馈 → 模型持续更新/迁移。
8. Runtime 设计：PackedFunc 统一调用 ABI 如何让多语言、多后端产物可以互操作；graph executor + RPC 如何支撑异构与远程设备部署。

**可以先跳过，以后遇到再看：**

- VDLA/VTA 具体硬件微架构细节（各 buffer 精确容量、指令队列的具体位宽设计）——理解"分层片上存储 + 显式同步"这个概念即可，具体数字对理解编译器设计意义不大。
- 论文当时 XGBoost cost model 的具体特征工程枚举（哪些 memory access/reuse 特征、具体 one-hot 维度）——理解"用 loop 程序结构特征训练一个相对排序模型"这个思路即可，特征列表已随后续版本迭代多次。
- TreeRNN 对比 XGBoost 的实验细节——这是一个"两个模型选型都可行，最终选了更快的"的工程取舍，不影响对整体框架的理解。
- 具体 FPGA PYNQ 开发板参数和各 benchmark 网络的详细层配置表（论文 Table 2）——这些是实验环境细节，用于复现实验时再查。
- AutoTVM 之后 TVM 社区演进出的 AutoScheduler/Ansor、MetaSchedule 的内部实现细节——先掌握论文原始机制的思想，后续接触现代 TVM 时再按需深入这些"下一代"实现。
