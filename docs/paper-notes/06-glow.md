# Glow：用"少量线性代数原语 + 渐进式 Graph Lowering"降低新硬件后端接入成本的神经网络编译器

> 论文元信息
> - 标题：*Glow: Graph Lowering Compiler Techniques for Neural Networks*
> - 作者/机构：Nadav Rotem, Jordan Fix 等，Facebook（现 Meta）
> - 年份：2018（arXiv v3 更新于 2019-04）
> - arXiv：https://arxiv.org/abs/1805.00907
> - 开源仓库：https://github.com/pytorch/glow

## 1. 它解决什么问题

传统机器学习框架（PyTorch、Caffe、TensorFlow）执行计算图的方式是 node-visitor：逐个节点调用对应 kernel。这在 CPU 上就已经效率不高，在异构加速器（DSA，domain specific architecture）大量出现之后，问题被进一步放大：**框架要支持一个新算子，就必须在每一种已支持的硬件上都实现一遍；反过来，硬件厂商要接入一个新加速器，就必须把框架已有的全部算子集在自己的硬件上都实现一遍。** 算子数（N）× 硬件后端数（M）的组合，使得工程成本随生态扩张呈 N×M 增长，任何一方的新增都会拖累另一方。

论文举了一个具体例子说明"直接把神经网络图翻译成通用底层代码，再指望 C 编译器优化"是不可行的：两个各自写入同一块内存的 for 循环，加上一个只读取其中一个元素的返回语句（图 1），GCC 和 LLVM 都无法证明第一个循环是死代码，也无法把读操作替换成编译期常量——因为要证明循环下标不溢出、指针不别名（alias）、语义与 C 标准一致，这些分析在通用编译器里代价太高。而这正是"从 7 层嵌套循环反向识别出这是一次卷积"的难度所在。

Glow 给出的方案是：在高层图上做"node lowering"——把 `FullyConnected`、`Convolution`、`Gradient`、`SGD` 等**语义复杂**的算子，拆解成矩阵乘、广播（broadcast）、element-wise 加减乘等一小组**线性代数原语**（linear algebra primitives）。新硬件后端不需要理解和实现 FullyConnected 这类高层语义，只需要把这一小组原语的 kernel 实现好，就能自动获得对整个算子集（包括训练用到的梯度算子）的支持。围绕这一核心思想，Glow 设计了两阶段强类型 IR：高层 dataflow graph IR 负责领域特定优化与自动微分，低层 instruction-based IR 负责内存与指令级优化。论文写作时，Cadence、Esperanto、Habana、Intel、Qualcomm 均已承诺在未来芯片中支持 Glow，可见其"降低新硬件接入门槛"的定位是面向真实产业需求的。

## 2. 整体运行框架

Glow 的编译 pipeline 可以用下面的 ASCII 图概括（对应论文 3.6 节的 9 步流程）：

```
                       ┌────────────────────────────┐
  模型文件 (ONNX/Caffe2) │        Graph Loader        │  或直接用 C++ 接口构造图
  ──────────────────────▶│  (一个 op -> 一个/多个 Node) │
                       └──────────────┬─────────────┘
                                      ▼
                       ┌────────────────────────────┐
                       │   High-Level IR (图 1)      │  dataflow / node-based
                       │  Function{Nodes} + Module   │  strongly-typed，支持
                       │  {Constant/Placeholder}     │  自动微分 & 图优化
                       └──────────────┬─────────────┘
                                      │ 2. Differentiate（若需要训练）
                                      ▼
                       ┌────────────────────────────┐
                       │   高层图优化（target-独立）    │  CSE / 常量折叠 / 死代码
                       │                              │  消除 / 冗余 transpose 消除 /
                       │                              │  conv+BN 合并 等
                       └──────────────┬─────────────┘
                                      │ 4. Node Lowering
                                      ▼
                       ┌────────────────────────────┐
                       │ High-Level IR（图 2）        │  节点已替换为线性代数原语
                       │ 只剩 MatMul/Add/Sub/Mul/... │  (仍是图，还没有内存/地址)
                       └──────────────┬─────────────┘
                                      │ 5. 再次优化（target 独立 + target 相关）
                                      │ 6. Scheduling（线性化，最小化内存占用）
                                      ▼
                       ┌────────────────────────────┐
                       │   IRGen（一对多翻译）         │
                       └──────────────┬─────────────┘
                                      ▼
                       ┌────────────────────────────┐
                       │   Low-Level IR（图 3）       │  instruction-based，
                       │  declare{显式内存} +         │  address-only；
                       │  program{指令序列}           │  可做内存分配/复用/
                       │                              │  in-place/指令调度
                       └──────────────┬─────────────┘
                                      │ 8. 低层 IR 优化
                                      ▼
                       ┌────────────────────────────┐
                       │ 后端特定优化 & CodeGen        │  CPU: LLVM IR + libjit
                       │ (target-specific graph/IR)  │  加速器: 各自 codegen
                       └──────────────┬─────────────┘
                                      ▼
                       ┌────────────────────────────┐
                       │        Glow Runtime         │  Partitioner / Provisioner /
                       │  AOT: 独立 object file       │  DeviceManager / Executor
                       │  JIT: 立即执行 + 调试信息      │
                       └────────────────────────────┘
```

各组件输入/输出与职责：

- **Graph Loader**：输入 ONNX/Caffe2 序列化模型（或用户直接调用 C++ API），输出 High-Level IR 图。这一步是"一个高层算子 → 一个或多个 Node"的直接翻译，还没有做任何优化。
- **High-Level IR**：dataflow、node-based、强类型的图。图被组织成 `Module{Function{Node}}`；`Constant`（编译期已知的权重）和 `Placeholder`（运行期才绑定 tensor 的输入/输出）是两种 storage node，归属于 Module，可被同一 Module 下的多个 Function 共享。
- **Differentiation**：在**lowering 之前**对图求导，因为 lowering 会破坏某些节点的原始语义（下节详述），求导必须看到语义完整的原图。
- **高层图优化**：在原始语义的图上做代数化简、CSE、死代码消除、常量折叠，以及 transpose 冗余消除、conv+BatchNorm 合并等领域特定优化。
- **Node Lowering**：把 `FullyConnected`、`Convolution`、`SGD`、`Regression` 等复杂节点，替换成 `MatMul`/`Add`/`Sub`/`Mul`/`Broadcast` 等原语组成的子图，仍然停留在 High-Level IR 层（还是图，还没有内存地址概念）。
- **IRGen**：把 lowering 后的 High-Level 图一次性翻译成 Low-Level IR 指令序列（一对多映射），这一步是可选的——如果某个后端有自己的一套软件栈更愿意直接消费 Node 级图，可以跳过 IRGen。
- **Low-Level IR**：instruction-based、operate on address（而不是抽象的 tensor 值）的表示，分为 `declare`（生命周期贯穿全程的显式内存区域）和 `program`（指令列表，含局部 `alloc`/`dealloc`）两段。每个操作数带 `@in`/`@out`/`@inout` 标注，供优化器判断 buffer 共享、拷贝消除是否合法。
- **后端特定优化与 CodeGen**：CPU 后端基于 LLVM，将 Low-Level IR 转成调用 `libjit`（用 C++/OpenCL vector syntax 手写的标准库）的 LLVM IR，再交给 LLVM 生成机器码；也可以在这一层继续构造更贴近硬件的 target-specific 图（例如 CPU 后端为不同 channel 规模的卷积生成专门的 `CPUConvDKKC8` 节点）。加速器后端可以实现自己的 codegen 路径。
- **Runtime**：`Partitioner` 按内存/耗时/通信成本切分子图到多设备，`Provisioner` 把子图编译并分配到具体设备，`DeviceManager` 管理单个物理加速器，`Executor` 负责异步调度与结果收集；产物既可以是可直接部署的独立 object file（AOT），也可以是 JIT 立即执行（附带调试信息，便于用调试器下断点或用 profiler 分析）。

**两级 IR 为什么要在这里切一刀**：高层图 IR 保留了"这是一次卷积""这是一次矩阵乘"这种领域语义，代数化简（如冗余 transpose 消除、conv+BN 合并）、自动微分都必须在语义完整、没有地址概念的图上进行——一旦下降到带内存地址的指令流，编译器就很难再反向证明两个 buffer 不别名、某个 loop 是否可以消除（正是图 1 揭示的困境）。而内存分配、buffer 复用、原地（in-place）操作、指令调度这些优化，恰恰需要"每个值都有明确的地址和生命周期"这个前提，这在图 IR 里是不存在的（图里的边只表示数据依赖，没有物理内存位置）。所以 Glow 选择：**先在没有内存概念的图上做完所有"看语义"的优化和 lowering，再一次性生成有内存概念的指令流，去做"看地址"的优化**。这也解释了为什么 node lowering 被安排在 IRGen 之前而不是之后：lowering 后的图仍是图结构，可以继续参与图级优化和调度决策，且 lowering 后暴露出的细粒度原语能让后续的 target-specific 优化有更多操作空间。

## 3. 核心特性逐条拆解

### 3.1 多级 IR 与渐进式 Lowering（Progressive Lowering）

**是什么**：编译流程中前几个阶段（图加载、微分、高层优化、node lowering、IRGen、低层优化）对所有后端都是共享、target-independent 的；越接近最终指令选择，IR 越 target-specific——CPU 后端在 Low-Level IR 之上还会构造自己的 target-specific 图（如为不同规模的卷积选择不同的 filter 内存布局和 tile 大小）。

**为什么这样设计**：这不是 Glow 独创，而是编译器和虚拟机的常见做法——逐步 canonicalize、优化、下降为指令流。好处是复用：两级公共 IR 意味着所有后端都能免费获得同一套图优化和内存优化，而不需要每个后端各自实现一遍。

**带来什么能力**：新增一个后端时，只需要在"分叉点"之后接入自己的 codegen 逻辑，之前所有阶段（微分、图优化、lowering、调度、内存分配）都是白拿的。

### 3.2 Node Lowering：一小组原语 + 自动微分的"顺带"获得

**是什么**：Lowering 把高层算子替换为线性代数原语组成的子图。论文给出的典型例子是 `FullyConnected` → 矩阵乘（MatMul）+ 广播加（broadcasted Add）；`SGD`（随机梯度下降）节点被拆成 `Sub`/`Mul`/`Add` 序列。Lowering 发生在 High-Level IR 阶段、IRGen 之前，且**必须排在自动微分之后**。

**为什么这样设计**：论文给出三点理由：(1) lowering 后的新图结构可能开启额外的图级优化机会；(2) 新的图结构会影响指令调度器的决策；(3) lowering 之后仍允许后端在这个更细粒度的图上做 target-specific 优化。至于"必须在微分之后"，是因为 lowering 不保语义（not semantics-preserving）——例如 `Regression` 节点在推理场景是 no-op，但在训练场景要被翻译成 element-wise subtract（用于计算误差）；如果在微分之前就把 Regression lowering 掉，微分器就无法知道该对哪个操作求导。

**带来什么能力（这是本节的核心）**：自动微分是在**未 lowering 的原图**上进行的——微分器只需要理解一小组高层节点（如 `Regression`、`FullyConnected`）如何求导，产生的梯度图（可能包含 `DivGrad`、`SGD` 等"求导专用"节点）随后会被送进**同一条 lowering pipeline**，被拆解成 `Sub`/`Mul`/`Add`/`Save` 等原语。也就是说，后端完全不需要单独实现 `DivGrad` 或 `SGD` 这样的指令，它们和推理阶段的算子一样，最终都收敛到同一小组线性代数原语上。论文图 2 展示的例子：对标量 `A` 做回归并自动微分后，得到的计算图只由 `Sub`、`Mul`、`Add`、`Save`、`Splat`（产生学习率常量）这些原语节点构成。

对比经典框架：如果一个框架不能自动生成融合 kernel，就必须为每一个未 lowering 的高层算子在每种硬件上手写 CUDA/CPU kernel（数以百计），这直接限制了它支持新硬件的速度和范围。

### 3.3 高层图优化

高层优化都作用在语义完整、无内存地址概念的图上，包括：常量折叠、CSE、死代码消除（针对 unused `Constant` 节点）、`Constant` 的编译期变换（如 transpose、量化，见 3.5）、冗余 transpose 消除，以及 `Convolution` 与 `BatchNorm` 的融合（论文 ResNet50 案例）。这些优化能够成立的前提是：图上的节点携带完整的领域语义（"这是一次卷积"），而不是一串通用的 load/store 指令——这正是 3.1 中反复强调的高层 IR 存在的意义。

### 3.4 低层 IR 的内存视角

**是什么**：Low-Level IR 是 instruction-based、operate-by-address 的表示。一个 Function 的 IR 分两段：`declare`（声明贯穿全程存活的全局内存区域，类似 C 里的全局变量）和 `program`（指令序列，`alloc`/`dealloc` 类似 LLVM 的 `alloca`，代表局部分配）。每个操作数带 `@in`（只读）/`@out`（只写）/`@inout`（读写）标注。

**为什么这样设计**：这类内存优化在高层图上是不可能做的，因为高层图里内存并不是一等公民（一个 Node 的输出只是一条数据边，没有物理地址）。只有把每个中间结果具体化为一段带地址的 buffer，编译器才能推理"这两个 buffer 是否可以复用""这次写是否可以原地（in-place）完成"。`@in`/`@out`/`@inout` 标注直接服务于这个目的：它们告诉优化器某个 buffer 的读写模式，从而判断拷贝消除（copy elimination）和 buffer 共享是否合法。此外，如果一个网络不是纯 inference-only（例如要支持训练），某些前向的中间结果必须为反向传播保留，这会限制部分内存复用优化——这个约束本身也只能在指令级 IR 上表达。

**带来什么能力**：(1) 静态内存分配——`alloc` 指令本身不真正分配内存，只是标记一段 activation 的生命周期起点，真正的分配由低层内存分配器统一完成，把所有 buffer 打包进一段 coalesced 内存区域，从而压低网络运行时的可变内存footprint；(2) 原地操作——element-wise 运算等可以被识别为可以原地改写输入 buffer；(3) 指令调度——指令级表示可以显式表达设备特定操作（如异步 DMA），调度器可以据此重排指令、用计算掩盖内存搬运延迟。

### 3.5 量化（Quantization）：Profile-Guided 两阶段流程

**是什么**：Glow 支持把浮点网络转换为有符号 8-bit 整数（Int8）网络。量化 tensor 的类型由底层元素类型（Int8）加上一对 `scale`/`offset` 参数描述，换算公式为 `value = (input - offset) * scale`。量化流程分两阶段：

1. **Profiling 阶段**：在图中插入专门的 profiling 节点，优化这个带 profiling 节点的图，跑一遍真实 inference，记录网络各阶段激活值（activation）的数值范围。
2. **量化重编译阶段**：用第一阶段收集到的 range 信息重新编译网络，把浮点子图替换为整数计算的"孤岛"，并对量化后的图做静态优化，目标是让量化网络的输出落在原浮点网络输出的可接受范围内。

**为什么这样设计**：网络不同部分的数值范围差异很大（有的在 0~1，有的在成百上千），单一的全局 scale 要么对小数值精度不足，要么把大数值截断，所以必须按"stage"（大致对应算子/子图）分别确定 scale/offset，这就需要先跑一遍 profiling 才能确定。

**带来什么能力**：量化图中会插入大量 `rescale` 节点（在不同数值范围之间转换），Glow 在编译期对这些节点做三类优化：(1) 尽量减少 float↔int 之间的转换次数——像 `transpose`、`concat` 这类算子本身可以在两种类型上操作，通过选择合适的 representation 减少转换；(2) 把 `rescale` 折叠进"产生数值的算子"里直接消除；(3) 主动 rescale 让硬件用更简单的方式实现某些算子——例如把 `max` 两侧都归一化到同一 scale，硬件就能用一次简单比较完成 `max`，而不需要先做数值对齐。训练感知（training-aware）量化在论文写作时被列为 future work。

### 3.6 AOT 编译产出与 JIT 的取舍

论文在 CPU 后端一节明确写到："The backend can emit a stand-alone object file to disk or execute code in just-in-time mode"——即 CPU 后端基于 LLVM，既可以把编译结果**离线**输出为一个独立的 object file，也可以**JIT**立即执行。这正是 Glow 项目广为人知的"bundle"式 AOT 部署能力的论文侧描述（论文正文没有直接使用"bundle"一词，这是补充说明，供读者对照 Glow 开源项目的 `model-compiler` 工具理解）。

两种模式的取舍：JIT 模式下 Glow 还会附带调试信息（debug info），可以在调试器里对具体算子下断点，或用 profiler 分析性能，适合开发迭代阶段；而离线产出的 object file（进一步打包成包含权重、符号表的 bundle）不依赖 Glow runtime 本身存在，只需要链接一个极薄的 C 运行时即可执行，特别适合**没有操作系统、内存高度受限**的嵌入式/端侧部署场景——这也是论文列出的多个 committed silicon 合作方（Cadence、Esperanto、Habana 等，很多面向嵌入式/DSP 场景）会关心的部署形态。

### 3.7 新后端的最小接入成本

论文没有给出一份具体的 C++ 接口清单，但从整体设计可以清楚推出一个新硬件后端至少需要做的事：

1. **实现一小组线性代数原语的 kernel**：因为 node lowering 已经把 `FullyConnected`、`Convolution`（在某些路径下）、`SGD` 等复杂算子都拆解成了 MatMul/Add/Sub/Mul/Broadcast 等原语，后端不需要理解和实现这些高层算子的语义，只需要吃透这一小组原语。CPU 后端自身的做法是编译一份小型 target-independent 标准库 `libjit`（C++ 写成，编译为 LLVM bitcode），在编译期把 tensor 维度、buffer 地址等参数替换为常量，交给 LLVM 优化和向量化。
2. **声明自己支持/不支持哪些节点或指令**：不同加速器能力差异很大（这是论文反复强调"each of their accelerators will likely differ in capabilities"的原因），后端需要有能力表达"这个原语我能处理，那个不能"，未支持的部分交由 Glow 的默认路径（如继续 lowering 或走 CPU fallback）处理。
3. **提供内存分配与调度的接入点**（可选覆盖默认策略）：Low-Level IR 的静态内存分配、指令调度默认由 Glow 公共设施完成，但后端可以按硬件特点（如需要表达异步 DMA）介入。
4. **提供 compile/codegen 方法**：把（可能经过后端自定义变换后的）Low-Level IR 或图，转换成该硬件可执行的代码或配置，并对接 Runtime 层的 `DeviceManager` 完成加载、内存搬运和执行。

论文还提到 Glow 使用自动代码生成工具 **ClassGen**（类比 LLVM 的 TableGen）来定义 Node 和 Instruction 类，自动生成相等性判断、哈希、克隆、打印、校验等样板方法（图 4 是 `AvgPool` 指令的 ClassGen 定义示例），这降低了框架维护者新增一个原语/指令时的工程成本，但这属于工程实现细节，不影响对整体架构的理解。

## 4. 使用示例

### 4.1 High-Level IR 中的自动微分结果（论文图 2，原文内容）

对标量 `A` 做回归并做自动微分后，Glow 转储出的高层图（论文原文示意，节点关系如下）：

```
Placeholder "Ex" (trainable=0)
Placeholder "A"  (trainable=1)
Placeholder "ret"(trainable=0)
Splat learningRateSplat = -3.000000e-02

Sub  rgn_grad  = A - Ex           // 回归的梯度：预测值减真实值
Mul  dx1       = rgn_grad * learningRateSplat
Add  newW      = A + dx1          // SGD 更新：W_new = W - lr * grad
Save ret       <- (regression 前向输出)
Save A_saveGrad <- newW           // 把更新后的值写回 Placeholder A
```

可以看到，图里只剩下 `Sub`、`Mul`、`Add`、`Splat`、`Save` 这些线性代数原语，没有出现 `Regression`、`SGD` 这些高层节点——它们已经被 lowering 掉了。这正对应 3.2 节的结论：微分产生的 `DivGrad`/`SGD` 等"求导专用"语义，最终都被同一条 lowering pipeline 收敛为同一组原语，后端不需要单独实现它们。

### 4.2 Low-Level IR 指令序列（论文图 3，原文抄录并逐行解释）

一个包含卷积、ReLU、池化的小网络，IRGen 之后（尚未做低层优化）的 IR：

```
declare {
  %input   = weight float<8 x 28 x 28 x 1>, broadcast, 0.0
  %filter  = weight float<16 x 5 x 5 x 1>, xavier, 25.0
  %filter0 = weight float<16>, broadcast, 0.100
  %weights = weight float<10 x 144>, xavier, 144.0
  %bias    = weight float<10>, broadcast, 0.100
  %selected = weight index<8 x 1>
  ...
  %result  = weight float<8 x 10>
}
program {
  %allo  = alloc float<8 x 28 x 28 x 16>
  %conv  = convolution [5 1 2 16] @out %allo, @in %input, @in %filter, @in %bias
  %allo0 = alloc float<8 x 28 x 28 x 16>
  %relu  = max0 @out %allo0, @in %allo
  %allo1 = alloc index<8 x 9 x 9 x 16 x 2>
  %allo2 = alloc float<8 x 9 x 9 x 16>
  %pool  = pool max [3 3 0] @out %allo2, @in %allo0, @inout %allo1
  ...
  %deal6 = dealloc @out %allo6
  %deal7 = dealloc @out %allo7
  %deal8 = dealloc @out %allo8
  %deal9 = dealloc @out %allo9
}
```

逐行解释：

- `declare` 段里的 `weight` 声明的是贯穿全程存活的显式内存区域，语法为 `weight <elem类型><shape>, <初始化策略>, <参数>`，例如 `%filter` 用 xavier 初始化、参数 25.0（对应 fan-in 相关的缩放）；`%input`/`%filter0`/`%bias` 用 broadcast（即用同一个常量填满整个 tensor）初始化。
- `%allo = alloc float<8 x 28 x 28 x 16>`：`alloc` **不真正分配内存**，只是标记一段 activation 生命周期的起点——真正的物理内存由低层内存分配器统一规划进一段 coalesced 缓冲区。
- `%conv = convolution [5 1 2 16] @out %allo, @in %input, @in %filter, @in %bias`：`[5 1 2 16]` 分别是 kernel size=5、stride=1、pad=2、output depth=16；`@out`/`@in` 标注了每个操作数的读写方向，供后续做拷贝消除/buffer 共享判断。
- `%relu = max0 @out %allo0, @in %allo`：ReLU 被表示为与 0 做逐元素 `max`（`max0`），这也是"用少量原语覆盖高层语义"的一个直接例子。
- `%pool = pool max [3 3 0] @out %allo2, @in %allo0, @inout %allo1`：`[3 3 0]` 是 kernel=3、stride=3、pad=0；`%allo1` 是 `@inout`（记录每个窗口最大值的索引，用于反向传播），体现了"forward 数据要为 backward 保留"这一约束如何在指令级 IR 里落地为一个额外 buffer。
- `dealloc` 标记对应 buffer 生命周期的结束，配合 `alloc` 共同界定每个 buffer 的存活区间，供内存分配器复用。

### 4.3 ClassGen：AvgPool 指令的声明（论文图 4，原文抄录）

```
BB.newInstr("AvgPool")
  .addOperand("Dest", OperandKind::Out)
  .addOperand("Src", OperandKind::In)
  .addMember(MemberType::VectorUnsigned, "Kernels")
  .addMember(MemberType::VectorUnsigned, "Strides")
  .addMember(MemberType::VectorUnsigned, "Pads")
  .autoIRGen()
  .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
  .addGradientInstr({"Dest"}, {"Dest", "Src"});
```

这段 DSL 声明了 `AvgPool` 指令有一个输出操作数（`Dest`）、一个输入操作数（`Src`），以及三个数组型成员（`Kernels`/`Strides`/`Pads`）；`.autoIRGen()` 让框架自动生成从对应 Node 到该 Instruction 的翻译代码；`.autoVerify(...)` 自动生成类型校验（要求 `Dest` 与 `Src` 元素类型一致）；`.addGradientInstr(...)` 声明该指令参与自动微分时，哪些操作数是梯度计算所需的。可以看到，新增一个原语级指令的边际成本，主要就是声明它的操作数和属性，样板代码（相等性判断、哈希、克隆、打印等）全部自动生成。

### 4.4 `model-compiler` / `image-classifier` 命令行工具（示意，非论文原文）

> 说明：论文正文没有给出具体命令行示例，以下命令按 Glow 开源项目公开的工具惯例整理，用于说明 AOT 编译与 profile-guided 量化在命令行层面大致如何组织，标注为**示意**。

```bash
# 第一步：跑 profiling，收集各激活值的数值范围，产出一份 profile YAML
./image-classifier images/*.png \
  -model=resnet50.onnx \       # 待编译的浮点模型
  -backend=CPU \               # 用哪个后端跑 profiling（通常用 CPU 即可）
  -dump-profile=profile.yaml   # 输出量化所需的 range 信息

# 第二步：结合 profile 信息，把模型 AOT 编译成一份可脱离 Glow runtime 部署的 bundle
./model-compiler \
  -model=resnet50.onnx \                  # 输入模型
  -backend=CPU \                          # 目标后端（也可以是接入的加速器后端）
  -load-profile=profile.yaml \            # 使用第一步收集的数值范围做量化
  -quantization-schema=symmetric \        # 量化格式：对称量化（也可选 asymmetric）
  -emit-bundle=./bundle_out \             # 产出目录：object file + 权重 + 头文件
  -network-name=resnet50                  # bundle 中生成符号的前缀名
```

产出的 bundle 通常包含：一份包含所有算子实现的 object file（`.o`）、一份权重二进制、以及一份声明输入/输出/权重符号的 C 头文件，部署方只需要把这三者链接进自己的固件/运行时即可执行推理，不需要 Glow 本身参与运行期。

### 4.5 自定义后端需要实现的 C++ 接口骨架（示意，非论文原文）

> 说明：论文没有给出具体接口代码，以下按论文 3.7/5 节描述的职责（判断算子支持度、生成代码、接入内存与调度）整理出的示意骨架，帮助理解"最小接入面"，标注为**示意**。

```cpp
// 示意：Glow 风格的 Backend 接口骨架，非论文原文
class MyAcceleratorBackend : public Backend {
public:
  // 声明后端标识，供 Runtime 匹配设备类型
  std::string getBackendName() const override { return "MyAccelerator"; }

  // 判断某个 Low-Level IR 指令 / Node 是否被该后端支持
  // 未支持的部分会被 Glow 继续 lowering 或回退到其他后端
  bool isOpSupported(const NodeInfo &NI) const override {
    switch (NI.getKind()) {
    case Kinded::Kind::MatMulNodeKind:
    case Kinded::Kind::AddNodeKind:
    case Kinded::Kind::ReluNodeKind:
      return true; // 只需要实现这一小组线性代数原语
    default:
      return false;
    }
  }

  // 把（已经 lowering + 优化过的）Function 编译为该硬件可执行的产物
  Expected<std::unique_ptr<CompiledFunction>>
  compile(Function *F, const BackendOptions &opts) const override {
    // 1. 可选：在此基础上继续做 target-specific 图变换
    // 2. 生成设备指令流 / 调用厂商 SDK
    // 3. 打包成 CompiledFunction，供 DeviceManager 加载执行
    ...
  }

  // 声明该后端偏好的内存对齐、是否支持某类内存优化等
  bool shouldLower(const Node *N) const override { return true; }
};
```

## 5. 关键实验/评估结论

论文的实验部分（第 7 节）规模不大，需要诚实说明其局限：只在**单台机器、单核 CPU**（Kaby Lake Intel Core i7-7600U，不支持 AVX-512）上，对**两个模型**（ResNet50、VGG19）、**一个 batch size 家族**（2/4/8，性能差异不大，论文只报告 batch=8 的结果）做了对比，对比对象是 TensorFlow-1.7（开启 XLA）和 TVM（LLVM 6.0，**未开启 auto-tuning、未用专门 schedule**）。没有任何加速器（GPU/ASIC/DSP）上的 benchmark 数据，尽管论文反复强调多个厂商承诺支持 Glow 的加速器芯片。

主要结论：

- **Glow 比 TensorFlow 快至多 2.7×**：原因是 TensorFlow 通过 Eigen 库实现卷积，走的是经典的 im2col + 矩阵乘路径；而 Glow 直接编译 direct convolution（配合 5.2 节的 operator stacking），避免了 im2col 展开数据带来的额外内存开销，并且做了 shape-aware 的代码生成（即知道具体张量形状后生成专门化代码，而不是通用形状的循环）。
- **Glow 比 TVM 快至多 1.3×**：但论文自己说明了这个对比的前提局限——TVM 没有开启 auto-tuning 和专门 schedule，作者预期开启后 TVM 的性能会提升。这说明该实验更适合解读为"**不依赖自动调优搜索、单纯靠手工编译期优化（direct conv + 向量化 + operator stacking）也能拿到有竞争力的性能**"，而不能得出"Glow 架构上限比 TVM 更高"的结论。TVM 同样不走 im2col，也是 lowering 到低层 IR（Halide-based）后生成向量化代码，两者在"避免 im2col、生成 tiling 友好的内存访问模式"这个层面是相似的，区别主要在于人工优化路线（Glow）与自动调优路线（TVM）的取舍。
- 除了这组量化对比，论文的其余论证基本是**定性**的：node lowering 减少了后端要实现的算子数量、内存优化降低了运行时可变内存占用、operator stacking 减少了对同一块内存的重复访问次数。这些论证没有配套的独立 ablation 数据（例如"关掉 operator stacking 后性能下降多少"），读者需要将其作为架构设计动机的说明，而非严格的量化证据来看待。

## 6. 与 TVM 的对比

Glow 和 TVM 是 2018 年前后几乎同期出现、目标高度相似（都是"神经网络图 → 多硬件后端"编译器）但技术路线明显分叉的两个系统，逐项对比如下。

**IR 层次设计**：Glow 的公共 IR 只有两级——High-Level IR（node-based dataflow graph，强类型，承载自动微分和图级代数优化）和 Low-Level IR（instruction-based，address-only，承载内存分配/复用/调度），往下则完全交给各后端自行处理（如 CPU 后端在 Low-Level IR 之上再叠加自己的 target-specific 图）。TVM/NNVM 是三级：Relay（图级 IR，类似 Glow 的 High-Level IR）→ TE（Tensor Expression，描述"计算是什么"而不涉及"怎么算"）→ TIR（Halide 衍生的循环级 IR，可以显式表达 tile/vectorize/parallel/unroll 等循环变换），最后再生成 LLVM/CUDA/Metal/OpenCL。**关键差异**在于 Glow 的低层 IR 没有"循环"这个概念给用户操作——循环结构完全隐藏在后端自己调用的 libjit kernel 和 LLVM 向量化器里；而 TVM 的 TIR 把循环变换显式建模为一等公民，可以在这一层做系统化的 schedule 搜索。

**算子集策略**：Glow 用 node lowering 把大量高层算子收敛到一小组线性代数原语，几乎不给"同一个算子多种实现方式"留操作空间——性能主要来自"为这组原语选一个合适的直接实现（如 direct convolution）+ LLVM 自动向量化 + operator stacking（自动融合连续的 data-parallel 算子）"这类工程化手段。TVM 走的是 TE 描述 computation + schedule 搜索空间的路线：同一段 computation 可以有大量不同的 schedule（不同的 tiling/vectorize/unroll/reorder 组合），AutoTVM/AutoScheduler 用 cost model 和搜索算法（模拟退火、机器学习引导搜索等）自动探索这个空间，为不同硬件生成不同的最优循环结构。简言之：**Glow 用"减少算子种类"换低接入成本；TVM 用"扩大同一算子的实现搜索空间"换极致性能，但需要付出搜索时间**。

**优化来源**：Glow 的优化基本是编译器工程师手工设计的 pass（代数化简、conv+BN 合并、operator stacking、CPU 后端针对不同 channel 规模选择不同 filter 内存布局），依赖人工经验积累。TVM 的核心卖点则是自动化调优——不再靠人工穷举 schedule，而是用搜索/学习算法在真实硬件上跑测量、拟合 cost model，逐步收敛到接近最优的 schedule。这是二者路线分歧的根本所在。

**量化方案**：两者都采用 profile-guided 的思路（先跑一遍网络收集数值统计，再据此生成量化图），细节上 Glow 更强调图级的代数优化——最小化 float/int 转换次数、把 rescale 折叠进产生数值的算子、为了让硬件用简单比较实现 `max` 而主动对齐两侧 scale。TVM 生态在量化之后仍然会把计算送进 schedule/auto-tuning 路径，去为量化后的整数计算搜索硬件最优实现，量化本身更像是"给后续调优換一种更省内存带宽的数据类型"，图级代数优化的比重相对较小。

**部署形态**：Glow 可以把编译产物输出为一个独立的 object file（进一步打包为包含权重和头文件的 bundle），部署时不依赖 Glow runtime 本身存在；同时也提供了一层完整的多设备 Runtime（Partitioner/Provisioner/DeviceManager/Executor）用于在同一台主机上管理多个模型、多张加速卡的并发推理请求。TVM 传统上产出一个 runtime module（如 `.so`/`.tar`），配合轻量的 TVM runtime 加载执行，其后也发展出面向不同部署场景的 runtime 变体。两者都支持"离线编译，产物部署"的模式，但 Glow 论文里对多加速器卡、多模型并发调度这层抽象的描述更具体、更偏向"数据中心异构加速卡池"的运维场景。

**新硬件接入方式**：Glow 的核心卖点就是"新硬件只需要实现一小组线性代数原语"，instruction selection、内存分配、图调度都由公共设施代劳，接入曲线相对平缓。TVM 要接入新硬件，需要对接其 codegen/schedule 体系——编写 tensorize intrinsics、定义硬件的 cost model，或者走 BYOC（Bring Your Own Codegen）机制注册一个自定义后端；工作量通常更大，但换来的是可以复用 auto-tuning 基础设施在新硬件上搜索出更优的实现。

**动态 shape 支持**：论文写作的时代，两者都基于**静态 shape** 假设。Glow 显式选择了严格类型系统（strongly-typed，不支持参数化 shape），不同 batch size 需要生成多份 Glow 图，或者在运行时重新 JIT 编译；论文提到考察过 Swift generics 式的参数化类型机制，但认为多数硬件加速器不支持这类抽象，因此放弃。TVM 同期的 Relay/TE 设计同样偏静态 shape（后续版本才逐步引入动态 shape 支持）。**这提示我们：动态 shape 在图级 DL 编译器的早期设计里普遍是"先放一放"的问题，这也正是我们这个项目要重点关注的前沿问题之一。**

**各自适合的场景**：Glow 更适合"目标硬件明确、希望快速把一个新 ASIC/DSP 接入生态、不愿意为 auto-tuning 投入大量搜索时间、且部署环境可能没有操作系统或内存高度受限"的场景（论文里 Cadence/Esperanto/Habana 等合作方多半符合这个画像）。TVM 更适合"需要在多样化硬件上压榨极致性能、有时间预算做离线调优、或者需要为新算子灵活表达 schedule"的场景，尤其是云端/数据中心这类对峰值性能敏感、可以承受一次性调优成本的场景。

## 7. 与本项目（算力网多后端分布式基础设施）的关联

- **"少量原语 + lowering" 对"后端最优计算图不一致"与"配置组合爆炸"的启示**：如果所有后端的最小支持面都收敛到同一组线性代数原语，那么"每个后端要维护一份自己的最优计算图"这个问题的组合规模，就从"任意高层算子的任意组合"降到了"这一组原语的组合"——配置空间被显著压缩。进一步地，配置组合爆炸问题（设备类型 × shape × batch × 精度 × 量化方式）里，如果所有后端都统一走同一份 lowering 结果，我们可以只对这一小组原语做一次面向"常见 shape/精度"组合的调优与缓存，再让每个后端自行决定要不要接管某个子图（类似 ExecuTorch 的 backend delegate、ONNX Runtime 的 Execution Provider 机制）——本质上是把"要不要接管"这个决策和"如何执行"这个决策解耦。
- **profile-guided 量化对"跨后端量化格式兼容"的启示**：Glow 的两阶段流程（先做与硬件无关的 profiling，收集数值 range；再依据 range 生成量化图）产出的中间产物——各阶段激活值的数值范围——本身是**硬件无关**的。这提示我们可以把"收集数值统计"和"选择目标量化格式（对称/非对称、per-tensor/per-channel、int8/int4）"彻底解耦成两个独立阶段：同一份 profiling 结果可以喂给多个后端各自的量化格式转换器，避免"每换一个后端就要重新跑一遍 profiling"的重复开销，这对多后端场景下量化格式的兼容性工程有直接借鉴价值。
- **AOT bundle 对"低成本后端切换"与边缘/异构部署的启示**：Glow 把编译产物做成不依赖 runtime 存在的独立 object file/bundle，这种形态天然适合"离线编译一次，部署到多个无 OS、资源受限设备"的场景。如果把 bundle 的产出格式设计为与具体 runtime 进一步解耦、可重定位、可并存多份候选实现，理论上就能支持"运行时按需在多份候选 bundle 之间切换所对应的后端实现"——这对我们关注的"运行状态变化后如何低成本切换执行后端"问题有直接参考意义：即把 compile-time 产物设计成可插拔、可横向比较的形式，而不是与某个特定运行环境强绑定的黑盒。

## 8. 学习这篇论文时的最小必要集

**必须掌握的 7 个点**：

1. 两级 IR 的边界划分——High-Level（图，语义完整，负责微分和代数优化）vs Low-Level（指令，带地址，负责内存和调度优化）——以及为什么必须在这里切一刀（图 1 揭示的"通用编译器无法反向识别领域语义"问题）。
2. Node Lowering 的核心思想：用一小组线性代数原语覆盖大量高层算子，从而把新硬件接入成本从"实现全部算子"降到"实现这一小组原语"。
3. Lowering 与自动微分的顺序关系——必须先微分再 lowering，因为 lowering 不保语义（`Regression` 节点在推理/训练下的不同行为是典型例子），以及自动微分产生的梯度节点如何顺带被同一条 lowering pipeline 收敛为原语。
4. 低层 IR 的内存视角：`@in`/`@out`/`@inout` 标注、`alloc` 只标记生命周期而非真正分配、静态内存分配把所有 buffer 打包进一段 coalesced 区域。
5. Profile-guided 量化的两阶段流程（profiling 收集 range → 依据 range 重新编译成量化图），以及 rescale 折叠、对齐 scale 以简化硬件比较这类图级优化。
6. Operator Stacking 与算子融合（fusion）的区别：融合需要为每种算子排列组合单独写 kernel，stacking 则是自动把连续的 data-parallel 算子拼接成一次内存遍历，不需要穷举排列组合。
7. Glow 与 TVM 在"人工设计优化 vs 自动搜索调优"这条根本路线上的分歧，及各自在新硬件接入成本与峰值性能之间的取舍。

**可以先跳过的内容**：

- ClassGen 的具体代码生成实现细节（属于工程基础设施，不影响理解整体架构）。
- Predication（谓词执行）机制的细节——这是面向 RNN 变长 batch 优化的一个具体特性，与两级 IR 的主线设计关系不大，可以后续用到时再回看。
- Runtime 层 Partitioner/Provisioner 具体的图切分算法细节（先建立"存在这样一层多设备调度抽象"的认识即可，具体切分策略属于工程实现）。
- CPU 后端里 x86 汇编、SIMD 内存布局变换（`CPUConvDKKC8`、5 维 filter layout）等偏硬件微架构优化的细节，这些是"一个具体后端如何做到极致性能"的案例，不是 Glow 架构设计本身要传达的核心信息。
