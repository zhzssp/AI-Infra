# AI 编译器前置核心概念：把工具串成知识体系

> **本文档的定位**
> - 这不是又一个项目说明书，而是**把仓库里所有学习文档与动手项目粘在一起的「公共词汇表」**。
> - 根目录 [`../README.md`](../README.md) 的目标是「算力网上大模型分布式运行基础设施」——需要同时会**分布式怎么切**、**异构后端怎么统一**、**运行时怎么决策**。LLVM / MLIR / IREE / TVM / ONNX / ExecuTorch / CUDA fatbin 各自讲透一块，但中间有一批**跨项目反复出现的概念**；缺了它们，你会觉得每本书都读懂了，却画不出一张总图。
> - 本文只讲这些**前置 / 横切**概念：是什么、为什么需要、在哪些项目里以什么名字出现。细节仍回各自学习文档。
>
> **怎么用**
> - **开局读一遍第 1～2 章**（总图 + 概念索引），建立坐标系。
> - 学某个项目卡住时，用[第 10 章对照表](#第-10-章-概念--深入材料对照)跳到对应章节补概念，再回去读项目文档。
> - 不要试图在这里替代 MLIR Dialect Conversion 或 IREE HAL 的精读——那是各专题文档的事。
>
> **一句话读法**：如果只有一小时，读[第 1 章总图](#第-1-章-一张总图从模型到异构设备)、[第 2 章概念索引](#第-2-章-概念索引按学习目标分组)、[第 9 章分布式与编译的接缝](#第-9-章-分布式与编译的接缝分片如何变成-ir)。

---

## 目录

- [第 1 章 一张总图：从模型到异构设备](#第-1-章-一张总图从模型到异构设备)
- [第 2 章 概念索引：按学习目标分组](#第-2-章-概念索引按学习目标分组)
- [第 3 章 图级抽象：计算图、算子、融合、划分](#第-3-章-图级抽象计算图算子融合划分)
- [第 4 章 调度与局部性：算法/调度分离、tiling、Roofline](#第-4-章-调度与局部性算法调度分离tilingroofline)
- [第 5 章 数据形态：tensor / buffer / layout / 量化 / 地址空间](#第-5-章-数据形态tensor--buffer--layout--量化--地址空间)
- [第 6 章 中间表示与变换：SSA、多层 IR、Pass、渐进 lowering](#第-6-章-中间表示与变换ssa多层-irpass渐进-lowering)
- [第 7 章 代码生成与变体：kernel、ISA 层级、多目标打包](#第-7-章-代码生成与变体kernelisa-层级多目标打包)
- [第 8 章 运行时抽象机：设备、队列、同步、AOT/JIT](#第-8-章-运行时抽象机设备队列同步aotjit)
- [第 9 章 分布式与编译的接缝：分片如何变成 IR](#第-9-章-分布式与编译的接缝分片如何变成-ir)
- [第 10 章 概念 → 深入材料对照](#第-10-章-概念--深入材料对照)
- [第 11 章 学习路径：什么时候读本文](#第-11-章-学习路径什么时候读本文)
- [附录：一页速查](#附录一页速查)

---

## 第 1 章 一张总图：从模型到异构设备

把仓库里所有材料压成**一张栈图**：最上面的「应用 / 框架」是**输入**（模型和并行策略从这里来），它下面才是**五层编译与执行栈**。每一层有自己的问题、自己的代表项目；**同一概念在不同层换名字，语义往往同构**。

> 后文说「五层栈」，指的就是下图中框架层以下的五层：**交换图 IR → 中端 → 执行规划 → 后端 IR/机器码 → 运行时**。  
> （不要和 [`paper-notes/01`](./paper-notes/01-efficient-training-distributed-infra.md) 里分布式训练的「六层结构」混淆——那是另一张图，讲的是集群侧分层。）

```
┌─ 应用 / 框架 ─────────────────────────────────────────────────┐
│  PyTorch / JAX / TensorFlow 模型；训练并行策略（DP/TP/PP…）     │
│  代表材料：01 分布式综述 · FlashAttention · PagedAttention      │
└────────────────────────────┬──────────────────────────────────┘
                             │ export / trace / 前端导入
                             ▼
┌─ 可交换图 IR（交换格式）──────────────────────────────────────┐
│  ONNX Graph · Torch ExportedProgram / Edge Dialect            │
│  问题：跨框架可移植；opset/版本；shape 推断                     │
│  代表材料：onnx-learning-guide · executorch-learning-guide    │
└────────────────────────────┬──────────────────────────────────┘
                             │ 图级优化 / 委托分区
                             ▼
┌─ 编译器中端（领域 IR + 调度）─────────────────────────────────┐
│  Relay/Relax / TOSA / StableHLO / linalg-on-tensors           │
│  融合 · layout · tiling · 代价模型 · 搜索                      │
│  代表材料：tvm-learning-guide · mlir-learning-guide · Halide  │
└────────────────────────────┬──────────────────────────────────┘
                             │ 渐进 lowering / Dialect Conversion
                             ▼
┌─ 执行规划 IR（何时算、何时搬、在哪跑）────────────────────────┐
│  IREE: flow（dispatch）→ stream（时间线）→ hal（设备对象）     │
│  或：ORT EP / ExecuTorch LoweredBackendModule                 │
│  代表材料：iree-learning-guide · onnx EP 章 · executorch 委托  │
└────────────────────────────┬──────────────────────────────────┘
                             │ codegen
                             ▼
┌─ 后端 IR → 机器码 ────────────────────────────────────────────┐
│  LLVM IR → SelectionDAG/MIR → 汇编                             │
│  NVPTX → PTX → SASS；fatbin / ExecutableVariant 多变体         │
│  代表材料：llvm-learning-guide · cuda-fatbin-learning-guide   │
└────────────────────────────┬──────────────────────────────────┘
                             │ 加载 / 派发
                             ▼
┌─ 运行时 ──────────────────────────────────────────────────────┐
│  HAL device/buffer/queue/fence · PackedFunc · EP 派发         │
│  集合通信 channel · KV Cache 分页 · 弹性迁移（开放问题）        │
└───────────────────────────────────────────────────────────────┘
```

**三个学习目标如何落在这张图上**：

| 学习目标（根 README §0） | 主要落在哪几层 | 关键横切概念（本文要补的） |
|--------------------------|----------------|---------------------------|
| ① 能表达分布式 | 应用层 + 图 IR +（未来）分片属性进编译器中端 | SPMD、sharding annotation、集合通信原语、状态表示 |
| ② 能适配异构后端 | 中端 → 执行规划 → 后端 IR → 运行时 | 委托/分区、layout、地址空间、多变体、HAL 抽象机 |
| ③ 能在运行时做决策 | 运行时 + 动态 shape / 分页 / 迁移 | AOT vs JIT、fence/timeline、配置组合、状态迁移粒度 |

> **读各专题文档之前，先记住一句话**：  
> **图决定「切成哪些算子」；调度决定「每个算子怎么访存/并行」；执行规划决定「何时提交、等谁」；后端决定「变成哪条机器指令」；运行时决定「在哪块硬件上真正发生」。**  
> 这正是五层各自负责的五个问题——混在一起，是学 AI 编译最常见的晕眩来源。

---

## 第 2 章 概念索引：按学习目标分组

下面每条都会在后文展开。这里先当目录用。

### 2.1 几乎所有编译栈共用（必补）

| # | 概念 | 一句话 | 深入见 |
|---|------|--------|--------|
| 1 | **计算图 / DFG** | 节点是算子，边是张量依赖 | §3.1 |
| 2 | **算子（op）语义分类** | injective / reduction / … 决定能不能融合 | §3.2 |
| 3 | **算子融合（fusion）** | 合并算子以减少中间结果写回 | §3.2 |
| 4 | **子图划分 / 委托（partition / delegate）** | 决定哪些节点交给哪个后端 | §3.3 |
| 5 | **算法 vs 调度（algorithm / schedule）** | 「算什么」与「怎么算」分离 | §4.1 |
| 6 | **Tiling / 循环变换** | 把迭代空间切块以适配缓存与并行 | §4.2 |
| 7 | **Roofline / 算力–带宽瓶颈** | 判断优化该砍 FLOPs 还是砍访存 | §4.3 |
| 8 | **Tensor vs Buffer（值语义 vs 存储）** | 不可变 SSA 值 vs 可原地更新的内存 | §5.1 |
| 9 | **Layout** | 同一逻辑张量在内存中的排布 | §5.2 |
| 10 | **量化（quantization）基础** | 低精度表示 + scale/zero-point | §5.3 |
| 11 | **地址空间 / 内存层次** | global/shared/local；HBM/SRAM/寄存器 | §5.4 |
| 12 | **SSA + CFG** | 单赋值 + 显式控制流（一切 IR 的底座） | §6.1 |
| 13 | **多层 IR / 渐进 lowering** | 每层只固化一类决定，避免过早丢失信息 | §6.2 |
| 14 | **Pass / 分析 vs 变换** | 编译器优化的组织单位 | §6.3 |
| 15 | **代价模型（cost model）** | 用估计代替穷举实测来做决策 | §6.4 |

### 2.2 对接异构后端时必补

| # | 概念 | 一句话 | 深入见 |
|---|------|--------|--------|
| 16 | **Kernel / Launch / Grid** | 设备上可执行的计算单元与启动配置 | §7.1 |
| 17 | **虚拟 ISA vs 真实 ISA**（PTX vs SASS） | 可迁移的中间码 vs 芯片原生码 | §7.2 |
| 18 | **多变体打包（fat binary / variant）** | 一份产物带多个目标实现，运行时选 | §7.3 |
| 19 | **设备抽象机**（device/buffer/queue） | 编译器眼中的「硬件能力接口」 | §8.1 |
| 20 | **异步执行与同步原语** | stream / event / fence / timeline semaphore | §8.2 |
| 21 | **AOT vs JIT** | 编译期定死 vs 运行期再特化 | §8.3 |

### 2.3 对接分布式与大模型负载时必补

| # | 概念 | 一句话 | 深入见 |
|---|------|--------|--------|
| 22 | **并行策略四要素** | 切什么 / 通什么 / 放哪 / 代价 | §9.1 |
| 23 | **SPMD + 分片标注** | 一份程序 + 每张量如何切到 mesh | §9.2 |
| 24 | **集合通信原语** | AllReduce / AllGather / ReduceScatter / AllToAll | §9.3 |
| 25 | **IO 感知算法**（FlashAttention 方法论） | 优化目标从 FLOPs 转向 HBM 访问次数 | §9.4 |
| 26 | **分页状态 / 块化迁移**（PagedAttention 启示） | 大状态按块管理才能低成本搬 | §9.5 |

---

## 第 3 章 图级抽象：计算图、算子、融合、划分

### 3.1 计算图（Computational Graph / DFG）

**是什么**：深度学习模型在编译器里的第一公民表示——**有向（通常无环）图**，节点是算子（MatMul、Softmax、LayerNorm…），边是张量（激活、权重、梯度）。

**为什么重要**：

- 图上的边天然表达**数据依赖**，这是融合、并行、内存规划、子图划分的共同输入。
- 与通用程序 CFG 不同：DL 图的节点粒度是「一次张量计算」，shape 常常（训练配置固定后）**静态可知**，因而能做更激进的编译期决策（静态内存规划、常量折叠）。

**在各项目中的名字**：

| 项目 | 叫法 |
|------|------|
| ONNX | `GraphProto` + `NodeProto` |
| PyTorch / ExecuTorch | `ExportedProgram` / FX graph / Edge Dialect |
| TVM | Relay / Relax 函数图 |
| IREE 前端 | StableHLO / TOSA / Torch 导入后的 MLIR 模块 |
| ORT | 会话图 + EP 分区后的子图 |

**必须建立的直觉**：图优化改的是**拓扑与节点粒度**；它不直接生成机器码。机器码是更下层的事。

### 3.2 算子语义分类与融合（Fusion）

**是什么**：把多个图节点合并成一个更大的可执行单元（通常对应一个 kernel），使中间张量**不必写回全局内存再读出**。

**经典四类**（TVM 的分类，已成社区共同语言）：

| 类别 | 含义 | 融合规则直觉 |
|------|------|--------------|
| **injective** | 输出每个元素只依赖输入对应位置（如 `add`、`relu`） | injective 彼此可融；可被 reduction / complex 的输入或输出侧吸收 |
| **reduction** | 沿某维聚合（如 `sum`、`softmax` 的归约部分） | 可融合输入侧的 injective |
| **complex-out-fusable** | 计算重，但输出可接 elementwise（如 `conv2d`） | 可融合输出侧 elementwise（conv+bias+relu） |
| **opaque** | 难以分析（如 `sort`） | 通常不融 |

**为什么这是「子图划分」的原型**：融合决策 =「这些节点要不要落在**同一个**可执行单元里」。TVM 用规则自动做；ORT EP / ExecuTorch Partitioner / IREE `flow.dispatch` 是同一问题在「多后端」场景下的推广——只不过边界从「融不融」变成了「给不给这个后端」。

详见 [`tvm-learning-guide.md`](./tvm-learning-guide.md) §2.2。

### 3.3 子图划分与委托（Partition / Delegate）

**是什么**：在一张完整计算图上，标出「由后端 A 执行」的连通子图，其余留给默认运行时（CPU EP、PyTorch 解释路径等）。边界处插入**数据交接**（可能伴随 layout/dtype/设备转换）。

**三种工业答案（同构对照）**：

| 系统 | 机制 | 谁声明能力 | 编译结果 |
|------|------|------------|----------|
| **ONNX Runtime EP** | `GetCapability` → `Compile` | EP | 分区后的可执行子图 |
| **ExecuTorch** | `Partitioner` 打 `delegation_tag` → `to_backend` | Partitioner + Backend | `LoweredBackendModule` + blob |
| **IREE** | 编译期形成 `flow.dispatch` / `hal.executable` | 编译流水线 + target | `.vmfb` 内嵌多 variant |

**分区边界的四种代价**（研究问题①②的物理根源）：

1. **数据拷贝**（host↔device 或 device↔device）  
2. **Layout / dtype 转换**  
3. **内存空间切换**（如 global ↔ shared 的管理边界）  
4. **同步点**（必须等上一子图完成才能开下一子图）

边界越多，这四项叠加越狠——所以「能划多大就划多大」和「划给最快的后端」经常冲突，需要代价模型。

详见 [`onnx-learning-guide.md`](./onnx-learning-guide.md) §7、[`executorch-learning-guide.md`](./executorch-learning-guide.md) §4–5、[`iree-learning-guide.md`](./iree-learning-guide.md) 第 2 章。

### 3.4 四条工业栈怎么选（学完各专题后回看这张表）

仓库里的四份「图级 / 部署级」专题文档讲的是**同一个问题的四种工业答案**。它们不是替代关系，选型取决于**你把决策权交给谁**：

| 栈 | 划分/优化的决策者 | 产物与运行时 | 适合的场景 | 对算力网目标缺什么 |
|----|------------------|--------------|-----------|-------------------|
| **ONNX Runtime EP**<br>[`onnx`](./onnx-learning-guide.md) · 动手 [`../onnx-delegate-lab/`](../onnx-delegate-lab/) | 各 EP 自己声明能吃什么（`GetCapability`），运行时按贪心优先级派发 | 单一 `.onnx` + 多 EP 插件，进程内派发 | 已有 ONNX 模型、要快速接一个新加速器 | 划分是贪心的，不做「算得快 vs 边界贵」的全局折中 |
| **ExecuTorch 委托**<br>[`executorch`](./executorch-learning-guide.md) · 同上 lab | 你写的 `Partitioner`（编译期打 tag） | `.pte` + delegate blob，极薄运行时 | 端侧 / 受限设备，要求确定性与小体积 | 不提供分布式调度，也不管跨节点 |
| **IREE**<br>[`iree`](./iree-learning-guide.md) | 编译流水线自己形成 `flow.dispatch`，并把时序与设备归属固化进 stream/hal | `.vmfb`（内嵌多 variant）+ HAL 运行时 | **多后端统一抽象**，要控制「何时搬、等谁、在哪跑」 | 单进程多设备为主；跨节点编排、容错、并行策略要自己补 |
| **TVM**<br>[`tvm`](./tvm-learning-guide.md) · 动手 [`../tvm-fatbin-lab/`](../tvm-fatbin-lab/) | 融合规则 + 搜索（AutoTVM/MetaSchedule）决定单个 kernel 怎么算得快 | 编译好的算子库 + PackedFunc 运行时 | 单算子/子图**性能压榨**与自动调优 | 层次比前三者低一层，不解决设备抽象与执行规划 |

**读法**：前三行是**「谁来划分」**的三种答案（同一决策，不同归属）；第四行是**「划完之后怎么算得快」**，与前三者正交，可以叠加使用。

**对本仓库的取舍**（与根 [`README.md`](../README.md) §0 优先级一致）：主线选 **IREE**，因为只有它把「执行规划」显式建成 IR 层；ONNX/ExecuTorch 提供**划分接口的工业对照**，用来理解研究问题①②；TVM 提供**代价模型与搜索**的方法论，用来回应研究问题⑥。

---

## 第 4 章 调度与局部性：算法/调度分离、tiling、Roofline

### 4.1 算法与调度分离（Algorithm / Schedule）

**是什么**（Halide 贡献的核心思想，TVM TE 直接继承）：

- **算法（algorithm / compute）**：用数学定义「输出每个元素怎么从输入算出来」，**不规定**循环顺序、并行、缓存。  
- **调度（schedule）**：在算法之上叠加变换（tile、reorder、cache_write…），生成具体循环与内存访问。

**为什么重要**：同一算法可以对应成千上万种实现；把二者耦合在手写 CUDA 里，就无法系统搜索，也无法跨硬件复用「算什么」的定义。

**在栈中的位置**：这是**算子级**优化的语言，不是图级融合。图级决定「这是一个 fused op」；调度决定「这个 fused op 内部循环长什么样」。

详见 [`paper-notes/04-halide.md`](./paper-notes/04-halide.md)、[`tvm-learning-guide.md`](./tvm-learning-guide.md) §3。

### 4.2 Tiling 与循环变换族

**Tiling（分块）**：把迭代空间切成小块，使一块数据在被踢出快速存储器之前被充分复用。

常见原语（名字来自 Halide/TVM，思想通用）：

| 原语 | 作用 |
|------|------|
| `split` / `tile` | 把一个循环变成两层（外层块、内层块内） |
| `reorder` | 改变循环嵌套顺序（决定遍历与局部性） |
| `fuse` | 合并相邻循环（便于绑定到线程维） |
| `parallel` / `bind` | 映射到 CPU 线程或 GPU block/thread |
| `vectorize` / `unroll` | SIMD 与展开 |
| `cache_read` / `cache_write` | 在更快存储层建立暂存缓冲 |
| `compute_at` | 把生产者计算嵌进消费者循环的某层（计算粒度） |

**权衡三角**（Halide 反复强调）：**局部性 ↔ 并行度 ↔ 冗余计算（+ 内存占用）**。拉高一个往往压低另一个——没有免费午餐，只有搜索或启发式。

### 4.3 Roofline：先问瓶颈在哪

**是什么**：把程序钉在「算力上限」与「带宽上限」两条线构成的屋顶上——

- 算术强度（Arithmetic Intensity）= FLOPs / Bytes  
- 强度低 → **带宽瓶颈**（优化应减少访存、提高复用、融合）  
- 强度高 → **算力瓶颈**（优化应向量化、用 Tensor Core、减少指令）

**为什么 AI 编译必提**：现代 GPU 上 Transformer 注意力等内核经常是**访存墙**，不是算力墙。FlashAttention 的方法论本质就是：用 tiling + 融合 + 重计算，把 HBM 访问次数打下来。不建立 Roofline 直觉，会误把「多算一点 FLOPs」当成优化。

详见 [`paper-notes/08-flash-attention.md`](./paper-notes/08-flash-attention.md)。

---

## 第 5 章 数据形态：tensor / buffer / layout / 量化 / 地址空间

### 5.1 Tensor（值）vs Buffer（存储）

| | Tensor（值语义） | Buffer / MemRef（存储语义） |
|--|------------------|-----------------------------|
| 可变性 | SSA 值，不可变；「更新」= 产生新值 | 内存位置，可被 store 原地改 |
| 别名 | 无别名（SSA） | 可能别名，分析困难 |
| 适合的优化 | 融合、tiling、shape 推导 | 内存复用、向量 load/store、与运行时对接 |
| 典型 IR | `tensor<…>`、Relay Tensor | `memref<…>`、HAL buffer |

**Bufferization**：把 tensor 程序变成 buffer 程序的过程。One-Shot Bufferize 的目标是**尽量原地复用**（少分配、少拷贝），依赖 destination-passing style（结果写进已有的 `outs` 缓冲）。

> 经验法则：**图级与高维变换尽量留在 tensor 世界；靠近硬件再 bufferize。** 过早 bufferize = 过早引入别名，后面融合/变换变难。

详见 [`mlir-learning-guide.md`](./mlir-learning-guide.md) §8。

### 5.2 Layout（数据布局）

**是什么**：逻辑上的 `[N,C,H,W]` 在内存里按什么顺序、是否分块（blocking）、是否与向量宽度对齐。

常见例子：

- `NCHW` vs `NHWC`（通道维位置影响卷积与向量化）  
- 更细的 **blocked layout**（如 `NCHW4c`）以适配 SIMD / Tensor Core tile  
- GPU 上还有 **swizzle** 等与 bank conflict 相关的排布

**传播**：producer 偏好与 consumer 偏好不一致时，插入 `layout_transform`；变换本身有代价，应尽量把冲突推到图的边缘或与计算融合。

这直接对应研究问题②「跨后端 layout 兼容」。

### 5.3 量化基础（够用即可）

**是什么**：用低比特整数（INT8/INT4…）近似浮点，配合 **scale** 与可选 **zero-point**：

\[
x_{float} \approx scale \times (x_{quant} - zero\_point)
\]

必知区分：

- **per-tensor** vs **per-channel**（scale 粒度）  
- **对称** vs **非对称**  
- 图中的 `QuantizeLinear` / `DequantizeLinear`（或等价节点）会在分区边界变成**额外算子**——又是边界代价。

本文不展开数值技巧；知道「量化是图上的一等公民，且制造跨后端摩擦」即可。Glow 的 profile-guided 量化、Campo 对 cast 开销的警告，都是同一主题。

### 5.4 地址空间与内存层次

| 层级 | 典型容量 | 带宽 | 谁管理 |
|------|----------|------|--------|
| 寄存器 | 极小 | 最高 | 编译器分配 |
| 片上 SRAM / shared | KB～MB | 很高 | 显式（GPU）或缓存（CPU） |
| HBM / 设备显存 | GB～十 GB | 高 | 运行时分配器 |
| 主机 DRAM | 更大 | PCIe/NVLink 受限 | OS / 运行时 |
| 网络另一端 | — | 更慢 | 集合通信 / RDMA |

**地址空间（address space）** 在 IR 里常被编码为指针属性（LLVM `addrspace(N)`、MLIR memref memory space、NVPTX generic/global/shared）。**错误的地址空间转换昂贵且易错**——GPU 路径上这是高频坑。

---

## 第 6 章 中间表示与变换：SSA、多层 IR、Pass、渐进 lowering

### 6.1 SSA 与 CFG（所有现代编译 IR 的底座）

- **SSA（Static Single Assignment）**：每个值只赋值一次；汇合用 `phi`（LLVM）或 **block argument**（MLIR）。  
- **CFG**：基本块 + 终结符显式列出后继。  
- **效果**：数据流分析变便宜；「寄存器无别名」成立。

你已在 [`llvm-hello-compile`](../llvm-hello-compile/) 里用 mem2reg 亲眼见过；MLIR 把同一思想推广到可嵌套 Region。

### 6.2 多层 IR 与渐进 Lowering

**单层 IR 的病**：过早降到 LLVM IR → 「这是一次卷积」永久消失 → 无法再做融合/tiling。

**渐进 lowering**：每层只固化一类决定——

```
领域算子语义  →  结构化张量计算（linalg）→ 循环（scf）→ 向量 → LLVM/PTX
     ↑ 可融合/可推导 shape        ↑ 可 tile/fuse         ↑ 可向量化
```

IREE 的 `linalg → flow → stream → hal` 是同一哲学在「执行规划」上的实例：  
flow 固化「怎么切成 dispatch」；stream 固化「时序与资源寿命」；hal 固化「用哪块设备对象」。

详见 [`mlir-learning-guide.md`](./mlir-learning-guide.md) §1、[`iree-learning-guide.md`](./iree-learning-guide.md) §2。

### 6.3 Pass：分析与变换

| 类型 | 行为 | 例子 |
|------|------|------|
| Analysis | 只读，算性质，可缓存 | 支配树、别名分析、SCEV |
| Transform | 改 IR，并声明保留了哪些分析 | 融合、DCE、Dialect Conversion |

组织方式：PassManager 按 Module / Function / Loop（LLVM）或任意嵌套（MLIR）调度。  
**写 lowering 时最重要的工程纪律**：改了什么就要正确 invalidate；否则下游用过期分析会静默出错。

### 6.4 代价模型（Cost Model）

**是什么**：用可计算的分数估计「某种划分 / 某种 schedule / 是否向量化」的好坏，避免每次都上真机穷举。

出现位置：

- TVM AutoTVM / MetaSchedule 的 ML cost model  
- LLVM TTI（TargetTransformInfo）  
- 你在 `mlir-toy-dialect` 里写的 `ToyCostOpInterface`（教学版）  
- 图划分时的「计算时间 vs 传输时间」估计  

**没有代价模型，自动搜索与自动分区都会退化成拍脑袋。**

---

## 第 7 章 代码生成与变体：kernel、ISA 层级、多目标打包

### 7.1 Kernel 与启动配置

在 GPU 类设备上：

- **Kernel**：设备上可执行的一段程序（对应 IREE 的 `hal.executable` export、CUDA `__global__`）。  
- **Launch configuration**：grid / block（或 workgroup）维度——决定并行度与 shared memory 用量。  
- **ABI**：kernel 如何拿到绑定的 buffer、常量、workgroup id（IREE 的 `hal.interface.*` 五件套是这一层的标准化）。

CPU 路径上「kernel」常退化成「一个函数」；概念仍在：可执行体 + 调用约定。

### 7.2 虚拟 ISA vs 真实 ISA

以 NVIDIA 为例（思想可推广）：

| | 虚拟（PTX / `compute_XX`） | 真实（SASS / `sm_XX`） |
|--|---------------------------|------------------------|
| 可移植性 | 同一 PTX 可 JIT 到更新架构 | 绑定具体芯片 |
| 加载成本 | 首次需 JIT | 直接加载 |
| 性能可控性 | 依赖驱动 JIT 质量 | 编译期完全可见 |

AMD 有自己的中间表示与机器码层级；CPU 则是 LLVM IR → 机器码。  
**共同点**：编译栈几乎总有「可移植中间码」和「最终机器码」两层。

详见 [`cuda-fatbin-learning-guide.md`](./cuda-fatbin-learning-guide.md)、[`llvm-learning-guide.md`](./llvm-learning-guide.md) §5。

### 7.3 多变体打包（Fat Binary / ExecutableVariant）

**问题**：设备微架构太多，预编译「所有配置 × 所有架构」会爆炸（研究问题⑥）。

**模式**：一份逻辑可执行体，内嵌**多个实现变体**；运行时按 capability / condition 选第一个可用者。

| 实现 | 载体 |
|------|------|
| CUDA fatbin | 一个 cubin/fatbin 含多份 PTX/SASS |
| IREE | `hal.executable` 内多个 `variant` |
| 工程折中 | 热配置预编译 + 冷配置 JIT |

这是「配置组合爆炸」最务实的工程答案之一——先理解模式，再谈搜索与缓存。

---

## 第 8 章 运行时抽象机：设备、队列、同步、AOT/JIT

### 8.1 设备抽象机（最小词汇表）

不管叫 HAL、EP 还是 CUDA Runtime，编译器眼中的硬件大致长这样：

```
Driver ──▶ Device ──▶ Allocator ──▶ Buffer (+ BufferView: shape/layout/dtype)
                │
                ├── CommandBuffer / Stream：录制要做的工作
                ├── Executable：设备代码
                └── Synchronize：semaphore / event / fence
```

**为什么要抽象**：编译器若直接依赖 `cudaMemcpyAsync` / `vkCmdDispatch`，每个后端重写整条流水线；有了抽象，**调度决策可以在编译期写成对抽象对象的操作序列**（IREE 的核心取向）。

详见 [`iree-learning-guide.md`](./iree-learning-guide.md) 第 4 章。

### 8.2 异步与同步

现代加速器默认**异步**：host 提交工作后立即返回；完成与否靠同步原语。

| 原语族 | 直觉 | 局限 |
|--------|------|------|
| CUDA Event | 「某个点已经过去」 | 难表达复杂多队列依赖与 host↔device 统一时间线 |
| Timeline Semaphore（IREE） | 64 位单调计数器，wait/signal 任意顺序 | 需要运行时补齐驱动缺口 |
| Fence | 常作为「一批 wait/signal 的宿主侧句柄」 | — |

**编译期**若不能显式表达依赖，就无法把「计算–通信重叠」「stream-ordered allocation」做进产物。这就是为什么 IREE 把 timeline 视为灵魂。

### 8.3 AOT vs JIT

| | AOT | JIT |
|--|-----|-----|
| 何时编译 | 部署前 | 运行中（首次或 shape 变化时） |
| 优点 | 启动快、可复现、可裁剪运行时 | 可按真实 shape/设备特化 |
| 缺点 | 变体爆炸或保守代码 | 首次延迟、工程复杂 |
| 典型 | IREE `.vmfb`、ExecuTorch `.pte`、fatbin 中的 SASS | PTX JIT、部分 TVM/ORT 路径、动态 shape 特化 |

算力网场景通常需要**混合**：稳态路径 AOT，变化路径 JIT 或预编译变体池。

---

## 第 9 章 分布式与编译的接缝：分片如何变成 IR

这是根 README「两条学习线交汇」的展开。没有这一章，分布式综述和 MLIR/IREE 会永远是两张皮。

### 9.1 并行策略的四要素（复习锚点）

对每一种并行（DP / TP / PP / SP / EP），始终问：

1. **切什么维度**（batch / 隐层 / 层 / 序列 / 专家）  
2. **通信什么原语**  
3. **放在哪个带宽域**（NVLink 域内 vs 跨节点）  
4. **主要代价**（算力气泡 / 通信体积 / 显存）

详见 [`paper-notes/01-efficient-training-distributed-infra.md`](./paper-notes/01-efficient-training-distributed-infra.md)。

### 9.2 SPMD 与分片标注（Sharding Annotation）

**SPMD（Single Program Multiple Data）**：所有设备跑**同一份程序**，差异由「每张量如何映射到 device mesh」描述。

**分片标注**：在 IR / 图上给张量挂属性，例如：

- Mesh-TensorFlow：显式 mesh + 命名维度  
- GSPMD：对中间张量推断注解  
- OneFlow SBP：Split / Broadcast / Partial  

**与编译器的关系**：

```
并行策略  ──编码为──▶  IR 上的 sharding / layout / device 属性
                ──Pass 推导──▶  插入通信算子、选择局部计算
                ──lowering──▶  每设备可执行体 + channel 集合通信
```

这正是「可编译的分布式」：策略不再是训练脚本里的 if/else，而是**可被 Pass 变换的 IR**。MLIR 的 Attribute / Interface 是挂这些注解的自然位置；IREE 的 `channel` 是通信在 HAL 层的落点。

### 9.3 集合通信原语（最小集）

| 原语 | 直觉 | 典型用途 |
|------|------|----------|
| AllReduce | 大家的值归约后再广播 | DP 梯度同步 |
| AllGather | 每卡拼出完整张量 | TP 的拼接 |
| ReduceScatter | 归约并按切分散开 | ZeRO / 部分 TP |
| AllToAll | 按分区重新洗牌 | MoE / 某些并行 |

编译器若能把 collective 与 compute 放进**同一条命令缓冲的依赖图**，才能谈 overlap。这是 IREE channel 设计与 NCCL 在概念上的对齐点。

### 9.4 IO 感知算法（负载侧方法论）

FlashAttention 教的不是又一个 kernel 细节，而是：

> 当 Roofline 显示带宽瓶颈时，**正确的复杂度度量是 HBM 访问次数**；允许重计算换存储；用 tiling 把工作集塞进 SRAM。

这套方法论直接指导编译器该把哪些融合/tiling 决策视为一等公民。

### 9.5 分页状态与低成本迁移

PagedAttention 教的是：

> 巨大的运行时状态（KV Cache）应按**块**分配与索引；块化之后，抢占、前缀共享、跨设备搬迁的粒度都变小。

映射到研究问题④「运行状态变化后低成本切换后端」：没有块化/并行无关的状态表示，迁移只能整包拷贝——太贵。Universal Checkpointing 等方向是训练侧的同类问题。

---

## 第 10 章 概念 → 深入材料对照

| 概念 | 先读本文 | 再深入 |
|------|----------|--------|
| 计算图 / opset / shape | §3.1 | [`onnx-learning-guide.md`](./onnx-learning-guide.md) |
| 融合四类 | §3.2 | [`tvm-learning-guide.md`](./tvm-learning-guide.md) §2.2 · [`paper-notes/05-tvm.md`](./paper-notes/05-tvm.md) |
| 委托 / 分区边界 | §3.3 | [`executorch-learning-guide.md`](./executorch-learning-guide.md) · onnx EP 章 · 动手 [`../onnx-delegate-lab/`](../onnx-delegate-lab/) |
| 四条工业栈怎么选 | §3.4 | [`onnx`](./onnx-learning-guide.md) · [`executorch`](./executorch-learning-guide.md) · [`iree`](./iree-learning-guide.md) · [`tvm`](./tvm-learning-guide.md) |
| 算法/调度、tiling | §4 | [`paper-notes/04-halide.md`](./paper-notes/04-halide.md) · tvm §3 |
| Roofline / IO 感知 | §4.3 · §9.4 | [`paper-notes/08-flash-attention.md`](./paper-notes/08-flash-attention.md) |
| tensor/buffer/layout | §5 | [`mlir-learning-guide.md`](./mlir-learning-guide.md) §8 · tvm layout 节 |
| SSA / Pass / CodeGen | §6 · §7 | [`llvm-learning-guide.md`](./llvm-learning-guide.md) · [`llvm-hello-compile`](../llvm-hello-compile/) |
| 渐进 lowering / Conversion | §6.2 | [`mlir-learning-guide.md`](./mlir-learning-guide.md) · [`mlir-toy-dialect`](../mlir-toy-dialect/) |
| HAL / fence / variant | §7.3 · §8 | [`iree-learning-guide.md`](./iree-learning-guide.md) |
| fatbin / compute vs sm | §7.2–7.3 | [`cuda-fatbin-learning-guide.md`](./cuda-fatbin-learning-guide.md) |
| 并行 / 分片 / 通信 | §9 | [`paper-notes/01-…`](./paper-notes/01-efficient-training-distributed-infra.md) |
| KV 分页 / 迁移直觉 | §9.5 | [`paper-notes/09-paged-attention-vllm.md`](./paper-notes/09-paged-attention-vllm.md) |
| 少量原语多后端 | （对照） | [`paper-notes/06-glow.md`](./paper-notes/06-glow.md) vs TVM |

---

## 第 11 章 学习路径：什么时候读本文

### 11.1 推荐节奏

| 时机 | 读哪些章 | 目的 |
|------|----------|------|
| **开局（第 0 周）** | §1 总图 + §2 索引 + 附录速查 | 知道整个栈有几层 |
| **学分布式综述同时** | §9 全部 | 把并行策略映射到「将来要进 IR 的属性」 |
| **进 MLIR / toy 之前** | §5.1 · §6 | tensor/buffer、渐进 lowering、Pass |
| **进 IREE 之前** | §3.3 · §7.1 · §8 | 划分、kernel、设备抽象机与同步 |
| **进 TVM / Halide 之前** | §4 | 算法/调度、tiling、Roofline |
| **进 ONNX / ExecuTorch 之前** | §3 全部 | 图、融合、委托边界代价 |
| **四条栈都学完之后** | §3.4 | 回看选型：谁把决策权交给了谁 |
| **碰 CUDA / 多后端打包时** | §7.2–7.3 | 虚拟/真实 ISA、变体 |
| **思考六个研究问题时** | §3.3 · §5.2–5.3 · §8.3 · §9 | 用同一套词汇写问题分析 |

### 11.2 必学 vs 可推迟

**现在就要建立直觉的**（否则专题文档读不进去）：

1. 五层栈总图（§1）  
2. 融合 vs 委托分区的同构（§3.2–3.3）  
3. 算法/调度分离 + Roofline（§4）  
4. tensor vs buffer + layout（§5.1–5.2）  
5. 渐进 lowering（§6.2）  
6. 设备抽象机 + 异步同步（§8）  
7. 分片标注如何成为 IR 属性（§9.2）  

**可推迟**：

- 具体集合通信算法（Ring/Tree）实现细节  
- 量化数值技巧（Hadamard、随机舍入等）  
- 某张 GPU 的 SASS 指令  
- 自动微分的正/反向模式实现  
- polyhedral 依赖分析公式  

### 11.3 自检（能讲清就算过关）

- [ ] 从 PyTorch 模型到 GPU 上跑完一次 matmul，中间经过哪五类表示？每层「固化什么决定」？  
- [ ] 为什么说「融合」和「EP 分区」是同一类决策？边界上贵在哪里？  
- [ ] 为什么不能把模型一步降成 LLVM IR？用「信息不可逆丢失」举一例。  
- [ ] tensor 世界和 buffer 世界各适合做什么优化？  
- [ ] 用四要素讲清 Tensor Parallel：切什么、通什么、放哪、代价。  
- [ ] fatbin 与 IREE ExecutableVariant 解决的是同一个什么问题？  
- [ ] 同一个模型要在算力网上跑，ORT EP / ExecuTorch / IREE / TVM 四条栈各自把哪个决策权交给了谁？（§3.4）

### 11.4 五个最小动作（把词汇变成手感）

本文只讲概念，动手在各专题文档里。但下面几件事**任何阶段都能做，且直接验证本文的核心同构**：

| 动作 | 命令 / 位置 | 验证了本文哪一节 |
|------|-------------|------------------|
| 看一次「同一段代码在不同层的样子」 | `cd llvm-hello-compile && bash scripts/run.sh`，对比 `02_sum_O0.ll` 与 `03a_sum_mem2reg.ll` | §6.1 SSA、§6.3 Pass |
| 看一次「渐进 lowering 丢了什么」 | `cd mlir-toy-dialect && bash scripts/all.sh`，对比 `x*4` 在 toy 层与 low 层的命运 | §6.2 渐进 lowering |
| 看一次「同一算法两种 schedule」 | `cd tvm-fatbin-lab && bash scripts/run_tvm.sh`，对比 `out/tvm/01_*.lower.txt` | §4 算法/调度 |
| 看一次「谁来划分、边界在哪」 | `cd onnx-delegate-lab && bash scripts/run.sh`，读 `out/ANALYSIS.md` | §3.3–3.4 委托/选型 |
| 看一次「一份产物里有几个变体」 | `cd tvm-fatbin-lab && bash scripts/run_fatbin.sh`（或教材第 8 章手写 `cuobjdump`） | §7.3 多变体打包 |

做完这些，本文 80% 的抽象都有了对应的实物。建议顺序：前两件随时做；后三件跟阶段 4→6（schedule → 划分 → fatbin）。

---

## 附录：一页速查

```
【五层栈】(输入:框架模型) → 交换图IR → 中端(融合/调度) → 执行规划(dispatch/stream/hal|EP) → 后端IR/机器码 → 运行时

【三个目标落点】
  表达分布式：分片标注进 IR + 集合通信进执行规划
  适配异构：  委托分区 + layout/地址空间 + 多变体 + HAL 抽象机
  运行时决策：AOT/JIT 混合 + fence/timeline + 块化状态迁移

【同构对照】
  融合规则     ≈  单后端上的「子图划分」
  ORT GetCapability / ET Partitioner / IREE flow.dispatch  ≈ 多后端划分
  CUDA fatbin  ≈  IREE ExecutableVariant  ≈  热配置预编译+冷配置JIT
  Halide schedule  ≈  TVM TE schedule  ≈  （思想）MLIR transform/tiling
  SSA+Pass     ≈  LLVM / MLIR 共用底座；MLIR 多了 Region 嵌套与 dialect

【数据】tensor(值/SSA) → bufferize → memref/buffer(存储/可别名)
【布局】逻辑形状 ≠ 物理 layout；跨后端要付转换税
【瓶颈】先 Roofline：带宽墙 → 融合/tiling/IO感知；算力墙 → 向量化/tensorize

【同步】异步提交是默认；没有显式依赖就没有 overlap
【分布式】策略 → sharding 属性 → Pass 插通信 → 每设备 executable + channel
```

---

## 维护约定

- 新增专题学习文档时：在[第 10 章对照表](#第-10-章-概念--深入材料对照)加一行「概念 → 该文档」。  
- 根 README 若调整学习目标表述：同步改[第 1 章表格](#第-1-章-一张总图从模型到异构设备)。  
- 本文保持「横切概念」角色，**不复制**各专题文档的 API 细节。
