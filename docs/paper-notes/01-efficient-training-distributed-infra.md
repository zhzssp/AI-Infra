# Efficient Training over Distributed Infra：大模型分布式训练系统的全景地图

> **导航**：[笔记索引](README.md) · [自学枢纽](../README.md)（阶段 1） · [横切概念](../learning-guides/ai-compiler-foundations-learning-guide.md) · [总规划 §3.1](../../README.md)  
> **配套阅读**：本篇讲「分布式要表达什么」；这些要求如何变成编译器里的 IR 属性，见 [`ai-compiler-foundations.md` §9](../learning-guides/ai-compiler-foundations-learning-guide.md#第-9-章-分布式与编译的接缝分片如何变成-ir)；落到设备与集合通信见 [`iree-learning-guide.md`](../learning-guides/iree-learning-guide.md) §4.8。  
> **本篇章号提示**：这是综述体例，§3–§9 对应论文各章；**框架在 §2，最小必要集在 §12**。

> **论文元信息**
> - 标题：*Efficient Training of Large Language Models on Distributed Infrastructures: A Survey*
> - 作者机构：上海人工智能实验室、香港中文大学、复旦大学、上海交通大学、南洋理工大学、北京大学
> - 发表：arXiv:2407.20018v1 [cs.DC]，2024-07-29
> - 链接：<https://arxiv.org/abs/2407.20018>
> - 类型：**综述（survey）**，不是单一系统论文。它的价值在于给出一张"分布式训练基础设施"的完整问题地图与技术选项清单。

---

## 0. 为什么这篇论文对我们最重要

这是老师给出的"算力网上大模型分布式运行基础设施"项目的**直接选题依据**。其他八篇论文（LLVM/MLIR/Halide/TVM/Glow/IREE/FlashAttention/PagedAttention）解决的是"**单个设备上怎么把算子编译得快**"，而这篇解决的是"**成千上万个设备怎么协同把一个模型训完**"。

两者的交汇点正是我们要做的事：**在异构、跨域的算力池上，为分布式大模型运行提供统一的编译 + 运行时基础设施**。所以读这篇论文的目标不是记住 400 篇引文，而是：

1. 建立**问题坐标系**：知道分布式训练一共有哪几类瓶颈，每类瓶颈的主流解法是什么。
2. 认清**关键约束**：通信拓扑、显存墙、故障率——这三个是所有设计的硬边界。
3. 找到**我们能切入的空隙**：算力网的"异构 + 跨域 + 多租户"恰好是这篇综述里覆盖最薄的一块（§4.3 异构并行、§3.4 调度）。

---

## 1. 论文的核心命题：SER 三角

论文把 LLM 训练系统的全部挑战归结为三个互相拉扯的目标，简称 **SER**：

| 维度 | 英文 | 含义 | 论文给的现实数据 |
|------|------|------|------------------|
| 可扩展性 | **S**calability | 万卡规模下仍能保持训练正确性与模型精度 | LLaMA-3 用 16K 张 H100 训练 54 天 |
| 效率 | **E**fficiency | 用 MFU（Model FLOPs Utilization）衡量算力利用率 | LLaMA-3 在 16K GPU 上 MFU 只有 **38%~41%** |
| 可靠性 | **R**eliability | 数周至数月的训练过程中能快速检测并恢复故障 | LLaMA-3 训练 54 天中断 **466 次**；Meta OPT-175B 理想 25 天、实际 57 天，**56% 的时间在处理故障** |

这三个数字是整篇论文的"锚"：
- **MFU 只有 40%** → 说明计算、通信、内存三方面都有巨大优化空间（§5、§6、§7 存在的理由）。
- **56% 时间浪费在故障上** → 说明容错不是"锦上添花"而是一等公民（§8 存在的理由）。
- **单节点 1.5% 的日故障率，扩到 1000 GPU 就是 84.8% 的日故障率** → 这是规模化最反直觉、也最致命的结论。

### LLM 训练负载的四个特征（决定了系统怎么设计）

1. **模型架构同质化**：几乎全是 Transformer（decoder-only）。这意味着可以为**一种架构**做极致的专用优化，而不必像过去 CNN/LSTM 时代那样追求通用性。
2. **规模与时长空前**：千亿参数 + TB 级数据 + 数周训练。
3. **软件栈高度专用化**：Megatron / DeepSpeed / Alpa 这类系统承担了主要优化。
4. **训练范式转变**：从"任务专用训练"变为"预训练基座 + 对齐微调 + 周期性评测"，从数据中心视角看是**一个大作业 + 大量小作业混合**的负载。

> **对我们的启示**：第 1 点是我们做基础设施的最大利好——目标算子集是收敛的（Attention + FFN + Norm + Embedding + 集合通信），不需要支持任意计算图。第 4 点提示算力网调度器必须同时处理"长时独占大作业"和"短时高并发小作业"。

---

## 2. 整体框架：论文的六层结构

论文把整个技术空间组织成六个层次，这也是我们后续建立知识体系的骨架：

```
┌──────────────────────────────────────────────────────────────────────┐
│ §8 容错 Fault Tolerance                                              │
│   故障分析 → 异常检测（统计监控/主动验证）→ 恢复（checkpoint / 无 ckpt）│
└──────────────────────────────────────────────────────────────────────┘
        ▲                       ▲                       ▲
┌───────┴────────┐   ┌──────────┴─────────┐   ┌─────────┴──────────┐
│ §5 计算优化    │   │ §6 内存优化        │   │ §7 通信优化        │
│ 算子优化       │   │ 激活重计算         │   │ 集合通信算法       │
│ （手工/编译器）│   │ 冗余消除(ZeRO/FSDP)│   │ 通信调度（重叠）   │
│ 混合精度       │   │ 碎片整理           │   │ 网内聚合 INA       │
│                │   │ Offload(CPU/SSD)   │   │                    │
└───────┬────────┘   └──────────┬─────────┘   └─────────┬──────────┘
        │                       │                       │
┌───────┴───────────────────────┴───────────────────────┴──────────────┐
│ §4 并行策略 Parallelism Schemes                                      │
│   混合并行：数据 DP / 张量 TP / 流水 PP / 序列 SP / 专家 EP           │
│   自动并行：Alpa、GSPMD、nnScaler …                                   │
│   异构并行：异构硬件（跨型号/跨地域）、异构模型（RLHF）               │
└──────────────────────────────────────────────────────────────────────┘
        ▲
┌───────┴──────────────────────────────────────────────────────────────┐
│ §3 基础设施 Infrastructure                                           │
│   AI 加速器 │ 网络（片间/节点间/拓扑/负载均衡与拥塞控制）             │
│   存储（checkpoint / 训练数据）│ 调度（作业调度 / 资源调度）          │
└──────────────────────────────────────────────────────────────────────┘
```

**读法建议**：自下而上读一遍建立约束感（硬件能力决定了软件能做什么），再自上而下读一遍建立设计感（优化目标决定了要用哪些硬件特性）。

---

## 3. §3 基础设施：硬件与网络的硬约束

### 3.1 加速器

- **NVIDIA**：核心是 SM（Streaming Multiprocessor）阵列 + 每 SM 的 shared memory + HBM。精度支持 FP32/TF32/FP16/BF16/FP8/INT8/FP4。Hopper 引入 **Transformer Engine**（FP8 + FP16 混合自动切换）。
- **其它**：AMD MI250X（ROCm，Frontier 超算，已有 ROCm 版 FlashAttention）、Intel/Habana GAUDI2（384 卡训 GPT-3 175B）、Google TPUv4（4096 芯片，约 60% peak FLOPS）、Graphcore IPU、Cerebras CS-2（晶圆级，85 万核，每核 48KB SRAM）。

> **对我们的启示**：算力网必然是多厂商混合的。**"同一个模型要在 NVIDIA/AMD/昇腾/寒武纪上都跑起来"这件事，正是 MLIR/IREE 多后端编译要解决的问题**——这就是本仓库两条学习线（分布式 + 编译器）真正汇合的地方。

### 3.2 网络：三个层次

**(1) 片间（chip-to-chip，节点内）**——五种拓扑：

| 拓扑 | 代表 | 关键数字 |
|------|------|----------|
| 树形 | PCIe | 3.0 ≈ 1 GB/s/lane，x16 ≈ 16 GB/s；4.0 翻倍；5.0 再翻倍 |
| Cube-Mesh | NVLink-1.0（DGX-1） | 每链路 160 GB/s 双向，8 卡立方网格 |
| 交换式全连接 | NVSwitch（DGX-2） | 任意两卡 300 GB/s → v2 600 GB/s → v3 900 GB/s |
| P2P 全连接 | AMD Infinity Fabric、Intel、昇腾 | 带宽受单链路限制 |
| 2D/3D-Torus | TPUv2 16×16、TPUv3 32×32、TPUv4 4×4×4 立方 | ICI 链路 |

**(2) 节点间（node-to-node）**：RDMA 是基础。**GPUDirect-RDMA** 绕过 CPU 直接 GPU 到 GPU。两大技术路线：**InfiniBand**（EDR 100G → HDR 200G → NDR 400G，专用网络）与 **RoCE**（v1 链路层 / v2 over UDP，复用以太网，字节跳动与 Meta 都在用）。iWARP 性能不足，基本不用。

**(3) 集群拓扑**：
- 分**前端网络**（作业管理、推理、存储）与**后端网络**（训练流量），优化重点是后端。
- HPC 传统拓扑：Clos/Fat-Tree（最常用）、Dragonfly+、Torus 等。Meta 上一代 24000 GPU 集群 = 8 个 pod，核心层 7:1 收敛比。
- **训练优化拓扑**：**rail-optimized**（不同服务器上相同 index 的 GPU 挂到同一 leaf 交换机，减少流量干扰）、**rail-only**（发现 GPT/OPT-175B 训练中 **99% 的 GPU 对之间没有任何流量**，于是干脆删掉 rail 之间的连接，跨 rail 走节点内互联转发）、阿里 **HPN**（2 层双平面，51.2Tbps 单芯片交换机，单 pod 支持 15000 GPU）、HammingMesh、BiGraph。
- **可重构拓扑**：SiP-ML、TopoOpt（拓扑与并行策略联合优化）、TPUv4 的光路交换 OCS（512 芯片可在 4×4×32 与 8×8×8 之间重构）。

**(4) 负载均衡与拥塞控制**：
- LLM 训练流量特征是**极少数超大流（elephant flow）+ 周期性突发**。传统 ECMP 基于哈希，几条大流撞到同一链路就拥塞。
- 解法：LLaMA-3 让集合通信库在两个 GPU 间建 **16 条流**并用 E-ECMP 按 RoCE 头部额外字段哈希打散；packet spraying（会乱序，需要 NIC 支持）；Ethereal（贪心分配路径）；HPN（在通信库内部做负载均衡）。
- 拥塞控制：PFC 保证无损但会 head-of-line blocking；DCQCN / TIMELY / Swift / HPCC / EQDS / RoCC。多作业场景：MLTCP（按每轮已发字节数调窗口，交错不同作业的通信相位）、CASSINI（按通信模式做作业放置）、MLT（**利用"靠后层的梯度更重要、数值更大的梯度更重要"，在交换机上按梯度重要性排队/丢包**）。

### 3.3 存储

- **Checkpoint 存储**：70B 模型的 checkpoint 就有 **980 GB**。用 Meta Tectonic、字节 HDFS、Ceph 对象存储。恢复时的常用技巧：**让一个 worker 从 HDFS 读，再广播给同 DP 组的其它 worker**，避开存储带宽瓶颈。
- **训练数据存储**：LLaMA-3 训了 15T tokens ≈ 30 TB；而预处理阶段的中间数据通常是最终数据集的 **100 倍以上**（WanJuan-CC 丢掉了 99% 的原始数据），总量到几十 PB。用 Lustre/GPFS/BeeGFS 并行文件系统 + Alluxio/JuiceFS/Quiver/Fluid 做缓存与预取。
- 注意一个细节：LLM 训练中**每个 token 通常只被看一次**，所以缓存的价值主要在"预取以掩盖 I/O 延迟"，而不在"复用"。

### 3.4 调度（与算力网最相关的一节）

分两类：
- **作业调度（workload scheduling）**：三种典型能力——**异构感知**（Gavel、Gandivafair，跨不同代 GPU 分配作业）、**作业打包**（FGD、Lucid，细粒度 GPU 共享）、**自适应伸缩**（Pollux、Sia，动态调整 GPU 数与超参）。针对 LLM 的新系统：**Crius**（在集群调度层联合考虑混合并行配置与硬件亲和性）、**Hydro**（缩小成代理模型做超参搜索，并把调参作业塞进预训练的流水线气泡里）、**Acme**（刻画 LLM 开发全流程的混合负载，解耦评测调度 + 故障诊断与自动恢复）。
- **资源调度**：Cassini（网络带宽相位错峰）、HIRE（交换机内网内计算调度）、SiloD（把数据缓存与远程 I/O 当作一等资源联合分配）、Synergy（优化 CPU 核分配而非按 GPU 比例分）、EnvPipe/Zeus/Perseus（能耗——利用流水线气泡降频、调 batch size 与功率上限、求迭代时间-能耗的 Pareto 前沿）。

> **对我们的启示**：算力网基础设施本质上就是 **"异构感知 + 自适应伸缩 + 拓扑感知放置"三者的合体**。Crius 的思路（调度器与并行策略联合决策）几乎就是我们要做的事情的雏形。

---

## 4. §4 并行策略：本篇最需要吃透的一章

### 4.0 编程模型前提：SPMD vs MPMD

- **SPMD（Single Program Multiple Data）**：所有设备跑同一份程序、处理不同数据。DP、TP、SP 都是 SPMD。类 MPI 范式。
- **MPMD（Multiple Program Multiple Data）**：不同设备跑程序的不同部分。PP（流水并行）本质是 MPMD。
- 自动并行和异构并行可以混用两者。

> **这是 README 里"掌握 SPMD 编程模型"的确切含义**：SPMD 的关键是"程序统一、数据分片"，因此可以用**张量分片标注（sharding annotation）**来表达并行策略——这正是 GSPMD / OneFlow SBP / PartIR 的做法，也是编译器能自动化分布式的原因。

### 4.1 五种基本并行（混合并行 = 它们的组合）

#### (a) 数据并行 DP —— 按 batch 维切分

引入**分片因子 F**（参数被切到多少个设备上，1 ≤ F ≤ W，W 为总设备数），可统一描述三种形态：

| F 取值 | 名称 | 代表 | 显存 | 通信 |
|--------|------|------|------|------|
| F = 1 | 全复制（vanilla DP） | PyTorch-DDP、Horovod | 每卡一份完整模型 | All-Reduce 梯度；按 bucket 分桶与反向计算重叠 |
| F = W | 全分片 | **ZeRO-3**、**FSDP**、Sharded Weight Update | 每卡 1/W 参数 | 最省显存但通信量 ≈ vanilla DP 的 **1.5 倍**；用 All-Gather 反分片 + Reduce-Scatter 分片 |
| 1 < F < W | 混合分片 | **MiCS**、FSDP 的 HYBRID_SHARD | 设备组成 N×M mesh，沿 N 分片、沿 M 复制 | 折中；FSDP 用 Reduce-Scatter 替代 All-Reduce 进一步省 |

**ZeRO 三阶段的精确账本**（论文 §6.2.1，混合精度下参数量 Φ）：

| 组件 | 字节数 | ZeRO-1 后 | ZeRO-2 后 | ZeRO-3 后 |
|------|--------|-----------|-----------|-----------|
| FP16 参数 | 2Φ | 2Φ | 2Φ | 2Φ/N |
| FP16 梯度 | 2Φ | 2Φ | 2Φ/N | 2Φ/N |
| 优化器状态（FP32 参数 + momentum + variance） | 12Φ | 12Φ/N | 12Φ/N | 12Φ/N |
| **合计** | **16Φ** | 4Φ + 12Φ/N | 2Φ + 14Φ/N | **16Φ/N** |

> **必背结论**：混合精度训练下，模型状态显存 = **16Φ 字节**（Φ 为参数个数）。7B 模型 ≈ 112 GB，单卡放不下——这一个式子就解释了 ZeRO/FSDP 为什么必须存在。

#### (b) 张量并行 TP —— 层内切分（intra-layer）

- **Megatron-TP（1-D）**：MLP 与 self-attention 各有两个参数矩阵，**第一个按列切、第二个按行切**，这样中间结果不需要通信，只在模块输出处插入 **All-Reduce**（前向一次、反向一次）。
- 通信的是**激活**而不是参数/梯度，数据量比 DP 小；但**很难与计算重叠**，所以必须放在高带宽域内（一般不跨节点，TP size ≤ 8 对应一个 NVSwitch 节点）。
- 高维扩展：Optimus（2-D，源自 SUMMA/Cannon 算法）、Tesseract（2.5-D）、3-D TP（完美负载均衡）。维度越高，通信量随规模增长越慢，但实现复杂度和小规模开销越大。

#### (c) 流水并行 PP —— 层间切分（inter-layer）

把层切成多个 **stage**，只在切点交换激活，通信频率低 → **适合跨节点、低带宽场景**（论文提到有工作用 PP 来利用地理分散的算力）。两个固有问题：

**问题 1：流水气泡（pipeline bubble）**

| 调度 | 机制 | 代价 |
|------|------|------|
| **GPipe** | fill-drain：所有 micro-batch 先全部前向，再全部反向 | 气泡大（warm-up + cool-down），激活显存高 |
| **PipeDream 1F1B** | 一个 micro-batch 前向完成就立刻做它的反向 | 气泡小，峰值显存低（异步场景） |
| **DAPPLE** | early backward：先注入固定数量 micro-batch，再 round-robin 交错前反向 | 同步语义下的 1F1B |
| **Interleaved 1F1B** | 每 GPU 承担**多个** stage（looping placement） | 气泡更小，但通信量与峰值显存上升 |
| **Zero Bubble** | 把反向拆成**激活梯度**和**参数梯度**两部分，用后者去填气泡 | 气泡近零，峰值显存上升 |
| **Chimera / Hanayo / V-Shape** | 双向流水 / 波浪流水 / 逆序放置双半 | 兼顾气泡与显存均衡（Chimera 需权重冗余） |
| **TeraPipe / Seq1F1B** | 沿**序列维**切 micro-batch，得到更细粒度的 token 并行 | TeraPipe 基于 GPipe 显存大；Seq1F1B 结合 1F1B 改善 |
| **DynaPipe** | 变长输入的动态 micro-batching + 显存感知调度 + 通信提前规划 | 面向多任务训练 |

**问题 2：显存不均衡** —— 靠前的 stage 要同时持有更多活跃 micro-batch 的激活。
解法：**BPipe / MPress**（运行时 D2D 把激活从重载 GPU 换到轻载 GPU）、**Chimera/Hanayo/V-Shape**（对称放置）、**mCAP**（增量 profiling 按峰值显存均分）、**Varuna**（流水 + 重计算的静态规则调度）、**AdaPipe**（不同 stage 用不同的重计算策略 + 自适应分区）。

#### (d) 序列并行 SP —— 按序列维切分（长上下文的关键）

动机：上下文窗口涨到百万 token，激活显存**线性增长**、attention 计算**平方增长**；全量重计算会带来约 **30%** 的额外开销；单纯加大 TP 度数通信开销又太大。

两条技术路线：

- **Ring 系（切序列维）**：Ring Self-Attention（环形传 K 再传 V）、**Megatron Context Parallel**（用 FlashAttention kernel + 去掉因果掩码下的无效计算 + 与对称 GPU 交换半个 chunk 来均衡负载 + 独立 CUDA stream 重叠 KV 通信）、DistFlashAttn、Striped Attention（每卡拿均匀分布的 token 子集而非连续块，解决因果掩码导致的负载不均）、BurstAttention（双缓冲重叠）、Blockwise Ring Attention、WallFacer。
- **All-to-All 系（切 head 维）**：**DeepSpeed-Ulysses** 用 All-to-All 把切分维度从 sequence 换到 head，天然负载均衡且可直接复用 FlashAttention；**但并行度受 head 数限制**，对 MQA/GQA 尤其吃紧。
- **融合**：**LoongTrain / USP** 把 GPU 组织成二维 mesh，先在 ulysses 组内 All-to-All 换维，再在 ring 组内做 Ring-Attention；LoongTrain 进一步提出 Double-Ring-Attention 榨干跨节点带宽。

**另外一个关键细节（§6.1.1）**：Megatron 的 **selective checkpointing** 只丢弃 attention 这类显存密集模块的激活；**DistFlashAttn 把 checkpoint 放在 FlashAttention kernel 的输出处而非 Transformer layer 边界**，从而反向时完全不用重算 attention；**LoongTrain 的 selective-checkpoint++** 把 attention 加白名单，保存输出与 `softmax_lse` 统计量。这三者说明：**重计算的粒度选择比"开不开重计算"重要得多**。

#### (e) 专家并行 EP —— MoE 的稀疏并行

- 基本结构：多个 expert FFN + 一个 gate 路由网络，每个 token 只激活少数 expert。**GShard** 首次把 MoE 扩到分布式 Transformer，expert 分布在不同 worker 上，靠 **All-to-All** 协作；Switch Transformer 用 top-1 路由；DeepSpeed-MoE 用共享 expert + 深层多放 expert。
- **三类问题与解法**：
  1. **稀疏激活与 GeMM 效率**：GeMM 要求各 expert 输入等长，于是要 token dropping/padding，浪费计算。**Megablocks** 用块稀疏矩阵乘支持不等 batch；**ScatterMoE** 的 ParallelLinear kernel 融合 grouped GeMM 与 scatter 读写。
  2. **通信优化**：Tutel（沿 expert capacity 维分组重叠、节点内先聚合小消息再跨节点）、FasterMoE（沿 expert 维切）、PipeMoE（建模通信/计算时间求最优分片数）、ScheMoE（把压缩/通信/expert 计算模块化后做自适应流水调度）、**Lina**（发现 All-to-All 与 AllReduce 重叠时 All-to-All 延迟被拉长，于是**优先保证 All-to-All 拿满带宽**）、Janus（数据不动、**搬 expert**，参数服务器式 + 拉取预取）、TA-MoE（拓扑感知路由 + 拓扑感知辅助损失）。
  3. **负载均衡**：FasterMoE 的 shadowing expert（把热门 expert 参数广播到所有 GPU）、SmartMoE（两阶段搜索 + 池内低成本切换）、FlexMoE（热 expert 多副本 + expand/shrink/migrate 三原语）、Prophet（迭代搜索 expert 放置 + 逐层调度隐藏开销）。

### 4.2 自动并行

统一的三步范式：**① 定义并行策略搜索空间 → ② 建性能模型 → ③ 设计搜索算法**。

按搜索空间分类：
- **DP + PP**：PipeDream（动态规划最小化最慢 stage）、DAPPLE（解析模型 + DP）、AutoPipe（模拟器 + 启发式 + 自动切 micro-batch）、RL 做 device placement。
- **DP + 算子切分**：OptCNN（沿输出张量所有可分维度切）、**FlexFlow**（SOAP 空间 = Sample/Operator/Attribute/Parameter，执行模拟器 + MCMC）、Tofu/HyPar（DP 最小化总通信量）、TensorOpt（给定显存预算的前沿追踪）、AutoMap（MCTS 选 PartIR 分区规则）。
- **DP + TP + PP**：Piper（两级 DP，先切流水再切算子，含重计算）、**Alpa**（把并行分成 **inter-operator** 与 **intra-operator** 两个层级，各层分别求解——这是最值得精读的一篇）、Unity（并行与代数变换在统一并行计算图上联合优化）、Aceso（迭代缓解瓶颈以压缩搜索时间）、nnScaler（三个原语组合搜索空间 + 专家加约束缩小空间）、AutoDDL（坐标下降迭代更新每层 SBP）。
- **系统支持层（对我们最关键）**：
  - **Mesh-TensorFlow**：把设备集群抽象成**多维 mesh**，把并行抽象成**切分迭代空间（张量维度）**；张量维与 mesh 维的映射关系就定义了并行策略（DP 切 batch 维、MP 切 hidden 维）。
  - **GSPMD**：在 JAX/XLA 上，用**简单的张量分片标注（sharding annotation）**统一表达各种并行，其余由编译器推导传播。
  - **OneFlow SBP**：Split / Broadcast / Partial-value 三种分布式张量属性 + placement，用户指定签名即可。
  - **PartIR**：把模型与分区解耦，用 schedule 增量地组合 SPMD 分片策略。
  - **Slapo**：类 TVM 的 **schedule 语言**，把并行与子图优化（算子融合、激活 checkpoint）作为 schedule 原语，与执行解耦、保留原模型结构以便渐进优化。

> **对我们最重要的一句话**：*Mesh-TensorFlow / GSPMD / OneFlow SBP / PartIR / Slapo 这一组工作，本质上就是"把分布式策略编译化"*——**用 IR + 标注 + Pass 来表达和推导并行策略**。这正是我们的项目要在 MLIR 上重做的事情，也是"分布式训练"与"编译器基础设施"两条学习线的真正交点。

### 4.3 异构并行（算力网的正题）

- **异构硬件**：HetPipe（切成多个 virtual worker，组内 PP、组间异步 DP）、AccPar（灵活张量分区 + DP 决策）、Whale（统一抽象 + 自动图优化 + 按硬件信息均衡负载）、AMP（异构感知性能模型）、HPH（按计算/通信比降序排 stage，整数规划最小化迭代时间）、**Pathways**（分片数据流模型 + 异步 gang-scheduling）、SDPipe（半去中心化：同步去中心化、调度集中化）、HAP（A* 搜索张量分片比例与通信方式）、PipePar（DP 分 stage，同时考虑 GPU 异构与带宽异构）。
- **跨地域低带宽**：Yuan et al.（切成 tasklet + 调度算法）、**SWARM parallelism**（等长 stage + 优先路由到低延迟稳定节点 + 设备在 stage 间自适应迁移）、FusionAI（DAG 切子图 + 负载均衡调度，面向消费级 GPU）、CocktailSGD（通信压缩）。
- **异构模型（RLHF）**：PPO 阶段有 actor/critic/reward/reference 四个模型，且分**生成（推理）**与**训练**两个过程。DeepSpeed-Chat 的 Hybrid Engine 在推理时用 TP 提吞吐、训练时用 ZeRO/LoRA 省显存；OpenRLHF 用 Ray + vLLM 把模型分散到不同设备；APP 提出 Interleaving 与 Separation 两种放置策略（生成与训练可独立运行，分置不同设备可消除串行，引入的通信可被计算掩盖）；ReaLHF 在子阶段间**重分布参数以切换最合适的并行模式**；PUZZLE 按阶段亲和性重排任务顺序。

---

## 5. §5 计算优化

### 5.1 算子优化

**手工优化（几乎全在 attention 上）**：
- 内存高效 attention：Rabe et al. 证明 self-attention 只需 **O(log n)** 额外内存（lazy softmax，把除以分母延迟到最后）。
- **FlashAttention 系列**（详见 `08-flash-attention.md`）：IO 感知 tiling + online softmax，把 matmul→softmax→matmul 融合进单个 CUDA kernel 以减少 HBM 访问；FA-2 增加序列维并行与 warp 级调度改进；FA-3 面向 H100，用异步 WGMMA + TMA 做 GEMM 与 softmax 的交错与 warp specialization 软流水。
- BPT 把 tiling 扩展到 FFN；SWattention（申威架构两级分块）；Cutlass 版 FA-2（TMA + WGMMA）。
- **变长序列**：FA-2 沿序列维不可分割地并行；**ByteTransformer** 做 padding-free，用 position array 记录有效 token 在原张量与打包张量间的映射，配合 grouped GEMM。

**编译器自动优化（与我们的编译器学习线直接相连）**：
- **kernel 级**：Halide、TVM（schedule 原语挖并行与局部性）、**Roller**（构造 Load/Store/Compute 的 tile kernel，scale-up-then-scale-out，大幅压缩搜索开销）、**Triton**（基于 C 的 tile 张量程序语言 + 机器相关 pass 做分层 tiling 与 shared memory 分配）、ALCOP（自动 load-compute 流水，多级 pipelining + 索引分析替换）。
- **图级融合**：Chimera（计算密集算子链分解成计算块 + 解析模型选最优块序 + 可替换 microkernel）、**Welder**（把计算图降到 **tile 级数据流图**，节点是算子 tile、边标注复用数据所在的存储层级，在 tile 级搜索跨存储层次最大化数据复用的融合组合）、PyTorch 2 的 **TorchDynamo + TorchInductor**、Slapo、JIT-Q（PIM 上的即时量化）。

> **Welder 的"tile 级数据流图 + 边上标注内存层级"这个建模方式，非常值得借鉴到多后端子图划分问题上**：把"数据在哪一级存储上复用"显式化，划分决策就变成了图上的优化问题。

### 5.2 混合精度

- **FP16 混合精度**：权重/激活/梯度用 FP16 做前反向，**保留一份 FP32 权重副本用于优化器累加**，并用 **loss scaling** 保住小梯度。Campo 用自动图重写优化 FP32↔FP16 的 cast 开销（cast 开销有时会吃掉低精度的收益），并用离线线性回归预测 cast 与执行时间。
- **BF16**：与 FP32 同动态范围，不需要调超参即可收敛（FP16 在 loss scalar 过低时会缓慢发散），BLOOM 用它；但快速 BF16 支持仅在 TPU 与 Ampere 及之后的 NVIDIA GPU 上。
- **FP8 及以下**：Wang et al.（chunk-based 累加 + 随机舍入）、Sun et al.（**前向与反向用不同的指数/尾数位配比**，因为两者对范围与精度的最优平衡点不同）、**FP8-LM**（H100 上的 FP8 自动混合精度框架，逐步把梯度、优化器状态、分布式并行都做成 FP8，用精度解耦与自动缩放解决上下溢）、Rouhani et al.（micro-scaled 格式，缩放因子绑定到张量的细粒度子块）。
- **低位定点**：Jetfire（INT8 数据流 + per-block 量化，线性算子用 INT32 WMMA 累加、非线性用 FP32）、Xi et al. 的 INT4（前向用块对角 Hadamard 变换抑制离群值再量化；反向用 bit splitting + leverage score sampling 选信息量大的梯度）、BitNet（1-bit 权重 + 8-bit 激活，signum 二值化，保留高精度隐权重与优化器状态）、BitNet b1.58（三值 {-1,0,1}）。

---

## 6. §6 内存优化

### 显存的四个组成部分（必须记住）

1. **模型状态（model states）**：参数 + 梯度 + 优化器状态 = **16Φ 字节**（混合精度）。
2. **激活（activations）**：前向产生、反向需要的张量。
3. **临时缓冲（temporary buffers）**：如梯度 All-Reduce 前把梯度打包进一个 flattened bucket。
4. **内存碎片（fragmentation）**：明明总空闲很多却分配失败。

对应四类解法：**重计算 / 冗余消除 / 碎片整理 / Offload**。

### 6.1 激活重计算

- **静态驱逐（static evicting）**：预先定好丢弃计划。Checkmate 用混合整数线性规划求最优重物化方案（**搜索空间太大，扛不住 LLM 规模**）；Megatron 的 selective checkpointing、FlashAttention 的融合 + selective checkpointing、DistFlashAttn 的 rematerialization-aware checkpointing、LoongTrain 的 selective-checkpoint++；Yuan et al. 精确度量每个激活张量的最小重建代价，枚举所有 checkpoint 方案得到**显存-计算的 Pareto 前沿**再选点。
- **动态驱逐（dynamic evicting）**：运行时决策。DTR（贪心在线算法）、MegTaiChi（利用运行时追踪的张量访问模式）、Coop（**用滑动窗口保证只驱逐连续内存块**，避免重计算本身制造碎片）。
- 现实结论：LLM 训练目前**以静态方案为主**，因为架构同质、静态规则够用；动态方案尚未被广泛采用。

### 6.2 冗余消除

- **全分片**：ZeRO-1/2/3、FSDP（见前文表格）。
- **部分分片**（核心权衡：**通信规模 vs 显存冗余**）：ZeRO++（在 ZeRO-3 之上于节点内维护参数的**二级分片**，反向从二级分片收集以减少跨节点通信；并对参数与梯度做量化压缩）、MiCS / FSDP 的组内分片组间复制、**AMSP / PaRO**（Full-Replica / Full-Sharding / Partial-Sharding 三种策略，**允许模型状态的每个组件独立选择分片策略**，AMSP 把它建成显存约束下最小化通信代价的优化问题并定制通信-计算重叠）、RTP（分片激活 + 旋转权重/梯度）。

### 6.3 碎片整理

- **张量级**：DL 框架一般用带内存池的 caching allocator。ROAM（联合优化算子执行顺序与张量分配，基于树搜索）、Imanishi et al.（建模成 2D bin-packing + 模拟退火优化拓扑序）、MegTaiChi / Coop（驱逐激活时考虑碎片）。局限：这些方法在 LLM 规模的分配次数与图规模下可扩展性存疑。
- **VMM 级（更实用）**：**GMLake** 与 **PyTorch expandable segments** 用 CUDA driver 的虚拟内存管理 API，把不连续的物理块通过虚拟地址映射**缝合（stitching）**成大块，几乎不需要数据搬移，对模型与上层优化技术透明、无需改用户代码。PyTorch 2.1 已集成 expandable segments。

### 6.4 Offload

- **CPU offload / 静态**：L2L（逐层搬）、**ZeRO-Offload**（参数留 GPU，优化器状态与梯度放 CPU，**优化器更新也放到 CPU 上算**；16 张 V100 训到 70B，但 GPU 显存会闲置且 CPU 优化器慢）、Elixir（pre-runtime profiling 搜索显存划分与 offload 的最优组合，把模型状态与优化器 chunk 都在 GPU/CPU 间划分）、Mobius（每 GPU 多 stage 并在 GPU/CPU 间动态换入换出 + 预取与 cross-mapping 缓解争用）。
- **CPU offload / 动态**：STRONGHOLD（动态调整"工作窗口"大小以最小化 GPU 停顿）、Harmony（启发式调度 + 减少交换 + 快速 P2P 交换）、TMOF（不相交交换 + 双向重叠协调，避免 PCIe 通道争用）、TSPLIT（切成 micro-tensor 做精细操作 + 模型引导的规划）、PatrickStar（chunk 化 + 运行时内存追踪 + chunk 驱逐 + 设备感知算子放置）。
- **SSD offload**：**ZeRO-Infinity**（模型状态可下到 CPU/NVMe，激活只到 CPU；32 节点 512×V100 支撑 32T 参数，但激活对 CPU 内存需求巨大——10T 模型约需 0.76 TB CPU 内存）、Fuyou（把激活也下到 SSD，把 SSD-CPU 通信当成新的优化维度，同步 out-of-core CPU 优化器与反向重叠）、Smart-Infinity（近存储处理做参数更新）、MoESys（GPU/CPU/SSD 组合存稀疏与稠密参数 + 2D 预取调度）。

---

## 7. §7 通信优化

### 各并行方式的通信画像（一张图就能记住）

论文 Fig.13 是 InternLM-2 102B 在 128 GPU（TP=8, PP=4, DP=4, ZeRO-1=4）上的通信热力图，拓扑优先级 **TP > DP/ZeRO-1 > PP**：

- **TP 的 All-Reduce**：走 NVSwitch 全连接，表现为对角线上 16 个密集方块（每块 = 一个节点）。
- **DP/ZeRO-1 的 ReduceScatter/AllGather**：跨节点，表现为四个 32×32 区域内的 6 条对称斜线。
- **PP 的 Send/Recv**：通信量最小，只是两条黄线。

**核心洞察**：*LLM 训练通信呈现清晰的层次性——绝大部分流量发生在小范围内，只有极少部分跨越整个集群。* 这直接支撑了 rail-only 拓扑那类"砍掉核心交换机省钱"的设计。

### 7.1 集合通信

- **预定义算法**（NCCL / RCCL / MPI 按拓扑与张量大小自动选择）：
  - **Ring**：张量切 chunk 逐个流水传，带宽最优；但**延迟随设备数线性增长**。
  - **Double Binary Tree**：利用"二叉树中一半以下是内部节点、一半以上是叶子"的性质构造两棵互补的树，解决 Ring 的延迟问题。
  - **Hybrid**：Two-level AllReduce（节点内 Reduce → 节点间 AllReduce → 节点内 Broadcast）、2D-Torus AllReduce / ACCL（节点内 ring ReduceScatter → 节点间 tree AllReduce → 节点内 ring AllGather）、BlueConnect（拆成大量可并行的 ReduceScatter/AllGather，各自映射到最合适的网络 fabric）、Plink（探测拓扑生成两级混合方案）。
- **算法综合（synthesized）**：GC3（面向数据的 DSL + 优化编译器）、SCCL（编码成 SMT 公式求 Pareto 最优精确调度）、**TACCL**（建成 MILP，用 communication sketch 抽象缩小搜索空间）、Blink（运行时探测可用链路构造拓扑，生成 packet generation tree 并直接生成 CUDA 代码）、P²（并行矩阵切分并行轴，生成拓扑感知的放置与归约策略）。

> **这一小节对我们特别有价值**：GC3/SCCL/TACCL/Blink 说明**集合通信本身也可以被"编译"**——从拓扑描述出发综合出最优通信算法与 kernel。在异构算力网上，拓扑千变万化，**通信算法综合几乎是必需能力**，而这与 MLIR 那套"IR + 约束求解 + codegen"的方法论完全同源。

### 7.2 通信调度（核心思想：重排通信以与计算重叠）

- **FIFO**：反向过程中每层梯度一算完就发（wait-free backprop）。Poseidon 用 FIFO 队列；GradientFlow / PyTorch DDP 把多个连续 AllReduce **融合成一个**（短暂等待以合并小张量，避免大量小消息）。
- **优先级**：FIFO 的问题是**反向产生通信的顺序与前向消费的顺序相反**，会导致通信阻塞计算。**P3** 把层切成固定大小 slice，按前向使用顺序赋优先级（第一层优先级最高），让当前层的梯度通信与下一层的前向计算重叠；TicTac 按计算图临界路径排序；**ByteScheduler**（统一抽象 + 贝叶斯优化自动调 partition size 与 credit size）、**PACE**（把 AllReduce 切片实现**可抢占**通信，避免大张量 head-of-line blocking；再用动态规划融合小张量）、Lina（MoE 中优先 All-to-All）。
- **分解式调度**：
  - **流水 stage 分解**：Breadth-First（把连续 stage 再细分并首尾相连成环，breadth-first 调度增大重叠）、Fold3D（all-in-all-out，每设备两个模型片段，一个片段的梯度同步与另一个的前/反向重叠）、TriRace（异步流水下推迟参数更新；把双向 P2P 拆成两个单向操作并按临界路径定优先级）。
  - **通信分解**：Wang et al.（把 AllGather/ReduceScatter 拆成细粒度 P2P，把 Einsum 拆成细粒度任务）、SYNDICATE（拆成 Motif，中央优化器用 MCMC 搜最优重叠计划）、**Centauri**（Primitive/Group/Workload 三种切分 + Workload-aware/Backward/Elastic 三种调度）、**DeAR**（把 AllReduce 拆回 AllGather + ReduceScatter，使后续操作能与前向重叠，无需等两步都完成）。
  - **计算分解**：**CoCoNet**（把矩阵乘输出切块，算完一块立刻启动该块的 AllReduce kernel，并精心安排喂给 matmul 的数据块顺序）、T3（软硬协同的 track-and-trigger + 计算增强内存）。
  - **乱序反向（ooo-backprop）**：反向产生两类梯度——**输出梯度**（算前一层用）与**权重梯度**（更新本层用，需 AllReduce）。把两者解耦后，权重梯度计算可以乱序调度，优先安排更关键的计算。Zero Bubble 就用了这个思路来填流水气泡。
  - 重计算与通信：Oases（把 AllReduce 固定为重计算单元的最后一个前向通信操作，并切 sub-batch 让两批的通信与计算重叠）、Lynx（OPT/HEU 两种重计算调度搜索）。

### 7.3 网内聚合（In-Network Aggregation, INA）

- **以太网 / 可编程交换机**：**SwitchML**（把集合通信卸载到可编程交换机，因交换机存储有限而**流式**聚合，一次处理有限数量的向量元素；两个局限——不能直接处理浮点，要用类 block floating-point 转成 32 位整数；主要基于 DPDK，RDMA 版难以集成进训练框架）、**FPISA**（用 P4 在交换机上直接实现浮点计算，FP16 张量可直接卸载）、NetReduce（兼容 RoCE，复用其拥塞控制与可靠性，FPGA 原型）、AllReduce-Switch、PANAMA（多作业间带宽分配）、ATP（多租户）。不适用的：Libra（参数服务器 + 稀疏模型）、iSwitch（RL 场景，需存整个梯度向量，扩不到 LLM）。
- **InfiniBand**：NVIDIA Mellanox **SHARP**（v1 EDR → v2 HDR：支持 Barrier/Reduce/AllReduce/Broadcast、16/32/64 位整数与浮点、GPUDirect RDMA、大向量流式线速归约，**已集成进 NCCL**，对训练框架透明）→ NDR 已生产可用；NVSwitch-v3 也集成了 SHARP。

---

## 8. §8 容错

### 8.1 故障画像（这些数字要记住）

| 集群 | 规模 | 故障频率 |
|------|------|----------|
| BLOOM | 384 GPU | 平均每周 1-2 次 GPU 故障 |
| Meta OPT-175B | 992 A100 | 两周内 40+ 次中断 |
| Acme | 1000+ A100 | 平均 1-2 天一次 |
| 字节 MegaScale | 12288 Ampere | 数周内 100+ 次 |
| Meta LLaMA-3 | 16384 H100 | 54 天内 **466 次**中断 |

- **规模放大效应**：阿里集群单节点日故障率 1.5%，扩到 1000 GPU 就是 **84.8% 的日故障率**。
- **原因分布**：硬件故障影响最严重（GPU 的 CUDA-Error/ECC-Error、NVLink、网络的 NCCL-Timeout/Connection-Error）。LLaMA-3 报告 **78% 是硬件问题**；阿里 C4 观察到 **约 82.5% 的错误局限在特定节点甚至单个设备上**（尽管用户看到的多是 NCCL error）。最新一代 GPU（A100/H100）错误率偏高。此外还有软件问题、模型本身的不稳定（loss spike、数值上下溢、梯度爆炸）、以及供电/散热等外部因素（机房高温 → GPU 过热 → NVLink-Error/ECC-Error 或训练速度不稳）。
- **两种浪费形态**：① 故障恢复（定位诊断 + 回滚 checkpoint + 重算）；② **性能退化（straggler）**——链路故障或异常慢节点不会崩溃但会拉低 MFU，更难发现。

### 8.2 异常检测

- **统计监控**：每 GPU 一个监控进程收集运行时统计，以心跳发往中心监控节点；不发心跳即视为失效。指标来源主要是 **NVIDIA DCGM**（SM block utilization、SM occupancy、SM pipe utilization、PCIe/NVLink 流量率等）。例：Vela 用 `DCGM_FI_DEV_ROW_REMAP_PENDING` 检测显存 row-remapping；MegaScale/Transom 分析训练日志；Unicron 检测 NCCL timeout/TCP timeout/task hang；**C4** 采集 RDMA IP、QP 号与传输层的消息数/大小/时长来发现变慢与挂死；PyTorch 内置 **NCCL flight recorder** 把集合通信元数据与栈回溯写入环形缓冲；Meta 的 **NCCLX** 与 PyTorch 协同设计，追踪每次通信的 kernel 与网络活动；Vela 的 Multi-NIC health checker 采集所有 2 节点对每端口的带宽；Transom 用 ML 算法做异常检测；TPUv4 每机一个 `healthd` 守护进程实时监控 ICI/PCIe/ASIC，发现严重症状即通知集群调度器驱逐或重调度作业。
- **主动验证**：核心矛盾是**验证时间 vs 准确率**。MegaScale 的轻量测试套件（主机内网络 + NCCL 测试）；Vela 的两级策略（周期性轻量测试 + 空闲时的侵入式测试）；TPUv4 的 preflight check；SuperBench（组件级基准套件 + selector 在验证时间与漏检代价间权衡）。

### 8.3 基于 checkpoint 的恢复

核心矛盾：**存得频 → I/O 开销大；存得疏 → 故障时丢的进度多**。持久化 checkpoint 分两个阶段：**snapshot**（GPU → CPU 内存拷贝）与 **persist**（CPU → 持久化存储写盘）。

- **同步**：DeepSpeed 默认、Varuna——在 DP rank 0 上停训做完整 checkpoint，两个阶段都让 GPU 空转。**JIT-Checkpointing** 换了个思路：既然大多数故障只涉及单 GPU/单网卡，那就**在故障发生后再即时打 checkpoint**，把浪费限制在最多一个 mini-batch。DLRover Flash-Checkpoint 用分布式缓存服务加速迁移。**Universal Checkpointing** 提供与并行策略解耦的通用 checkpoint 表示，**可按需在不同并行策略之间转换 checkpoint**。
- **snapshot-stall**：解耦两阶段，只在 snapshot 期间停训，persist 交给后台 CPU 进程异步做。Check-N-Run、TorchSnapshot（张量分块 + 多线程写盘，让 persist 更早开始）、MegaScale 与 InternEvo（停训数秒抓状态，异步传到分布式文件系统；恢复时**指定一个 worker 读再广播给同 DP 组**；InternEvo 还把 checkpoint 从热存储异步搬到冷存储省钱）。
- **异步**：DeepFreeze（后台跑轻/重两级持久化，checkpoint 分片到各 DP GPU 摊 I/O）、CheckFreq（把 snapshot/persist 与下一轮前反向流水化，保证在下次参数更新前完成；动态调 checkpoint 频率）、LightCheck（逐层 checkpoint 流水）、DataStates-LLM（预分配 pinned host memory 解决主机内存分配慢；计算/snapshot/persist 逐层流水）、FastPersist（识别全异步 persist 的风险并与下一轮参数更新同步；双缓冲 pinned memory 提升 SSD 带宽利用；只用部分 DP rank 写 checkpoint 以降低硬件争用）。
- **内存内 checkpoint**：**Gemini**（存到其它计算节点的 CPU 内存，配 checkpoint 放置策略 + 流量调度算法降低对训练的干扰）、**REFT**（异步缓存到主机内存与 Redis 这类内存存储，用**纠删码实现 RAIM5**——RAID5 把 Disk 换成 Memory——来抵御节点故障）。局限：持久性不如存储方案，**实践上需要内存 + 持久化的混合策略**。

### 8.4 无 checkpoint 恢复

- **热迁移（live migration）**：利用 DP 副本之间的天然冗余。故障后动态重配并行策略（用剩余健康实例或加入新实例），把当前模型状态搬过去继续训。**Parcae**（三种迁移机制，通信开销各不相同，用于在不同并行策略间搬状态）、**Oobleck**（维护一组预定义 pipeline template，故障时快速实例化新的异构流水线）。
- **模块冗余**：不恢复状态，而是把计算路由到冗余模块。**Bamboo**（把冗余 stage 放在持有相邻 stage 的同一 GPU 上，平时利用流水气泡做冗余计算，故障时立刻转正）、SlipStream（把故障节点的计算路由到其它 DP 流水线的节点）、**SWARM**（面向连接差、异构、不可靠的设备，冗余计算 + 实例迁移结合）。

> **对我们的启示**：算力网天然是"节点会来会走"的环境，所以 **Oobleck 的 pipeline template + Parcae 的策略间状态迁移 + Universal Checkpointing 的并行无关表示**这三件事组合起来，几乎就是算力网弹性训练的答案骨架。而"并行无关的状态表示"又和 README §5.4 的"低成本后端切换需要统一的状态序列化格式"是同一个问题。

---

## 9. §9 结论与展望

- 传统数字电路计算系统受摩尔定律与 Dennard Scaling 的物理与经济约束，难以满足 LLM 需求。
- 论文押注的方向是**大规模光电集成（silicon photonics）**：光计算 + 光网络的混合数据中心。已有工作：TopoOpt（联合优化光网络拓扑与并行策略）、TPUv4 的 OCS 光路交换动态重构 3D-Torus、Taichi（分布式衍射-干涉混合光子计算架构，百万神经元级，160 TOPS/W）。

---

## 10. 使用示例：把论文概念对应到能跑的代码

> 说明：论文本身是综述、不提供代码。以下示例是把论文里的机制映射到主流框架的**实际可执行形态**，用于建立"概念 ↔ 旋钮"的对应关系。

### 10.1 3D 并行的进程组划分（理解 rank 布局）

```python
# 假设 world_size = 128, TP=8, PP=4, DP=4  （8 * 4 * 4 = 128）
# 论文 Fig.13 的拓扑优先级：TP > DP/ZeRO-1 > PP
# 含义是：TP 组的 rank 必须相邻（落在同一 NVSwitch 节点内），PP 组的 rank 可以隔得最远。
#
# Megatron 的 rank 排布顺序为 tp -> dp -> pp（低位变化最快的维度放通信最重的并行）
def build_groups(world_size=128, tp=8, dp=4, pp=4):
    groups = {"tp": [], "dp": [], "pp": []}
    # TP 组：连续 tp 个 rank 为一组 —— 走 NVLink/NVSwitch，做 All-Reduce
    for i in range(world_size // tp):
        groups["tp"].append(list(range(i * tp, (i + 1) * tp)))
    # DP 组：跨节点，做 ReduceScatter/AllGather（ZeRO-1）
    for p in range(pp):
        for t in range(tp):
            base = p * (tp * dp)
            groups["dp"].append([base + d * tp + t for d in range(dp)])
    # PP 组：通信量最小，只做 Send/Recv，可以放在最远的位置
    for d in range(dp):
        for t in range(tp):
            groups["pp"].append([p * (tp * dp) + d * tp + t for p in range(pp)])
    return groups
```

**要点**：`TP > DP > PP` 这个优先级不是随便定的——它由 §7 那张通信热力图决定：**通信量最大、最难与计算重叠的并行维度必须落在带宽最高的域内。** 在算力网的异构拓扑上，这个映射需要自动求解，而不是手写常量。

### 10.2 Megatron-LM 启动命令（把论文旋钮找出来）

```bash
torchrun --nproc_per_node 8 --nnodes 16 pretrain_gpt.py \
  --tensor-model-parallel-size 8          `# §4.1.2 TP：只在节点内，因为 All-Reduce 难重叠` \
  --pipeline-model-parallel-size 4        `# §4.1.3 PP：跨节点，只传 stage 边界激活` \
  --num-layers-per-virtual-pipeline-stage 2 `# Interleaved 1F1B：一卡多 stage，减气泡但增通信` \
  --context-parallel-size 2               `# §4.1.4 SP：Megatron Context Parallel（ring 系）` \
  --sequence-parallel                     `# Megatron-SP：把非 TP 区域也沿序列维切，省冗余激活` \
  --recompute-granularity selective       `# §6.1.1 选择性重计算：只丢 attention 这类显存密集模块` \
  --use-flash-attn                        `# §5.1.1 FlashAttention kernel` \
  --bf16                                  `# §5.2.1 BF16：与 FP32 同动态范围，无需调 loss scale` \
  --overlap-grad-reduce                   `# §7.2 通信调度：梯度归约与反向计算重叠` \
  --overlap-param-gather                  `# 参数 All-Gather 与前向重叠` \
  --use-distributed-optimizer             `# ≈ ZeRO-1：优化器状态分片到 DP 组` \
  --global-batch-size 1024 --micro-batch-size 1  `# micro-batch 数决定流水气泡率`
```

**气泡率的估算**（1F1B 同步流水）：\( \text{bubble ratio} \approx \dfrac{p - 1}{m} \)，其中 \(p\) 为 stage 数、\(m\) 为 micro-batch 数。上例 \(p=4\)、\(m = 1024 / (1 \times 4\text{个 DP}) = 256\)，气泡率约 1.2%。**这个公式解释了为什么大 global batch 是流水并行的前提。**

### 10.3 PyTorch FSDP：分片因子 F 的三种取值

```python
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.device_mesh import init_device_mesh

# 对应论文 §4.1.1 的三种 sharding factor
#   F = 1  -> NO_SHARD        （等价 DDP，显存最多、通信最少）
#   F = W  -> FULL_SHARD      （ZeRO-3，显存 16Φ/N，通信约 1.5x DDP）
#   1<F<W  -> HYBRID_SHARD    （组内分片、组间复制，减少跨节点通信规模）
mesh = init_device_mesh("cuda", (16, 8), mesh_dim_names=("replica", "shard"))

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,  # 节点内分片，节点间复制
    device_mesh=mesh,
    auto_wrap_policy=transformer_auto_wrap_policy,     # 以 Transformer layer 为分片单元
    limit_all_gathers=True,      # 限制预取深度：控制 All-Gather 造成的显存峰值
    forward_prefetch=True,       # §7.2 通信调度：提前 All-Gather 下一层参数
    use_orig_params=True,
)
```

### 10.4 DeepSpeed ZeRO-3 + Offload 配置（内存优化的旋钮全景）

```json
{
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param":     { "device": "cpu", "pin_memory": true },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_max_live_parameters": 1e9,
    "reduce_bucket_size": 5e7,
    "zero_quantized_weights": true,
    "zero_hpz_partition_size": 8
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": true
  }
}
```

| 配置项 | 对应论文机制 |
|--------|--------------|
| `stage: 3` | §6.2.1 全分片，模型状态 16Φ → 16Φ/N |
| `offload_optimizer` / `offload_param` | §6.4.1 ZeRO-Offload |
| `zero_quantized_weights` + `zero_hpz_partition_size` | §6.2.2 **ZeRO++**：参数量化 + 节点内二级分片 |
| `overlap_comm` / `*_bucket_size` | §7.2 FIFO 融合 + 通信计算重叠 |
| `contiguous_gradients` / `contiguous_memory_optimization` | §6.3 碎片整理 |
| `partition_activations` | §6.1 激活分片与重计算 |

### 10.5 通信与碎片相关的环境变量

```bash
# §7.1 集合通信算法选择：Ring 带宽优、Tree 延迟优
export NCCL_ALGO=Tree                  # 也可 Ring / CollnetDirect / NVLS
export NCCL_PROTO=Simple               # LL / LL128 / Simple
export NCCL_NVLS_ENABLE=1              # NVSwitch-v3 上的 SHARP（§7.3 网内聚合）
export NCCL_COLLNET_ENABLE=1           # InfiniBand SHARP 卸载
export NCCL_IB_HCA=mlx5_0,mlx5_1       # 绑定网卡，配合 rail-optimized 拓扑
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO                 # 排查实际选中的算法与拓扑
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000   # §8.2.1 NCCL flight recorder（故障诊断）

# §6.3.2 VMM 级碎片整理：expandable_segments 通过虚拟地址映射缝合非连续块
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 10.6 一个简化的"性能模型"骨架（自动并行的核心）

自动并行系统（§4.2）都需要这样一个代价函数。写一个最粗糙的版本能极大帮助理解各并行维度的取舍：

```python
def estimate_iter_time(cfg, model, cluster):
    """给定并行配置，估算单轮迭代时间。这是所有自动并行系统的骨架。"""
    tp, pp, dp, sp = cfg.tp, cfg.pp, cfg.dp, cfg.sp
    m = cfg.global_batch // (cfg.micro_batch * dp)      # micro-batch 数

    # 1) 计算时间：FLOPs / (每卡算力 * 有效利用率)
    flops_per_micro = 6 * model.params * cfg.micro_batch * model.seq_len   # 前向+反向≈6ΦBT
    t_comp = flops_per_micro / (cluster.flops_per_gpu * cfg.mfu) / (tp * pp)

    # 2) TP 通信：每层两次 All-Reduce 激活，走节点内高带宽
    vol_tp = 2 * model.layers * cfg.micro_batch * model.seq_len * model.hidden * 2  # bytes
    t_tp = 2 * (tp - 1) / tp * vol_tp / cluster.bw_intra          # ring All-Reduce 系数

    # 3) DP 通信：梯度 ReduceScatter + 参数 AllGather，走跨节点带宽，可与反向重叠
    vol_dp = model.params * 2 / tp / pp
    t_dp = max(0, 2 * (dp - 1) / dp * vol_dp / cluster.bw_inter - t_comp * cfg.overlap_ratio)

    # 4) PP：气泡 + stage 间 Send/Recv
    t_bubble = (pp - 1) / m * (t_comp + t_tp)
    t_pp = model.layers / pp * 0 + (pp - 1) * cfg.act_bytes / cluster.bw_inter

    # 5) 显存约束（16Φ 模型状态 + 激活），超了这个配置直接不可行
    mem = 16 * model.params / (tp * pp * (dp if cfg.zero3 else 1)) + estimate_act(cfg, model)
    if mem > cluster.mem_per_gpu:
        return float("inf")

    return m * (t_comp + t_tp) + t_dp + t_bubble + t_pp
```

**这个骨架揭示的权衡**：TP 增大 → 计算变快但 `t_tp` 线性增长且难重叠；PP 增大 → 显存下降但气泡上升；DP 增大 → 吞吐上升但 `t_dp` 上升；ZeRO-3 → 显存除以 N 但通信 ×1.5。**在异构集群上，`bw_intra` / `bw_inter` / `flops_per_gpu` 都变成了 per-device 的向量，问题从"选一个配置"变成"求一个不均匀分配"——这就是算力网场景真正的新问题。**

---

## 11. 与本项目（算力网上大模型分布式运行基础设施）的关联

把这篇综述与 README §6 的六个研究问题对齐，可以看到相当明确的映射：

| README 的研究问题 | 本篇论文里的对应机制 | 算力网场景下的新挑战 |
|---|---|---|
| ① 子图划分与跨设备传输最小化 | §4.2 自动并行（Alpa 的 inter/intra-operator 分层、FlexFlow 的 SOAP + 模拟器）、§5.1.2 Welder 的 tile 级数据流图 | 设备算力与带宽都不均匀，划分要同时解决"切哪里"和"每块给谁"，且带宽矩阵是动态的 |
| ② 跨后端 layout / 量化 / 内存空间兼容 | §5.2 混合精度（FP16/BF16/FP8/INT8/INT4 的格式差异与 cast 开销，Campo 用图重写优化 cast） | 不同厂商加速器支持的精度集不同（有的无 BF16、有的只有 INT8），跨设备传输时必须插入转换，转换代价要进代价模型 |
| ③ 动态 shape / 动态 batch / KV Cache | §4.1.4 序列并行、§4.1.3 DynaPipe 的变长动态 micro-batching、§5.1.1 ByteTransformer 的 padding-free | 训练侧是变长序列 + 动态 micro-batch；推理侧是 PagedAttention（见 `09-paged-attention-vllm.md`） |
| ④ 运行状态变化后的低成本后端切换 | §8.4 热迁移（Parcae 的策略间状态迁移、Oobleck 的 pipeline template）、§8.3.1 **Universal Checkpointing 的并行无关表示**、§4.3.2 ReaLHF 的参数重分布 | 这几乎就是我们要做的核心能力：**需要一个与并行策略、与后端都无关的状态表示** |
| ⑤ 后端最优计算图不一致 | §5.1.2 算子自动优化（Roller/Triton/Welder 在不同后端搜出不同 kernel）、§4.3.1 异构硬件并行 | 同一模型在不同后端上不仅 kernel 不同，**最优并行策略也不同**，需要 per-device 的策略与图 |
| ⑥ 配置组合爆炸 | §4.2 自动并行的搜索空间与性能模型（Aceso 压缩搜索时间、nnScaler 用专家约束缩空间、SmartMoE 的策略池）、§7.1.2 通信算法综合 | 维度更多：设备型号 × 拓扑 × 并行度 × 精度 × 序列长度。**"用专家约束缩小空间 + 分层缓存"是目前最现实的路线** |

**三条最值得投入的切入点**：

1. **把并行策略编译化**。Mesh-TensorFlow / GSPMD / OneFlow SBP / PartIR 已经证明"分片标注 + 编译器传播推导"可行。在 MLIR 上做一套面向异构算力池的分片方言（sharding dialect），是我们这个项目最自然的技术形态，而且正好把 README 的第二、三阶段（MLIR/TVM）与第一阶段（分布式）接起来。
2. **异构感知的代价模型**。§4.3 的异构并行工作（AMP、HAP、PipePar、HPH）都在做同一件事：把"设备异构 + 带宽异构"塞进代价模型再求解。算力网的特殊性在于**拓扑与可用性还会随时间变化**，所以代价模型必须能在线更新（对应 §3.4 的 Crius）。
3. **并行无关的状态表示 + 弹性恢复**。Universal Checkpointing + Oobleck + Parcae 这条线，是算力网"节点随时进出"这一前提下的必需能力，也直接回答 README 的问题 ④。

---

## 12. 学习这篇论文的最小必要集

### 必须掌握（这些是后续所有讨论的公共词汇）

1. **SER 三角与三个锚定数字**：MFU 40%、故障 466 次/54 天、56% 时间浪费。
2. **显存账本**：混合精度下模型状态 = **16Φ 字节**；ZeRO-1/2/3 分别把哪一部分除以 N。
3. **五种并行的切分维度与通信原语**：DP 切 batch（All-Reduce / ReduceScatter+AllGather）、TP 切 hidden（All-Reduce 激活，节点内）、PP 切 layer（Send/Recv，跨节点）、SP 切 sequence 或 head（Ring P2P 或 All-to-All）、EP 切 expert（All-to-All）。**要能说出每一种"切什么、通什么、放哪里、为什么"。**
4. **流水气泡公式** \((p-1)/m\) 与 GPipe / 1F1B / Interleaved 1F1B / Zero Bubble 的区别。
5. **通信层次性洞察**：TP > DP > PP 的带宽需求排序，以及"99% 的 GPU 对之间没有流量"这个结论。
6. **重计算的粒度选择**：selective checkpointing 与"把 checkpoint 放在 FlashAttention 输出处"的区别。
7. **通信-计算重叠的三种手段**：FIFO 分桶融合、优先级调度（P3 的"按前向顺序定优先级"）、分解式（CoCoNet 切输出块、DeAR 拆 AllReduce）。
8. **SPMD 的分片标注范式**：Mesh-TensorFlow 的 mesh 抽象、GSPMD 的 sharding annotation、OneFlow 的 SBP。**这是分布式与编译器的交汇点，务必吃透。**
9. **容错的三条路线**：快照-停顿式 checkpoint、内存内 checkpoint、无 checkpoint 恢复（热迁移 / 模块冗余）。

### 可以先跳过（遇到再查）

- §3.2 的具体拓扑细节（Dragonfly+、BCube、DCell、HammingMesh 的构造方式）与拥塞控制算法的具体机制（DCQCN/HPCC/Swift 的公式）——**只需要知道"训练流量是少数超大流 + 周期性突发，传统 ECMP 会撞流"这个结论**。
- §3.3 各种并行文件系统与缓存系统的对比细节。
- §4.1.2 的 2-D / 2.5-D / 3-D 张量并行的矩阵乘算法推导（SUMMA、Cannon）——先掌握 Megatron 1-D 的列切/行切即可。
- §4.1.5 MoE 的绝大部分具体系统（除了 GShard 的基本范式、All-to-All 通信瓶颈、以及"负载不均衡"这个核心问题）。
- §5.2.2/5.2.3 的 FP8/INT4/1-bit 的具体数值技巧（Hadamard 变换、leverage score sampling、随机舍入）。
- §6.4 各 offload 系统的细节差异——**知道"CPU offload 换来容量、代价是 PCIe 带宽与 CPU 优化器慢"就够了**。
- §7.3 网内聚合的硬件实现细节（P4 编程、FPGA 原型）——只需知道 **SHARP 已集成进 NCCL，是生产可用的**。
- §8 中各 checkpoint 系统的流水细节；先掌握"snapshot / persist 两阶段"的分解思想。
- §9 的光电计算展望。
- 全部 400+ 篇引文——只在需要深入某个具体机制时按名字去查。

### 建议的动手验证（把纸面知识变成手感）

1. 用两张卡（或单卡模拟）跑一次 DDP vs FSDP，用 `torch.cuda.max_memory_allocated()` 验证 **16Φ → 16Φ/N** 的显存账本。
2. 开关 `--recompute-granularity selective`，测量显存与吞吐的变化，验证"全量重计算约 30% 开销"这个数字。
3. 用 `NCCL_DEBUG=INFO` 观察 NCCL 实际选了 Ring 还是 Tree，以及在不同张量大小下的切换点。
4. 用 PyTorch profiler 抓一次训练迭代的 timeline，亲眼看到 All-Reduce 与反向计算的重叠（或不重叠）。
5. 把 §10.6 的性能模型骨架填上真实的机器参数，扫一遍 (TP, PP, DP) 的组合，看看最优配置在哪里——这是理解自动并行最快的方式。

---

## 13. 延伸阅读优先级

按对我们项目的相关度排序，这几篇是本综述里最值得读原文的：

1. **Alpa**（OSDI'22）：inter-operator + intra-operator 的分层自动并行，最接近我们要做的"编译式分布式"。
2. **GSPMD** / **Mesh-TensorFlow**：分片标注 + 编译器传播的范式源头。
3. **Megatron-LM**（TP）+ **Megatron-SP / Context Parallel**（SP）：工业界事实标准的切分方式。
4. **ZeRO** / **FSDP**：显存优化的基础设施。
5. **MegaScale**（NSDI'24）：万卡工程实践的全景，故障与 straggler 的真实处理方式。
6. **Oobleck** / **Parcae** / **Universal Checkpointing**：弹性与状态迁移。
7. **TACCL** / **SCCL**：集合通信算法综合——异构拓扑下的必需能力。
8. **Welder** / **Roller** / **Triton**：算子与图级编译，与本仓库的编译器学习线直接衔接。
