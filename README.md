# AI-Infra

面向 **「算力网上大模型分布式运行基础设施」** 的学习与实践仓库。

> **怎么用这套仓库**：本文件是**总规划**（学什么 / 不学什么 / 学到什么程度）。  
> **按阶段「读什么 → 做什么 → 验什么」** 的完整自学路径，见 **[`docs/README.md`](docs/README.md)**（自学体系枢纽）。

---

## 仓库怎么组成一套自学体系

```
                    ┌─────────────────────────────────────┐
                    │  README.md（本文件）= 总规划          │
                    │  目标 · 优先级 · 必学/跳过 · 时间表   │
                    │  验收标准 · 六个研究问题              │
                    └──────────────────┬──────────────────┘
                                       │ 指导学什么
                                       ▼
┌──────────────────────┐    读→做→验闭环     ┌──────────────────────────┐
│  docs/ 知识库         │ ◄─────────────────▶ │  动手项目                 │
│  · paper-notes/ 论文  │                      │  · llvm-hello-compile    │
│  · llvm-learning-…   │                      │  · mlir-toy-dialect      │
│  · iree-learning-…   │                      └──────────────────────────┘
└──────────┬───────────┘
           │ 细节不够再翻
           ▼
      paper/ 原文 PDF
```

| 目录 | 角色 | 入口 |
|------|------|------|
| **[`docs/README.md`](docs/README.md)** | **自学体系枢纽**：每阶段读什么 / 做什么 / 过关标准 | **日常导航从这里进** |
| [`docs/paper-notes/`](docs/paper-notes/) | 论文精读笔记（框架 · 核心特性 · 最小必要集） | [笔记索引](docs/paper-notes/README.md) |
| [`docs/llvm-learning-guide.md`](docs/llvm-learning-guide.md) | LLVM 官方文档蒸馏（四层 IR · New PM · CodeGen） | 配 [`llvm-hello-compile/`](llvm-hello-compile/) |
| [`docs/iree-learning-guide.md`](docs/iree-learning-guide.md) | IREE 核心概念 + **HAL 详解** | P0 主教材之一 |
| [`llvm-hello-compile/`](llvm-hello-compile/) | 动手：源码→IR→Pass→汇编；手写 New PM 插件 | `bash scripts/run.sh` |
| [`mlir-toy-dialect/`](mlir-toy-dialect/) | 动手：双 dialect · Region · Interface · Dialect Conversion | `bash scripts/all.sh` |
| [`paper/`](paper/) | 原始论文 PDF | 笔记说不清时再打开 |

**两个动手项目在主线上的位置**：

1. **`llvm-hello-compile`**（P2 地基，建议先跑一遍）：建立「单层 IR + Pass + CodeGen」手感 → 对应 [`docs/llvm-learning-guide.md`](docs/llvm-learning-guide.md)。
2. **`mlir-toy-dialect`**（P0 主战场）：建立「多层 dialect + 渐进 lowering」手感 → 对应 [`docs/paper-notes/03-mlir.md`](docs/paper-notes/03-mlir.md)；核心机制已齐，下一刀是 Toy→linalg→llvm 端到端。

两者同构（TableGen/ODS、lit/FileCheck、分析型 vs 变换型 Pass），对照着做会快很多。详细阶段地图见 [`docs/README.md`](docs/README.md)。

---

## 0. 先读这一段：结论先行

老师的任务是**做一套适配算力网上大模型分布式运行的基础设施**。把这个目标拆开，它需要三种能力：

1. **能表达分布式**——知道大模型怎么被切到成千上万张卡上，切分策略如何用 IR 与标注来表达（而不是手写）。
2. **能适配异构后端**——算力网必然是多厂商、多型号、跨地域的混合池，同一个模型要在不同后端上跑起来，需要统一的编译栈与设备抽象。
3. **能在运行时做决策**——设备会变、负载会变、故障会发生，静态编译一次跑到底的假设在算力网上不成立。

对照这三种能力，学习路线上的项目优先级如下（**这就是本文档最重要的一张表**）：

| 优先级 | 项目 | 为什么是这个优先级 | 建议投入 |
|--------|------|---------------------|----------|
| **P0** | **分布式训练综述** | 项目的正题。不懂并行策略与通信/显存约束，后面所有编译优化都无的放矢 | 20% |
| **P0** | **MLIR（深化）** | 我们要做的基础设施几乎一定是 MLIR 形态。已有 toy dialect 基础，需补齐 Conversion / Interface / linalg | 20% |
| **P0** | **IREE + HAL** | **多后端统一运行时的现成样板**：device/buffer/command buffer/多目标变体，直接对应「异构算力池的统一设备抽象」 | 15% |
| **P1** | **TVM** | 图级编译器的完整范式：融合、layout、调度、代价模型与搜索。对应「子图划分」与「配置组合爆炸」 | 12% |
| **P1** | **ONNX + ExecuTorch/ORT 多后端委托** | 工程上绕不开的交换格式，且第五阶段六个研究问题就是从委托机制里长出来的 | 10% |
| **P1** | **FlashAttention + PagedAttention** | 大模型负载的**真实形态**。不了解它们，做出来的基础设施会脱离实际负载 | 8% |
| **P2** | **LLVM（复习）** | 已完成 `llvm-hello-compile`，概念够用，按需回查 | 5% |
| **P2** | **Halide** | 只为理解「算法与调度分离」这一个思想，1~2 天足够 | 4% |
| **P2** | **Glow** | 与 TVM 对比阅读，理解「少量原语 + lowering」的低成本多后端策略 | 3% |
| **P2** | **CUDA fatbin** | 半天。理解「一个二进制携带多架构变体」——它和 IREE ExecutableVariant 是同一个思想 | 3% |

**一句话总结优先级逻辑**：
> **先把「分布式怎么切」和「多后端怎么统一」这两件事搞清楚（P0），再补「怎么编译得快」（P1），最后回填编译器基础与历史脉络（P2）。**

> **本阶段的学习原则**（这条原则贯穿全文）：
> **只学「一直要用的核心运行框架 + 核心特性」。** 每个项目下面都明确列出了「必须掌握」与「先跳过」两栏——凡是列在「先跳过」里的，遇到再学，现在花时间是浪费。判断标准很简单：**这个知识点在我未来三个月的编码里会不会反复出现？** 会，就现在学；不会，就记下名字，遇到再查。

---

## 1. 目标与技术形态

### 1.1 「算力网上大模型分布式运行基础设施」到底要做什么

结合调研论文 [Efficient Training over Distributed Infra](docs/paper-notes/01-efficient-training-distributed-infra.md) 和第五阶段的研究问题，这套基础设施大致要回答四个工程问题：

```
                        一个大模型（Transformer）
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
  ① 怎么切？               ② 切完放哪？              ③ 怎么执行？
  并行策略的表达            异构设备的映射             统一运行时
  （DP/TP/PP/SP/EP）        （算力/带宽/显存异构）      （device/buffer/
  用 IR + 分片标注表达       拓扑感知的放置             command buffer）
        │                        │                        │
        └────────────────────────┼────────────────────────┘
                                 ▼
                         ④ 变了怎么办？
                  设备退出/负载变化/故障 → 重划分、状态迁移
                  （并行无关的状态表示 + 弹性恢复）
```

- **① 和 ② 的答案在分布式训练那条线上**：Mesh-TensorFlow 的 mesh 抽象、GSPMD 的分片标注、OneFlow 的 SBP、Alpa 的 inter/intra-operator 分层，都是把「并行策略」变成「编译器可推导的 IR 属性」。
- **③ 的答案在 IREE HAL 那条线上**：device / buffer / command buffer / executable variant 就是一套现成的、经过工业验证的硬件抽象层。
- **④ 是当前最开放的问题**，也最可能是我们的贡献点：Universal Checkpointing（并行无关的状态表示）+ Oobleck（pipeline template 快速重构）+ Parcae（策略间状态迁移）是目前最接近答案的三块拼图。

### 1.2 两条学习线在哪里交汇

这是理解整份学习计划的关键。很多人会把「分布式训练」和「编译器」当成两件事，其实它们的交点非常明确：

```
分布式训练线                                       编译器线
─────────────                                     ─────────
手写并行策略                                       LLVM IR / Pass
  Megatron TP、DeepSpeed ZeRO                        ↓
       ↓                                          MLIR：多层 dialect + 渐进 lowering
自动并行：Alpa / FlexFlow                              ↓
       ↓                                          TVM：图级优化 + 调度 + 搜索
把并行策略编译化 ★★★                                   ↓
  Mesh-TensorFlow：mesh 抽象  ────────┐            IREE：flow → stream → hal
  GSPMD：sharding annotation          │                ↓
  OneFlow：SBP 签名                   ├──── 交汇 ───→ 用 IR 表达分布式，
  PartIR：分区与模型解耦              │              用 Pass 推导分片，
  Slapo：schedule 原语                │              用 HAL 抽象异构设备
                            ─────────┘
```

**所以「MLIR + IREE」不是编译器爱好者的自娱自乐，而是实现「可编译的分布式」最现实的工程载体。** 这也是为什么它们和分布式综述并列 P0。

---

## 2. 学习路线全景

> **可执行的「读→做→验」阶段表**（含每个阶段打开哪份文档、跑哪个脚本）在 [`docs/README.md`](docs/README.md) §2。下面这张图只标优先级与顺序。

```
┌─ 阶段 0（建议先跑，P2 地基）───────────────────────────────────┐
│  llvm-hello-compile  +  docs/llvm-learning-guide.md             │
│    └─ 源码→IR→Pass→汇编；SSA / New PM 手感                      │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─ P0 ───────────────────────────────────────────────────────────┐
│  ① 分布式训练基础 ← docs/paper-notes/01-…                       │
│  ② MLIR 深化      ← 03-mlir.md  +  mlir-toy-dialect/            │
│  ③ IREE + HAL     ← docs/iree-learning-guide.md（学完 MLIR 立刻接）│
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─ P1 ───────────────────────────────────────────────────────────┐
│  TVM · ONNX/ORT/ExecuTorch 委托 · FlashAttention · PagedAttention│
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─ P2（回填）· Halide · Glow · CUDA fatbin · LLVM 深挖 CodeGen ──┐
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─ 收束：六个研究问题（§6），边学边对照「已有工作参照」───────────┐
└─────────────────────────────────────────────────────────────────┘
```

> **注意路线顺序的一处调整**：原路线把 IREE 放在第四阶段（运行时与部署）。但对我们的项目而言，**IREE HAL 是「异构设备统一抽象」的直接答案，应该在学完 MLIR 之后立刻接上，不要等到 TVM 之后。** TVM 更多是「单算子怎么编快」，与我们的核心命题相隔一层。

---

## 3. P0 项目详解

### 3.1 分布式训练基础

📄 **精读笔记**：[`01-efficient-training-distributed-infra.md`](docs/paper-notes/01-efficient-training-distributed-infra.md)
📌 **原文**：[Efficient Training of LLMs on Distributed Infrastructures: A Survey](https://arxiv.org/abs/2407.20018)

#### 必须掌握的「核心运行框架」

能徒手画出这张六层图，并说清每层解决什么问题：

```
容错（异常检测 / checkpoint / 无 checkpoint 恢复）
   ↑
计算优化 · 内存优化 · 通信优化
   ↑
并行策略（混合并行 / 自动并行 / 异构并行）
   ↑
基础设施（加速器 / 网络 / 存储 / 调度）
```

#### 必学核心特性（按重要性排序）

| # | 知识点 | 掌握标准 |
|---|--------|----------|
| 1 | **SPMD 分片标注范式** ★最重要 | 能解释 Mesh-TensorFlow 的 device mesh、GSPMD 的 sharding annotation、OneFlow 的 SBP（Split/Broadcast/Partial）三者在表达同一个并行策略时分别怎么写。**这是我们项目的技术原型** |
| 2 | **显存账本 16Φ** | 能背出：混合精度下模型状态 = 参数 2Φ + 梯度 2Φ + 优化器状态 12Φ = **16Φ 字节**；ZeRO-1/2/3 分别把哪一项除以 N |
| 3 | **五种并行的四要素** | 对 DP/TP/PP/SP/EP，能说出「**切什么维度 / 通信什么原语 / 放在哪个带宽域 / 主要代价是什么**」 |
| 4 | **3D 并行的 rank 布局** | 能解释为什么优先级是 **TP > DP/ZeRO > PP**，并能手写进程组划分代码 |
| 5 | **流水气泡** | 记住 \((p-1)/m\)；能说清 GPipe / 1F1B / Interleaved 1F1B / Zero Bubble 的差别与各自代价 |
| 6 | **通信-计算重叠的三种手段** | FIFO 分桶融合、优先级调度（P3 按前向顺序定优先级）、分解式（CoCoNet 切输出块、DeAR 拆 AllReduce） |
| 7 | **重计算的粒度选择** | 理解 selective checkpointing，以及「把 checkpoint 放在 FlashAttention 输出处」为什么更优 |
| 8 | **容错三条线** | snapshot-stall checkpoint / 内存内 checkpoint / 无 checkpoint 恢复（热迁移、模块冗余）。**重点看 Universal Checkpointing 的「并行无关表示」** |

#### 先跳过（遇到再学）

- 网络拓扑的构造细节（Dragonfly+、BCube、DCell、HammingMesh）与拥塞控制算法公式（DCQCN/HPCC/Swift）。只需记住结论：**训练流量 = 少数超大流 + 周期性突发，传统 ECMP 会撞流**。
- 2-D / 2.5-D / 3-D 张量并行的矩阵乘算法推导（SUMMA、Cannon）。先掌握 Megatron 1-D 的列切/行切。
- MoE 的具体系统清单（Tutel / FasterMoE / SmartMoE / FlexMoE 等）。只需掌握 GShard 的基本范式 + All-to-All 瓶颈 + 负载不均衡这三点。
- FP8 / INT4 / 1-bit 的数值技巧（Hadamard 变换、leverage score sampling、随机舍入）。
- 各 offload 系统的对比细节。记住「CPU offload 换容量，代价是 PCIe 带宽和 CPU 优化器慢」即可。
- 网内聚合的硬件实现（P4 编程、FPGA 原型）。只需知道 **SHARP 已集成进 NCCL，生产可用**。
- 光电计算展望、400+ 篇引文。

#### 动手验收

1. 单机双卡跑 DDP vs FSDP，用 `torch.cuda.max_memory_allocated()` **实测验证 16Φ → 16Φ/N**。
2. 开关 selective recompute，测量显存与吞吐变化。
3. `NCCL_DEBUG=INFO` 观察 NCCL 在不同张量大小下选 Ring 还是 Tree。
4. 用 PyTorch profiler 抓一次迭代 timeline，**亲眼确认 All-Reduce 是否与反向计算重叠**。
5. 填一个简易性能模型（笔记 §10.6 有骨架），扫 (TP, PP, DP) 组合找最优配置。

---

### 3.2 MLIR 深化

📄 **精读笔记**：[`03-mlir.md`](docs/paper-notes/03-mlir.md)
📌 **原文**：[MLIR: A Compiler Infrastructure for the End of Moore's Law](https://arxiv.org/abs/2002.11054)
🛠 **动手主战场**：[`mlir-toy-dialect/`](mlir-toy-dialect/) —— 先 `bash scripts/all.sh` 跑通九组演示与 lit；项目 README 有「MLIR 核心特性覆盖对照表」
🧭 **阶段导航**：[`docs/README.md` 阶段 2](docs/README.md#阶段-2mlir-深化p0约-2-周)

#### 必须掌握的「核心运行框架」

```
IR 数据结构：Operation ⊃ Region ⊃ Block ⊃ Operation（可嵌套）
             Operation 持有 Operand(Value) / Result(Value) / Attribute / Type
             Value 满足 SSA，Block 有 block argument（替代 phi 节点）

Dialect = 一组 Op + Type + Attribute + Interface 的命名空间

变换基础设施：PassManager → Pass → RewritePattern → DialectConversion
声明式定义：  ODS/TableGen(.td) 生成 Op 的 C++ 类
正确性保障：  Verifier + Trait + Location
```

#### 必学核心特性

| # | 知识点 | 掌握标准 | 你已有的基础 |
|---|--------|----------|--------------|
| 1 | **ODS 定义 Op** | 熟练写 `.td`：arguments / results / assemblyFormat / traits / hasVerifier / hasCanonicalizer | ✅ 已有 `ToyOps.td`、`LowOps.td` |
| 2 | **RewritePattern + Canonicalization** | 能写 `matchAndRewrite`，理解 greedy driver 的收敛行为 | ✅ 已有强度削减 pass |
| 3 | **DialectConversion 全套** ★ | `ConversionTarget` 的 legal/illegal/dynamicallyLegal、`TypeConverter` 与 materialization、partial vs full conversion 的区别 | ⚠️ **需要补**——这是跨 dialect lowering 的核心，也是我们做多后端 lowering 的主武器 |
| 4 | **Interface** ★ | 理解 OpInterface / TypeInterface 为什么能让通用 pass 跨 dialect 复用；至少用过 `MemoryEffectsOpInterface`、`LoopLikeOpInterface` | ⚠️ **需要补** |
| 5 | **内置 dialect 地图** | 说得出 `func / arith / scf / cf / memref / tensor / linalg / vector / gpu / llvm` 各自的抽象层级与典型 op | ⚠️ 需要补 |
| 6 | **linalg + bufferization** ★ | 理解 linalg on tensors 的结构化语义、tiling 与 fusion 为什么在 linalg 上做、bufferization 把 tensor 变 memref 的时机 | ⚠️ **需要补**——IREE 与所有现代 ML 编译栈都建在 linalg 上 |
| 7 | **Pass pipeline 与调试** | 熟练用 `mlir-opt --pass-pipeline=...`、`--print-ir-after-all`、`--mlir-print-op-generic`、`--debug-only=dialect-conversion` | ✅ 部分已有 |
| 8 | **渐进式 lowering 的设计哲学** | 能论述「为什么不一步降到 LLVM IR」，以及每一层应该固化什么决定 | ✅ 已有直觉，需系统化 |

#### 先跳过（遇到再学）

- PDL / PDLL 声明式重写语言（先知道存在，写 C++ pattern 足够）。
- Transform dialect 的完整用法（知道它是「用 IR 描述调度」即可，做 tiling 时再学）。
- Python bindings、手写 custom assembly parser/printer（用 `assemblyFormat` 覆盖 95% 场景）。
- 外围 dialect：`spirv` / `emitc` / `async` / `pdl` / `acc` / `omp` 的细节。
- MLIR 的 C API、ExecutionEngine 内部实现。

#### 动手验收（在现有 `mlir-toy-dialect` 上继续做）

1. **给 Toy dialect 加一个 `TypeConverter`**：把自定义的 `!toy.tensor<...>` 转成 `memref<...>`，走完整的 `applyPartialConversion` 流程。
2. **实现一个 OpInterface**（例如 `ToyShapeInferenceOpInterface`），写一个通用 shape inference pass，验证「同一个 pass 能作用于多个 dialect 的 op」。
3. **把 Toy 降到 linalg**，再用现成的 `-linalg-tile` / `-convert-linalg-to-loops` 一路降到 `scf` + `memref`，最后到 `llvm` dialect 并用 `mlir-cpu-runner` 跑出结果。**这条端到端链路走通，MLIR 就算过关了。**
4. 给每一步加 lit 测试（已有 lit 基础设施，直接扩展）。

---

### 3.3 IREE 与 HAL

📘 **主学习材料**：[`docs/iree-learning-guide.md`](docs/iree-learning-guide.md) —— 基于官方文档与源码整理的 IREE 核心概念全景 + **HAL 详解**，本阶段以它为准
📄 **论文简记**：[`07-tinyiree.md`](docs/paper-notes/07-tinyiree.md) —— TinyIREE 只给初步印象，非必读
📌 **官方文档**：[IREE HAL Dialect](https://iree.dev/reference/mlir-dialects/HAL/) · [Invocation Execution Model](https://github.com/iree-org/iree/blob/main/docs/website/docs/developers/design-docs/invocation-execution-model.md) · [CUDA HAL Driver](https://iree.dev/developers/design-docs/cuda-hal-driver/)

#### 必须掌握的「核心运行框架」

**这是本阶段最重要的一张图**——它就是「多后端统一运行时」的参考答案：

```
前端（StableHLO / TOSA / Torch / ONNX）
   ↓
linalg on tensors ······················ 结构化算子，可 tiling/fusion
   ↓
flow dialect ··························· 把张量程序切成 dispatch region
   ↓                                      ★ 这一层就是「子图划分」
stream dialect ························· 异步执行、资源与时间线
   ↓                                      ★ 这一层决定「什么时候搬数据」
hal dialect ···························· 设备 / 缓冲 / 命令缓冲 / 可执行体
   ↓                                      ★ 这一层是「硬件抽象」
各后端 codegen（LLVM-CPU / SPIR-V / CUDA / ROCm / VMVX）
   ↓
VM bytecode 或 C 源码
   ↓
Runtime：HAL driver + VM
```

#### 必学核心特性

| # | 知识点 | 为什么必学 |
|---|--------|-----------|
> 下表的每一项在 [`docs/iree-learning-guide.md`](docs/iree-learning-guide.md) 里都有对应章节，学的时候按章对照。

| # | 知识点 | 为什么必学 | 对应章节 |
|---|--------|-----------|---------|
| 1 | **执行模型六概念** ★ | program / module / context / invocation / timeline / fence。核心认知是「invocation 是**排布**工作到时间线，不是**执行**工作」——整个 IREE 栈（含 HAL）都是这套乱序执行模型的复述 | 第 3 章 |
| 2 | **五层 dialect 各自固化了什么决定** ★ | 「渐进式 lowering」在真实系统里的最佳范例。要能说出：flow 固化「算子怎么分组成 dispatch」，stream 固化「执行时序 + 资源生命周期 + 设备归属」，hal 固化「用哪个设备、哪块 buffer、哪个 fence」 | 第 2 章 |
| 3 | **HAL 完整对象模型** ★ | driver / device / allocator / buffer / buffer_view / command_buffer / executable(_cache) / semaphore / fence / event / channel。**这就是我们做算力网设备抽象的词汇表**，要能默画对象关系图 | 4.2–4.5 |
| 4 | **timeline semaphore 语义** ★★ | 64 位单调递增 / signal-before-wait 与 wait-before-signal 都支持 / host↔device 四方向全覆盖 / 同值可多次等待。**这是 HAL 里最硬也最关键的知识点**，必须搞懂为什么 `CUevent` 满足不了以及 IREE 怎么补 | 4.6 |
| 5 | **ExecutableVariant 多目标变体机制** ★ | 一个 `hal.executable` 里放多个 target-specific variant，运行时按 condition 选第一个可用者（fat binary）。**这是「低成本切换后端」的结构基础，也是回应「配置组合爆炸」的现成方案** | 4.7 |
| 6 | **kernel ABI 五个 op** ★ | `hal.interface.binding.subspan` / `constant.load` / `workgroup.id` / `.count` / `.size`。接任何新后端，打通这五个是第一件事 | 4.7.4 |
| 7 | **tensor 边界三个 op** | `hal.tensor.import`(wait fence) / `.barrier`(signal fence) / `.export`。张量世界与 buffer 世界的接缝，也是 fence 绑定资源的落地形态 | 4.9 |
| 8 | **stream-ordered allocation** | 分配/释放本身带 fence、排在时间线上、可在设备侧远程执行。**直接决定峰值显存**，且内存不足时自动串行化而非 OOM | 3.4 / 4.5 |
| 9 | **channel 集合通信** ★ | `iree_hal_channel_t` 对标 `ncclComm_t`/`MPI_Comm`，`split` 可切 TP/PP/DP 子组；collective 与 dispatch 同处一个命令缓冲，**编译器可做计算-通信 overlap**。与我们的项目最直接相关 | 4.8 |
| 10 | **`iree-compile` / `iree-run-module` 关键 flag** | 尤其 `--compile-to=<phase>`——**这是理解 IREE 最高效的单一工具** | 第 5 章 |

#### 先跳过（遇到再学）

- TinyIREE 论文里 MCU / 裸机移植的具体细节、特定 RISC-V 扩展。
- HAL 的 instrument / profile 系列，以及 `pool` / `file` / `IO/Parameters`。
- 各后端 codegen pass 的内部实现（先当黑盒）。
- VM bytecode 的指令格式、emitc / C 源码输出细节（知道存在即可）。
- IREE 的构建系统与 CI 细节（用预编译包即可）。

#### 动手验收

1. 用 `--compile-to=flow/stream/hal` **逐相位 dump IR**，在 `hal` 那份里找出并读懂：`hal.executable`/`variant`/`export` 的嵌套结构、`hal.command_buffer.dispatch`、`hal.device.queue.execute` 的 wait/signal fence、函数边界的 `hal.tensor.import/export`。
2. 用 `iree-compile` 把同一个小模型分别编到 **CPU 和 Vulkan/CUDA** 两个后端，`iree-run-module` 跑出一致结果。
3. `iree-run-module --dump_devices` 看设备能力；同一产物分别用 `local-sync` 与 `local-task` 跑，对比行为差异。
4. 精读 `cuda-hal-driver.md` 的 semaphore 一节，能复述 IREE 用哪几个数据结构补齐 `CUevent` 的能力缺口。
5. 写一个最小的 C 程序调用 IREE runtime 完成推理。

---

## 4. P1 项目详解

### 4.1 TVM

📄 **精读笔记**：[`05-tvm.md`](docs/paper-notes/05-tvm.md) · 📌 [OSDI 2018](https://arxiv.org/abs/1802.04799)

**核心运行框架**：`前端 → 图 IR(Relay/Relax) → 图级优化 → TE → schedule → TIR → CodeGen → Runtime(PackedFunc + graph executor)`

**必学核心特性**：

1. **算子融合的四类算子与融合规则**（injective / reduction / complex-out-fusable / opaque）——**这是「子图划分」问题最经典的解法，必须吃透**。
2. **数据布局（layout）变换与传播**——对应研究问题②「跨后端 layout 兼容」。
3. **TE + schedule 原语**：`split / tile / reorder / fuse / vectorize / unroll / parallel / bind / cache_read / cache_write / compute_at`。要能对着 `tvm.lower(..., simple_mode=True)` 的输出，说清每个原语对循环结构做了什么。
4. **tensorize**——对接 TensorCore 这类张量指令。异构后端接入的关键机制。
5. **AutoTVM / Ansor 的搜索闭环**：schedule 模板 → 搜索空间 → ML cost model → 真机实测 → **tuning log 复用**。**tuning log 复用直接对应研究问题⑥「配置组合爆炸」的缓存策略。**
6. **PackedFunc 统一 ABI 与 graph executor**——理解运行时如何跨语言、跨设备派发。

**先跳过**：VTA/FPGA 硬件细节、XGBoost 的特征工程、Relay pass 全集、TVM 源码目录结构、Relax 的最新演进（知道它解决动态 shape 即可）。

**动手验收**：用 TE 写一个 matmul，给出 naive 与 tiled+cache_write+vectorize 两个 schedule，打印 TIR 对比循环结构差异；再跑一次 AutoTVM 调优并复用 tuning log。

---

### 4.2 ONNX + 多后端委托（ExecuTorch / ONNX Runtime EP）

📌 [ONNX](https://onnx.ai/) · [ExecuTorch `to_backend`](https://pytorch.org/executorch/) · [ORT Execution Providers](https://onnxruntime.ai/docs/execution-providers/)

**为什么它是 P1 而不是 P2**：第五阶段那六个研究问题，**全部是从多后端委托机制里长出来的**。不亲手用一次 partitioner，那六个问题就只是纸上的句子。

**必学核心特性**：

| 主题 | 必须掌握 |
|------|----------|
| ONNX 模型结构 | `ModelProto / GraphProto / NodeProto / TensorProto / ValueInfoProto` 的层次；initializer 与 input 的区别 |
| opset 与版本管理 | opset version 如何决定算子语义；版本转换的坑 |
| shape inference | `onnx.shape_inference.infer_shapes` 能推出什么、推不出什么（动态维度） |
| 图的读写与修改 | 用 `onnx.helper` 手工构图、插入/删除节点、改 initializer。**这是必备的工程手感** |
| ORT EP 机制 ★ | `GetCapability`（EP 声明它能吃下哪些子图）→ `Compile`（编译子图）→ 运行时按分区派发。**这就是「子图划分」的工业接口** |
| ExecuTorch 委托 ★ | `Partitioner` → `to_backend` → `LoweredBackendModule`。理解分区边界处的数据拷贝开销从哪来 |
| 分区带来的代价 | 每个分区边界都可能引入 layout 转换、内存空间切换、同步点——**这正是研究问题①②的根源** |

**先跳过**：ONNX 全部算子的语义（用到再查）、ONNX 训练相关扩展、各具体 EP（TensorRT/OpenVINO/QNN）的内部实现。

**动手验收**：
1. 用 `onnx.helper` 手工构造一个 3 层小图并跑通；再解析一个真实模型，插入一个自定义节点。
2. 在 ORT 里同时启用 CPU EP 和另一个 EP，**打印出实际的分区结果**（哪些节点去了哪个 EP），观察分区边界在哪里、为什么在那里。
3. 用 ExecuTorch 写一个最小 Partitioner，故意只接管部分算子，观察产生了几个 delegate 子图。

---

### 4.3 LLM 负载的真实形态：FlashAttention + PagedAttention

📄 **笔记**：[`08-flash-attention.md`](docs/paper-notes/08-flash-attention.md) · [`09-paged-attention-vllm.md`](docs/paper-notes/09-paged-attention-vllm.md)

**为什么必学**：做「大模型基础设施」却不了解大模型实际怎么算、显存实际被什么占满，设计一定会跑偏。这两篇分别代表**训练侧的访存优化**和**推理侧的内存管理**。

**FlashAttention 必学**：
1. **瓶颈在 HBM 访存而非 FLOPs**——GPU 内存层次（HBM / SRAM / register）的带宽与容量数量级。
2. **online softmax 的重标定公式**（running max \(m_i\) 与 running sum \(\ell_i\) 如何更新）。能手推一遍。
3. **IO 复杂度结论**：标准实现 vs FlashAttention 的 HBM 访问次数（用 \(N, d, M\) 表示）。
4. **反向重计算的取舍**：为什么存 \(O(N)\) 统计量 + 重算，比存 \(O(N^2)\) 矩阵更快。
5. **方法论**：这是「融合 + tiling + 重计算」三件套最有说服力的实例——**它证明了编译器优化的核心杠杆在哪里**。

**PagedAttention / vLLM 必学**：
1. **三类内存浪费**：内部碎片（按最大长度预留）、外部碎片、无法共享。
2. **分页机制**：block table、逻辑块→物理块映射、按需分配、block size 的取舍。
3. **内存共享**：copy-on-write + 引用计数、prefix caching。
4. **调度与抢占**：iteration-level 连续批处理；显存不足时 **swap vs recompute** 两种恢复策略的取舍。
5. **对我们的启示**：**block 化的 KV Cache 是「运行状态低成本迁移」的天然载体**——块粒度迁移比整体迁移便宜得多，这直接连到研究问题④。

**先跳过**：block-sparse 的理论证明、附录全部推导、vLLM 各版本的工程细节。

---

## 5. P2 项目（回填，按需）

### 5.1 LLVM（复习为主）

📘 **主学习材料**：[`docs/llvm-learning-guide.md`](docs/llvm-learning-guide.md) —— 基于官方文档蒸馏的核心链路（四层 IR、New Pass Manager、后端 CodeGen 七阶段、TableGen、MLIR 接缝）
📄 **论文笔记**：[`02-llvm.md`](docs/paper-notes/02-llvm.md)（2004 CGO 论文，讲设计动机与历史）
🛠 **动手项目**：[`llvm-hello-compile/`](llvm-hello-compile/) —— `bash scripts/run.sh`，先读 `out/ANALYSIS.md`
🧭 **阶段导航**：[`docs/README.md` 阶段 0](docs/README.md#阶段-0编译器地基半天1-天可与阶段-1-并行)

**只需确认掌握**：SSA 形式与无限虚拟寄存器、类型系统与显式类型信息、`load/store` 内存模型与 `getelementptr`、New Pass Manager（四层嵌套、`PreservedAnalyses`、analysis 缓存与失效）、别名分析的四种回答、link-time 跨模块优化的意义、前后端解耦。

**回查时优先看这三处**：`poison`/`undef`/`freeze` 语义（写 lowering 时最容易踩 UB）、`noalias`/`align`/`!alias.scope` 这些"决定 LLVM 能帮你多少"的属性、后端七阶段的总图。

**先跳过**：LangRef 全文通读、调试信息与异常处理 metadata、具体 target backend 的编码细节、GC/statepoint。

> 你已经写过 `CountIR` 和 `InjectLogging` 两个 pass，LLVM 的动手门槛已经跨过了。**这一项现在只需要在遇到 MLIR 的 llvm dialect 时回查即可，不要重新投入大块时间。** 唯一值得补的是后端 CodeGen 那一段——2004 年的论文完全没讲，而它决定了"MLIR 交给 LLVM 之后到底发生了什么"。

### 5.2 Halide（1~2 天）

📄 [`04-halide.md`](docs/paper-notes/04-halide.md)

**只学一件事：算法与调度分离（algorithm/schedule decoupling）。** 具体到四个点：
1. 函数式的算法定义为什么让调度获得自由。
2. `compute_at` / `store_at` 控制的「计算粒度 vs 存储粒度」。
3. 循环变换原语：`split / reorder / tile / fuse / unroll / vectorize / parallel`。
4. **核心权衡三角：局部性 ↔ 并行度 ↔ 冗余计算（+ 内存占用）**。

看懂「同一个 blur 算法在不同 schedule 下生成的循环嵌套如何变化」这一组例子，这一项就结束了。**其余（图像处理案例、autoscheduler 内部、bounds inference 的区间分析细节）全部跳过。**

### 5.3 Glow（半天，对比阅读）

📄 [`06-glow.md`](docs/paper-notes/06-glow.md)

**只学三点**：
1. **两级 IR**：高层图 IR（做代数化简与自动微分）与低层指令 IR（做内存分配复用与指令调度）的分工，以及为什么在这里切一刀。
2. **node lowering 到少量线性代数原语**——这是「降低新后端接入成本」的另一条路线，与 TVM 的「TE + 搜索」形成对照。**对我们做多后端很有启发**。
3. **profile-guided 量化** 与 **AOT bundle**。

其余全部跳过。读完后能回答一句话：**「面对一个新硬件后端，Glow 让厂商实现什么？TVM 让厂商实现什么？各自的代价是什么？」**

### 5.4 CUDA fatbin（半天）

**只学三点**：
1. fatbin 的结构：一个二进制里如何同时携带多个架构的代码。
2. **`compute_XX`（PTX，虚拟架构）vs `sm_XX`（SASS，真实架构）** 的区别，以及运行时 JIT 兜底机制。
3. 用 `cuobjdump` / `nvdisasm` 查看一个 `.o` 里到底有哪些架构的代码。

**为什么值得花这半天**：它和 IREE 的 `ExecutableVariant`、和「配置组合爆炸」是同一个问题的三种说法。亲手 dump 一次，「多变体打包 + 运行时选择」这个设计模式就会变成肌肉记忆。

**跳过**：SASS 指令集细节、PTX 汇编编程。

---

## 6. 第五阶段：多后端协同的前沿研究问题

这是路线最终聚焦的研究课题，来自 ExecuTorch 与 ONNX Runtime 多后端委托机制衍生的六个关键挑战。**下面为每个问题补充了「已有工作参照」——即在我们读过的论文里，谁已经部分回答了这个问题。带着这些参照去想，比空想有效得多。**

### 问题 1：子图划分与跨设备传输最小化

> 怎么划分才能减少跨设备传输？

- 不同算子在不同后端上的执行效率不同，子图划分是一个**带通信代价的图分割优化问题**。
- 需考虑设备间带宽（PCIe / NVLink / 网络）、数据搬移开销与计算时间的关系。
- **已有工作参照**：
  - TVM 的算子融合规则（四类算子）——**融合本质上就是一种划分决策**。
  - IREE 的 `flow.dispatch` region 形成——工业系统里的划分实现。
  - Welder 把计算图降到 **tile 级数据流图**，边上标注「复用数据所在的存储层级」，在 tile 级搜索融合组合。**这个建模方式非常值得借鉴**。
  - Alpa 的 inter-operator 划分 + 动态规划。
- **可能的切入点**：图划分算法在 DL 计算图上的适配、基于 cost model 的动态划分、把「存储层级」显式建进图模型。

### 问题 2：跨后端的 Layout、量化格式与内存空间兼容性

> 不同 backend 的 layout、量化格式和内存空间如何兼容？

- 不同后端对张量 layout 要求不同（NCHW vs NHWC vs 自定义），量化方案也不同（per-tensor vs per-channel、对称 vs 非对称）。
- 需要插入 layout / quantization 转换节点，但**转换本身引入开销**。
- **已有工作参照**：
  - TVM 的 layout 变换与传播机制。
  - Campo 用**自动图重写**优化 FP32↔FP16 的 cast 开销——论文明确指出「cast 开销有时会吃掉低精度的收益」。
  - Glow 的 profile-guided 量化与 rescale 优化。
  - IREE 的 `hal.buffer_view` 携带 layout/element type 信息。
- **可能的切入点**：把转换代价纳入划分决策；把转换与计算融合；设计跨后端的统一量化元数据表示。

### 问题 3：动态 Shape 场景的处理

> 动态 batch、序列长度和 KV Cache 怎么处理？

- LLM 推理中 batch size 与序列长度动态变化，静态编译策略失效。
- KV Cache 的内存管理与分配模式特殊。
- **已有工作参照**：
  - **PagedAttention / vLLM** —— 推理侧的完整答案（分页、block table、抢占）。
  - DynaPipe 的变长动态 micro-batching、ByteTransformer 的 padding-free。
  - TVM Relay 的静态假设 → Relax 的动态 shape 演进。
- **可能的切入点**：symbolic shape propagation、动态 shape 下的 JIT 特化策略、KV Cache 感知的图优化、**编译器与运行时的职责边界划分**（哪些决定必须留到运行时）。

### 问题 4：运行状态变化后的低成本后端切换

> 运行状态变化后，能否低成本切换执行后端？

- 设备负载变化、功耗限制、模型热升级、**算力网中节点的进出**，都需要在不同后端间迁移。
- 关键难点是快速迁移中间状态（activations、KV Cache、optimizer states）。
- **已有工作参照**（这是我们最可能出成果的方向）：
  - **Universal Checkpointing**：与并行策略解耦的通用 checkpoint 表示，**可按需在不同并行策略之间转换**。
  - **Oobleck**：预定义 pipeline template，故障时快速实例化新的异构流水线。
  - **Parcae**：三种迁移机制，在不同并行策略间搬运模型状态。
  - **ReaLHF**：在子阶段间重分布参数以切换最合适的并行模式。
  - **PagedAttention 的块化 KV Cache**：块粒度迁移天然比整体迁移便宜。
- **可能的切入点**：设计**统一的、与并行策略和后端都无关的状态序列化格式** + 零拷贝/块粒度迁移机制。

### 问题 5：后端最优计算图的不一致性

> 后端的最优计算图不一样。

- GPU 上的最优融合模式可能不适用于 NPU/DSP，反之亦然。
- 多后端场景下需为同一模型维护多份「最优」计算图，存储与编译开销大。
- **已有工作参照**：
  - Glow 的「少量原语 + node lowering」——**用降低表达能力换取后端接入成本**。
  - IREE 的 `ExecutableVariant`——同一 dispatch 携带多个目标的编译产物。
  - Roller / Triton / Welder 在不同后端上搜出的 kernel 结构本就不同。
- **可能的切入点**：后端无关的中间表示 + 后端特定的 lowering pass；graph rewriting 的增量编译。

### 问题 6：配置组合爆炸问题

> 设备类型、Shape、batch、序列长度、精度和量化方式组合后，变体数量会爆炸。

- 每个维度的取值组合形成笛卡尔积，预编译所有变体不可行。**这是多后端编译最核心的工程挑战。**
- **已有工作参照**：
  - **AutoTVM 的 tuning log 复用 + 迁移学习**——「已经调过的配置不要重复调」。
  - Aceso 的迭代瓶颈缓解（压缩搜索时间）、nnScaler 的「专家加约束缩小搜索空间」、SmartMoE 的「策略池 + 池内低成本切换」。
  - CUDA fatbin / IREE ExecutableVariant 的多变体打包。
  - TACCL / SCCL 把通信算法综合建成 MILP/SMT 问题求解。
- **可能的切入点**：
  - 参数化编译（编译一份代码，运行时特化）。
  - 编译缓存与增量编译。
  - **「常用配置预编译 + 冷配置 JIT」的分层策略**（最现实的工程路线）。
  - 把问题建模为在线学习 / bandit 选择。

---

## 7. 时间安排建议

按每周约 20 小时估算的 12 周计划。**这不是硬性排期，而是一个投入比例的参照**——关键是保证 P0 拿到一半以上的时间。  
每周具体打开哪份文档、跑哪个脚本，跟 [`docs/README.md`](docs/README.md) §2 阶段表对齐即可。

| 周次 | 主题 | 产出物 |
|------|------|--------|
| **W0（可选，0.5～1 天）** | 跑通 `llvm-hello-compile` + 扫 LLVM 学习文档总图 | `out/ANALYSIS.md` 能讲；知道四层 IR 长什么样 |
| **W1-W2** | 分布式训练综述精读 + 动手验证 | 能画六层图；DDP/FSDP 显存实测报告；性能模型脚本 |
| **W3** | 补 SPMD 分片标注（GSPMD / Mesh-TF / SBP 三篇速读） | 一份「同一个并行策略在三种标注体系下怎么写」的对照笔记 |
| **W4-W5** | MLIR 深化：Conversion + Interface + linalg | `mlir-toy-dialect` 先 `all.sh` 过关，再打通 Toy → linalg → scf → llvm |
| **W6-W7** | IREE + HAL | 同一模型编到两个后端跑通；四层 IR 快照对照笔记；C runtime 调用示例 |
| **W8** | TVM | matmul 的两个 schedule 对比 + 一次 AutoTVM 调优 |
| **W9** | ONNX + ORT EP + ExecuTorch 委托 | 打印真实的 EP 分区结果；最小 Partitioner |
| **W10** | FlashAttention + PagedAttention | online softmax 手推；vLLM 关键参数与论文机制的对照表 |
| **W11** | P2 回填：Halide + Glow + CUDA fatbin | 三份短笔记；一次 `cuobjdump` 实操 |
| **W12** | 收束：针对六个研究问题写一份**自己的**问题分析 | 选定 1~2 个方向作为切入点，形成初步方案 |

### 如果只有 4 周（应急最短路径）

0. **半天**：`llvm-hello-compile` 跑通 + 扫 [`docs/llvm-learning-guide.md`](docs/llvm-learning-guide.md) 总图（建立 SSA/Pass 手感，后面学 MLIR 不发飘）。
1. **W1**：分布式综述的 §4（并行策略）+ §6（内存）+ §7（通信）三章，跳过其余；做完 DDP/FSDP 实测。
2. **W2**：`mlir-toy-dialect` 的 `all.sh` + DialectConversion；把 toy 往 linalg 推一刀。
3. **W3**：IREE 全部（[`docs/iree-learning-guide.md`](docs/iree-learning-guide.md)）——这是投入产出比最高的一周。
4. **W4**：ONNX + ORT EP 分区实操；带着六个研究问题回看前三周的笔记。

---

## 8. 学习方法与验收标准

> 分阶段的「读→做→验」清单与卡住时的索引表，见 [`docs/README.md`](docs/README.md) §5～§6。

### 8.1 三条硬性原则

1. **每个项目先画框架图，再抠细节。** 如果画不出「输入是什么 → 经过哪几层 → 每层固化了什么决定 → 输出是什么」，说明还没入门，抠细节是浪费时间。
2. **动手环节不可跳过。** 至少硬门槛两条：`llvm-hello-compile` 跑通并讲清 mem2reg；`mlir-toy-dialect` 跑通并讲清 Dialect Conversion 与 OpInterface。再加上 IREE 双后端、ORT 分区打印、`cuobjdump`——纸面知识才会变成手感。
3. **读论文带着同一个问题**：*「这个设计如何应对『大模型 + 异构多后端 + 动态变化』的场景？它的局限在哪里？」* 每篇笔记的最后一节都在回答这个问题，读完对照检查自己的答案。

### 8.2 阶段性验收（能不能讲清楚）

给自己（或组会）讲清楚下面这几件事，就算通过：

- [ ] 用五分钟讲清「一个 70B 模型是怎么被切到 128 张卡上的」，包括每一维并行切什么、通什么、放哪里。
- [ ] 解释「为什么 MLIR 要分这么多层 dialect，一步降到 LLVM IR 不行吗」（最好能结合 `mlir-toy-dialect` 里 `x*4` 在 toy/low 两层的不同命运）。
- [ ] 画出 IREE 的 `linalg → flow → stream → hal` 四层，说清每层固化了什么决定。
- [ ] 说清「一个算子在 ORT 里从模型文件到某个 EP 上执行，中间经过了哪些决策点」。
- [ ] 用 FlashAttention 为例，论证「融合 + tiling + 重计算」为什么是编译器优化的核心杠杆。
- [ ] 对六个研究问题中的至少两个，说出「已有工作做到了哪一步、还差什么、我可能怎么做」。
- [ ] **动手硬门槛**：`llvm-hello-compile` 的 `out/ANALYSIS.md` 能讲；`mlir-toy-dialect` 的 `--toy-to-low-convert` / `--toy-print-cost` 能讲。

### 8.3 保持跟踪

该领域发展极快（尤其 LLM 训练与推理系统）。建议持续关注 **MLSys / OSDI / SOSP / ASPLOS / NSDI / ISCA** 的相关论文，以及 MLIR、IREE、vLLM 三个项目的 upstream 动态。

---

## 9. 材料索引

> 带「读→做→验」的完整自学路径见 **[`docs/README.md`](docs/README.md)**；论文笔记目录见 [`docs/paper-notes/README.md`](docs/paper-notes/README.md)。

### 论文精读笔记

| # | 笔记 | 论文 | 优先级 | 一句话定位 |
|---|------|------|--------|-----------|
| 01 | [分布式训练综述](docs/paper-notes/01-efficient-training-distributed-infra.md) | Efficient Training of LLMs on Distributed Infrastructures | **P0** | 大模型分布式训练的全景问题地图 |
| 02 | [LLVM](docs/paper-notes/02-llvm.md) | LLVM: A Compilation Framework | P2 | 编译器 IR 与 Pass 基础设施的鼻祖（设计动机与历史；机制见下方专题文档） |
| 03 | [MLIR](docs/paper-notes/03-mlir.md) | MLIR: A Compiler Infrastructure for the End of Moore's Law | **P0** | 可扩展的多层 IR 与渐进式 lowering |
| 04 | [Halide](docs/paper-notes/04-halide.md) | Halide | P2 | 算法与调度分离范式的源头 |
| 05 | [TVM](docs/paper-notes/05-tvm.md) | TVM (OSDI 2018) | P1 | 端到端 DL 编译器 + 自动调优 |
| 06 | [Glow](docs/paper-notes/06-glow.md) | Glow: Graph Lowering Compiler Techniques | P2 | 少量原语 + 多级 lowering 的低成本多后端 |
| 07 | [TinyIREE（简记）](docs/paper-notes/07-tinyiree.md) | TinyIREE (IEEE Micro 2022) | P2 | 只建立"IREE 长什么样"的初步印象；正式材料见下方专题文档 |
| 08 | [FlashAttention](docs/paper-notes/08-flash-attention.md) | FlashAttention | P1 | IO 感知的 kernel 设计范式 |
| 09 | [PagedAttention / vLLM](docs/paper-notes/09-paged-attention-vllm.md) | Efficient Memory Management for LLM Serving | P1 | KV Cache 的分页内存管理与调度 |

### 专题文档与枢纽

| 文档 | 来源 | 优先级 | 一句话定位 |
|------|------|--------|-----------|
| **[自学体系枢纽](docs/README.md)** | 本仓库编排 | — | **日常导航**：每阶段读什么 / 做什么 / 验什么；卡住时的索引表 |
| [IREE 学习文档：核心概念 + HAL 详解](docs/iree-learning-guide.md) | IREE 官方文档 + 主干源码 | **P0** | 老师要求的 IREE HAL 主学习材料：执行模型、dialect 流水线、HAL 全对象模型、timeline semaphore、executable variant、集合通信 |
| [LLVM 学习文档：核心链路 + 概念蒸馏](docs/llvm-learning-guide.md) | LLVM 官方文档主干版本 | P2 | 补足 02 号论文笔记没覆盖的部分：四层 IR 总图、New Pass Manager、**后端 CodeGen 七阶段**、TableGen 与 MLIR ODS 的同源关系 |

### 外部资源

| 资源 | 说明 |
|------|------|
| [LLVM LangRef](https://llvm.org/docs/LangRef.html) | 按需查阅，不必通读。重点：Type System、Poison Values、Intrinsic Functions |
| [LLVM CodeGenerator](https://llvm.org/docs/CodeGenerator.html) | 后端七阶段、SelectionDAG、寄存器分配、MC 层的权威说明 |
| [LLVM NewPassManager](https://llvm.org/docs/NewPassManager.html) | 四层嵌套、analysis 缓存与失效、`opt -passes` 语法 |
| [MLIR 官方文档](https://mlir.llvm.org/docs/) | 重点看 Dialect Conversion、ODS、Interfaces、Linalg 四篇 |
| [IREE HAL Dialect](https://iree.dev/reference/mlir-dialects/HAL/) | **P0 必读**，HAL 的类型与 op 参考 |
| [IREE CUDA HAL Driver 设计文档](https://iree.dev/developers/design-docs/cuda-hal-driver/) | **P0 必读**，唯一讲透"HAL 抽象与真实硬件落差"的文档 |
| [IREE Invocation Execution Model](https://github.com/iree-org/iree/blob/main/docs/website/docs/developers/design-docs/invocation-execution-model.md) | **P0 必读**，timeline / fence / stream-ordered allocation |
| [IREE 部署配置](https://iree.dev/guides/deployment-configurations/) | target backend 与 HAL device 的对应关系 |
| [ONNX 官方](https://onnx.ai/) | 算子集定义、shape inference、graph optimization |
| [ONNX Runtime EP 文档](https://onnxruntime.ai/docs/execution-providers/) | Execution Provider 的注册与分区机制 |
| [ExecuTorch 文档](https://pytorch.org/executorch/) | `to_backend` 与 Partitioner API |
| CUDA fatbin | `cuobjdump` / `nvdisasm` 的手册 + NVCC 编译流程文档 |
