# AI-Infra

This repo is for AI Infra learning, including a variety of resources.

---

## 学习路线：AI 大模型负载多 AI 后端编译

### 路线总览

该学习路线围绕 **"AI 大模型负载在多后端上的高效编译与执行"** 这一核心命题展开，从分布式训练基础 → 编译器基础设施 → 图级 DL 编译器 → 运行时部署 → 多后端协同的前沿问题，形成了一条逐层递进、理论与实践并重的知识链路。

```
分布式训练基础 ──→ 编译器基础设施（LLVM → MLIR）──→ 图级 DL 编译器（TVM/Glow/Halide）
                                                            │
                                                            ↓
                           多后端协同前沿问题 ←── 运行时与部署（IREE/CUDA fatbin/ONNX/ExecuTorch）
```

---

### 第一阶段：分布式训练基础

| 资源 | 说明 |
|------|------|
| [Efficient Training over Distributed Infra](https://arxiv.org/abs/2407.20018) | 分布式训练综述，覆盖数据并行、模型并行、流水线并行、ZeRO、FSDP 等主流策略，是理解大模型训练中通信与计算协同的起点 |

**学习目标**：理解大模型训练为何需要多卡/多机协同，掌握 SPMD 编程模型的基本概念，为后续理解"编译后的计算图如何映射到多设备"打下基础。

---

### 第二阶段：编译器基础设施

| 资源 | 说明 |
|------|------|
| [LLVM LangRef](https://llvm.org/docs/LangRef.html) | LLVM IR 语言参考手册，掌握 SSA 形式、类型系统、基本块、控制流等编译器 IR 的核心概念 |
| **MLIR：实现一个非常小的 MLIR Dialect** | 动手实践环节——通过亲手写一个 Dialect（定义 Ops、Types、Lowering Pass），真正理解 MLIR 的"多层 IR、渐进式 lowering"设计哲学 |
| [MLIR: A Compiler Infrastructure for the End of Moore's Law](https://arxiv.org/abs/2002.11054) | MLIR 论文，阐述为何需要可扩展的编译器 IR，以及 Dialect 机制如何解决领域特定编译器的碎片化问题 |
| [Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing](https://dl.acm.org/doi/10.1145/3150211) | 将算法（algorithm）与调度（schedule）解耦的经典工作，TVM 的设计思想直接来源于此 |

**学习目标**：
- LLVM：建立编译器 IR 的系统认知（SSA、Pass Manager、CodeGen 流程）
- MLIR 动手实践：理解 Dialect、Pass、Conversion 的核心 API，亲手写一个端到端的 lowering pipeline
- MLIR 论文：理解 "Progressive Lowering" 的设计动机和应用场景
- Halide：理解 "计算与调度分离" 范式，这是 TVM 乃至所有 DL 编译器的核心设计理念

> **为什么 LLVM → MLIR → 动手写 Dialect 的顺序很重要**：
> MLIR 的很多概念（Operation、BasicBlock、Region、SSA）直接继承自 LLVM。先看 LLVM LangRef 建立 IR 基础认知，再读 MLIR 论文理解其扩展性诉求，最后动手实现一个小 Dialect 完成从"知道"到"会用"的跨越。这三个步骤不可跳过。

---

### 第三阶段：图级深度学习编译器

| 资源 | 说明 |
|------|------|
| [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://arxiv.org/abs/1802.04799)（OSDI 2018） | 端到端 DL 编译器的里程碑工作，提出 Relay IR + TE + AutoTVM 三层架构 |
| [Glow: Graph Lowering Compiler Techniques for Neural Networks](https://arxiv.org/abs/1805.00907) | Meta 的 DL 编译器，重点理解其多级 IR lowering 策略和 ahead-of-time 编译方案 |

**学习目标**：
- TVM：理解 Relay（高层图 IR）→ TE/TIR（张量级 IR）→ CodeGen 的完整 lowering pipeline，以及 AutoTVM/AutoScheduler 的自动调优思路
- Glow：与 TVM 形成对比学习，理解不同编译器在 IR 设计和 lowering 策略上的取舍
- Halide（归入第二阶段但在此深化）：理解 compute/schedule 分离在此阶段如何落地为 TVM 的 TE + schedule primitive

---

### 第四阶段：运行时与部署

| 资源 | 说明 |
|------|------|
| [IREE HAL](https://iree.dev/reference/mlir-dialects/HAL/) | IREE 的硬件抽象层 Dialect，理解如何用 MLIR 表达 device、buffer、command buffer 等硬件概念 |
| **CUDA fatbin** | 理解 CUDA 的 fat binary 格式——一个二进制如何包含针对多种 GPU 架构（sm_80/sm_90 等）的代码，以及运行时如何选择最优版本 |
| **ONNX 项目** | 掌握 ONNX 作为模型交换标准的算子集定义、版本管理、shape inference 和 graph optimization 机制 |
| **ExecuTorch & ONNX Runtime 多后端委托** | 两个主流的运行时多后端方案：ExecuTorch 的 `to_backend` API 和 ONNX Runtime 的 Execution Provider 机制 |

**学习目标**：
- IREE：理解 MLIR-based 编译器如何从 high-level ML framework 一路 lowering 到具体硬件后端
- CUDA fatbin：具备分析和理解 GPU 二进制的能力，为后端的性能调优和 debug 打基础
- ONNX：具备分析和修改 ONNX 模型图的能力
- ExecuTorch / ONNX Runtime：理解运行时如何将统一的计算图"拆开"分发给不同后端执行

---

### 第五阶段：多后端协同的前沿研究问题

这是本路线最终聚焦的核心研究课题，来自 ExecuTorch 和 ONNX Runtime 多后端委托机制衍生的 6 个关键挑战：

#### 1. 子图划分与跨设备传输最小化

> 怎么划分才能减少跨设备传输？

- 不同算子在不同后端上的执行效率不同，子图划分是一个带通信代价的图分割优化问题
- 需考虑设备间带宽（PCIe/NVLink/网络）、数据搬移开销与计算时间的关系
- 可能的切入点：图划分算法（如 Kernighan–Lin、Spectral Partitioning）在 DL 计算图上的适配，或基于 cost model 的动态划分策略

#### 2. 跨后端的 Layout、量化格式与内存空间兼容性

> 不同 backend 的 layout、量化格式和内存空间如何兼容？

- 不同后端对张量 layout 的要求不同（NCHW vs NHWC vs 自定义 layout），量化方案也不同（per-tensor vs per-channel，对称 vs 非对称）
- 需要插入 layout/quantization 转换节点，但转换本身引入开销
- 可能的切入点：在划分决策中纳入转换代价，或将转换与计算融合（如 FuseLayoutTransform）

#### 3. 动态 Shape 场景的处理

> 动态 batch、序列长度和 KV Cache 怎么处理？

- LLM 推理中 batch size 和序列长度动态变化，静态编译策略失效
- KV Cache 的内存管理和分配模式特殊（如 PagedAttention）
- 可能的切入点：动态 shape 下的 just-in-time compilation 策略、symbolic shape propagation、以及 KV Cache 感知的计算图优化

#### 4. 运行状态变化后的低成本后端切换

> 运行状态变化后，能否低成本切换执行后端？

- 设备负载变化、功耗限制、模型热升级等场景需要在不同后端间迁移
- 关键难点是如何快速迁移中间状态（activations、KV Cache、optimizer states）
- 可能的切入点：设计统一的状态序列化格式与零拷贝迁移机制

#### 5. 后端最优计算图的不一致性

> 后端的最优计算图不一样。

- GPU 上的最优融合模式可能不适用于 NPU/DSP，反之亦然
- 多后端场景下需要为同一模型维护多份"最优"计算图，存储和编译开销大
- 可能的切入点：设计"后端无关"的中间表示 + 后端特定的 lowering passes，或探索 graph rewriting 的增量编译方案

#### 6. 配置组合爆炸问题

> 设备类型、Shape、batch、序列长度、精度和量化方式组合后，变体数量会爆炸

- 每个维度的取值组合形成笛卡尔积，预编译所有变体不可行
- 这是多后端编译最核心的工程挑战之一
- 可能的切入点：
  - 参数化编译（parameterized compilation）——编译一份代码，运行时特化
  - 编译缓存与增量编译策略
  - 基于 profiling 的"常用配置预编译 + 冷配置 JIT"的分层策略
  - 将问题建模为在线学习或 bandit 选择问题

---

### 学习建议

1. **按阶段循序渐进**：第一阶段到第四阶段是知识积累，第五阶段的研究问题需要在实践中反复思考
2. **动手实践是关键**：MLIR Dialect 实现、ONNX 模型解析与修改、写一个简单的 CUDA kernel 并查看 fatbin，这些动手环节能大幅加深理解
3. **论文阅读要带着问题**：读每个论文时都问自己——"这个设计如何应对大模型 + 多后端的场景？它的局限在哪里？"
4. **保持对前沿论文的跟踪**：该领域发展极快（尤其 LLM 推理优化），建议持续关注 MLSys、OSDI、ASPLOS、ISCA 等会议的论文
