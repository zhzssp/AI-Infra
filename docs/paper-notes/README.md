# 论文精读笔记索引

> **自学体系入口**：[`../README.md`](../README.md)（本目录在整套体系中的位置、每阶段读→做→验）。  
> **总规划 / 优先级 / 验收**：仓库根目录 [`../../README.md`](../../README.md)。  
> **横切词汇表**（把论文与各项目粘成一张图）：[`../learning-guides/ai-compiler-foundations-learning-guide.md`](../learning-guides/ai-compiler-foundations-learning-guide.md)——开局建议先扫总图与概念索引。

本目录为 [`paper/`](../../paper/) 下每篇论文的中文精读笔记。笔记遵循同一套骨架：

1. **它解决什么问题** —— 动机与当时的技术背景
2. **整体运行框架** —— ASCII 流程图 + 每个组件的输入/输出/职责
3. **核心特性逐条拆解** —— 每条按「是什么 → 为什么这样设计 → 带来什么能力」展开
4. **使用示例** —— 可读的代码 / IR / 命令行，逐条注释
5. **关键实验结论** —— 论文数据说明了什么
6. **与本项目的关联** —— 对「算力网上大模型分布式运行基础设施」的启示
7. **最小必要集** —— 必须掌握的点 vs 可以先跳过的内容

> **读法**：时间紧张时，只读**「整体运行框架」+「最小必要集」**两节，其余按需展开。

**骨架的三个例外**（章号与上表不同，按下表跳转）：

| 笔记 | 差异 | 「最小必要集」在哪 |
|------|------|-------------------|
| [01 分布式综述](01-efficient-training-distributed-infra.md) | 综述体例：§3–§9 对应论文各章 | **§12**（框架在 §2） |
| [06 Glow](06-glow.md) · [08 FlashAttention](08-flash-attention.md) | 多一节横向对比 / 后续演进 | **§8** |
| [08](08-flash-attention.md) · [09](09-paged-attention-vllm.md) | **自选补充**，不按骨架逐节读 | 见下方「P1 自选补充」表的「读到哪停」 |
| [07 TinyIREE](07-tinyiree.md) | 刻意写成简记，无「使用示例」「最小必要集」 | 用 [`../learning-guides/iree-learning-guide.md`](../learning-guides/iree-learning-guide.md) §7 代替 |

---

## 按优先级排列

### P0 —— 立即投入，占一半以上时间

| 笔记 | 论文 | 核心收获 |
|------|------|----------|
| [01 分布式训练综述](01-efficient-training-distributed-infra.md) | *Efficient Training of LLMs on Distributed Infrastructures: A Survey*（arXiv 2407.20018） | 五种并行的切分维度与通信原语、16Φ 显存账本、通信-计算重叠、**SPMD 分片标注范式**、容错三条线 |
| [03 MLIR](03-mlir.md) | *MLIR: A Compiler Infrastructure for the End of Moore's Law* | Operation/Region/Block、Dialect、渐进式 lowering。**机制详解见 [`../learning-guides/mlir-learning-guide.md`](../learning-guides/mlir-learning-guide.md)** |

> **IREE 不在这张表里**：老师要求的 IREE HAL 内容不来自某篇论文，而是官方文档与源码。主学习材料是专题文档 [`../learning-guides/iree-learning-guide.md`](../learning-guides/iree-learning-guide.md)，它同样是 **P0**。
>
> 同理，[`../learning-guides/llvm-learning-guide.md`](../learning-guides/llvm-learning-guide.md) 是从 LLVM 官方文档蒸馏的专题文档（P2），补足 02 号笔记没覆盖的后端 CodeGen 与现代 Pass 基础设施。

### P1 —— 紧随其后

| 笔记 | 论文 | 核心收获 |
|------|------|----------|
| [05 TVM](05-tvm.md) | *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning*（OSDI 2018） | 四类融合、layout、TE+schedule、tensorize、AutoTVM。**工程机制见 [`../learning-guides/tvm-learning-guide.md`](../learning-guides/tvm-learning-guide.md)** |

#### P1 自选补充 —— 两篇负载侧论据（不在主链上）

> 08 / 09 是额外补充的论文，**没有配套动手项目**，也不构成一条技术栈。核心结论已蒸馏进 [`../learning-guides/ai-compiler-foundations-learning-guide.md`](../learning-guides/ai-compiler-foundations-learning-guide.md) §4.3 / §9.4 / §9.5——**只读 foundations 也不断链**。要读就按下表的身份穿插，不要按「精读七节骨架」逐节啃。

| 笔记 | 论文 | 以什么身份融入 | 读到哪停 |
|------|------|----------------|----------|
| [08 FlashAttention](08-flash-attention.md) | *FlashAttention* | **优化杠杆的实证**：进 TVM/schedule 之前建立「为什么要 tiling」的动机 | 瓶颈是 HBM 访存不是 FLOPs；融合+tiling+重计算怎么配合。公式看懂即可，**不要求手推** |
| [09 PagedAttention / vLLM](09-paged-attention-vllm.md) | *Efficient Memory Management for LLM Serving with PagedAttention*（SOSP 2023） | **研究问题的样本**：根 README §6 问题③（编译期 vs 运行时边界）与④（低成本迁移） | 三类浪费、block table 分页；**重点是「与编译器视角的关系」一段** |

### P2 —— 回填，按需

| 笔记 | 论文 | 核心收获 |
|------|------|----------|
| [02 LLVM](02-llvm.md) | *LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation* | SSA 与类型系统、Pass 基础设施、link-time 优化、前后端解耦。**这篇讲的是 2004 年的设计动机；今天 LLVM 的实际结构见 [`../learning-guides/llvm-learning-guide.md`](../learning-guides/llvm-learning-guide.md)** |
| [04 Halide](04-halide.md) | *Halide* | **算法与调度分离**、compute_at/store_at、局部性 ↔ 并行度 ↔ 冗余计算的权衡三角 |
| [06 Glow](06-glow.md) | *Glow: Graph Lowering Compiler Techniques for Neural Networks* | 两级 IR 分工、node lowering 到少量线性代数原语、profile-guided 量化、AOT bundle |
| [07 TinyIREE（简记）](07-tinyiree.md) | *TinyIREE*（IEEE Micro 2022） | 只为建立 IREE 的初步印象：dispatch region 是核心调度单位、"缩小"靠替换运行时组装方式而非砍编译能力。**深入内容见 [`../learning-guides/iree-learning-guide.md`](../learning-guides/iree-learning-guide.md)** |

---

## 阅读路径建议

```
开局：../learning-guides/ai-compiler-foundations-learning-guide.md（总图 + 公共词汇）
                │
                ▼
01 分布式训练综述  ──┐
                     ├──→  理解「要解决什么问题」
（可选）04 Halide ───┘      —— Halide 只看「算法/调度分离」这一节

03 MLIR  ──→  IREE 专题文档  ──→  05 TVM
   │          （../learning-guides/iree-learning-guide.md）  │
   │             │  ↑ 先花 20 分钟扫一遍     └─ 图级优化与自动调优的完整范式
   │             │    07 TinyIREE 建立印象
   │             └─ 多后端统一运行时的工业样板（本路线的核心参考）
   └─ 所有现代 ML 编译栈的公共基础设施

（自选补充，旁挂，不占主链）
08 FlashAttention ─┐ 进 05 TVM 之前扫结论：为什么值得做融合/tiling
09 PagedAttention ─┘ 收束时读：研究问题③④的样本（编译期 vs 运行时 / 块化迁移）

02 LLVM  +  06 Glow
   └─ 遇到具体问题时回查即可
      LLVM 的机制细节走 ../learning-guides/llvm-learning-guide.md，
      02 号笔记只回答「当年为什么这样设计」
```

- **完整自学路径**（读什么 → 做什么 → 验什么）：[`../README.md`](../README.md)
- **横切前置概念**：[`../learning-guides/ai-compiler-foundations-learning-guide.md`](../learning-guides/ai-compiler-foundations-learning-guide.md)
- **优先级、必学/跳过、时间表、研究问题**：仓库根目录 [`../../README.md`](../../README.md)
- **动手项目**：[`../../llvm-hello-compile/`](../../llvm-hello-compile/) · [`../../mlir-toy-dialect/`](../../mlir-toy-dialect/) · [`../../tvm-fatbin-lab/`](../../tvm-fatbin-lab/) · [`../../onnx-delegate-lab/`](../../onnx-delegate-lab/)
