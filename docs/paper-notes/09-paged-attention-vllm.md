# PagedAttention / vLLM：用操作系统的分页思想管理 LLM 推理的 KV Cache

> **导航**：[笔记索引](README.md) · [自学枢纽](../README.md)（阶段 5） · [横切概念](../ai-compiler-foundations.md) §9.5（分页状态 / 块化迁移）  
> **配套阅读**：本篇是研究问题④「运行状态低成本迁移」最直接的技术参照——**块化的状态才搬得动**；训练侧的访存优化对照见 [`08-flash-attention.md`](08-flash-attention.md)。

> 论文元信息：*Efficient Memory Management for Large Language Model Serving with PagedAttention*
> Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
> UC Berkeley / Stanford University / UC San Diego，SOSP 2023
> arXiv: https://arxiv.org/abs/2309.06180　代码：https://github.com/vllm-project/vllm

## 1. 它解决什么问题

LLM 推理服务要提高吞吐（throughput），本质上要靠**把足够多的请求 batch 在一起**跑，摊薄模型权重加载和 kernel 启动的固定开销。但 batch 能做多大，不是被算力（FLOPS）卡住的，而是被显存卡住的——具体说，是被 **KV Cache** 卡住的。

在一张 40GB 的 A100 上跑 13B 模型，约 65% 显存是常驻的模型参数，30% 多是 KV Cache，剩下一小部分是激活值（activation）。参数大小是固定的，激活值也很小，所以**谁能把 KV Cache 管理得更省、更满，谁就能塞进更多并发请求，谁的吞吐就更高**。这就是本文的核心战场。

KV Cache 之所以难管理，是因为它和传统深度学习里的张量完全不同：

- **大**：以 13B 的 OPT 模型为例，单个 token 的 KV Cache 需要 800KB（`2(K/V) × 5120(hidden) × 40(layers) × 2(bytes, FP16)`），一条 2048 token 的请求就要 1.6GB。几十 GB 显存也只能撑几十个并发请求。
- **动态增长、生命周期不可预知**：请求要生成多少 token 事先不知道，KV Cache 是随着 decode 一步步"长出来"的，且什么时候结束也不知道。

而现有系统（论文对比的 FasterTransformer、Orca）为了满足深度学习框架"张量必须连续存储"的要求，**按请求的最大可能长度预先分配一段连续显存**。这带来三类浪费，论文用 Fig.2 做了量化（见下表，数值为占 KV Cache 总空间的百分比）：

| 浪费类型 | 成因 | Orca(Max) | Orca(Pow2) | Orca(Oracle) | vLLM |
|---|---|---|---|---|---|
| 实际 token 状态利用率 | —— | 20.4% | 30.3% | 38.2%（近似，见下） | **96.3%** |
| **预留（reservation）** | 已知未来会用但当前空着，为将来 token 保留 | 高 | 中 | 低 | 几乎无 |
| **内部碎片（internal fragmentation）** | 按最大长度预分配，实际长度远小于最大长度，永远用不到的部分 | 高（如 57.3%） | 中 | 低 | 几乎无 |
| **外部碎片（external fragmentation）** | 不同请求预分配的块大小不同，allocator（如 buddy allocator）留下的、谁都用不上的空隙 | 有 | 有 | 有 | 无（等大块） |

论文原始结论：**现有系统里真正用来存 token 状态的显存只有 20.4%~38.2%**，其余全被这三类浪费吃掉；vLLM 把有效利用率提升到 **96.3%**。这不是"锦上添花"的优化，而是数量级的差距——直接决定了能塞进 batch 的请求数量，也就直接决定吞吐。

第二个问题是**无法共享**：并行采样（parallel sampling）、beam search 等解码算法本质上会让同一个请求产生多个共享前缀的序列，理论上这些序列可以共享一部分 KV Cache。但现有系统把每个序列的 KV Cache 存成独立的连续大块，物理上没法共享，只能各自占一份完整拷贝。

**prefill 与 decode 两阶段的特征差异**（这是理解一切调度设计的前提）：

- **Prefill（prompt 阶段）**：一次性把整条 prompt 的所有 token 喂进去，用矩阵乘矩阵（GEMM）并行计算出所有位置的 K/V 和第一个输出 token 的概率分布。这一阶段**compute-bound**，能充分利用 GPU 并行度。
- **Decode（自回归生成阶段）**：每一步只喂 1 个新 token（上一步采样出来的），用矩阵乘向量（GEMV）计算下一个 token 的概率，且这一步依赖前面所有步的 KV Cache——不能跨步并行。这一阶段**memory-bound**，GPU 算力严重欠打满，是单请求延迟的主要来源，也是"为什么必须靠 batching 多个请求的 decode 步骤来提高吞吐"的原因。

正是因为 decode 阶段要靠 batching 撑吞吐，而 batch 能开多大又被 KV Cache 显存卡住，"如何最省、最灵活地管理 KV Cache"就成了 LLM 推理服务基础设施的核心问题——这正是本文要解决的。

## 2. 整体运行框架

vLLM 采用**中心化调度（centralized scheduler）+ 分布式 GPU worker** 的架构：

```
                        ┌─────────────────────────────────────────┐
                        │              请求队列 (waiting)           │
                        │   Request(prompt, SamplingParams, ...)   │
                        └───────────────────┬───────────────────────┘
                                            │ FCFS 排队
                                            ▼
                        ┌─────────────────────────────────────────┐
                        │             Scheduler（每个 iteration）    │
                        │  - 决定这一步 batch 里放哪些序列(FCFS)     │
                        │  - 判断显存是否够, 触发抢占/换入换出        │
                        │  - 为新 token 申请逻辑块 -> 物理块         │
                        └───────────────────┬───────────────────────┘
                                            │ 控制消息: token ids + block table
                                            ▼
                        ┌─────────────────────────────────────────┐
                        │            KV Cache Manager               │
                        │  - block table: 逻辑块 -> 物理块 映射      │
                        │  - GPU Block Allocator / CPU Block Allocator│
                        │  - 引用计数 / copy-on-write 判定            │
                        └───────────────────┬───────────────────────┘
                                            │ broadcast 到各 GPU worker
                                            ▼
        ┌───────────────┬───────────────────┬───────────────────┐
        │  Worker 0      │   Worker 1        │  ...  Worker N-1  │
        │ Model Shard 0  │  Model Shard 1     │      Model Shard  │
        │ Cache Engine 0 │  Cache Engine 1    │      Cache Engine │
        │ (物理 KV block) │  (物理 KV block)    │                   │
        └───────┬───────┴─────────┬─────────┴───────────┬───────┘
                │ tensor-parallel all-reduce（worker 间直接同步，无需过 scheduler）
                ▼
        ┌─────────────────────────────────────────┐
        │       PagedAttention CUDA kernel          │
        │  按 block table 间接寻址, 逐块读取 K/V,     │
        │  计算注意力, 写回新生成的 K/V 到物理块       │
        └───────────────────┬───────────────────────┘
                            ▼
                     采样出新 token，回传给 Scheduler
                     （继续下一 iteration，或结束释放显存）
```

**各组件职责**：

- **Scheduler**：全局唯一，每个 **iteration（解码步）**做一次决策——这是 Orca 提出的 *iteration-level scheduling* 思路的延续：不是"整批请求一起等到全部完成才换下一批"，而是每一步都可以有请求完成退出、有新请求加入。调度粒度精细到"一次前向传播"，所以排队延迟只是"等一步"，不是"等一整个 batch 跑完"。
- **KV Cache Manager**：维护 block table（逻辑块到物理块的映射）与 GPU/CPU 两个 block allocator。它决定"这一步谁能拿到新的物理块""谁的物理块要被抢占换出"。它是纯粹的**元数据管理层**，不接触真实的 K/V 数据搬运。
- **分布式 Worker**：真正持有模型权重分片（tensor parallel shard）和物理 KV block 数据（cache engine）。多个 worker 共享同一份 scheduler 广播下来的 block table，各自在自己持有的 attention head 子集上执行 PagedAttention kernel，然后通过 all-reduce 同步中间结果——这一步不需要 scheduler 介入，worker 间点对点同步。
- **PagedAttention kernel**：真正的计算发生地，接收 block table 作为寻址依据，把"取 K/V"从"读一段连续内存"变成"按 block table 查表 + 逐块读取再拼接"。

**与操作系统虚拟内存/分页的类比**：

| OS 虚拟内存 | vLLM |
|---|---|
| 进程（process） | 请求的一个序列（sequence） |
| 字节（byte） | token |
| 虚拟页 / 逻辑页（logical page） | 逻辑 KV block（logical KV block） |
| 物理页（physical page） | 物理 KV block（physical KV block） |
| 页表（page table） | block table |
| 按需分页（demand paging） | 按需分配物理块，最后一块之前的空间从不浪费 |
| fork 时的 copy-on-write | 并行采样 / beam search 下的 copy-on-write |
| 换出到磁盘 swap 区 | 换出到 CPU RAM |

这个类比在大部分设计动机上是成立的，但**有两处需要改造，不能照搬 OS 的做法**：

1. **OS 的页面置换是"猜哪个页面未来最久不会被访问"（如 LRU），而 vLLM 采用 all-or-nothing 的整序列换入换出**——因为一个请求的所有 block 在下一次被处理时是"整批一起访问"的（一次 attention 计算要用到该序列全部历史 KV），细粒度地只换出部分页面没有意义，反而增加复杂度。
2. **OS 换出页面后只能从磁盘换回原数据；vLLM 除了 swap 还有"recomputation（重算）"这个 OS 里不存在的恢复手段**——因为 vLLM 知道生成过程的语义：被抢占序列的 prompt + 已生成 token 可以拼成一个新 prompt，用一次 prefill 重新算出全部 KV Cache，往往比等待 PCIe 传输更快。这是"应用语义反哺系统设计"的典型例子。

## 3. 核心机制逐条拆解

### 3.1 PagedAttention kernel

**是什么**：把一个序列的 KV Cache 切分成固定大小（block size `B`）的 **KV block**，每块存 `B` 个 token 在某一层、某个注意力头上的 key/value 向量。attention 计算被重写成分块累加的形式：

```
A_ij = softmax_j( q_i^T · K_j / sqrt(d) )   # 对第 j 个 block 内的 key 计算注意力分数
o_i  = Σ_j  V_j · A_ij^T                     # 对各 block 的贡献累加，得到最终输出
```

kernel 在执行时，不再假设一个序列的所有 K/V 存在一段连续显存里，而是拿着这个序列的 block table，**逐块查出物理地址再去读**，读到之后照常做 QK^T、softmax、加权求和。

**为什么这样设计**：深度学习框架的张量运算默认要求"连续内存",但 KV Cache 天生动态增长、长度不可预知，"连续"和"动态"是根本冲突的。分块之后，只要 kernel 认识"块的间接寻址"，物理内存布局就可以完全打散，管理层可以像 OS 管理物理页一样自由分配/回收/共享这些块。

**带来什么能力**：这是整篇论文所有其它机制（按需分配、共享、swap、抢占）的**地基**——没有 PagedAttention 提供"非连续存储也能算注意力"的能力，后面所有内存管理技巧都无从谈起。

**block size 的取舍**：block 越小，kernel 内并行处理的 token 越少，读取和 launch 开销占比越高，GPU 利用率越低；block 越大，最后一块里"预留但未填满"的内部碎片就越大，且共享粒度变粗（两个序列要共享，就必须整块完全一致才能共享，块越大越难对齐），可共享性下降。论文的 ablation（§7.2）显示 ShareGPT 数据集下 block size 16~128 都表现较好，但 Alpaca 这种短序列数据集下大 block size 明显掉性能（序列本身比 block 还短，内部碎片占比飙升）。**vLLM 默认 block size = 16**，是"kernel 效率"与"碎片/共享粒度"之间的折衷点。

### 3.2 Block table 与按需分配

**是什么**：block table 记录一个序列的逻辑块序号 → 物理块编号的映射，以及该块当前"已填充了多少个 token（#filled）"。块从左到右顺序填充，只有当前块填满才会去申请下一个新物理块。

**为什么这样设计**：与"预先按最大长度分配一大段连续显存"相反，vLLM 只在真正需要的时候才伸手要显存——prompt 有多少 token 就分配多少块，decode 每生成够 `B` 个新 token 才多申请 1 块。

**带来什么能力**：内部碎片被压缩到**最多一个 block 的大小**（也就是最后那个未填满的块），无论请求的最大长度设多大，浪费的上限都是固定的、很小的一块，而不是随最大长度线性增长。同时因为所有物理块大小完全一致，**外部碎片被彻底消除**——分配器永远只需要找"一个空闲块"，不存在"大小不匹配拼不出连续空间"的问题。这正是 Fig.2 中 vLLM 96.3% 利用率的来源。

### 3.3 内存共享：copy-on-write 与引用计数

**是什么**：当多个逻辑序列（同一请求的不同采样结果、beam search 的不同候选、共享前缀的不同请求）的某些逻辑块映射到**同一个物理块**时，给该物理块记一个**引用计数（reference count）**。只要没人要往这个块里写新东西，大家可以一直共享读取；一旦某个序列要往一个引用计数 > 1 的块里追加新 token，就触发 **copy-on-write**：先申请一个新物理块，把旧内容拷过去，引用计数减 1，再在新块上写入——这跟 OS 里 `fork()` 之后父子进程共享物理页、直到某一方要写才真正拷贝的机制完全一致。

具体场景：

- **并行采样（parallel sampling）**：多个输出共享同一份 prompt 的 KV，prompt 阶段只存一份物理块（引用计数 = 采样数）；到 decode 阶段各自采样出不同 token，只有当某个采样序列写到"仍与别人共享"的最后一块时才 copy-on-write。论文实验：这带来 6.1%~9.8%（Alpaca）/16.2%~30.5%（ShareGPT）的显存节省。
- **Beam search**：不仅共享 prompt，解码过程中各 beam 候选之间共享的块也会随着 top-k 剪枝动态变化——某个候选被淘汰，其独占的物理块引用计数归零即释放；新分裂出来的候选复用父节点的共享块。这类似 OS 里"多次 fork 形成的进程树"。共享粒度比并行采样更细、更动态，节省也更大：37.6%~55.2%（Alpaca）/44.3%~66.3%（ShareGPT）。
- **共享前缀（prefix sharing / shared prompt）**：多个不同用户请求共享同一段系统提示词（system prompt）或 few-shot 示例前缀时，服务方可以提前把这段前缀的 KV Cache 算好并常驻，新请求直接把自己的逻辑块指向这些预先计算好的物理块（最后一块标记为 copy-on-write），只需要对用户自己那部分输入做 prefill。这是现代 vLLM `enable_prefix_caching` 的思想来源。

**为什么这样设计**：连续内存布局下，共享是不可能的——两个序列的 KV 存在两段独立的连续空间里，没有"部分重叠"的表达能力。分块之后，"共享"退化成"两个逻辑块指向同一个物理块编号"，是一个纯元数据操作，不需要拷贝任何真实数据。

**带来什么能力**：把本来会被完整复制 N 份的 prompt/前缀 KV Cache，压缩成 1 份物理存储 + N 份轻量元数据指针，尤其在长 prompt、多输出候选场景下（如程序生成的多候选补全、机器翻译的 beam search）显存节省非常可观，且因为不需要真实数据拷贝（除了 copy-on-write 触发的那一小块），几乎没有额外计算开销。

### 3.4 调度与抢占（preemption）

**是什么**：vLLM 采用**先来先服务（FCFS）**策略保证公平性——最先到达的请求最先被处理，抢占时最后到达的请求最先被踢出去。调度发生在**每个 iteration**（迭代级调度，继承自 Orca 的思路）：每一步决定"这一批 batch 里放哪些序列"，处理完就有请求可能完成退出、腾出空间给排队中的新请求进入,不需要等整批全部跑完。

当显存耗尽、无法再给正在运行的序列分配新物理块时，触发**抢占**：

- **驱逐单位是整个 sequence group**（一个请求下所有相关序列，比如一个 beam search 的所有候选）而不是单个 block——因为处理一个序列必须访问它的所有历史 KV，"all-or-nothing"驱逐，不做细粒度页面置换算法（不需要猜"未来最久不用的页"）。
- **恢复手段一：Swapping（换出到 CPU 内存）**。把被驱逐序列的物理块整体拷贝到 CPU RAM（对应 KV Cache Manager 里的 CPU Block Allocator），显存腾出来给别的请求；被抢占的请求完成排队后再整块拷回 GPU。由于 CPU 侧占用的块数不会超过 GPU 侧总块数，CPU RAM 的 swap 空间是有界的。**代价是 PCIe 传输带宽**，且实验显示 block size 越小，swap 单次传输的数据块越碎，有效带宽越低，开销越大。
- **恢复手段二：Recomputation（重算）**。直接丢弃被驱逐序列的 KV Cache，重新调度时把"原始 prompt + 已经生成的 token"拼成一个新的 prompt，走一次 prefill 重新并行算出所有位置的 KV Cache。因为 prefill 阶段是并行的 GEMM，往往比逐 token decode 生成同样长度的序列快得多，所以"重算"的代价可能远低于"从头生成"的代价。且它的开销和 block size 无关（不依赖 KV block 本身，直接从 token id 重算）。
- **取舍**：小 block size 时 recomputation 更划算（swap 因为传输太碎而效率低）；大 block size 时二者接近，swap 略占优；论文实测 recomputation 的开销从不超过 swapping 延迟的 20%（在小 block size 下），中等 block size（16~64）二者端到端性能相近。这给了系统设计者一个简单的经验法则：**block size 较小时优先选 recomputation，block size 较大或 PCIe 带宽充足时可以考虑 swapping**。

### 3.5 分布式执行：tensor parallelism 下的共享 block table

**是什么**：vLLM 支持 Megatron-LM 式张量并行——线性层按矩阵分块，attention 层按注意力头切分到不同 GPU，各 GPU（SPMD 进程）处理相同的输入 token，但只负责一部分注意力头，计算完通过 **all-reduce** 同步中间结果。

**为什么这样设计**：因为所有 GPU shard 处理的是**同一批 token、同一组序列位置**，只是隐藏维度上切开了,所以它们需要的 KV Cache 逻辑结构（block table）是完全一样的——只是每个 worker 真正存储的物理数据只是自己那部分 attention head 的 K/V。于是 vLLM 让**调度器维护单一、全局的 KV Cache Manager**，把同一份 block table 广播给所有 worker；每个 worker 收到 token ids + block table 之后自己去读/写自己那部分物理块，计算完了通过 NCCL all-reduce 直接互相同步，**不需要经过 scheduler**。

**带来什么能力**：内存管理逻辑与模型并行执行逻辑完全解耦——scheduler 只需要算一次"分配/换出/共享"的决策，就能同时驱动所有 GPU shard，worker 之间在一个 iteration 内也不需要为了内存管理而额外同步，通信开销只在 all-reduce 这一处,把分布式场景下的内存管理复杂度做到了最低。

### 3.6 与其它优化的组合

论文自己提到的组合/边界情况（§5.1, §8）：

- **kernel 融合**：为了压低 PagedAttention 引入的间接寻址开销，vLLM 实现了 (1) fused reshape + block write（新 K/V 生成后直接 reshape 成 block 布局再按 block table 写入，合并成一个 kernel）；(2) fused block read + attention（读 block table、取 K/V、算 attention 融合，每个 GPU warp 负责读一个 block 以保证 coalesced memory access）；(3) fused block copy（copy-on-write 的多个碎片化 block 拷贝合并成一次 kernel launch，避免大量小的 `cudaMemcpyAsync` 调用）。测下来间接寻址导致 attention kernel 本身延迟增加 20%~26%（相对 FasterTransformer 手写 kernel），但因为只影响 attention 算子，端到端仍大幅领先。
- **FlashAttention**：论文在相关工作中明确指出 FlashAttention 是"用 tiling 和 IO 感知的 kernel 优化降低 attention 计算时的峰值显存与 IO 开销"，与 PagedAttention 是**互补而非竞争**关系——一个管"怎么存"，一个管"怎么算得更快更省中间内存"；现实中的 vLLM 实现已经把两者结合（用 FlashAttention/FlashInfer 之类的 kernel 后端配合 block 化存储）。
- **量化、chunked prefill**：**本论文完全没有涉及**，这些是 vLLM 项目后续演进出的特性，不属于这篇 SOSP 2023 论文的内容,阅读时要注意区分"论文提出的机制"与"vLLM 项目现在已经有的机制"。

## 4. 使用示例

vLLM 的编程接口把论文中的机制封装成了可调参数。下面按"论文机制 → 对应参数"的方式过一遍最常用的用法。

离线批量推理（对应§4.3/§4.5 的调度与批处理）：

```python
from vllm import LLM, SamplingParams

prompts = [
    "The future of AI infrastructure is",
    "PagedAttention works by",
]
# SamplingParams 对应论文里"每个序列独立的解码配置"
# n>1 时触发 parallel sampling 的 prompt KV 共享（§4.4）
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128, n=2)

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    gpu_memory_utilization=0.9,   # 对应 §4.2: KV Cache Manager 可用的显存上限
    block_size=16,                # 对应 §4.1/§7.2: PagedAttention 的 block size
    max_num_seqs=256,             # 对应 §4.5: 一个 iteration 里 batch 的最大序列数
    max_num_batched_tokens=4096,  # 对应 §4.5: 一个 iteration 里 batch 的最大 token 数（prefill+decode 混合计入）
    swap_space=4,                 # GB, 对应 §4.5: CPU block allocator 的 swap 容量
    enable_prefix_caching=True,   # 对应 §4.4: 共享前缀(shared prefix)机制
    tensor_parallel_size=1,       # 对应 §4.6: Megatron 式张量并行 GPU 数
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    for seq in output.outputs:
        print(seq.text)
```

启动 OpenAI 兼容的在线服务（对应论文里持续运行、iteration-level 调度的 serving engine）：

```bash
vllm serve meta-llama/Llama-2-7b-hf \
    --gpu-memory-utilization 0.9 \
    --block-size 16 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 4096 \
    --swap-space 4 \
    --enable-prefix-caching \
    --tensor-parallel-size 2
```

**关键参数与论文机制的对照**：

| 参数 | 调的是论文里的哪个机制 |
|---|---|
| `gpu_memory_utilization` | 决定 KV Cache Manager 能向 GPU Block Allocator 申请多少显存（§4.2），间接决定能开多大 batch |
| `block_size` | PagedAttention 的 block 大小（§4.1, §7.2），影响 kernel 效率 vs 内部碎片/共享粒度的取舍 |
| `max_num_seqs` | 每个 iteration 里 Scheduler 允许同时在跑的序列数上限（§4.5 iteration-level scheduling） |
| `max_num_batched_tokens` | 每个 iteration 里 prefill+decode 混合计入的 token 数上限，控制 compute-bound(prefill) 与 memory-bound(decode) 的混合比例 |
| `swap_space` | CPU Block Allocator 的容量，对应抢占恢复手段中的 swapping（§4.5） |
| `enable_prefix_caching` | 共享前缀机制（§4.4 Shared prefix），让不同请求共享同一段系统提示词的物理块 |
| `tensor_parallel_size` | Megatron 式张量并行的 GPU 数（§4.6），启用后所有 shard 共享同一份 block table |

**block table 映射的伪代码**（逻辑 token 位置 → 逻辑块 → 物理块 → kernel 寻址）：

```python
# 简化示意：论文 Fig.6 描述的映射过程

def logical_position_to_physical_address(seq, token_pos, block_size):
    # 1. 逻辑 token 位置 -> 逻辑块号 + 块内偏移（等价于 OS 里"虚拟地址 -> 页号 + 页内偏移"）
    logical_block_idx = token_pos // block_size
    offset_in_block = token_pos % block_size

    # 2. 查 block table：逻辑块号 -> 物理块号（这是纯元数据操作，不搬运真实数据）
    #    block_table[seq_id] 是这条序列的 List[BlockTableEntry]
    entry = seq.block_table[logical_block_idx]
    physical_block_id = entry.physical_block_id
    # entry.num_filled 记录该块已经写了多少个 token，用于判断是否需要申请新块

    # 3. 物理块号 + 块内偏移 -> 真实显存地址，供 PagedAttention kernel 直接读取 K/V
    physical_address = cache_engine.block_base_addr(physical_block_id) + offset_in_block * kv_slot_size
    return physical_address


def paged_attention_forward(query_i, seq, block_size):
    # kernel 内部：按逻辑块顺序，逐块从物理内存取出 K/V，累加注意力
    output = 0.0
    for logical_block_idx in range(ceil(seq.length / block_size)):
        entry = seq.block_table[logical_block_idx]
        K_block, V_block = cache_engine.read_block(entry.physical_block_id)  # 间接寻址读取
        attn_score = softmax(query_i @ K_block.T / sqrt(head_dim))
        output += attn_score @ V_block
    return output
```

## 5. 关键实验结论

- **总体吞吐**：在 ShareGPT 数据集上，vLLM 相比 Orca(Oracle)（假设系统预知真实输出长度的理论上限）快 **1.7×~2.7×**，相比 Orca(Max)（按模型最大长度预留）快 **2.7×~8×**，相比 FasterTransformer 快达 **22×**。说明"消除内部/外部碎片和预留浪费"带来的收益，不是理论上的边角优化,而是接近整数倍的吞吐差距，且差距在长序列、大模型场景下更明显（KV Cache 占比越大,PagedAttention 的收益越显著）。
- **并发 batch 规模**：在同样的显存下，vLLM 对 OPT-13B 平均能同时处理的请求数是 Orca(Oracle) 的 **2.2×**，是 Orca(Max) 的 **4.3×**。这直接验证了"省下来的碎片显存变成了额外的并发容量"这一因果链条。
- **并行采样与 beam search 下的显存共享收益**：并行采样节省 6.1%~9.8%（Alpaca）/16.2%~30.5%（ShareGPT）；beam search 由于共享粒度更细、更动态，节省 37.6%~55.2%（Alpaca）/44.3%~66.3%（ShareGPT）。说明**解码算法越复杂、候选间共享结构越丰富，PagedAttention 的共享收益越大**——这也解释了为什么论文强调"改进随更复杂的解码算法而更明显"。
- **共享前缀（few-shot prompt）**：1-shot（80 token 前缀）共享带来 1.67× 吞吐提升，5-shot（341 token 前缀）带来 3.58× 提升——前缀越长，被复用的计算和存储越多,收益越大。
- **Kernel 微观开销 vs 端到端收益**：PagedAttention 的间接寻址让 attention kernel 本身慢了 20%~26%（相对高度优化的 FasterTransformer kernel），但因为这一开销只发生在 attention 算子上（其余如 Linear 层不受影响），且被"能塞进更多 batch"的收益远远盖过,端到端仍然是数倍提升。这说明**局部的、可控的 kernel 效率损失,换来全局的内存利用率提升,是一笔非常划算的交易**——工程决策时要看端到端指标，不要被局部 benchmark 吓退。
- **block size 敏感性**：block size 16~128 在长序列数据集（ShareGPT）上表现相近；短序列数据集（Alpaca）上大 block size 显著掉性能（内部碎片占比随序列变短而急剧上升）。**默认 16** 是"kernel 并行效率"与"碎片/共享粒度"折衷后的经验值,不是理论最优,依赖工作负载的长度分布。
- **swap vs recompute**：recomputation 的开销与 block size 无关，始终不超过 swapping 延迟的 20%（小 block size 时）；block size 增大后二者接近。这给出一个可操作的经验规则：**block size 小时选 recompute，block size 适中偏大、PCIe 带宽充足时 swap 更优**。

## 6. 与本项目（算力网上大模型分布式运行基础设施）的关联

这是理解这篇论文对你当前研究方向价值的重点章节，逐条对应你要面对的问题。

**对"动态 batch / 动态序列长度 / KV Cache"这一研究问题的直接回应**：论文最核心的洞察是——LLM 推理的"动态性"根源在 KV Cache 的长度和生命周期不可预知,而**把动态性从"张量必须连续"这一僵化约束里解耦出来的方法就是分页**：用固定大小的块 + 间接寻址表，把一个本质上动态增长的数据结构,拆成"很多个大小固定、可以按需分配/释放/共享的小单元"。这是解决"动态 batch/动态 shape 下资源管理"问题的一个通用范式,不只适用于 GPU 显存,任何"总量不可预知、但可以分块管理"的资源都可以套用这个思路。

**block 化对"运行状态低成本切换后端/迁移中间状态"的启示**：因为 KV Cache 被切成了大小一致、边界清晰的物理块，"迁移一个请求的运行状态"这件事在理论上可以退化成**"迁移它 block table 里列出的那些块"**——迁移粒度从"一整个不可分割的张量"变成"若干个独立可寻址的小块",这为构建"块级别的迁移/复制通道"提供了数据结构基础：swap 到 CPU 内存本质上就是这样一种"块粒度迁移"，只是目的地是本机 CPU RAM。这提示我们，如果要做"跨设备/跨节点迁移正在运行的推理状态"（比如为了负载均衡、故障恢复、或者 prefill/decode 分离部署），**block 粒度的搬运单元和引用计数机制是可以直接复用的底层构件**，问题变成"把这条搬运通道从 PCIe 换成 NVLink/网络"。

**在异构算力池上做分页 KV Cache 会遇到的新问题**（论文完全没有讨论，是留给你的开放问题）：

- **block size 与 kernel 后端绑定**：论文的 block size 选择（16）是针对特定 GPU 架构（A100）、特定 attention kernel 实现调出来的经验值。换到不同型号的 GPU（甚至 NPU），warp 大小、显存带宽、cache line 大小都不同，最优 block size 会漂移，而且**不同后端的 PagedAttention kernel 实现（CUDA/ROCm/其它加速卡的算子库）可能被绑定在特定的 block size 或内存对齐方式上**——这意味着"块大小"从一个纯粹的性能调参,变成了一个跨后端调度时必须协调的兼容性约束。
- **跨设备/跨节点搬移的带宽层级差异巨大**：同机 PCIe、同机 NVLink、跨节点网络（RDMA/以太网）的带宽和延迟差异是几个数量级的。论文里的 swap 只考虑了"GPU 显存 ↔ 同机 CPU RAM"这一种通道，其开销模型（§7.3 的微观benchmark）完全建立在 PCIe 带宽假设上。在算力网场景下，"把 KV Cache block 从一块卡搬到另一块卡"要考虑清楚这些块究竟经过哪条物理通道，以及是否需要为每条通道单独建模"swap vs recompute"的取舍点（比如跨节点网络下,recompute 可能几乎总是比网络传输划算）。
- **prefix cache 的一致性与放置**：`enable_prefix_caching` 这种共享前缀机制，在单机场景下就是"物理块共享 + 引用计数"这么简单；但放到多节点、多副本的算力池里，同一段前缀的 KV Cache 应该缓存在哪个节点、如何保证多个副本间的一致性（要不要做类似分布式缓存的失效/更新协议）、请求路由要不要考虑"这个前缀缓存在哪"来做亲和性调度——这些都是论文完全没有涉及、需要在分布式基础设施层面重新设计的问题。

**与编译器视角的关系（为什么这种动态内存管理很难由静态编译器表达）**：图级 DL 编译器（如 XLA/TVM/MLIR 体系）擅长处理**编译期形状已知、生命周期可静态分析**的张量，可以做静态的内存规划、算子融合、buffer 复用。但 KV Cache 的长度依赖运行时才能确定的采样结果（何时遇到 EOS、beam search 的动态剪枝），这是**运行时才能揭示的动态信息**，编译器在图构建阶段根本看不到。这正是"动态 shape"问题的一个具体样本，也呼应了"配置组合爆炸"——如果试图为每一种可能的序列长度都生成一份静态编译的图/kernel，组合数是不可控的。**vLLM 给出的分工方式是**：把"内存到底怎么分配、共享、换入换出"这件事完全从编译期挪到**运行时的调度层+专门设计的 kernel**（PagedAttention 用间接寻址一次性适配所有长度，不需要为每个长度单独编译 kernel），编译器只需要负责生成"认识 block table、支持间接寻址"这一种通用 kernel,不用管上层运行时怎么组织这些块。**这是一个值得记住的分工原则：动态性强、依赖运行时反馈的决策（调度、内存分配）交给运行时；形状无关、可以一次编译多次复用的计算逻辑（attention 的分块计算本身）交给编译器/kernel 工程去做好通用化**。

**与后续工作的关系（明确标注：均非本论文内容）**：prefill 和 decode 两阶段特征差异巨大（compute-bound vs memory-bound），本论文虽然分析了这个差异，但**并没有提出"把两阶段分离部署在不同硬件/不同集群"**这个思路——这是后续工作如 DistServe、Mooncake 等做的事情（prefill/decode disaggregation）。这些后续系统正是在 vLLM 的 block 化 KV Cache 基础上，进一步探讨"prefill 产生的 KV Cache 要不要、怎么跨节点传输给专门做 decode 的机器"，这与本节前面提到的"跨节点搬移带宽层级"问题直接相关，是你后续可以专门去读的延伸方向,但不属于本论文讨论范围。

## 7. 学习这篇论文时的最小必要集

**必须掌握（5-7 个点）**：

1. **三类内存浪费的根因**：预留、内部碎片、外部碎片,均源于"连续内存分配 + 长度不可预知"的根本冲突,以及量化的严重程度（有效利用率仅 20%~38%）。
2. **PagedAttention 的分块 + 间接寻址**：attention 计算怎么从"一段连续 K/V"变成"逐块查表读取再累加"，公式本质是把 softmax 加权求和拆成分块累加。
3. **block table 与虚拟内存类比**：逻辑块/物理块/block table 三者关系，以及为什么这个设计能把内部碎片压到"最多一个块"、消除外部碎片。
4. **copy-on-write + 引用计数** 如何实现并行采样、beam search、共享前缀三种场景下的内存共享，以及触发拷贝的具体时机（有人要写且引用计数 > 1 时）。
5. **调度与抢占**：FCFS + iteration-level scheduling 是什么、为什么按 sequence group 整体驱逐而不是细粒度页面置换、swap 与 recompute 两种恢复手段的开销模型和取舍规律。
6. **分布式场景下单一 block table 的设计**：为什么 tensor parallel 下所有 GPU shard 可以共享同一份逻辑映射,理解"内存管理逻辑"与"计算执行逻辑"的解耦。
7. **关键量化结论的因果链**：碎片消除 → 有效利用率提升 → 并发 batch 变大 → 吞吐提升 2-4×，能把这条链条讲清楚就说明真正理解了论文的贡献所在，而不只是记住了数字。

**可以先跳过的内容**：

- §5.1 具体的 CUDA kernel 融合实现细节（fused reshape/block write 等）——工程优化细节，理解"存在这类优化、目的是压低间接寻址开销"即可，不必深究实现。
- §6 中针对不同 baseline（FasterTransformer、Orca 三个变体）的具体实验配置差异——记住结论量级（2-4×）和"为什么 vLLM 更强"的原理即可，不必逐个复现实验设置。
- 参考文献里关于其它 serving 系统（Clipper、Clockwork、DVABatch 等）的介绍性内容——这些是背景性related work，除非要系统调研 serving 领域全貌，否则可以只留个"存在这些方向"的印象。
- 数学符号推导本身（Eq.1-4）——公式是自然语言描述"分块累加注意力"的严谨表达，理解自然语言版本后，公式可以一带而过。
