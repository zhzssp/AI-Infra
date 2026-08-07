# FlashAttention：用 IO 感知的 tiling + 重计算，把 attention 从"算得快但搬得慢"变成真正的 wall-clock 加速

> **导航**：[笔记索引](README.md) · [自学枢纽](../README.md)（阶段 5，自选补充；建议穿插进阶段 4 开头） · [横切概念](../ai-compiler-foundations.md) §4.3（Roofline）、§9.4（IO 感知方法论）  
> **在学习路线中的定位（自选补充）**：本篇**不在主链上**，也没有配套动手项目。它以一个身份融入——**「融合 + tiling + 重计算」的实证**，用来在学 TVM/schedule 之前建立「为什么值得做这些变换」的动机。  
> 结论已蒸馏进 [`../ai-compiler-foundations.md`](../ai-compiler-foundations.md) §4.3 / §9.4，**跳过本篇不断链**。要读就抓「HBM 访存是瓶颈」这条主线，**online softmax 与 IO 复杂度公式看懂即可，不必手推**。  
> **配套阅读**：推理侧的对应物见 [`09-paged-attention-vllm.md`](09-paged-attention-vllm.md)。  
> **本篇章号提示**：多一节「后续演进」，因此**最小必要集在 §8**。

> 论文元信息：
> - 标题：*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*
> - 作者机构：Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré（Stanford University / University at Buffalo, SUNY）
> - 发表信息：NeurIPS 2022（预印本 2022-05-27，v2 2022-06-24）
> - arXiv：[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
> - 代码：<https://github.com/HazyResearch/flash-attention>（后续演进为 Dao-AILab/flash-attention）

---

## 0. 一句话总结

标准 attention 实现的瓶颈不是浮点运算（FLOPs）不够多，而是 \(N\times N\) 的中间矩阵在 GPU 显存（HBM）与片上高速缓存（SRAM）之间来回搬运；FlashAttention 通过**分块（tiling）+ 在线 softmax（online softmax）+ 反向传播重计算（recomputation）**，把整个 attention 算子融合成一个 kernel，只在 SRAM 内完成绝大部分计算，从而把 HBM 访问次数从 \(\Theta(N^2)\) 量级降到 \(\Theta(N^2 d^2 M^{-1})\)，在不改变数值结果（**exact attention**，不是近似）的前提下获得数倍的实测加速与线性显存占用。这是理解"AI 编译器为什么要做算子融合、循环切分（tiling）、重计算"这一整套优化手段最有说服力的手写 kernel 案例。

---

## 1. 它解决什么问题

### 1.1 标准 attention 为什么慢：瓶颈是访存，不是算力

给定 \(Q, K, V \in \mathbb{R}^{N\times d}\)（\(N\) 是序列长度，\(d\) 是 head 维度，通常 \(N \gg d\)，例如 GPT-2 中 \(N=1024, d=64\)），标准实现分三步：

\[
S = QK^\top \in \mathbb{R}^{N\times N},\quad P = \mathrm{softmax}(S) \in \mathbb{R}^{N\times N},\quad O = PV \in \mathbb{R}^{N\times d}
\]

这三步在 PyTorch/TensorFlow 这类框架里会被翻译成三到五个独立 kernel（matmul → mask → softmax → dropout → matmul），**每个 kernel 都要把输入从 HBM 读入、计算完再把结果写回 HBM**。问题在于：

- \(S\) 和 \(P\) 都是 \(N\times N\) 的矩阵，必须完整物化（materialize）到 HBM，显存占用是 \(O(N^2)\)；
- softmax、mask、dropout 这些逐元素/归约操作本身 FLOPs 很少，但需要把整块 \(N\times N\) 矩阵读一遍、写一遍——它们是**访存受限（memory-bound）**而不是**算力受限（compute-bound）**的操作；
- 因此 attention 的实际运行时间主要由"读写 HBM 的字节数"决定，而不是由矩阵乘法的浮点运算量决定。

这正是论文提出的核心论点：**近似 attention（稀疏化、低秩分解等）方法专注于降低 FLOPs，却忽视了访存开销，所以理论复杂度更低，实测 wall-clock 却经常不加速**——FlashAttention 反其道而行，用**更多的 FLOPs**（因为反向传播要重计算）换取**更少的 HBM 访问**，结果是真实更快。

### 1.2 GPU 内存层次：算得快、搬得慢的数量级差距

论文以 A100 GPU 为例列出内存层次的带宽与容量（Figure 1 左）：

| 层级 | 带宽 | 容量 |
|---|---|---|
| GPU SRAM（片上，每个 SM 独享） | ~19 TB/s | ~20 MB（108 个 SM，每个 192KB） |
| GPU HBM（显存） | 1.5–2.0 TB/s | 40–80 GB |
| CPU DRAM（主存） | ~12.8 GB/s | \(>\)1 TB |

SRAM 比 HBM 快一个数量级，但容量小三个数量级以上。近年来 GPU 计算能力的增长速度远超显存带宽的增长速度（论文引用 V100→A100→H100 的架构演进），这意味着越来越多算子从"算力受限"滑向"访存受限"——**这也是为什么 IO 感知（IO-awareness）会成为设计高性能 kernel 的第一原则**：一个算子如果频繁地把大张量在 HBM 与 SRAM 之间搬运，那无论矩阵乘法本身多快，总运行时间都会被这些搬运锁死。

Kernel fusion（把多个逐元素/归约算子融合进一个 kernel，中间结果留在 SRAM/寄存器里不落地到 HBM）是应对访存受限算子的标准手段，编译器（XLA、TorchScript 等）能自动融合许多简单的逐元素算子，但对于训练场景——中间结果还要留给反向传播用——朴素融合的收益有限，这正是 FlashAttention 要额外解决的问题。

---

## 2. 整体运行框架 / 算法机制

### 2.1 执行结构总览（对应论文 Figure 1 左）

```
                     HBM（显存，慢，大）                         SRAM（片上，快，小）
   ┌───────────────────────────────────┐            ┌───────────────────────────┐
   │  Q [N×d]   K [N×d]   V [N×d]       │            │                           │
   │  O [N×d]（增量累加输出）              │            │   当前块 Kj, Vj  (Bc×d)     │
   │  ℓ [N]（running sum，逐行归一化因子）  │  外层循环 j │   当前块 Qi, Oi  (Br×d)     │
   │  m [N]（running max，逐行最大值）     │──加载───→ │   当前块 Sij = Qi·Kjᵗ       │
   │                                      │            │   在 SRAM 内做:            │
   │            ▲          │              │            │    softmax 局部统计       │
   │            │ 写回增量  │ 内层循环 i      │            │    rescale 累加到 Oi       │
   │            │  更新     │ 逐块加载 Q/O/ℓ/m│            │                           │
   └────────────┴──────────┴──────────────┘            └───────────────────────────┘

外层循环（over j = 1..Tc）：把 K、V 按列分块，每块只从 HBM 搬一次到 SRAM
  内层循环（over i = 1..Tr）：把 Q、O、ℓ、m 按行分块搬入 SRAM
    → 在 SRAM 上计算 Sij = Qi Kjᵗ（Br×Bc 小矩阵，绝不落地到 HBM）
    → 计算局部 softmax 统计量 (m̃ij, ℓ̃ij) 并用 online softmax 公式与历史 (mi, ℓi) 合并
    → rescale 后累加进 Oi，把新的 Oi、ℓi、mi 写回 HBM（只有 O(N) 大小的统计量落地）
```

**从不出现的东西**：完整的 \(N\times N\) 矩阵 \(S\) 或 \(P\) 从未整体存在于 HBM 中——它只以 \(B_r \times B_c\) 的小块形式短暂存在于 SRAM 里，用完即弃。这是与标准实现最本质的区别。

### 2.2 逐步算法讲解（对应论文 Algorithm 1）

设 SRAM 大小为 \(M\)，块大小取 \(B_c = \lceil M/4d \rceil\)，\(B_r = \min(\lceil M/4d\rceil, d)\)（保证 \(K_j, V_j, Q_i, O_i, S_{ij}\) 都能放进 SRAM）。把 \(Q\) 按行分成 \(T_r = \lceil N/B_r\rceil\) 块，\(K, V\) 按行分成 \(T_c = \lceil N/B_c\rceil\) 块。

**外层循环** `for j = 1..Tc`：把 \(K_j, V_j\) 从 HBM 加载到 SRAM（每个 \(K,V\) 块在整个算法执行期间只被加载一次）。

**内层循环** `for i = 1..Tr`：把 \(Q_i, O_i, \ell_i, m_i\) 加载到 SRAM，执行：

1. 计算局部注意力分数：\(S_{ij} = Q_i K_j^\top \in \mathbb{R}^{B_r\times B_c}\)
2. 计算该块的局部统计量（沿列方向，即每行内部）：
   \[
   \tilde m_{ij} = \mathrm{rowmax}(S_{ij}),\quad \tilde P_{ij} = \exp(S_{ij} - \tilde m_{ij}),\quad \tilde\ell_{ij} = \mathrm{rowsum}(\tilde P_{ij})
   \]
3. **在线合并（online softmax 的核心一步）**，把"历史看到的所有块"的统计量 \((m_i, \ell_i)\) 与"当前块"的统计量 \((\tilde m_{ij}, \tilde\ell_{ij})\) 合并成新的全局统计量：
   \[
   m_i^{\text{new}} = \max(m_i, \tilde m_{ij}),\qquad
   \ell_i^{\text{new}} = e^{m_i - m_i^{\text{new}}}\,\ell_i \;+\; e^{\tilde m_{ij} - m_i^{\text{new}}}\,\tilde\ell_{ij}
   \]
4. **重标定（rescale）后累加输出**：旧的 \(O_i\) 是按旧最大值 \(m_i\) 归一化过的，要先用 \(e^{m_i - m_i^{\text{new}}}\) 撤销旧归一化、乘回旧的 \(\ell_i\)，再加上当前块贡献 \(e^{\tilde m_{ij}-m_i^{\text{new}}}\tilde P_{ij}V_j\)，最后统一除以新的 \(\ell_i^{\text{new}}\) 得到正确归一化的输出：
   \[
   O_i \leftarrow \mathrm{diag}(\ell_i^{\text{new}})^{-1}\Big(\mathrm{diag}(\ell_i)\,e^{m_i-m_i^{\text{new}}}\,O_i \;+\; e^{\tilde m_{ij}-m_i^{\text{new}}}\,\tilde P_{ij} V_j\Big)
   \]
5. 把 \(O_i, \ell_i \leftarrow \ell_i^{\text{new}}, m_i \leftarrow m_i^{\text{new}}\) 写回 HBM（只有 \(O(N)\) 大小的向量/矩阵落地，不含任何 \(N\times N\) 的量）。

跑完全部 \(T_c \times T_r\) 次内层迭代后，\(O\) 就精确等于 \(\mathrm{softmax}(QK^\top)V\)。论文用对外层循环下标 \(j\) 做归纳法证明了这一点（Appendix C，Theorem 1）——关键在于每次合并操作在数学上等价于把"目前处理过的所有列"重新做一次完整 softmax，只是增量地完成。

### 2.3 数值稳定性为什么在分块下依然成立

Softmax 的标准数值稳定实现（safe softmax）要减去最大值防止 \(e^x\) 溢出：
\[
m(x) := \max_i x_i,\quad f(x) := [e^{x_1-m(x)}, \dots, e^{x_B-m(x)}],\quad \ell(x) := \sum_i f(x)_i,\quad \mathrm{softmax}(x) := f(x)/\ell(x)
\]

分块的关键推导是：对于拼接向量 \(x = [x^{(1)}, x^{(2)}]\)，其全局统计量可以用两个子块各自的统计量组合出来（**代数聚合，algebraic aggregation**）：
\[
m(x) = \max(m(x^{(1)}), m(x^{(2)}))
\]
\[
f(x) = \big[e^{m(x^{(1)})-m(x)}f(x^{(1)}),\; e^{m(x^{(2)})-m(x)}f(x^{(2)})\big]
\]
\[
\ell(x) = e^{m(x^{(1)})-m(x)}\ell(x^{(1)}) + e^{m(x^{(2)})-m(x)}\ell(x^{(2)})
\]

这就是 2.2 节第 3、4 步公式的来源：只要维护 \((m, \ell)\) 这两个 \(O(N)\) 大小的统计量，就可以"边看一块、边合并"地算出与一次性看到全部数据完全相同的 safe softmax 结果，而不需要把整行 \(N\) 个分数同时放进内存。这也正是"online softmax / streaming softmax"名字的来源——它把一个归约操作变成了可以流式处理、增量更新的形式，和在线算法里维护 running max/running sum 的思想是一致的。

### 2.4 反向传播：为什么"存 O(N) 统计量 + 重计算"比"存 O(N²) 矩阵"更快

反向传播需要用到前向的 \(S, P\)（\(N\times N\)）来计算 \(dQ, dK, dV\)。直觉上"存下来复用"应该比"重新算一遍"更省时间，但论文指出：**决定运行时间的是 HBM 访问次数，不是 FLOPs**。

- 若存储 \(P\)：反向传播要从 HBM **读** \(O(N^2)\) 大小的 \(P\)，这本身就是 \(\Theta(N^2)\) 级别的访存；
- 若不存储，只存前向的输出 \(O\) 与统计量 \((\ell, m)\)（大小 \(O(N)\)）：反向传播时把 \(Q,K,V\) 的块重新载入 SRAM，**在片上重新算一遍** \(S_{ij}, P_{ij}\)（这部分是纯计算，不产生额外 HBM 访问），然后直接算梯度。

数学上，反向传播可以解析地写成（不需要对角雅可比矩阵的显式构造）：
\[
dV = P^\top dO,\qquad dP = dO\,V^\top,\qquad D_i = \mathrm{rowsum}(dO_i \odot O_i) = do_i^\top o_i
\]
\[
dS_{ij} = P_{ij} \odot (dP_{ij} - D_i),\qquad dQ = \tau\, dS\, K,\qquad dK = \tau\, dS^\top Q
\]
其中 \(D_i\) 是 softmax 反向传播里 \(P_{i:}^\top dP_{i:}\) 这一项的化简结果——利用 \(D_i = do_i^\top o_i\) 这一恒等式，避免了对 \(N\) 长度向量做归约，只需要对 \(d\) 维向量做点积，同样可以在分块、只用 \(O(d)\) 额外内存的情况下完成。

也就是说，重计算带来了**更多的矩阵乘法 FLOPs**（因为 \(S,P\) 要重新算一次），但节省的是**一整个 \(N\times N\) 矩阵的 HBM 读操作**——而这部分省下来的访存时间远大于多花的计算时间，这就是"even with the increased FLOPs due to recomputation, our algorithm runs faster"的原因，本质上是一种**选择性梯度检查点（selective gradient checkpointing）**：checkpoint 的对象是"是否存中间激活"，代价是重算，但因为重算发生在快速的 SRAM/寄存器上而不产生额外访存，所以是净赢。

### 2.5 IO 复杂度分析：标准实现 vs FlashAttention

设 \(N\) 为序列长度，\(d\) 为 head 维度，\(M\) 为 SRAM 大小，且 \(d \le M \le Nd\)。论文给出（Theorem 2）：

\[
\text{标准 attention 的 HBM 访问次数：}\ \Theta(Nd + N^2)
\]
\[
\text{FlashAttention 的 HBM 访问次数：}\ \Theta\!\left(\frac{N^2 d^2}{M}\right)
\]

推导要点：标准实现要把整个 \(N\times N\) 的 \(S\) 写入 HBM 再读出来做 softmax，再写 \(P\)，这几步都是 \(\Theta(N^2)\) 级别的访问。FlashAttention 中，\(K,V\) 的每个元素只从 HBM 加载一次（外层循环 \(T_c\) 次，每次 \(\Theta(M)\) 大小），但对 \(Q,O\) 要重复扫描 \(T_c = \Theta(Nd/M)\) 遍（因为外层循环每换一个 \(K,V\) 块就要重新过一遍所有 \(Q\) 块），每遍 \(\Theta(Nd)\)，因此总访问量是 \(\Theta(Nd \cdot T_c) = \Theta(N^2 d^2/M)\)。

**这个式子说明了什么**：
- 因为典型情况下 \(d\)（64–128）远小于 \(M\)（片上内存约 100KB 量级，換算成元素个数后 \(d^2 \ll M\)），所以 FlashAttention 的 HBM 访问比标准实现少好几倍到接近一个数量级（论文实测最高约 9 倍）；
- 访问次数只依赖 \(N^2 d^2/M\)，**块大小越大（\(M\) 越大）访问次数越少**——这解释了 Figure 2 中"增大 block size 能持续降低运行时，直到块大小超过某个阈值后被计算量本身限制"的现象；
- 论文进一步证明了**下界**（Proposition 3）：对于所有 \(M \in [d, Nd]\) 的范围，不存在一种精确 attention 算法能在渐进意义上比 \(\Theta(N^2d^2/M^{-1})\) 更优——也就是说 FlashAttention 在 IO 复杂度这个指标上是**（在给定 SRAM 容量约束下）渐进最优**的，这是一种典型的"给定内存层次约束求最优访存调度"的结果，与数据库/数值线性代数里的 I/O 复杂度理论（external-memory model）是同一套方法论。

### 2.6 Block-sparse FlashAttention（简述）

把 FlashAttention 的思路直接套到近似 attention 上：给定块级稀疏 mask \(M \in \{0,1\}^{N/B_r \times N/B_c}\)，算法与 Algorithm 1 完全相同，只是**跳过 mask 中为 0 的块**（不进行任何计算，也不产生任何 HBM 访问）。其 IO 复杂度变为：
\[
\Theta\!\left(Nd + \frac{N^2 d^2}{M}\,s\right)
\]
其中 \(s\) 是非零块的比例。这意味着稀疏比例 \(s\) 直接线性地降低了原本占主导的 \(N^2\) 项，当 \(s = \Theta(1/\sqrt N)\) 或 \(\Theta(\log N / N)\) 时，可以把复杂度进一步降到 \(\Theta(N\sqrt N)\) 或 \(\Theta(N\log N)\)。因为块内计算依然复用了 FlashAttention 的融合 kernel，block-sparse FlashAttention 比此前所有近似 attention 实现都快（论文用固定的 butterfly 稀疏模式做下游实验，实测比 dense FlashAttention 再快 2–4×）。

---

## 3. 核心特性逐条拆解

### 3.1 Kernel fusion（算子融合）
- **是什么**：把 \(QK^\top \to \mathrm{mask} \to \mathrm{softmax} \to \mathrm{dropout} \to \cdot V\) 这一整条链路写成一个 CUDA kernel，一次性完成，而不是拆成 5 个独立 kernel 各自读写 HBM。
- **为什么这样设计**：链路中除了两次矩阵乘法是算力受限的，中间的 mask/softmax/dropout 都是访存受限的逐元素/归约操作；如果分成独立 kernel，每个都要完整读一遍、写一遍 \(N\times N\) 的中间结果，访存开销随环节数线性叠加。融合后中间结果只存在于寄存器/SRAM，永不落地 HBM。
- **带来什么能力**：单个融合 kernel 消除了 4 次多余的 \(N\times N\) 级别 HBM 读写，这是 Figure 1 中 7.6× attention 计算加速的直接来源，也是编译器"算子融合"优化在真实 kernel 里能带来数量级收益的具体证据。

### 3.2 Tiling（分块）
- **是什么**：把 \(Q,K,V\) 沿序列维度切成 \(B_r \times d\) / \(B_c \times d\) 的小块，每次只把一小块搬进 SRAM 参与计算。
- **为什么这样设计**：SRAM 容量（~20MB/GPU，每个 SM 独享约 192KB）远小于 \(N\times N\) 或哪怕 \(N\times d\) 矩阵的大小，无法一次性放下整块数据；同时 softmax 是"耦合"所有列的归约操作（每一行的归一化需要看到该行全部分数），朴素分块会破坏归约的正确性，必须配合 online softmax 才能保持数学等价。
- **带来什么能力**：tiling 是让"融合大算子"在有限片上内存下可行的前提；它把访存模式从"整块读整块写"变成"小块流式读、增量式写"，直接对应 IO 复杂度分析中 \(N^2d^2/M\) 这个结果——tiling 粒度（块大小）与 IO 复杂度是直接的数学关系，这也是为什么"tiling"是所有 DL 编译器（TVM/Halide/MLIR affine/Triton）里最基础、最核心的循环变换。

### 3.3 Online / streaming softmax
- **是什么**：用 \((m_i, \ell_i)\) 两个 \(O(N)\) 大小的统计量代替完整的一行 softmax 分数，每处理完一个新块就用重标定公式把新块的贡献合并进已有统计量。
- **为什么这样设计**：softmax 本质上是一个"先看到全部数据才能归一化"的全局归约算子，与 tiling（只能看到局部数据）天然冲突；论文借用数值稳定 softmax 里 \(\max\)-\(\mathrm{shift}\) 的推导，证明这个归约可以写成**满足结合律的增量合并算子**（algebraic aggregation），从而可以像"运行中的最大值/运行中的求和"一样流式更新。
- **带来什么能力**：这是让"分块计算注意力"在数学上严格等价于"一次性计算完整 softmax"的核心技巧，没有它，tiling 只能用于近似算法（如某些线性 attention），而 online softmax 让 FlashAttention 成为**精确（exact）** 算法——这一点是它区别于此前所有近似 attention 方法、能同时兼顾速度与模型质量的关键。

### 3.4 Backward 重计算（recomputation）
- **是什么**：反向传播时不读取前向存储的 \(N\times N\) 注意力矩阵，而是只用存下的 \(O, \ell, m\)（\(O(N)\)）把 \(Q,K,V\) 的相关块重新载入 SRAM，在片上重新计算出 \(S_{ij}, P_{ij}\) 后直接算梯度。
- **为什么这样设计**：如果存储 \(P\)，反向传播不可避免要产生 \(\Theta(N^2)\) 级别的 HBM 读取；而"多算一次矩阵乘法"这部分 FLOPs 的开销远小于"少一次 \(N\times N\) 矩阵的 HBM 读写"节省的时间——这是用计算换访存的经典权衡（trade FLOPs for memory access），而不是传统意义上"用计算换显存容量"的梯度检查点（gradient checkpointing）。
- **带来什么能力**：反向传播同时获得了**更快的速度**（更少 HBM 访问）和**更少的显存占用**（只需 \(O(N)\) 额外内存），是罕见的"两者都要"而非"二者权衡"的优化——这打破了"梯度检查点必然牺牲速度换显存"的一般认知，前提是重算发生在片上、几乎不产生新的访存。

### 3.5 IO 感知的复杂度分析方法论
- **是什么**：不用传统的 FLOPs 计数分析算法复杂度，而是显式地对"从 HBM 读了多少字节、写了多少字节"建模，并给出关于 \(N, d, M\) 的渐进复杂度，还证明了在给定 SRAM 容量约束下的下界。
- **为什么这样设计**：现代 GPU 的算力增长速度远超显存带宽增长速度（V100→A100→H100 的架构趋势），越来越多算子从算力受限走向访存受限，FLOPs 已经不能预测真实的 wall-clock 时间；这套方法论借用了数据库/外存算法（external-memory algorithms）里成熟的 I/O 复杂度分析框架（Aggarwal & Vitter 1988）。
- **带来什么能力**：给出了一种可迁移的分析范式——任何涉及大中间张量的算子（不仅是 attention），都可以按"分块大小 \(B\) → SRAM 约束 \(M\) → 总 HBM 访问次数关于 \(N,d,M\) 的表达式"这套流程分析并指导 kernel 设计；这正是理解"为什么 AI 编译器要做 tiling/fusion/重计算"这些优化背后统一的理论依据，而不是零散的经验规则。

---

## 4. 使用示例

### 4.1 PyTorch 内置接口：`scaled_dot_product_attention`

PyTorch 2.x 把 FlashAttention（及 memory-efficient attention、纯数学实现）整合进统一算子，运行时按硬件与输入形状自动选择 backend：

```python
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# q, k, v: [batch, num_heads, seq_len, head_dim]
q = torch.randn(2, 16, 4096, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 16, 4096, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 16, 4096, 64, device="cuda", dtype=torch.float16)

# 让 PyTorch 自动挑选最优 backend（FlashAttention / memory-efficient / math）
out = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True,       # 因果掩码（自回归解码常用），等价于论文中的 mask 融合进 kernel
    scale=None,           # 默认 1/sqrt(d)，对应论文中的 softmax 缩放常数 tau
)

# 显式强制走 FlashAttention backend，便于做对比测速
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out_flash = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

### 4.2 `flash-attn` 官方库接口

`flash_attn_func` 用于定长 batch，`flash_attn_varlen_func` 用于变长序列（避免 padding 浪费，训练/推理混合长度场景常用）：

```python
from flash_attn import flash_attn_func, flash_attn_varlen_func

# 定长场景：q,k,v 形状为 [batch, seqlen, num_heads, head_dim]（注意维度顺序与 SDPA 不同）
out = flash_attn_func(
    q, k, v,
    dropout_p=0.0,
    softmax_scale=None,   # None 则默认 1/sqrt(head_dim)
    causal=True,          # 是否使用因果掩码
    window_size=(-1, -1), # 滑动窗口 attention，(-1,-1) 表示不限制
)

# 变长场景：把 batch 内所有序列拼接成一维，用 cu_seqlens 记录每条序列的累计起止位置
# cu_seqlens_q/k: [batch+1]，例如 batch 内三条长度为 [3,5,2] 的序列对应 [0,3,8,10]
out_varlen = flash_attn_varlen_func(
    q_packed, k_packed, v_packed,   # [total_tokens, num_heads, head_dim]，已去除 padding
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    dropout_p=0.0,
    causal=True,
)
```

关键参数含义：
- `causal`：是否融合因果掩码（对应论文 Algorithm 2 中的 `mask` 函数，掩码同样在 kernel 内完成，不产生额外 HBM 访问）；
- `softmax_scale`：对应论文中的 \(\tau\)（通常为 \(1/\sqrt d\)）；
- `cu_seqlens_q/k`（cumulative sequence lengths）：变长接口把 batch 内所有序列首尾拼接成一维 token 流，用累计长度数组标记每条序列的边界，从而避免对 padding 位置做无意义计算——这是训练阶段处理不等长样本、以及推理阶段处理不同 prompt 长度请求的标准做法。

### 4.3 前向 kernel 骨架（Triton 风格伪代码，示意）

```python
# 示意代码：仅用于说明 tiling + online softmax 在 kernel 里的形态，非可直接运行的生产实现
import triton
import triton.language as tl

@triton.jit
def flash_attn_fwd_kernel(
    Q, K, V, O,            # HBM 中的指针
    stride_qm, stride_qd,  # 省略其余 stride 参数
    seq_len, head_dim,
    BLOCK_M: tl.constexpr,  # 对应论文 B_r
    BLOCK_N: tl.constexpr,  # 对应论文 B_c
):
    row_block_idx = tl.program_id(0)          # 内层循环下标 i，由 GPU 并行网格承担
    q_rows = row_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)

    # 从 HBM 加载当前 Q 块到 SRAM（寄存器/shared memory），整段循环内只加载一次
    q = tl.load(Q + q_rows[:, None] * stride_qm + tl.arange(0, head_dim)[None, :] * stride_qd)

    # 初始化 running 统计量：对应论文的 m_i(初始 -inf)、l_i(初始 0)、O_i(初始 0)
    m_i = tl.full((BLOCK_M,), value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)   # O_i 的累加器

    # 外层循环：遍历 K, V 的所有列块（对应论文 j = 1..Tc）
    for col_block_start in range(0, seq_len, BLOCK_N):
        k = tl.load(K + ...)   # 加载当前 K_j 块到 SRAM
        v = tl.load(V + ...)   # 加载当前 V_j 块到 SRAM

        s_ij = tl.dot(q, k.T) * softmax_scale               # S_ij = Q_i K_j^T

        m_ij = tl.max(s_ij, axis=1)                          # 局部行最大值 m~_ij
        p_ij = tl.exp(s_ij - m_ij[:, None])                  # exp(S_ij - m~_ij)
        l_ij = tl.sum(p_ij, axis=1)                          # 局部行和 l~_ij

        m_new = tl.maximum(m_i, m_ij)                        # 合并出新的全局 max
        alpha = tl.exp(m_i - m_new)                          # 旧统计量的 rescale 因子
        beta = tl.exp(m_ij - m_new)                          # 新块的 rescale 因子

        l_new = alpha * l_i + beta * l_ij                    # 合并出新的全局 sum
        acc = acc * alpha[:, None] + beta[:, None] * tl.dot(p_ij, v)  # 累加输出并 rescale

        m_i, l_i = m_new, l_new                               # 更新 running 统计量，进入下一块

    out = acc / l_i[:, None]                                  # 最终统一归一化
    tl.store(O + q_rows[:, None] * stride_qm + ..., out)      # 写回 HBM，仅此一次
```

这段伪代码里，`alpha`/`beta` 两个 rescale 因子就是 2.2 节公式里的 \(e^{m_i-m_i^{\text{new}}}\) 与 \(e^{\tilde m_{ij}-m_i^{\text{new}}}\)；整段内层 `for` 循环只在最后写一次 `O`，中间任何 \(N\times N\) 级别的量都没有出现。

### 4.4 对比测速脚本骨架

```python
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

def naive_attention(q, k, v, causal=True):
    # 朴素实现：显式物化 S 和 P，模拟"标准 attention"的访存模式
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / d ** 0.5     # 物化 S: [B, H, N, N]
    if causal:
        mask = torch.triu(torch.ones(scores.shape[-2:], device=q.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)            # 物化 P: [B, H, N, N]
    return probs @ v

def benchmark(fn, q, k, v, warmup=10, iters=50):
    for _ in range(warmup):
        fn(q, k, v)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(q, k, v)
    end.record()
    torch.cuda.synchronize()

    latency_ms = start.elapsed_time(end) / iters
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    return latency_ms, peak_mem_gb

if __name__ == "__main__":
    for seq_len in [512, 1024, 2048, 4096, 8192]:
        q = torch.randn(2, 16, seq_len, 64, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q); v = torch.randn_like(q)

        flash_fn = lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)

        lat_naive, mem_naive = benchmark(naive_attention, q, k, v)
        lat_flash, mem_flash = benchmark(flash_fn, q, k, v)

        # 关键指标：
        # 1) 延迟（ms）：直接反映 wall-clock 加速比
        # 2) 峰值显存（GB）：验证 O(N) vs O(N^2) 的显存增长曲线差异
        # 3) HBM 带宽利用率（可用 nsight compute / torch.profiler 采集，
        #    naive 版本会明显更早打满带宽而 FLOPs 利用率却很低——这正是"memory-bound"的证据）
        print(f"N={seq_len:>5} | naive: {lat_naive:6.2f}ms {mem_naive:5.2f}GB "
              f"| flash: {lat_flash:6.2f}ms {mem_flash:5.2f}GB "
              f"| speedup={lat_naive/lat_flash:.2f}x")
```

该脚本应重点观察：随着 `seq_len` 增大，naive 版本的显存呈 \(O(N^2)\) 增长很快 OOM，而 flash 版本呈 \(O(N)\) 线性增长；同时用 `torch.profiler` 或 `nsys`/`ncu` 采集 HBM 读写字节数与 SM 利用率，可以直接验证"naive 版本的算力利用率很低但 HBM 带宽利用率很高"这一 memory-bound 特征。

---

## 5. 关键实验结论

- **BERT-large 训练**：在 8×A100 上，FlashAttention 训练 BERT-large（序列长度 512）比 MLPerf 1.1 的 Nvidia 官方训练速度记录快 **15%**（17.4 分钟 vs 20.0 分钟）——说明即使在中等序列长度下，去掉 attention 的冗余 HBM 访问也能带来端到端可感知的提速，而不只是理论数字。
- **GPT-2 训练**：序列长度 1K 时，比 HuggingFace 实现快达 **3×**，比 Megatron-LM 快 **1.7×**；且困惑度（perplexity）完全一致——说明加速是"零精度损失"的，因为算法是精确（exact）计算，不是近似。
- **长序列下的加速与训练效率的联动**：FlashAttention 支持 GPT-2 用 **4 倍**的上下文长度（4K vs 1K）训练，仍比 Megatron-LM 在 1K 上下文下快 **30%**，同时困惑度还改善了 **0.7**——这说明"更快 + 更省显存"直接转化为"能训练更长上下文并且质量更好"，而不仅仅是速度指标的提升。
- **Long-Range Arena（LRA）**：FlashAttention 比标准 attention 快 **2.4×**，block-sparse FlashAttention 快 **2.8×**，且是所有近似 attention 基线（Linformer、Performer、Reformer 等）里最快的——说明"精确但 IO 高效"可以同时在速度上超过"近似但 IO 低效"的方法。
- **长文档分类**：把序列长度从 512 扩到 16K，MIMIC-III 上提升 4.3 分，ECtHR 上（8K 长度）提升 8.5 分——说明长上下文能力不只是"跑得动"，是能直接换来下游任务质量提升的。
- **Path-X / Path-256（长程依赖挑战任务）**：此前所有 Transformer 变体在这两个任务上都只能达到随机猜测水平（因为需要建模 16K/64K 长度的长程依赖且显存/速度撑不住）；FlashAttention 是**第一个**在 Path-X（16K，61.4% 准确率）上超过随机水平的 Transformer，block-sparse FlashAttention 是第一个在 Path-256（64K，63.1% 准确率）上超过随机水平的序列模型——说明"IO 效率的提升可以直接兑换成模型能力的解锁"，而不只是同等能力下的效率优化。
- **纯 attention 算子微观benchmark**：在常见序列长度（128–2K）范围内比标准实现快最多 **3×**，比标准实现节省最多 **20×** 显存；序列长度超过 1K 后部分近似方法开始在速度上反超，但 block-sparse FlashAttention 在所有测试长度下都快于全部已知的精确/近似/稀疏 attention 实现——说明 IO 感知的收益在中短序列上最显著，长序列上仍需要配合稀疏化才能保持领先。
- **HBM 访问是决定性因子的直接证据**（Figure 2）：GPT-2 medium 配置下，FlashAttention 的 FLOPs（75.2 GFLOPs）反而比标准实现（66.6 GFLOPs）更高（因为反向重计算），但 HBM 读写量从 40.3GB 降到 4.4GB，运行时间从 41.7ms 降到 7.3ms——这组数据是全文最核心的证据，直接证明了"FLOPs 更多但访存更少→更快"这一反直觉结论。

---

## 6. 后续演进（简述，非本论文内容）

> 以下内容基于公开资料的已有知识补充，不属于本论文正文。

- **FlashAttention-2**（2023）：重新设计了 work partitioning（工作划分），减少非矩阵乘操作（non-matmul FLOPs，如 rescale、统计量更新等在 GPU 上效率远低于 Tensor Core 矩阵乘的操作）的占比，把 attention 内外层循环重新排布、在 warp 间更均衡地划分并行工作，实测比 FlashAttention 再快约 2 倍，把 GPU 利用率从约 25–40% 提升到 50–73%。
- **FlashAttention-3**（2024）：针对 Hopper 架构（H100）设计，利用 warp-specialized 异步执行（让不同 warp 组分别负责数据搬运与计算的流水重叠）、FP8 低精度计算配合误差补偿，进一步压榨 Tensor Core 与 TMA（Tensor Memory Accelerator）异步数据搬运的能力，在 H100 上相比 FA2 有显著提速，同时保持精度可用。
- **FlashDecoding / FlashDecoding++**：把 FlashAttention 的 tiling 思路应用到**推理阶段的解码（decode）**场景——decode 阶段 query 长度通常为 1（每次只生成一个 token），而 KV 长度可能很长，原本"沿 Q 分块"的并行策略在此场景下并行度不足（GPU 大量核心闲置）；FlashDecoding 改为**沿 KV 长度维度切分**做并行 reduction，解决了长上下文解码时 GPU 利用率低的问题。
- **与 PagedAttention 的关系**：PagedAttention（vLLM 提出）解决的是**KV Cache 的显存管理**问题（借鉴操作系统虚拟内存分页的思想，把 KV Cache 分成非连续的块管理，解决多请求变长序列下的显存碎片与浪费），而 FlashAttention/FlashDecoding 解决的是**attention 计算本身的访存效率**问题——两者是互补而非竞争关系：vLLM 等推理框架通常同时使用分页式 KV Cache 管理 + FlashAttention/FlashDecoding 风格的分块计算 kernel，共同构成高效 LLM 推理引擎的两大支柱。

---

## 7. 与本项目（AI 编译器 / 多后端基础设施）的关联

### 7.1 为什么这是"融合 + tiling + 重计算"作为编译器优化核心杠杆的最佳范例

FlashAttention 几乎是一个"教科书式"的展示：三种看似独立的优化手段（kernel fusion 消除中间结果落地、tiling 让大张量在有限片上内存下可处理、recomputation 用计算换访存）组合在一起，在**不改变算法语义**的前提下带来数量级的实测加速。这恰好对应 AI 编译器（TVM、Halide、MLIR 的 affine/linalg 变换、XLA 的 fusion pass）里三个最基础也最重要的循环/图变换：算子融合（operator fusion）、循环切分与内存层次映射（tiling + memory hierarchy mapping）、重计算换存储（rematerialization）。理解 FlashAttention 的价值不在于"记住这一个算子怎么写"，而在于建立一个心智模型：**几乎所有涉及大中间张量、访存受限的算子优化，都可以用"IO 复杂度关于 tiling 大小和硬件内存层次的表达式"来指导设计**——这正是本项目后续学习 MLIR、图级 DL 编译器时反复会用到的分析框架。

### 7.2 手写 kernel 与编译器自动生成的边界

论文在 Limitations 部分坦承了这一点：FlashAttention 需要针对每种硬件、每种变体手写 CUDA kernel（后续用 Triton 降低了一部分门槛），而不能仅在 PyTorch 层描述算法就自动获得 IO 最优的实现。原因在于：

- **依赖关系跨越归约边界**：softmax 是一个跨越整行（跨越所有列块）的全局归约，编译器要自动发现"可以用 online softmax 的代数聚合性质把这个归约重写成增量式合并"，需要具备对特定数学结构（结合律、可结合聚合算子）的领域知识，而不是通用的循环变换能推导出来的；
- **硬件相关的资源分配决策**（block size \(B_r, B_c\) 如何依据 SRAM 容量 \(M\)、寄存器数量、warp 调度设定）目前仍高度依赖对具体 GPU 架构（occupancy、bank conflict、shared memory bank 等）的手工调优，通用编译器的自动 tiling 策略（如 TVM 的 auto-tuning、MLIR 的 tiling pass）在搜索空间和领域知识覆盖上还难以完全替代专家手写；
- **重计算的选择本身是一个语义级决策**（哪些中间量该存、哪些该重算），这依赖于对"哪部分计算量小、访存量大"的算子级判断，通用编译器的 rematerialization pass 通常基于代价模型做局部决策，难以像 FlashAttention 这样做跨越前向反向的全局设计。

这也是为什么 Halide、Triton 这类"表达 tiling/调度但仍需要人工指定 schedule"的领域特定语言（DSL）会长期存在——它们把"正确性描述"与"性能调度"解耦，让专家能相对高效地手写出接近最优的 kernel，而不是完全依赖编译器自动发现。

### 7.3 多后端场景下的移植与变体管理挑战

FlashAttention 有大量变体：causal/non-causal、有无 dropout、有无 mask、定长/变长（varlen）、MHA/GQA/MQA 不同 KV 头数、滑动窗口、ALiBi/RoPE 位置编码融合……每种变体在 CUDA/Triton/ROCm（AMD）/其他 AI 加速芯片上都可能需要重新实现或调优。这正对应多后端基础设施中的两个经典难题：

- **后端最优计算图不一致**：同一个"attention"语义，在 NVIDIA GPU 上的最优实现（tiling 大小、是否用 Tensor Core、TMA 异步搬运）与在国产 AI 芯片、AMD GPU 上的最优实现可能完全不同，编译器/运行时需要为不同后端维护不同的 kernel 选择与生成策略，而不能假设一套 IR 能自动映射到所有后端的最优形式；
- **配置组合爆炸**：causal × varlen × 头数变体 × 精度（FP16/BF16/FP8）× 是否融合位置编码，几种维度交叉后 kernel 变体数量迅速膨胀，手写维护成本很高——这正是当前 AI Infra 领域推动"kernel 生成器/编译器（如 Triton、TVM TensorIR、Mosaic）"、"算子库参数化模板"等方案的动机：把 IO 感知的调度参数（block size 等）做成可搜索/可配置的模板，而不是每种组合都手写一份。

### 7.4 与分布式训练的关系

显存节省直接影响并行策略的选择空间：当 attention 的激活显存从 \(O(N^2)\) 降到 \(O(N)\) 后，同样的显存预算下可以支持更长的序列或更大的 micro-batch，这意味着：

- 减少了对**张量并行/流水并行**切分粒度的显存压力，某些原本需要跨卡切分才能塞进显存的长序列场景，现在单卡就能处理更大规模，简化了并行策略选择；
- 与**序列并行（sequence parallelism）**天然契合：序列并行本身就是把 \(N\) 维度切分到多张卡上处理，而 FlashAttention 的 tiling 思想（分块处理 Q/K/V 并用统计量合并）与序列并行需要解决的"跨卡合并局部 attention 结果"问题在数学结构上是一致的；
- **Ring Attention** 正是这种联系的直接产物：把 K/V 沿序列维度切分到不同设备上，各设备之间以环形（ring）方式传递 K/V 块，每个设备用类似 FlashAttention 的 online softmax 增量合并局部结果，从而把单卡 IO 感知的 tiling 思想扩展成了跨设备的分布式版本，实现近乎无限长上下文的分布式训练/推理——这也是论文 Limitations 部分提到的"多 GPU IO 感知方法"这一未来方向的具体实现路径。

---

## 8. 学习这篇论文时的最小必要集

**必须掌握（5–7 个点）：**

1. **问题定位**：attention 慢的根本原因是 HBM 访存量而非 FLOPs，标准实现要物化 \(O(N^2)\) 的 \(S,P\) 矩阵。
2. **Online softmax 的重标定公式**：\(m_i^{\text{new}}=\max(m_i,\tilde m_{ij})\)，\(\ell_i^{\text{new}}=e^{m_i-m_i^{\text{new}}}\ell_i+e^{\tilde m_{ij}-m_i^{\text{new}}}\tilde\ell_{ij}\)，以及输出 \(O_i\) 用同样的 rescale 因子累加——这是全文最核心的数学机制，必须能自己推一遍。
3. **IO 复杂度结论**：标准 \(\Theta(Nd+N^2)\) vs FlashAttention \(\Theta(N^2d^2/M)\)，以及"\(d^2 \ll M\) 所以快很多"这一结论的直观含义。
4. **重计算的取舍**：backward 不存 \(P\)，只存 \(O(N)\) 统计量，用重算 \(S,P\) 换取避免读 \(O(N^2)\) 的 HBM 访问——理解"多 FLOPs 但少访存 = 更快"这个反直觉但正确的权衡。
5. **算法是精确（exact）而非近似**：这是 FlashAttention 与此前大量近似 attention 方法的根本区别，速度提升不以牺牲模型质量为代价。
6. **Kernel fusion 的作用**：把多个逐元素/归约算子融合进一个 kernel 避免中间结果落地，是理解一切后续优化的前提认知。
7. **实验证据链**：Figure 2 里"FLOPs 更高但 HBM 访问更低→运行时间更短"这组对比数据，是整篇论文论点的实证支柱，建议记住这组数字（75.2 vs 66.6 GFLOPs；4.4GB vs 40.3GB；7.3ms vs 41.7ms）。

**可以先跳过：**

- Block-sparse FlashAttention 的具体理论证明（Proposition 4 的完整推导）及 butterfly 稀疏模式的结构化矩阵理论背景（Appendix A 里关于 structured matrices 的综述）——只需理解"稀疏比例 \(s\) 线性降低 \(N^2\) 项"这个结论即可；
- Appendix C 里 Theorem 1/2 的完整数学归纳证明细节——理解结论和推导要点（分块大小如何由 \(M\) 约束推出）即可，不必逐行验证证明；
- 与 Rabe and Staats [66] 方法的详细对比（Appendix B.5）——这是相关工作的差异化讨论，了解"FlashAttention 优化的是访存次数而不仅是总内存占用"这一句话即可；
- Long-Range Arena 各子任务的具体实验设置与调参细节（脚注中提到 LRA 结果对训练细节敏感）——这是实验复现层面的内容，对理解算法机制不是必需的。
