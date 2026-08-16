# `iterator_types` 与 Verifier：声明式语义和 IR 安全网

> 来源：[`mlir-learning-guide.md`](../learning-guides/mlir-learning-guide.md) §8.2

## 是什么

`iterator_types` 是 `linalg.generic` 这类结构化 op 中的核心声明：

```mlir
iterator_types = ["parallel", "parallel", "reduction"]
```

它告诉编译器：

- 哪些维是**可并行**的
- 哪些维是**归约**的
- 这个迭代空间的“语义角色”是什么

它不是具体循环代码，也不是“一定正确的算法证明”；它更像“给编译器的语义契约”。

## 它决定了什么

### 1. 是否能并行

如果某个维是 `parallel`，说明这一维上的不同点之间没有依赖，编译器可以把它看成可并行维：

- vectorize
- split / tile
- GPU/CPU thread 映射
- fuse 到更大并行块

### 2. 是否有归约

如果某个维是 `reduction`，说明它不是独立点计算，而是要沿该维累积：

```text
C[m, n] += sum_k A[m, k] * B[k, n]
```

`k` 是归约维，编译器会把它当成“需要汇总”的维，而不是普通并行维。

### 3. 结构化 op 能否被进一步分析/融合

对 `linalg.generic` 来说，`iterator_types` + `indexing_maps` 是融合和重排的核心依据：

- 只看结构，不必理解算子具体语义
- 只要两个 op 的访问模式 + 迭代角色对齐，编译器就能判断是否能 fuse、tile、reorder

## 和 `indexing_maps` 的关系

```mlir
#A = affine_map<(m, n, k) -> (m, k)>
#B = affine_map<(m, n, k) -> (k, n)>
#C = affine_map<(m, n, k) -> (m, n)>

iterator_types = ["parallel", "parallel", "reduction"]
```

联合起来读：

- `m, n` 是输出上的并行维
- `k` 是归约维
- `A` / `B` / `C` 分别说明在当前迭代点访问哪些元素

这比单看 body 里的 `mulf` / `addf` 更有利于编译器做全局分析。

## 它不是“绝对真理”，而是“声明”

你可以把它理解成：

```text
“我声明这个维是 reduction / parallel”
```

但这不等于“这一定真的能实现”。

例如你可能写错：

```mlir
iterator_types = ["reduction"]
```

但实际 body 里并不是在做归约，或者访问模式不支持这个归约语义。这样时常会出现：

1. 结构看起来合法
2. verifier 也没直接报错
3. 但后续 pass 按 reduction 的语义去做优化
4. 最终结果错误，或者 codegen 产生坏代码

也就是说：**声明不等于语义证明**。

## Verifier 是什么

Verifier 是 MLIR 里**内置的 IR 合法性检查机制**，它负责检查：

- Op 的操作数数目是否正确
- 类型是否匹配
- 区域/Block/terminator 是否满足结构约束
- SSA 是否有效
- Dialect 特定语义是否满足基本不变式

它主要检查“IR 是否结构上合理”，不是证明算法语义一定正确。

### Verifier 典型检查对象

- `Block` 必须以 terminator 结束
- 结果类型和 operand 类型必须一致
- op 的属性和 input/output 数量必须匹配
- `indexing_maps` 的个数要和操作数匹配
- `iterator_types` 的维数要与迭代空间/访问模式相符

## Verifier 和 Pass 的分工

| 概念 | 主要作用 | 典型检查对象 |
|------|----------|--------------|
| **Verifier** | 保证 IR 合法 | type、SSA、block、terminator、dialect 约束 |
| **Pass** | 对合规 IR 做优化/变换 | fusion、tiling、canonicalization、lowering |

所以：

- Verifier 负责“保底安全门”
- Pass 负责“在安全门后做优化”

## 什么时候会被 Verifier 拦住

只要是明显结构不合法，通常就会直接报错，例如：

- operation 数量不对
- block 没有 terminator
- `iterator_types` 维度和 map 数量明显不匹配
- operand 类型与 op 定义不符

这类错误通常比较直接：**IR 不合法，编译器拒绝继续**。

## 什么时候不会被 Verifier 直接拦住

如果声明看起来“结构合理”，但语义其实错了，verifier 往往不会从字面量声明里证明出“你必须错了”；它可能只是让 IR 继续流过，例如：

```mlir
iterator_types = ["parallel", "reduction"]
```

如果 body 实际上不遵守 reduction 语义，后续 pass 可能会错误地并行化/合并，最后才在后面的优化或代码生成阶段暴露问题。

这说明：

- verifier 是**必要的安全网**，但不是“算法语义证明器”
- 语义层面的正确性，更多依赖 pass/分析者和开发者自己保证

## 结论

`iterator_types` 不是“绝对可实现”的断言，而是**编译器分析时使用的结构化声明**。它告诉编译器“哪些维是并行，哪些维是归约”，从而决定后续能否做 vectorize、fuse、tile、分块等优化。

而 verifier 则负责把“明显不合法的 IR”拦掉，保证后续 pass 运行在一个结构正确、类型正确、SSA 正确的 IR 上；但它并不负责证明你的语义声明一定真。

## 一句话记忆

**`iterator_types` 是把循环的语义角色写进 IR；`verifier` 是把 IR 的合法性门禁写进框架。**
