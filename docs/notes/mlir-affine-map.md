# Affine Map：循环索引到张量访问的声明

> 来源：[`mlir-learning-guide.md`](../mlir-learning-guide.md) §8.2

## 是什么

`affine_map` 不是一段要执行的计算，它是一种**声明式的访问模式说明**：

```mlir
(m, n, k) -> (m, k)
```

它的语义是：

- 当前迭代点是 `(m, n, k)`
- 对应的张量访问位置是 `(m, k)`
- 编译器把它当成“这个 op 如何从循环变量映射到数据索引”的静态声明

所以它更像“索引公式”或“访问契约”，而不是运行时执行代码。

## 它在 MLIR/ Linalg 里干什么

它的主要作用是让编译器静态分析访问模式，而不是直接算值：

1. **合法性检查**：索引是不是仿射表达式、维度是否匹配、是否和迭代域一致。
2. **依赖分析**：判断两个访问是否同一内存、是否冲突、是否可并行/可融合。
3. **变换优化**：给 loop fusion、tiling、reordering、vectorization 提供基础信息。
4. **把“计算是什么”和“访问模式是什么”分开**：`linalg.generic` 的 body 描述标量计算；`indexing_maps` 描述访问模式。

## 与 `indexing_maps` 的关系

`affine_map` 是单个映射；`indexing_maps` 是多个映射拼成一个列表：

```mlir
#A = affine_map<(m, n, k) -> (m, k)>
#B = affine_map<(m, n, k) -> (k, n)>
#C = affine_map<(m, n, k) -> (m, n)>

#matmul_accesses = [#A, #B, #C]
```

它表示：

- 第一个 operand 访问 `A[m, k]`
- 第二个 operand 访问 `B[k, n]`
- 输出写 `C[m, n]`

所以 `linalg.generic` 的语义可以直接理解成：

```text
C[m, n] += A[m, k] * B[k, n]
```

这里的关键是：**不需要看 op 名字，只要看访问映射和 iterator 类型就能判断结构。**

## 为什么说它是 affine，而不是任意函数

`affine_map` 只允许“仿射表达式”：

```mlir
(d0, d1) -> (d0 + 1)
(d0, d1) -> (2 * d0 + 3 * d1)
(d0, d1) -> (d0, d1)
```

其中：

- 循环变量能线性组合
- 能带常数偏移
- 适合做多面体/依赖分析

不属于 affine map 的典型例子：

```text
(d0, d1) -> (d0 * d1)
(d0, d1) -> (f(d0))
(d0, d1) -> (g(d1))
```

这些不能直接放进 `affine_map`，因为它们不满足“仿射/线性”的静态分析要求。

## 若不是线性映射怎么办

如果索引需要复杂计算，比如：

```text
A[f(d0), g(d1)]
```

那么通常不是把它硬塞进 `affine_map`，而是把复杂索引计算放到 body 或更通用的 IR 里：

```mlir
%idx0 = ...  // 计算 f(d0)
%idx1 = ...  // 计算 g(d1)
%v = tensor.extract %A[%idx0, %idx1] : tensor<...>
```

也就是说：

- `affine_map` 负责“访问模式是线性的/可分析的”
- 更复杂的索引计算由普通 `arith` / `scf` / 其他 op 负责

## 它不是所有 MLIR op 都有的通用字段

不是所有 op 都带 `affine_map`，但它在以下场景里特别常见：

- `affine` dialect（循环 / load / store）
- `linalg.generic` / `linalg.matmul` 这样的结构化 tensor op
- 依赖分析、fusion、tiling、vectorization 等 pass

它最典型的用途是：

> 描述“当前循环迭代点访问哪一个 tensor 元素”。

## 一个最短记忆法

- `affine_map`：索引公式 / 访问声明
- `indexing_maps`：把一组访问公式挂到 Linalg op 上
- `iterator_types`：哪些维是并行的、哪些是归约的
- 编译器用途：合法性检查 + 依赖分析 + 变换优化

## 例子：逐元素加法

```mlir
#id = affine_map<(d0, d1) -> (d0, d1)>

%out = linalg.generic {
    indexing_maps = [#id, #id, #id],
    iterator_types = ["parallel", "parallel"]
  } ins(%a, %b : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%init : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %acc: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<4x8xf32>
```

这里：

- `#id` 表明每个点读 `a[d0, d1]`、`b[d0, d1]`
- 输出写 `out[d0, d1]`
- 两个维都可并行

整个 op 的结构就被“访问模式 + 迭代类型”完整表达出来了。

## 结论

`affine_map` 本身不是执行计算的代码，而是**编译器用来描述“循环索引到张量访问”的声明**。它最关键的价值不在“跑起来”，而在于：**让编译器能静态分析循环结构、依赖关系和优化机会。**

这也是为什么它在 MLIR 的 `affine` 与 `linalg` 里那么有价值：只要是“仿射访问模式”，编译器就能基于它做更强的重排、融合和代码生成。
