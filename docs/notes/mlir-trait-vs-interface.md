# Trait vs Interface：能力边界、候选与正确性

> 来源：[`mlir-learning-guide.md`](../mlir-learning-guide.md) §3.4 · 第 4 章  
> 相关：[mlir-type-attr-interface.md](./mlir-type-attr-interface.md)

## 内置与自定义

- 常用 Trait 多是**内置**的：`Pure`、`Commutative`、`Terminator`、`IsolatedFromAbove`、`SingleBlock`…
- **可以自定义** Trait / Interface；实践上：布尔标签用 Trait，要「做事」用 Interface。

| | 自定义 Trait | 自定义 Interface |
|--|-------------|------------------|
| 像什么 | 贴纸 `hasTrait` | 方法契约 `dyn_cast` 后调用 |
| 擅长 | 是/否、门禁 | 打分、按 op 定制重写 |
| 选法 | 只要分类 | 行为因 op 而异 |

## 融合例子：边界在哪

设定：`toy.relu` / `toy.add` 挂 `[Elementwise]`；`toy.matmul` 不挂。

**Trait 能做的：**

```text
两相邻 op 都有 Elementwise + Pure + 数据流相连
→ 合成 fused_elementwise
```

门禁级优化：过滤候选、简单改写。  
**做不到：** 估计省多少访存、各 op 如何编进融合体、按代价决定融不融。

**Interface 能做的**（示意 `FusableOpInterface`）：

- `isElementwise()` / `estimateBytesTouched()` / `mergeInto(builder)`
- 同一 FusionPass 可服务 `toy.*` 与 `low.*`（跨 dialect）
- 接近 `BufferizableOpInterface` 的用法：Pass 只认契约

叠用：`Trait` 便宜过滤 → `Interface` 打分与生成。

> **Trait = 门禁；Interface = 策略与重写。**

## 候选是谁：不是一次性全体

Trait/Interface 是**过滤器**，不负责「选出全世界」。

```text
按窗口遍历 IR（单 Op / producer–consumer 对 / 某个 Region）
  → 问 hasTrait / dyn_cast<Interface>
  → 合格才变换
```

融合候选通常是**当前这一对**相邻 op，不是所有 Elementwise 收进一个大集合一次融完。  
贪心 pattern 多轮局部匹配直到固定点——仍是逐步、局部。

## 正确性从哪来

保证的是**每步语义等价**，不是局部贪心 ⇒ 全局最优。

1. **适用条件要够强**：数据流、单消费者、无非法副作用、类型兼容…  
2. **局部保义可组合**：每步 ≡ 则多步仍 ≡（与是否一次看完全局无关）  
3. **安全网**：Verifier、SSA/dominance、Interface 契约、测试  

| | |
|--|--|
| 正确性 | 规则在条件下保义 + IR 不变量 |
| 最优性 | 不保证（phase ordering：先融 A 可能挡住更好的 B） |

条件太松，即使全局看一遍也会错；条件够强，局部候选也可以对。
