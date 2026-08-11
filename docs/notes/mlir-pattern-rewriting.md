# Pattern Rewriting：机制、子图与 Dialect Conversion

> 来源：[`mlir-learning-guide.md`](../mlir-learning-guide.md) 第 5 章 · 第 6 章

## 是什么

**Pattern rewriting = 通用重写机制**：匹配一块局部 IR → 替换成另一块。  
**不专属于优化**——也用于 canonicalize、lowering、方言转换等。

## 「子图」如何定义

不是先画一张独立几何图，而是：

1. **根（root）**：如 `OpRewritePattern<MulOp>`，扫到该 op 时试这条规则  
2. **沿 use-def 边检查**：`getDefiningOp` / `hasOneUse` / 属性与类型约束  
3. **不满足 → `failure()`**；满足 → 用 `PatternRewriter` 创建/替换  

例：`mul(x, 1) → x` —— 子图 = Mul 根 + 一侧操作数是常量 1。  
更大子图同理：`add` 为根，检查 `lhs` 的 defining op 是否为 `mul`，再替换。

驱动器：工作列表上的当前 op 为根，只向**局部邻居**长出子图；不是一次匹配全程序。

## 细粒度 vs 宏观效果

- **细的是匹配视野**（局部邻域），不是「只能优化小算子」——根可以是 `linalg.matmul` 等大 op。  
- 单条 pattern **不保证**端到端最优。宏观靠：高层 dialect 保留结构、Pass 流水线、带分析的专用 Pass / 搜索。  
- Pattern 常当「刷子」或 lowering 手段；全局调度别只靠一条超级子图。

## 和 Dialect Conversion 的关系

跨 dialect 时**大量仍用 pattern**（多为 `ConversionPattern`），但外面还有 Conversion 框架：

| | 普通贪心 Rewrite | Dialect Conversion |
|--|------------------|-------------------|
| Pattern | `RewritePattern` | `ConversionPattern`（+ remapped operands） |
| 做完 | 无规则再命中 | illegal op 均合法化 |
| 额外 | — | `ConversionTarget` + 可选 `TypeConverter` |
| 漏规则 | 可能静默残留 | 常失败并指出 |

> **方言转换 = Pattern 做重写 + Conversion 管合法/类型/是否做完。**

## 一句话

**Pattern rewriting 是局部重写引擎；子图用「根 + use-def 约束」在代码里定义；dialect 转换也走这套（ConversionPattern），并不是「只做细粒度计算优化」。**
