# LLVM Pass 调度：analysis 与 transform 如何交错

> 来源：[`llvm-learning-guide.md`](../learning-guides/llvm-learning-guide.md) §3.4–§3.6

## 是什么

LLVM 的 pass 调度不是“先全做 analysis，再全做 transform”，而是由 `PassManager` 按预先构造好的 pipeline 执行。transform pass 在运行时按需向 `AnalysisManager` 请求分析结果，analysis 命中缓存就直接复用，失效了才重算。

因此，analysis 和 transform 是在**同一条流水线里交错出现**的：pass 负责改 IR，analysis 负责提供证据，框架负责缓存、失效和重算。

## 调度由谁控制

1. **pipeline 构建者**决定 pass 的顺序和层级
   - 例如 `PassBuilder` 的 `default<O2>`
   - 或者程序员手写 `ModulePassManager` / `FunctionPassManager`
2. **PassManager** 负责按层执行 pass，并通过 adaptor 嵌套子层 pass
3. **AnalysisManager** 负责分析结果缓存
4. **PreservedAnalyses** 负责告诉框架哪些分析仍然有效

## 关键规则

1. **调度顺序通常是显式设计好的，不是自动搜索最优解**。
2. **analysis 只在需要时计算**，不会预先把所有分析都跑一遍。
3. **transform pass 改 IR 后必须诚实声明失效关系**，否则后续 pass 可能复用过期结果。
4. **默认 pipeline 的好效果来自经验化的 pass 顺序**，例如先规范化、再内联、再清理、最后向量化。

## 为什么这样设计

- 避免无谓计算：很多分析结果不一定会被用到
- 避免过期信息：IR 一改，相关分析就可能失效
- 让 pass 可以复用缓存：同一个分析在多个 pass 间共享
- 让优化顺序可控：前面的 pass 为后面的 pass 创造机会

## 记忆方式

可以把 LLVM 想成：

> **PassManager 负责排班，AnalysisManager 负责体检报告缓存，Transform pass 负责真正动刀。**

