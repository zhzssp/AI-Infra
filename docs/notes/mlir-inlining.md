# 内联（Inlining）与 DialectInlinerInterface

> 来源：[`mlir-learning-guide.md`](../mlir-learning-guide.md) §4.3  
> 相关：LLVM 中端的 `inline`（CGSCC pass）；`llvm-hello-compile` 里 `-O2` 后小函数消失

## 编译里的 inlining 是什么

把**被调函数的函数体嵌进调用点**，去掉这次 `call`：

```c
int square(int x) { return x * x; }
int f(int a) { return square(a) + 1; }
// 内联后概念上：
int f(int a) { return (a * a) + 1; }
```

收益：

1. 少一次调用约定（传参、跳转、返回）开销  
2. **更重要**：打开调用点周围的后续优化（如 `square(3)` → 直接折成 `9`）

代价：代码可能变大 → 编译器按启发式**选择性**内联，不是能融全融。

## 和 `#include` 的差别

都像「把别处代码弄到使用点」，但：

| | `#include` | inlining |
|--|------------|----------|
| 何时 | 预处理，文本粘贴 | 中端，在 IR/函数上做 |
| 对象 | 整份头文件 | **一次调用**对应的函数体 |
| 是否总做 | 写了就包含 | 按代价决定 |

## 「dialect 参与 inlining」指什么

MLIR 有通用 **Inliner Pass**，但各方言的函数/调用/Region 约束不同，不能写死「凡 call 都内联」。

方言通过 **`DialectInlinerInterface`**（DialectInterface，挂在整个 dialect 上）回答例如：

- 能不能**内联进来**（别的函数体能否嵌进我的 region）  
- 能不能**内联出去**（我的函数能否被嵌进调用方）  
- 内联时对本 dialect 的 op 如何处理  

≠ OpInterface（那是单个 op 的能力，如 bufferize）。

## 一句话

**Inlining = 调用点展开函数体；dialect 参与 = 用 `DialectInlinerInterface` 告诉通用 Inliner 本方言允不允许、怎么内联。**
