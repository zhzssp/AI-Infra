# Block argument、phi 与到达定值

> 来源：[`mlir-learning-guide.md`](../learning-guides/mlir-learning-guide.md) §2.3 · [`llvm-learning-guide.md`](../learning-guides/llvm-learning-guide.md) §2.3  
> 相关：[llvm-phi.md](./llvm-phi.md)

## 分工：控制流 vs 数据流

| | 管什么 | 例子 |
|--|--------|------|
| 条件分支 | **控制流**：下一跳去哪 | `br` / `cf.cond_br` |
| `phi` / block argument | **数据流汇合**：多路径定值合成一个 SSA 名 | 见下 |

`phi` **不是**分支逻辑本身；分支造成汇合点，phi/block-arg 解决「两边各有一个定义，汇合后用哪个」。

## Block argument 是什么

基本块入口的**形参**。跳进来时由前驱 terminator **传值**，进块后统一用这个名字：

```mlir
cf.br ^merge(%a : i32)    // 边带来的定值
cf.br ^merge(%b : i32)

^merge(%r: i32):          // 接收 → 之后只用 %r
```

与 LLVM 语义等价：

```llvm
%r = phi i32 [ %a, %bb1 ], [ %b, %bb2 ]
```

只是「边 → 值」写在跳转上，而不是写在后继块首的 phi 表里。MLIR 这样更利于改 CFG、嵌套 Region、做 verifier。

## 表达的是到达定值，不是活跃变量

**到达定值**：这个 use **可能读到**前面哪些赋值。  
**活跃变量**：这个定值之后**还会不会被读**（另一类分析）。

Phi / block-arg = 把汇合点上**多条到达定值**写成 IR 的显式合并：

```text
定值1 (%a)     定值2 (%b)
     \            /
      v          v
   合成一个 SSA 名 %r   ← phi 或 block argument
           │
           v
        use(%r)        ← 只有一个到达定值：%r
```

不是「前驱传来的所有活跃变量都变成参数」；只有**不同前驱上定义不同**的值才需要合并。

## SSA 值如何对应到「变量」

进 SSA 后 IR 里通常**没有**一等「变量」，只有**值**。对应关系在**构造时**建立：

| 层 | 例子 |
|----|------|
| 源变量 / `alloca` | C 的 `r`，或 `%r = alloca` |
| SSA 值 | `%r1`、`%r2`、phi/block-arg 的 `%r3`（`r` 的不同版本） |

实务路径：前端用 `alloca`+`store`/`load` 表示变量 → `mem2reg` 提升为 SSA 值 + 必要的 phi。  
之后优化只认 def–use；「对应源变量谁」靠构造对应关系与可选调试信息，不是运行时绑定。

## 作用

1. 维持 SSA：每个名字单次定值，use 有唯一 def  
2. 把到达定值固化进 IR，方便常量传播、DCE、GVN 等  
3. 控制流与数据流分工清楚：分支改边，phi/block-arg 在边上合并数据  
