# poison、UB 与「坏路径不会被用到」

> 来源：[`llvm-learning-guide.md`](../llvm-learning-guide.md) §2.7

## poison 表达什么

**poison** = 出错运算的**延期标记**（deferred error），不是「空」。

- 像 SQL `NULL` 的地方：会**传染**（多数指令：操作数 poison → 结果 poison）
- 不像 `NULL`：不表示缺失/未知，而表示「这次运算已踩线，结果不可信」
- **不是**异常捕获：不为运行时恢复服务，只为优化器投机

典型来源：违反 `nuw`/`nsw`/`inbounds` 等承诺。反直觉：`and %poison, 0` 数值必为 0，**仍是 poison**。

## UB 是什么

**UB（Undefined Behavior）** = 规范不允许发生的情况；一旦发生，编译器可做任何事（崩、看似正常、删代码……）。

| | poison | UB |
|--|--------|-----|
| 角色 | 「算坏了」的标签，还可继续传 | 标签用到**禁位**后引爆 |
| 禁位例子 | — | `store`/`load` 指针、`br` 条件、除数、`noundef` 参数等 |

poison 是贴纸；UB 是贴纸被当成真地址去用了。

## `select` 例外

`select` **不**按「任一操作数 poison → 全体 poison」：

```llvm
%r = select i1 %c, i32 %a, i32 %b
```

- `%c` 是 poison → 结果 poison
- `%c == true` → 结果是 `%a`；**另一臂 `%b` 即使 poison 也不污染 `%r`**

未选中的路可以带着 poison——支持 if-conversion：两路都算再选，坏路不能弄脏好路。

## 「坏路径实际不会被用到」

不是运行时去避开错误，而是：

> 你挂了 `nsw`/`inbounds` 等承诺 → 优化器有权假定**违反承诺的情况永不发生**，并按此改代码。若实际会违反，进入 UB。

poison 把「若发生了」先记成标签；真正用到禁位才引爆。`select` 未选中的臂 = 语义上「这条坏结果没被用到」。

## 和异常的对照

| | try/catch | poison |
|--|-----------|--------|
| 给谁 | 运行时恢复 | 编译期优化 |
| 行为 | 有定义的跳转 | 标签；误用 → UB |
| 意图 | 别崩、可恢复 | 假设犯规路径不发生 |
