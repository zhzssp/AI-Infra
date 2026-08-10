# Op / Region / Block 与高级语言、Dialect 范围

> 来源：[`mlir-learning-guide.md`](../mlir-learning-guide.md) §2.1 · §3.1

## 嵌套关系

```text
Operation
 └─ Region（0..N，语义由所属 Op 规定）
      └─ Block（Region 内的基本块）
           └─ 更多 Operation（可再带 Region）…
```

一切都是 Op：`builtin.module`、`func.func` 也不例外。

## 和高级语言怎么对应

| 源码构造 | 通常落到 |
|----------|----------|
| 语句 / 表达式 / 声明 | **Op**（`addi`、`scf.if`、`func.return`…） |
| 函数体、then/else、循环体 | 所属 Op 的 **Region** |
| Region 里的直线段 | **Block** |

不是「一句语句 = 一个 Block」。`if` 本身是 Op；大括号里的两坨才是它的两个 Region。

```text
builtin.module                         ← 顶层 Op（文件根）
  └─ func.func @main                   ← 也是 Op
       └─ Region（函数体）
            └─ Block(s) + 内部的 scf.if 等
                 scf.if → Region then / Region else
```

## Block ≈ 基本块，但图可能还没拍平

在 **SSACFG Region** 里，Block 就是经典基本块（单入、terminator 结尾）。

差别：高层 `scf` 用 Region 嵌套包住 if/for，这些 Block 是**该 Region 内**的 BB，还不是整函数扁平 CFG。要到 `scf → cf` 之后才变成教科书那张多 BB 的网。

## Dialect 里你通常定义什么（排除法）

Dialect ≈ Op + Type + Attribute + Interface 的命名空间。

**一般不必自己再定义**（上游标准 dialect 已有）：

`builtin` / `func` / `arith` / `math` / `scf` / `cf` / `tensor` / `memref` / `linalg` / `vector` / `llvm` …

**自定义 dialect 只补领域层**：标准库表达不了或不想过早丢掉的算子/类型/属性（如 `toy.add`、`!toy.num`）。

`module`、`func` 不是「魔法非 Op」，只是**已写好、直接用**的普通 Op。
