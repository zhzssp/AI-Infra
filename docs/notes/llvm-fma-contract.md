# FMA 与 `contract` 标志

> 来源：[`llvm-learning-guide.md`](../llvm-learning-guide.md) §2.6（指令标志 / 浮点 fast-math）

## FMA 是什么

**FMA（Fused Multiply-Add）**：一条运算算 `a*b+c`，中间结果**只舍入一次**。

| 写法 | 舍入 |
|------|------|
| `t = a*b; r = t+c` | 两次（严格 IEEE 默认路径） |
| `r = fma(a,b,c)` | 一次（内部更高精度乘完再加） |

更快，且常更准，但与「先 mul 再 add」**不一定 bit-identical**。LLVM 对应 `@llvm.fma`，后端落到 `vfmadd` / `ffma` 等。

## `contract` 开关

浮点 fast-math 标志，挂在 `fmul`/`fadd` 上，表示：**允许把表达式收缩成融合形式**（典型：`a*b` + `c` → FMA）。

```llvm
%t = fmul contract float %a, %b
%r = fadd contract float %t, %c
; → 可变成 call @llvm.fma.f32(...)
```

- **有 `contract`**：允许改舍入语义以生成 FMA  
- **无 `contract`**：即使硬件有 FMA，编译器也常故意不用  
- **`fast`**：一揽子放宽（含 `contract`、`reassoc` 等）

`contract` = 软件侧许可；硬件 FMA = 能否便宜地真融合。

## 和「算子融合」的关系

本质都是把 mul+add 合成更粗的单元，但层级不同：

| | FMA / `contract` | 图上的算子融合 |
|--|------------------|----------------|
| 层级 | 指令 / 算术表达式 | 图 / kernel（如 Conv+BN+ReLU） |
| 主要收益 | 少舍入 + 少指令 | 少全局访存、少 launch |
| 语义代价 | 改变浮点舍入（需授权） | 通常保持约定语义 |

AI 栈常叠用：图融合减访存，落到后端再用 `contract`/FMA 吃硬件乘加。

## 要不要硬件支持

**真 FMA（一次舍入 + 通常一条指令）需要硬件 FMA。**  
无硬件时：多半退回两次运算（不是真 FMA），或用很慢的软件模拟。现代 GPU/多数 CPU（x86 FMA3、ARM 等）都有；matmul / 点积路径几乎总依赖它。
