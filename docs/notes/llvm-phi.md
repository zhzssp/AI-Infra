# LLVM IR 中的 Phi

> 来源：[`llvm-learning-guide.md`](../llvm-learning-guide.md) §2.3

## 是什么

**Phi（`phi`）是 LLVM IR 里在控制流汇合点“选值”的指令**，名字来自数学里的 φ 函数。

LLVM IR 是 **SSA（Static Single Assignment）**：每个变量只能被定义一次。分支两边各自算出一个值、汇合后再用时，不能写两次赋值，所以用 `phi`：

```c
int r;
if (cond)
  r = a;
else
  r = b;
// 这里要用 r
```

对应 IR：

```llvm
merge:
  %r = phi i32 [ %a, %then ], [ %b, %else ]
```

含义：**从 `%then` 进来则 `%r = %a`；从 `%else` 进来则 `%r = %b`。**

## 两条规则

1. **`phi` 必须放在基本块最开头**（所有非 phi 指令之前）
2. **每个前驱基本块恰好对应一个入口**——`[%value, %pred]` 成对出现

## 和 MLIR 的关系

| LLVM IR | MLIR |
|---------|------|
| `phi` | block argument（如 `^bb2(%r: i32):`） |

语义等价；MLIR 用 block argument 避免了「phi 必须在块首」「前驱要对齐」等结构约束。

## 什么时候会看到

跑 `mem2reg` 或 `-O2` 后，栈上的 `alloca` / `load` / `store` 常被提升成寄存器形式的 SSA，**汇合点就会出现 `phi`**（「`alloca` 消失、`phi` 出现」）。

## 延伸

phi 与 MLIR block argument、到达定值、SSA 值和源变量的对应：见 [mlir-block-arg-ssa.md](./mlir-block-arg-ssa.md)。
