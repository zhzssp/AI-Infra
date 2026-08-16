# examples/upstream —— 上游 dialect 的最小可跑样例

> **这些文件不属于 `toy` dialect，也不参与本项目的构建。**
> 它们用 conda 环境里现成的 `mlir-opt` / `mlir-translate` 跑，`toy-opt` 不认识它们。

## 为什么要有这个目录

`toy` 是一个刻意做小的**标量 `i32`** dialect。这个选择让它把 Operation / Region /
Trait / Interface / Dialect Conversion 讲得极干净，代价是有三件事它**根本表达不了**：

| 讲不了的东西 | 因为 | 落在哪个文件 |
|-------------|------|------------|
| 张量语义、索引映射、结构化 op | `toy` 只有标量，没有 shape 概念 | `01-linalg-generic.mlir` |
| bufferization（tensor → memref） | 没有张量就没有"值语义 vs 缓冲"的分野 | `02-bufferize.mlir` |
| 与 LLVM 的真实接缝（memref descriptor） | `toy` 降到 `low` 就停了，没接真后端 | `03-memref-to-llvm.mlir` |

本项目 README「刻意留白」一节把 bufferization 和「降到 LLVM Dialect」列为**遇到再学**——
这个判断没变。这里补的不是完整子系统，而是**每个概念一份最小可跑的实物**，
让学习指南里讲这三节时有代码可指，而不是凭空写一段 IR。

## 主角是同一个

三个文件用的都是全仓库的图级主角 `tiny_mlp`（`Gemm → Relu → Add`）的片段，
batch 取 2，所以 shape 是 `[2,3] @ [3,4] → [2,4]`。

这样一来：

```text
onnx-delegate-lab 里的 tiny_mlp        ← 站 ①② 表示与划分
        ↓ 同一个模型
examples/upstream/01 的 linalg 版      ← 站 ⑤ 多层降低
        ↓ 同一个 Relu
examples/upstream/03 降出来的 .ll      ← 站 ⑤ 交给 站 ⑥ 的那一刻
        ↓ 同一段 select
llvm-hello-compile 里 relu_sum 的 IR   ← 站 ⑥ 指令生成
```

链路全图见 [`../../../docs/learning-guides/00-end-to-end-pipeline.md`](../../../docs/learning-guides/00-end-to-end-pipeline.md)。

## 跑法

```bash
cd mlir-toy-dialect
bash scripts/run_upstream.sh          # 一次跑完三个文件，产物落 out/upstream/
```

不想跑脚本时，每个 `.mlir` 文件头部都写了可以直接复制的 `mlir-opt` 命令。

## 版本提示

conda 环境里是 MLIR 17.0.6。跨版本最容易变的两处：

- `--finalize-memref-to-llvm` 在老版本里叫 `--convert-memref-to-llvm`（脚本会自动试两个名字）。
- 样例里一律用 `arith.cmpf` + `arith.select` 而不是 `arith.maxf` / `arith.maximumf`——
  后者在 LLVM 18 前后改过名字，前者所有版本都能跑，而且和 `relu_sum` 降出来的 IR 同形。
