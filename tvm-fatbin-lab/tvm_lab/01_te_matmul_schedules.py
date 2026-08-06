#!/usr/bin/env python3
"""Step 01 — TE compute + two schedules: naive vs tiled+cache_write+vectorize.

Core concepts:
  - algorithm / schedule decoupling (Halide → TVM)
  - schedule primitives: tile, reorder, cache_write, compute_at, vectorize
  - read tvm.lower(..., simple_mode=True) and explain loop structure
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tvm_lab.common import banner, out_dir, require_tvm, write_text


def build_matmul(N: int = 64):
    require_tvm()
    from tvm import te

    A = te.placeholder((N, N), name="A", dtype="float32")
    B = te.placeholder((N, N), name="B", dtype="float32")
    k = te.reduce_axis((0, N), name="k")
    C = te.compute(
        (N, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="C",
    )
    return A, B, C


def schedule_naive(C):
    from tvm import te

    return te.create_schedule(C.op)


def schedule_tiled(C, tile: int = 16):
    """tiled + cache_write + compute_at + vectorize — classic locality schedule."""
    from tvm import te

    s = te.create_schedule(C.op)
    CL = s.cache_write(C, "global")
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], tile, tile)
    s[CL].compute_at(s[C], yo)
    _, yi_inner = s[CL].split(s[CL].op.axis[1], factor=8)
    s[CL].vectorize(yi_inner)
    return s


def main() -> None:
    banner("01 TE matmul：naive vs tiled+cache_write+vectorize")
    tvm = require_tvm()

    A, B, C = build_matmul(64)
    s_naive = schedule_naive(C)
    A2, B2, C2 = build_matmul(64)
    s_tiled = schedule_tiled(C2, tile=16)

    lower_naive = str(tvm.lower(s_naive, [A, B, C], simple_mode=True))
    lower_tiled = str(tvm.lower(s_tiled, [A2, B2, C2], simple_mode=True))

    out = out_dir() / "tvm"
    write_text(out / "01_matmul_naive.lower.txt", lower_naive)
    write_text(out / "01_matmul_tiled.lower.txt", lower_tiled)

    guide = """# 阅读指引：两份 lower 输出差在哪

打开 `01_matmul_naive.lower.txt` 与 `01_matmul_tiled.lower.txt`，对照标注：

| 原语 | 在 tiled 版里看什么 |
|------|---------------------|
| `tile` | 外层块循环 + 内层块内循环（四层嵌套） |
| `cache_write` | 出现临时缓冲（CL）承接累加 |
| `compute_at` | CL 的计算嵌进 C 的某个 tile 层内部 |
| `vectorize` | 最内层标成向量化（SIMD） |

**心智模型**：TE compute 只定义「每个输出元素怎么算」；schedule 决定循环顺序、暂存与向量化。
同一算法可以对应成千上万种实现——这就是搜索（AutoTVM）的前提。
"""
    write_text(out / "01_READING.md", guide)

    print("\n--- naive lower（前 40 行）---")
    print("\n".join(lower_naive.splitlines()[:40]))
    print("\n--- tiled lower（前 50 行）---")
    print("\n".join(lower_tiled.splitlines()[:50]))
    print("\n[OK] 步骤 01 完成。先读 out/tvm/01_READING.md，再对比两份 .lower.txt。")


if __name__ == "__main__":
    main()
