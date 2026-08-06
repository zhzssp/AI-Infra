#!/usr/bin/env python3
"""Step 05 — AutoTVM search loop + tuning-log reuse (config explosion antidote).

Core concepts:
  - schedule template → search space → measure → best config
  - apply_history_best: reuse log without re-searching (research question ⑥)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tvm_lab.common import banner, out_dir, require_tvm, write_text


def register_template():
    tvm = require_tvm()
    from tvm import autotvm, te

    @autotvm.template("ai_infra/matmul_lab")
    def matmul(N):
        A = te.placeholder((N, N), name="A", dtype="float32")
        B = te.placeholder((N, N), name="B", dtype="float32")
        k = te.reduce_axis((0, N), name="k")
        C = te.compute((N, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
        s = te.create_schedule(C.op)

        cfg = autotvm.get_config()
        cfg.define_split(
            "tile_i", C.op.axis[0], num_outputs=2, filter=lambda e: e.size[-1] in [4, 8, 16]
        )
        cfg.define_split(
            "tile_j", C.op.axis[1], num_outputs=2, filter=lambda e: e.size[-1] in [4, 8, 16]
        )
        xo, xi = cfg["tile_i"].apply(s, C, C.op.axis[0])
        yo, yi = cfg["tile_j"].apply(s, C, C.op.axis[1])
        s[C].reorder(xo, yo, xi, yi, k)
        return s, [A, B, C]

    return matmul


def main() -> None:
    banner("05 AutoTVM：搜索闭环 + tuning log 复用")
    tvm = require_tvm()
    from tvm import autotvm

    N = 64
    n_trial = int(os.environ.get("AUTOTVM_TRIALS", "12"))
    matmul = register_template()
    task = autotvm.task.create("ai_infra/matmul_lab", args=(N,), target="llvm")

    log_path = out_dir() / "tvm" / "05_autotvm.log"
    log_file = str(log_path)

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=20),
        runner=autotvm.LocalRunner(
            number=1, repeat=1, min_repeat_ms=0, enable_cpu_cache_flush=False, timeout=20
        ),
    )

    space_size = len(task.config_space)
    trials = min(n_trial, space_size)
    print(f"  搜索空间大小: {space_size}")
    print(f"  n_trial={trials}（环境变量 AUTOTVM_TRIALS 可调）")
    print(f"  log → {log_file}")

    t0 = time.time()
    tuner = autotvm.tuner.RandomTuner(task)
    tuner.tune(
        n_trial=trials,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(log_file)],
    )
    tune_sec = time.time() - t0

    t1 = time.time()
    with autotvm.apply_history_best(log_file):
        with tvm.target.Target("llvm"):
            s, arg_bufs = matmul(N)
            func = tvm.build(s, arg_bufs, target="llvm")
    reuse_sec = time.time() - t1

    a = tvm.nd.array(np.random.randn(N, N).astype("float32"))
    b = tvm.nd.array(np.random.randn(N, N).astype("float32"))
    c = tvm.nd.array(np.zeros((N, N), dtype="float32"))
    func(a, b, c)
    ok = np.allclose(c.numpy(), a.numpy() @ b.numpy(), rtol=1e-3, atol=1e-3)

    best_excerpt = ""
    if log_path.exists():
        lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        best_excerpt = lines[-1][:500] if lines else "(empty log)"

    note = f"""# AutoTVM 闭环实验结果

- 矩阵规模: {N}×{N}
- config space 大小: {space_size}
- n_trial: {trials}
- 搜索耗时: {tune_sec:.2f}s
- **复用 log 再 build 耗时: {reuse_sec:.2f}s**（不应再跑完整搜索）
- 数值正确: {ok}

## 用自己的话记住三句

1. **搜索空间**：这里是 `tile_i` / `tile_j` 的合法切分组合。
2. **实测 / cost model**：本 demo 用 RandomTuner + 真机测量；生产里用 XGBTuner 减少实测次数。
3. **log 复用**：`apply_history_best` = 「已经调过的配置不要重复调」——研究问题⑥「配置组合爆炸」的缓存策略。

最后一条 tuning 记录（摘录）：
```
{best_excerpt}
```
"""
    write_text(out_dir() / "tvm" / "05_READING.md", note)
    print(note)
    print("[OK] 步骤 05 完成。")


if __name__ == "__main__":
    main()
