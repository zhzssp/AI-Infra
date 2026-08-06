#!/usr/bin/env python3
"""Step 04 — tensorize: map a loop tile onto a micro-kernel (heterogeneous hook).

Core concepts:
  - vectorize = SIMD on one axis; tensorize = replace a loop nest tile with an intrinsic
  - this is the hook for Tensor Core / vendor matrix units
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tvm_lab.common import banner, out_dir, require_tvm, write_text


def _intrin_func(ins, outs):
    from tvm import tir

    aa, bb = ins
    cc = outs[0]
    ib = tir.ir_builder.create()
    with ib.for_range(0, 16, name="i") as i:
        with ib.for_range(0, 4, name="j") as j:
            acc = ib.allocate("float32", (1,), name="acc", scope="local")
            acc[0] = tir.const(0.0, "float32")
            with ib.for_range(0, 16, name="k") as k:
                acc[0] = acc[0] + aa.vload([i, k]) * bb.vload([k, j])
            ib.emit(cc.vstore([i, j], acc[0]))
    return ib.get()


def main() -> None:
    banner("04 tensorize：把循环块替换成张量微内核")
    tvm = require_tvm()
    from tvm import te

    M, N, K = 64, 64, 64
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    s = te.create_schedule(C.op)

    # Tile to match the intrinsic shape: 16 x 4, with k factor 16.
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], 16, 4)
    ko, ki = s[C].split(k, factor=16)
    s[C].reorder(xo, yo, ko, xi, yi, ki)

    status = "tensorize applied"
    try:
        a = te.placeholder((16, 16), name="a")
        b = te.placeholder((16, 4), name="b")
        kk = te.reduce_axis((0, 16), name="kk")
        c = te.compute(
            (16, 4),
            lambda i, j: te.sum(a[i, kk] * b[kk, j], axis=kk),
            name="c",
        )
        Ab = te.decl_buffer(a.shape, a.dtype, name="A", offset_factor=1, strides=[te.var("sa"), 1])
        Bb = te.decl_buffer(b.shape, b.dtype, name="B", offset_factor=1, strides=[te.var("sb"), 1])
        Cb = te.decl_buffer(c.shape, c.dtype, name="C", offset_factor=1, strides=[te.var("sc"), 1])
        gemm = te.decl_tensor_intrin(c.op, _intrin_func, binds={a: Ab, b: Bb, c: Cb})
        s[C].tensorize(xi, gemm)
        lowered = str(tvm.lower(s, [A, B, C], simple_mode=True))
    except Exception as e:
        lowered = str(tvm.lower(s, [A, B, C], simple_mode=True))
        status = (
            f"tensorize skipped ({type(e).__name__}: {e}); "
            "showing tiled IR — still enough to contrast with vectorize"
        )

    out = out_dir() / "tvm"
    write_text(out / "04_tensorize.lower.txt", lowered)
    guide = f"""# tensorize 阅读指引

状态：{status}

## vectorize vs tensorize

| | vectorize | tensorize |
|--|-----------|-----------|
| 替换什么 | 一条循环轴 → SIMD 指令 | **一整块循环嵌套** → 张量指令 / 微内核 |
| 典型硬件 | AVX / NEON | Tensor Core / WMMA / 厂商 MMA |
| 异构接入 | 有限 | **关键钩子**：新硬件 = 实现 intrinsic + schedule 里 `tensorize` |

在 `04_tensorize.lower.txt` 里找：tile 后的 `xi/yi/ki` 区域是否被替换成 intrinsic / 微内核。
即使本机 TE 版本对 buffer bind 较严导致回退，也请记住：
**异构后端接入的编译器接口 =「声明一块可替换的计算 + 绑到硬件原语」。**
"""
    write_text(out / "04_READING.md", guide)
    print("\n".join(lowered.splitlines()[:50]))
    print(f"\n[OK] 步骤 04 完成（{status}）。读 out/tvm/04_READING.md。")


if __name__ == "__main__":
    main()
