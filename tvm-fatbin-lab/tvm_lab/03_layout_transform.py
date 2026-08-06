#!/usr/bin/env python3
"""Step 03 — Layout transform & why cross-backend boundaries pay tax.

Core concepts:
  - logical shape ≠ physical layout (NCHW vs NHWC)
  - layout_transform inserts conversion ops (research question ②)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tvm_lab.common import banner, out_dir, require_tvm, write_text


def main() -> None:
    banner("03 Layout：NCHW → NHWC 变换与边界代价")
    tvm = require_tvm()
    from tvm import relay

    n, c, h, w = 1, 3, 8, 8
    x = relay.var("x", shape=(n, c, h, w), dtype="float32")
    # Prefer NHWC for a hypothetical consumer (many mobile / DSP backends).
    y = relay.layout_transform(x, src_layout="NCHW", dst_layout="NHWC")
    # A trivial injective op after transform — shows layout is sticky.
    y = relay.nn.relu(y)
    mod = tvm.IRModule.from_expr(relay.Function([x], y))
    ir_text = mod.astext(show_meta_data=False)

    # Execute to prove semantic equivalence of element order.
    with tvm.transform.PassContext(opt_level=1):
        lib = relay.build(mod, target="llvm")
    dev = tvm.cpu(0)
    # Prefer graph executor; fall back if API renamed.
    try:
        from tvm.contrib import graph_executor

        module = graph_executor.GraphModule(lib["default"](dev))
    except Exception:
        from tvm.contrib.executor import GraphModule

        module = GraphModule(lib["default"](dev))

    inp = np.arange(n * c * h * w, dtype="float32").reshape(n, c, h, w)
    module.set_input("x", inp)
    module.run()
    out_nhwc = module.get_output(0).numpy()
    expected = np.transpose(np.maximum(inp, 0), (0, 2, 3, 1))
    ok = np.allclose(out_nhwc, expected)

    out = out_dir() / "tvm"
    write_text(out / "03_layout_ir.txt", ir_text)
    note = f"""# Layout 变换实验结果

- 输入形状 (NCHW): {inp.shape}
- 输出形状 (NHWC): {out_nhwc.shape}
- 数值正确: {ok}

## 为什么这是研究问题②的原型

跨后端委托时，若 EP_A 偏好 NCHW、EP_B 偏好 NHWC，分区边界必须插入 `layout_transform`
（或等价的 Transpose / Reformat）。变换本身吃带宽——有时比多跑一个 `add` 更贵。

好的划分会把 layout 冲突推到图边缘，或与计算融合；坏的划分会在每个边界都付转换税。
"""
    write_text(out / "03_READING.md", note)
    print(ir_text)
    print(f"\n[OK] 步骤 03 完成。数值正确={ok}。读 out/tvm/03_READING.md。")


if __name__ == "__main__":
    main()
