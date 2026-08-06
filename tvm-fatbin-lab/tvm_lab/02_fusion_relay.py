#!/usr/bin/env python3
"""Step 02 — Graph-level fusion via Relay (four operator classes in action).

Core concepts:
  - injective / reduction / complex-out-fusable / opaque
  - fusion = deciding which nodes share one executable unit (subgraph partition prototype)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tvm_lab.common import banner, out_dir, require_tvm, write_text


def main() -> None:
    banner("02 Relay 图级融合：四类算子规则的实物")
    tvm = require_tvm()
    from tvm import relay

    # dense (complex-out-fusable) → add (injective) → relu (injective)
    # then an explicit fusion barrier, then another relu (new group).
    x = relay.var("x", shape=(1, 16), dtype="float32")
    w = relay.const(np.random.randn(8, 16).astype("float32"))
    b = relay.const(np.random.randn(8).astype("float32"))
    y = relay.nn.dense(x, w)
    y = relay.add(y, b)
    y = relay.nn.relu(y)
    # stop_fusion ≈ opaque boundary for teaching（真实 opaque 如 sort 也会切开）
    try:
        y = relay.annotation.stop_fusion(y)
    except Exception:
        # Fallback: cast round-trip often breaks fusion chains on older stacks.
        y = relay.cast(relay.cast(y, "float16"), "float32")
    y = relay.nn.relu(y)

    mod = tvm.IRModule.from_expr(relay.Function([x], y))
    before = mod.astext(show_meta_data=False)

    with tvm.transform.PassContext(opt_level=3):
        mod_opt = relay.transform.FoldConstant()(mod)
        mod_opt = relay.transform.FuseOps()(mod_opt)
    after = mod_opt.astext(show_meta_data=False)

    out = out_dir() / "tvm"
    write_text(out / "02_relay_before_fuse.ir.txt", before)
    write_text(out / "02_relay_after_fuse.ir.txt", after)

    guide = """# 阅读指引：融合前后 IR

对比 `02_relay_before_fuse.ir.txt` 与 `02_relay_after_fuse.ir.txt`：

1. `dense → add → relu`：典型 **complex-out-fusable + injective**，应融成同一 `fn` / fused function。
2. `stop_fusion`（或 cast 屏障）：模拟 **opaque** 切开点。
3. 之后的 `relu`：通常开启新的融合组。

**与「子图划分」的同构**：融合决策 =「这些节点要不要落在同一个可执行单元」。
ORT EP / ExecuTorch Partitioner / IREE `flow.dispatch` 是同一问题在多后端场景下的推广。

四类速记：
| 类别 | 例子 | 规则直觉 |
|------|------|----------|
| injective | add / relu | 彼此可融；可被两侧吸收 |
| reduction | sum / softmax 归约 | 可融合输入侧 injective |
| complex-out-fusable | conv / dense | 可融合输出侧 injective |
| opaque | sort / stop_fusion | 通常不融，切开边界 |
"""
    write_text(out / "02_READING.md", guide)

    print("\n--- FuseOps 之后（节选）---")
    print("\n".join(after.splitlines()[:60]))
    print("\n[OK] 步骤 02 完成。对照 out/tvm/02_READING.md 看融合切开点。")


if __name__ == "__main__":
    main()
