#!/usr/bin/env python3
"""Step 06 — PackedFunc ABI + end-to-end build/run of a tiny Relay graph.

Core concepts:
  - PackedFunc: unified runtime calling convention across languages/devices
  - graph executor: topological dispatch of PackedFuncs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tvm_lab.common import banner, out_dir, require_tvm, write_text


def main() -> None:
    banner("06 PackedFunc + graph executor：运行时派发")
    tvm = require_tvm()
    from tvm import relay

    w_np = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype="float32")  # (units=2, in=4)
    x = relay.var("x", shape=(1, 4), dtype="float32")
    w = relay.const(w_np)
    y = relay.nn.dense(x, w)  # (1, 2)
    y = relay.nn.relu(y)
    mod = tvm.IRModule.from_expr(relay.Function([x], y))

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target="llvm")

    out = out_dir() / "tvm"
    graph_json = None
    for getter in (
        lambda: lib.get_graph_json(),
        lambda: lib.graph_json,
        lambda: getattr(lib, "graph", None),
    ):
        try:
            graph_json = getter()
            if callable(graph_json):
                graph_json = graph_json()
            if graph_json is not None:
                break
        except Exception:
            continue
    if graph_json is not None:
        if not isinstance(graph_json, str):
            graph_json = json.dumps(graph_json, indent=2)
        write_text(out / "06_graph.json", graph_json)

    from tvm.contrib import graph_executor

    module = graph_executor.GraphModule(lib["default"](tvm.cpu(0)))
    inp = np.array([[-1.0, 2.0, -3.0, 4.0]], dtype="float32")
    module.set_input("x", inp)
    module.run()
    result = module.get_output(0).numpy()
    expected = np.maximum(inp @ w_np.T, 0)
    ok = np.allclose(result, expected)

    note = f"""# PackedFunc / graph executor 阅读指引

- 数值正确: {ok}
- 输出: {result.tolist()}
- 期望: {expected.tolist()}

## 运行时在干什么

```
Relay 图 ──build──▶  graph JSON（拓扑） + 多个算子内核（每个是 PackedFunc）
                           │
                           ▼
              GraphModule.run：按拓扑序 set_input → 调 PackedFunc → get_output
```

**PackedFunc** 是跨语言 / 跨设备的统一 ABI：Python / C++ 都能用同一套调用约定派发。
对照 IREE：那里的「派发」发生在 HAL `command_buffer.dispatch`；TVM 这里发生在 graph executor。

若存在 `06_graph.json`，打开看节点列表与 `func_name`——融合后的节点名往往带 `fused_`。
"""
    write_text(out / "06_READING.md", note)
    print(note)
    print("[OK] 步骤 06 完成。")


if __name__ == "__main__":
    main()
