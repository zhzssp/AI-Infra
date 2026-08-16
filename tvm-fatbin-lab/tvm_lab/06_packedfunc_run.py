#!/usr/bin/env python3
"""Step 06 — PackedFunc ABI + end-to-end build/run of the protagonist graph.

Core concepts:
  - PackedFunc: unified runtime calling convention across languages/devices
  - graph executor: topological dispatch of PackedFuncs
  - graph JSON 的 storage_id：静态内存规划的结果（相同 id = 共用同一块内存）

模型仍是全仓库主角 tiny_mlp（Gemm→Relu→Add），与 02 同一个图；
区别是 02 停在 IR 上看融合，这里一路 build 到能跑，看运行时怎么派发。

对应 docs/learning-guides/tvm-learning-guide.md §2.4 与 §6.2。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tvm_lab.common import banner, out_dir, require_tvm, write_text

# 与 02_fusion_relay.py / iree-lab/models/tiny_mlp.mlir 逐位相同
W = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0],
     [1.0, 1.0, 1.0]],
    dtype="float32",
)
B = np.full((4,), 0.5, dtype="float32")
BIAS2 = np.full((4,), 1.0, dtype="float32")
X = np.array([[1.0, 2.0, 3.0],
              [-4.0, -5.0, -6.0]], dtype="float32")
EXPECTED = np.array([[2.5, 3.5, 4.5, 7.5],
                     [1.0, 1.0, 1.0, 1.0]], dtype="float32")


def fetch_graph_json(lib) -> str | None:
    """graph JSON 的取法各版本不一，挨个试。"""
    for getter in (
        lambda: lib.get_graph_json(),
        lambda: lib.graph_json,
        lambda: getattr(lib, "graph", None),
    ):
        try:
            g = getter()
            if callable(g):
                g = g()
            if g is None:
                continue
            return g if isinstance(g, str) else json.dumps(g, indent=2)
        except Exception:
            continue
    return None


def analyze_storage(graph_json: str) -> str:
    """从 graph JSON 里把 storage_id 拿出来：相同 id = 共用同一块内存。"""
    try:
        g = json.loads(graph_json)
    except Exception as e:  # noqa: BLE001
        return f"（graph JSON 解析失败：{e}）"

    nodes = g.get("nodes", [])
    attrs = g.get("attrs", {})
    sids = attrs.get("storage_id", [None, []])[1]
    shapes = attrs.get("shape", [None, []])[1]
    names = [n.get("name", "?") for n in nodes]
    kinds = [n.get("op", "?") for n in nodes]

    lines = ["| # | op | name | shape | storage_id |", "|---|----|------|-------|-----------|"]
    for i, name in enumerate(names):
        sid = sids[i] if i < len(sids) else "?"
        shp = shapes[i] if i < len(shapes) else "?"
        lines.append(f"| {i} | `{kinds[i]}` | `{name}` | {shp} | **{sid}** |")

    uniq = len(set(sids)) if sids else 0
    lines.append("")
    lines.append(f"张量条目 {len(sids)} 个，去重后的 storage_id **{uniq}** 个。")
    lines.append("")
    lines.append("**这两个数字的差就是内存规划省下的东西**：id 相同的条目共用同一块内存。")
    lines.append("`GraphModule` 初始化时按去重后的 id 申请存储池，而不是每个张量各要一块。")
    return "\n".join(lines)


def main() -> None:
    banner("06 PackedFunc + graph executor：运行时派发（主角模型 tiny_mlp）")
    tvm = require_tvm()
    from tvm import relay

    x = relay.var("x", shape=(2, 3), dtype="float32")
    y = relay.nn.dense(x, relay.const(W))     # Gemm
    y = relay.add(y, relay.const(B))          # Gemm 的 C
    y = relay.nn.relu(y)                      # Relu
    y = relay.add(y, relay.const(BIAS2))      # Add
    mod = tvm.IRModule.from_expr(relay.Function([x], y))

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target="llvm")

    out = out_dir() / "tvm"
    graph_json = fetch_graph_json(lib)
    storage_table = "（本次未能取到 graph JSON，跳过内存规划分析）"
    fused_names: list[str] = []
    if graph_json is not None:
        write_text(out / "06_graph.json", graph_json)
        storage_table = analyze_storage(graph_json)
        try:
            fused_names = [
                n["attrs"]["func_name"]
                for n in json.loads(graph_json).get("nodes", [])
                if n.get("op") == "tvm_op"
            ]
        except Exception:
            fused_names = []

    from tvm.contrib import graph_executor

    module = graph_executor.GraphModule(lib["default"](tvm.cpu(0)))
    module.set_input("x", X)
    module.run()
    result = module.get_output(0).numpy()
    ok = bool(np.allclose(result, EXPECTED, atol=1e-5))

    print(f"\n  输出: {result.tolist()}")
    print(f"  期望: {EXPECTED.tolist()}")
    print(f"  数值正确: {ok}")
    if fused_names:
        print(f"\n  实际会被调用的 kernel（{len(fused_names)} 个）：")
        for fn in fused_names:
            print(f"    - {fn}")
        print("  名字带 fused_ 的就是融合组——一个名字 = 一次 PackedFunc 调用。")

    note = f"""# PackedFunc / graph executor 阅读指引

模型：全仓库主角 **tiny_mlp**（`x[2,3] → Gemm(W[4,3], b) → Relu → Add(bias2)`），
与 `02_fusion_relay.py` 是同一个图。02 停在 IR 上看融合，这里一路 build 到能跑。

- 数值正确: {ok}
- 输出: {result.tolist()}
- 期望: {EXPECTED.tolist()}（与 `iree-lab/models/tiny_mlp.mlir` 文末手算一致）

## 运行时在干什么

```
Relay 图 ──build──▶  graph JSON（拓扑 + 内存规划） + 每个融合组一个内核（PackedFunc）
                           │
                           ▼
              GraphModule.run：按拓扑序 set_input → 调 PackedFunc → get_output
```

实际会被调用的 kernel：{fused_names if fused_names else "（未取到）"}

**名字带 `fused_` 的就是融合组**——一个名字 = 一次 PackedFunc 调用。
源码写了 4 个 op，这里若只剩 1～2 个 kernel，就是 `FuseOps` 的功劳，
和 `02_READING.md` 里数 `fn (` 是同一件事的两个观察角度：
02 看的是**编译期的分组**，这里看的是**运行期真正被调用了几次**。

## 内存规划：storage_id

{storage_table}

## PackedFunc 是什么

跨语言 / 跨设备的统一 ABI：Python / C++ 都用同一套调用约定派发，
参数打包成 `TVMValue` 数组 + type code 数组。
**代价是每次调用有打包开销**，所以粒度不能太细——这反过来又是要融合的一个理由。

对照 IREE：那里的「派发」发生在 HAL `command_buffer.dispatch`，
调度序列在编译期就被编成了 VM 字节码；TVM 这里是 graph executor 在运行期读 JSON 拓扑。
**一个把调度编译掉了，一个留到运行期解释**——这是两套系统最根本的分歧，
见 [`iree-learning-guide.md`](../../../docs/learning-guides/iree-learning-guide.md) 第 1 章。
"""
    write_text(out / "06_READING.md", note)
    print("\n[OK] 步骤 06 完成。读 out/tvm/06_READING.md。")


if __name__ == "__main__":
    main()
