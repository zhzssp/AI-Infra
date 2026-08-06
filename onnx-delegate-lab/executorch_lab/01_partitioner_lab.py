#!/usr/bin/env python3
"""Step ET — Minimal Partitioner: per-node vs connected-component tagging.

If ExecuTorch is installed, runs a real to_edge/to_backend path.
Otherwise runs a torch.fx simulation that still teaches delegation_tag / boundaries.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_common import banner, out_dir, write_text


class ToyModel:
    """+ * - / * +  — portable ops (-,/) deliberately split add/mul regions."""

    def forward_ops(self):
        return ["add", "mul", "sub", "div", "mul", "add"]


def simulate_partition(ops: List[str], mode: str) -> dict:
    """mode: per_node | connected — tag only add/mul."""
    tags = {}
    node_tags = []
    if mode == "per_node":
        pid = 0
        for i, op in enumerate(ops):
            if op in ("add", "mul"):
                tag = f"addmul_{pid}"
                pid += 1
                tags[tag] = "DemoBackend"
                node_tags.append((i, op, tag))
            else:
                node_tags.append((i, op, None))
    else:
        # Connected components of add/mul; sub/div break the chain.
        pid = 0
        current = None
        for i, op in enumerate(ops):
            if op in ("add", "mul"):
                if current is None:
                    current = f"addmul_{pid}"
                    pid += 1
                    tags[current] = "DemoBackend"
                node_tags.append((i, op, current))
            else:
                current = None
                node_tags.append((i, op, None))

    # Boundaries: edge between tagged region and untagged (or different tags).
    boundaries = []
    for i in range(len(node_tags) - 1):
        t0, t1 = node_tags[i][2], node_tags[i + 1][2]
        if t0 != t1:
            boundaries.append(
                {
                    "between": f"{node_tags[i][1]}#{i} → {node_tags[i+1][1]}#{i+1}",
                    "tags": [t0, t1],
                }
            )

    delegate_subgraphs = len(tags)
    return {
        "mode": mode,
        "node_tags": node_tags,
        "delegation_tags": tags,
        "delegate_subgraph_count": delegate_subgraphs,
        "boundary_count": len(boundaries),
        "boundaries": boundaries,
    }


def try_real_executorch(out: Path) -> Optional[dict]:
    try:
        import torch
        from torch.export import export
    except Exception as e:
        return {"status": "no_torch", "error": str(e)}

    try:
        from executorch.exir import to_edge
        from executorch.exir.backend.backend_details import DelegationSpec, PartitionResult
        from executorch.exir.backend.partitioner import Partitioner
        from executorch.exir.dialects._ops import ops as exir_ops
    except Exception as e:
        return {"status": "no_executorch", "error": str(e)}

    _ADD = exir_ops.edge.aten.add.Tensor
    _MUL = exir_ops.edge.aten.mul.Tensor
    _DEMO = "BackendWithCompilerDemo"

    class AddMulPartitioner(Partitioner):
        def __init__(self, connected: bool) -> None:
            self.connected = connected
            self.delegation_spec = DelegationSpec(_DEMO, [])
            self.partition_tags: Dict[str, DelegationSpec] = {}

        def partition(self, exported_program):
            graph = exported_program.graph_module.graph
            partition_id = 0
            current = None
            for node in graph.nodes:
                if node.op != "call_function":
                    continue
                if node.target not in (_ADD, _MUL):
                    current = None
                    continue
                if self.connected:
                    if current is None:
                        current = f"addmul_{partition_id}"
                        partition_id += 1
                        self.partition_tags[current] = self.delegation_spec
                    node.meta["delegation_tag"] = current
                else:
                    tag = f"addmul_{partition_id}"
                    partition_id += 1
                    node.meta["delegation_tag"] = tag
                    self.partition_tags[tag] = self.delegation_spec
            return PartitionResult(
                tagged_exported_program=exported_program,
                partition_tags=self.partition_tags,
            )

    class Model(torch.nn.Module):
        def forward(self, x, y):
            x = x + y
            x = x * y
            x = x - y
            x = x / y
            x = x * y
            x = x + y
            return x

    results = {"status": "real_executorch", "runs": []}
    m = Model().eval()
    inputs = (torch.randn(1, 3), torch.randn(1, 3))

    for connected, name in ((False, "per_node"), (True, "connected")):
        try:
            ep = export(m, inputs)
            edge = to_edge(ep)
            edge2 = edge.to_backend(AddMulPartitioner(connected=connected))
            prog = edge2.to_executorch()
            pte = out / f"01_addmul_{name}.pte"
            with open(pte, "wb") as f:
                f.write(prog.buffer)
            # Count call_delegate / lowered modules if visible.
            g = edge2.exported_program().graph
            gtxt = str(g)
            n_delegate = gtxt.count("call_delegate") + gtxt.count("lowered_module")
            results["runs"].append(
                {
                    "mode": name,
                    "pte": str(pte.name),
                    "graph_excerpt": gtxt[:1500],
                    "delegate_markers_approx": n_delegate,
                    "partition_tags": list(AddMulPartitioner(connected).partition_tags.keys())
                    if False
                    else "see graph",
                }
            )
            # Re-run partitioner just to capture tag count cleanly.
            part = AddMulPartitioner(connected=connected)
            # Export fresh for counting tags
            edge_tmp = to_edge(export(m, inputs))
            pr = part.partition(edge_tmp.exported_program())
            results["runs"][-1]["tag_count"] = len(pr.partition_tags)
            results["runs"][-1]["tags"] = list(pr.partition_tags.keys())
        except Exception as e:
            results["runs"].append({"mode": name, "error": f"{type(e).__name__}: {e}"})

    return results


def main() -> None:
    banner("ET Partitioner：per-node vs connected（边界个数实验室）")
    out = out_dir() / "executorch"
    ops = ToyModel().forward_ops()

    sim_per = simulate_partition(ops, "per_node")
    sim_conn = simulate_partition(ops, "connected")
    real = try_real_executorch(out)

    import json

    write_text(
        out / "01_partition_compare.json",
        json.dumps({"simulate": {"per_node": sim_per, "connected": sim_conn}, "real": real}, indent=2),
    )

    real_status = real.get("status") if isinstance(real, dict) else "unknown"
    real_note = {
        "real_executorch": "已跑通真实 ExecuTorch 路径，见 JSON 里 runs / .pte。",
        "no_executorch": "未安装 executorch——使用模拟结果教学；安装后重跑即可落盘 .pte。\n"
        "  参考: https://pytorch.org/executorch/",
        "no_torch": "未安装 torch——仅模拟。pip/conda 安装 torch 后再装 executorch。",
    }.get(real_status, str(real)[:200])

    guide = f"""# ExecuTorch Partitioner 阅读指引

## 模型（故意制造边界）

```
x = x + y   # 可委托
x = x * y   # 可委托
x = x - y   # portable → 切开
x = x / y   # portable → 切开
x = x * y   # 可委托
x = x + y   # 可委托
```

## 模拟结果（不依赖 ExecuTorch）

| 策略 | delegate 子图数 | 边界数 | 含义 |
|------|-----------------|--------|------|
| per_node（一节点一 tag） | {sim_per['delegate_subgraph_count']} | {sim_per['boundary_count']} | 边界最多，拷贝/同步税最高 |
| connected（连通 add/mul 共 tag） | {sim_conn['delegate_subgraph_count']} | {sim_conn['boundary_count']} | `-`/`/` 切开后两段区域 |

per_node 边界细节: {[b['between'] for b in sim_per['boundaries']]}
connected 边界细节: {[b['between'] for b in sim_conn['boundaries']]}

## 真实 ExecuTorch

状态: `{real_status}`
{real_note}

## Partitioner 契约（必须记住）

1. `partition` **只写** `node.meta["delegation_tag"]` + `partition_tags`
2. **不要**在 partition 阶段改计算语义
3. 同 tag → 一个子图 → 一次 preprocess → 一个 blob → `call_delegate`
4. 未打 tag 的算子留在 portable runtime → **分区边界**

## 接到研究问题①②

- ① 子图划分：tag 粒度 = 图分割决策；本实验用子图数/边界数量化
- ② 边界代价：每多一条边界，就可能多一次拷贝 / layout / 同步

对照 ORT EP（本 lab ONNX 轨步骤 03）与 IREE `flow.dispatch`：
决策时刻分别是 **Session 构建期 / 导出期 / 编译相位**。
"""
    write_text(out / "01_READING.md", guide)
    print(guide)
    print("[OK] ExecuTorch 步骤完成。")


if __name__ == "__main__":
    main()
