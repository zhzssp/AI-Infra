#!/usr/bin/env python3
"""Step ET — Minimal Partitioner: per-node vs connected-component tagging.

模型 = 主角 tiny_mlp 叠两层，中间夹一个 softmax：

    [Gemm → Relu → Add] → Softmax → [Gemm → Relu → Add]
     └──── 可委托 ────┘    portable   └──── 可委托 ────┘

为什么这样搭：主角三件套在任何后端上都被支持，单独一份会被整个吃掉，
**没有边界就量不出边界代价**。夹一个 portable 的 softmax，图被迫切成两段，
于是 per_node 与 connected 两种打 tag 策略会给出完全不同的子图数与边界数——
这正是研究问题①②的最小实验室。

If ExecuTorch is installed, runs a real to_edge/to_backend path.
Otherwise runs a simulation that still teaches delegation_tag / boundaries.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_common import banner, out_dir, write_text

# 只有 softmax 留给 portable runtime；其余（含权重搬运类 op）都视为可委托。
# 用「排除法」而不是「白名单」，是因为 Linear 在 Edge Dialect 里会被分解成
# addmm / permute_copy / view_copy 等若干形态，白名单容易随版本漏项。
PORTABLE_HINTS = ("softmax",)

# 与 iree-lab/models/tiny_mlp.mlir、tvm_lab/02、onnx_lab/01 同一组权重，
# 便于跨系统手算对拍：x=[1,2,3] → 第一段输出 [2.5, 3.5, 4.5, 7.5]
W1 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
B1 = 0.5
BIAS2 = 1.0


class ToyModel:
    """主角叠两层，中间夹 softmax —— 与真实 torch 模型的算子序列对应。"""

    def forward_ops(self) -> List[str]:
        return [
            "addmm",    # Gemm  ┐
            "relu",     # Relu  ├ 第一份 tiny_mlp
            "add",      # Add   ┘
            "softmax",  # ← portable，切开
            "addmm",    # Gemm  ┐
            "relu",     # Relu  ├ 第二份 tiny_mlp
            "add",      # Add   ┘
        ]


def is_delegatable(op: str) -> bool:
    return not any(h in op for h in PORTABLE_HINTS)


def simulate_partition(ops: List[str], mode: str) -> dict:
    """mode: per_node | connected —— 两种打 tag 策略，算法差别只有一处。"""
    tags: Dict[str, str] = {}
    node_tags = []
    pid = 0
    current = None

    for i, op in enumerate(ops):
        if not is_delegatable(op):
            current = None
            node_tags.append((i, op, None))
            continue
        if mode == "per_node":
            # 每个可委托节点自成一 tag —— 合法，但把能连起来的一段拆碎了
            tag = f"mlp_{pid}"
            pid += 1
            tags[tag] = "DemoBackend"
            node_tags.append((i, op, tag))
        else:
            # 连通分量：只要上一个节点也被 tag 了，就沿用同一个 tag
            if current is None:
                current = f"mlp_{pid}"
                pid += 1
                tags[current] = "DemoBackend"
            node_tags.append((i, op, current))

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

    return {
        "mode": mode,
        "node_tags": node_tags,
        "delegation_tags": tags,
        "delegate_subgraph_count": len(tags),
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
        from executorch.exir.backend.backend_details import DelegationSpec
        from executorch.exir.backend.partitioner import PartitionResult, Partitioner
    except Exception as e:
        return {"status": "no_executorch", "error": str(e)}

    _DEMO = "BackendWithCompilerDemo"

    class TinyMlpPartitioner(Partitioner):
        """per_node 与 connected 的差别只有 `self.connected` 那一个分支。"""

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
                if not is_delegatable(str(node.target)):
                    current = None
                    continue
                if self.connected:
                    if current is None:
                        current = f"mlp_{partition_id}"
                        partition_id += 1
                        self.partition_tags[current] = self.delegation_spec
                    node.meta["delegation_tag"] = current
                else:
                    tag = f"mlp_{partition_id}"
                    partition_id += 1
                    node.meta["delegation_tag"] = tag
                    self.partition_tags[tag] = self.delegation_spec
            return PartitionResult(
                tagged_exported_program=exported_program,
                partition_tags=self.partition_tags,
            )

    class TinyMlpx2(torch.nn.Module):
        """主角 tiny_mlp 叠两层，中间夹 softmax。"""

        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(3, 4)
            self.fc2 = torch.nn.Linear(4, 4)
            with torch.no_grad():
                # 与 iree-lab / tvm_lab / onnx_lab 同一组权重
                self.fc1.weight.copy_(torch.tensor(W1))
                self.fc1.bias.fill_(B1)
                self.fc2.weight.copy_(torch.eye(4))
                self.fc2.bias.zero_()
            self.register_buffer("bias2", torch.full((4,), BIAS2))

        def forward(self, x):
            h = torch.relu(self.fc1(x)) + self.bias2      # 第一份 tiny_mlp
            p = torch.softmax(h, dim=-1)                  # portable，切开
            return torch.relu(self.fc2(p)) + self.bias2   # 第二份 tiny_mlp

    results = {"status": "real_executorch", "runs": []}
    m = TinyMlpx2().eval()
    inputs = (torch.tensor([[1.0, 2.0, 3.0]]),)

    with torch.no_grad():
        stage1 = torch.relu(m.fc1(inputs[0])) + m.bias2
    results["stage1_output"] = stage1.tolist()
    results["stage1_expected"] = [[2.5, 3.5, 4.5, 7.5]]

    # 相位 dump：同一个模型在 ATen / Edge 两层各长什么样。
    # 这是「Edge Dialect 多约束了什么」最直接的证据——对比两份文本即可。
    try:
        ep0 = export(m, inputs)
        write_text(out / "01_aten.graph.txt", str(ep0.graph))
        edge0 = to_edge(ep0)
        write_text(out / "01_edge.graph.txt", str(edge0.exported_program().graph))
        results["dialect_dumps"] = ["01_aten.graph.txt", "01_edge.graph.txt"]
    except Exception as e:
        results["dialect_dumps"] = f"{type(e).__name__}: {e}"

    for connected, name in ((False, "per_node"), (True, "connected")):
        try:
            edge = to_edge(export(m, inputs))
            # 先单独跑一次 partition，干净地统计 tag 数（to_backend 会消费掉图）
            part = TinyMlpPartitioner(connected=connected)
            pr = part.partition(to_edge(export(m, inputs)).exported_program())

            lowered = edge.to_backend(TinyMlpPartitioner(connected=connected))
            prog = lowered.to_executorch()
            pte = out / f"01_tiny_mlp_{name}.pte"
            with open(pte, "wb") as f:
                f.write(prog.buffer)

            gtxt = str(lowered.exported_program().graph)
            results["runs"].append(
                {
                    "mode": name,
                    "pte": pte.name,
                    "pte_bytes": pte.stat().st_size,
                    "tag_count": len(pr.partition_tags),
                    "tags": list(pr.partition_tags.keys()),
                    "delegate_markers_approx": gtxt.count("call_delegate")
                    + gtxt.count("lowered_module"),
                    "graph_excerpt": gtxt[:1500],
                }
            )
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
        json.dumps(
            {"simulate": {"per_node": sim_per, "connected": sim_conn}, "real": real},
            indent=2,
            ensure_ascii=False,
        ),
    )

    real_status = real.get("status") if isinstance(real, dict) else "unknown"
    real_note = {
        "real_executorch": "已跑通真实 ExecuTorch 路径，见 JSON 里 runs / .pte。",
        "no_executorch": "未安装 executorch——使用模拟结果教学；安装后重跑即可落盘 .pte。\n"
        "  参考: https://pytorch.org/executorch/",
        "no_torch": "未安装 torch——仅模拟。pip/conda 安装 torch 后再装 executorch。",
    }.get(real_status, str(real)[:200])

    guide = f"""# ExecuTorch Partitioner 阅读指引

## 模型 = 主角 tiny_mlp 叠两层，中间夹一个 portable 算子

```
x ─▶ Gemm ─▶ Relu ─▶ Add ─▶ Softmax ─▶ Gemm ─▶ Relu ─▶ Add ─▶ y
     └──── 可委托 ────┘      portable  └──── 可委托 ────┘
```

单独一份 tiny_mlp 会被整个吃掉——**没有边界就量不出边界代价**。
夹一个 softmax 把图切成两段，per_node 与 connected 才有差别可看。

算子序列：`{ops}`

## 模拟结果（不依赖 ExecuTorch）

| 策略 | delegate 子图数 | 边界数 | 含义 |
|------|-----------------|--------|------|
| per_node（一节点一 tag） | {sim_per['delegate_subgraph_count']} | {sim_per['boundary_count']} | 每个 op 单独下发，中间张量全部物化 |
| connected（连通区共 tag） | {sim_conn['delegate_subgraph_count']} | {sim_conn['boundary_count']} | softmax 切开后正好两段 |

per_node 边界细节: {[b['between'] for b in sim_per['boundaries']]}
connected 边界细节: {[b['between'] for b in sim_conn['boundaries']]}

**两个策略都合法、都能跑出正确数值**——差别全在性能。
per_node 下 `Relu` 自成一个 delegate，意味着 `h` 必须写回内存再读出来：
Gemm 的输出循环里顺手做一次 `max(0,·)` 的机会**被永久关闭**，
下游 TVM 的 `FuseOps`、IREE 的 `flow.dispatch` 谁都补不回来。
这就是链路总图断链表第 ② 行标「不能补救」的原因。

## 真实 ExecuTorch

状态: `{real_status}`
{real_note}

装好 torch + executorch 后还会多出两份相位 dump：

- `01_aten.graph.txt` —— ATen Dialect，`torch.ops.aten.*` 命名空间
- `01_edge.graph.txt` —— Edge Dialect，`executorch_exir_dialects_edge__ops_*`，且已 dtype 特化

**对比着读这两份**，重点看 `self.fc1(x)` 变成了什么：
`nn.Linear` 在 Edge 层会被拆成 `permute_copy` + `addmm`（或 `t_copy` + `addmm`）。
这就是 Partitioner 要用「排除法」而不是白名单挑算子的原因——
一个 `Linear` 到底落成哪几个 edge op，是随版本变的。

## Partitioner 契约（必须记住）

1. `partition` **只写** `node.meta["delegation_tag"]` + `partition_tags`
2. **不要**在 partition 阶段改计算语义
3. 同 tag → 一个子图 → 一次 preprocess → 一个 blob → `call_delegate`
4. 未打 tag 的算子留在 portable runtime → **分区边界**

## 接到研究问题①②

- ① 子图划分：tag 粒度 = 图分割决策；本实验用子图数/边界数量化
- ② 边界代价：每多一条边界，就可能多一次拷贝 / layout / 同步 / 融合机会丢失

对照 ORT EP（本 lab ONNX 轨步骤 03，同样是「主角 + 一个 portable 尾巴」）
与 IREE `flow.dispatch`（`iree-lab/scripts/run_phases.sh`）：
决策时刻分别是 **导出期 / Session 构建期 / 编译相位**。
"""
    write_text(out / "01_READING.md", guide)
    print(guide)
    print("[OK] ExecuTorch 步骤完成。")


if __name__ == "__main__":
    main()
