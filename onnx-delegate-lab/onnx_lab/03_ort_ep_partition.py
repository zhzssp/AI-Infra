#!/usr/bin/env python3
"""Step 03 — Observe ORT EP partitioning (GetCapability → Compile → dispatch).

Prints available providers, runs CPU-only vs (optional) accel+CPU, dumps profiling
and a partition observation report focused on research questions ①②.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_common import banner, out_dir, require_onnx, write_text


def build_partition_demo_model(onnx):
    """A slightly richer graph: Gemm+Relu may go to accel EP; Softmax often stays CPU."""
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(1)
    W = numpy_helper.from_array(rng.standard_normal((8, 4), dtype=np.float32), name="W")
    b = numpy_helper.from_array(rng.standard_normal((8,), dtype=np.float32), name="b")

    nodes = [
        helper.make_node("Gemm", ["x", "W", "b"], ["h"], name="gemm"),
        helper.make_node("Relu", ["h"], ["h_act"], name="relu"),
        helper.make_node("Softmax", ["h_act"], ["prob"], name="softmax", axis=1),
        helper.make_node("ReduceSum", ["prob"], ["y"], name="reducesum", keepdims=0),
    ]
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [])
    graph = helper.make_graph(nodes, "partition_demo", [X], [Y], initializer=[W, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = onnx.IR_VERSION
    return model


def run_with_providers(ort, model_path: Path, providers, tag: str, out: Path):
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.optimized_model_filepath = str(out / f"03_optimized_{tag}.onnx")
    # Keep graph opts on so fused regions appear in optimized model.
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        sess = ort.InferenceSession(str(model_path), so, providers=providers)
    except Exception as e:
        return {"error": str(e), "requested": providers, "active": []}

    active = sess.get_providers()
    x = np.random.randn(2, 4).astype(np.float32)
    sess.run(None, {"x": x})
    prof = sess.end_profiling()

    events = []
    provider_hits = {}
    try:
        with open(prof, encoding="utf-8") as f:
            data = json.load(f)
        # ORT profile is either a list or {"traceEvents": [...]} depending on version.
        raw = data if isinstance(data, list) else data.get("traceEvents", data.get("events", []))
        for ev in raw:
            args = ev.get("args", {}) if isinstance(ev, dict) else {}
            name = ev.get("name") or args.get("op_name") or ""
            prov = args.get("provider") or args.get("provider_type") or ev.get("cat") or ""
            if name and prov:
                events.append({"name": name, "provider": prov})
                provider_hits[prov] = provider_hits.get(prov, 0) + 1
    except Exception as e:
        events = [{"parse_error": str(e), "profile_path": prof}]

    return {
        "requested": providers,
        "active": active,
        "profile_path": prof,
        "optimized_model": so.optimized_model_filepath,
        "provider_hit_counts": provider_hits,
        "sample_events": events[:40],
    }


def conceptual_boundary_note(nodes):
    """Teaching note when a second EP is unavailable."""
    lines = [
        "本机若只有 CPU EP，仍可从算子语义推断「典型」边界位置：",
        "",
        "| 节点 | 常见归属直觉 |",
        "|------|--------------|",
    ]
    for n in nodes:
        guess = "常被 CUDA/TRT 类 EP 优先吃掉" if n in ("Gemm", "MatMul", "Conv", "Relu") else "常留在 CPU 或单独成区（归约/特殊实现）"
        lines.append(f"| {n} | {guess} |")
    lines += [
        "",
        "打开双 EP（如 `CUDAExecutionProvider` + `CPUExecutionProvider`）后，",
        "用 profile / optimized graph **验证**边界是否落在 Softmax / ReduceSum 之前——",
        "那就是研究问题①的最小实验室。",
    ]
    return "\n".join(lines)


def main() -> None:
    banner("03 ORT EP：打印 providers + 分区观察（研究问题①②）")
    onnx, ort = require_onnx()
    from onnx import checker

    out = out_dir() / "onnx"
    model = build_partition_demo_model(onnx)
    checker.check_model(model)
    model_path = out / "03_partition_demo.onnx"
    onnx.save(model, model_path)

    available = ort.get_available_providers()
    print(f"  可用 providers: {available}")

    cpu_report = run_with_providers(ort, model_path, ["CPUExecutionProvider"], "cpu", out)

    accel = None
    for cand in ("CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider"):
        if cand in available:
            accel = cand
            break

    dual_report = None
    if accel:
        dual_report = run_with_providers(
            ort, model_path, [accel, "CPUExecutionProvider"], "dual", out
        )

    op_types = [n.op_type for n in model.graph.node]
    concept = conceptual_boundary_note(op_types)

    report = {
        "available_providers": available,
        "cpu_only": cpu_report,
        "dual": dual_report,
        "graph_nodes": [{"name": n.name, "op": n.op_type} for n in model.graph.node],
    }
    write_text(out / "03_partition_report.json", json.dumps(report, indent=2, ensure_ascii=False))

    dual_section = "未检测到第二 EP（CUDA/Dml/CoreML）。已完成 CPU 轨；装好加速 EP 后重跑本步即可看到真实跨 EP 边界。"
    if dual_report and not dual_report.get("error"):
        dual_section = f"""请求 providers: {dual_report['requested']}
实际 active: {dual_report['active']}
profile: `{Path(dual_report['profile_path']).name}`
provider 命中统计: {dual_report.get('provider_hit_counts')}
optimized 图: `{Path(dual_report['optimized_model']).name}`

**请回答三句**：
1. 哪些连续节点被同一 EP 吃成一块？
2. 边界落在哪两个 op 之间、为何？
3. 边界可能付出哪类代价（拷贝 / layout / 内存空间 / 同步）？
"""

    guide = f"""# 03 阅读指引：ORT EP 分区 = 工业界「子图划分」接口

## 机制三拍（对照教材）

```
GetCapability  →  EP 声明能吃哪些子图
Compile        →  编译子图为 EP 内部可执行体
运行时派发      →  Session 按分区调用各 EP
```

## 本机观察

- 可用 providers: `{available}`
- CPU-only active: `{cpu_report.get('active')}`
- 图节点顺序: {op_types}

### 双 EP 结果

{dual_section}

### 概念对照（无第二 EP 时也读）

{concept}

## 四类边界代价（研究问题①②）

| 代价 | 在 ORT 里怎么出现 |
|------|------------------|
| 数据拷贝 | host↔device / device↔device |
| Layout 转换 | Transpose / Reformat 插在边界 |
| 内存空间切换 | 不同 EP 分配器不互认 |
| 同步点 | 跨流 / 跨 EP 必须 wait |

完整 JSON 见 `03_partition_report.json`。对照：
- ExecuTorch Partitioner（本 lab 下一步）
- IREE `flow.dispatch`（编译期划分）
- TVM 融合（单后端划分原型，见 `tvm-fatbin-lab`）
"""
    write_text(out / "03_READING.md", guide)
    print(guide)
    print("[OK] 步骤 03 完成。")


if __name__ == "__main__":
    main()
