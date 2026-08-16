#!/usr/bin/env python3
"""Step 02 — Inspect + mutate an ONNX graph (insert Identity, tweak initializer)."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_common import banner, out_dir, require_onnx, write_text


def replace_initializer(graph, name: str, array: np.ndarray) -> None:
    from onnx import numpy_helper

    for i, t in enumerate(list(graph.initializer)):
        if t.name == name:
            del graph.initializer[i]
            break
    graph.initializer.append(numpy_helper.from_array(array, name=name))


def insert_identity_after(graph, producer_output: str, new_name: str) -> None:
    """在 producer_output 之后插一个 Identity 探针，并把原下游改接到 new_name。

    两件事缺一不可，缺了哪件 checker 都会当场拒绝：
    1. **改接下游**——否则 new_name 无人使用，Identity 成了死节点；
    2. **插在生产者紧后面**——ONNX 要求 node 列表拓扑有序（§3.1）。
       如果图省事写成 graph.node.append(...)，探针会排在消费它的 add 之后，
       checker 直接报 "input 'h_act_probe' is not output of any previous nodes"。
    """
    from onnx import helper

    producer_idx = next(
        (i for i, n in enumerate(graph.node) if producer_output in n.output), -1
    )
    if producer_idx < 0:
        raise ValueError(f"没有任何节点产出 {producer_output}")

    for node in graph.node:
        for i, inp in enumerate(list(node.input)):
            if inp == producer_output:
                node.input[i] = new_name
    for out in graph.output:
        if out.name == producer_output:
            out.name = new_name

    probe = helper.make_node("Identity", [producer_output], [new_name], name="probe_identity")
    graph.node.insert(producer_idx + 1, probe)


def main() -> None:
    banner("02 ONNX：巡检 + 插入 Identity + 改 initializer")
    onnx, ort = require_onnx()
    from onnx import checker, shape_inference

    src = out_dir() / "onnx" / "01_tiny_mlp.onnx"
    if not src.exists():
        print(f"[ERROR] 先跑 01。缺少 {src}", file=sys.stderr)
        sys.exit(1)

    model = onnx.load(src)
    g = model.graph
    init = {t.name for t in g.initializer}
    hist = Counter((n.domain or "", n.op_type) for n in g.node)
    runtime_inputs = [i.name for i in g.input if i.name not in init]

    # Insert Identity after relu output h_act.
    insert_identity_after(g, "h_act", "h_act_probe")
    # Zero-out bias2 to make output change observable.
    replace_initializer(g, "bias2", np.zeros(4, dtype=np.float32))

    checker.check_model(model)
    model = shape_inference.infer_shapes(model)
    out_path = out_dir() / "onnx" / "02_tiny_mlp_mutated.onnx"
    onnx.save(model, out_path)

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    x = np.ones((2, 3), dtype=np.float32)
    (y_mut,) = sess.run(None, {"x": x})

    sess0 = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    (y0,) = sess0.run(None, {"x": x})

    guide = f"""# 02 阅读指引：改图工程手感

## 巡检清单（真实模型也按这个扫）

- IR / opset / 节点数 / initializer 数
- 算子直方图（EP 能不能吃，先看直方图）
- **真正 runtime inputs**（去掉 initializer 同名项）

本模型：
- nodes: {len(g.node)}（含插入的 Identity）
- inits: {len(g.initializer)}
- histogram: {dict(hist)}
- runtime inputs: {runtime_inputs}

## 做了什么

1. 在 `h_act` 后插入 `Identity` 探针 → 下游边改接到 `h_act_probe`
2. 把 initializer `bias2` 改成全 0
3. `checker` + `infer_shapes` + ORT 再跑

### 插节点的两个硬约束

- **改接下游**：新名字必须有人用，否则探针是死节点
- **插在生产者紧后面**：`graph.node` 必须拓扑有序。
  写成 `graph.node.append(probe)` 会让探针排在消费它的 `add` 之后，
  `checker` 报 `input 'h_act_probe' is not output of any previous nodes`。
  想亲眼看这条报错，把 `insert_identity_after` 里的 `insert` 换成 `append` 再跑一次。

## 数值对照（同一输入 x=ones）

- 原图输出均值: {float(y0.mean()):.6f}
- 改图输出均值: {float(y_mut.mean()):.6f}
- 是否变化: {not np.allclose(y0, y_mut)}

改完图务必：`checker.check_model` + 数值冒烟。这是委托/分区实验前的基本功。

> 拓扑序在 MLIR 里的对应物叫 Dominance，见
> [`docs/learning-guides/onnx-learning-guide.md` §3.1](../../../docs/learning-guides/onnx-learning-guide.md)。
"""
    write_text(out_dir() / "onnx" / "02_READING.md", guide)
    print(guide)
    print("[OK] 步骤 02 完成。")


if __name__ == "__main__":
    main()
