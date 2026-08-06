#!/usr/bin/env python3
"""Step 01 — Build a 3-layer ONNX graph with helper, check, infer_shapes, run ORT CPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_common import banner, out_dir, require_onnx, write_text


def main() -> None:
    banner("01 ONNX：helper 构图 + checker + shape infer + ORT 跑通")
    onnx, ort = require_onnx()
    from onnx import TensorProto, checker, helper, numpy_helper, shape_inference

    rng = np.random.default_rng(0)
    W = numpy_helper.from_array(rng.standard_normal((4, 3), dtype=np.float32), name="W")
    b = numpy_helper.from_array(rng.standard_normal((4,), dtype=np.float32), name="b")
    bias2 = numpy_helper.from_array(np.ones(4, dtype=np.float32), name="bias2")

    gemm = helper.make_node("Gemm", ["x", "W", "b"], ["h"], name="gemm")
    relu = helper.make_node("Relu", ["h"], ["h_act"], name="relu")
    add = helper.make_node("Add", ["h_act", "bias2"], ["y"], name="add")

    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 4])
    graph = helper.make_graph(
        [gemm, relu, add],
        "tiny_mlp",
        inputs=[X],
        outputs=[Y],
        initializer=[W, b, bias2],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="onnx-delegate-lab",
    )
    model.ir_version = onnx.IR_VERSION
    checker.check_model(model)
    model = shape_inference.infer_shapes(model)

    out = out_dir() / "onnx"
    path = out / "01_tiny_mlp.onnx"
    onnx.save(model, path)

    # Shape summary after inference.
    shapes = []
    for vi in list(model.graph.value_info) + list(model.graph.output):
        t = vi.type.tensor_type
        dims = []
        for d in t.shape.dim:
            dims.append(d.dim_value if d.dim_value else (d.dim_param or "?"))
        shapes.append(f"{vi.name}: {dims}")

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = rng.standard_normal((2, 3), dtype=np.float32)
    (y,) = sess.run(None, {"x": x})

    init_names = {t.name for t in model.graph.initializer}
    runtime_inputs = [i.name for i in model.graph.input if i.name not in init_names]

    guide = f"""# 01 阅读指引：ONNX 结构层次

## 层次（默画）

```
ModelProto
  └─ GraphProto  (tiny_mlp)
       ├─ NodeProto[]     gemm → relu → add
       ├─ initializer[]   W, b, bias2   ← 权重只进这里，不进 graph.input
       ├─ input[]         x             ← 真正要喂的运行时输入
       └─ output[]        y
```

## 本步结果

- 模型文件: `{path.name}`
- IR version: {model.ir_version}
- opset: {[(o.domain, o.version) for o in model.opset_import]}
- runtime inputs: {runtime_inputs}
- initializer: {sorted(init_names)}
- infer_shapes 后可见: {shapes}
- ORT CPU 输出 shape: {tuple(y.shape)}（期望 (2, 4)）
- 数值冒烟 OK: {y.shape == (2, 4)}

## 记住

1. **ONNX ≠ 运行时**——本文件是交换 IR；ORT 才是执行器。
2. `initializer` vs `graph.input`：权重在 initializer，不要误当成要喂的输入。
3. `dim_param` / `None`：batch 维动态时，`infer_shapes` 能推出通道维，推不出具体 batch。
"""
    write_text(out / "01_READING.md", guide)
    print(guide)
    print("[OK] 步骤 01 完成。")


if __name__ == "__main__":
    main()
