#!/usr/bin/env python3
"""Step 02 — Graph-level fusion via Relay (four operator classes in action).

Core concepts:
  - injective / reduction / complex-out-fusable / opaque
  - fusion = deciding which nodes share one executable unit (subgraph partition prototype)

主角模型与全仓库统一：tiny_mlp = Gemm(W:[4,3], b) → Relu → Add(bias2)，x:[2,3]。
同一个模型的其它面孔：
  onnx-delegate-lab/onnx_lab/01_build_and_infer.py   ONNX GraphProto
  mlir-toy-dialect/examples/upstream/01-linalg-generic.mlir   linalg IR
  iree-lab/models/tiny_mlp.mlir                      真编真跑（数值答案在文件末尾）
链路位置见 docs/learning-guides/00-end-to-end-pipeline.md 站 ③。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tvm_lab.common import banner, out_dir, require_tvm, write_text

# 与 iree-lab/models/tiny_mlp.mlir、onnx_lab/01 逐位相同的权重。
# 取这组数是为了能手算：W 前三行是单位阵，第四行求和。
W = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0],
     [1.0, 1.0, 1.0]],
    dtype="float32",
)
B = np.full((4,), 0.5, dtype="float32")       # Gemm 的 C
BIAS2 = np.full((4,), 1.0, dtype="float32")   # 第二个 Add
X = np.array([[1.0, 2.0, 3.0],
              [-4.0, -5.0, -6.0]], dtype="float32")
EXPECTED = np.array([[2.5, 3.5, 4.5, 7.5],
                     [1.0, 1.0, 1.0, 1.0]], dtype="float32")


def main() -> None:
    banner("02 Relay 图级融合：四类算子规则的实物（主角模型 tiny_mlp）")
    tvm = require_tvm()
    from tvm import relay

    # ---------------------------------------------------------------------
    # 建图：Gemm → Relu → Add，后面再挂一个屏障 + relu 用来演示 opaque 切开
    # ---------------------------------------------------------------------
    # ★ 注意 nn.dense 的权重布局是 [units, in_dim] = [4, 3]，
    #   也就是「转置过的」——与 ONNX Gemm(transB=1)、
    #   linalg 里 #mapW = (m,n,k)->(n,k) 说的是同一件事。
    #   三套系统都选了 [out, in]，因为它让每个输出通道的权重在内存里连续。
    x = relay.var("x", shape=(2, 3), dtype="float32")
    w = relay.const(W)
    b = relay.const(B)
    bias2 = relay.const(BIAS2)

    y = relay.nn.dense(x, w)      # complex-out-fusable：能吸收输出侧的 injective
    y = relay.add(y, b)           # injective —— 这就是 Gemm 的 C
    y = relay.nn.relu(y)          # injective
    y = relay.add(y, bias2)       # injective
    # 到这里为止就是主角模型。下面是为了演示 opaque 额外加的两个节点。

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

    # ---------------------------------------------------------------------
    # 数一数融合组：融合后每个 `fn (` 就是一个将来会变成 kernel 的单元
    # ---------------------------------------------------------------------
    n_fn_before = before.count("fn (")
    n_fn_after = after.count("fn (")
    print(f"\n  融合前的函数数：{n_fn_before}    融合后：{n_fn_after}")
    print("  主角模型那 4 个 op（dense/add/relu/add）若归到同一个 fn，")
    print("  说明中间张量不落 DRAM —— 这正是站 ③ 那条优化目标。")

    # ---------------------------------------------------------------------
    # 跑一遍对答案：与 iree-lab 的手算结果必须一致
    # 融合是「不改语义的重排」，这行断言就是它的验收标准。
    # ---------------------------------------------------------------------
    numeric = "（本次未执行）"
    try:
        # 只留主角模型部分（去掉演示 opaque 用的尾巴），这样才好和别的 lab 对答案
        y2 = relay.nn.relu(relay.add(relay.nn.dense(x, w), b))
        y2 = relay.add(y2, bias2)
        mod2 = tvm.IRModule.from_expr(relay.Function([x], y2))
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod2, target="llvm")
        from tvm.contrib import graph_executor

        m = graph_executor.GraphModule(lib["default"](tvm.cpu()))
        m.set_input("x", tvm.nd.array(X))
        m.run()
        got = m.get_output(0).numpy()
        ok = np.allclose(got, EXPECTED, atol=1e-5)
        numeric = f"{got.tolist()}  {'✅ 与手算一致' if ok else '⚠️ 不一致'}"
        print(f"\n  数值校验：{numeric}")
        if ok:
            print("  与 iree-lab/models/tiny_mlp.mlir 文末的手算、onnx_lab/01 的输出三方一致：")
            print("  三套编译器对同一个模型给出同一个答案，融合/相位/EP 都没改语义。")
    except Exception as e:  # noqa: BLE001
        print(f"\n  [跳过数值校验] {type(e).__name__}: {e}")
        print("  （建图与融合的观察不受影响；执行需要带 LLVM 的 TVM 构建）")

    guide = f"""# 阅读指引：融合前后 IR

模型是全仓库统一的主角 **tiny_mlp**：`x[2,3] → Gemm(W[4,3], b) → Relu → Add(bias2) → y[2,4]`，
后面额外挂了「屏障 + relu」用来演示 opaque 切开点。

对比 `02_relay_before_fuse.ir.txt` 与 `02_relay_after_fuse.ir.txt`：

1. `dense → add → relu → add`：典型 **complex-out-fusable + 三个 injective**，应融成同一个 `fn`。
2. `stop_fusion`（或 cast 屏障）：模拟 **opaque** 切开点。
3. 之后的 `relu`：通常开启新的融合组。

本次统计：融合前 `fn (` 出现 {n_fn_before} 次，融合后 {n_fn_after} 次。
数值校验：{numeric}

## 同一个决定，三套系统三种做法

| 系统 | 靠什么判断能不能融 | 在哪看 |
|------|------------------|--------|
| TVM | `OpPatternKind` 四分类（本文件） | `02_relay_after_fuse.ir.txt` 的 `fn` 分组 |
| IREE | `linalg` 的 `iterator_types` + `indexing_maps` | `iree-lab/out/phases/tiny_mlp.flow.mlir` 数 `flow.dispatch` |
| ORT / ExecuTorch | 后端自报 capability，再按连通性切 | `onnx-delegate-lab/out/` 的边界张量计数 |

**问题是同一个**：这几个 op 要不要落在同一个可执行单元里。
TVM 用算子分类回答（简单、够用），IREE 用循环结构回答（不必认识算子，能推广到新算子），
委托框架用后端能力回答（受限于别人支持什么）。**三种答案的代价各不相同**——
这是 [`00-end-to-end-pipeline.md`](../../../docs/learning-guides/00-end-to-end-pipeline.md) 站 ②③ 的核心。

四类速记：
| 类别 | 例子 | 规则直觉 |
|------|------|----------|
| injective | add / relu | 彼此可融；可被两侧吸收 |
| reduction | sum / softmax 归约 | 可融合输入侧 injective |
| complex-out-fusable | conv / dense | 可融合输出侧 injective |
| opaque | sort / stop_fusion | 通常不融，切开边界 |

## 权重布局：三套系统的同一个选择

`nn.dense` 的权重是 `[units, in_dim] = [4, 3]`，即「转置过的」。
ONNX 里靠 `Gemm(transB=1)` 表达，linalg 里靠 `#mapW = (m,n,k)->(n,k)` 表达。
**三者说的是同一件事**：让每个输出通道的权重在内存里连续。
注意 linalg 那种写法连转置 op 都不需要——换个下标顺序而已。
"""
    write_text(out / "02_READING.md", guide)

    print("\n--- FuseOps 之后（节选）---")
    print("\n".join(after.splitlines()[:60]))
    print("\n[OK] 步骤 02 完成。对照 out/tvm/02_READING.md 看融合切开点。")


if __name__ == "__main__":
    main()
