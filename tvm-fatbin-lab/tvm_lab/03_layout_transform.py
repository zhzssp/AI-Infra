#!/usr/bin/env python3
"""Step 03 — Layout transform & why cross-backend boundaries pay tax.

Core concepts:
  - logical shape ≠ physical layout (NCHW vs NHWC)
  - layout_transform inserts conversion ops (research question ②)
  - ConvertLayout: 偏好传播 + 把转换外推到图边界（本步骤第二部分）

两部分的分工：
  A. 最小实验：一次显式 layout_transform + relu，跑出数值证明「只是换了排列」
  B. 传播实验：conv2d(NHWC) → relu，跑 ConvertLayout 到 NCHW，
     看转换节点是被机械插入还是被推到边界

对应 docs/learning-guides/tvm-learning-guide.md §2.3。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tvm_lab.common import banner, out_dir, require_tvm, write_text


def part_a(tvm, out: Path) -> tuple[bool, str]:
    """显式 layout_transform：证明它只改排列、不改语义，但要付一次全量搬运。"""
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
    ok = bool(np.allclose(out_nhwc, expected))

    write_text(out / "03_layout_ir.txt", ir_text)
    print(ir_text)
    print(f"  A 部分：输入 {inp.shape} (NCHW) → 输出 {out_nhwc.shape} (NHWC)，数值正确={ok}")
    return ok, ir_text


def part_b(tvm, out: Path) -> tuple[int, int]:
    """ConvertLayout：偏好传播能不能把转换推到图边界。

    造一个 NHWC 的 conv2d → relu，要求 conv 用 NCHW。
    看 ConvertLayout 之后 layout_transform 节点有几个、落在哪。
    """
    from tvm import relay

    # 从 TF 侧导入的典型形态：全图 NHWC，权重 HWIO
    x = relay.var("x", shape=(1, 8, 8, 4), dtype="float32")
    w = relay.const(np.random.randn(3, 3, 4, 4).astype("float32"))
    y = relay.nn.conv2d(
        x, w,
        data_layout="NHWC", kernel_layout="HWIO",
        padding=[1, 1, 1, 1], channels=4, kernel_size=[3, 3],
    )
    y = relay.nn.relu(y)          # layout-agnostic：传播能穿过它
    mod = tvm.IRModule.from_expr(relay.Function([x], y))
    before = mod.astext(show_meta_data=False)

    # 要求 conv2d 走 NCHW（CPU 上的常见偏好）
    seq = tvm.transform.Sequential([
        relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}),
        relay.transform.FoldConstant(),   # 权重侧的 transform 输入全是常量，应被吃掉
    ])
    with tvm.transform.PassContext(opt_level=3):
        mod_after = seq(mod)
    after = mod_after.astext(show_meta_data=False)

    n_before = before.count("layout_transform")
    n_after = after.count("layout_transform")

    write_text(out / "03b_convertlayout_before.ir.txt", before)
    write_text(out / "03b_convertlayout_after.ir.txt", after)
    print(f"\n  B 部分：layout_transform 节点数 {n_before} → {n_after}")
    print("  期望：conv 换到 NCHW 后，转换被推到图的输入/输出边界，")
    print("  relu 跟着 conv 一起走 NCHW（它是 layout-agnostic，只改类型不改语义），")
    print("  权重那次转换被 FoldConstant 直接算掉——所以剩下的应该只有边界那两个。")
    return n_before, n_after


def main() -> None:
    banner("03 Layout：NCHW → NHWC 变换、传播与边界代价")
    tvm = require_tvm()
    out = out_dir() / "tvm"

    ok, _ = part_a(tvm, out)

    try:
        n_before, n_after = part_b(tvm, out)
        part_b_note = f"`layout_transform` 节点数：{n_before} → {n_after}（ConvertLayout + FoldConstant 之后）"
    except Exception as e:  # noqa: BLE001
        n_before = n_after = -1
        part_b_note = f"（本次跳过：{type(e).__name__}: {e}）"
        print(f"\n  [跳过 B 部分] {type(e).__name__}: {e}")

    note = f"""# Layout 变换实验结果

## A. 显式 layout_transform：只改排列，但要付搬运

- NCHW → NHWC，数值正确: {ok}
- 产物：`03_layout_ir.txt`

`layout_transform` 是一次**全量读 + 全量写**的纯搬运：一个 FLOP 都不算，
却要把整张张量过一遍内存。所以它出现在哪、出现几次，直接是性能问题。

## B. ConvertLayout：偏好传播，把转换推到边界

{part_b_note}

对比 `03b_convertlayout_before.ir.txt` 与 `03b_convertlayout_after.ir.txt`：

1. **before**：全图 NHWC，一个 transform 都没有——但 conv 跑在不被偏好的 layout 上，
   慢在算子实现里，**图上看不出来**。
2. **after**：conv 换成 NCHW。如果实现只会机械插入，会在 conv 两侧各插一个
   （数据转入、结果转回），加上权重那个共 3 个，**单独看是负优化**。
   真正的做法是让偏好沿边**传播**：`nn.relu` 是 layout-agnostic 的，
   换 layout 只需改它的张量类型，于是转换被推到图边界；
   权重侧那次输入全是常量，被 `FoldConstant` 吃掉。

**算子对 layout 的三种态度**决定了传播能推多远：

| 态度 | 例子 | 传播时怎么处理 |
|------|------|---------------|
| layout-sensitive | `nn.conv2d` / `nn.dense` | 声明偏好，是传播的**源头** |
| layout-agnostic | `nn.relu` / `add` | 跟着上游走，只改类型 |
| layout-fixed（带轴参数） | `concatenate(axis=1)` / `softmax(axis=-1)` | 能跟，但**必须同步改写 axis**；改不动就在此处停下、插 transform |

> 自己试：把 `nn.relu` 换成 `relay.concatenate([y, y], axis=3)`，
> 再跑一遍看 `layout_transform` 的数量变成几个。

## 为什么这是研究问题②的原型

跨后端委托时，若 EP_A 偏好 NCHW、EP_B 偏好 NHWC，分区边界必须插入 `layout_transform`
（或等价的 Transpose / Reformat）。变换本身吃带宽——有时比多跑一个 `add` 更贵。

好的划分会把 layout 冲突推到图边缘，或与计算融合；坏的划分会在每个边界都付转换税。

**这一条和「融合」是同一枚硬币的两面**：融合问的是「相邻两个 op 能不能共用一个循环」，
layout 问的是「相邻两个 op 对内存排布的要求合不合」。两者都不合时，中间张量就必须落地。
对照 `02_READING.md` 与 `onnx-delegate-lab/out/` 的边界张量计数一起看。
"""
    write_text(out / "03_READING.md", note)
    print(f"\n[OK] 步骤 03 完成。数值正确={ok}。读 out/tvm/03_READING.md。")


if __name__ == "__main__":
    main()
