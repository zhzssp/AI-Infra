"""16Φ 显存账手算器。纯 CPU，不 import torch.distributed，也不需要卡。

这是所有实测的对照基线：**16Φ 是模型状态，是显存的下界，不是训练总显存**。
实测一定更大，差额来自激活、临时缓冲、碎片——把差额归因清楚才算真懂。

一个值得注意的巧合：混合精度与纯 fp32 的合计**都是 16Φ**，但构成完全不同
（2+2+12 vs 4+4+8）。说得清这个巧合的成因，才算 16 不是一个魔数。
"""

from __future__ import annotations

import argparse

GIB = 2**30


def model_states(
    params: int, precision: str = "mixed", zero_stage: int = 0, n: int = 1
) -> dict[str, float]:
    """模型状态各项字节数。zero_stage: 0=DDP, 1/2/3 = ZeRO-1/2/3, n = 分片组大小。"""
    if precision == "mixed":
        # fp16/bf16 前反向各一份 + 优化器侧 fp32 master weight/m/v
        p, g, opt = 2.0 * params, 2.0 * params, 12.0 * params
    elif precision == "fp32":
        p, g, opt = 4.0 * params, 4.0 * params, 8.0 * params
    else:
        raise ValueError(f"未知精度口径: {precision}")

    # 阶梯是累加的：ZeRO-1 切优化器状态，2 再切梯度，3 再切参数
    if zero_stage >= 1:
        opt /= n
    if zero_stage >= 2:
        g /= n
    if zero_stage >= 3:
        p /= n
    return {"param": p, "grad": g, "optim": opt, "total": p + g + opt}


def activations(
    batch: int,
    seq: int,
    hidden: int,
    layers: int,
    expand: int = 4,
    bytes_per_elem: int = 2,
    recompute: bool = False,
) -> float:
    """粗估。不重计算时每层要留 norm 输出、fc1 输出、act 输出、fc2 输出；
    重计算时每层只留输入，反向时按需重算。

    注意这一项**不随 ZeRO 分片下降**——要压它得靠重计算或序列并行。
    把激活也算进被除项，是 FSDP 实验里最常见的算错方式。
    """
    per_token = hidden if recompute else (2 + 2 * expand) * hidden
    return float(batch * seq * layers * per_token * bytes_per_elem)


def _fmt(nbytes: float) -> str:
    return f"{nbytes:>18,.0f} B  = {nbytes / GIB:>9.3f} GiB"


def main() -> None:
    ap = argparse.ArgumentParser(description="16Φ 显存账手算器")
    ap.add_argument("--params", type=int, default=None, help="参数量；不给则由 hidden/layers 算")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--expand", type=int, default=4)
    ap.add_argument("--precision", choices=["mixed", "fp32"], default="mixed")
    ap.add_argument("--zero", type=int, choices=[0, 1, 2, 3], default=0)
    ap.add_argument("--n", type=int, default=1, help="分片组大小（卡数）")
    ap.add_argument("--activations", action="store_true", help="额外输出激活粗估")
    ap.add_argument("--recompute", action="store_true", help="激活按重计算口径估")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--ladder", action="store_true", help="打印 ZeRO 0~3 的完整阶梯")
    args = ap.parse_args()

    if args.params is not None:
        params = args.params
        src = f"--params {params:,}"
    else:
        from model import params_closed_form

        params = params_closed_form(args.hidden, args.layers, args.expand)
        src = f"hidden={args.hidden} layers={args.layers} expand={args.expand}"

    print("=" * 72)
    print(f"  参数量 Φ = {params:,}  ({params / 1e6:.2f} M)     来源: {src}")
    print(f"  精度口径 = {args.precision}   ZeRO-{args.zero}   分片组 n = {args.n}")
    print("=" * 72)

    st = model_states(params, args.precision, args.zero, args.n)
    for key, label in [
        ("param", "参数     param"),
        ("grad", "梯度     grad "),
        ("optim", "优化器   optim"),
    ]:
        print(f"  {label} : {_fmt(st[key])}   ({st[key] / params:.2f} Φ)")
    print("  " + "-" * 68)
    print(f"  模型状态 total : {_fmt(st['total'])}   ({st['total'] / params:.2f} Φ)")

    if args.activations:
        act = activations(
            args.batch, args.seq, args.hidden, args.layers, args.expand, 2, args.recompute
        )
        print()
        print(f"  激活粗估（batch={args.batch} seq={args.seq}"
              f"{' recompute' if args.recompute else ''}）")
        print(f"           act   : {_fmt(act)}")
        print(f"  合计（状态+激活）: {_fmt(st['total'] + act)}")
        share = act / (st["total"] + act) if (st["total"] + act) > 0 else 0
        print(f"  激活占比 = {share:.1%}", end="")
        if share > 0.15:
            print("   ← 想验模型状态相关结论请调大 hidden 或调小 batch/seq")
        else:
            print("   ← 模型状态占主导，适合验 ZeRO/FSDP 的显存判据")

    if args.ladder:
        print()
        print(f"  ZeRO 阶梯（n = {args.n}，{args.precision} 口径）")
        print("  " + "-" * 68)
        print(f"  {'级别':<10}{'param':>10}{'grad':>10}{'optim':>10}{'total':>12}{'每卡 GiB':>12}")
        for z in (0, 1, 2, 3):
            s = model_states(params, args.precision, z, args.n)
            print(
                f"  ZeRO-{z:<6}"
                f"{s['param']/params:>9.2f}Φ"
                f"{s['grad']/params:>9.2f}Φ"
                f"{s['optim']/params:>9.2f}Φ"
                f"{s['total']/params:>11.2f}Φ"
                f"{s['total']/GIB:>12.3f}"
            )
        print()
        print("  注意第一级的收益最大（16Φ→10Φ 时 n=2），而 2→3 只差 1Φ。")
        print("  卡少时优先上 ZeRO-1；卡多时 ZeRO-3 的 1/n 才拉得开差距。")


if __name__ == "__main__":
    main()
