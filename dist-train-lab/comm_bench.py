"""集合通信微基准：扫消息大小，测 all_reduce / all_gather 的耗时与带宽。

要测出的是一个**双段模型**：

    耗时 ≈ 固定延迟 + 消息大小 / 带宽

小消息落在延迟主导区（涨 16 倍字节，耗时几乎不变），大消息落在带宽主导区
（涨 N 倍字节，耗时涨 N 倍）。两段之间的拐点直接决定了 DDP 为什么要做梯度分桶：
反向会产生大量小梯度张量，逐个 all_reduce 全落在延迟区，所以要攒成桶推到带宽区。

**在没有 NVLink / P2P 的机器上（如 RTX 4090/5090 这类消费卡），这个脚本测出的
带宽会比显存带宽低一到两个数量级。** 那个比值是本 lab 后续所有结论的背景常数。
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import time

import torch
import torch.distributed as dist

import common


def bus_factor(op: str, world: int) -> float:
    """算法带宽 → 总线带宽的换算。

    ring all_reduce = reduce-scatter + all-gather，每个 rank 实际收发
    2(N-1)/N × S 字节；all_gather 是 (N-1)/N × S。用 busbw 才能跨 N 比较，
    也才能和厂商标称带宽对得上。
    """
    if world < 2:
        return 0.0
    if op == "all_reduce":
        return 2.0 * (world - 1) / world
    if op == "all_gather":
        return float(world - 1) / world
    return 1.0


def run_op(op: str, buf: torch.Tensor, out: torch.Tensor | None) -> None:
    if op == "all_reduce":
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    elif op == "all_gather":
        dist.all_gather_into_tensor(out, buf)
    else:
        raise ValueError(op)


def bench_one(
    op: str, numel: int, device: torch.device, world: int, iters: int, warmup: int
) -> dict[str, float]:
    buf = torch.ones(numel, dtype=torch.float32, device=device)
    out = (
        torch.empty(numel * world, dtype=torch.float32, device=device)
        if op == "all_gather"
        else None
    )

    for _ in range(warmup):  # 首次调用含建链开销，不预热会把它算进第一个点
        run_op(op, buf, out)
    common.sync(device)
    dist.barrier()  # 对齐各 rank 的计时起点

    t0 = time.perf_counter()
    for _ in range(iters):
        run_op(op, buf, out)
    common.sync(device)
    elapsed = (time.perf_counter() - t0) / iters

    nbytes = numel * 4
    algbw = nbytes / elapsed if elapsed > 0 else 0.0
    return {
        "bytes": nbytes,
        "time_s": elapsed,
        "algbw_GBps": algbw / 1e9,
        "busbw_GBps": algbw * bus_factor(op, world) / 1e9,
    }


def find_knee(rows: list[dict]) -> dict[str, float] | None:
    """拐点：从这里开始 busbw 达到平台值的 80%。"""
    if len(rows) < 4:
        return None
    peak = max(r["busbw_GBps"] for r in rows)
    if peak <= 0:
        return None
    for r in rows:
        if r["busbw_GBps"] >= 0.8 * peak:
            return {"knee_bytes": r["bytes"], "peak_busbw_GBps": peak}
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="集合通信微基准")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--backend", choices=["gloo", "nccl"], default=None)
    ap.add_argument("--ops", default="all_reduce,all_gather")
    ap.add_argument("--min-exp", type=int, default=10, help="最小元素数 2^k")
    ap.add_argument("--max-exp", type=int, default=26, help="最大元素数 2^k（2^26*4B = 256 MiB）")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    backend = common.pick_backend(args.backend)
    rank, world, local_rank = common.setup(backend)
    device = common.resolve_device(args.device, local_rank)

    if world < 2:
        common.rank0("[warn] world_size=1，集合通信退化为恒等，测出的带宽没有意义。")

    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    rows: list[dict] = []

    for op in ops:
        for k in range(args.min_exp, args.max_exp + 1):
            numel = 2**k
            # all_gather 的输出是输入的 world 倍，大 world 时要防显存爆掉
            if op == "all_gather" and numel * world * 4 > 4 * 2**30:
                continue
            try:
                r = bench_one(op, numel, device, world, args.iters, args.warmup)
            except RuntimeError as e:  # OOM 等，跳过该点而不是中断整轮
                common.rank0(f"  [skip] {op} 2^{k}: {type(e).__name__}: {e}")
                break
            r.update({"op": op, "backend": backend, "world_size": world})
            rows.append(r)
            common.rank0(
                f"  {op:<11} {r['bytes']/2**20:>9.3f} MiB  "
                f"{r['time_s']*1e6:>10.1f} us  "
                f"algbw {r['algbw_GBps']:>7.2f} GB/s  busbw {r['busbw_GBps']:>7.2f} GB/s"
            )

    if rank == 0:
        out_path = args.out or (common.out_dir() / f"comm_{backend}_ws{world}.csv")
        cols = ["op", "backend", "world_size", "bytes", "time_s", "algbw_GBps", "busbw_GBps"]
        p = pathlib.Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r[c] for c in cols})
        print(f"\n[写出] {p}")

        for op in ops:
            sub = [r for r in rows if r["op"] == op]
            knee = find_knee(sub)
            if knee:
                print(
                    f"  {op}: 平台带宽 {knee['peak_busbw_GBps']:.2f} GB/s，"
                    f"拐点约 {knee['knee_bytes']/2**20:.3f} MiB"
                )
                print(
                    f"      DDP 默认 bucket_cap_mb=25 → "
                    f"{'落在拐点右侧（带宽区），默认值合理' if knee['knee_bytes'] <= 25*2**20 else '落在拐点左侧，这台机器上可考虑调大'}"
                )

        # 这台机器最重要的一个比值
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device.index or 0)
            ar = [r for r in rows if r["op"] == "all_reduce"]
            if ar:
                peak_comm = max(r["busbw_GBps"] for r in ar)
                p2p = common.p2p_matrix()
                print()
                print(f"  卡型 {props.name}  sm_{props.major}{props.minor}")
                if p2p and p2p.get("any_p2p") is False:
                    print("  P2P：**全部关闭**（消费级 GeForce 的典型情况）")
                    print("       NCCL 只能走 host 中转的 shared-memory 传输。")
                print(f"  卡间 all_reduce 总线带宽峰值 ≈ {peak_comm:.1f} GB/s")
                print("  ↑ 把它和单卡显存带宽（5090 约 1790 GB/s）相除，就是这台机器的")
                print("    「带宽悬崖」。悬崖越陡，划分决策的通信代价越不能忽略。")

    common.teardown()


if __name__ == "__main__":
    main()
