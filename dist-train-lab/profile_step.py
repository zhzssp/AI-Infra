"""一步训练里的通信占比：把 comm_bench 的延迟/带宽模型接到真实训练步上。

要回答两个问题：

1. **为什么要分桶**：反向产生大量小梯度张量，逐个 all_reduce 全落在延迟主导区。
   `--bucket-mb 1` vs `25` vs `200`，通信次数会单调下降，但耗时不是单调的。
2. **为什么大桶也不好**：最后一个桶要等反向**全部结束**才能开始通信，
   完全丧失与反向计算重叠的机会。

**overlap 只在 GPU 上能观察到**：gloo 在 CPU 上没有独立的通信 stream，
通信与计算基本串行，CPU 版只能验证「通信次数 vs 分桶」这一半。
"""

from __future__ import annotations

import argparse
import contextlib
import time

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import common
from mem_ledger import model_states
from model import build_model, count_params, synthetic_batch
from train_single import ledger_kind

NCCL_KEYS = ("nccl", "allreduce", "all_reduce", "c10d")


def profile_gpu(step_fn, n_active: int = 3) -> dict:
    from torch.profiler import ProfilerActivity, profile, schedule

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=2, active=n_active),
        record_shapes=False,
    ) as prof:
        for _ in range(3 + n_active):
            step_fn()
            prof.step()

    comm_us, comm_calls, total_us = 0.0, 0, 0.0
    for evt in prof.key_averages():
        dev_us = getattr(evt, "device_time_total", None)
        if dev_us is None:
            dev_us = getattr(evt, "cuda_time_total", 0.0)
        total_us += max(dev_us, 0.0)
        name = evt.key.lower()
        if any(k in name for k in NCCL_KEYS):
            comm_us += max(dev_us, 0.0)
            comm_calls += evt.count

    out = common.out_dir() / "traces"
    out.mkdir(parents=True, exist_ok=True)
    return {
        "comm_time_s": comm_us / 1e6 / n_active,
        "comm_calls": comm_calls // max(n_active, 1),
        "device_time_s": total_us / 1e6 / n_active,
        "_prof": prof,
    }


def profile_cpu(ddp, x, y, opt, iters: int) -> dict:
    """降级版：手工打点。

    做法是量两次反向：一次用 no_sync（纯本地计算），一次正常（含 all_reduce），
    差值就是通信的**串行**耗时。注意这个口径在 GPU 上会严重低估——那里通信
    与反向是重叠的，差值反而接近 0。所以它只适合 CPU/gloo。
    """
    loss_fn = torch.nn.MSELoss()

    def measure(sync: bool) -> float:
        for _ in range(2):  # 预热
            opt.zero_grad(set_to_none=True)
            with contextlib.nullcontext() if sync else ddp.no_sync():
                loss_fn(ddp(x), y).backward()
        t0 = time.perf_counter()
        for _ in range(iters):
            opt.zero_grad(set_to_none=True)
            with contextlib.nullcontext() if sync else ddp.no_sync():
                loss_fn(ddp(x), y).backward()
        return (time.perf_counter() - t0) / iters

    t_local = measure(sync=False)
    t_sync = measure(sync=True)
    opt.zero_grad(set_to_none=True)

    return {
        "comm_time_s": max(t_sync - t_local, 0.0),
        # gloo 下 DDP 仍按 bucket 分批 all_reduce，但没有 profiler 就数不到真实次数
        "comm_calls": None,
        "device_time_s": None,
        "_prof": None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="一步训练的通信占比")
    common.add_common_args(ap)
    ap.add_argument("--bucket-mb", type=int, default=25)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--profile-iters", type=int, default=3)
    ap.add_argument("--export-trace", action="store_true")
    args = ap.parse_args()

    backend = common.pick_backend(args.backend)
    rank, world, local_rank = common.setup(backend)
    device = common.resolve_device(args.device, local_rank)

    torch.manual_seed(1234)
    model = build_model(args.hidden, args.layers, args.expand).to(device)
    model.train()
    ddp = DDP(
        model,
        device_ids=[local_rank] if backend == "nccl" else None,
        bucket_cap_mb=args.bucket_mb,
        gradient_as_bucket_view=True,
    )
    opt = torch.optim.AdamW(ddp.parameters(), lr=args.lr)
    params = count_params(model)
    x, y = synthetic_batch(args.batch, args.seq, args.hidden, device, seed=1234 + rank)
    loss_fn = torch.nn.MSELoss()

    counter = {"i": 0}

    def step() -> None:
        i = counter["i"]
        counter["i"] = i + 1
        syncing = (i % args.accum) == (args.accum - 1)
        with contextlib.nullcontext() if syncing else ddp.no_sync():
            (loss_fn(ddp(x), y) / args.accum).backward()
        if syncing:
            opt.step()
            opt.zero_grad(set_to_none=True)

    step_time = common.timed_steps(step, args.steps, device, warmup=args.warmup)

    if device.type == "cuda":
        prof_res = profile_gpu(step, n_active=args.profile_iters)
    else:
        prof_res = profile_cpu(ddp, x, y, opt, iters=args.profile_iters)

    prof = prof_res.pop("_prof", None)
    tag = args.tag or f"prof_bk{args.bucket_mb}"

    if prof is not None and rank == 0:
        sort_key = "device_time_total"
        try:
            print(prof.key_averages().table(sort_by=sort_key, row_limit=20))
        except (KeyError, AssertionError):
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        if args.export_trace:
            tp = common.out_dir() / "traces" / f"{tag}.json"
            tp.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(tp))
            print(f"[写出] {tp}   ← 用 chrome://tracing 或 Perfetto 打开看重叠")

    ratio = prof_res["comm_time_s"] / step_time if step_time > 0 else None
    ledger = model_states(params, ledger_kind(args.precision), 0, world)
    rec = common.build_record(
        tag=tag,
        backend=backend,
        device=device,
        hidden=args.hidden,
        layers=args.layers,
        params=params,
        batch_per_rank=args.batch,
        seq=args.seq,
        precision=args.precision,
        step_time_s=step_time,
        mem=common.MemProbe(device).read(),
        ledger_16phi_bytes=int(ledger["total"]),
        extra={
            "strategy": "ddp",
            "bucket_mb": args.bucket_mb,
            "accum": args.accum,
            "comm_ratio": ratio,
            **prof_res,
        },
    )
    common.dump(rec, common.record_path(tag, world, rank))

    common.rank0(f"bucket_mb={args.bucket_mb} accum={args.accum} world={world}")
    common.rank0(f"  每步       = {step_time*1e3:.2f} ms")
    common.rank0(f"  通信耗时   = {prof_res['comm_time_s']*1e3:.2f} ms")
    common.rank0(f"  通信次数   = {prof_res['comm_calls'] if prof_res['comm_calls'] is not None else '不可用（需 GPU profiler）'}")
    if ratio is not None:
        common.rank0(f"  通信占比   = {ratio:.1%}")
        if device.type == "cpu":
            common.rank0("  ↑ CPU/gloo 口径只是【串行耗时】占比，不含重叠信息。")
            common.rank0("    真正的 overlap 只能在 GPU 上用 profiler 的时间轴看到。")

    common.teardown()


if __name__ == "__main__":
    main()
