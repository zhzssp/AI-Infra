"""DistributedDataParallel。

本脚本要证明的核心不变量：**各 rank 喂的是不同数据，但梯度 all_reduce 取平均后
参数保持逐比特一致**。这个不变量和有没有 GPU 毫无关系，所以 CPU/gloo 两进程
就能把 DDP 学明白——上机时只需把 --device cuda --backend nccl 换上去。
"""

from __future__ import annotations

import argparse
import contextlib

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import common
from mem_ledger import model_states
from model import build_model, count_params, set_activation_checkpointing, synthetic_batch
from train_single import ledger_kind


def flat_params(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()])


def flat_grads(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
            for p in module.parameters()
        ]
    )


def check_consistency(ddp, x, y, world: int) -> dict[str, float]:
    """两项检查，对应 DDP 的两条语义。"""
    loss_fn = torch.nn.MSELoss()

    # 检查一：参数一致性。DDP 的核心不变量。
    flat = flat_params(ddp.module)
    bufs = [torch.empty_like(flat) for _ in range(world)]
    dist.all_gather(bufs, flat)
    param_diff = max((bufs[0] - b).abs().max().item() for b in bufs)

    # 检查二：梯度确实被【平均】了，而不是求和。
    ddp.zero_grad(set_to_none=True)
    with ddp.no_sync():  # 只算本地梯度，不触发 all_reduce
        loss_fn(ddp(x), y).backward()
    g_local = flat_grads(ddp.module).clone()

    ddp.zero_grad(set_to_none=True)
    loss_fn(ddp(x), y).backward()  # 这一次会触发 all_reduce
    g_sync = flat_grads(ddp.module).clone()

    locals_ = [torch.empty_like(g_local) for _ in range(world)]
    dist.all_gather(locals_, g_local)
    grad_diff = (g_sync - torch.stack(locals_).mean(0)).abs().max().item()

    ddp.zero_grad(set_to_none=True)
    return {"param_diff": param_diff, "grad_diff": grad_diff}


def main() -> None:
    ap = argparse.ArgumentParser(description="DDP")
    common.add_common_args(ap)
    ap.add_argument("--check-consistency", action="store_true")
    ap.add_argument("--bucket-mb", type=int, default=25, help="梯度分桶大小")
    ap.add_argument("--accum", type=int, default=1, help="用 no_sync 累积 K-1 步再同步一次")
    ap.add_argument("--zero1", action="store_true", help="用 ZeroRedundancyOptimizer 做真正的 ZeRO-1")
    ap.add_argument("--activation-checkpointing", type=int, choices=[0, 1], default=0)
    args = ap.parse_args()

    backend = common.pick_backend(args.backend)
    rank, world, local_rank = common.setup(backend)
    device = common.resolve_device(args.device, local_rank)

    # 模型初始化种子必须相同；DDP 构造时还会再广播一次 rank0 的参数兜底
    torch.manual_seed(1234)
    model = build_model(args.hidden, args.layers, args.expand).to(device)
    set_activation_checkpointing(model, bool(args.activation_checkpointing))
    model.train()

    ddp = DDP(
        model,
        device_ids=[local_rank] if backend == "nccl" else None,
        bucket_cap_mb=args.bucket_mb,
        gradient_as_bucket_view=True,
    )

    if args.zero1:
        from torch.distributed.optim import ZeroRedundancyOptimizer

        # FSDP 的枚举里没有精确对应 ZeRO-1 的档位（SHARD_GRAD_OP 已经是 ZeRO-2），
        # PyTorch 里 ZeRO-1 的对应物就是这个：只切优化器状态。
        opt = ZeroRedundancyOptimizer(
            ddp.parameters(), optimizer_class=torch.optim.AdamW, lr=args.lr
        )
    else:
        opt = torch.optim.AdamW(ddp.parameters(), lr=args.lr)

    params = count_params(model)
    # 各 rank 必须拿到不同数据，否则 all_reduce 前后梯度相同，等于没验证
    x, y = synthetic_batch(args.batch, args.seq, args.hidden, device, seed=1234 + rank)
    loss_fn = torch.nn.MSELoss()

    counter = {"i": 0}

    def step() -> None:
        i = counter["i"]
        counter["i"] = i + 1
        # 累积期内用 no_sync 抑制 all_reduce，只在最后一步同步
        syncing = (i % args.accum) == (args.accum - 1)
        ctx = contextlib.nullcontext() if syncing else ddp.no_sync()
        with ctx:
            loss = loss_fn(ddp(x), y) / args.accum  # 累积必须缩放，否则等效放大了 lr
            loss.backward()
        if syncing:
            opt.step()
            opt.zero_grad(set_to_none=True)

    probe = common.MemProbe(device)
    probe.reset()
    step_time = common.timed_steps(step, args.steps, device, warmup=args.warmup)
    mem = probe.read()

    checks = check_consistency(ddp, x, y, world) if args.check_consistency else {}

    zero_stage = 1 if args.zero1 else 0
    ledger = model_states(params, ledger_kind(args.precision), zero_stage=zero_stage, n=world)
    tag = args.tag or ("ddp_zero1" if args.zero1 else "ddp")
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
        mem=mem,
        ledger_16phi_bytes=int(ledger["total"]),
        extra={
            "strategy": "ddp_zero1" if args.zero1 else "ddp",
            "bucket_mb": args.bucket_mb,
            "accum": args.accum,
            "zero_stage": zero_stage,
            "ledger_kind": ledger_kind(args.precision),
            "activation_checkpointing": bool(args.activation_checkpointing),
            **checks,
        },
    )
    common.dump(rec, common.record_path(tag, world, rank))

    common.rank0(f"world_size={world}  backend={backend}  device={device}")
    print(f"  [rank {rank}] local_rank={local_rank}  每步 {step_time*1e3:.2f} ms")
    if mem["peak_alloc_bytes"] is not None:
        print(f"  [rank {rank}] 峰值 alloc {mem['peak_alloc_bytes']/2**30:.3f} GiB")
    if checks:
        common.rank0(f"  param_diff = {checks['param_diff']:.3e}   （应为 0）")
        common.rank0(f"  grad_diff  = {checks['grad_diff']:.3e}   （应 < 1e-6）")
    common.rank0(f"  模型状态手算 = {ledger['total']/2**30:.3f} GiB  (ZeRO-{zero_stage}, n={world})")
    common.rank0(f"  tokens/s(全局) = {rec['tokens_per_s_global']:,.1f}")

    common.teardown()


if __name__ == "__main__":
    main()
