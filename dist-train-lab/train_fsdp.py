"""FullyShardedDataParallel：把 16Φ 切成 16Φ/N，代价是通信量约 1.5 倍。

三档 sharding_strategy 与 ZeRO 分级的对应（注意**不是一一对应**）：

    NO_SHARD       ≈ ZeRO-0 = DDP
    SHARD_GRAD_OP  ≈ ZeRO-2（同时切了梯度与优化器状态）
    FULL_SHARD     ≈ ZeRO-3
    HYBRID_SHARD     1 < F < W，跨机时才有意义

FSDP 的枚举里**没有 ZeRO-1**。PyTorch 里 ZeRO-1 的对应物是
ZeroRedundancyOptimizer + DDP，见 train_ddp.py --zero1。
"""

from __future__ import annotations

import argparse
import functools

import torch

import common
from mem_ledger import model_states
from model import FFNBlock, build_model, count_params, set_activation_checkpointing, synthetic_batch


STRATEGY_TO_ZERO = {"none": 0, "grad_op": 2, "full": 3, "hybrid": 3}


def build_fsdp(model, args, local_rank: int, device: torch.device):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "none": ShardingStrategy.NO_SHARD,
        "hybrid": ShardingStrategy.HYBRID_SHARD,
    }[args.strategy]

    # 以 FFNBlock 为分片单元。传 None 会让整个模型成为一个单元——前向时要
    # all_gather 全部参数，等于没分片。这是 FSDP 最典型的配置错误。
    policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={FFNBlock})

    mp = None
    if args.precision == "amp":
        # 只有这样才真正把参数压到 2 字节，对上「混合精度 2+2+12」那本账。
        # torch.autocast 不改变参数存储，对的是 fp32 那一栏。
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )

    kwargs = dict(
        sharding_strategy=strategy,
        auto_wrap_policy=policy,
        mixed_precision=mp,
        limit_all_gathers=True,  # 限制预取深度，压住 all_gather 造成的显存尖峰
        forward_prefetch=True,
        use_orig_params=True,
    )
    if device.type == "cuda":
        kwargs["device_id"] = local_rank
    return FSDP(model, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description="FSDP")
    common.add_common_args(ap)
    ap.add_argument("--strategy", choices=["none", "grad_op", "full", "hybrid"], default="full")
    ap.add_argument("--activation-checkpointing", type=int, choices=[0, 1], default=0)
    args = ap.parse_args()

    backend = common.pick_backend(args.backend)
    rank, world, local_rank = common.setup(backend)
    device = common.resolve_device(args.device, local_rank)

    torch.manual_seed(1234)
    model = build_model(args.hidden, args.layers, args.expand).to(device)
    n_ckpt = set_activation_checkpointing(model, bool(args.activation_checkpointing))
    params = count_params(model)  # 必须在 wrap 之前数，wrap 后参数已被分片
    model.train()

    try:
        fsdp = build_fsdp(model, args, local_rank, device)
    except Exception as e:
        common.rank0(
            f"[SKIP] FSDP 构造失败：{type(e).__name__}: {e}\n"
            "  FSDP 对 gloo/CPU 的支持随版本而异。无卡时请改用：\n"
            "    python mem_ledger.py --hidden ... --ladder --n 8\n"
            "  做纸面推演，或用 train_ddp.py --zero1 验 ZeRO-1 那一档。"
        )
        common.teardown()
        raise SystemExit(2)

    # optimizer 必须在 wrap 之后构造：参数对象已被 FSDP 替换掉了
    opt = torch.optim.AdamW(fsdp.parameters(), lr=args.lr)

    x, y = synthetic_batch(args.batch, args.seq, args.hidden, device, seed=1234 + rank)
    loss_fn = torch.nn.MSELoss()

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        loss_fn(fsdp(x), y).backward()
        opt.step()

    probe = common.MemProbe(device)
    probe.reset()
    step_time = common.timed_steps(step, args.steps, device, warmup=args.warmup)
    mem = probe.read()

    zero_stage = STRATEGY_TO_ZERO[args.strategy]
    # FSDP + MixedPrecision 才是经典的 2+2+12；否则是 fp32 的 4+4+8
    kind = "mixed" if args.precision == "amp" else "fp32"
    ledger = model_states(params, kind, zero_stage=zero_stage, n=world)

    tag = args.tag or f"fsdp_{args.strategy}"
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
            "strategy": f"fsdp_{args.strategy}",
            "sharding_strategy": args.strategy,
            "zero_stage": zero_stage,
            "ledger_kind": kind,
            "activation_checkpointing": bool(args.activation_checkpointing),
            "ac_blocks_patched": n_ckpt,
        },
    )
    common.dump(rec, common.record_path(tag, world, rank))

    common.rank0(f"strategy={args.strategy} (≈ZeRO-{zero_stage})  world={world}  backend={backend}")
    print(f"  [rank {rank}] 每步 {step_time*1e3:.2f} ms", end="")
    if mem["peak_alloc_bytes"] is not None:
        print(f"   峰值 alloc {mem['peak_alloc_bytes']/2**30:.3f} GiB")
    else:
        print()
    common.rank0(f"  模型状态手算 = {ledger['total']/2**30:.3f} GiB  [{kind} 口径, n={world}]")
    common.rank0(f"  tokens/s(全局) = {rec['tokens_per_s_global']:,.1f}")

    common.teardown()


if __name__ == "__main__":
    main()
