"""单进程 baseline（CPU 或单卡）。所有多卡实验的对照组。

吞吐口径统一成两个数：samples_per_s 与 tokens_per_s，且**都取全局值**
（乘 world_size）。否则 DDP 的「线性加速」会被口径悄悄吃掉。
"""

from __future__ import annotations

import argparse
import contextlib

import torch

import common
from mem_ledger import model_states
from model import build_model, count_params, params_closed_form, set_activation_checkpointing, synthetic_batch


def ledger_kind(precision: str) -> str:
    """torch 原生 AMP 与 DeepSpeed 式「混合精度」不是一回事，这里必须分清。

    经典混合精度（DeepSpeed/Megatron，也是 16Φ 那张表的出处）：
        fp16 参数 2Φ + fp16 梯度 2Φ + (fp32 master 4Φ + m 4Φ + v 4Φ) = 16Φ
    torch.autocast：参数与梯度**始终是 fp32**，只在算子入口临时 cast，
        所以模型状态是 4Φ + 4Φ + 8Φ = 16Φ —— 合计同样 16Φ，但构成是 fp32 那一栏。

    结论：用 `--precision amp` 实测时，要对的是 fp32 口径的账。真正能把
    参数压到 2Φ 的是 FSDP 的 MixedPrecision(param_dtype=bf16)，见 train_fsdp.py。
    """
    return "fp32"


def build_step(model, opt, x, y, device, precision):
    loss_fn = torch.nn.MSELoss()
    use_amp = precision == "amp" and device.type == "cuda"

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_amp
            else contextlib.nullcontext()
        )
        with ctx:
            loss = loss_fn(model(x), y)
        # bf16 的动态范围与 fp32 相同，不需要 GradScaler（fp16 才需要）
        loss.backward()
        opt.step()

    return step


def main() -> None:
    ap = argparse.ArgumentParser(description="单进程 baseline")
    common.add_common_args(ap)
    ap.add_argument("--activation-checkpointing", type=int, choices=[0, 1], default=0)
    args = ap.parse_args()

    device = common.resolve_device(args.device)
    torch.manual_seed(1234)

    model = build_model(args.hidden, args.layers, args.expand).to(device)
    n_ckpt = set_activation_checkpointing(model, bool(args.activation_checkpointing))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    params = count_params(model)
    closed = params_closed_form(args.hidden, args.layers, args.expand)
    x, y = synthetic_batch(args.batch, args.seq, args.hidden, device, seed=1234)

    probe = common.MemProbe(device)
    probe.reset()
    step = build_step(model, opt, x, y, device, args.precision)
    step_time = common.timed_steps(step, args.steps, device, warmup=args.warmup)
    mem = probe.read()

    ledger = model_states(params, ledger_kind(args.precision), zero_stage=0, n=1)
    tag = args.tag or f"single_{args.device}"
    rec = common.build_record(
        tag=tag,
        backend="none",
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
            "strategy": "single",
            "activation_checkpointing": bool(args.activation_checkpointing),
            "ac_blocks_patched": n_ckpt,
            "ledger_kind": ledger_kind(args.precision),
        },
    )
    path = common.dump(rec, common.out_dir() / f"{tag}.json")

    print(f"参数量 Φ      = {params:,}  (闭式解 {closed:,}，误差 {abs(params-closed)/closed:.2e})")
    print(f"模型状态 16Φ  = {ledger['total']/2**30:.3f} GiB   [{ledger_kind(args.precision)} 口径]")
    if mem["peak_alloc_bytes"] is not None:
        alloc = mem["peak_alloc_bytes"]
        print(f"实测峰值 alloc = {alloc/2**30:.3f} GiB   差额 {(alloc-ledger['total'])/2**30:+.3f} GiB")
        print(f"实测峰值 resvd = {mem['peak_reserved_bytes']/2**30:.3f} GiB")
    else:
        print("实测峰值      = 不可用（CPU 上 max_memory_allocated 无意义）")
    print(f"每步耗时      = {step_time*1e3:.2f} ms")
    print(f"tokens/s      = {rec['tokens_per_s_global']:,.1f}")
    print(f"[写出] {path}")


if __name__ == "__main__":
    main()
