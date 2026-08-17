"""最小张量并行：列切 + all_gather，行切 + all_reduce。

SPMD 的核心等式：**分片 + 对应的 collective = 语义等价**。

每个中间张量都有一个「分布式状态」，三种体系叫法不同但说的是一件事：

    OneFlow SBP   S(dim)      B            P
    DTensor       Shard(dim)  Replicate()  Partial()
    本文件注释     Split(dim)  Broadcast    Partial

`Partial`（部分和）这个状态必须存在：行切之后的中间结果既不是完整值也不是
分片，缺了它就无法在类型系统里做分片传播推导。

在**没有 NVLink** 的机器上跑 --bench，你会看到 TP 的耗时被 all_reduce 完全支配。
这就是「TP size 必须 ≤ 单节点卡数、且节点内要有高速互联」的定量依据。
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.distributed as dist

import common


class CollectiveCounter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def note(self, name: str) -> None:
        self.calls.append(name)

    def summary(self) -> str:
        from collections import Counter

        c = Counter(self.calls)
        return ", ".join(f"{k}×{v}" for k, v in c.items()) or "无"


def column_parallel(x, w, rank, world, ctr):
    """列切：W 按 out 维切 → 各 rank 得到 y 的一部分 → all_gather 拼回。

    x  : Broadcast   （各 rank 相同）
    W  : Split(1)    （按输出维切）
    y  : Split(-1)   → all_gather → Broadcast
    """
    w_shard = w.chunk(world, dim=1)[rank].contiguous()
    y_local = x @ w_shard
    out = [torch.empty_like(y_local) for _ in range(world)]
    dist.all_gather(out, y_local)
    ctr.note("all_gather")
    return torch.cat(out, dim=-1)


def row_parallel(x, w, rank, world, ctr):
    """行切：W 按 in 维切 → x 的最后一维也要【同步切】→ 部分和 → all_reduce。

    x  : Split(-1)
    W  : Split(0)
    y  : Partial     → all_reduce → Broadcast
    """
    w_shard = w.chunk(world, dim=0)[rank].contiguous()
    x_shard = x.chunk(world, dim=-1)[rank].contiguous()
    y_partial = x_shard @ w_shard  # Partial：部分和，不是最终值
    dist.all_reduce(y_partial, op=dist.ReduceOp.SUM)
    ctr.note("all_reduce")
    return y_partial


def megatron_ffn(x, w1, w2, rank, world, ctr):
    """Megatron 的 FFN：先列切再行切，整块**只需要一次 collective**。

    x ─Broadcast─▶ fc1(Split(1)) ─▶ h: Split(-1) ─GELU─▶ 仍是 Split(-1)
                                                            │
                                        逐元素算子不改变分片状态，无需通信
                                                            ▼
                                      fc2(Split(0)) ─▶ y: Partial ─all_reduce─▶ B

    反过来「先行切再列切」的话，中间要先 all_reduce 恢复成 Broadcast 才能进下一层，
    整块要两次通信。顺序不是随便定的。
    """
    w1_shard = w1.chunk(world, dim=1)[rank].contiguous()  # Split(1)
    h_local = torch.nn.functional.gelu(x @ w1_shard)  # Split(-1)
    w2_shard = w2.chunk(world, dim=0)[rank].contiguous()  # Split(0)，与 h 的切分对齐
    y_partial = h_local @ w2_shard  # Partial
    dist.all_reduce(y_partial, op=dist.ReduceOp.SUM)
    ctr.note("all_reduce")
    return y_partial


def run_dtensor(x, w, hidden, world) -> dict[str, str]:
    """L3-DIST-13：同一件事交给框架按标注自动推导，不手写任何 collective。"""
    try:
        from torch.distributed.device_mesh import init_device_mesh

        try:
            from torch.distributed.tensor import Replicate, Shard, distribute_tensor
        except ImportError:  # 较早版本
            from torch.distributed._tensor import Replicate, Shard, distribute_tensor
    except ImportError as e:
        return {"status": f"unavailable: {e}"}

    mesh = init_device_mesh(x.device.type, (world,))
    dw = distribute_tensor(w, mesh, [Shard(1)])  # 列切，对应 SBP 的 S(1)
    dx = distribute_tensor(x, mesh, [Replicate()])  # 对应 SBP 的 B
    dy = dx @ dw  # 没写任何 collective，框架按标注推导出局部计算
    placements_col = str(dy.placements)
    y_full = dy.redistribute(mesh, [Replicate()]).to_local()  # 自动插入 all_gather

    ref = x @ w
    diff_col = (y_full - ref).abs().max().item()

    # 行切：dy 的状态应该是 Partial，redistribute 时框架会插 all_reduce
    dw_row = distribute_tensor(w, mesh, [Shard(0)])
    dx_row = distribute_tensor(x, mesh, [Shard(1)])
    dy_row = dx_row @ dw_row
    placements_row = str(dy_row.placements)
    y_row = dy_row.redistribute(mesh, [Replicate()]).to_local()
    diff_row = (y_row - ref).abs().max().item()

    return {
        "status": "ok",
        "col_placements": placements_col,
        "col_max_abs_diff": f"{diff_col:.3e}",
        "row_placements": placements_row,
        "row_max_abs_diff": f"{diff_row:.3e}",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="最小张量并行")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--backend", choices=["gloo", "nccl"], default=None)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--check", default="col,row,ffn")
    ap.add_argument("--count-collectives", action="store_true")
    ap.add_argument("--dtensor", action="store_true")
    ap.add_argument("--bench", action="store_true", help="计时，看 TP 的通信占比")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args()

    backend = common.pick_backend(args.backend)
    rank, world, local_rank = common.setup(backend)
    device = common.resolve_device(args.device, local_rank)

    h = args.hidden
    torch.manual_seed(1234)  # 各 rank 必须持有相同的 w 与 x，才能谈「与单卡等价」
    x = torch.randn(args.batch, h, device=device)
    w = torch.randn(h, 4 * h, device=device)
    w1 = torch.randn(h, 4 * h, device=device)
    w2 = torch.randn(4 * h, h, device=device)

    checks = [c.strip() for c in args.check.split(",") if c.strip()]
    ctr = CollectiveCounter()
    results: dict[str, float] = {}

    if "col" in checks:
        ref = x @ w
        got = column_parallel(x, w, rank, world, ctr)
        results["col"] = (got - ref).abs().max().item()

    if "row" in checks:
        ref = x @ w
        got = row_parallel(x, w, rank, world, ctr)
        results["row"] = (got - ref).abs().max().item()

    if "ffn" in checks:
        ref = torch.nn.functional.gelu(x @ w1) @ w2
        before = len(ctr.calls)
        got = megatron_ffn(x, w1, w2, rank, world, ctr)
        results["ffn"] = (got - ref).abs().max().item()
        ffn_collectives = len(ctr.calls) - before

    common.rank0(f"world={world}  backend={backend}  device={device}  hidden={h}")
    ok_all = True
    for k, v in results.items():
        # 归约顺序变了，浮点结果不会逐比特相同——这是正常的，别用 atol=0
        ok = v < args.atol
        ok_all = ok_all and ok
        common.rank0(f"  {k:<4} max_abs_diff = {v:.3e}   {'✔' if ok else '✘ 超出 atol'}")

    if "ffn" in checks:
        common.rank0(f"  FFN 整块用了 {ffn_collectives} 次 collective（应为 1）")
    if args.count_collectives:
        common.rank0(f"  collective 统计: {ctr.summary()}")

    if args.bench and world > 1:
        # 对照组：同样的计算量，但不切分、不通信
        def tp_step():
            megatron_ffn(x, w1, w2, rank, world, CollectiveCounter())

        def local_step():
            torch.nn.functional.gelu(x @ w1) @ w2

        for fn in (tp_step, local_step):
            for _ in range(5):
                fn()
        common.sync(device)
        dist.barrier()

        t0 = time.perf_counter()
        for _ in range(args.iters):
            tp_step()
        common.sync(device)
        t_tp = (time.perf_counter() - t0) / args.iters

        t0 = time.perf_counter()
        for _ in range(args.iters):
            local_step()
        common.sync(device)
        t_local = (time.perf_counter() - t0) / args.iters

        common.rank0("")
        common.rank0(f"  TP({world} 卡) 每次 FFN  = {t_tp*1e6:>10.1f} us")
        common.rank0(f"  单卡完整 FFN            = {t_local*1e6:>10.1f} us")
        if t_tp > 0:
            common.rank0(f"  加速比 = {t_local/t_tp:.2f}x   （理想是 {world}x）")
            common.rank0(
                "  ↑ 若远低于 1，说明 all_reduce 的代价已经完全盖过了切分省下的计算。"
            )
            common.rank0(
                "    在没有 NVLink / P2P 的机器上这是**预期结果**，而不是配置错误——"
            )
            common.rank0("    它正是「TP 必须放在高带宽域内」这条工程约束的来源。")

    if args.dtensor:
        common.rank0("")
        info = run_dtensor(x, w, h, world)
        for k, v in info.items():
            common.rank0(f"  dtensor.{k} = {v}")
        common.rank0("  ↑ 列切时 placements 应含 Shard(1)，行切时应含 Partial。")
        common.rank0("    能在 placements 里看到 Partial，是这条最直接的过关证据。")

    common.teardown()
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
