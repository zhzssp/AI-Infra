"""所有脚本共用的四件事：进程组、显存计量、吞吐计时、产物落盘。

设计原则：**没有 GPU 也要能跑**。进程模型与集合通信语义在 CPU/gloo 上完全一致，
只有「性能与真实显存行为」需要真卡。所以本文件里凡是碰 CUDA 的地方都有 CPU 分支。
"""

from __future__ import annotations

import json
import os
import pathlib
import platform
import time
from typing import Any, Callable

import torch
import torch.distributed as dist


# --------------------------------------------------------------------------
# 进程组
# --------------------------------------------------------------------------


def pick_backend(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    return "nccl" if torch.cuda.is_available() else "gloo"


def setup(backend: str | None = None) -> tuple[int, int, int]:
    """torchrun 注入 RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT。

    返回 (rank, world_size, local_rank)。未经 torchrun 启动时退化为单进程。
    """
    backend = pick_backend(backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 必须在 init_process_group 之前绑定设备：否则 NCCL 建链时所有 rank 都
    # 认为自己在 cuda:0，轻则性能塌陷，重则死锁。
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("backend=nccl 但 torch.cuda.is_available() 为 False")
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size(), local_rank


def teardown() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world() -> int:
    return dist.get_world_size() if is_dist() else 1


def rank0(*args: Any, **kwargs: Any) -> None:
    if get_rank() == 0:
        print(*args, **kwargs)


def resolve_device(arg_device: str, local_rank: int = 0) -> torch.device:
    if arg_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda 但本机没有可用的 CUDA 设备")
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# 显存计量
# --------------------------------------------------------------------------


class MemProbe:
    """allocated 是张量真实占用；reserved 是 caching allocator 向驱动要的量。

    两者之差 ≈ 碎片 + 尚未归还的缓存块。只看 allocated 会低估真实压力，
    只看 nvidia-smi 又会把 CUDA context（数百 MB）算进来——三个口径都不一样。
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.enabled = device.type == "cuda"

    def reset(self) -> None:
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()

    def read(self) -> dict[str, int | None]:
        if not self.enabled:
            return {"peak_alloc_bytes": None, "peak_reserved_bytes": None}
        return {
            "peak_alloc_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
        }


# --------------------------------------------------------------------------
# 吞吐计时
# --------------------------------------------------------------------------


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_steps(
    step_fn: Callable[[], Any],
    n_steps: int,
    device: torch.device,
    warmup: int = 3,
) -> float:
    """返回每步平均秒数。

    warmup 不能省的两个理由，都会让账对不上：
      1. Adam 的 exp_avg / exp_avg_sq 是**第一次 step() 时**才分配的（共 8Φ）。
         只跑一步测显存，会少掉一大块。
      2. NCCL 建链、cuBLAS handle、CUDA context 都在首次调用时才付费。
    """
    for _ in range(warmup):
        step_fn()
    sync(device)
    if is_dist():
        dist.barrier()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        step_fn()
    sync(device)  # 不同步的话测到的是 kernel「下发」耗时，吞吐会虚高一个数量级
    return (time.perf_counter() - t0) / max(n_steps, 1)


# --------------------------------------------------------------------------
# 环境画像（这台机器的常数，所有实验都要引用）
# --------------------------------------------------------------------------


def device_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device_type": device.type,
    }
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        try:
            # NCCL 在 Windows 上不存在，别让画像函数把整个实验带崩
            nccl_ver = ".".join(str(v) for v in torch.cuda.nccl.version())
        except (AttributeError, RuntimeError, OSError):
            nccl_ver = None
        info.update(
            {
                "device_name": props.name,
                "capability": f"sm_{props.major}{props.minor}",
                "total_mem_bytes": int(props.total_memory),
                "multi_processor_count": int(props.multi_processor_count),
                "cuda": torch.version.cuda,
                "nccl": nccl_ver,
            }
        )
    else:
        info["device_name"] = platform.processor() or "cpu"
    return info


def p2p_matrix() -> dict[str, Any] | None:
    """卡间能否直连。消费级 GeForce（含 RTX 4090/5090）的 P2P 是关闭的，
    NCCL 只能退回经 host 中转的 shared-memory 传输——这正是本 lab 最重要的
    背景常数：它让「通信代价」在墙钟时间上放大到无法忽视。
    """
    if not torch.cuda.is_available():
        return None
    n = torch.cuda.device_count()
    if n < 2:
        return {"n_gpus": n, "note": "单卡，无 P2P 可测"}
    matrix = [
        [bool(torch.cuda.can_device_access_peer(i, j)) if i != j else True for j in range(n)]
        for i in range(n)
    ]
    any_p2p = any(matrix[i][j] for i in range(n) for j in range(n) if i != j)
    return {"n_gpus": n, "can_access_peer": matrix, "any_p2p": any_p2p}


# --------------------------------------------------------------------------
# 产物落盘
# --------------------------------------------------------------------------


def project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def out_dir() -> pathlib.Path:
    d = pathlib.Path(os.environ.get("OUT_DIR", project_root() / "out"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def dump(record: dict[str, Any], path: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return p


def record_path(tag: str, world: int, rank: int) -> pathlib.Path:
    """每个 rank 写自己的文件。分片实验里各 rank 峰值可能不同，
    只看 rank 0 会漏掉负载不均。"""
    return out_dir() / f"{tag}_ws{world}_r{rank}.json"


def build_record(
    *,
    tag: str,
    backend: str,
    device: torch.device,
    hidden: int,
    layers: int,
    params: int,
    batch_per_rank: int,
    seq: int,
    precision: str,
    step_time_s: float,
    mem: dict[str, int | None],
    ledger_16phi_bytes: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    world = get_world()
    tokens_global = batch_per_rank * seq * world / step_time_s if step_time_s > 0 else 0.0
    rec: dict[str, Any] = {
        "tag": tag,
        "backend": backend,
        "world_size": world,
        "rank": get_rank(),
        "hidden": hidden,
        "layers": layers,
        "params": params,
        "batch_per_rank": batch_per_rank,
        "seq": seq,
        "precision": precision,
        "step_time_s": step_time_s,
        # 吞吐口径必须是全局值，否则 DDP 的「线性加速」会被口径吃掉
        "samples_per_s_global": batch_per_rank * world / step_time_s if step_time_s > 0 else 0.0,
        "tokens_per_s_global": tokens_global,
        "ledger_16phi_bytes": ledger_16phi_bytes,
    }
    rec.update(mem)
    rec.update(device_info(device))
    if extra:
        rec.update(extra)
    return rec


def add_common_args(parser) -> None:
    """所有训练脚本共用的参数，保证各实验口径一致。"""
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--backend", choices=["gloo", "nccl"], default=None)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--expand", type=int, default=4)
    parser.add_argument("--batch", type=int, default=4, help="per-rank micro-batch")
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--precision", choices=["fp32", "amp"], default="fp32")
    parser.add_argument("--tag", type=str, default=None)
