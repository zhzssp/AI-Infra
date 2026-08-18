# 检验体系 08｜分布式训练

> **导航**：[检验体系总纲](./README.md) · [总规划 §3.1 分布式训练基础](../../README.md#31-分布式训练基础) · [论文精读 01](../paper-notes/01-efficient-training-distributed-infra.md) · [横切概念 §9 分片如何变成 IR](../learning-guides/ai-compiler-foundations-learning-guide.md#第-9-章-分布式与编译的接缝分片如何变成-ir)
>
> 分级定义（L0 复现 / L1 改一处 / L2 加组件 / L3 打通）与资源标签的含义，以 [`./README.md`](./README.md) §2、§3 为准，本文不重复。

**本册的动手项目是 [`dist-train-lab/`](../../dist-train-lab/)**，已按 [§2](#2-实验脚手架设计dist-train-lab) 的设计建好。所以这份文档承担两件事——

1. 说明**脚手架的设计意图**（[§2](#2-实验脚手架设计dist-train-lab)）：每个文件干什么、关键 API 是哪几个、哪些坑要提前写进代码。**读它是为了理解 lab 里的代码为什么那样写**；想跳过直接跑，见下面的「先跑起来」。
2. 在脚手架之上给出**分级检验条目**（[§3](#3-检验条目)）。

> **先跑起来**（无卡也能做完 13 条里的 6 条）：
>
> ```bash
> bash setup.sh --torch cpu          # 仓库根目录，不需要 root；已配好则跳过
> cd dist-train-lab
> bash scripts/run_cpu_smoke.sh      # 几十秒，覆盖 L0 全部 + L1-DIST-04 + L3-DIST-11/13 主判据
> ```
>
> 有多卡时再跑 `bash scripts/run_8gpu_wall.sh`——它把 [L2-DIST-06/07/08](#l2-dist-06ddp-vs-单卡显存几乎不降吞吐接近线性) 与 [L3-DIST-11](#l3-dist-11手写最小张量并行列切--all_gather) 的性能部分打包成四组对照，按实际卡数自动生成 `world_size = 2, 4, 8, ...` 序列。

学习文档（[根 README §3.1](../../README.md#31-分布式训练基础)、[paper-notes/01](../paper-notes/01-efficient-training-distributed-infra.md)）考的是**口述**：16Φ 是怎么来的、ZeRO-1/2/3 各除以谁、SPMD 三种标注在说什么。本文考的是**动手**：把同一个知识点变成一条能跑出数字的命令，数字对不上就是没懂。

---

## 1. 先看这个：哪些现在就能做，哪些必须等机器

这一节直接决定你的排期。**不要因为「没有卡」就把整个分布式模块往后排**——进程模型、集合通信语义、显存账本、SPMD 标注这四块核心知识，**全部可以在一台没有 GPU 的笔记本上学完并验证**。

| 现在就能做（笔记本 CPU，`本地`） | 必须等机器 |
|---------------------------------|-----------|
| [L0-DIST-01](#l0-dist-01建起-lab-骨架并在-cpu-上跑通单进程-baseline) 建 lab 骨架 + 单进程 baseline | [L1-DIST-05](#l1-dist-05单卡显存峰值实测并与-16φ-手算账对差额) 单卡显存峰值实测（`单卡GPU`） |
| [L0-DIST-02](#l0-dist-02cpugloo-两进程跑通-ddp并验证-all_reduce-后各-rank-参数一致) **CPU + gloo 两进程 DDP**，验证 all_reduce 语义 | [L2-DIST-06](#l2-dist-06ddp-vs-单卡显存几乎不降吞吐接近线性) DDP vs 单卡（`多卡GPU`） |
| [L0-DIST-03](#l0-dist-0316φ-显存账手算脚本) 16Φ 显存账手算脚本 | [L2-DIST-07](#l2-dist-07fsdp-vs-ddp显存随卡数近似-1n吞吐略降) FSDP vs DDP 对比表（`多卡GPU`） |
| [L1-DIST-04](#l1-dist-04comm_bench-在-cpugloo-上扫消息大小找延迟带宽拐点) comm_bench 找延迟/带宽拐点 | [L2-DIST-08](#l2-dist-08切换-fsdp-的-sharding_strategy-三档映射回-zero-123) sharding_strategy 三档对比（`多卡GPU`） |
| [L2-DIST-10](#l2-dist-10统计一步训练里的通信占比解释-bucketing-与-overlap) 通信占比打点（gloo 降级版可做） | [L2-DIST-09](#l2-dist-09开关激活重计算量出用算力换显存的兑换率) 激活重计算 trade-off（`多卡GPU`） |
| [L3-DIST-11](#l3-dist-11手写最小张量并行列切--all_gather) 手写最小张量并行（数值等价部分可做） | [L3-DIST-12](#l3-dist-122-节点跨机-ddpfsdp对比同机与跨机的通信耗时) 跨机通信（`多机多卡`） |
| [L3-DIST-13](#l3-dist-13spmd-三种标注体系对照表并用-dtensor-实操验证) SPMD 三种标注对照 + DTensor 实操 | |

**排期建议**：

```text
第 1~2 天（无卡）  建 lab 骨架 → L0-DIST-01/02/03 → L1-DIST-04
                   产出：进程模型与 collective 语义的手感 + 一张 16Φ 账本表
第 3 天（无卡）    L3-DIST-11 的 CPU 数值等价版 + L3-DIST-13
                   产出：能用代码解释 GSPMD/SBP 在说什么（这条对项目最值钱）
第 4 天（无卡）    L2-DIST-10 的 gloo 降级版，把 profiling 脚本调通
                   产出：机器到手当天就能直接开跑，不浪费机时
—— 以上全部完成后再申请机器（见 §5），带着脚本和预测去跑 ——
拿到卡后半天       L1-DIST-05 → L2-DIST-06/07/08/09（这四条是主判据）
有集群再补         L3-DIST-12
```

**为什么把「无卡条目」排在前面**：机时是稀缺资源。把脚本、判据、预测都在 CPU 上准备好，上机后只是把 `--device cpu --backend gloo` 换成 `--device cuda --backend nccl`，一个下午能跑完全部 L2。反过来，等到有卡才开始写脚本，会把机时浪费在调 `torchrun` 参数上。

**降级方案的诚实边界**：`单卡GPU` 及以上的条目都给了降级方案，但要清楚降级后**拿不到什么**——

| 降级能保住的 | 降级一定拿不到的 |
|-------------|-----------------|
| 逻辑正确性（进程组划分、collective 语义、数值等价） | 真实显存峰值与其构成（激活、碎片、cudnn workspace 的实际占比） |
| 相对趋势的方向（分片后应该变小、重计算后应该变慢） | 任何**比值型结论**（1.8x 加速、显存降到 0.55 倍） |
| 代码能跑通、API 用对了 | 通信与计算能否真正重叠（gloo 没有独立 stream，overlap 行为完全不同） |

所以：**降级版能让你「不被阻塞地学会」，但不能让你「拿到可写进报告的数字」。** L2 的四条主判据最终仍需真机复跑一遍。

---

## 2. 实验脚手架设计：`dist-train-lab/`

> 本节是**设计说明**。对应代码已在 [`dist-train-lab/`](../../dist-train-lab/) 里建好，命名与布局对齐仓库既有 lab（[`tvm-fatbin-lab/`](../../tvm-fatbin-lab/)、[`onnx-delegate-lab/`](../../onnx-delegate-lab/)）：脚本平铺、`scripts/*.sh` 放启动命令、产物进 `out/` 且不入库。
>
> **[L0-DIST-01](#l0-dist-01建起-lab-骨架并在-cpu-上跑通单进程-baseline) 现在有两种做法**：想练手就先把 `dist-train-lab/` 挪走、照本节说明自己写一遍，写完与仓库版本对读；时间紧就直接跑现成的，重点放在读懂 `common.py` 里那两个「必须踩到的坑」为什么要那样处理。

### 2.1 目录结构

```text
dist-train-lab/
├── README.md               # 一页说明：怎么建、怎么跑、产物在哪
├── requirements.txt        # torch / psutil（可选） / matplotlib（可选，画拐点图）
├── .gitignore              # out/ 不入库（只把最终对比表贴进 docs/notes/）
├── common.py               # 进程组初始化、计时器、显存计量、产物落盘（所有脚本共用）
├── model.py                # 参数量可调的小模型（改 hidden 就能把显存压力做出来）
├── mem_ledger.py           # 16Φ 显存账手算器（纯 CPU，不 import torch.distributed）
├── train_single.py         # 单进程 baseline（CPU 或单卡）
├── train_ddp.py            # DistributedDataParallel
├── train_fsdp.py           # FullyShardedDataParallel，可切 sharding_strategy
├── comm_bench.py           # all_reduce / all_gather 的带宽与延迟微基准
├── profile_step.py         # 一步训练的通信占比统计（torch.profiler 或手工打点）
├── tp_min.py               # 最小张量并行（列切 + all_gather），做到 L3 再写
├── report.py               # 读 out/*.json → 生成 markdown 对比表
├── scripts/
│   ├── env.sh              # 定位 python/torchrun，探测卡数与 P2P，统一实验档位
│   ├── run_cpu_smoke.sh    # 无卡冒烟：单进程 + 2 进程 gloo DDP，几十秒跑完
│   ├── run_8gpu_wall.sh    # 多卡主线：通信墙四组对照，按实际卡数自动扩展
│   ├── run_ddp.sh
│   ├── run_fsdp.sh         # 参数化 sharding_strategy
│   ├── run_comm_bench.sh
│   ├── run_multinode.sh    # 2 节点，torchrun 的 rdzv 参数在这里
│   └── clean.sh
└── out/                    # 产物：*.json（每次 run 一条记录）+ *.csv（comm_bench 扫描）
```

`run_8gpu_wall.sh` 是设计里没有、实现时补上的一个脚本，它把四组对照串成一条主线：
**A 通信带宽 vs 卡数 → B DDP 扩展效率 → C 张量并行会不会塌 → D ZeRO 阶梯**。
四组共用同一个模型与口径，所以 B/C/D 的缺口都能拿 A 组测出的带宽去解释——
这正是[研究问题 1](../../README.md#问题-1子图划分与跨设备传输最小化)所说的
「带通信代价的图分割」在真实硬件上的样子。

### 2.2 环境与版本自查（先做这一步）

PyTorch 的 FSDP 有 **FSDP1**（`FullyShardedDataParallel` 类包装，用 `ShardingStrategy` 枚举）与 **FSDP2**（`fully_shard` 函数式 API，用 `reshard_after_forward` 布尔量 + `DeviceMesh`）两代，**入口路径随版本变动**。本文的示例以 FSDP1 为准（枚举名与 ZeRO 分级的对应关系最直白），**具体以你本地 torch 版本为准**：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import torch.distributed.fsdp as f; print(sorted(n for n in dir(f) if 'shard' in n.lower() or 'Shard' in n))"
python -c "import torch.distributed as d; print(d.is_available(), d.is_gloo_available(), d.is_nccl_available())"
```

- 第二条能打印出 `ShardingStrategy`、`FullyShardedDataParallel` → 走本文 FSDP1 写法。
- 若同时打印出 `fully_shard` → 你的版本也支持 FSDP2；较早的版本里它在 `torch.distributed._composable.fsdp` 下。两者**不能混用**在同一模型上。
- DTensor（[L3-DIST-13](#l3-dist-13spmd-三种标注体系对照表并用-dtensor-实操验证) 会用）公开路径是 `torch.distributed.tensor`，较早版本在 `torch.distributed._tensor`。

**Windows 说明**：纯 CPU / gloo 的条目在原生 Windows + PowerShell 下可直接跑（把 `.sh` 里的 `torchrun ...` 那一行复制出来执行即可）；`nccl` 与 FSDP 相关条目请在 Linux 或 WSL2 下做。

`requirements.txt` 建议内容：

```text
torch>=2.1
psutil>=5.9        # CPU 侧 RSS 计量（可选）
matplotlib>=3.7    # 画 comm_bench 拐点图（可选）
```

### 2.3 各文件的职责与关键 API

#### `common.py` —— 所有脚本共用的四件事

职责：①初始化/销毁进程组；②统一的显存计量；③统一的吞吐计时；④统一的产物落盘格式。

```python
import os, json, time, pathlib, torch, torch.distributed as dist

def setup(backend: str | None = None):
    """torchrun 会注入 RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT。"""
    backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    dist.init_process_group(backend=backend)          # 关键 API
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if backend == "nccl":
        torch.cuda.set_device(local_rank)             # 必须，否则所有 rank 挤在 cuda:0
    return dist.get_rank(), dist.get_world_size(), local_rank

def teardown():
    dist.barrier()
    dist.destroy_process_group()

class MemProbe:
    """显存计量：allocated 看真实占用，reserved 看 caching allocator 拿了多少（差值≈碎片+缓存）。"""
    def reset(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()      # 关键 API
    def read(self):
        if not torch.cuda.is_available():
            return {"peak_alloc": None, "peak_reserved": None}
        return {"peak_alloc": torch.cuda.max_memory_allocated(),      # 关键 API
                "peak_reserved": torch.cuda.max_memory_reserved()}

def timed_steps(step_fn, n_steps: int, warmup: int = 3):
    """前 warmup 步不计时（含 CUDA context、Adam 状态惰性分配、NCCL 建链）。"""
    for _ in range(warmup):
        step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_steps

def dump(record: dict, path: str):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
```

**两个必须踩到的坑，提前写进代码里**：

- **`warmup` 不能省**：Adam 的 `exp_avg` / `exp_avg_sq` 是**第一次 `optimizer.step()` 时才分配**的。只跑 1 步测显存，会少掉 8Φ（混合精度口径下的 momentum + variance），账永远对不上。
- **`torch.cuda.synchronize()` 不能省**：CUDA kernel 异步下发，不同步的话你测的是「下发耗时」而不是「执行耗时」，吞吐会虚高十倍以上。

#### `model.py` —— 参数量可调的小模型

职责：给一个**参数量随 `hidden` 平方增长**的模型，这样改一个数字就能把显存压力做出来，且参数量有闭式解便于手算 16Φ。

```python
import torch.nn as nn

class FFNBlock(nn.Module):
    """Transformer 的 FFN 结构（占 Transformer 参数量的大头），去掉 attention 以避免
    形状与 mask 干扰显存账；需要真 attention 时把 nn.MultiheadAttention 加回来即可。"""
    def __init__(self, hidden: int, expand: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1  = nn.Linear(hidden, expand * hidden)     # 列切对象（TP 的第一个矩阵）
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(expand * hidden, hidden)     # 行切对象（TP 的第二个矩阵）
    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))

def build_model(hidden=1024, layers=8, expand=4):
    return nn.Sequential(*[FFNBlock(hidden, expand) for _ in range(layers)],
                         nn.LayerNorm(hidden))

def count_params(model):
    return sum(p.numel() for p in model.parameters())
```

**参数量闭式解**（用于和 `mem_ledger.py` 对账）：每层 ≈ `2 * expand * hidden²`（两个 Linear 的权重）`+ (expand+1) * hidden`（两个 bias）`+ 2 * hidden`（LayerNorm）。`expand=4` 时每层约 `8 * hidden²`。

**参数量档位建议**（决定你能不能观察到显存差异）：

| hidden | layers | Φ（约） | 16Φ（约） | 用途 |
|--------|--------|--------|-----------|------|
| 256 | 4 | 2.1 M | 34 MB | CPU 冒烟，秒级 |
| 1024 | 8 | 67 M | 1.1 GB | 单卡 baseline |
| 2048 | 12 | 403 M | 6.4 GB | **L2 主判据用这一档**：模型状态足够大，能盖过激活噪声 |
| 4096 | 12 | 1.6 G | 26 GB | 单卡放不下，用来演示「必须分片」 |

**关键设计点**：L2 的显存判据要成立，必须让**模型状态占主导**。所以主判据统一用大 `hidden` + 小 `batch`/`seq`（如 `--batch 4 --seq 256`），把激活占比压到 15% 以下；反过来验证激活相关结论（[L2-DIST-09](#l2-dist-09开关激活重计算量出用算力换显存的兑换率)）时再调大 `batch`/`seq`。

#### `mem_ledger.py` —— 16Φ 显存账手算器

职责：输入参数量与配置，输出各项显存与合计，**不依赖 GPU、不 import 分布式**。这是 [L0-DIST-03](#l0-dist-0316φ-显存账手算脚本) 的产物，也是后面所有实测的对照基线。

```python
def model_states(params: int, precision: str = "mixed", zero_stage: int = 0, n: int = 1):
    """返回模型状态各项字节数。zero_stage: 0=DDP, 1/2/3 = ZeRO-1/2/3, n = 分片组大小。"""
    if precision == "mixed":       # fp16/bf16 前反向 + fp32 master weight
        p, g, opt = 2 * params, 2 * params, 12 * params   # 12Φ = master 4Φ + m 4Φ + v 4Φ
    elif precision == "fp32":      # 纯 fp32 + Adam
        p, g, opt = 4 * params, 4 * params, 8 * params    # 8Φ = m 4Φ + v 4Φ
    else:
        raise ValueError(precision)
    if zero_stage >= 1: opt /= n   # ZeRO-1：切优化器状态
    if zero_stage >= 2: g   /= n   # ZeRO-2：再切梯度
    if zero_stage >= 3: p   /= n   # ZeRO-3 / FSDP FULL_SHARD：再切参数
    return {"param": p, "grad": g, "optim": opt, "total": p + g + opt}

def activations(batch, seq, hidden, layers, expand=4, bytes_per_elem=2, recompute=False):
    """粗估：不重计算时每层要留 norm 输出、fc1 输出、act 输出；重计算时只留每层输入。"""
    per_token = hidden if recompute else (2 + 2 * expand) * hidden
    return batch * seq * layers * per_token * bytes_per_elem
```

**这个脚本要能打印出的关键事实**：mixed 与 fp32 两种口径**合计都是 16Φ**，但构成完全不同（`2+2+12` vs `4+4+8`）。能说清这个巧合的成因，才算真懂 16Φ 不是一个魔数。

#### `train_single.py` —— 单卡/单进程 baseline

职责：给出对照组。参数 `--device {cpu,cuda} --hidden --layers --batch --seq --steps --precision {fp32,amp}`。

关键流程：`build_model` → `.to(device)` → `torch.optim.AdamW` → 循环 `forward / loss.backward() / opt.step() / opt.zero_grad(set_to_none=True)` → 记录 `MemProbe.read()` 与 `timed_steps` → `dump` 到 `out/single_<tag>.json`。

吞吐口径统一为两个数：`samples_per_s = batch / step_time`、`tokens_per_s = batch * seq / step_time`。**多进程时用全局值**（乘 `world_size`），否则 DDP 的「线性加速」会被口径吃掉。

#### `train_ddp.py` —— DistributedDataParallel

```python
from torch.nn.parallel import DistributedDataParallel as DDP

rank, world, local_rank = setup(args.backend)
model = build_model(args.hidden, args.layers).to(device)
ddp = DDP(model,
          device_ids=[local_rank] if args.backend == "nccl" else None,  # gloo/CPU 传 None
          bucket_cap_mb=args.bucket_mb,          # 默认 25：梯度分桶大小，L2-DIST-10 会改它
          gradient_as_bucket_view=True)          # 让梯度直接视图化到 bucket，省一份拷贝
```

必备的三个开关（后面条目会用到）：

- `--check-consistency`：跑完一步后 all_gather 各 rank 的参数并比对（[L0-DIST-02](#l0-dist-02cpugloo-两进程跑通-ddp并验证-all_reduce-后各-rank-参数一致)）。
- `--bucket-mb N`：改分桶大小（[L2-DIST-10](#l2-dist-10统计一步训练里的通信占比解释-bucketing-与-overlap)）。
- `--accum K`：用 `with ddp.no_sync():` 累积 K-1 步再同步一次，用来观察通信次数与耗时的关系。

数据侧用 `torch.utils.data.distributed.DistributedSampler`，或者直接按 rank 生成不同随机种子的合成数据（本 lab 不关心收敛，合成数据更简单，但**必须保证各 rank 数据不同**，否则 all_reduce 的效果观察不到）。

#### `train_fsdp.py` —— FullyShardedDataParallel

```python
import functools
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

STRATEGY = {"full": ShardingStrategy.FULL_SHARD,        # ≈ ZeRO-3
            "grad_op": ShardingStrategy.SHARD_GRAD_OP,  # ≈ ZeRO-2
            "none": ShardingStrategy.NO_SHARD,          # ≈ ZeRO-0 / DDP
            "hybrid": ShardingStrategy.HYBRID_SHARD}    # 1 < F < W，跨机时才有意义

model = FSDP(
    build_model(args.hidden, args.layers).to(device),
    sharding_strategy=STRATEGY[args.strategy],
    auto_wrap_policy=functools.partial(transformer_auto_wrap_policy,
                                       transformer_layer_cls={FFNBlock}),  # 以 block 为分片单元
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                   reduce_dtype=torch.float32,
                                   buffer_dtype=torch.bfloat16) if args.amp else None,
    limit_all_gathers=True,     # 限制预取深度，控制 all_gather 造成的显存尖峰
    forward_prefetch=True,      # 提前 all_gather 下一层参数，换取重叠
    use_orig_params=True,
    device_id=local_rank,
)
```

`--activation-checkpointing` 开关（[L2-DIST-09](#l2-dist-09开关激活重计算量出用算力换显存的兑换率) 用）两种写法任选：

```python
# 写法 A：直接在 FFNBlock.forward 里用（最少侵入，也适用于单卡）
from torch.utils.checkpoint import checkpoint
out = checkpoint(self.block_body, x, use_reentrant=False)

# 写法 B：FSDP 官方配套（对 wrap 单元统一施加）
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl)
apply_activation_checkpointing(
    model, checkpoint_wrapper_fn=functools.partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
    check_fn=lambda m: isinstance(m, FFNBlock))
```

> **注意**：`apply_activation_checkpointing` 位于带下划线的私有模块路径，不同版本可能移动。跑不通时先用写法 A，结论完全一样。

`--zero1` 开关（[L2-DIST-08](#l2-dist-08切换-fsdp-的-sharding_strategy-三档映射回-zero-123) 用）：FSDP 的枚举里**没有精确对应 ZeRO-1 的那一档**（`SHARD_GRAD_OP` 同时切了梯度和优化器状态 = ZeRO-2）。PyTorch 里 ZeRO-1 的对应物是 `torch.distributed.optim.ZeroRedundancyOptimizer`，配 DDP 使用：

```python
from torch.distributed.optim import ZeroRedundancyOptimizer
opt = ZeroRedundancyOptimizer(ddp.parameters(), optimizer_class=torch.optim.AdamW, lr=1e-4)
```

#### `comm_bench.py` —— 集合通信微基准

职责：扫消息大小，测 all_reduce / all_gather 的耗时与带宽，输出 CSV。

```python
sizes = [2**k for k in range(10, 29)]          # 1 KiB ~ 256 MiB（元素数，非字节）
for numel in sizes:
    buf = torch.ones(numel, dtype=torch.float32, device=device)
    for _ in range(warmup):                     # 首次调用含建链开销，必须预热
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    sync(); dist.barrier(); t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)   # 关键 API
    sync(); t = (time.perf_counter() - t0) / iters
    nbytes = numel * 4
    algbw = nbytes / t                                  # 算法带宽
    busbw = algbw * 2 * (world - 1) / world             # ring all_reduce 的总线带宽换算
```

`all_gather` 用 `dist.all_gather_into_tensor(out, inp)`（新接口，比 `dist.all_gather(list, t)` 少一次拼接）。CSV 列：`op,backend,world_size,bytes,time_s,algbw_GBps,busbw_GBps`。

**为什么 busbw 要乘 `2(N-1)/N`**：ring all_reduce = reduce-scatter + all-gather，每个 rank 实际收发 `2(N-1)/N × S` 字节。用 busbw 才能跨 N 比较，也才能和厂商标称带宽对得上。

#### `profile_step.py` —— 通信占比

```python
from torch.profiler import profile, ProfilerActivity, schedule
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             schedule=schedule(wait=1, warmup=2, active=3),
             record_shapes=True) as prof:
    for _ in range(6):
        train_step(); prof.step()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
prof.export_chrome_trace("out/trace_ddp.json")     # 用 chrome://tracing 或 Perfetto 打开
```

通信 kernel 在表里的名字：NCCL 侧是 `ncclDevKernel_AllReduce_*` / `nccl:all_reduce`，框架侧是 `c10d::allreduce_`。**没有 GPU 时的降级**：用 `time.perf_counter()` 在 `loss.backward()` 前后与 `dist.all_reduce` 前后手工打点，只能得到「串行耗时占比」，得不到重叠信息。

#### `report.py` —— 生成对比表

读 `out/*.json`，按 `tag` 汇总成 markdown 表（策略 × 峰值显存 × 吞吐 × 相对基线的比值），直接贴进 `docs/notes/`。[L2-DIST-07](#l2-dist-07fsdp-vs-ddp显存随卡数近似-1n吞吐略降) 要求的对比表由它产出。

### 2.4 产物格式（`out/`）

每次 run 落一个 JSON，字段固定，`report.py` 才能自动对比：

```json
{
  "tag": "fsdp_full_shard",
  "backend": "nccl", "world_size": 2, "rank": 0,
  "hidden": 2048, "layers": 12, "params": 402849792,
  "batch_per_rank": 4, "seq": 256, "precision": "amp",
  "peak_alloc_bytes": 7516192768, "peak_reserved_bytes": 8321499136,
  "step_time_s": 0.184, "tokens_per_s_global": 11130.4,
  "ledger_16phi_bytes": 6445596672,
  "torch": "2.4.0", "device_name": "NVIDIA A100-SXM4-40GB"
}
```

**每个 rank 都写自己的文件**（`out/<tag>_ws{N}_r{rank}.json`）。分片类实验中各 rank 的峰值可能不同，只看 rank 0 会漏掉负载不均。

### 2.5 建 lab 的推荐顺序

`common.py` → `model.py` → `train_single.py`（CPU 跑通）→ `mem_ledger.py` → `train_ddp.py`（gloo 两进程跑通）→ `comm_bench.py` → 其余按需要写。**前五个文件在无卡笔记本上就能全部完成并验证**。

---

## 3. 检验条目

### L0：复现

#### L0-DIST-01｜建起 lab 骨架，并在 CPU 上跑通单进程 baseline

- **检验什么**：这条通过 = 你真的掌握了「一次训练迭代的显存与吞吐该怎么被测量」——`reset_peak_memory_stats` / `max_memory_allocated` 的配对使用、warmup 为什么不能省、吞吐口径为什么必须统一
- **前置**：无。`pip install torch`
- **资源**：`本地`
- **预计耗时**：2h（含建目录与写 `common.py`）

**任务**：按 [§2.1](#21-目录结构) 建出 `dist-train-lab/` 的目录骨架，实现 `common.py`（`setup` 之外的部分）、`model.py`、`train_single.py` 三个文件。`train_single.py` 必须支持 `--device cpu`，跑完向 `out/single_cpu.json` 写一条符合 [§2.4](#24-产物格式out) 字段规范的记录，并在 stdout 打印参数量、每步耗时、tokens/s。

**先预测再动手**：

1. `hidden` 从 256 改到 512，参数量会变成几倍？每步耗时会变成几倍？（这两个倍数一样吗）
2. 如果把 `warmup` 设成 0，测出的每步耗时会偏大还是偏小？为什么？
3. `optimizer.zero_grad()` 与 `zero_grad(set_to_none=True)` 对显存的影响有区别吗？

**验收命令**：

```bash
cd dist-train-lab
python train_single.py --device cpu --hidden 256 --layers 4 --batch 8 --seq 128 --steps 20
python -c "import json; r=json.load(open('out/single_cpu.json')); print(r['params'], r['tokens_per_s_global'])"
```

**通过标准**：`out/single_cpu.json` 存在且含全部必需字段；打印的 `params` 与 `model.py` 的闭式解 `layers * 8 * hidden²`（`expand=4`）相对误差 < 2%；`hidden` 加倍后重跑，`params` 变为约 4 倍。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 参数量与闭式解对不上（差得远） | 没算清 Linear 的 `in×out + out`，或忘了 LayerNorm 的 `weight+bias` |
| 每步耗时抖动超过 30% | warmup 没做，或把数据构造放进了计时区间 |
| tokens/s 算出来比 samples/s 还小 | 吞吐口径写反了（`batch*seq/t` 而非 `t/(batch*seq)`） |

---

#### L0-DIST-02｜CPU/gloo 两进程跑通 DDP，并验证 all_reduce 后各 rank 参数一致

> **这条是本册最重要的入门条目。** 它证明一件事：**没有 GPU 也能把分布式的进程模型、rank/world_size、集合通信语义完整学明白。** 分布式训练里真正难的是「谁在跟谁通信、通信的是什么、通信之后不变量是什么」，这三件事和有没有卡毫无关系。等到有卡时，你要做的只是把 `gloo` 换成 `nccl`。

- **检验什么**：这条通过 = 你真的掌握了「DDP 的核心不变量是**梯度 all_reduce 取平均后各 rank 参数保持完全一致**」，以及 rank / world_size / local_rank / 进程组这套 SPMD 进程模型
- **前置**：[L0-DIST-01](#l0-dist-01建起-lab-骨架并在-cpu-上跑通单进程-baseline)
- **资源**：`本地`（纯 CPU，两个进程跑在同一台笔记本上）
- **预计耗时**：2h

**任务**：实现 `common.py` 的 `setup()/teardown()` 与 `train_ddp.py`。用 `torchrun --standalone --nproc_per_node=2` 启动两个 **CPU** 进程，backend 用 `gloo`，`DDP(model, device_ids=None)`。**两个 rank 必须喂不同的数据**（用 `torch.manual_seed(1234 + rank)` 生成合成 batch，但模型初始化种子必须相同，或依赖 DDP 构造时的参数广播）。

实现 `--check-consistency`，在训练若干步后做两项检查：

```python
# 检查一：参数一致性 —— DDP 的核心不变量
flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
bufs = [torch.empty_like(flat) for _ in range(world)]
dist.all_gather(bufs, flat)
param_diff = max((bufs[0] - b).abs().max().item() for b in bufs)

# 检查二：梯度确实被平均了 —— all_reduce 的语义
opt.zero_grad(set_to_none=True)
with ddp.no_sync():                 # 只算本地梯度，不触发 all_reduce
    loss_fn(ddp(x), y).backward()
g_local = torch.cat([p.grad.reshape(-1).clone() for p in model.parameters()])
opt.zero_grad(set_to_none=True)
loss_fn(ddp(x), y).backward()       # 这一次会触发 all_reduce
g_sync = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
locals_ = [torch.empty_like(g_local) for _ in range(world)]
dist.all_gather(locals_, g_local)
grad_diff = (g_sync - torch.stack(locals_).mean(0)).abs().max().item()
```

**先预测再动手**：

1. 两个 rank 喂了**不同**的数据，为什么训练若干步后参数还能**逐比特相同**？这个不变量是靠什么维持的？
2. `param_diff` 你预测是精确的 `0.0`，还是 `1e-7` 量级的浮点误差？（想清楚 all_reduce 在各 rank 上产生的是不是同一个数）
3. 如果**故意**让两个 rank 的模型初始化种子不同，并且跳过 DDP 的初始参数广播，`param_diff` 会是多少？训练久了会收敛还是发散？

**验收命令**：

```bash
cd dist-train-lab
torchrun --standalone --nproc_per_node=2 train_ddp.py \
  --device cpu --backend gloo --hidden 256 --layers 4 --batch 8 --seq 128 \
  --steps 20 --check-consistency
# 单进程对照（world_size=1 时 all_reduce 应退化为恒等）
torchrun --standalone --nproc_per_node=1 train_ddp.py \
  --device cpu --backend gloo --hidden 256 --layers 4 --steps 20 --check-consistency
```

**通过标准**：

- 日志出现 `world_size=2` 且两个 rank 都打印了自己的 `rank` / `local_rank`；
- `param_diff == 0.0`（严格逐比特一致；若出现 `1e-8` 量级也可接受，但要能解释来源）；
- `grad_diff < 1e-6`（fp32 口径），即 `g_sync == mean(g_local_0, g_local_1)`；
- `--nproc_per_node=1` 时 `grad_diff` 同样 `< 1e-6`（此时 all_reduce 退化为恒等）。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 卡在 `init_process_group` 不动 | 没理解**所有 rank 必须都到齐**才能完成 rendezvous；漏了 `--standalone` 或端口被占 |
| `param_diff` 明显不为 0 | 两个 rank 用了不同的初始化且没走 DDP 广播；或某些参数没参与 forward（DDP 不会同步未使用的参数，需 `find_unused_parameters`） |
| `grad_diff ≈ g_local`（差一倍） | 以为 DDP 做的是 SUM。它是 **SUM 后除以 world_size**，即平均 |
| 两 rank 喂了相同数据还沾沾自喜 | 没意识到这样 all_reduce 前后梯度不变，等于什么都没验证 |
| 程序结束时挂住 | 漏了 `destroy_process_group()`，或某个 rank 提前 `return` 导致 barrier 永远等不齐 |

---

#### L0-DIST-03｜16Φ 显存账手算脚本

- **检验什么**：这条通过 = 你真的掌握了「混合精度下模型状态 = 参数 2Φ + 梯度 2Φ + 优化器状态 12Φ = 16Φ 字节」的**构成**，而不只是记住数字 16
- **前置**：无（可与 L0-DIST-01 并行做）
- **资源**：`本地`
- **预计耗时**：0.5h

**任务**：写 `mem_ledger.py`（骨架见 [§2.3](#23-各文件的职责与关键-api)），命令行接受 `--params`（或 `--hidden --layers` 自动算）、`--precision {mixed,fp32}`、`--zero {0,1,2,3}`、`--n`，输出一张表：参数 / 梯度 / 优化器状态 / 合计，单位同时给 bytes 与 GiB。再加 `--activations` 输出激活粗估。

**先预测再动手**：

1. 12Φ 里的三个 4Φ 分别是什么？为什么 fp32 master weight 必须留一份，不能直接用 fp16 参数更新？
2. 纯 fp32 训练的合计也是 16Φ。这是巧合还是有必然性？两种口径下**哪一项**差最多？
3. 7B 模型用 ZeRO-3 切到 8 卡，每卡模型状态是多少 GiB？还能塞进 24GB 的卡吗（别忘了激活）？

**验收命令**：

```bash
cd dist-train-lab
python mem_ledger.py --hidden 2048 --layers 12 --precision mixed --zero 0
python mem_ledger.py --params 7000000000 --precision mixed --zero 3 --n 8
python mem_ledger.py --hidden 2048 --layers 12 --precision fp32 --zero 0
```

**通过标准**：

- 第一条输出 `total ≈ 16 × params` 字节（相对误差 < 0.1%）；
- 第二条输出 `total ≈ 16 × 7e9 / 8 = 14 GiB`（约 13.04 GiB，按 GiB 换算），且三项各自都被 8 除；
- 第三条 `total` 同样 ≈ 16Φ，但 `param`/`grad` 是 4Φ、`optim` 是 8Φ；
- `--zero 1` 时只有 `optim` 被 N 除，`--zero 2` 时 `optim` 与 `grad` 被除，`--zero 3` 时三项全除——这个阶梯必须能从输出里一眼看出来。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 把 12Φ 写成 8Φ | 忘了 fp32 master weight 这一份（只记了 Adam 的 m 和 v） |
| ZeRO-1 把梯度也除了 | 没记住分级：1 切优化器状态、2 加梯度、3 加参数 |
| 认为 16Φ 就是训练总显存 | 忘了激活、临时缓冲、碎片。16Φ 只是**模型状态**，是显存的**下界** |
| GB / GiB 混用导致对不上 | 1 GiB = 2³⁰，1 GB = 10⁹。做实测对账时这 7% 的差会让你怀疑人生 |

---

### L1：改一处

#### L1-DIST-04｜comm_bench 在 CPU/gloo 上扫消息大小，找延迟/带宽拐点

- **检验什么**：这条通过 = 你真的掌握了「集合通信的耗时 = 固定延迟 + 消息大小/带宽」这个双段模型，以及它为什么直接决定了 **DDP 要做梯度分桶（bucketing）**
- **前置**：[L0-DIST-02](#l0-dist-02cpugloo-两进程跑通-ddp并验证-all_reduce-后各-rank-参数一致)
- **资源**：`本地`（CPU + gloo，两进程）
- **预计耗时**：2h

**任务**：实现 `comm_bench.py`（骨架见 [§2.3](#23-各文件的职责与关键-api)），扫 `2^10 ~ 2^26` 个 float32 元素（4 KiB ~ 256 MiB），对 `all_reduce` 与 `all_gather_into_tensor` 各测一遍，输出 `out/comm_gloo_ws2.csv`。再写十行代码在 log-log 坐标上做两段线性拟合，把**拐点**（斜率从约 0 变成约 1 的位置）打印出来。

**先预测再动手**：

1. 消息从 4 KiB 涨到 64 KiB（16 倍），耗时会涨 16 倍吗？从 16 MiB 涨到 256 MiB 呢？
2. 拐点大概会落在哪个量级？拐点左边耗时由什么主导，右边由什么主导？
3. DDP 默认 `bucket_cap_mb=25`。25 MiB 落在你测出的拐点左边还是右边？这个默认值是想避免什么？

**验收命令**：

```bash
cd dist-train-lab
torchrun --standalone --nproc_per_node=2 comm_bench.py \
  --device cpu --backend gloo --ops all_reduce,all_gather --out out/comm_gloo_ws2.csv
python -c "
import csv; rows=[r for r in csv.DictReader(open('out/comm_gloo_ws2.csv')) if r['op']=='all_reduce']
small=[r for r in rows if int(r['bytes'])<=2**16]; large=[r for r in rows if int(r['bytes'])>=2**22]
f=lambda rs: (float(rs[-1]['time_s'])/float(rs[0]['time_s']), int(rs[-1]['bytes'])/int(rs[0]['bytes']))
print('small seg time_ratio/size_ratio =', f(small)); print('large seg =', f(large))"
```

**通过标准**：

- CSV 至少 15 行有效数据，`busbw_GBps` 单调上升后趋于平台；
- **小消息段**（≤ 64 KiB）：字节数涨 16 倍，耗时涨幅 < 3 倍（log-log 斜率 < 0.3）；
- **大消息段**（≥ 4 MiB）：字节数涨 N 倍，耗时涨幅在 0.7N ~ 1.3N 之间（斜率 ∈ [0.8, 1.2]）；
- 能指出一个具体的拐点字节数，并说明在此之上 `busbw` 已达平台值的 80% 以上。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 曲线全程是直线（无拐点） | 扫描范围太窄，或没预热（首次调用的建链开销盖过了一切） |
| 小消息测出的耗时是纳秒级 | 忘了 `dist.barrier()` 对齐起点，或迭代次数太少被计时精度吃掉 |
| 认为拐点位置是硬件常数 | 拐点由 backend + 拓扑 + 消息数共同决定。gloo/CPU 与 nccl/NVLink 的拐点差一两个数量级 |
| 说不出这和 bucketing 的关系 | 没建立因果：反向会产生**大量小梯度张量**，逐个 all_reduce 全落在延迟主导区，所以要攒成桶推到带宽主导区 |

---

#### L1-DIST-05｜单卡显存峰值实测，并与 16Φ 手算账对差额

- **检验什么**：这条通过 = 你真的掌握了「实测显存 = 模型状态 + 激活 + 临时缓冲 + 碎片」这个四项构成，且能定量地把差额归因
- **前置**：[L0-DIST-01](#l0-dist-01建起-lab-骨架并在-cpu-上跑通单进程-baseline)、[L0-DIST-03](#l0-dist-0316φ-显存账手算脚本)
- **资源**：`单卡GPU`
- **预计耗时**：2h

**任务**：在单卡上用三档配置跑 `train_single.py --device cuda`，每档都记录 `max_memory_allocated` 与 `max_memory_reserved`，与 `mem_ledger.py` 的手算值列成一张三列表（手算模型状态 / 实测 allocated / 差额），并把差额拆成「激活估算 + 其余」。三档：`(hidden=1024, layers=8)`、`(2048, 12)`、以及固定模型只改 `--batch` 从 4 到 16。

**先预测再动手**：

1. 实测 allocated 会比 16Φ 大还是小？大多少（预测一个百分比）？
2. `--batch` 从 4 涨到 16，多出来的显存**应该只来自哪一项**？涨幅是线性的吗？
3. `max_memory_reserved - max_memory_allocated` 这个差值代表什么？它随训练步数增加是变大还是稳定？

**验收命令**：

```bash
cd dist-train-lab
for cfg in "1024 8" "2048 12"; do set -- $cfg; \
  python train_single.py --device cuda --hidden $1 --layers $2 --batch 4 --seq 256 --steps 20 \
    --tag single_h$1_l$2; done
for b in 4 8 16; do \
  python train_single.py --device cuda --hidden 2048 --layers 12 --batch $b --seq 256 --steps 20 \
    --tag single_b$b; done
python report.py --glob 'out/single_*.json' --cols peak_alloc_bytes,peak_reserved_bytes,ledger_16phi_bytes
```

**通过标准**：

- 每一档都满足 `peak_alloc > ledger_16phi`（模型状态是**下界**，实测必须更大）；
- 差额 `peak_alloc - 16Φ` 落在 `mem_ledger.py` 激活估算的 0.5x ~ 2.0x 区间内（粗估允许这个精度）；
- `--batch` 三档中，`peak_alloc - 16Φ` 随 batch **近似线性增长**（4→16 时增长 3±1 倍），而 16Φ 项完全不变；
- `peak_reserved ≥ peak_alloc`，且能说出差值的来源。

**降级方案（无 GPU 时）**：

用 `--device cpu` + `psutil.Process().memory_info().rss` 采样代替，或纯用 `mem_ledger.py` 做**纸面对账**：手算三档的 16Φ 与激活估算，画出「显存 vs batch」的预测曲线。
**降级后拿不到的**：真实峰值数字、`reserved - allocated` 的碎片量、cudnn/cuBLAS workspace 的额外占用，以及 caching allocator 的行为。也就是说，**「差额归因」这条主线在降级下只能是理论推演，无法证实**。等拿到卡时这条必须重跑。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 实测比 16Φ **小** | 只跑了 1 步，Adam 的 `exp_avg`/`exp_avg_sq` 还没分配（它们在首次 `step()` 时才建） |
| 差额远大于激活估算 | 没做 `reset_peak_memory_stats()`，把上一次 run 的峰值带进来了；或 AMP 下同时存在 fp16 与 fp32 两份副本没算 |
| batch 翻倍显存纹丝不动 | 模型状态占绝对主导，激活项被淹没——说明档位选错了，要么调大 batch/seq，要么调小 hidden |
| 用 `nvidia-smi` 的数字对账 | `nvidia-smi` 看到的是进程级显存（含 CUDA context 约几百 MB + caching allocator 缓存），**不能**与 `max_memory_allocated` 直接比 |

---

### L2：加组件（主判据）

#### L2-DIST-06｜DDP vs 单卡：显存几乎不降，吞吐接近线性

- **检验什么**：这条通过 = 你真的掌握了「**vanilla DP（分片因子 F=1）省的是时间，不是显存**」——每卡仍持有完整的 16Φ 模型状态
- **前置**：[L0-DIST-02](#l0-dist-02cpugloo-两进程跑通-ddp并验证-all_reduce-后各-rank-参数一致)、[L1-DIST-05](#l1-dist-05单卡显存峰值实测并与-16φ-手算账对差额)
- **资源**：`多卡GPU`（同机 2 卡）
- **预计耗时**：半天

**任务**：用 `(hidden=2048, layers=12)`（Φ≈4.0e8，16Φ≈6.4 GB）跑两组对照，两组都要做：

- **A 组（固定 per-rank micro-batch）**：单卡 `--batch 4` vs 2 卡各 `--batch 4`（global batch 翻倍）。这组用来判**显存**。
- **B 组（固定 global batch）**：单卡 `--batch 8` vs 2 卡各 `--batch 4`。这组用来判**吞吐**。

用 `--seq 256` 把激活占比压到 15% 以下（先用 `mem_ledger.py --activations` 确认）。

**先预测再动手**：

1. A 组里 2 卡的**每卡**峰值显存，相对单卡是降一半、基本不变、还是略升？略升的话多出来的是什么？
2. B 组里 2 卡的总吞吐能到单卡的几倍？拿不到 2.0 倍的原因有几个，分别是什么？
3. 如果把模型换成 `hidden=256`（Φ 很小），DDP 的加速比会变好还是变差？为什么？

**验收命令**：

```bash
cd dist-train-lab
# A 组：固定 per-rank batch，看显存
python train_single.py --device cuda --hidden 2048 --layers 12 --batch 4 --seq 256 --steps 30 --tag A_single_b4
torchrun --standalone --nproc_per_node=2 train_ddp.py --device cuda --backend nccl \
  --hidden 2048 --layers 12 --batch 4 --seq 256 --steps 30 --tag A_ddp_b4
# B 组：固定 global batch，看吞吐
python train_single.py --device cuda --hidden 2048 --layers 12 --batch 8 --seq 256 --steps 30 --tag B_single_b8
torchrun --standalone --nproc_per_node=2 train_ddp.py --device cuda --backend nccl \
  --hidden 2048 --layers 12 --batch 4 --seq 256 --steps 30 --tag B_ddp_b4x2
python report.py --glob 'out/A_*.json,out/B_*.json'
```

**通过标准**（机器可判定）：

| 判据 | 数值条件 |
|------|---------|
| A 组显存不降 | `peak_alloc(A_ddp, 每个 rank) ∈ [0.9, 1.15] × peak_alloc(A_single)`，即 **±10%~15% 以内** |
| 模型状态确实全量复制 | `peak_alloc(A_ddp) > 16Φ`（即 > 6.4 GB），而不是 8Φ |
| B 组吞吐接近线性 | `tokens_per_s_global(B_ddp) ≥ 1.6 × tokens_per_s_global(B_single)`（≥80% 扩展效率） |
| 数值仍然正确 | `--check-consistency` 输出 `param_diff == 0` |

若 A 组出现峰值**上升** 5%~15%，是正常的：DDP 的梯度 bucket 是一块额外的连续缓冲区（约等于梯度总量的一个桶）。能解释这一点比数字本身更重要。

**降级方案（只有单卡或无卡）**：

用 `--device cpu --backend gloo --nproc_per_node=2` 跑同样两组，验证**流程与数值**正确（`param_diff == 0`、loss 下降）；显存结论用 `mem_ledger.py --zero 0 --n 2` 做纸面推演（输出应显示三项都没被 N 除）。
**降级后拿不到的**：显存的 ±10% 判据（CPU 上 `max_memory_allocated` 直接不可用）、任何加速比数字（gloo 在同机多进程上受 CPU 核数与内存带宽制约，扩展效率不具参考性）。**A、B 两组的核心结论都必须真机复测。**

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 2 卡显存降了将近一半 | 用的是 B 组配置（固定 global batch，per-rank batch 减半），降的是**激活**不是模型状态。混淆了这两组的用途 |
| 加速比只有 1.1x | 模型太小，通信占比过高；或没设 `torch.cuda.set_device(local_rank)` 导致两进程挤在 cuda:0 |
| 加速比 > 2.0x | 单卡 baseline 的 batch 没对齐（拿 A 组单卡去比 B 组 DDP），口径错了 |
| 两个 rank 显存差异很大 | 数据没均分，或只在 rank 0 上做了额外的日志/checkpoint 操作 |

---

#### L2-DIST-07｜FSDP vs DDP：显存随卡数近似 1/N，吞吐略降

- **检验什么**：这条通过 = 你真的掌握了「**全分片（F=W，ZeRO-3/FSDP）把 16Φ 变成 16Φ/N，代价是通信量约 1.5 倍**」这笔交易
- **前置**：[L2-DIST-06](#l2-dist-06ddp-vs-单卡显存几乎不降吞吐接近线性)
- **资源**：`多卡GPU`（同机 2 卡；有 4 卡更好，能验证 1/N 的趋势）
- **预计耗时**：半天

**任务**：实现 `train_fsdp.py`（骨架见 [§2.3](#23-各文件的职责与关键-api)），用与 [L2-DIST-06](#l2-dist-06ddp-vs-单卡显存几乎不降吞吐接近线性) A 组**完全相同**的模型与 per-rank batch，跑 `FULL_SHARD`。产出 `report.py` 生成的对比表：

| 策略 | world_size | 每卡峰值显存 | 全局吞吐 | 相对 DDP 显存 | 相对 DDP 吞吐 |
|------|-----------|-------------|---------|--------------|--------------|
| single | 1 | | | | |
| DDP | 2 | | | | |
| FSDP FULL_SHARD | 2 | | | | |
| （有 4 卡则再加）FSDP FULL_SHARD | 4 | | | | |

**先预测再动手**：

1. **FSDP 开 FULL_SHARD 后，2 卡显存是单卡的 1/2 还是更少？为什么不是精确 1/2？** 至少说出两个让它偏离 1/2 的因素。
2. 吞吐相对 DDP 会降多少？降的原因是「通信总量变大」还是「通信次数变多」，还是两者都有？
3. 如果 `auto_wrap_policy` 传 `None`（整个模型作为一个分片单元），显存会更省还是更费？为什么？

**验收命令**：

```bash
cd dist-train-lab
torchrun --standalone --nproc_per_node=2 train_fsdp.py --device cuda --backend nccl \
  --strategy full --hidden 2048 --layers 12 --batch 4 --seq 256 --steps 30 --tag fsdp_full_ws2
# 有 4 卡再加这一条
torchrun --standalone --nproc_per_node=4 train_fsdp.py --device cuda --backend nccl \
  --strategy full --hidden 2048 --layers 12 --batch 4 --seq 256 --steps 30 --tag fsdp_full_ws4
python report.py --glob 'out/A_single_b4.json,out/A_ddp_b4*.json,out/fsdp_full_ws*.json' --markdown out/table_fsdp.md
```

**通过标准**（机器可判定）：

| 判据 | 数值条件 |
|------|---------|
| 显存显著下降 | `peak_alloc(FSDP,ws=2) < peak_alloc(DDP,ws=2) - 0.35 × 16Φ`（16Φ≈6.4 GB 时，至少省 2.2 GB） |
| 逼近但不等于 1/N | `peak_alloc(FSDP,ws=2) ∈ [0.5, 0.8] × peak_alloc(DDP,ws=2)`；**低于 0.5 说明你算错了**（激活没被分片） |
| 4 卡继续下降但收敛 | `peak_alloc(ws=4) < peak_alloc(ws=2)`，且 `peak_alloc(ws=4) − peak_alloc(ws=2)` 的绝对值小于 `peak_alloc(ws=2) − peak_alloc(ws=1)`（边际收益递减） |
| 吞吐略降 | `tput(FSDP)/tput(DDP) ∈ [0.6, 1.0]`。若 > 1.0，多半是 DDP 被显存逼到了更小的有效 batch，需检查配置对齐 |
| 产出对比表 | `out/table_fsdp.md` 存在且四列齐全 |

**降级方案（只有单卡）**：

单卡上用 `--nproc_per_node=1` 跑 FSDP 只能验证 API 用法（此时不产生分片）。更有价值的降级是**用 `mem_ledger.py` 做完整推演**：分别输出 `--zero 0 --n 2` 与 `--zero 3 --n 2`，得到 16Φ vs 8Φ，再加上**不随分片变化的激活项**，算出理论比值（应落在 0.5~0.8，正好对上判据）。
**降级后拿不到的**：`all_gather` 临时缓冲造成的显存尖峰（这正是「为什么不是精确 1/2」的主因之一）、真实吞吐损失、`limit_all_gathers` / `forward_prefetch` 的效果。**这条的对比表必须真机产出。**

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| FSDP 显存几乎没降 | `auto_wrap_policy` 没生效，整个模型成了一个分片单元——前向时要 all_gather 全部参数，等于没分片。这是最典型的 FSDP 配置错误 |
| 显存降到远低于 1/2 | 忘了激活**不随 FULL_SHARD 分片**（要分片激活得靠 SP 或 `activation_checkpointing`），说明你把激活也算进被除项了 |
| 吞吐比 DDP 还高 | 两组的 batch / 精度 / steps 没对齐；或 DDP 那组开了 AMP 而 FSDP 没开 |
| 报 `use_orig_params` 相关错误 | optimizer 在 FSDP 包装**之前**构造了。必须先 wrap 再建 optimizer（参数对象已被替换） |

---

#### L2-DIST-08｜切换 FSDP 的 `sharding_strategy` 三档，映射回 ZeRO-1/2/3

- **检验什么**：这条通过 = 你真的掌握了「**ZeRO 三级各自切的是哪一项**」——不是背表，而是能从显存数字反推出切了谁
- **前置**：[L2-DIST-07](#l2-dist-07fsdp-vs-ddp显存随卡数近似-1n吞吐略降)、[L0-DIST-03](#l0-dist-0316φ-显存账手算脚本)
- **资源**：`多卡GPU`（同机 2 卡）
- **预计耗时**：2h

**任务**：给 `train_fsdp.py` 的 `--strategy` 跑满三档 `none / grad_op / full`（对应 `NO_SHARD` / `SHARD_GRAD_OP` / `FULL_SHARD`），模型与 batch 与上一条完全一致。再用 `ZeroRedundancyOptimizer` + DDP 补上**真正的 ZeRO-1**（FSDP 枚举里没有这一档）。产出四行对比表，并在每行后面写上**手算预期值**：

| 配置 | 对应 ZeRO 级 | 模型状态手算（N=2，混合精度口径） | 实测每卡峰值 |
|------|-------------|--------------------------------|-------------|
| DDP / `NO_SHARD` | ZeRO-0 | 16Φ | |
| DDP + `ZeroRedundancyOptimizer` | ZeRO-1 | 4Φ + 12Φ/2 = 10Φ | |
| FSDP `SHARD_GRAD_OP` | ZeRO-2 | 2Φ + 14Φ/2 = 9Φ | |
| FSDP `FULL_SHARD` | ZeRO-3 | 16Φ/2 = 8Φ | |

**先预测再动手**：

1. 上表里 ZeRO-1 → ZeRO-2 只差 1Φ，ZeRO-2 → ZeRO-3 也只差 1Φ，而 ZeRO-0 → ZeRO-1 差了 6Φ。为什么**第一级的收益最大**？这对「只有 2 卡时该选哪一级」有什么启示？
2. `SHARD_GRAD_OP` 的英文直译是「切梯度与优化器状态」。参数在前向后**保留还是释放**？这决定了它比 `FULL_SHARD` 多占多少、又快多少。
3. 把 N 从 2 改成 8，四档的差距会拉大还是缩小？极限情况（N→∞）四档分别趋近于多少 Φ？

**验收命令**：

```bash
cd dist-train-lab
for s in none grad_op full; do \
  torchrun --standalone --nproc_per_node=2 train_fsdp.py --device cuda --backend nccl \
    --strategy $s --hidden 2048 --layers 12 --batch 4 --seq 256 --steps 30 --tag zero_$s; done
torchrun --standalone --nproc_per_node=2 train_ddp.py --device cuda --backend nccl --zero1 \
  --hidden 2048 --layers 12 --batch 4 --seq 256 --steps 30 --tag zero_1_zro
python mem_ledger.py --hidden 2048 --layers 12 --precision mixed --zero 1 --n 2
python mem_ledger.py --hidden 2048 --layers 12 --precision mixed --zero 2 --n 2
python mem_ledger.py --hidden 2048 --layers 12 --precision mixed --zero 3 --n 2
python report.py --glob 'out/zero_*.json' --markdown out/table_zero.md
```

**通过标准**（机器可判定）：

- 严格单调：`peak(none) > peak(zero1) > peak(grad_op) ≥ peak(full)`；
- `peak(none)` 与 `peak(DDP)`（上一条 A 组）相差 < 10%——**`NO_SHARD` 就是 DDP**；
- 相邻档的**差值**与手算的 Φ 倍数同号且量级吻合：`peak(none) − peak(zero1) ≈ 6Φ`（±30%），`peak(grad_op) − peak(full) ≈ 1Φ`（±60%，这一档差值小，容易被噪声淹没）；
- 若 `grad_op` 与 `full` 的差异落在噪声里（< 2%），把 `hidden` 提到 4096 重跑一次，差值应随 Φ 线性放大——**能通过放大 Φ 让差值浮出噪声，本身就是过关的证据**。

**降级方案（只有单卡或无卡）**：

纯用 `mem_ledger.py` 跑满 `--zero 0/1/2/3 × --n 2/4/8`，画出「ZeRO 级别 × 卡数 → 每卡模型状态」的表格，验证三条阶梯的分母关系。API 层面可在 CPU/gloo 上跑 `--strategy` 三档确认不报错（FSDP 对 gloo 的支持随版本而异，报错就只做 `ZeroRedundancyOptimizer` 那一档）。
**降级后拿不到的**：`SHARD_GRAD_OP` 与 `FULL_SHARD` 之间那 1Φ 的实测差（这恰恰是最能体现「你分得清 ZeRO-2 和 ZeRO-3」的一档），以及三档的吞吐排序。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 找不到「ZeRO-1」对应的 `ShardingStrategy` | 没意识到 FSDP 的枚举与 ZeRO 分级**不是一一对应**：`SHARD_GRAD_OP` 已经是 ZeRO-2，ZeRO-1 要用 `ZeroRedundancyOptimizer` |
| `grad_op` 比 `full` 还省 | 实验没对齐（steps/batch/AMP 不同），或峰值取自不同 rank |
| 三档差异全在 1% 以内 | Φ 太小，模型状态被激活淹没。放大 `hidden` 而不是放大 `batch` |
| 以为 ZeRO-3 通信量和 ZeRO-1 一样 | 忘了 ZeRO-3 要在前向、反向各 all_gather 一次参数，通信量约为 DDP 的 1.5 倍 |

---

#### L2-DIST-09｜开关激活重计算，量出「用算力换显存」的兑换率

- **检验什么**：这条通过 = 你真的掌握了「重计算把**激活**显存换成额外前向计算」，且知道它换的**不是模型状态**——它与 ZeRO 是正交的两把刀
- **前置**：[L2-DIST-07](#l2-dist-07fsdp-vs-ddp显存随卡数近似-1n吞吐略降)
- **资源**：`多卡GPU`（单卡也能做出结论，2 卡是为了和 FSDP 叠加观察）
- **预计耗时**：2h

**任务**：给 `train_fsdp.py` / `train_single.py` 加 `--activation-checkpointing` 开关（写法见 [§2.3](#23-各文件的职责与关键-api)）。这次要**反过来配置**：用**小 hidden + 大 batch/seq**（如 `--hidden 1024 --layers 12 --batch 16 --seq 1024`）让激活占主导，否则看不出效果。开关各跑一次，记录显存与吞吐。

**先预测再动手**：

1. 开重计算后，显存降的是四项构成里的哪一项？16Φ 那一项会变吗？
2. 吞吐预计降多少？（提示：全量重计算 ≈ 多做一次前向，而前向约占前反向总量的 1/3）
3. 如果此时模型状态占 90%、激活只占 10%，重计算还值得开吗？这说明「该不该开重计算」取决于什么？

**验收命令**：

```bash
cd dist-train-lab
for ac in 0 1; do \
  torchrun --standalone --nproc_per_node=2 train_fsdp.py --device cuda --backend nccl \
    --strategy full --hidden 1024 --layers 12 --batch 16 --seq 1024 --steps 20 \
    --activation-checkpointing $ac --tag ac$ac; done
python mem_ledger.py --hidden 1024 --layers 12 --activations --batch 16 --seq 1024
python mem_ledger.py --hidden 1024 --layers 12 --activations --batch 16 --seq 1024 --recompute
python report.py --glob 'out/ac*.json'
```

**通过标准**（机器可判定）：

- 显存下降：`peak_alloc(ac1) < peak_alloc(ac0)`，且下降量 ≥ `mem_ledger --activations` 估算激活量的 **60%**；
- 模型状态未变：`peak_alloc(ac1) > 16Φ/N`（即重计算没有动模型状态那一项）；
- 吞吐代价在合理区间：`tput(ac1)/tput(ac0) ∈ [0.6, 0.95]`。**若比值 > 0.98，说明重计算根本没生效**（`use_reentrant`、`check_fn` 没匹配上，或包装位置不对）；
- 能用一句话给出兑换率：省了 X GB，慢了 Y%，X/Y 是多少 GB/%。

**降级方案（只有单卡 / 无卡）**：

单卡上做完全相同的开关对比，结论一致（重计算与并行策略正交，不需要多卡）。无卡时用 `mem_ledger.py --activations` 的 `recompute=True/False` 两次输出算出理论省下的比例（应约为 `1 - 1/(2+2*expand)` ≈ 90%），吞吐代价只能引用论文的「全量重计算约 30% 开销」这个数。
**降级后拿不到的**：真实的兑换率数字（它强依赖于 kernel 的算力/带宽比与 recompute 的粒度），以及「重计算与 all_gather 抢带宽」这类叠加效应。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 显存一点没降 | `check_fn` 没匹配到任何模块；或用了写法 A 但把 `checkpoint` 套在了没有中间激活的地方 |
| 吞吐也没降 | 同上——**显存和吞吐同时不变 = 重计算根本没跑** |
| 报 `use_reentrant` 警告后行为异常 | 新版 PyTorch 要求显式指定；`use_reentrant=False` 是推荐值，reentrant 版对 RNG 与 grad 的处理有额外坑 |
| 认为重计算能替代 ZeRO | 混淆了显存四项构成：重计算动的是激活，ZeRO 动的是模型状态。大模型上**两个都要开** |

---

#### L2-DIST-10｜统计一步训练里的通信占比，解释 bucketing 与 overlap

- **检验什么**：这条通过 = 你真的掌握了「DDP 为什么要做梯度分桶与通信-计算重叠」——把 [L1-DIST-04](#l1-dist-04comm_bench-在-cpugloo-上扫消息大小找延迟带宽拐点) 的延迟/带宽模型接到真实训练步上
- **前置**：[L1-DIST-04](#l1-dist-04comm_bench-在-cpugloo-上扫消息大小找延迟带宽拐点)、[L2-DIST-06](#l2-dist-06ddp-vs-单卡显存几乎不降吞吐接近线性)
- **资源**：`本地`（gloo 降级版可做全部三组对照）或 `多卡GPU`（完整版，能看到重叠）
- **预计耗时**：半天

**任务**：实现 `profile_step.py`。做三组对照，每组给出「一步耗时」与「通信相关耗时占比」：

1. `--bucket-mb 1`（大量小消息）
2. `--bucket-mb 25`（PyTorch 默认）
3. `--bucket-mb 200`（几乎不分桶）

再加一组 `--accum 4`（用 `ddp.no_sync()` 累积 4 步只同步 1 次），观察通信**次数**与总耗时的关系。GPU 版用 `torch.profiler` 读 `ncclDevKernel_AllReduce_*` 的 `cuda_time_total` 与 `count`；CPU 版用手工打点，只统计串行耗时。

**先预测再动手**：

1. `bucket-mb` 从 25 降到 1，通信总字节数**变了吗**？总耗时呢？用 [L1-DIST-04](#l1-dist-04comm_bench-在-cpugloo-上扫消息大小找延迟带宽拐点) 的拐点解释这个差异。
2. `bucket-mb` 从 25 升到 200，是不是越大越好？（想想：最后一个桶要等到**反向全部结束**才能开始通信，这会破坏什么）
3. `--accum 4` 后，每步平均通信耗时应该降到约 1/4。总吞吐会提升 4 倍吗？为什么不会？

**验收命令**：

```bash
cd dist-train-lab
# 完整版（多卡 GPU）
for b in 1 25 200; do \
  torchrun --standalone --nproc_per_node=2 profile_step.py --device cuda --backend nccl \
    --hidden 2048 --layers 12 --batch 4 --seq 256 --bucket-mb $b --tag bk$b; done
torchrun --standalone --nproc_per_node=2 profile_step.py --device cuda --backend nccl \
  --hidden 2048 --layers 12 --batch 4 --seq 256 --accum 4 --tag accum4
# 降级版（本地 CPU/gloo，只看串行耗时）
for b in 1 25 200; do \
  torchrun --standalone --nproc_per_node=2 profile_step.py --device cpu --backend gloo \
    --hidden 512 --layers 8 --batch 8 --seq 128 --bucket-mb $b --tag cpu_bk$b; done
python report.py --glob 'out/bk*.json,out/accum4.json,out/cpu_bk*.json' --cols step_time_s,comm_time_s,comm_calls
```

**通过标准**（机器可判定）：

- **通信次数**随 bucket 增大单调下降：`comm_calls(bk1) > comm_calls(bk25) > comm_calls(bk200)`；
- **小桶更慢**：`step_time(bk1) ≥ 1.05 × step_time(bk25)`（至少慢 5%），且能指出慢的部分是延迟项；
- **大桶不一定更快**：给出 `step_time(bk200)` 与 `bk25` 的比较结果并解释（无论谁快，都要用「最后一个桶无法与反向重叠」来分析）；
- `--accum 4`：`comm_calls` 降到约 1/4，`step_time` 下降但 **< 4 倍**的通信节省幅度；
- 完整版还需给出一个具体的通信占比数字（`comm_time / step_time`），并在 trace 里指出 all_reduce kernel 与反向 kernel 的时间轴**是否有重叠区间**。

**降级方案（无 GPU）**：

上面命令块里的「降级版」三条即为降级方案，可完整跑出通信**次数**与**串行耗时**的关系，足以验证 bucketing 的动机。
**降级后拿不到的**：**overlap 本身**。gloo 在 CPU 上没有独立的通信 stream，通信与计算基本串行，因此「反向计算与 all_reduce 重叠」这个 DDP 最核心的性能机制在降级版里根本观察不到，占比数字也没有参考价值。**「解释 overlap」这半条必须真机。**

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| profiler 表里找不到 nccl kernel | 没加 `ProfilerActivity.CUDA`；或 `schedule` 的 `active` 窗口落在了没有通信的步上 |
| 三档 bucket 耗时完全一样 | 模型参数太少，只够装一个桶。加大 `hidden`/`layers` 让梯度总量远超 25 MB |
| 认为 bucket 越大越好 | 没理解重叠的前提：桶要在**反向途中**就填满才能提前发。全塞一个桶 = 完全串行 |
| `--accum` 后 loss 曲线变了 | 累积时没对 loss 做 `/accum` 缩放——这是梯度累积的经典错误 |

---

### L3：打通

#### L3-DIST-11｜手写最小张量并行（列切 + all_gather）

> **这条是把「读论文读来的切分标注」落成代码的关键一步。** Mesh-TensorFlow 的 mesh 维、GSPMD 的 sharding annotation、OneFlow 的 SBP，讲的都是同一件事：**一个张量沿哪个维度切、切到哪些设备上、以及切完之后要补哪个 collective 才能恢复语义等价**。这条做完，[L3-DIST-13](#l3-dist-13spmd-三种标注体系对照表并用-dtensor-实操验证) 的标注表就不再是纸面名词。

- **检验什么**：这条通过 = 你真的掌握了「Megatron 1-D TP 的列切/行切」以及「**分片 + 对应 collective = 语义等价**」这个 SPMD 的核心等式
- **前置**：[L0-DIST-02](#l0-dist-02cpugloo-两进程跑通-ddp并验证-all_reduce-后各-rank-参数一致)
- **资源**：`多卡GPU`（数值等价部分在 `本地` CPU/gloo 上即可完成）
- **预计耗时**：半天

**任务**：写 `tp_min.py`，对同一个 `nn.Linear(hidden, 4*hidden)` 实现两种切法，并各自验证与单卡结果数值一致：

```python
# 列切（column parallel）：W [in, out] 按 out 维切 → 各 rank 得到 y 的一部分 → all_gather 拼回
w_shard = w.chunk(world, dim=1)[rank].contiguous()      # 标注：Split(1)
y_local = x @ w_shard                                    # x 是 Broadcast（各 rank 相同）
out = [torch.empty_like(y_local) for _ in range(world)]
dist.all_gather(out, y_local)
y_tp = torch.cat(out, dim=-1)                            # 恢复语义等价

# 行切（row parallel）：W 按 in 维切 → x 也要按最后一维切 → 各 rank 得到部分和 → all_reduce
w_shard = w.chunk(world, dim=0)[rank].contiguous()      # 标注：Split(0)
x_shard = x.chunk(world, dim=-1)[rank].contiguous()
y_partial = x_shard @ w_shard                            # 标注：Partial（部分和，不是最终值）
dist.all_reduce(y_partial, op=dist.ReduceOp.SUM)         # Partial → Broadcast
```

再把两者串起来复现 Megatron 的 FFN：**`fc1` 列切 → GELU（逐元素，无需通信）→ `fc2` 行切 → 一次 all_reduce**，验证整块 FFN 与单卡等价，并数出**整个 block 只需要一次 collective**。

**先预测再动手**：

1. 列切之后为什么用 `all_gather`、行切之后为什么用 `all_reduce`？把这两个选择和「切的是哪个维度、切完得到的是分片还是部分和」对应起来。
2. Megatron 的 FFN 为什么是「先列切再行切」而不是「先行切再列切」？后者中间要插几次通信？
3. 数值上 `y_tp` 与单卡 `y_ref` 会**逐比特相同**吗？（提示：矩阵乘的归约顺序变了）你打算用多大的 `atol`？

**验收命令**：

```bash
cd dist-train-lab
# 数值等价（无卡也能做，这是本条的主判据）
torchrun --standalone --nproc_per_node=2 tp_min.py --device cpu --backend gloo \
  --hidden 512 --check col,row,ffn
# 有卡时再跑一遍，并观察通信次数
torchrun --standalone --nproc_per_node=2 tp_min.py --device cuda --backend nccl \
  --hidden 4096 --check col,row,ffn --count-collectives
```

**通过标准**：

- 三项检查都输出 `max_abs_diff` 且满足 `< 1e-4`（fp32，`torch.allclose(atol=1e-4, rtol=1e-3)` 为真）；
- 脚本打印出每种切法用的 collective 名称与调用次数，且 **FFN 整块只有 1 次 all_reduce**；
- 能对每个中间张量写出它的分布式状态（`Split(dim)` / `Broadcast` / `Partial`）——把这三个词标在代码注释里就是过关证据。

**降级方案**：本条的主判据（数值等价）在 `本地` CPU/gloo 上**可以完整完成**，这是少数不需要卡的 L3。
**降级后拿不到的**：TP 的真实性能画像——`all_reduce` 在 NVLink 域内与跨 PCIe 的巨大差异，也就无法体会「为什么 TP size 必须 ≤ 单节点卡数」。这一半结论要留到有卡时补。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 行切后忘了切 `x`，结果形状不匹配 | 没理解行切要求「W 的 in 维」与「x 的最后一维」**同步切分**，这是分片传播规则 |
| 行切后用了 all_gather | 没分清「分片（拼接恢复）」与「部分和（求和恢复）」——这正是 SBP 里 `S` 与 `P` 的区别 |
| bias 加了两遍 | 行切时 bias 只能在 all_reduce **之后**加一次（或只让 rank 0 持有）。这是 Megatron 实现里的经典细节 |
| 用 `atol=0` 判定失败就认为写错了 | 归约顺序改变导致浮点结果不逐比特相同是**正常**的；分不清「数值等价」与「逐比特相同」 |

---

#### L3-DIST-12｜2 节点跨机 DDP/FSDP，对比同机与跨机的通信耗时

- **检验什么**：这条通过 = 你真的掌握了「**带宽域**决定并行策略的放置」——为什么 TP 只能在节点内，DP/ZeRO 可以跨节点，PP 最适合跨低带宽链路
- **前置**：[L1-DIST-04](#l1-dist-04comm_bench-在-cpugloo-上扫消息大小找延迟带宽拐点)、[L2-DIST-07](#l2-dist-07fsdp-vs-ddp显存随卡数近似-1n吞吐略降)
- **资源**：`多机多卡`（2 节点 × 2 卡起；有 IB/RoCE 更能看出差异）
- **预计耗时**：半天（含集群环境调通）

**任务**：用 `scripts/run_multinode.sh` 在 2 节点上跑三组：

1. `comm_bench.py` 的**同机 2 卡** vs **跨机 2 卡**（每节点 1 卡）——固定几个消息大小对比 `busbw`；
2. DDP 的同机 4 卡 vs 跨机 4 卡（2×2）——比较每步耗时；
3. 用 `NCCL_DEBUG=INFO` 抓出 NCCL 实际选的算法（Ring / Tree / NVLS）与检测到的拓扑。

启动方式（两节点各执行一次，`--node_rank` 不同）：

```bash
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=2 \
  --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 train_ddp.py --device cuda --backend nccl ...
```

**先预测再动手**：

1. 同机 NVLink 与跨机 IB 的 `busbw` 大约差几倍？（先查一下你的机器：NVLink 双向带宽 vs 网卡速率）
2. 跨机后受影响更大的是**小消息**还是**大消息**？为什么？这对 bucket 大小的选择有什么影响？
3. 如果必须在这 4 卡上跑 TP=4，与 TP=2 + DP=2（TP 组落在节点内）相比，哪个快？差多少量级？

**验收命令**：

```bash
# 每个节点上分别执行（NODE_RANK 换成 0 / 1）
cd dist-train-lab
NCCL_DEBUG=INFO bash scripts/run_multinode.sh --node-rank $NODE_RANK --master $MASTER_ADDR \
  --nproc 2 --job comm_bench --out out/comm_nccl_2n2g.csv
NCCL_DEBUG=INFO bash scripts/run_multinode.sh --node-rank $NODE_RANK --master $MASTER_ADDR \
  --nproc 2 --job ddp --hidden 2048 --layers 12 --batch 4 --seq 256 --tag ddp_2n2g
# 同机对照（单节点 4 卡，若有）
torchrun --standalone --nproc_per_node=4 comm_bench.py --device cuda --backend nccl --out out/comm_nccl_1n4g.csv
```

**通过标准**：

- 同一消息大小（如 64 MiB）下，`busbw(同机) / busbw(跨机) ≥ 2`，并能报出这个比值；
- DDP 每步耗时：`step_time(跨机) > step_time(同机)`，且增量能被「通信量 ÷ 跨机带宽差」量级解释；
- `NCCL_DEBUG=INFO` 日志中能定位到实际使用的算法与通道数（关键字：`Ring`、`Tree`、`NVLS`、`via NET/IB`、`P2P/direct pointer`），并说明跨机时哪些链路走了 NET；
- 用上述数据回答预测问题 3，给出「TP 不跨机」的定量依据。

**降级方案（没有集群）**：

在单机上用 `NCCL_P2P_DISABLE=1`（禁用 NVLink 直连，强制走 PCIe/共享内存）或 `NCCL_SHM_DISABLE=1` 制造一个**慢链路**，对比开关前后的 `busbw` 与 DDP 每步耗时。这能定性复现「带宽域差异如何影响并行策略」。无卡时用 `comm_bench` 的 gloo 数据 + 手算：把 [§2.3](#23-各文件的职责与关键-api) 的通信量公式代入不同带宽，算出各并行维度的通信时间。
**降级后拿不到的**：真实的跨机延迟量级（微秒 vs 毫秒）、网络抖动与 straggler 现象、NCCL 的跨机拓扑发现与算法切换行为。**这些恰恰是 L3 的全部价值，所以这条降级后只能算「预习」，不能算通过。**

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| rendezvous 超时 | 没打通节点间端口，或 `MASTER_ADDR` 用了内网/外网中错误的一个 |
| 跨机带宽只有几百 MB/s | NCCL 退回到 TCP socket 了。查 `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME` 是否绑到正确网卡 |
| 同机跨机差异不明显 | 消息太小（还在延迟主导区），或同机本来就没有 NVLink（纯 PCIe 机型） |
| 认为跨机只是「慢一点」 | 没建立「带宽域」的概念：差 5~10 倍意味着某些并行方式（TP）在跨机时**根本不可用**，而不是慢一点 |

---

#### L3-DIST-13｜SPMD 三种标注体系对照表，并用 DTensor 实操验证

- **检验什么**：这条通过 = 你真的掌握了「Mesh-TensorFlow / GSPMD / OneFlow SBP **在讲同一件事**」——分片标注是把并行策略编译化的技术原型，也是本仓库分布式线与编译器线的交汇点（见 [`../learning-guides/ai-compiler-foundations-learning-guide.md` §9.2](../learning-guides/ai-compiler-foundations-learning-guide.md#第-9-章-分布式与编译的接缝分片如何变成-ir)）
- **前置**：[L3-DIST-11](#l3-dist-11手写最小张量并行列切--all_gather)
- **资源**：`本地`（纸面练习；DTensor 验证部分用 CPU/gloo 两进程）
- **预计耗时**：2h

**任务**：分两步。

**第一步（纸面）**：对 [L3-DIST-11](#l3-dist-11手写最小张量并行列切--all_gather) 里那个 `nn.Linear(hidden, 4*hidden)` 的**列切**方案，写出四种体系各自的标注写法与关键概念对照：

| 体系 | 设备抽象 | 权重 W 的标注 | 输入 x 的标注 | 输出 y 的状态 | 恢复语义需要的 collective |
|------|---------|--------------|--------------|--------------|------------------------|
| Mesh-TensorFlow | 命名维 device mesh | 沿 `out` 维映射到 mesh 维 | 各 mesh 维上复制 | 沿 `out` 维分片 | all_gather（若下游需要完整张量） |
| GSPMD (XLA/JAX) | mesh + `sharding annotation` | 对 W 标注切分，其余由编译器**传播推导** | 标 replicated | 编译器推导出分片 | 编译器自动插入 |
| OneFlow SBP | placement + SBP 签名 | `S(1)` | `B` | `S(1)` | `S(1) → B` 即 all_gather |
| PyTorch DTensor | `DeviceMesh` | `Shard(1)` | `Replicate()` | `Shard(1)` | `redistribute` 到 `Replicate()` |

**要在表下面用自己的话写清三件事**：

1. 三种体系里，**「部分和」这个状态**分别叫什么（SBP 的 `P` / GSPMD 里的 partial reduction / DTensor 的 `Partial`），以及为什么必须有这个状态——没有它，行切之后的中间结果就无法在类型系统里表达。
2. **谁来插通信**：Megatron 是人手写；GSPMD / SBP / DTensor 是**编译器或运行时按标注推导后自动插入**。这一步就是 [`../learning-guides/ai-compiler-foundations-learning-guide.md` §9.2](../learning-guides/ai-compiler-foundations-learning-guide.md#第-9-章-分布式与编译的接缝分片如何变成-ir) 说的「策略 → IR 属性 → Pass 插通信」。
3. 为什么这套东西对本仓库的项目（算力网上的分布式基础设施）是核心：策略从训练脚本里的 if/else 变成了**可被 Pass 变换的 IR 属性**。

**第二步（实操）**：用 PyTorch DTensor 把上表最后一行跑通，让纸面标注变成可执行代码：

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard, Replicate   # 较早版本：torch.distributed._tensor
mesh = init_device_mesh("cpu", (2,))
dw = distribute_tensor(w, mesh, [Shard(1)])          # 列切，对应 SBP 的 S(1)
dx = distribute_tensor(x, mesh, [Replicate()])       # 对应 SBP 的 B
dy = dx @ dw                                          # 框架按标注自动决定局部计算
y_full = dy.redistribute(mesh, [Replicate()]).to_local()   # 自动插入 all_gather
```

**先预测再动手**：

1. `dx @ dw` 这一行你**没有写任何 collective**，但结果是对的。框架是根据什么推导出「这里不需要通信」的？
2. 若把 `dw` 改成 `Shard(0)`（行切）而 `dx` 仍是 `Replicate()`，会发生什么——报错、自动重分布，还是算出错误结果？先预测再试。
3. `dy` 在行切场景下的状态应该是 `Partial`。`redistribute` 到 `Replicate()` 时框架会插入哪个 collective？和你在 [L3-DIST-11](#l3-dist-11手写最小张量并行列切--all_gather) 里手写的是同一个吗？

**验收命令**：

```bash
cd dist-train-lab
python -c "import torch; print(torch.__version__)"
python -c "import torch.distributed.tensor as t; print(hasattr(t,'distribute_tensor'), hasattr(t,'Shard'))"
torchrun --standalone --nproc_per_node=2 tp_min.py --device cpu --backend gloo --dtensor --hidden 512
```

**通过标准**：

- 对照表四行填满，且「部分和」在三种体系里的名字都写对；
- DTensor 版与单卡 `y_ref` 的 `max_abs_diff < 1e-4`；
- 打印出 `dy.placements`，列切时为 `(Shard(dim=1),)`、行切时为 `(Partial(...),)`——**能看到 `Partial` 出现在 placements 里，是这条最直接的过关证据**；
- 能指出 `redistribute` 在两种情况下插入的 collective 分别是 all_gather 与 all_reduce，与 [L3-DIST-11](#l3-dist-11手写最小张量并行列切--all_gather) 手写的一致。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| `import torch.distributed.tensor` 失败 | torch 版本较早，DTensor 还在 `torch.distributed._tensor`。先跑 §2.2 的版本自查 |
| 认为 GSPMD 需要给每个张量都标注 | GSPMD 的核心正是**只标少数张量，其余由编译器传播推导**。这是它和手写 Megatron 的本质差别 |
| 说不清 `Partial` 存在的必要性 | 没意识到它是**类型系统里的一个状态**：行切的中间结果既不是完整值也不是分片，缺了这个状态就无法做分片传播推导 |
| 把三种体系当成三个不同的技术 | 它们是同一范式的三种语法。能互相翻译，才算真读懂了 [paper-notes/01 §4.2](../paper-notes/01-efficient-training-distributed-infra.md) |

---

## 4. 条目 × 资源需求速查

| 编号 | 一句话 | 级别 | 资源 | 耗时 | 有降级方案 | 降级后仍能通过 |
|------|--------|------|------|------|-----------|--------------|
| [L0-DIST-01](#l0-dist-01建起-lab-骨架并在-cpu-上跑通单进程-baseline) | 建 lab 骨架 + CPU 单进程 baseline | L0 | `本地` | 2h | — | — |
| [L0-DIST-02](#l0-dist-02cpugloo-两进程跑通-ddp并验证-all_reduce-后各-rank-参数一致) | CPU/gloo 两进程 DDP，验证参数一致 | L0 | `本地` | 2h | — | — |
| [L0-DIST-03](#l0-dist-0316φ-显存账手算脚本) | 16Φ 显存账手算脚本 | L0 | `本地` | 0.5h | — | — |
| [L1-DIST-04](#l1-dist-04comm_bench-在-cpugloo-上扫消息大小找延迟带宽拐点) | comm_bench 找延迟/带宽拐点 | L1 | `本地` | 2h | — | — |
| [L1-DIST-05](#l1-dist-05单卡显存峰值实测并与-16φ-手算账对差额) | 单卡显存峰值 vs 16Φ 手算账 | L1 | `单卡GPU` | 2h | 纸面对账 | 否（拿不到真实峰值） |
| [L2-DIST-06](#l2-dist-06ddp-vs-单卡显存几乎不降吞吐接近线性) | DDP vs 单卡：显存不降、吞吐近线性 | L2 | `多卡GPU` | 半天 | gloo 验逻辑 | 否（无显存与加速比） |
| [L2-DIST-07](#l2-dist-07fsdp-vs-ddp显存随卡数近似-1n吞吐略降) | FSDP vs DDP 对比表 | L2 | `多卡GPU` | 半天 | ledger 推演 | 否（对比表须真机） |
| [L2-DIST-08](#l2-dist-08切换-fsdp-的-sharding_strategy-三档映射回-zero-123) | sharding_strategy 三档 → ZeRO-1/2/3 | L2 | `多卡GPU` | 2h | ledger 全扫 | 部分（分级关系可验，1Φ 差额不可验） |
| [L2-DIST-09](#l2-dist-09开关激活重计算量出用算力换显存的兑换率) | 激活重计算的显存/吞吐兑换率 | L2 | `多卡GPU`（单卡亦可） | 2h | 单卡即可 | 是（单卡完整可做） |
| [L2-DIST-10](#l2-dist-10统计一步训练里的通信占比解释-bucketing-与-overlap) | 通信占比、bucketing 与 overlap | L2 | `本地` / `多卡GPU` | 半天 | gloo 打点 | 部分（bucketing 可验，overlap 不可验） |
| [L3-DIST-11](#l3-dist-11手写最小张量并行列切--all_gather) | 手写最小张量并行（列切/行切） | L3 | `多卡GPU`（数值部分 `本地`） | 半天 | CPU/gloo | 是（主判据可在本地完成） |
| [L3-DIST-12](#l3-dist-122-节点跨机-ddpfsdp对比同机与跨机的通信耗时) | 跨机 DDP/FSDP 通信对比 | L3 | `多机多卡` | 半天 | `NCCL_P2P_DISABLE` 造慢链路 | 否 |
| [L3-DIST-13](#l3-dist-13spmd-三种标注体系对照表并用-dtensor-实操验证) | SPMD 三种标注对照 + DTensor 实操 | L3 | `本地` | 2h | — | — |

**统计**：共 13 条。按级别：L0 三条、L1 两条、L2 五条、L3 三条。按资源：`本地` 六条（另有两条的主判据可在本地完成）、`单卡GPU` 一条、`多卡GPU` 五条、`多机多卡` 一条。

**入门线**（对齐 [`./README.md`](./README.md) §2）：L0 三条全做 + L1 至少一条 + **L2 至少两条**（建议 [L2-DIST-06](#l2-dist-06ddp-vs-单卡显存几乎不降吞吐接近线性) 与 [L2-DIST-07](#l2-dist-07fsdp-vs-ddp显存随卡数近似-1n吞吐略降)，它们直接对应根 README §3.1「动手验收」的第 1 条）。

**没有机器时的最大可达进度**：L0 全部 + L1-DIST-04 + L2-DIST-10 的一半 + L3-DIST-11 的主判据 + L3-DIST-13 全部 = **13 条里能实质完成 6 条**，且覆盖了「SPMD 进程模型 / 集合通信语义 / 显存账本 / 分片标注」四块核心知识。剩下的 7 条全部是**性能与真实显存行为**，必须真机。

---

## 5. 机器申请清单

> **已经有多卡机器的话，本节可以跳过**，直接 `bash scripts/run_8gpu_wall.sh`。下表是给还需要申请的人用的。
>
> **如果你的卡是消费级 GeForce（RTX 40/50 系）**：它们**没有 NVLink，且 P2P 被驱动关闭**，
> NCCL 只能退回经 host 中转的 shared-memory 传输，卡间带宽通常只有几十 GB/s——
> 与单卡显存带宽差**一到两个数量级**。
>
> 这会让下面几条的结论和「教科书版」不同，而且**差异本身就是最有价值的部分**：
>
> | 条目 | 在 NVLink 机器上 | 在消费级多卡机器上 |
> |------|-----------------|-------------------|
> | [L2-DIST-06](#l2-dist-06ddp-vs-单卡显存几乎不降吞吐接近线性) 扩展效率 | 容易达到 ≥ 80% | 可能明显低于 80%，缺口由通信解释 |
> | [L2-DIST-07](#l2-dist-07fsdp-vs-ddp显存随卡数近似-1n吞吐略降) FSDP 吞吐 | 相对 DDP 降一点 | 降得更多（FSDP 通信量约 1.5 倍，悬崖越陡越吃亏） |
> | [L3-DIST-11](#l3-dist-11手写最小张量并行列切--all_gather) TP 加速比 | TP=2/4 通常正收益 | **可能 < 1，即负优化** |
>
> 所以判据里的数值区间（如「≥ 1.6x」）要按实际互联条件放宽，**但因果解释不能放宽**：
> 无论测出多少，都要能用「通信字节数 ÷ `comm_bench` 实测带宽」把差值算平。
> 拿不到教科书数字不算失败；**说不清为什么拿不到才算**。

> 下表可直接贴进邮件或消息。话术模板见 [`./README.md` §6.2](./README.md#62-申请机器时可以直接发的话术)——本表是它的「分布式部分」明细，两者配合使用：先用模板说明背景与产出承诺，再附上本表说明具体规格。

| # | 实验内容 | 卡数 / 节点 | 互联要求 | 环境 | 预计机时 | 对应检验条目 | 不做的后果 |
|---|---------|-----------|---------|------|---------|-------------|-----------|
| 1 | 单卡显存峰值实测与 16Φ 账对齐 | 1 卡 | 无 | CUDA 12.x，PyTorch ≥ 2.1 | 1 h | [L1-DIST-05](#l1-dist-05单卡显存峰值实测并与-16φ-手算账对差额) | 显存构成只能停留在理论推演 |
| 2 | DDP vs 单卡：显存与吞吐 | 同机 2 卡 | NVLink 或 PCIe 均可 | 同上 | 2 h | [L2-DIST-06](#l2-dist-06ddp-vs-单卡显存几乎不降吞吐接近线性) | 无法证实「DP 省时间不省显存」 |
| 3 | FSDP vs DDP + ZeRO 三档 | 同机 2 卡（4 卡更佳） | 同上 | 同上 | 3 h | [L2-DIST-07](#l2-dist-07fsdp-vs-ddp显存随卡数近似-1n吞吐略降)、[L2-DIST-08](#l2-dist-08切换-fsdp-的-sharding_strategy-三档映射回-zero-123) | **本模块最核心的实测结论缺失** |
| 4 | 激活重计算 trade-off | 1~2 卡 | 无 | 同上 | 1 h | [L2-DIST-09](#l2-dist-09开关激活重计算量出用算力换显存的兑换率) | 拿不到「用算力换显存」的兑换率 |
| 5 | 通信占比与 overlap 观测 | 同机 2 卡 | 同上 | 同上 + `torch.profiler` | 2 h | [L2-DIST-10](#l2-dist-10统计一步训练里的通信占比解释-bucketing-与-overlap) | 看不到通信与反向的重叠 |
| 6 | 最小张量并行的性能画像 | 同机 2 卡 | **NVLink 优先** | 同上 | 1 h | [L3-DIST-11](#l3-dist-11手写最小张量并行列切--all_gather) | 无法解释「TP 为什么不能跨机」 |
| 7 | 跨机通信与拓扑影响 | 2 节点 × 2 卡 | IB / RoCE 优先，千兆以太亦可对比 | 同上 + `NCCL_DEBUG=INFO` | 3 h | [L3-DIST-12](#l3-dist-122-节点跨机-ddpfsdp对比同机与跨机的通信耗时) | 缺少并行策略放置的定量依据 |

**打包申请建议**（合并成两次申请，减少沟通成本）：

| 批次 | 规格 | 连续机时 | 覆盖条目 | 优先级 |
|------|------|---------|---------|--------|
| 第一批 | **同机 2 卡**（任意 Ampere 及以上，单卡显存 ≥ 16 GB），CUDA 12.x | **半天（4~6 h）** | #1 #2 #3 #4 #5 #6 | **必需**：入门线的两条 L2 主判据全在这一批 |
| 第二批 | **2 节点 × 2 卡**，节点间 IB/RoCE | 半天（3~4 h） | #7 | 可选：L3 加深项，无则以单机降级方案顶替并注明 |

**申请时可以一并说明的三点**（提高通过率）：

1. **脚本已就绪**：所有实验脚本已在本地 CPU（gloo backend）上跑通逻辑，上机只改 `--device cuda --backend nccl` 两个参数，不占用机器做开发调试。
2. **单卡显存需求可控**：主判据用的模型规模为 `hidden=2048, layers=12`（约 4 亿参数，16Φ ≈ 6.4 GB），16 GB 卡即可完成全部 L2；不需要大显存卡。
3. **产出可复现**：每次 run 落一份 JSON，最终产出「策略 × 显存 × 吞吐」对比表与实验记录，提交到仓库的 `docs/notes/`。

---

## 6. 记录方式

每完成一条，按 [`./README.md` §6.1](./README.md#61-怎么记录建议) 在 `docs/notes/` 追加一条短记录，重点三样：**预测 vs 实际**（不一致的地方最值钱）、**卡在哪一步**、**验收命令与输出片段**。

本册额外建议保留两份产物，它们是申请机器与写报告时最直接的材料：

- `out/table_fsdp.md`（[L2-DIST-07](#l2-dist-07fsdp-vs-ddp显存随卡数近似-1n吞吐略降) 的策略 × 显存 × 吞吐对比表）
- `out/table_zero.md`（[L2-DIST-08](#l2-dist-08切换-fsdp-的-sharding_strategy-三档映射回-zero-123) 的四档 ZeRO 手算 vs 实测表）

这两张表加上 [L3-DIST-13](#l3-dist-13spmd-三种标注体系对照表并用-dtensor-实操验证) 的标注对照表，就是「分布式训练基础」这一阶段最完整的过关证据。
