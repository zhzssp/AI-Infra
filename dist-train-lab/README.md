# dist-train-lab —— 分布式训练：把通信代价测出来

> **一句话**：前五个 lab 教你「一张图怎么被编译」，这个 lab 教你「一张图大到单卡装不下时，
> 切到多卡上要付什么代价」。
>
> **链路位置**：站 ②（划分）在**设备维度**上的展开。前面几个 lab 划分的是算子，
> 这里划分的是**张量与模型状态**，但要回答的是同一个问题——
> [链路总纲](../docs/learning-guides/00-end-to-end-pipeline.md)里那句「划分是一个带通信代价的优化问题」。

---

## 一、主角还是那张图，只是大了

全仓库共用一套主角。这个 lab 加的是**第三级尺度**：

| 级别 | 主角 | 尺度 | 在哪里 |
|------|------|------|--------|
| 算子级 | `axpy` | 一层循环 | [`llvm-hello-compile`](../llvm-hello-compile/) |
| 图级 | `tiny_mlp`（`Gemm→Relu→Add`） | `2×3 @ 4×3` | onnx / mlir / tvm / iree 四个 lab |
| **规模级** | **FFN block**（`Gemm→GELU→Add`） | `hidden→4·hidden→hidden` | **本 lab** |
| **分布式级** | **N 层堆叠** | 约 0.4~1.6 G 参数 | **本 lab** |

放大的是**骨架**，不是换了个模型：

```
tiny_mlp :  x ──Gemm(W[4,3], b)──▶ Relu ──▶ Add(bias2) ──▶ y
FFNBlock :  x ──LayerNorm──▶ fc1 ──▶ GELU ──▶ fc2 ──▶ Add(x) ──▶ y
                              ▲        ▲               ▲
                            同一个   同一个          同一个
                           Gemm 角色  激活角色       逐元素加
```

**诚实地说，这不是严格同构**：FFN 比 `tiny_mlp` 多了一个降维矩阵 `fc2` 和一个 `LayerNorm`。
但被放大的正是那根骨架，所以你在前五个 lab 里建立的直觉（权重按 `[out,in]` 存、
归约维在哪、哪些算子是逐元素因而能融合）在这里全部继续成立。

`model.py` 里保留了**原尺寸的 `TinyMlp`** 作数值锚点——喂 `x=[1,2,3]` 必须得到
`[2.5, 3.5, 4.5, 7.5]`，与另外四个 lab 逐位一致。跑 `python model.py` 可验证。

为什么要放大：`tiny_mlp` 只有约 64 次浮点运算，单张 RTX 5090 跑完它约需 `6e-13` 秒——
**比硬件小十二个数量级**。不放大就什么现象都观察不到。

---

## 二、这个 lab 到底要你看见什么

一句话：**通信可以比计算还贵。**

在有 NVLink 的机器上，这件事被硬件掩盖了（NVLink 约 900 GB/s vs HBM 约 3350 GB/s，
只差 3.7 倍），学习者很难对「划分要付通信代价」产生切身感受。而在**消费级多卡机器**上
（RTX 4090 / 5090 这类，没有 NVLink 且 P2P 被关闭），卡间带宽只有几十 GB/s，
和显存带宽差**一到两个数量级**。

代价被放大，于是变得可测。这个 lab 的四组主线实验就是围绕这个悬崖设计的：

| 组 | 测什么 | 在没有 NVLink 的机器上的**预期结论** |
|----|--------|-----------------------------------|
| A | 集合通信带宽 vs 卡数 | 峰值带宽只有显存带宽的百分之几 |
| B | DDP 扩展效率 | 明显低于线性，缺口能被 A 组的带宽解释掉 |
| C | 张量并行加速比 | **可能 < 1，即负优化** |
| D | FSDP / ZeRO 阶梯 | 显存降到约 1/N，吞吐付出可测的代价 |

**C 组测出负优化不是配置错误，是这条工程约束的来源**：TP size 必须 ≤ 单节点卡数，
且节点内要有高速互联。教科书上这句话是结论，在这台机器上它是可复现的实验。

---

## 三、一键启动

### 没有 GPU 也能开始（推荐先做）

分布式里真正难的三件事——**谁在跟谁通信、通信的是什么、通信之后不变量是什么**——
和有没有卡毫无关系。

```bash
bash setup.sh --torch cpu                # 在仓库根目录跑，不需要 root
cd dist-train-lab
bash scripts/run_cpu_smoke.sh            # 几十秒跑完
```

只想单独装这个 lab 的话，`pip install -r requirements.txt` 也够。根目录的
[`setup.sh`](../setup.sh) 是把六个 lab 的依赖一次配齐，细节见
[`docs/environment-setup.md`](../docs/environment-setup.md)。

这一轮会验证：主角数值锚点、16Φ 显存账、单进程 baseline、两进程 DDP 的
`param_diff == 0`、集合通信拐点、张量并行的数值等价。

**但它拿不到任何比值型结论**：CPU 上 `max_memory_allocated` 无意义，
gloo 没有独立通信 stream（观察不到重叠），同机多进程的加速比不具参考性。

### 有多卡时

```bash
bash scripts/run_8gpu_wall.sh            # 四组对照，约 30~60 分钟
```

脚本会按实际卡数自动生成 `world_size = 2, 4, 8, ...` 的序列。产物：

```
out/comm_nccl_ws*.csv     A 组：通信带宽扫描
out/table_scaling.md      B 组：DDP 扩展效率
out/table_zero.md         B'/D 组：显存与 ZeRO 阶梯
```

### 单项运行

```bash
bash scripts/run_ddp.sh 8 -- --bucket-mb 1 --tag bk1     # 改分桶大小
bash scripts/run_fsdp.sh 8                               # 三档 sharding_strategy
bash scripts/run_comm_bench.sh 8
NPROC=8 bash scripts/run_fsdp.sh                         # 等价写法
```

---

## 四、装 torch 的两个坑（会直接卡住你）

1. **RTX 50 系（Blackwell, `sm_120`）必须装 CUDA 12.8 及以上的 wheel**：

   ```bash
   bash setup.sh --torch cu128    # 或 pip install torch --index-url https://download.pytorch.org/whl/cu128
   ```

   装了 cu124 及更早的版本，第一次 kernel launch 会报
   `no kernel image is available for execution on the device`。

2. **多卡还需要 NCCL ≥ 2.26**，否则在 Blackwell 消费卡上连初始化都会失败
   （`ncclMaxSharedMem ... exceeds device maxSharedMem`）。torch 的 wheel 自带 NCCL：

   ```bash
   python -c "import torch; print(torch.cuda.nccl.version())"
   ```

`scripts/env.sh` 启动时会打印卡型、compute capability 与 **P2P 是否可用**，先看那一行。

---

## 五、文件职责

| 文件 | 干什么 |
|------|--------|
| `common.py` | 进程组、显存计量、吞吐计时、产物落盘。所有脚本共用 |
| `model.py` | `TinyMlp`（数值锚点）+ `FFNBlock`（放大版）+ 参数量闭式解 |
| `mem_ledger.py` | 16Φ 显存账手算器。纯 CPU，是所有实测的对照基线 |
| `train_single.py` | 单进程 baseline |
| `train_ddp.py` | DDP，含 `--check-consistency` / `--bucket-mb` / `--accum` / `--zero1` |
| `train_fsdp.py` | FSDP 三档 `sharding_strategy` |
| `comm_bench.py` | 集合通信微基准，扫消息大小找延迟/带宽拐点 |
| `profile_step.py` | 一步训练的通信占比，验证 bucketing 与 overlap |
| `tp_min.py` | 最小张量并行（列切/行切/Megatron FFN）+ DTensor |
| `report.py` | 读 `out/*.json` → markdown 对比表 |

---

## 六、三个最容易算错的地方

这三条在代码里都有注释兜底，但值得先知道：

**① `warmup` 不能省。** Adam 的 `exp_avg` / `exp_avg_sq` 是**第一次 `optimizer.step()`
时才分配**的（共 8Φ）。只跑一步测显存，会得出「实测比 16Φ 还小」的荒谬结论。

**② torch 原生 AMP 不是那本 16Φ 账里的「混合精度」。**
经典混合精度（DeepSpeed/Megatron）是 `fp16 参数 2Φ + fp16 梯度 2Φ + fp32 master/m/v 12Φ`；
而 `torch.autocast` 的参数与梯度**始终是 fp32**，只在算子入口临时 cast，
所以模型状态是 `4Φ + 4Φ + 8Φ`。**合计都是 16Φ，但构成不同**——用 `--precision amp`
实测时要对的是 fp32 那一栏。真正把参数压到 2Φ 的是 FSDP 的
`MixedPrecision(param_dtype=bf16)`，见 `train_fsdp.py`。

**③ 激活不随 ZeRO 分片下降。** FULL_SHARD 把模型状态切成 1/N，但激活纹丝不动。
把激活也算进被除项，就会预期「显存降到精确 1/2」，然后被实测打脸。
要压激活得靠重计算（`--activation-checkpointing 1`）或序列并行——
**它和 ZeRO 是正交的两把刀，大模型上两个都要开。**

---

## 七、过关标准

对应 [`docs/checkpoints/08-distributed.md`](../docs/checkpoints/08-distributed.md) 的 13 条检验。

**无卡即可完成**（6 条，覆盖四块核心知识）：

- [ ] `param_diff == 0`：两 rank 喂**不同**数据，为什么参数还能逐比特一致？
- [ ] `grad_diff < 1e-6`：all_reduce 做的是**平均**不是求和
- [ ] 说得清 16Φ 的构成，以及混合精度与 fp32 合计都是 16Φ 为什么是巧合
- [ ] 指出 comm_bench 的拐点，并解释它和 DDP 梯度分桶的因果
- [ ] 列切配 all_gather、行切配 all_reduce，且 Megatron FFN 整块只要 1 次通信
- [ ] 在 DTensor 的 `placements` 里看到 `Partial`，并说清这个状态为什么必须存在

**需要多卡**（这台机器上全部可做）：

- [ ] A 组：卡间带宽是单卡显存带宽的百分之几？
- [ ] B 组：DDP 8 卡扩展效率是多少？缺口能被 A 组带宽解释吗？
- [ ] B' 组：DDP 每卡显存**不降**（±15% 内），且 `peak_alloc > 16Φ`
- [ ] C 组：TP 加速比。若 < 1，用 all_reduce 字节数 ÷ A 组带宽验证这个差值
- [ ] D 组：`peak(none) > peak(zero1) > peak(grad_op) ≥ peak(full)` 严格单调
- [ ] D 组：ZeRO-0→1 省 6Φ 而 2→3 只省 1Φ，说得清为什么第一级最划算

---

## 八、在自学体系中的位置

| 项目 | 回答的问题 |
|------|-----------|
| `onnx-delegate-lab` | 谁来划分算子、边界付什么代价 |
| `tvm-fatbin-lab` | 划完怎么算得快、多变体怎么打包 |
| `iree-lab` | 一条工业流水线怎么把这些串成相位 |
| **`dist-train-lab`（本项目）** | **划分推广到设备维度：切张量与模型状态，通信代价是多少** |

往上接 [`docs/paper-notes/01-efficient-training-distributed-infra.md`](../docs/paper-notes/01-efficient-training-distributed-infra.md)
（16Φ、ZeRO、并行策略的来源），往下接
[横切概念 §9](../docs/learning-guides/ai-compiler-foundations-learning-guide.md#第-9-章-分布式与编译的接缝分片如何变成-ir)
（分片标注怎么变成 IR 属性——`tp_min.py` 里手写的那些 collective，
在 GSPMD / SBP / DTensor 里是**编译器按标注自动插入**的）。

这也是仓库[六个研究问题](../README.md#6-第五阶段多后端协同的前沿研究问题)里
**第 1 问（子图划分与跨设备传输最小化）**最直接的实验场：
A 组给出通信代价的常数，B/C/D 组给出不同划分方式的实测代价——
把它们凑起来就是一个真实的 cost model 输入。
