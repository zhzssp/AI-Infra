# tvm-fatbin-lab —— TVM 核心概念 + CUDA fatbin 一键学习项目

一条命令走完两条轨道：

1. **TVM**：算法/调度分离 → 融合 → layout → tensorize → AutoTVM（含 log 复用）→ PackedFunc  
2. **CUDA fatbin**：双架构打包 → `compute_*` vs `sm_*` → `cuobjdump` → 与 IREE `ExecutableVariant` 同构  

对应根 README **§4.1（TVM）** 与 **§5.4（CUDA fatbin）** 的动手验收。

## 在自学体系中的位置

| | |
|--|--|
| **角色** | P1/P2 动手：图级编译 + 多变体打包 |
| **总规划** | [`../README.md`](../README.md) §4.1 · §5.4 |
| **配套教材** | [`../docs/learning-guides/tvm-learning-guide.md`](../docs/learning-guides/tvm-learning-guide.md) · [`../docs/learning-guides/cuda-fatbin-learning-guide.md`](../docs/learning-guides/cuda-fatbin-learning-guide.md) |
| **横切词汇** | [`../docs/learning-guides/ai-compiler-foundations-learning-guide.md`](../docs/learning-guides/ai-compiler-foundations-learning-guide.md) §3–4 · §7.3 |
| **阶段导航** | [`../docs/README.md`](../docs/README.md) 阶段 4 / 阶段 6 |
| **对照项目** | [`../llvm-hello-compile/`](../llvm-hello-compile/) · [`../mlir-toy-dialect/`](../mlir-toy-dialect/) · [`../onnx-delegate-lab/`](../onnx-delegate-lab/)（划分/委托，正交） |

> **主线故事**：TVM 解决「同一算法怎么算得快」；fatbin 解决「同一份产物怎么携带多架构实现」。  
> 二者都是「配置组合爆炸」的解药——一个靠**搜索 + 缓存**，一个靠**多变体打包 + 运行时选择**。

---

## 一、你将学到什么

| 步骤 | 核心概念 | 产物（`out/`） |
|------|----------|----------------|
| **T01** | TE compute vs schedule；`tile` / `cache_write` / `compute_at` / `vectorize` | `tvm/01_matmul_*.lower.txt` |
| **T02** | 四类融合（injective / reduction / complex-out-fusable / opaque） | `tvm/02_relay_*fuse*.txt` |
| **T03** | layout 变换；跨后端转换税（研究问题②） | `tvm/03_*` |
| **T04** | tensorize vs vectorize；异构接入钩子 | `tvm/04_*` |
| **T05** | AutoTVM 搜索闭环 + **tuning log 复用**（研究问题⑥） | `tvm/05_autotvm.log` |
| **T06** | PackedFunc ABI + graph executor 派发 | `tvm/06_*` |
| **F01** | fatbin 多镜像结构 | `cuda/add.fatbin` |
| **F02** | `compute_*`（PTX）vs `sm_*`（SASS）；JIT 退路 | `cuda/dump_*.txt` |
| **F03** | fatbin ↔ IREE ExecutableVariant 同构 | `cuda/READING.md` |

```
同一 matmul
   ├─ TVM：TE → schedule / 融合 / 搜索 → 快实现 + PackedFunc 派发
   └─ fatbin / variant：多 ISA（或多 backend）打进同一逻辑产物 → 运行时选择
```

---

## 二、目录结构

```
tvm-fatbin-lab/
├── README.md
├── requirements.txt          # apache-tvm 等
├── tvm_lab/                  # 6 个 Python 步骤（可单独跑）
│   ├── 01_te_matmul_schedules.py
│   ├── 02_fusion_relay.py
│   ├── 03_layout_transform.py
│   ├── 04_tensorize_demo.py
│   ├── 05_autotvm_tune.py
│   └── 06_packedfunc_run.py
├── cuda/
│   └── add.cu                # 最小 device kernel
├── scripts/
│   ├── env.sh                # 探测 python / tvm / nvcc
│   ├── run.sh                # 【主入口】两轨 + ANALYSIS.md
│   ├── run_tvm.sh
│   ├── run_fatbin.sh
│   └── clean.sh
└── out/                      # 运行后生成
```

---

## 三、一键运行

```bash
cd tvm-fatbin-lab

# 依赖（TVM 轨）
pip install -r requirements.txt
# 或：conda activate 含 apache-tvm 的环境

# CUDA 轨需要 Toolkit：nvcc、cuobjdump 在 PATH 中

bash scripts/run.sh
```

跑完后**先打开** [`out/ANALYSIS.md`](out/ANALYSIS.md)。

分轨重跑：

```bash
bash scripts/run_tvm.sh
bash scripts/run_fatbin.sh

# AutoTVM 试验次数（默认 12，可加大）
AUTOTVM_TRIALS=30 bash scripts/run_tvm.sh

# fatbin 架构号（默认 75+80；旧 Toolkit 可改）
SM_A=70 SM_B=75 bash scripts/run_fatbin.sh
```

单独跑某一步：

```bash
python tvm_lab/01_te_matmul_schedules.py
```

> Windows：与 `llvm-hello-compile` 相同，请在 **Git Bash / WSL** 下跑 `bash scripts/run.sh`。  
> 若本机缺 TVM 或 CUDA，对应轨会标记 `missing-deps` 并继续生成报告，不阻塞另一轨。

---

## 四、过关标准

- [ ] 指着两份 matmul `lower`，讲清至少 4 个 schedule 原语  
- [ ] 四类融合各举一例能融 / 不能融，并能指出 `argsort` 切开点  
- [ ] 说明 `apply_history_best` 如何避免重复搜索  
- [ ] `cuobjdump -lelf` 看到两个 `sm_*`；with_ptx 的 `-lptx` 非空  
- [ ] 口述 fatbin 与 IREE `ExecutableVariant` 的同构（问题 / 打包 / 选择 / 退路）  

---

## 五、刻意不做（保持小而纯）

- VTA / FPGA、Relay pass 全集、MetaSchedule 深挖  
- 手写 PTX、SASS 调优、完整 CUDA Runtime 教程  
- 多机分布式调优  

需要深度时回教材；本项目只把**核心概念变成可落盘的实物**。
