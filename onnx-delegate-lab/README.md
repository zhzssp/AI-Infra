# onnx-delegate-lab —— ONNX / ORT EP + ExecuTorch Partitioner 学习项目

一条命令走完「多后端委托」两条工业路径：

1. **ONNX + ORT EP**：helper 构图 → 改图 → **打印 EP 分区**（GetCapability → Compile → 派发）  
2. **ExecuTorch Partitioner**：只接管部分算子 → 对比 per-node / connected tag → **数边界与 delegate 子图**

对应根 README **§4.2** 动手验收；对准研究问题 **①②**。

> **不要与 `tvm-fatbin-lab` 合并**：那边回答「划完怎么算得快 / 多变体打包」；这边回答「谁来划分、边界付什么代价」。同构，分层不同——见 [`docs/learning-guides/ai-compiler-foundations-learning-guide.md`](../docs/learning-guides/ai-compiler-foundations-learning-guide.md) §3.3–3.4。

## 在自学体系中的位置

| | |
|--|--|
| **角色** | P1 动手：交换格式 + 子图委托 |
| **链路位置** | 第 **① 站（表示）** 与第 **② 站（划分）**，见 [`../docs/learning-guides/00-end-to-end-pipeline.md`](../docs/learning-guides/00-end-to-end-pipeline.md) |
| **总规划** | [`../README.md`](../README.md) §4.2 |
| **配套教材** | [`../docs/learning-guides/onnx-learning-guide.md`](../docs/learning-guides/onnx-learning-guide.md) · [`../docs/learning-guides/executorch-learning-guide.md`](../docs/learning-guides/executorch-learning-guide.md) |
| **横切词汇** | [`../docs/learning-guides/ai-compiler-foundations-learning-guide.md`](../docs/learning-guides/ai-compiler-foundations-learning-guide.md) §3.3（四类边界代价）· §3.4（四栈选型） |
| **阶段导航** | [`../docs/README.md`](../docs/README.md) 阶段 4 |
| **正交对照** | [`../tvm-fatbin-lab/`](../tvm-fatbin-lab/)（融合 / AutoTVM / fatbin） |

## 主角模型：`tiny_mlp`

全仓库共用一套主角：**四个 lab 跑的是同一张图**（本 lab、`mlir-toy-dialect`、`iree-lab`、`tvm-fatbin-lab`），
另外两个各取一级尺度——`llvm-hello-compile` 钻进最内层算子，`dist-train-lab` 把这根骨架放大到 GPU 尺寸，
第五个 `llvm-hello-compile` 钻进这张图的最内层算子。所以各阶段的数字可以横着比：

```
tiny_mlp:  x[2,3] ─▶ Gemm(W[4,3], b) ─▶ Relu ─▶ Add(bias2) ─▶ y[2,4]
                     ↑ W 存成 [out,in]，靠 transB=1 对齐
```

本 lab 里它出现三次，每次都为了制造一个可观测的边界：

| 出现在 | 形态 | 为什么这么改 |
|--------|------|------------|
| `onnx_lab/01` | 原样（batch 动态） | 讲 IR 结构与 shape 推断，越干净越好 |
| `onnx_lab/03` | 后面接 `Softmax + ReduceSum`，batch 固定成 2 | 三个算子谁都支持，**整图被一个 EP 吃掉就没有边界可看** |
| `executorch_lab/01` | 叠两层，中间夹 `Softmax` | 同理；两段之间必须断开，`per_node` 与 `connected` 才有差别 |

同一份权重（`W` 用 `[out,in]` 布局）也出现在 [`../iree-lab/models/tiny_mlp.mlir`](../iree-lab/models/tiny_mlp.mlir) 与
[`../tvm-fatbin-lab/tvm_lab/02_fusion_relay.py`](../tvm-fatbin-lab/tvm_lab/02_fusion_relay.py)，
喂 `x=[1,2,3]` 时四个系统都该给出 `[2.5, 3.5, 4.5, 7.5]`。

---

## 一、你将学到什么

| 步骤 | 核心概念 | 产物 |
|------|----------|------|
| **O01** | Model/Graph/Node；initializer vs runtime input；shape infer；`transB` 作为属性 | `out/onnx/01_*` |
| **O02** | helper 插节点 / 改 initializer；**拓扑序**为何不能 `append`；checker | `out/onnx/02_*` |
| **O03** | ORT providers；分区观察；四类边界代价 | `out/onnx/03_*` |
| **E01** | Partitioner 契约；tag 粒度 ↔ 边界数；Edge Dialect 两层 dump | `out/executorch/01_*` |
| **E02** | ExecuTorch / ORT / IREE 三系统对照 | `out/executorch/02_THREE_SYSTEMS.md` |

```
同一张 tiny_mlp
  ├─ ORT：Session 构建期多 EP 协商划分        ← O03
  ├─ ExecuTorch：导出期 Partitioner 打 tag    ← E01
  ├─ TVM：编译期 FuseOps（单后端划分原型）     ← ../tvm-fatbin-lab/tvm_lab/02
  └─ IREE：编译相位形成 flow.dispatch         ← ../iree-lab/scripts/run_phases.sh
```

**三个系统的差别不在"能不能划分"，在"什么时候决定"。** 跑完这两条轨，
再去 `iree-lab` 数一次 `flow.dispatch`，你手上就有同一张图的四组数字。

---

## 二、目录结构

```
onnx-delegate-lab/
├── README.md
├── requirements.txt
├── lab_common.py
├── onnx_lab/
│   ├── 01_build_and_infer.py
│   ├── 02_mutate_graph.py
│   └── 03_ort_ep_partition.py
├── executorch_lab/
│   ├── 01_partitioner_lab.py    # 主角叠两层；真实 ET 或概念模拟
│   └── 02_compare_three_systems.py
├── scripts/
│   ├── env.sh / run.sh / run_onnx.sh / run_executorch.sh / clean.sh
└── out/                         # 运行后生成
```

---

## 三、一键运行

```bash
cd onnx-delegate-lab
pip install -r requirements.txt          # onnx + onnxruntime
# 可选：按官方文档安装 torch + executorch，以落盘真实 .pte

bash scripts/run.sh                      # Git Bash / WSL
# 先打开 out/ANALYSIS.md
```

分轨：

```bash
bash scripts/run_onnx.sh
bash scripts/run_executorch.sh
```

> 未安装 ExecuTorch 时，E01 仍会用 **模拟 tag** 产出边界数对比报告；装好后重跑即可生成 `.pte`。  
> 未安装 CUDA 等第二 EP 时，O03 仍完成 CPU 轨 + 概念边界说明；有 `CUDAExecutionProvider` 时自动跑双 EP 并写 profile。

---

## 四、过关标准

- [ ] 默画 ONNX 结构层次；分清 initializer / runtime input  
- [ ] 说清 `transB=1` 为什么是属性不是输入，以及这对下游融合意味着什么  
- [ ] 解释插节点时为何必须 `insert` 而不能 `append`（拓扑序）  
- [ ] 说出 ORT 三拍，并能解释一次分区边界（真实或概念）  
- [ ] 解释 Partitioner 为何只打 tag、partition 期不改图  
- [ ] 对比 per-node vs connected：谁边界更多、为何连到研究问题①  
- [ ] **说出 per_node 粒度下丢掉的那次融合机会，后面哪个 lab 能补、哪个补不回来**  
- [ ] 用一张表对比 ET / ORT / IREE 的**决策时刻**  

---

## 五、刻意不做

- 各厂商 EP / backend 内部实现（TensorRT builder、QNN…）  
- ONNX 全量算子手册、训练扩展  
- 分布式调度、跨节点编排（那是 IREE/HAL + 分布式综述的线）  
