# AI-Infra 自学体系枢纽

> 这是整套自学材料的**导航页**。  
> 优先级、必学/跳过清单、时间表、研究问题 → 看根目录 [`../README.md`](../README.md)。  
> 本页只回答一件事：**每一阶段读什么、做什么、做到什么算过关。**

---

## 0. 仓库四层分工

```
AI-Infra/
├── README.md                 ← ① 总规划：目标 / 优先级 / 必学·跳过 / 验收 / 时间表 / 研究问题
├── docs/                     ← ② 知识库（本目录）
│   ├── README.md             ←    你正在读：自学体系枢纽
│   ├── ai-compiler-foundations.md ← 横切前置概念（开局读总图+§2.4，其余随阶段分轮补）
│   ├── checkpoints/          ←    知识落地检验体系：每个工具一册动手过关标准
│   ├── paper-notes/          ←    论文精读笔记（为什么、框架、最小必要集）
│   ├── mlir-learning-guide.md←    MLIR（Conversion / Interface / Linalg）
│   ├── iree-learning-guide.md←    IREE（核心 + HAL）
│   ├── tvm-learning-guide.md ←    TVM（图编译 + 调度搜索）
│   ├── onnx-learning-guide.md←    ONNX IR + ORT EP
│   ├── executorch-learning-guide.md← ExecuTorch（to_backend / Partitioner）
│   ├── llvm-learning-guide.md←    LLVM（四层 IR / New PM / CodeGen）
│   └── cuda-fatbin-learning-guide.md← CUDA fatbin（半天）
├── llvm-hello-compile/       ← ③ 动手项目 A：LLVM 编译全链路 + 自定义 Pass
├── mlir-toy-dialect/         ← ③ 动手项目 B：MLIR Dialect / Conversion / Interface
├── tvm-fatbin-lab/           ← ③ 动手项目 C：TVM 调度/融合/AutoTVM + CUDA fatbin
├── onnx-delegate-lab/        ← ③ 动手项目 D：ONNX/ORT EP + ExecuTorch Partitioner
└── paper/                    ← ④ 原文 PDF（遇到细节再翻）
```

| 层 | 职责 | 怎么用 |
|----|------|--------|
| **总规划** | 决定学什么、不学什么、学到什么程度 | 每周回看优先级表与验收清单 |
| **知识库** | 把论文/官方文档蒸馏成可读笔记 | 开局先扫 [foundations](./ai-compiler-foundations.md)；再按阶段读专题文档 |
| **动手项目** | 把概念变成可执行产物 | 先跑通脚本，再对照产物读笔记 |
| **检验体系** | 判定「真会」还是「看过」 | 按 [`checkpoints/`](./checkpoints/README.md) 逐条改代码、跑命令、比产物 |
| **原文** | 权威出处 | 笔记说不清时再打开 PDF |

**铁律**：规划里写「先跳过」的，知识库里也标了跳过——不要被长文档带偏。

---

## 1. 一条主线：读 → 做 → 验

每一个知识点都必须走完这个闭环，否则只是「看过」：

```
   ┌────────────┐     ┌────────────┐     ┌────────────┐
   │  读材料    │ ──▶ │  动手项目   │ ──▶ │  对照验收   │
   │ 笔记/专题  │     │ 脚本/实验  │     │ checkpoints │
   └────────────┘     └────────────┘     └─────┬──────┘
         ▲                                     │
         └──────── 讲不清楚就回查文档 ──────────┘
```

> 阶段地图里的「验」是**口述题**（能不能讲清）。口述过了不等于会做——动手判定在 [`checkpoints/`](./checkpoints/README.md)：每个工具一册，把知识点绑到「加一个组件 + 跑出预期差异」上。

四个动手项目在主线上的位置：

| 项目 | 角色 | 何时做 |
|------|------|--------|
| [`llvm-hello-compile/`](../llvm-hello-compile/) | **编译器地基**：源码→IR→Pass→汇编；手写 New PM Pass | 学 MLIR 之前或卡在 `llvm` dialect 时复习（P2，但建议先跑一遍） |
| [`mlir-toy-dialect/`](../mlir-toy-dialect/) | **MLIR 心智模型**：双 dialect、Region、Interface、Dialect Conversion | P0 第二阶段主战场；后续打通 Toy→linalg→llvm 端到端 |
| [`tvm-fatbin-lab/`](../tvm-fatbin-lab/) | **TVM + fatbin**：调度/融合/AutoTVM + 多架构打包 | 阶段 4（TVM 轨）/ 阶段 6（CUDA 轨）；`bash scripts/run.sh` → `out/ANALYSIS.md` |
| [`onnx-delegate-lab/`](../onnx-delegate-lab/) | **委托/分区**：ONNX 构图 + ORT EP + ExecuTorch Partitioner | 阶段 4；研究问题①② 的实验室 |

> LLVM 项目回答「单层 IR + Pass 怎么工作」；MLIR 项目回答「多层 dialect + 渐进 lowering 怎么工作」。  
> 两者同构（ODS≈TableGen、lit/FileCheck、analysis 失效），对照着做会快很多。  
> P1 两个 lab 正交：TVM 练「算得快」；ONNX/ET 练「谁来划分」。默认 TVM → ONNX；时间紧或冲研究问题则先 ONNX。

---

## 2. 阶段地图（完整自学路径）

按顺序走。每一行都是「读什么 → 做什么 → 过关标准」。

### 开局｜横切词汇表（0.5 天，贯穿全程回查）

| | 内容 |
|--|------|
| **读** | [`ai-compiler-foundations.md`](./ai-compiler-foundations.md) 第 1 章总图 + **[§2.4 同步路线表](./ai-compiler-foundations.md#24-与工具路线同步的概念学习路线)** + 附录速查，**读完就停**。该文**不是开局读完的文档**：26 条概念按 W0～W8 分批挂在下面的阶段表上，每轮开头补下一个工具要用的那几节 |
| **做** | 自己默画五层栈（框架模型是输入；五层 = 交换 IR → 中端 → 执行规划 → 后端 IR/机器码 → 运行时），在每层旁写上仓库里对应的文档名；再照 §2.4 抄一份「本轮该补哪几节」的清单贴在手边 |
| **验** | ① 默画出五层栈并说出每层固化什么决定；② 说得出四根 ★★ 柱子各自**在第几轮**补（分片标注 W1／渐进 lowering W2／设备抽象机+同步 W3／划分与边界代价 W3 引入·W5 坐实）——**此刻不要求讲得清内容**，那是各轮的事 |
| **何时回查** | 每轮开头按 foundations [§2.4 路线表](./ai-compiler-foundations.md#24-与工具路线同步的概念学习路线)取本轮概念；卡住时用 [§11.1 回查触发表](./ai-compiler-foundations.md#111-开局读什么--卡住时回查什么)（症状→章节）或 [第 10 章对照表](./ai-compiler-foundations.md#第-10-章-概念--深入材料对照) |

### 阶段 0｜编译器地基（半天～1 天，可与阶段 1 并行）

| | 内容 |
|--|------|
| **读** | 先补 foundations **§6.1 SSA + §6.3 Pass**（§6.2 渐进 lowering 留到阶段 2）→ [`llvm-learning-guide.md`](./llvm-learning-guide.md) 第 1～3 章 + 附录；需要动机时扫 [`paper-notes/02-llvm.md`](./paper-notes/02-llvm.md) §1～2 |
| **做** | `cd llvm-hello-compile && bash scripts/run.sh` → 先读 `out/ANALYSIS.md` |
| **验** | 能指着 `02_sum_O0.ll` vs `03a_sum_mem2reg.ll` 讲清 SSA/`phi`；能说出 CountIR（分析）与 InjectLogging（变换）的差别 |
| **回查触发** | 写 MLIR→LLVM lowering 踩到 `poison`/`noalias`/向量化失败 → 回 [`llvm-learning-guide.md`](./llvm-learning-guide.md) 第 2.7、4.2、5 章 |

### 阶段 1｜分布式训练基础（P0，约 1 周；对应周次表 W1）

| | 内容 |
|--|------|
| **读** | 先补 foundations **§9.1–9.3 + §5.4 内存层次**（§9.4–9.5 属自选，跳过）→ [`paper-notes/01-efficient-training-distributed-infra.md`](./paper-notes/01-efficient-training-distributed-infra.md)：先 §2 框架 + §12 最小必要集，再抠 SPMD / 16Φ / 五种并行；三种标注体系对照着扫即可 |
| **做** | 根 README §3.1 入门验收：优先 DDP vs FSDP 显存实测（1～2 条即可） |
| **验** | 五分钟讲清「70B 怎么切到 128 卡」；背出 16Φ；三种分片标注能对照说同一策略 |
| **串到后面** | 「并行策略要变成 IR 属性」→ 阶段 2/3 的 MLIR + IREE 才有的放矢 |

### 阶段 2｜MLIR 入门（P0，约 1 周；对应周次表 W2）

| | 内容 |
|--|------|
| **读** | 先补 foundations **§6.2 渐进 lowering + §5.1 tensor/buffer + §6.4 代价模型（浅）** → 扫 [`paper-notes/03-mlir.md`](./paper-notes/03-mlir.md) → [`mlir-learning-guide.md`](./mlir-learning-guide.md)（Conversion / Interface / Linalg 三章最重）+ [`mlir-toy-dialect/README.md`](../mlir-toy-dialect/README.md) |
| **做** | `cd mlir-toy-dialect && bash scripts/all.sh`；对照读 `ConvertToyToLow.cpp` 与 `ToyCostPass.cpp`。Toy→llvm 端到端属加深，不挡入门 |
| **验** | 默画 Operation⊃Region⊃Block；说清 greedy vs Dialect Conversion；解释 Interface 为何让一个 Pass 跨 dialect 工作 |
| **与 LLVM 项目对照** | `toy-opt` ≈ `opt`；`--toy-to-low-convert` ≈ 一次跨 IR 变换；lit 写法两边一样 |

### 阶段 3｜IREE + HAL（P0，约 1 周；对应周次表 W3）——主线核心

| | 内容 |
|--|------|
| **读** | 先补 foundations **§8.1–8.2 设备抽象机+同步、§3.3 划分（先建立直觉）、§7.1 kernel、§7.3 variant** → 扫 [`paper-notes/07-tinyiree.md`](./paper-notes/07-tinyiree.md) → [`iree-learning-guide.md`](./iree-learning-guide.md)（HAL 第 4 章最重） |
| **做** | `--compile-to` 至少 dump 两层；单后端（CPU 即可）`iree-run-module` 跑通。双后端 / C runtime 属加深 |
| **验** | 画出 `linalg→flow→stream→hal`；口述 HAL 对象关系与 timeline semaphore 为何强于 `CUevent` |
| **串到项目目标** | HAL 的 device/buffer/command_buffer/variant ≈ 算力网「异构设备统一抽象」的现成词汇表 |

### 阶段 4｜图级编译与多后端委托（P1，约 1～1.5 周；对应周次表 W4+W5）

| | 内容 |
|--|------|
| **读** | **W4 前**补 foundations §4.3 Roofline → §3.2 融合 → §4.1–4.2 算法/调度+tiling → §5.2 layout，然后 TVM：[`05-tvm.md`](./paper-notes/05-tvm.md) → [`tvm-learning-guide.md`](./tvm-learning-guide.md)<br>**W5 前**补 §3.3 四类边界代价 + §8.3 AOT/JIT，然后委托：[`onnx-learning-guide.md`](./onnx-learning-guide.md) + [`executorch-learning-guide.md`](./executorch-learning-guide.md) |
| **做（建议顺序）** | ① `tvm-fatbin-lab` TVM 轨；② `onnx-delegate-lab`。时间紧 / 冲研究问题：对调。CUDA fatbin 轨留到阶段 6 |
| **验** | 四类融合直觉；对着 lower 讲主要 schedule 原语；指出 EP/Partitioner 边界；对照 ET / ORT / IREE 三种划分答案 |

### 阶段 5（自选补充）｜负载侧论据，不占独立周次

> 这两篇是**额外补充**的论文，不在主链上，也没有配套动手项目；结论已蒸馏进 [foundations](./ai-compiler-foundations.md) §4.3 / §9.4 / §9.5，**跳过原文不断链**。  
> 要读就按下面两个身份穿插进已有阶段，不要当成第五条技术栈。

| 篇目 | 穿插到哪 | 只需回答 | 用时 |
|------|----------|----------|------|
| [08 FlashAttention](./paper-notes/08-flash-attention.md) | **阶段 4 开头**（学 schedule 之前） | 为什么值得做融合 / tiling / 重计算：瓶颈是 HBM 访存而非 FLOPs | 约 0.5 天（公式看懂即可，不手推） |
| [09 PagedAttention](./paper-notes/09-paged-attention-vllm.md) | **阶段 6 收束**（与研究问题一起） | 哪些决策编译器静态做不了；块化状态为何才搬得动（问题③④） | 约 0.5～1 天 |

### 阶段 6｜回填与收束（P2 + 研究问题观点；约 1～2 周；对应周次表 W6～W8）

| | 内容 |
|--|------|
| **读** | 先补 foundations §7.2 虚拟/真实 ISA（回看 §7.3）→ Halide / Glow 短读 + [`cuda-fatbin-learning-guide.md`](./cuda-fatbin-learning-guide.md)；需要时回 llvm 学习文档 CodeGen<br>写观点前：foundations §3.4 四条栈选型（此时才有判断力）+ [09 PagedAttention](./paper-notes/09-paged-attention-vllm.md)（自选，问题③④的样本） |
| **做** | `tvm-fatbin-lab` CUDA 轨（有环境时）；对照 IREE [`ExecutableVariant`](./iree-learning-guide.md#47-设备代码executable--variant--export) |
| **验** | 对根 README §6 **六个**研究问题写出观点笔记：价值 / 已有边界 / **自己的判断**（不要求实现方案） |

---

## 3. 材料总表（按角色）

### 3.1 专题学习文档（机制向，可当教材）

| 文档 | 优先级 | 配套动手 | 一句话 |
|------|--------|----------|--------|
| **[AI 编译器前置核心概念](./ai-compiler-foundations.md)** | **开局读总图+§2.4；其余分轮补** | 默画五层栈；每轮开头按 §2.4 取那几节 | 横切词汇表：把各项目粘成一张总图 |
| [IREE 学习文档](./iree-learning-guide.md) | **P0** | `iree-compile` / `iree-run-module`（见根 README §3.3） | 多后端统一运行时 + HAL 全景 |
| [MLIR 学习文档](./mlir-learning-guide.md) | **P0** | [`mlir-toy-dialect/`](../mlir-toy-dialect/) | Conversion / Interface / Linalg / Bufferize |
| [TVM 学习文档](./tvm-learning-guide.md) | P1 | [`tvm-fatbin-lab/`](../tvm-fatbin-lab/) TVM 轨 | 融合/layout/调度原语/调优闭环/PackedFunc |
| [ONNX 学习文档](./onnx-learning-guide.md) | P1 | [`onnx-delegate-lab/`](../onnx-delegate-lab/) ONNX 轨 | IR 层次 + **GetCapability→Compile** |
| [ExecuTorch 学习文档](./executorch-learning-guide.md) | P1 | [`onnx-delegate-lab/`](../onnx-delegate-lab/) ExecuTorch 轨 | `to_backend` + 分区边界代价 |
| [LLVM 学习文档](./llvm-learning-guide.md) | P2 | [`llvm-hello-compile/`](../llvm-hello-compile/) | 四层 IR、New PM、后端七阶段、与 MLIR 接缝 |
| [CUDA fatbin 学习文档](./cuda-fatbin-learning-guide.md) | P2（半天） | [`tvm-fatbin-lab/`](../tvm-fatbin-lab/) CUDA 轨 | 多架构镜像打包；同构于 IREE `ExecutableVariant` |

**材料怎么配合**：论文笔记讲「为什么」→ [foundations](./ai-compiler-foundations.md) 讲「用什么词粘起来」→ 各 `*-learning-guide.md` 讲「这个项目怎么工作」→ 动手项目把概念跑通。

### 3.2 论文精读笔记（动机 + 框架 + 必要集）

完整索引与阅读顺序见 [`paper-notes/README.md`](./paper-notes/README.md)。

| 优先级 | 笔记 | 配套 |
|--------|------|------|
| P0 | [01 分布式训练综述](./paper-notes/01-efficient-training-distributed-infra.md) | 实测 DDP/FSDP |
| P0 | [03 MLIR](./paper-notes/03-mlir.md) | [`mlir-learning-guide.md`](./mlir-learning-guide.md) + [`mlir-toy-dialect/`](../mlir-toy-dialect/) |
| P1 | [05 TVM](./paper-notes/05-tvm.md) | TVM→[`tvm-learning-guide.md`](./tvm-learning-guide.md)；委托→ONNX/ExecuTorch 学习文档 |
| P1（自选补充） | [08 FlashAttention](./paper-notes/08-flash-attention.md) · [09 PagedAttention](./paper-notes/09-paged-attention-vllm.md) | 论据而非技术栈：FA→阶段 4 动机；PA→研究问题③④。结论见 foundations §4.3/§9.4/§9.5 |
| P2 | [02 LLVM](./paper-notes/02-llvm.md) · [04 Halide](./paper-notes/04-halide.md) · [06 Glow](./paper-notes/06-glow.md) · [07 TinyIREE 简记](./paper-notes/07-tinyiree.md) | LLVM→学习文档；TinyIREE→IREE 学习文档 |

### 3.3 动手项目速查

| 项目 | 一键入口 | 建议阅读产物 | 对应知识库 |
|------|----------|--------------|------------|
| [llvm-hello-compile](../llvm-hello-compile/) | `bash scripts/run.sh` | `out/ANALYSIS.md`，再对比 `02_*.ll` / `03a_*.ll` / `05_*.s` | [llvm-learning-guide](./llvm-learning-guide.md) |
| [mlir-toy-dialect](../mlir-toy-dialect/) | `bash scripts/all.sh` | `scripts/run.sh` 九组演示 + `test/*.mlir` | [mlir-learning-guide](./mlir-learning-guide.md) · [03-mlir](./paper-notes/03-mlir.md) |
| [tvm-fatbin-lab](../tvm-fatbin-lab/) | `bash scripts/run.sh` | `out/ANALYSIS.md` + `out/tvm/*` + `out/cuda/*` | [tvm-learning-guide](./tvm-learning-guide.md) · [cuda-fatbin](./cuda-fatbin-learning-guide.md) |
| [onnx-delegate-lab](../onnx-delegate-lab/) | `bash scripts/run.sh` | `out/ANALYSIS.md` + `out/onnx/*` + `out/executorch/*` | [onnx](./onnx-learning-guide.md) · [executorch](./executorch-learning-guide.md) |

### 3.4 检验体系（动手过关标准）

总纲与分级定义见 [`checkpoints/README.md`](./checkpoints/README.md)：L0 复现 → L1 改一处 → **L2 加组件（主判据）** → L3 打通。每条都有可执行的验收命令与机器可判定的通过标准，并标注了资源需求（本地 / 需 GPU / 需集群）。

| 分册 | 主判据（L2 举例） | 最低资源 |
|------|------------------|---------|
| [01 LLVM](./checkpoints/01-llvm.md) | 写一个新 Pass 并注册到正确 PM 层级，补 FileCheck 测试 | 本地+工具链 |
| [02 MLIR](./checkpoints/02-mlir.md) | 新增 `toy.sub`：ODS → fold → 两版 lowering → lit 测试 | 本地+工具链 |
| [03 IREE](./checkpoints/03-iree.md) | 逐相位 dump + 改模型观察 dispatch 划分变化 | 本地+工具链 |
| [04 TVM](./checkpoints/04-tvm.md) | 新增算子并写多套 schedule，给出量化对比 | 本地+工具链 |
| [05 ONNX + ORT](./checkpoints/05-onnx-ort.md) | 手写图优化 pass + 观察 EP 分区边界变化 | 本地+工具链 |
| [06 ExecuTorch](./checkpoints/06-executorch.md) | 写自定义 Partitioner 并对比边界账 | 本地+工具链 |
| [07 CUDA fatbin](./checkpoints/07-cuda-fatbin.md) | 构造指定架构组合的 fatbin 并用 dump 验证 | 本地+工具链 |
| [08 分布式训练](./checkpoints/08-distributed.md) | DDP vs FSDP 显存实测，印证 16Φ 账 | **多卡GPU（需申请）** |

**绝大多数条目在个人开发机上就能做完**；需要申请机器的部分集中在 08 分布式与各册的 L3 性能项，申请话术见 [checkpoints §6.2](./checkpoints/README.md#62-申请机器时可以直接发的话术)。

---

## 4. 概念穿线（为什么这些材料要放在一起）

自学时最容易「学成孤岛」。**完整横切词汇与五层栈总图**在 [`ai-compiler-foundations.md`](./ai-compiler-foundations.md)；下面只钉「动手项目 ↔ 编译栈」的最短对照：

```
llvm-hello-compile                mlir-toy-dialect                 IREE
─────────────────                ────────────────                 ────
Module/Function/BB     ≈         Module/func.func/Block
Instruction            ≈         Operation
phi                    ≈         block argument（改进）
Pass + PreservedAnalyses ≈       Pass + DialectConversion
TableGen (.td)         =         ODS (.td) 同一套语言
opt / llc              ≈         toy-opt                          iree-compile
.ll → .s               ≈         toy → low → (linalg→) llvm       linalg→flow→stream→hal
无限虚拟寄存器          →         多层 dialect 保留领域语义         HAL 把设备抽象钉死

tvm-fatbin-lab（融合/schedule）     ≈     onnx-delegate-lab（EP/Partitioner）
单后端「划进一个 kernel」                 多后端「谁吃哪段子图」
算法/调度 + AutoTVM                       边界四类代价（拷贝/layout/内存/同步）
```

**对项目目标的汇合点**（与根 README §1.2、foundations §1/§9 同一张图）：

- 分布式线给出「要表达什么」（分片、并行、容错状态）——foundations §9  
- MLIR 给出「用什么结构表达」（dialect / attribute / interface）  
- IREE HAL 给出「落到异构设备时用什么词汇」（device / buffer / fence / variant）——foundations §8  
- TVM / ONNX·ET 给出「子图怎么划、划完怎么算得快」——foundations §3–4；动手见两 lab  
- LLVM / fatbin 给出「最后一公里机器码与多变体怎么来」——foundations §7

---

## 5. 怎么用这套体系（三种节奏）

### A. 标准 6～8 周（推荐，入门上手）

按根 README §7.2 周次表走（合计约 80～120 小时）；每周固定动作：

1. 打开本页对应阶段的「读」  
2. 完成「做」（停在入门验收，不主动做加深项）  
3. 用「验」自检；不过关只回查列出的文档章节，不另开战场  

### B. 应急 3～4 周

根 README §7.3：foundations + llvm → 分布式核心 → MLIR `all.sh` → IREE dump + **先** [`onnx-delegate-lab`](../onnx-delegate-lab/) → 研究问题观点笔记；有余力再补 [`tvm-fatbin-lab`](../tvm-fatbin-lab/) TVM 轨。

### C. 只缺某一块时（索引式）

| 我卡住了… | 去读 | 去做 |
|-----------|------|------|
| 各项目怎么拼成一张图 / 公共词汇 | [ai-compiler-foundations](./ai-compiler-foundations.md) §1–2、§10–11 | 默画五层栈 |
| ORT / ExecuTorch / IREE / TVM 该用哪个 | [ai-compiler-foundations](./ai-compiler-foundations.md) §3.4 选型对照 | 对着自己的场景填一遍表 |
| SSA / Pass / 汇编从哪来 | foundations §6 → llvm-learning-guide §1–3、§5 | llvm-hello-compile |
| Dialect / Conversion / Interface | foundations §5–6 → [mlir-learning-guide](./mlir-learning-guide.md) §4/§6/§8 | mlir-toy-dialect 的 convert/cost/region 测试 |
| 设备抽象 / fence / 多后端 | foundations §8 → iree-learning-guide §3–4 | `--compile-to` dump + 双后端跑通 |
| fatbin / 多变体打包 | foundations §7 → [cuda-fatbin-learning-guide](./cuda-fatbin-learning-guide.md) + IREE §4.7 | 双 `-gencode` + `cuobjdump -lelf/-lptx` |
| 并行怎么切 / 显存为什么爆 | foundations §9 → 01 综述 §4/§6/§7 | DDP/FSDP 实测 |
| 子图划分 / 委托 | foundations §3 → [onnx](./onnx-learning-guide.md) + [executorch](./executorch-learning-guide.md)；TVM 融合见 [tvm §2.2](./tvm-learning-guide.md) | [`onnx-delegate-lab`](../onnx-delegate-lab/) |
| schedule / AutoTVM | foundations §4 → [tvm-learning-guide](./tvm-learning-guide.md) §3–5；动机见 [05-tvm](./paper-notes/05-tvm.md) | matmul 两 schedule + 一次调优 |
| 为什么要融合/tiling；大状态为何搬不动 | foundations §4.3 / §9.4 / §9.5；要实证再翻 08 / 09（自选） | 对着 Roofline 说瓶颈；用块化解释问题③④ |

---

## 6. 全局验收（学完这套体系之后）

能给组会讲清楚下面几条，就算「自学体系走通」（与根 README §8.2 一致）：

- [ ] 默画 AI 编译栈五层，每层点出仓库里对应的文档（foundations §1）  
- [ ] 70B → 128 卡：切什么、通什么、放哪里；并说明分片如何变成 IR 属性  
- [ ] 为什么不能一步降到 LLVM IR（结合 toy dialect 多层分工的例子）  
- [ ] IREE `linalg → flow → stream → hal` 每层固化什么  
- [ ] 一个算子从 ONNX 到某个 EP：中间有哪些决策点  
- [ ] 「融合 + tiling + 重计算」为何是核心杠杆（Roofline 口径即可；FA 是可选实证）  
- [ ] 六个研究问题均能说出价值、已有边界与自己的观点（不要求实现方案）  

四条动手硬门槛（入门线；没做等于没学）：

- [ ] `llvm-hello-compile` 跑通，能讲解 mem2reg 前后 IR  
- [ ] `mlir-toy-dialect` 跑通，能讲解 `--toy-to-low-convert` 与 `--toy-print-cost`  
- [ ] `tvm-fatbin-lab` 跑通（至少 TVM 轨），能讲解两份 matmul lower；有 CUDA 时再讲 fatbin dump  
- [ ] `onnx-delegate-lab` 跑通（至少 ONNX 轨），能讲解 EP 分区边界与 Partitioner tag 策略对比  

**以上四条只是 L0（跑通 + 能讲）**。要判定「真的会做」，继续走 [`checkpoints/`](./checkpoints/README.md)：每册至少完成两条 L2（自己加一个组件并跑出预期差异）。

---

## 7. 维护约定

- **讲机制必须带「示例精讲」**：任何一节只要在讲机制（Pass、Conversion、融合、划分、打包……），就要有一个 `#### 示例精讲：<主题>` 小节，形态固定为**最小输入 → 文本 IR/命令/产物 → 对象结构或调用链 → 逐步注释 → 并排对照表 → 一句自测**。参考基准：[`notes/llvm-mlir-pass-ir-unit.md`](./notes/llvm-mlir-pass-ir-unit.md)。  
  判据很简单：**如果读完还得靠额外笔记才懂，这一节就没写完**。工具输出若非逐字复制，必须标注「形态示意（以本地版本为准）」。
- **新论文笔记** → 放进 `paper-notes/`，按现有七节骨架写，并更新 [`paper-notes/README.md`](./paper-notes/README.md) 与根 README §9。  
- **新专题文档**（官方文档蒸馏）→ 放在 `docs/` 根下，在本页 §3.1 与根 README「专题学习文档」表各加一行；若引入新的横切概念，同步补 [`ai-compiler-foundations.md`](./ai-compiler-foundations.md) 对应章节与对照表。  
- **新增检验条目** → 放进 [`checkpoints/`](./checkpoints/README.md) 对应分册，必须凑齐六栏：检验什么 / 前置 / 资源 / 任务 / **先预测再动手** / 可执行的验收命令 + 机器可判定的通过标准 + 常见失败映射。缺「通过标准」的条目不收——那样又退回成口述题。  
- **动手项目升级**（例如 toy 打通到 linalg）→ 改项目自己的 README，并回写根 README 对应阶段的「动手验收」与本页阶段地图。  
- **优先级变化** → 只改根 README §0 总表，本页阶段顺序跟着总表走。
