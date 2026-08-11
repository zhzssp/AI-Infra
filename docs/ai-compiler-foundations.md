# AI 编译器前置核心概念：把工具串成知识体系

> **本文档的定位**
> - 这不是又一个项目说明书，而是**把仓库里所有学习文档与动手项目粘在一起的「公共词汇表」**。
> - 根目录 [`../README.md`](../README.md) 的目标是「算力网上大模型分布式运行基础设施」——需要同时会**分布式怎么切**、**异构后端怎么统一**、**运行时怎么决策**。LLVM / MLIR / IREE / TVM / ONNX / ExecuTorch / CUDA fatbin 各自讲透一块，但中间有一批**跨项目反复出现的概念**；缺了它们，你会觉得每本书都读懂了，却画不出一张总图。
> - 本文只讲这些**前置 / 横切**概念：是什么、为什么需要、在哪些项目里以什么名字出现。细节仍回各自学习文档。
>
> **怎么用**
> - **开局读一遍第 1～2 章**（总图 + 概念索引），建立坐标系。
> - **本文不是「全读」文档，也不是开局一次读完的文档**：第 2 章每条概念都标了**优先级**（★★ / ★ / ◆ / ○）与**它直接服务于老师指定的哪个工具/项目**；标 ○ 的属于宏观 AI-Infra 体系，知道名字即可。
> - **[§2.4 给出与根 README 周次表同步的概念路线](#24-与工具路线同步的概念学习路线)**：每一轮只补下一个工具真正要用到的那几节，概念比工具早半步、不早一整轮。
> - 学某个项目卡住时，用[第 10 章对照表](#第-10-章-概念--深入材料对照)跳到对应章节补概念，再回去读项目文档。
> - 不要试图在这里替代 MLIR Dialect Conversion 或 IREE HAL 的精读——那是各专题文档的事。
>
> **一句话读法**：如果只有一小时，读[第 1 章总图](#第-1-章-一张总图从模型到异构设备)、[§2.4 同步路线表](#24-与工具路线同步的概念学习路线)，然后**只**读该表 W0 行点到的那几节。

---

## 目录

- [第 1 章 一张总图：从模型到异构设备](#第-1-章-一张总图从模型到异构设备)
- [第 2 章 概念索引：按学习目标分组](#第-2-章-概念索引按学习目标分组)
  - [2.0 优先级标注怎么读](#20-优先级标注怎么读) · **[2.4 与工具路线同步的概念学习路线](#24-与工具路线同步的概念学习路线)** ←「哪一周学哪几条」看这里
- [第 3 章 图级抽象：计算图、算子、融合、划分](#第-3-章-图级抽象计算图算子融合划分)
- [第 4 章 调度与局部性：算法/调度分离、tiling、Roofline](#第-4-章-调度与局部性算法调度分离tilingroofline)
- [第 5 章 数据形态：tensor / buffer / layout / 量化 / 地址空间](#第-5-章-数据形态tensor--buffer--layout--量化--地址空间)
- [第 6 章 中间表示与变换：SSA、多层 IR、Pass、渐进 lowering](#第-6-章-中间表示与变换ssa多层-irpass渐进-lowering)
- [第 7 章 代码生成与变体：kernel、ISA 层级、多目标打包](#第-7-章-代码生成与变体kernelisa-层级多目标打包)
- [第 8 章 运行时抽象机：设备、队列、同步、AOT/JIT](#第-8-章-运行时抽象机设备队列同步aotjit)
- [第 9 章 分布式与编译的接缝：分片如何变成 IR](#第-9-章-分布式与编译的接缝分片如何变成-ir)
- [第 10 章 概念 → 深入材料对照](#第-10-章-概念--深入材料对照)
- [第 11 章 学习路径：什么时候读本文](#第-11-章-学习路径什么时候读本文)
- [附录：一页速查](#附录一页速查)

---

## 第 1 章 一张总图：从模型到异构设备

把仓库里所有材料压成**一张栈图**：最上面的「应用 / 框架」是**输入**（模型和并行策略从这里来），它下面才是**五层编译与执行栈**。每一层有自己的问题、自己的代表项目；**同一概念在不同层换名字，语义往往同构**。

> 后文说「五层栈」，指的就是下图中框架层以下的五层：**交换图 IR → 中端 → 执行规划 → 后端 IR/机器码 → 运行时**。  
> （不要和 [`paper-notes/01`](./paper-notes/01-efficient-training-distributed-infra.md) 里分布式训练的「六层结构」混淆——那是另一张图，讲的是集群侧分层。）

```
┌─ 应用 / 框架 ─────────────────────────────────────────────────┐
│  PyTorch / JAX / TensorFlow 模型；训练并行策略（DP/TP/PP…）     │
│  代表材料：01 分布式综述                                        │
│  （FA / PA 也落在这一层——是负载侧论据，不在五层栈内；见 §4.3/§9.4/§9.5）│
└────────────────────────────┬──────────────────────────────────┘
                             │ export / trace / 前端导入
                             ▼
┌─ 可交换图 IR（交换格式）──────────────────────────────────────┐
│  ONNX Graph · Torch ExportedProgram / Edge Dialect            │
│  问题：跨框架可移植；opset/版本；shape 推断                     │
│  代表材料：onnx-learning-guide · executorch-learning-guide    │
└────────────────────────────┬──────────────────────────────────┘
                             │ 图级优化 / 委托分区
                             ▼
┌─ 编译器中端（领域 IR + 调度）─────────────────────────────────┐
│  Relay/Relax / TOSA / StableHLO / linalg-on-tensors           │
│  融合 · layout · tiling · 代价模型 · 搜索                      │
│  代表材料：tvm-learning-guide · mlir-learning-guide · Halide  │
└────────────────────────────┬──────────────────────────────────┘
                             │ 渐进 lowering / Dialect Conversion
                             ▼
┌─ 执行规划 IR（何时算、何时搬、在哪跑）────────────────────────┐
│  IREE: flow（dispatch）→ stream（时间线）→ hal（设备对象）     │
│  或：ORT EP / ExecuTorch LoweredBackendModule                 │
│  代表材料：iree-learning-guide · onnx EP 章 · executorch 委托  │
└────────────────────────────┬──────────────────────────────────┘
                             │ codegen
                             ▼
┌─ 后端 IR → 机器码 ────────────────────────────────────────────┐
│  LLVM IR → SelectionDAG/MIR → 汇编                             │
│  NVPTX → PTX → SASS；fatbin / ExecutableVariant 多变体         │
│  代表材料：llvm-learning-guide · cuda-fatbin-learning-guide   │
└────────────────────────────┬──────────────────────────────────┘
                             │ 加载 / 派发
                             ▼
┌─ 运行时 ──────────────────────────────────────────────────────┐
│  HAL device/buffer/queue/fence · PackedFunc · EP 派发         │
│  集合通信 channel · KV Cache 分页 · 弹性迁移（开放问题）        │
└───────────────────────────────────────────────────────────────┘
```

**三个学习目标如何落在这张图上**：

| 学习目标（根 README §0） | 主要落在哪几层 | 关键横切概念（本文要补的） |
|--------------------------|----------------|---------------------------|
| ① 能表达分布式 | 应用层 + 图 IR +（未来）分片属性进编译器中端 | SPMD、sharding annotation、集合通信原语、状态表示 |
| ② 能适配异构后端 | 中端 → 执行规划 → 后端 IR → 运行时 | 委托/分区、layout、地址空间、多变体、HAL 抽象机 |
| ③ 能在运行时做决策 | 运行时 + 动态 shape / 分页 / 迁移 | AOT vs JIT、fence/timeline、配置组合、状态迁移粒度 |

> **读各专题文档之前，先记住一句话**：  
> **图决定「切成哪些算子」；调度决定「每个算子怎么访存/并行」；执行规划决定「何时提交、等谁」；后端决定「变成哪条机器指令」；运行时决定「在哪块硬件上真正发生」。**  
> 这正是五层各自负责的五个问题——混在一起，是学 AI 编译最常见的晕眩来源。

---

## 第 2 章 概念索引：按学习目标分组

下面每条都会在后文展开。这里先当目录用。

### 2.0 优先级标注怎么读

本文的概念**并不同等重要**。当前阶段的任务是「先学好老师指定的工具/项目」，所以每条都按**它对那些工具的作用**分了四档：

| 标记 | 含义 | 怎么处理 |
|------|------|----------|
| **★★** | **骨架**：老师指定内容的组织方式本身就是它；不懂就有一整块读不下去 | 该轮**必须吃透**（四根柱子分散在 W1～W3，见 §2.4） |
| **★** | **硬前提**：专题文档或动手项目里直接、反复用到 | 对应工具那一轮**开头就补** |
| **◆** | **强杠杆**：补了之后「为什么这样设计」立刻通；也是写研究问题观点的词汇 | 同轮**顺手补**，读一遍能复述即可 |
| **○** | **体系辅助**：宏观 AI-Infra 概念，与老师指定内容只是间接相关 | **只记名字 + 一句话结论**，实际项目里再补 |

> **一句话**：★★ 与 ★ 是「学不学得懂」的问题，◆ 是「理解得深不深」的问题，○ 是「知识体系完不完整」的问题——后者可以推迟。
>
> **优先级 ≠ 学习顺序**：优先级说的是「值不值得花时间」，顺序由 [§2.4](#24-与工具路线同步的概念学习路线) 的轮次决定。★★ 的分片标注在 W1，★ 的 SSA 在 W0——**先学 W0 的 ★，不是先学所有 ★★**。

### 2.1 几乎所有编译栈共用

> 「轮次」列 = 第一次补它的时机，对应根 [`README.md`](../README.md) §7.2 的 W0～W8；`Wx→Wy` 表示 Wx 先建立直觉、Wy 再坐实。完整安排见 [§2.4](#24-与工具路线同步的概念学习路线)。

| # | 概念 | 优先级 | 轮次 | 直接服务于（老师指定内容） | 一句话 | 深入见 |
|---|------|--------|------|---------------------------|--------|--------|
| 1 | **计算图 / DFG** | ★ | W0→W5 | ONNX `GraphProto` · TVM Relay · IREE 前端导入；`onnx-delegate-lab` 第一步就是构图 | 节点是算子，边是张量依赖 | §3.1 |
| 2 | **算子（op）语义分类** | ★ | W4 | TVM 必学第 1 条（四类融合规则） | injective / reduction / … 决定能不能融合 | §3.2 |
| 3 | **算子融合（fusion）** | ★ | W4 | TVM 图级优化；同时是「委托」的单后端原型 | 合并算子以减少中间结果写回 | §3.2 |
| 4 | **子图划分 / 委托** | **★★** | W3→W5 | ORT EP + ExecuTorch Partitioner + IREE `flow.dispatch`；整个 `onnx-delegate-lab`；**六个研究问题的源头** | 决定哪些节点交给哪个后端 | §3.3 |
| 5 | **算法 vs 调度** | ★ | W4 | TVM TE/schedule；**Halide 唯一要学的思想** | 「算什么」与「怎么算」分离 | §4.1 |
| 6 | **Tiling / 循环变换** | ★ | W4 | TVM schedule 原语（`tvm-fatbin-lab` 步骤 01）；MLIR linalg tiling | 把迭代空间切块以适配缓存与并行 | §4.2 |
| 7 | **Roofline / 算力–带宽瓶颈** | ◆ | W4 开头 | 给 TVM / Halide 提供动机：为什么值得花力气融合与 tiling | 判断优化该砍 FLOPs 还是砍访存 | §4.3 |
| 8 | **Tensor vs Buffer** | ★ | W2 | MLIR linalg + Bufferization（必学三章之一）；IREE `flow`(tensor) → `hal`(buffer) | 不可变 SSA 值 vs 可原地更新的内存 | §5.1 |
| 9 | **Layout** | ◆ | W4 | TVM layout 传播；EP/Partitioner 边界代价；研究问题② | 同一逻辑张量在内存中的排布 | §5.2 |
| 10 | **量化基础** | ○ | W5（只读结论） | 仅间接：只需知道「量化在图上是一等公民，且制造跨后端摩擦」。数值技巧根 README 已列入「先跳过」 | 低精度表示 + scale/zero-point | §5.3 |
| 11 | **地址空间 / 内存层次** | ◆ | W1 | 01 综述的显存账本与带宽域；解释「tiling 为何有效」；MLIR memref memory space | global/shared/local；HBM/SRAM/寄存器 | §5.4 |
| 12 | **SSA + CFG** | ★ | W0 | LLVM 全部；`llvm-hello-compile` 的 mem2reg 就是它；MLIR block argument | 单赋值 + 显式控制流（一切 IR 的底座） | §6.1 |
| 13 | **多层 IR / 渐进 lowering** | **★★** | W2 | **MLIR 主战场 + IREE 五层 dialect 的组织方式本身**；`mlir-toy-dialect` | 每层只固化一类决定，避免过早丢失信息 | §6.2 |
| 14 | **Pass / 分析 vs 变换** | ★ | W0→W2 | LLVM New PM · MLIR Pass/Dialect Conversion；两个动手项目都在写 Pass | 编译器优化的组织单位 | §6.3 |
| 15 | **代价模型** | ◆ | W2→W4 | `mlir-toy-dialect` 的 `ToyCostOpInterface`；AutoTVM/MetaSchedule 搜索；研究问题⑥ | 用估计代替穷举实测来做决策 | §6.4 |

### 2.2 对接异构后端

| # | 概念 | 优先级 | 轮次 | 直接服务于（老师指定内容） | 一句话 | 深入见 |
|---|------|--------|------|---------------------------|--------|--------|
| 16 | **Kernel / Launch / Grid** | ◆ | W3 | IREE kernel ABI 五件套；CUDA fatbin；TVM GPU codegen | 设备上可执行的计算单元与启动配置 | §7.1 |
| 17 | **虚拟 ISA vs 真实 ISA** | ◆ | W6 | `cuda-fatbin-learning-guide` 的核心（`compute_xx` vs `sm_xx`）；LLVM NVPTX | 可迁移的中间码 vs 芯片原生码 | §7.2 |
| 18 | **多变体打包** | ◆ | W3→W6 | IREE `ExecutableVariant` ≈ fatbin（同构对照）；研究问题⑥ | 一份产物带多个目标实现，运行时选 | §7.3 |
| 19 | **设备抽象机** | **★★** | W3 | **IREE HAL 第 4 章——老师点名要求的内容**；也是算力网设备抽象的词汇表 | 编译器眼中的「硬件能力接口」 | §8.1 |
| 20 | **异步执行与同步原语** | **★★** | W3 | IREE timeline semaphore——根 README 标注的**最硬也最关键**知识点 | stream / event / fence / timeline semaphore | §8.2 |
| 21 | **AOT vs JIT** | ◆ | W5 | IREE / ORT / ExecuTorch 三者「决策时刻」差异；研究问题③ | 编译期定死 vs 运行期再特化 | §8.3 |

### 2.3 对接分布式与大模型负载

> **提醒**：这一组容易被当成「宏观 AI-Infra」而推迟，但 [01 分布式综述](./paper-notes/01-efficient-training-distributed-infra.md) 本身就是老师指定的 **P0 正题**，所以 §9.1–9.3 属于必补；只有 §9.4–9.5（来自自选补充的两篇论文）可以推迟。

| # | 概念 | 优先级 | 轮次 | 直接服务于（老师指定内容） | 一句话 | 深入见 |
|---|------|--------|------|---------------------------|--------|--------|
| 22 | **并行策略四要素** | ◆ | W1 | 01 综述的复习锚点（是 checklist，不是新概念） | 切什么 / 通什么 / 放哪 / 代价 | §9.1 |
| 23 | **SPMD + 分片标注** | **★★** | W1 | 01 综述标注的「★最重要」；**「可编译的分布式」与 MLIR/IREE 的接缝** | 一份程序 + 每张量如何切到 mesh | §9.2 |
| 24 | **集合通信原语** | ◆ | W1 | 01 综述；IREE HAL `channel`（§4.8） | AllReduce / AllGather / ReduceScatter / AllToAll | §9.3 |
| 25 | **IO 感知算法** | ○ | W4 可选 | 仅服务自选补充的 FlashAttention；**结论在 §9.4 读完即可，不必读原文** | 优化目标从 FLOPs 转向 HBM 访问次数 | §9.4 |
| 26 | **分页状态 / 块化迁移** | ○ | W7 可选 | 仅服务自选补充的 PagedAttention；写研究问题③④观点时再读 | 大状态按块管理才能低成本搬 | §9.5 |

### 2.4 与工具路线同步的概念学习路线

**这是本文最重要的一张表。** 上面 26 条概念**不是一次学完的**——它们按批次挂在根 [`README.md`](../README.md) §7.2 的周次表上，每一轮只补「下一个工具真正要用到的那几条」。

> **一句话原则**：**概念永远比工具早半步，不早一整轮。** 每轮开头花半天到一天补概念，然后立刻进工具/动手项目验证；下一轮再补下一批。
>
> 注意四根 **★★ 骨架**（渐进 lowering、划分/委托、设备抽象机+同步、分片标注）**是分散在第 1～3 轮的**，不要试图在开局一次补完——它们各自要靠对应的工具才能坐实。

```
工具轨（README §7.2）      概念轨（本文，每轮只补下面这几节）
─────────────────────      ──────────────────────────────────
W0  llvm-hello-compile  ◄─ §1 总图 · §3.1 直觉 · §6.1 SSA · §6.3 Pass
W1  分布式综述          ◄─ §9.1 · §9.2 ★★ · §9.3 · §5.4
W2  MLIR / toy-dialect  ◄─ §6.2 ★★ · §5.1 · §6.4（浅）
W3  IREE + HAL          ◄─ §8.1 ★★ · §8.2 ★★ · §3.3 ★★（引入）· §7.1 · §7.3（引入）
W4  TVM / tvm-lab       ◄─ §3.2 · §4.1 · §4.2 · §4.3 · §5.2 ·（§6.4 深化）
W5  ONNX/ET / onnx-lab  ◄─ §3.3 深化（四类边界代价）· §8.3 · §3.1 深化 · §5.3（结论）
W6  P2 回填             ◄─ §7.2 ·（§7.3 深化：fatbin）
W7–W8 研究问题收束      ◄─ §3.4 选型 · §9.5 ·（回看 §3.3 · §5.2 · §8.3 · §9）
```

| 轮次 | 同期在学的工具/项目 | 本轮要补的概念 | 为什么**这时候**补 | 补完能回答 |
|------|--------------------|----------------|-------------------|-----------|
| **W0**<br>开局 + LLVM | `llvm-hello-compile` | §1 五层栈总图<br>§3.1 计算图（只要直觉）<br>**§6.1 SSA + CFG** ★<br>**§6.3 Pass / 分析 vs 变换** ★ | 这两条在 `mem2reg` 与 `CountIR`/`InjectLogging` 里**当场看得见**，是最便宜的入门 | `phi` 从哪来；「优化 = 一串可插拔的 Pass」 |
| **W1**<br>分布式 | 01 分布式综述 | §9.1 并行四要素 ◆<br>**§9.2 SPMD + 分片标注** ★★（柱子④）<br>§9.3 集合通信原语 ◆<br>§5.4 地址空间 / 内存层次 ◆ | 综述的 16Φ 账本与通信重叠**必须**有带宽层级直觉才读得动；分片标注是后面 MLIR/IREE 的接缝 | 70B 怎么切到 128 卡；策略如何变成 IR 属性 |
| **W2**<br>MLIR | `mlir-toy-dialect` | **§6.2 多层 IR / 渐进 lowering** ★★（柱子①）<br>§5.1 Tensor vs Buffer ★<br>§6.4 代价模型 ◆（浅层）<br>（§6.3 Pass 深化到 Dialect Conversion） | toy→low 那一刀就是渐进 lowering 的实物；`ToyCostOpInterface` 正好是代价模型的教学版 | 为什么不能一步降到 LLVM IR；tensor/buffer 各适合什么优化 |
| **W3**<br>IREE | IREE + HAL | **§8.1 设备抽象机** ★★ + **§8.2 异步与同步** ★★（柱子③）<br>**§3.3 划分 / 委托** ★★ 引入（柱子②上半）<br>§7.1 Kernel / Launch ◆<br>§7.3 多变体打包 ◆ 引入 | HAL 第 4 章直接就是 §8；`flow.dispatch` 是「编译期划分」，先在这里见一次；kernel ABI 五件套要 §7.1 打底 | `linalg→flow→stream→hal` 各固化什么；为什么 `CUevent` 不够 |
| **W4**<br>TVM | `tvm-fatbin-lab` TVM 轨 | §3.2 算子分类 + 融合 ★<br>§4.1 算法 vs 调度 ★<br>§4.2 Tiling / 循环变换 ★<br>§4.3 Roofline ◆<br>§5.2 Layout ◆<br>（§6.4 深化到 AutoTVM 搜索） | 先有 Roofline 才知道为什么值得 tiling，再学 schedule 原语才不是背命令；layout 是 TVM 必学第 2 条 | 四类融合；对着 `tvm.lower` 讲原语；带宽墙 vs 算力墙 |
| **W5**<br>委托 | `onnx-delegate-lab` | **§3.3 深化：四类边界代价** ★★（柱子②下半）<br>§8.3 AOT vs JIT ◆<br>§3.1 深化：opset / shape 推断 ★<br>§5.3 量化 ○（只看结论） | 此时已见过 IREE 的编译期划分，正好对照 ORT 的 session 期、ET 的导出期——「决策时刻」差异靠 §8.3 说清 | 边界为什么贵；三种划分归属的差别；研究问题①②在问什么 |
| **W6**<br>P2 回填 | Halide / Glow / CUDA fatbin | §7.2 虚拟 ISA vs 真实 ISA ◆<br>（§7.3 深化：fatbin 实物）<br>（§4.1 回看 Halide 源头） | `compute_xx` vs `sm_xx` 只有在动手 dump 时才记得住；此时回看 §7.3 就能把 fatbin ≈ ExecutableVariant 钉死 | 多变体打包解决什么；PTX 与 SASS 各自代价 |
| **W7–W8**<br>收束 | §6 六个研究问题 | §3.4 四条工业栈选型 ◆<br>§9.5 分页状态 / 块化迁移 ○<br>（回看 §3.3 · §5.2 · §8.3 · §9） | 四条栈都学过才有判断力去填选型表；这一轮的概念是**写观点用的词汇**，不是新知识 | 每条栈把决策权交给了谁；哪些决定必须留到运行时 |

**同步纪律（三条）**

1. **不预习下一轮的概念**——读了没有工具坐实，两周后照样忘。
2. **每轮结束回填一句话**：用自己的话把本轮概念与刚跑通的产物连起来（例：「`flow.dispatch` 就是 §3.3 说的划分，只是决策者是编译器」）。
3. **卡住时不要往前学，往回查**——用[第 10 章对照表](#第-10-章-概念--深入材料对照)定位到本轮或更早轮次的那一节。

**始终可以跳过的（○）**：量化数值技巧（§5.3 只看结论）、IO 感知（§9.4，跟 FA 一起可选）、分页状态（§9.5，收束时可选），以及 [§11.2](#112-必学-vs-可推迟) 列出的其余可推迟项。

---

## 第 3 章 图级抽象：计算图、算子、融合、划分

### 3.1 计算图（Computational Graph / DFG）

**是什么**：深度学习模型在编译器里的第一公民表示——**有向（通常无环）图**，节点是算子（MatMul、Softmax、LayerNorm…），边是张量（激活、权重、梯度）。

**为什么重要**：

- 图上的边天然表达**数据依赖**，这是融合、并行、内存规划、子图划分的共同输入。
- 与通用程序 CFG 不同：DL 图的节点粒度是「一次张量计算」，shape 常常（训练配置固定后）**静态可知**，因而能做更激进的编译期决策（静态内存规划、常量折叠）。

**在各项目中的名字**：

| 项目 | 叫法 |
|------|------|
| ONNX | `GraphProto` + `NodeProto` |
| PyTorch / ExecuTorch | `ExportedProgram` / FX graph / Edge Dialect |
| TVM | Relay / Relax 函数图 |
| IREE 前端 | StableHLO / TOSA / Torch 导入后的 MLIR 模块 |
| ORT | 会话图 + EP 分区后的子图 |

**必须建立的直觉**：图优化改的是**拓扑与节点粒度**；它不直接生成机器码。机器码是更下层的事。

### 3.2 算子语义分类与融合（Fusion）

**是什么**：把多个图节点合并成一个更大的可执行单元（通常对应一个 kernel），使中间张量**不必写回全局内存再读出**。

**经典四类**（TVM 的分类，已成社区共同语言）：

| 类别 | 含义 | 融合规则直觉 |
|------|------|--------------|
| **injective** | 输出每个元素只依赖输入对应位置（如 `add`、`relu`） | injective 彼此可融；可被 reduction / complex 的输入或输出侧吸收 |
| **reduction** | 沿某维聚合（如 `sum`、`softmax` 的归约部分） | 可融合输入侧的 injective |
| **complex-out-fusable** | 计算重，但输出可接 elementwise（如 `conv2d`） | 可融合输出侧 elementwise（conv+bias+relu） |
| **opaque** | 难以分析（如 `sort`） | 通常不融 |

**为什么这是「子图划分」的原型**：融合决策 =「这些节点要不要落在**同一个**可执行单元里」。TVM 用规则自动做；ORT EP / ExecuTorch Partitioner / IREE `flow.dispatch` 是同一问题在「多后端」场景下的推广——只不过边界从「融不融」变成了「给不给这个后端」。

详见 [`tvm-learning-guide.md`](./tvm-learning-guide.md) §2.2。

#### 示例精讲：conv → bias_add → relu 融合前后的访存账

**最小具体输入**：`y = relu(bias_add(conv2d(x, w), b))`。输入 `x`：`1×64×56×56` f32；卷积核 `w`：`64×64×3×3`，pad 1、stride 1，故输出仍是 `1×64×56×56`；`b`：长度 64。

先把一份激活张量的大小算出来，后面全部以它为单位：

```text
S = 64 × 56 × 56 × 4 B = 200704 × 4 B = 802,816 B ≈ 0.77 MiB
W（权重）= 64 × 64 × 3 × 3 × 4 B = 147,456 B ≈ 0.14 MiB
FLOPs = 2 × 200704 × (64×3×3) = 2 × 200704 × 576 ≈ 2.31 × 10⁸
```

**融合前：三个节点、两个中间张量落地**

```text
%t1 = conv2d(%x, %w)        // complex-out-fusable
%t2 = bias_add(%t1, %b)     // injective
%y  = relu(%t2)             // injective
```

**融合后：一个 kernel，中间结果不出寄存器**

```text
%y = fused_conv2d_bias_relu(%x, %w, %b)

// 内部循环体（伪代码）：
for each (n, f, oh, ow):
    acc = 0f
    for kh, kw, ci:  acc += x[n, ci, oh+kh-1, ow+kw-1] * w[f, ci, kh, kw]
    acc = acc + b[f]            // ← bias_add 被吸收到输出侧
    y[n, f, oh, ow] = max(acc, 0f)   // ← relu 也被吸收；acc 从未离开寄存器
```

能这么做的**依据就是四类分类**：conv 是 complex-out-fusable（可吸收输出侧 elementwise），bias_add 与 relu 都是 injective（可被吸收）。若中间夹一个 opaque 算子，这条链就断在那里。

**访存计数**（悲观口径：假设中间张量每次都往返 HBM，不计缓存命中）

| | kernel 数 | 中间张量分配 | 中间张量访存 | 含输入/权重的总 bytes | 算术强度 |
|--|-----------|--------------|--------------|----------------------|----------|
| 融合前 | 3 | 2（`%t1`、`%t2`） | 写 S + 读写 2S + 读写 2S = **5S ≈ 3.83 MiB** | S + W + 5S = 6S + W ≈ 4.96 MB | 2.31×10⁸ / 4.96×10⁶ ≈ **47 FLOP/B** |
| 融合后 | 1 | 0 | 只写最终输出 = **1S ≈ 0.77 MiB** | S + W + S = 2S + W ≈ 1.75 MB | 2.31×10⁸ / 1.75×10⁶ ≈ **132 FLOP/B** |
| 差 | −2 | −2 | **省 4S ≈ 3.06 MiB** | 约 1/2.8 | **约 2.8 倍** |

**呼应 Roofline**（[§4.3](#43-roofline先问瓶颈在哪)）：分子 FLOPs **一次都没少**，减少的全是分母 bytes——所以融合是把这个算子在 Roofline 上**向右推**，从更靠近带宽屋檐的位置推向算力屋顶。如果这台机器的机器平衡点（peak FLOP/s ÷ peak B/s）落在 47 与 132 之间，那么这一次融合正好把它从「带宽受限」搬到「算力受限」——收益最大的一格。顺带省掉的还有 2 次 kernel launch 与 2 块临时显存。

> **自测**：如果 `%t1` 还被另一个分支消费（残差连接），上表「省 4S」里哪一部分省不掉？

### 3.3 子图划分与委托（Partition / Delegate）

**是什么**：在一张完整计算图上，标出「由后端 A 执行」的连通子图，其余留给默认运行时（CPU EP、PyTorch 解释路径等）。边界处插入**数据交接**（可能伴随 layout/dtype/设备转换）。

**三种工业答案（同构对照）**：

| 系统 | 机制 | 谁声明能力 | 编译结果 |
|------|------|------------|----------|
| **ONNX Runtime EP** | `GetCapability` → `Compile` | EP | 分区后的可执行子图 |
| **ExecuTorch** | `Partitioner` 打 `delegation_tag` → `to_backend` | Partitioner + Backend | `LoweredBackendModule` + blob |
| **IREE** | 编译期形成 `flow.dispatch` / `hal.executable` | 编译流水线 + target | `.vmfb` 内嵌多 variant |

**分区边界的四种代价**（研究问题①②的物理根源）：

1. **数据拷贝**（host↔device 或 device↔device）  
2. **Layout / dtype 转换**  
3. **内存空间切换**（如 global ↔ shared 的管理边界）  
4. **同步点**（必须等上一子图完成才能开下一子图）

边界越多，这四项叠加越狠——所以「能划多大就划多大」和「划给最快的后端」经常冲突，需要代价模型。

详见 [`onnx-learning-guide.md`](./onnx-learning-guide.md) §7、[`executorch-learning-guide.md`](./executorch-learning-guide.md) §4–5、[`iree-learning-guide.md`](./iree-learning-guide.md) 第 2 章。

#### 示例精讲：5 节点链切一刀，四类代价各落在哪条边

**最小具体输入**：一条 5 节点链。后端 X（某加速器）声明只支持 `conv2d` 与 `relu`；`topk` / `add` / `softmax` 留给默认 CPU 运行时。

```text
// 划分前的伪 IR（括号里是能力归属）
%t1 = conv2d(%x, %w)      // A —— 后端 X 支持
%t2 = relu(%t1)           // B —— 后端 X 支持
%t3 = topk(%t2)           // C —— X 不支持   ← 切口只能落在 B|C 之间
%t4 = add(%t3, %bias)     // D —— CPU
%y  = softmax(%t4)        // E —— CPU
```

给每条边编号，方便下面指认：

```text
 %x ──e0──▶ A(conv2d) ──e1──▶ B(relu) ──e2──▶ C(topk) ──e3──▶ D(add) ──e4──▶ E(softmax) ──▶ %y
            └────────── 后端 X 的子图 ─────────┘  ▲  └──────── 默认 CPU 运行时 ────────┘
                                                  │
                                          e2 = 唯一的边界边
```

**划分后的伪 IR**

```text
%t2 = call @subgraph_X(%x, %w)   // 一次委托调用；内部是 conv2d → relu（还可再融合）
// ══════════════ 切口：所有边界代价都压在 e2 这一条边上 ══════════════
%t3 = topk(%t2)                  // 以下留在默认运行时
%t4 = add(%t3, %bias)
%y  = softmax(%t4)
```

注意 `e1` 被**吞进子图内部**了：它不再是边界，反而成了可融合的候选（对照 [§3.2](#32-算子语义分类与融合fusion)）。这就是「边界越少越好」的机械含义——**每条被吞进去的边省掉一整套下面四项代价**。

**四类代价 ↔ 图上的位置**

| 代价 | 落在哪 | 在这个例子里具体是什么 | 各系统里的名字 |
|------|--------|------------------------|----------------|
| ① 数据拷贝 | **e2 这条边上** | `%t2`（子图输出）从 X 的显存拷回 CPU 可读内存；不共享地址空间时无法避免 | ORT 的 EP 边界 copy、ET 的 delegate 输入/输出搬运 |
| ② Layout / dtype 转换 | **e2 上，表现为新增算子** | X 内部按 NHWC+fp16 算，CPU 的 `topk` 期待 NCHW+fp32 → e2 实际展开成 `relu → cast → layout_transform → topk` | `layout_transform`、`QuantizeLinear`/`Dequantize` 对 |
| ③ 内存空间切换 | **e2 两端 buffer 的归属** | 左端由 X 的分配器管（device global），右端要在 host 分配器里另拿一块；跨界的是「谁管这块内存」，不是数据内容 | 地址空间（[§5.4](#54-地址空间与内存层次)）、HAL allocator |
| ④ 同步点 | **e2 上的一条时间边** | `topk` 读 `%t2` 之前必须等整个 `@subgraph_X` 完成并 signal——这条依赖不产生数据，只产生等待 | fence / event / timeline semaphore（[§8.2](#82-异步与同步)） |

**要记住的两点**

1. 前三项是**空间**代价（多一次搬、多一个算子、多一次分配），第四项是**时间**代价（流水线被截断，无法与相邻工作重叠）。前三项能靠 layout 协商压小，第四项只能靠**减少边界数量**或让边界两侧异步化。
2. 代价是按**边界数**收费，不是按节点数。所以「把 X 能吃的节点尽量连成一片」通常比「把每个 X 能吃的节点都交给 X」更划算——后者会切出一堆碎片，每片都要付一遍 ①②③④。

> **自测**：如果后端 X 其实也支持 `add` 与 `softmax`（只有 `topk` 不支持），边界会变成几条？四类代价各要付几遍？

### 3.4 四条工业栈怎么选（学完各专题后回看这张表）

仓库里的四份「图级 / 部署级」专题文档讲的是**同一个问题的四种工业答案**。它们不是替代关系，选型取决于**你把决策权交给谁**：

| 栈 | 划分/优化的决策者 | 产物与运行时 | 适合的场景 | 对算力网目标缺什么 |
|----|------------------|--------------|-----------|-------------------|
| **ONNX Runtime EP**<br>[`onnx`](./onnx-learning-guide.md) · 动手 [`../onnx-delegate-lab/`](../onnx-delegate-lab/) | 各 EP 自己声明能吃什么（`GetCapability`），运行时按贪心优先级派发 | 单一 `.onnx` + 多 EP 插件，进程内派发 | 已有 ONNX 模型、要快速接一个新加速器 | 划分是贪心的，不做「算得快 vs 边界贵」的全局折中 |
| **ExecuTorch 委托**<br>[`executorch`](./executorch-learning-guide.md) · 同上 lab | 你写的 `Partitioner`（编译期打 tag） | `.pte` + delegate blob，极薄运行时 | 端侧 / 受限设备，要求确定性与小体积 | 不提供分布式调度，也不管跨节点 |
| **IREE**<br>[`iree`](./iree-learning-guide.md) | 编译流水线自己形成 `flow.dispatch`，并把时序与设备归属固化进 stream/hal | `.vmfb`（内嵌多 variant）+ HAL 运行时 | **多后端统一抽象**，要控制「何时搬、等谁、在哪跑」 | 单进程多设备为主；跨节点编排、容错、并行策略要自己补 |
| **TVM**<br>[`tvm`](./tvm-learning-guide.md) · 动手 [`../tvm-fatbin-lab/`](../tvm-fatbin-lab/) | 融合规则 + 搜索（AutoTVM/MetaSchedule）决定单个 kernel 怎么算得快 | 编译好的算子库 + PackedFunc 运行时 | 单算子/子图**性能压榨**与自动调优 | 层次比前三者低一层，不解决设备抽象与执行规划 |

**读法**：前三行是**「谁来划分」**的三种答案（同一决策，不同归属）；第四行是**「划完之后怎么算得快」**，与前三者正交，可以叠加使用。

**对本仓库的取舍**（与根 [`README.md`](../README.md) §0 优先级一致）：主线选 **IREE**，因为只有它把「执行规划」显式建成 IR 层；ONNX/ExecuTorch 提供**划分接口的工业对照**，用来理解研究问题①②；TVM 提供**代价模型与搜索**的方法论，用来回应研究问题⑥。

---

## 第 4 章 调度与局部性：算法/调度分离、tiling、Roofline

### 4.1 算法与调度分离（Algorithm / Schedule）

**是什么**（Halide 贡献的核心思想，TVM TE 直接继承）：

- **算法（algorithm / compute）**：用数学定义「输出每个元素怎么从输入算出来」，**不规定**循环顺序、并行、缓存。  
- **调度（schedule）**：在算法之上叠加变换（tile、reorder、cache_write…），生成具体循环与内存访问。

**为什么重要**：同一算法可以对应成千上万种实现；把二者耦合在手写 CUDA 里，就无法系统搜索，也无法跨硬件复用「算什么」的定义。

**在栈中的位置**：这是**算子级**优化的语言，不是图级融合。图级决定「这是一个 fused op」；调度决定「这个 fused op 内部循环长什么样」。

详见 [`paper-notes/04-halide.md`](./paper-notes/04-halide.md)、[`tvm-learning-guide.md`](./tvm-learning-guide.md) §3。

#### 示例精讲：一个 matmul 算法 + 两个 schedule

**最小具体输入**：方阵 matmul，`C[i,j] = Σ_k A[i,k] * B[k,j]`。下面**算法一个字都不改**，只换 schedule。（对应动手脚本 [`../tvm-fatbin-lab/tvm_lab/01_te_matmul_schedules.py`](../tvm-fatbin-lab/tvm_lab/01_te_matmul_schedules.py)，那里 `N=64`、`tile=16`。）

**算法（只说「算什么」）**

```python
A = te.placeholder((N, N), name="A", dtype="float32")
B = te.placeholder((N, N), name="B", dtype="float32")
k = te.reduce_axis((0, N), name="k")
C = te.compute((N, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
# 这四行里没有任何「循环顺序 / 分块 / 并行 / 缓存」的信息
```

**Schedule 0（默认）**

```python
s = te.create_schedule(C.op)     # 什么也不加
```

**Schedule 1（tile + reorder）**

```python
s = te.create_schedule(C.op)
io, ii = s[C].split(C.op.axis[0], factor=16)   # i → (i.outer, i.inner)
jo, ji = s[C].split(C.op.axis[1], factor=16)   # j → (j.outer, j.inner)
s[C].reorder(io, jo, ii, ji)                   # 等价于 s[C].tile(i, j, 16, 16)
```

**两份循环 nest 并排**

> 输出形态示意（以本地工具版本为准）：`tvm.lower(s, [A, B, C], simple_mode=True)` 的结构，索引算式已简写。

```text
Schedule 0（默认）                        Schedule 1（tile 16×16 + reorder）
─────────────────────────────────         ─────────────────────────────────────
for (i, 0, N)                             for (i.outer, 0, N/16)
  for (j, 0, N)                             for (j.outer, 0, N/16)
    C[i, j] = 0f                              for (i.inner, 0, 16)
    for (k, 0, N)                               for (j.inner, 0, 16)
      C[i, j] += A[i, k] * B[k, j]                 C[..] = 0f
                                                   for (k, 0, N)
                                                     C[..] += A[..] * B[..]
```

**局部性发生了什么**

1. Schedule 0：最内两层是 `j` 和 `k`。固定一个 `i`，`j` 循环要把**整个 B**扫一遍；`i` 前进一格，B 又得从头扫一遍。若 B 放不进快速存储，B 就被重读 `N` 次。
2. Schedule 1：最内两层变成 `i.inner`/`j.inner`（各 16）。一个 `(i.outer, j.outer)` 块内只碰 A 的 16 行、B 的 16 列，**工作集从「整个 B」缩到「两条 16×N 的带」**，而这两条带在块内被复用 16 次。
3. 代价：块循环数变多，`k` 仍然是最内的完整归约，所以**冗余计算为零、并行粒度变粗**——这正是 §4.2 权衡三角里「局部性 ↑ / 并行度形态改变」的那一格。

| | 算法（compute） | 调度（schedule） |
|--|-----------------|------------------|
| 回答的问题 | 输出每个元素等于什么 | 按什么顺序算、暂存在哪、谁并行 |
| 上面哪几行 | `te.placeholder` / `te.reduce_axis` / `te.compute` | `split` / `reorder`（以及 §4.2 的 `cache_read` 等） |
| 换硬件时 | 基本不动 | 几乎全部要重写或重搜 |
| 改错的后果 | **数值结果错** | 只是慢（合法变换保语义） |

> 这就是「可搜索」的来源：算法固定 ⇒ 所有 schedule 都在实现同一个数学定义 ⇒ 搜索器基本只按性能排序。（注意例外：一旦某个 schedule 改动了浮点**归约顺序**（split `k`、`rfactor`、带重结合的向量化），结果不再逐位相同——语义仍对，但数值会有末位差异。）

> **自测**：Schedule 1 里把 `reorder` 改成 `(io, ii, jo, ji)`，最内层的工作集会变大还是变小？

### 4.2 Tiling 与循环变换族

**Tiling（分块）**：把迭代空间切成小块，使一块数据在被踢出快速存储器之前被充分复用。

常见原语（名字来自 Halide/TVM，思想通用）：

| 原语 | 作用 |
|------|------|
| `split` / `tile` | 把一个循环变成两层（外层块、内层块内） |
| `reorder` | 改变循环嵌套顺序（决定遍历与局部性） |
| `fuse` | 合并相邻循环（便于绑定到线程维） |
| `parallel` / `bind` | 映射到 CPU 线程或 GPU block/thread |
| `vectorize` / `unroll` | SIMD 与展开 |
| `cache_read` / `cache_write` | 在更快存储层建立暂存缓冲 |
| `compute_at` | 把生产者计算嵌进消费者循环的某层（计算粒度） |

**权衡三角**（Halide 反复强调）：**局部性 ↔ 并行度 ↔ 冗余计算（+ 内存占用）**。拉高一个往往压低另一个——没有免费午餐，只有搜索或启发式。

#### 示例精讲：`split` 与 `cache_read` 各改了什么

沿用 [§4.1](#41-算法与调度分离algorithm--schedule) 的 matmul，但把规模放大到 `N = 1024`（f32），这样访存账才算得有意义。

**第一步：`split` —— 一个循环变两层**

```mlir
// 前：一层循环，迭代空间 1024
scf.for %i = %c0 to %c1024 step %c1 {
  use(%i)
}

// 后：split(factor=16) —— 64 个块 × 块内 16
scf.for %io = %c0 to %c64 step %c1 {
  scf.for %ii = %c0 to %c16 step %c1 {
    %t = arith.muli %io, %c16 : index
    %i = arith.addi %t, %ii : index        // 还原出原来的 %i
    use(%i)
  }
}
```

`split` 本身**不改变任何一次计算**，只是把迭代空间重新参数化——它的价值全在于「切出来的两层可以分别被 `reorder` / `bind` / `vectorize` 处置」。注意前提：16 整除 1024；否则要留边界块或加谓词。

**第二步：`cache_read` —— 给 B 建一层显式暂存**

```python
BB = s.cache_read(B, "shared", [C])   # 把 B 的 tile 读进快速存储
s[BB].compute_at(s[C], jo)            # 每个 (i.outer, j.outer) 块开头搬一次
```

```text
// 前：每次乘法都从 B 的原位置读（命不命中缓存靠运气）
for (i.outer) for (j.outer) for (i.inner) for (j.inner) for (k)
    C[..] += A[i, k] * B[k, j]

// 后：块入口多了一个搬运阶段，计算阶段只读暂存
for (i.outer) for (j.outer) {
    for (k) for (j.inner)                       // 搬运：N × 16 个元素
        B.shared[k, j.inner] = B[k, j.outer*16 + j.inner]
    // ── 这里需要一次 barrier：搬完才能算 ──
    for (i.inner) for (j.inner) for (k)
        C[..] += A[i, k] * B.shared[k, j.inner]  // 每个搬进来的元素被用 16 次
}
```

**复用次数 / 并行度对照**（`N=1024`，tile `16×16`，f32；假设快速存储放得下 16 行 A 与 16 列 B 的带，但放不下整个 B）

| 版本 | 每个 B 元素平均被用几次 | 从下一级存储读入的元素数 | 可并行单元数 | 额外快速存储 |
|------|------------------------|--------------------------|--------------|--------------|
| 默认 `i,j,k` | 1（`i` 前进就得重读整个 B） | ≈ N³ = 1.07×10⁹（≈ 4.3 GB） | `i`：1024 路 | 0 |
| `split`+`reorder`（tile 16×16） | 16（块内被 `i.inner` 复用） | ≈ 2N³/16 = 1.34×10⁸（≈ 0.54 GB） | `(io,jo)`：64×64 = 4096 块 | 0（靠缓存自动命中） |
| 再加 `cache_read(B→shared)` | 16（同上，但**由程序保证**） | 同上 | 同上，块内可再绑 256 线程 | B 的带 = 1024×16×4 B = 64 KiB |

**三条读法**

1. 访存量降到约 1/8，FLOPs 一点没变——所以这是一次**纯粹的带宽侧优化**，Roofline 上是向右移（见 [§4.3](#43-roofline先问瓶颈在哪)）。
2. `split` 只提供结构，真正省访存的是 `reorder` 之后**最内层工作集变小**；`cache_read` 把「命中靠缓存运气」变成「命中由代码保证」，代价是多一次搬运 + 一次 barrier。
3. 上表最后一格 64 KiB 已经超过常见 GPU 每 block 的 shared memory 预算——这就是真实 schedule 还要**再 split `k`**（k-tile）的原因：三层 tiling 不是炫技，是被容量逼出来的。

> **自测**：把 tile 从 16 改成 64，上表哪一列会先撞墙？

### 4.3 Roofline：先问瓶颈在哪

**是什么**：把程序钉在「算力上限」与「带宽上限」两条线构成的屋顶上——

- 算术强度（Arithmetic Intensity）= FLOPs / Bytes  
- 强度低 → **带宽瓶颈**（优化应减少访存、提高复用、融合）  
- 强度高 → **算力瓶颈**（优化应向量化、用 Tensor Core、减少指令）

**为什么 AI 编译必提**：现代 GPU 上 Transformer 注意力等内核经常是**访存墙**，不是算力墙。FlashAttention 的方法论本质就是：用 tiling + 融合 + 重计算，把 HBM 访问次数打下来。不建立 Roofline 直觉，会误把「多算一点 FLOPs」当成优化。

想要实证再翻 [`paper-notes/08-flash-attention.md`](./paper-notes/08-flash-attention.md)（**自选补充**：本节结论已够用，原文只是把它坐实）。

---

## 第 5 章 数据形态：tensor / buffer / layout / 量化 / 地址空间

### 5.1 Tensor（值）vs Buffer（存储）

| | Tensor（值语义） | Buffer / MemRef（存储语义） |
|--|------------------|-----------------------------|
| 可变性 | SSA 值，不可变；「更新」= 产生新值 | 内存位置，可被 store 原地改 |
| 别名 | 无别名（SSA） | 可能别名，分析困难 |
| 适合的优化 | 融合、tiling、shape 推导 | 内存复用、向量 load/store、与运行时对接 |
| 典型 IR | `tensor<…>`、Relay Tensor | `memref<…>`、HAL buffer |

**Bufferization**：把 tensor 程序变成 buffer 程序的过程。One-Shot Bufferize 的目标是**尽量原地复用**（少分配、少拷贝），依赖 destination-passing style（结果写进已有的 `outs` 缓冲）。

> 经验法则：**图级与高维变换尽量留在 tensor 世界；靠近硬件再 bufferize。** 过早 bufferize = 过早引入别名，后面融合/变换变难。

详见 [`mlir-learning-guide.md`](./mlir-learning-guide.md) §8。

#### 示例精讲：同一个 `c = a + b` 的 tensor 版与 memref 版

**最小具体输入**：`c = a + b`，紧接着 `d = relu(c)`，长度 8 的 f32 向量。两段 IR 语义完全一样，唯一的区别是「`c` 是一个**值**，还是一块**内存**」。

**版本 A：tensor（值语义）**

```mlir
func.func @add_relu_tensor(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
  %z  = arith.constant 0.0 : f32
  %e0 = tensor.empty() : tensor<8xf32>
  %c  = linalg.add ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
                   outs(%e0 : tensor<8xf32>) -> tensor<8xf32>   // %c 是新的 SSA 值
  %e1 = tensor.empty() : tensor<8xf32>
  %d  = linalg.generic
          {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
           iterator_types = ["parallel"]}
          ins(%c : tensor<8xf32>) outs(%e1 : tensor<8xf32>) {
  ^bb0(%x: f32, %o: f32):
    %r = arith.maximumf %x, %z : f32
    linalg.yield %r : f32
  } -> tensor<8xf32>
  return %d : tensor<8xf32>
}
```

**版本 B：memref（存储语义）** —— bufferize 之后的同一段计算

```mlir
func.func @add_relu_memref(%a: memref<8xf32>, %b: memref<8xf32>, %c: memref<8xf32>) {
  %z  = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {                  // 循环 1：add
    %x = memref.load %a[%i] : memref<8xf32>
    %y = memref.load %b[%i] : memref<8xf32>
    %s = arith.addf %x, %y : f32
    memref.store %s, %c[%i] : memref<8xf32>           // 中间结果**落地**
  }
  scf.for %i = %c0 to %c8 step %c1 {                  // 循环 2：relu
    %v = memref.load %c[%i] : memref<8xf32>
    %r = arith.maximumf %v, %z : f32
    memref.store %r, %c[%i] : memref<8xf32>           // 原地覆盖，旧值消失
  }
  return
}
```

**两版的依赖边**

```text
版本 A：边 = SSA 依赖
  %a ─┐
      ├─ linalg.add ──▶ %c ──▶ linalg.generic(relu) ──▶ %d
  %b ─┘                 ▲
                        └── 单定义 + 单使用 → 这条边可融合

版本 B：边 = 「对同一块内存的读写顺序」
  循环1 ──store──▶ [ %c 这块内存 ] ──load──▶ 循环2
                        ▲
                        └── 能否合并两个循环，取决于别名分析能否证明
                            %a / %b / %c 互不重叠（调用方可以传同一块进来）
```

**逐条读**

1. A 的 `%c` 只有一个定义、一个使用：融合就是把 relu 的计算体搬进 add 的循环体，整个 `%c` 张量根本不必存在（可退化成寄存器里的一个 `f32`）。
2. A 的 `tensor.empty()` 不是分配，只是给 destination-passing style 一个形状占位；真分配推迟到 bufferization——这正是 One-Shot Bufferize 能「原地复用」的余地。
3. B 的 `%c` 是函数参数里的一块内存：循环 2 从它读、又写回它。这里没有 SSA 边可看，只有 store/load 的先后。
4. B 的 `memref.store %r, %c[%i]` 是原地覆盖：一旦执行，add 的结果永久消失——想再读它的 Pass 已经晚了。

**并排对照**

| | 版本 A（tensor） | 版本 B（memref） |
|--|------------------|------------------|
| `%c` 是什么 | SSA 值，不可变 | 内存位置，可被 store 改写 |
| 哪条边允许融合 | `linalg.add → %c → relu` 这条 SSA 边 | 无 SSA 边；只有「先写后读同一 memref」的顺序约束 |
| 判定融合合法性要什么 | 看 use-def 就够 | 先做别名分析，证明不重叠 |
| 中间结果能否消掉 | 能（融合后 `%c` 不落地） | 已落地，只能靠后续内存优化补救 |
| 适合的优化 | 融合、tiling、shape 推导 | 内存复用、向量化 load/store、对接运行时 |

> 一句话：**可融合的那条边是版本 A 里的 `%c`**。bufferize 之后它退化成「对同一块内存的读写顺序」，融合的门槛就从「看一眼 use-def」升级成「证明不别名」——这就是 §5.1 那条经验法则的机械原因。

> **自测**：要把版本 B 的两个 `scf.for` 合成一个，编译器必须先证明哪几个 memref 之间的什么性质？

### 5.2 Layout（数据布局）

**是什么**：逻辑上的 `[N,C,H,W]` 在内存里按什么顺序、是否分块（blocking）、是否与向量宽度对齐。

常见例子：

- `NCHW` vs `NHWC`（通道维位置影响卷积与向量化）  
- 更细的 **blocked layout**（如 `NCHW4c`）以适配 SIMD / Tensor Core tile  
- GPU 上还有 **swizzle** 等与 bank conflict 相关的排布

**传播**：producer 偏好与 consumer 偏好不一致时，插入 `layout_transform`；变换本身有代价，应尽量把冲突推到图的边缘或与计算融合。

这直接对应研究问题②「跨后端 layout 兼容」。

### 5.3 量化基础（够用即可）

**是什么**：用低比特整数（INT8/INT4…）近似浮点，配合 **scale** 与可选 **zero-point**：

\[
x_{float} \approx scale \times (x_{quant} - zero\_point)
\]

必知区分：

- **per-tensor** vs **per-channel**（scale 粒度）  
- **对称** vs **非对称**  
- 图中的 `QuantizeLinear` / `DequantizeLinear`（或等价节点）会在分区边界变成**额外算子**——又是边界代价。

本文不展开数值技巧；知道「量化是图上的一等公民，且制造跨后端摩擦」即可。Glow 的 profile-guided 量化、Campo 对 cast 开销的警告，都是同一主题。

### 5.4 地址空间与内存层次

| 层级 | 典型容量 | 带宽 | 谁管理 |
|------|----------|------|--------|
| 寄存器 | 极小 | 最高 | 编译器分配 |
| 片上 SRAM / shared | KB～MB | 很高 | 显式（GPU）或缓存（CPU） |
| HBM / 设备显存 | GB～十 GB | 高 | 运行时分配器 |
| 主机 DRAM | 更大 | PCIe/NVLink 受限 | OS / 运行时 |
| 网络另一端 | — | 更慢 | 集合通信 / RDMA |

**地址空间（address space）** 在 IR 里常被编码为指针属性（LLVM `addrspace(N)`、MLIR memref memory space、NVPTX generic/global/shared）。**错误的地址空间转换昂贵且易错**——GPU 路径上这是高频坑。

---

## 第 6 章 中间表示与变换：SSA、多层 IR、Pass、渐进 lowering

### 6.1 SSA 与 CFG（所有现代编译 IR 的底座）

- **SSA（Static Single Assignment）**：每个值只赋值一次；汇合用 `phi`（LLVM）或 **block argument**（MLIR）。  
- **CFG**：基本块 + 终结符显式列出后继。  
- **效果**：数据流分析变便宜；「寄存器无别名」成立。

你已在 [`llvm-hello-compile`](../llvm-hello-compile/) 里用 mem2reg 亲眼见过；MLIR 把同一思想推广到可嵌套 Region。

### 6.2 多层 IR 与渐进 Lowering

**单层 IR 的病**：过早降到 LLVM IR → 「这是一次卷积」永久消失 → 无法再做融合/tiling。

**渐进 lowering**：每层只固化一类决定——

```
领域算子语义  →  结构化张量计算（linalg）→ 循环（scf）→ 向量 → LLVM/PTX
     ↑ 可融合/可推导 shape        ↑ 可 tile/fuse         ↑ 可向量化
```

IREE 的 `linalg → flow → stream → hal` 是同一哲学在「执行规划」上的实例：  
flow 固化「怎么切成 dispatch」；stream 固化「时序与资源寿命」；hal 固化「用哪块设备对象」。

详见 [`mlir-learning-guide.md`](./mlir-learning-guide.md) §1、[`iree-learning-guide.md`](./iree-learning-guide.md) §2。

#### 示例精讲：conv + relu 在三层的形态，以及「conv」是在哪一刀之后消失的

**最小具体输入**：`out = relu(conv2d(in, flt))`。NHWC 输入 `1×8×8×4`，HWCF 卷积核 `3×3×4×4`，stride 1、无 padding，因此输出 `1×6×6×4`。

**第 1 层：linalg on tensors —— 「这是一次卷积」本身是一个 op**

```mlir
// 片段：省略 func 头；%in / %flt 是参数，%c0f 是 0.0 : f32
%z    = tensor.empty() : tensor<1x6x6x4xf32>
%init = linalg.fill ins(%c0f : f32) outs(%z : tensor<1x6x6x4xf32>) -> tensor<1x6x6x4xf32>
%conv = linalg.conv_2d_nhwc_hwcf
          {strides = dense<1> : tensor<2xi64>, dilations = dense<1> : tensor<2xi64>}
          ins(%in, %flt : tensor<1x8x8x4xf32>, tensor<3x3x4x4xf32>)
          outs(%init : tensor<1x6x6x4xf32>) -> tensor<1x6x6x4xf32>
%z2   = tensor.empty() : tensor<1x6x6x4xf32>
%out  = linalg.generic {iterator_types = ["parallel","parallel","parallel","parallel"], …}
          ins(%conv : tensor<1x6x6x4xf32>) outs(%z2 : tensor<1x6x6x4xf32>) {
  ^bb0(%x: f32, %o: f32):
    %r = arith.maximumf %x, %c0f : f32
    linalg.yield %r : f32
} -> tensor<1x6x6x4xf32>
```

这一层能做：把 relu 融进 conv 的输出侧（conv 属 complex-out-fusable，见 [§3.2](#32-算子语义分类与融合fusion)）、换 layout、推导 shape。因为 op 名字还在，Pattern 可以直接匹配 `linalg.conv_2d_nhwc_hwcf`。

**第 2 层：scf + memref —— 循环骨架已定，conv 还在**

```mlir
scf.for %oh = %c0 to %c6 step %c3 {
  scf.for %ow = %c0 to %c6 step %c3 {
    // 输出 3×3 的 tile，配 3×3 卷积核、stride 1 → 需要 5×5 的输入窗口
    %in_t  = memref.subview %In[0, %oh, %ow, 0] [1, 5, 5, 4] [1, 1, 1, 1]
                : memref<1x8x8x4xf32> to memref<1x5x5x4xf32, strided<[…]>>
    %out_t = memref.subview %Out[0, %oh, %ow, 0] [1, 3, 3, 4] [1, 1, 1, 1]
                : memref<1x6x6x4xf32> to memref<1x3x3x4xf32, strided<[…]>>
    linalg.conv_2d_nhwc_hwcf ins(%in_t, %Flt : …) outs(%out_t : …)   // ← 名字还在
  }
}
```

这一层能做：改 tile 大小、把外层绑到 block/thread、给 `%in_t` 加一层 shared memory 暂存。**tile 尺寸已被固化**，但「这是卷积」这条信息还没丢。

**第 3 层：纯 load / mulf / addf / store —— 语义已经丢了**

```mlir
scf.for %oh = %c0 to %c6 step %c1 {
 scf.for %ow = %c0 to %c6 step %c1 {
  scf.for %f = %c0 to %c4 step %c1 {
   %acc = scf.for %kh = %c0 to %c3 step %c1 iter_args(%a0 = %c0f) -> f32 {
     %a1 = scf.for %kw = %c0 to %c3 step %c1 iter_args(%a2 = %a0) -> f32 {
       %a3 = scf.for %ci = %c0 to %c4 step %c1 iter_args(%a4 = %a2) -> f32 {
         %ih = arith.addi %oh, %kh : index
         %iw = arith.addi %ow, %kw : index
         %x  = memref.load %In[%c0, %ih, %iw, %ci]  : memref<1x8x8x4xf32>
         %w  = memref.load %Flt[%kh, %kw, %ci, %f]  : memref<3x3x4x4xf32>
         %m  = arith.mulf %x, %w : f32
         %s  = arith.addf %a4, %m : f32
         scf.yield %s : f32
       }
       scf.yield %a3 : f32
     }
     scf.yield %a1 : f32
   }
   %r = arith.maximumf %acc, %c0f : f32
   memref.store %r, %Out[%c0, %oh, %ow, %f] : memref<1x6x6x4xf32>
}}}
```

只剩「六层循环 + `mulf`/`addf`」。它与一段手写的、语义完全不同的六层循环在 IR 上**无法区分**。

**结构树对照**

```text
第 1 层  func.func
         └─ linalg.conv_2d_nhwc_hwcf   ← 按 op 名就能匹配
            + linalg.generic (relu)
第 2 层  func.func
         └─ scf.for / scf.for                （tile 已固化）
            └─ linalg.conv_2d_nhwc_hwcf   ← 仍能匹配
第 3 层  func.func
         └─ scf.for × 6
            └─ memref.load / arith.mulf / arith.addf / memref.store
                              ↑ 没有任何 op 叫 conv
```

| 层 | 谁还能匹配到 conv | 这一层可做的变换 | 这一层固化了什么 |
|----|-------------------|------------------|------------------|
| linalg on tensors | 能（op 名 + indexing maps） | 融合、layout、shape 推导、tiling 决策 | 还几乎没固化 |
| scf + memref | 能（循环里那条 named op） | 调 tile、绑线程、加 shared 暂存 | 循环结构、内存与别名 |
| 纯循环 + arith | **不能** | 只剩通用循环优化（unroll、LICM、向量化） | 迭代顺序与逐元素运算 |

**哪一层之后 Pass 再也匹配不到 conv**：**第 2 层 → 第 3 层那一刀（linalg → loops）**。只要 `linalg.conv_2d_nhwc_hwcf` 还在，`RewritePattern` 就能按名字找到它；一旦展成 `scf.for` + `arith.*`，要重新认出「这是卷积」只能靠 idiom recognition / raising，代价高且不可靠。所以**所有需要知道「这是卷积」的决定（融合、layout、tile 策略、要不要换 Tensor Core kernel）都必须在这一刀之前做完**——这就是「渐进 lowering」的全部理由。

> **自测**：如果一个 Pass 想在第 3 层把这段循环替换成 Tensor Core 指令，它必须先恢复出哪些信息？

### 6.3 Pass：分析与变换

| 类型 | 行为 | 例子 |
|------|------|------|
| Analysis | 只读，算性质，可缓存 | 支配树、别名分析、SCEV |
| Transform | 改 IR，并声明保留了哪些分析 | 融合、DCE、Dialect Conversion |

组织方式：PassManager 按 Module / Function / Loop（LLVM）或任意嵌套（MLIR）调度。  
**写 lowering 时最重要的工程纪律**：改了什么就要正确 invalidate；否则下游用过期分析会静默出错。

### 6.4 代价模型（Cost Model）

**是什么**：用可计算的分数估计「某种划分 / 某种 schedule / 是否向量化」的好坏，避免每次都上真机穷举。

出现位置：

- TVM AutoTVM / MetaSchedule 的 ML cost model  
- LLVM TTI（TargetTransformInfo）  
- 你在 `mlir-toy-dialect` 里写的 `ToyCostOpInterface`（教学版）  
- 图划分时的「计算时间 vs 传输时间」估计  

**没有代价模型，自动搜索与自动分区都会退化成拍脑袋。**

---

## 第 7 章 代码生成与变体：kernel、ISA 层级、多目标打包

### 7.1 Kernel 与启动配置

在 GPU 类设备上：

- **Kernel**：设备上可执行的一段程序（对应 IREE 的 `hal.executable` export、CUDA `__global__`）。  
- **Launch configuration**：grid / block（或 workgroup）维度——决定并行度与 shared memory 用量。  
- **ABI**：kernel 如何拿到绑定的 buffer、常量、workgroup id（IREE 的 `hal.interface.*` 五件套是这一层的标准化）。

CPU 路径上「kernel」常退化成「一个函数」；概念仍在：可执行体 + 调用约定。

### 7.2 虚拟 ISA vs 真实 ISA

以 NVIDIA 为例（思想可推广）：

| | 虚拟（PTX / `compute_XX`） | 真实（SASS / `sm_XX`） |
|--|---------------------------|------------------------|
| 可移植性 | 同一 PTX 可 JIT 到更新架构 | 绑定具体芯片 |
| 加载成本 | 首次需 JIT | 直接加载 |
| 性能可控性 | 依赖驱动 JIT 质量 | 编译期完全可见 |

AMD 有自己的中间表示与机器码层级；CPU 则是 LLVM IR → 机器码。  
**共同点**：编译栈几乎总有「可移植中间码」和「最终机器码」两层。

详见 [`cuda-fatbin-learning-guide.md`](./cuda-fatbin-learning-guide.md)、[`llvm-learning-guide.md`](./llvm-learning-guide.md) §5。

### 7.3 多变体打包（Fat Binary / ExecutableVariant）

**问题**：设备微架构太多，预编译「所有配置 × 所有架构」会爆炸（研究问题⑥）。

**模式**：一份逻辑可执行体，内嵌**多个实现变体**；运行时按 capability / condition 选第一个可用者。

| 实现 | 载体 |
|------|------|
| CUDA fatbin | 一个 cubin/fatbin 含多份 PTX/SASS |
| IREE | `hal.executable` 内多个 `variant` |
| 工程折中 | 热配置预编译 + 冷配置 JIT |

这是「配置组合爆炸」最务实的工程答案之一——先理解模式，再谈搜索与缓存。

---

## 第 8 章 运行时抽象机：设备、队列、同步、AOT/JIT

### 8.1 设备抽象机（最小词汇表）

不管叫 HAL、EP 还是 CUDA Runtime，编译器眼中的硬件大致长这样：

```
Driver ──▶ Device ──▶ Allocator ──▶ Buffer (+ BufferView: shape/layout/dtype)
                │
                ├── CommandBuffer / Stream：录制要做的工作
                ├── Executable：设备代码
                └── Synchronize：semaphore / event / fence
```

**为什么要抽象**：编译器若直接依赖 `cudaMemcpyAsync` / `vkCmdDispatch`，每个后端重写整条流水线；有了抽象，**调度决策可以在编译期写成对抽象对象的操作序列**（IREE 的核心取向）。

详见 [`iree-learning-guide.md`](./iree-learning-guide.md) 第 4 章。

### 8.2 异步与同步

现代加速器默认**异步**：host 提交工作后立即返回；完成与否靠同步原语。

| 原语族 | 直觉 | 局限 |
|--------|------|------|
| CUDA Event | 「某个点已经过去」 | 难表达复杂多队列依赖与 host↔device 统一时间线 |
| Timeline Semaphore（IREE） | 64 位单调计数器，wait/signal 任意顺序 | 需要运行时补齐驱动缺口 |
| Fence | 常作为「一批 wait/signal 的宿主侧句柄」 | — |

**编译期**若不能显式表达依赖，就无法把「计算–通信重叠」「stream-ordered allocation」做进产物。这就是为什么 IREE 把 timeline 视为灵魂。

#### 示例精讲：两条队列 + 一个 semaphore，重叠是怎么被「表达」出来的

**最小具体输入**：一个梯度张量被切成 2 块；每块先在设备上算（`mm`），再跨设备 AllReduce（`ar`）。设 `mm` 一块耗 3 个单位时间，`ar` 一块耗 2 个单位。两条队列：`Q0` 计算、`Q1` 通信。

**写法 (a)：只有「host 等全部完成」这一种同步**

```text
Q0: dispatch @mm(块0)
Q0: dispatch @mm(块1)
host: device.synchronize()        // 整批等：IR 里没有「块0 已就绪」这个概念
Q1: all_reduce(块0)
Q1: all_reduce(块1)
host: device.synchronize()
```

**写法 (b)：用 timeline semaphore 逐块表达依赖**

> 伪 IR 示意（形态贴近 IREE 的 stream/hal，具体 op 名以所用版本为准）

```text
%c = semaphore.create initial(0)     // 计算时间线
%n = semaphore.create initial(0)     // 通信时间线

Q0: dispatch @mm(块0)      wait(—)        signal(%c = 1)
Q0: dispatch @mm(块1)      wait(—)        signal(%c = 2)
Q1: all_reduce(块0)        wait(%c >= 1)  signal(%n = 1)   // 块0 算完即可发车
Q1: all_reduce(块1)        wait(%c >= 2)  signal(%n = 2)
Q0: dispatch @apply(块0)   wait(%n >= 1)  signal(%c = 3)   // 通信回来就能用
host: fence.wait(%n >= 2)                                  // 只在最后等一次
```

**时间轴对照**（示意；每个方块标了自己的起止时刻）

```text
(a) 粗同步：IR 里只有「整批等」
  Q0 计算  [ mm 块0 : 0→3 ][ mm 块1 : 3→6 ]
  Q1 通信                                    [ ar 块0 : 6→8 ][ ar 块1 : 8→10 ]
  host                     ↑ 等全部计算完     ↑ 等全部通信完
  总时长 = 6（计算）+ 4（通信）= 10

(b) semaphore：逐块依赖
  Q0 计算  [ mm 块0 : 0→3 ][ mm 块1 : 3→6 ]
  Q1 通信                   [ ar 块0 : 3→5 ]   [ ar 块1 : 6→8 ]
                            ↑ wait(%c>=1) 已满足 → 与 mm 块1 并行跑
  总时长 = 8   （ar 块0 完全藏进了 mm 块1 的时间里）
```

**逐条读**

1. (a) 与 (b) 的**设备工作量完全相同**，差别只在「IR 里有没有一条能被调度器利用的细粒度依赖」。省下的 2 个单位是纯粹的表达力收益。
2. `wait(%c >= 1)` 这种「等某个计数值」的写法，让 signal 与 wait **不必按提交顺序配对**：编译期可以先发 `Q1` 的 wait，再发 `Q0` 的 signal，运行时自会对齐。CUDA event 更像「录制一个已发生的点」，跨多队列 + host/device 统一时间线时表达力不够（见上表的「局限」列）。
3. 关键结论：**重叠不是运行时「聪明」出来的，是编译期写进产物里的**。如果降级到 (a) 那种只有整批同步的形态，运行时无论多聪明都看不出「块0 已经可以发车」——依赖信息已经在 IR 里丢了。

| | 写法 (a) 粗同步 | 写法 (b) timeline semaphore |
|--|-----------------|------------------------------|
| IR 里的依赖粒度 | 整批（队列级） | 每块（工作项级） |
| 能否 overlap 计算与通信 | 不能 | 能 |
| host 参与次数 | 每个阶段一次 | 只在最末尾一次 |
| 块数变多时 | 总时长 = 计算 + 通信 | 总时长 ≈ 计算 + 一个块的通信 |
| 谁做决定 | host 代码（编译器看不见） | 编译期固化在产物里 |

> **自测**：若把块数从 2 提到 8（每块 `mm` 0.75、`ar` 0.5 单位），写法 (b) 的总时长趋近于哪个数？

### 8.3 AOT vs JIT

| | AOT | JIT |
|--|-----|-----|
| 何时编译 | 部署前 | 运行中（首次或 shape 变化时） |
| 优点 | 启动快、可复现、可裁剪运行时 | 可按真实 shape/设备特化 |
| 缺点 | 变体爆炸或保守代码 | 首次延迟、工程复杂 |
| 典型 | IREE `.vmfb`、ExecuTorch `.pte`、fatbin 中的 SASS | PTX JIT、部分 TVM/ORT 路径、动态 shape 特化 |

算力网场景通常需要**混合**：稳态路径 AOT，变化路径 JIT 或预编译变体池。

---

## 第 9 章 分布式与编译的接缝：分片如何变成 IR

这是根 README「两条学习线交汇」的展开。没有这一章，分布式综述和 MLIR/IREE 会永远是两张皮。

### 9.1 并行策略的四要素（复习锚点）

对每一种并行（DP / TP / PP / SP / EP），始终问：

1. **切什么维度**（batch / 隐层 / 层 / 序列 / 专家）  
2. **通信什么原语**  
3. **放在哪个带宽域**（NVLink 域内 vs 跨节点）  
4. **主要代价**（算力气泡 / 通信体积 / 显存）

详见 [`paper-notes/01-efficient-training-distributed-infra.md`](./paper-notes/01-efficient-training-distributed-infra.md)。

### 9.2 SPMD 与分片标注（Sharding Annotation）

**SPMD（Single Program Multiple Data）**：所有设备跑**同一份程序**，差异由「每张量如何映射到 device mesh」描述。

**分片标注**：在 IR / 图上给张量挂属性，例如：

- Mesh-TensorFlow：显式 mesh + 命名维度  
- GSPMD：对中间张量推断注解  
- OneFlow SBP：Split / Broadcast / Partial  

**与编译器的关系**：

```
并行策略  ──编码为──▶  IR 上的 sharding / layout / device 属性
                ──Pass 推导──▶  插入通信算子、选择局部计算
                ──lowering──▶  每设备可执行体 + channel 集合通信
```

这正是「可编译的分布式」：策略不再是训练脚本里的 if/else，而是**可被 Pass 变换的 IR**。MLIR 的 Attribute / Interface 是挂这些注解的自然位置；IREE 的 `channel` 是通信在 HAL 层的落点。

### 9.3 集合通信原语（最小集）

| 原语 | 直觉 | 典型用途 |
|------|------|----------|
| AllReduce | 大家的值归约后再广播 | DP 梯度同步 |
| AllGather | 每卡拼出完整张量 | TP 的拼接 |
| ReduceScatter | 归约并按切分散开 | ZeRO / 部分 TP |
| AllToAll | 按分区重新洗牌 | MoE / 某些并行 |

编译器若能把 collective 与 compute 放进**同一条命令缓冲的依赖图**，才能谈 overlap。这是 IREE channel 设计与 NCCL 在概念上的对齐点。

### 9.4 IO 感知算法（负载侧方法论）

> **§9.4–9.5 的定位**：这两节把 FlashAttention / PagedAttention 两篇**自选补充**论文的结论蒸馏在此。它们是**论据**，不是五层栈里的一层——读完本节，主线就不缺环；原文只在你要实证或深挖研究问题时才需要。

FlashAttention 教的不是又一个 kernel 细节，而是：

> 当 Roofline 显示带宽瓶颈时，**正确的复杂度度量是 HBM 访问次数**；允许重计算换存储；用 tiling 把工作集塞进 SRAM。

这套方法论直接指导编译器该把哪些融合/tiling 决策视为一等公民——**这就是进 TVM / Halide 之前值得先扫一眼它的原因**。

### 9.5 分页状态与低成本迁移

PagedAttention 教的是：

> 巨大的运行时状态（KV Cache）应按**块**分配与索引；块化之后，抢占、前缀共享、跨设备搬迁的粒度都变小。

映射到研究问题④「运行状态变化后低成本切换后端」：没有块化/并行无关的状态表示，迁移只能整包拷贝——太贵。Universal Checkpointing 等方向是训练侧的同类问题。

它同时是研究问题③的样本：KV 长度由运行时采样结果决定，编译期根本看不到，**所以「内存怎么分配」这类决策必须留在运行时**，编译器只负责生成认识 block table 的通用 kernel。这条「编译期 vs 运行时职责边界」的划法，比记住分页机制本身更值钱。

---

## 第 10 章 概念 → 深入材料对照

> 「优先级」含义见 [§2.0](#20-优先级标注怎么读)：★★ 骨架 / ★ 硬前提 / ◆ 强杠杆 / ○ 可推迟。「轮次」含义见 [§2.4](#24-与工具路线同步的概念学习路线)。**表已按轮次排序**，所以从上往下也就是学习顺序。

| 轮次 | 概念 | 优先级 | 先读本文 | 再深入 |
|------|------|--------|----------|--------|
| **W0** | SSA / Pass / CodeGen | ★ | §6.1 · §6.3 · §7 | [`llvm-learning-guide.md`](./llvm-learning-guide.md) · [`llvm-hello-compile`](../llvm-hello-compile/) |
| **W0**→W5 | 计算图 / opset / shape | ★ | §3.1 | [`onnx-learning-guide.md`](./onnx-learning-guide.md) |
| **W1** | 分片标注进 IR | **★★** | §9.2 | [`paper-notes/01-…`](./paper-notes/01-efficient-training-distributed-infra.md) |
| **W1** | 并行四要素 / 集合通信 | ◆ | §9.1 · §9.3 | 同上 · IREE `channel` §4.8 |
| **W1** | 内存层次 / 地址空间 | ◆ | §5.4 | 同上（显存账本与带宽域） |
| **W2** | 渐进 lowering / Conversion | **★★** | §6.2 | [`mlir-learning-guide.md`](./mlir-learning-guide.md) · [`mlir-toy-dialect`](../mlir-toy-dialect/) |
| **W2** | tensor/buffer | ★ | §5.1 | [`mlir-learning-guide.md`](./mlir-learning-guide.md) §8 |
| **W3** | HAL / fence（设备抽象机 + 同步） | **★★** | §8.1–8.2 | [`iree-learning-guide.md`](./iree-learning-guide.md) 第 4 章 |
| **W3**→W5 | 委托 / 分区边界 | **★★** | §3.3 | [`executorch-learning-guide.md`](./executorch-learning-guide.md) · onnx EP 章 · 动手 [`../onnx-delegate-lab/`](../onnx-delegate-lab/) |
| **W3**→W6 | kernel / variant | ◆ | §7.1 · §7.3 | [`iree-learning-guide.md`](./iree-learning-guide.md) §4.7 |
| **W4** | Roofline | ◆ | §4.3 | 实证见 [`08-flash-attention.md`](./paper-notes/08-flash-attention.md)（自选补充） |
| **W4** | 融合四类 | ★ | §3.2 | [`tvm-learning-guide.md`](./tvm-learning-guide.md) §2.2 · [`paper-notes/05-tvm.md`](./paper-notes/05-tvm.md) |
| **W4** | 算法/调度、tiling | ★ | §4.1–4.2 | [`paper-notes/04-halide.md`](./paper-notes/04-halide.md) · tvm §3 |
| **W4** | layout | ◆ | §5.2 | tvm layout 节 · EP 边界代价 |
| **W5** | AOT vs JIT（决策时刻） | ◆ | §8.3 | [`iree-learning-guide.md`](./iree-learning-guide.md) · ORT session / ET 导出期 |
| **W5** | 量化 | ○ | §5.3（只看结论） | [`paper-notes/06-glow.md`](./paper-notes/06-glow.md) profile-guided 量化 |
| **W6** | fatbin / compute vs sm | ◆ | §7.2–7.3 | [`cuda-fatbin-learning-guide.md`](./cuda-fatbin-learning-guide.md) |
| **W6** | 少量原语多后端 | ○（对照） | — | [`paper-notes/06-glow.md`](./paper-notes/06-glow.md) vs TVM |
| **W7–W8** | 四条工业栈怎么选 | ◆（学完再看） | §3.4 | [`onnx`](./onnx-learning-guide.md) · [`executorch`](./executorch-learning-guide.md) · [`iree`](./iree-learning-guide.md) · [`tvm`](./tvm-learning-guide.md) |
| **W7–W8** | KV 分页 / 迁移直觉 | ○ | §9.5 | [`paper-notes/09-paged-attention-vllm.md`](./paper-notes/09-paged-attention-vllm.md)（自选补充） |

---

## 第 11 章 学习路径：什么时候读本文

### 11.1 开局读什么 · 卡住时回查什么

**按周次的完整安排见 [§2.4 同步路线表](#24-与工具路线同步的概念学习路线)**（W0→W8 每轮补哪几节、为什么这时候补）。本节只补两件 §2.4 没说的事：**开局那半天读什么**，以及**中途卡住时按什么触发去回查**。

**开局半天（W0 之前）**：§1 总图 → §2.0 优先级读法 → §2.4 路线表 → [附录速查](#附录一页速查)。**读完就停**，不要顺着往第 3 章读下去。

**回查触发表**（不是阅读顺序，是「出现这个症状就翻这一节」）：

| 症状 | 翻哪节 |
|------|--------|
| 看到 `%3 = phi` 或不理解「为什么变量只赋值一次」 | §6.1 |
| 一份 IR 里同时出现 `tensor<...>` 和 `memref<...>`，分不清界 | §5.1 |
| 不理解为什么要经过这么多层 dialect / 为什么不能一步降到底 | §6.2 |
| 看到 `flow.dispatch` / `EPContext` / `lowered_backend_module`，分不清谁做了决定 | §3.3 |
| 看到 `hal.device` / `semaphore` / `command_buffer` 不知各管什么 | §8.1–8.2 |
| 看到 `split` / `reorder` / `cache_write` 不知在优化什么 | §4.1–4.2 |
| 看到 `compute_75` 与 `sm_75` 并列出现 | §7.2 |
| 看到 `shard` / `mesh` / `DeviceMesh` 属性 | §9.2 |
| 要写「为什么这个决定该留到运行时」 | §8.3 |
| 要写六个研究问题的观点，缺词汇 | §3.3 · §5.2 · §8.3 · §9 |

### 11.2 必学 vs 可推迟

> 完整的优先级标注（每条概念 + 它服务于老师指定的哪个工具）见 [§2.0–2.3](#20-优先级标注怎么读)，**分几轮学见 [§2.4](#24-与工具路线同步的概念学习路线)**；本节只给两头的结论。

**这一路必须真正建立直觉的七块**（缺一块就有一整份专题文档读不进去）。注意括号里的轮次——**它们不在开局一起学**，四根 ★★ 柱子分别落在 W1／W2／W3：

1. 五层栈总图（§1）— W0，唯一需要开局就有的  
2. SSA + Pass（§6.1、§6.3）— W0，`llvm-hello-compile` 直接在写  
3. **分片标注如何成为 IR 属性（§9.2）** ← 柱子④ — W1  
4. **渐进 lowering（§6.2）** ← 柱子① — W2；tensor vs buffer（§5.1）同轮  
5. **设备抽象机 + 异步同步（§8.1–8.2）** ← 柱子③ — W3  
6. 算法/调度分离 + tiling（§4.1–4.2）— W4  
7. **融合 vs 委托分区的同构 + 四类边界代价（§3.2–3.3）** ← 柱子② — W3 引入、W5 坐实  

**可推迟到实际项目里补**（属宏观 AI-Infra 体系，与老师指定内容只是间接相关）：

- 量化数值技巧（Hadamard、随机舍入等）——§5.3 只需知道「量化制造跨后端摩擦」  
- IO 感知算法（§9.4）与分页状态（§9.5）——来自自选补充的两篇论文，读结论即可  
- 具体集合通信算法（Ring/Tree）实现细节  
- 某张 GPU 的 SASS 指令  
- 自动微分的正/反向模式实现  
- polyhedral 依赖分析公式  

### 11.3 自检（能讲清就算过关）

- [ ] 从 PyTorch 模型到 GPU 上跑完一次 matmul，中间经过哪五类表示？每层「固化什么决定」？  
- [ ] 为什么说「融合」和「EP 分区」是同一类决策？边界上贵在哪里？  
- [ ] 为什么不能把模型一步降成 LLVM IR？用「信息不可逆丢失」举一例。  
- [ ] tensor 世界和 buffer 世界各适合做什么优化？  
- [ ] 用四要素讲清 Tensor Parallel：切什么、通什么、放哪、代价。  
- [ ] fatbin 与 IREE ExecutableVariant 解决的是同一个什么问题？  
- [ ] 同一个模型要在算力网上跑，ORT EP / ExecuTorch / IREE / TVM 四条栈各自把哪个决策权交给了谁？（§3.4）

### 11.4 五个最小动作（把词汇变成手感）

本文只讲概念，动手在各专题文档里。但下面几件事**任何阶段都能做，且直接验证本文的核心同构**：

| 动作 | 命令 / 位置 | 验证了本文哪一节 |
|------|-------------|------------------|
| 看一次「同一段代码在不同层的样子」 | `cd llvm-hello-compile && bash scripts/run.sh`，对比 `02_sum_O0.ll` 与 `03a_sum_mem2reg.ll` | §6.1 SSA、§6.3 Pass |
| 看一次「渐进 lowering 丢了什么」 | `cd mlir-toy-dialect && bash scripts/all.sh`，对比 `x*4` 在 toy 层与 low 层的命运 | §6.2 渐进 lowering |
| 看一次「同一算法两种 schedule」 | `cd tvm-fatbin-lab && bash scripts/run_tvm.sh`，对比 `out/tvm/01_*.lower.txt` | §4 算法/调度 |
| 看一次「谁来划分、边界在哪」 | `cd onnx-delegate-lab && bash scripts/run.sh`，读 `out/ANALYSIS.md` | §3.3–3.4 委托/选型 |
| 看一次「一份产物里有几个变体」 | `cd tvm-fatbin-lab && bash scripts/run_fatbin.sh`（或教材第 8 章手写 `cuobjdump`） | §7.3 多变体打包 |

做完这些，本文 80% 的抽象都有了对应的实物。建议顺序：前两件随时做；后三件跟阶段 4→6（schedule → 划分 → fatbin）。

---

## 附录：一页速查

```
【五层栈】(输入:框架模型) → 交换图IR → 中端(融合/调度) → 执行规划(dispatch/stream/hal|EP) → 后端IR/机器码 → 运行时

【三个目标落点】
  表达分布式：分片标注进 IR + 集合通信进执行规划
  适配异构：  委托分区 + layout/地址空间 + 多变体 + HAL 抽象机
  运行时决策：AOT/JIT 混合 + fence/timeline + 块化状态迁移

【同构对照】
  融合规则     ≈  单后端上的「子图划分」
  ORT GetCapability / ET Partitioner / IREE flow.dispatch  ≈ 多后端划分
  CUDA fatbin  ≈  IREE ExecutableVariant  ≈  热配置预编译+冷配置JIT
  Halide schedule  ≈  TVM TE schedule  ≈  （思想）MLIR transform/tiling
  SSA+Pass     ≈  LLVM / MLIR 共用底座；MLIR 多了 Region 嵌套与 dialect

【数据】tensor(值/SSA) → bufferize → memref/buffer(存储/可别名)
【布局】逻辑形状 ≠ 物理 layout；跨后端要付转换税
【瓶颈】先 Roofline：带宽墙 → 融合/tiling/IO感知；算力墙 → 向量化/tensorize

【同步】异步提交是默认；没有显式依赖就没有 overlap
【分布式】策略 → sharding 属性 → Pass 插通信 → 每设备 executable + channel
```

---

## 维护约定

- 新增专题学习文档时：在[第 10 章对照表](#第-10-章-概念--深入材料对照)加一行「概念 → 该文档」，并标上优先级与轮次（口径见 [§2.0](#20-优先级标注怎么读) / [§2.4](#24-与工具路线同步的概念学习路线)）。  
- 新增横切概念时：在[第 2 章](#第-2-章-概念索引按学习目标分组)对应分组加一行，**必须填「优先级」「轮次」「直接服务于哪个老师指定内容」**——填不出后两者，就说明它当前属 ○，不要拉进必学清单；同时把它挂进 [§2.4](#24-与工具路线同步的概念学习路线) 某一轮，**不允许有「无主」概念**。  
- 根 README §7.2 周次表若调整：同步改 [§2.4](#24-与工具路线同步的概念学习路线) 的轮次列与双轨图；调整学习目标表述则同步改[第 1 章表格](#第-1-章-一张总图从模型到异构设备)。  
- 本文保持「横切概念」角色，**不复制**各专题文档的 API 细节。
