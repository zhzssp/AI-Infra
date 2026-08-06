# TVM 学习文档：端到端图编译 + 调度搜索全景

> **本文档的定位**
> - 基于 **Apache TVM 官方文档**（https://tvm.apache.org/docs/）与社区策略讨论整理，讲的是**今天的 Apache TVM 作为工程系统长什么样**。
> - 与 [`paper-notes/05-tvm.md`](./paper-notes/05-tvm.md) 分工明确：那篇是 **OSDI 2018 论文**笔记，讲"当年为什么要这样设计"（N×M 组合爆炸、图级/算子级联合优化、AutoTVM 动机）。本文讲**结构与机制**，把仍在用的论文概念落到可动手的 API 与流水线上。
> - 算法/调度分离的思想源头见 [`paper-notes/04-halide.md`](./paper-notes/04-halide.md)；与 IREE/MLIR 路线对照见 [`iree-learning-guide.md`](./iree-learning-guide.md) 与本文第 7 章。
> - 服务目标：根 README §4.1 的必学点——融合规则、layout 传播、TE+schedule、tensorize、调优闭环、PackedFunc——以及研究问题①（子图划分）、②（跨后端 layout）、⑥（配置组合爆炸）。
> - **先修**：[`ai-compiler-foundations.md`](./ai-compiler-foundations.md) §3（图与融合）、§4（算法/调度、tiling、Roofline）、§5.2（layout）。
> - **动手项目**：[`../tvm-fatbin-lab/`](../tvm-fatbin-lab/) —— `bash scripts/run.sh`，先读 `out/ANALYSIS.md`。
> - **正交对照（多后端划分）**：[`../onnx-delegate-lab/`](../onnx-delegate-lab/) —— 融合≈单后端划分；EP/Partitioner≈多后端划分。
>
> **主要信息源**
> - 官方文档：https://tvm.apache.org/docs/（TE / TIR / AutoTVM / AutoScheduler / MetaSchedule / Runtime）
> - OSDI'18 论文仍在用的概念：四类融合、layout 传播、TE+schedule、tensorize、PackedFunc（见 [`05-tvm.md`](./paper-notes/05-tvm.md)）
> - MetaSchedule RFC / 社区策略（discuss.tvm.apache.org）：**TE-compute 保留；TE-schedule 对新技术工作弃用，改走 TensorIR + MetaSchedule；调优统一进 MetaSchedule**
> - 学习优先级说明：经典 TE+schedule 原语仍是**必学心智模型**；动手时用它们对照 `tvm.lower`，新项目再切 TensorIR/MetaSchedule
>
> **一句话读法**：如果只有两小时，读[第 1 章总图](#第-1-章-一张总图今天的-tvm-流水线)、[第 2.2 四类融合](#22-算子融合四类规则重点)、[第 3 章每个 schedule 原语的前后对比](#第-3-章-te--schedule-原语每个都看懂循环变化)、[第 5 章调优闭环](#第-5-章-autotvm--ansor--metaschedule搜索闭环)、[附录速查](#附录一页速查)。

---

## 目录

- [第 1 章 一张总图：今天的 TVM 流水线](#第-1-章-一张总图今天的-tvm-流水线)
- [第 2 章 图级 IR 与图级优化](#第-2-章-图级-ir-与图级优化)
  - [2.1 Relay → Relax：静态到动态 shape](#21-relay--relax静态到动态-shape)
  - [**2.2 算子融合：四类规则（重点）**](#22-算子融合四类规则重点)
  - [2.3 数据布局变换与传播](#23-数据布局变换与传播)
  - [2.4 常量折叠与静态内存规划](#24-常量折叠与静态内存规划)
- [**第 3 章 TE + schedule 原语：每个都看懂循环变化**](#第-3-章-te--schedule-原语每个都看懂循环变化)
- [第 4 章 tensorize：异构后端的钩子](#第-4-章-tensorize异构后端的钩子)
- [第 5 章 AutoTVM / Ansor / MetaSchedule：搜索闭环](#第-5-章-autotvm--ansor--metaschedule搜索闭环)
- [第 6 章 Runtime：PackedFunc ABI + Graph/VM Executor](#第-6-章-runtimepackedfunc-abi--graphvm-executor)
- [第 7 章 与 AI-Infra 目标的对接（含 vs IREE/MLIR）](#第-7-章-与-ai-infra-目标的对接含-vs-ireemlir)
- [第 8 章 学习路径：最小必要集与动手清单](#第-8-章-学习路径最小必要集与动手清单)
- [附录：一页速查](#附录一页速查)

---

## 第 1 章 一张总图：今天的 TVM 流水线

**Apache TVM** 是一套端到端深度学习编译栈：上接多框架前端，下接 CPU / GPU / 加速器，中间用**图级优化**决定"切成哪些算子"，用**张量级调度 + 自动搜索**决定"每个算子怎么算快"。

一句话记住它和别人的区别：

> 传统框架把"算得快"外包给**厂商算子库**（cuDNN、ACL……）；  
> **TVM 用编译器生成算子代码**，并用 ML 引导的搜索在巨大配置空间里找高性能 schedule——图级决策与算子级代码生成是同一条流水线的两端。

### 1.1 端到端流水线（必背）

```
  前端模型（PyTorch / ONNX / TensorFlow / …）
           │  frontend import
           ▼
  ┌────────────────────────────────────────────────────────────┐
  │  图 IR：Relay（经典，偏静态 shape） / Relax（现代，动态 shape）│
  │  节点 = 算子，边 = 张量依赖                                   │
  └────────────────────────────────────────────────────────────┘
           │  图级优化（Graph opts）
           │  · 算子融合（四类规则）
           │  · layout 变换与传播
           │  · 常量折叠 / 静态内存规划 / 死代码消除 …
           ▼
  优化后的计算图（每个节点 ≈ 一个将要 codegen 的 fused op）
           │  对每个 fused op
           ▼
  ┌────────────────────────────────────────────────────────────┐
  │  TE compute："算什么"（index formula，无循环顺序）            │
  │       ↓                                                    │
  │  【经典路径】TE schedule 原语 → Lower → TIR                  │
  │  【现代路径】TensorIR（TIR 手写/变换）+ MetaSchedule 搜索     │
  └────────────────────────────────────────────────────────────┘
           │
           ▼  CodeGen
      LLVM（CPU） / CUDA·OpenCL·Metal（GPU） / 加速器 backend
           │
           ▼
  可部署模块（lib + graph/VM 程序 + params）
           │
           ▼
  Runtime：PackedFunc 统一 ABI
           + Graph Executor（拓扑序） / Relax VM / 旧 Relay VM
```

| 阶段 | 回答的问题 | 核心产物 |
|------|-----------|---------|
| **Frontend** | 模型从哪来 | 统一图 IR（Relay/Relax `IRModule`） |
| **Graph opts** | 哪些算子合成一个 kernel；用什么 layout | 更少节点、带偏好 layout 的图 |
| **TE compute** | 每个输出元素怎么算 | 声明式张量表达式（与 schedule 解耦） |
| **Schedule / TensorIR** | 循环怎么嵌套、并行、缓存、向量化 | 具体循环结构（TIR） |
| **Auto-tuning** | 在组合爆炸里选哪组 knob | tuning log / database |
| **CodeGen** | 落到哪条 ISA / 驱动 API | `.so` / PTX / OpenCL … |
| **Runtime** | 谁按什么顺序调用这些函数 | PackedFunc + executor/VM |

### 1.2 现代演进：你该站在哪条路上学

社区（discuss.tvm.apache.org 策略讨论 + MetaSchedule RFC）的共识可以压成三句话：

| 组件 | 今天的定位 | 学习建议 |
|------|-----------|---------|
| **TE compute** | **保留**。仍是描述"算什么"的好接口 | 必学 |
| **TE schedule** | 对**新工作弃用**；新代码走 TensorIR 变换 + MetaSchedule | **仍必学心智模型**——原语语义没变，只是 API 载体换了 |
| **Relay** | 成熟、静态 shape 友好；新特性重心在 Relax | 会用 `relay.build` 即可，不背 pass 全集 |
| **Relax** | 为**动态 shape** / 符号 shape 设计的下一代图 IR | 知道动机即可，深挖可跳过 |
| **AutoTVM** | 模板式搜索的经典形态 | 动手跑通一次；理解闭环 |
| **Ansor（auto_scheduler）** | 模板无关搜索 | 理解"不写 template 也能搜" |
| **MetaSchedule** | **当前统一调优框架**（吸收 AutoTVM + Ansor） | 知道它是终点；动手可先 AutoTVM |

> **为什么学习顺序仍从经典 TE+schedule 开始**：TensorIR 的 `split`/`reorder`/`vectorize` 等变换，语义上就是 schedule 原语的 TIR 版；AutoTVM 的 knob 空间也是这些原语的参数化。先在 `tvm.lower(..., simple_mode=True)` 里把循环看穿，再读 MetaSchedule 文档不会迷路。

### 1.3 与 Halide / IREE 的一句话对照

```
Halide（见 paper-notes/04-halide.md）
  算法 ↔ 调度 解耦；面向图像 stencil；CPU/GPU 为主
        │  TVM 继承 compute/schedule，并加上
        │  memory scope / tensorize / virtual thread / 图级 DL 优化 / AutoTVM
        ▼
TVM
  图 IR + TE/TIR + 搜索；重心是「单机多后端把算子编快」
        │  对照
        ▼
IREE（见 iree-learning-guide.md）
  MLIR 多 dialect 渐进 lowering + HAL；重心是「把调度也编译进产物、统一设备抽象」
```

---

## 第 2 章 图级 IR 与图级优化

图级优化的输出**先验地决定**了算子级要生成什么代码：融合决定"一个 kernel 里有哪些计算"，layout 决定"tile/向量化对齐哪一维"。这一章对应根 README 必学点 1、2。

### 2.1 Relay → Relax：静态到动态 shape

| | **Relay**（经典） | **Relax**（现代 / Unity） |
|--|------------------|---------------------------|
| Shape | 编译期大多静态已知，特化 kernel | 符号 shape / 动态维，运行时推导 |
| 执行 | `graph_executor` 为主；另有 Relay VM | `relax.VirtualMachine` |
| 适用 | CNN、固定输入尺寸推理 | LLM、变长 seq/batch |
| 你要掌握的深度 | 会 import → `relay.build` → 跑通 | **知道它解决动态 shape 即可（先跳过深挖）** |

论文时代的图 IR 是 NNVM；随后社区用 **Relay** 替换并增强；再往后 **Relax** 把"每个具体 shape 特化一份"的假设拆掉。对 AI-Infra 而言：研究问题③（动态 shape）在 TVM 侧的答案就是这条演进线——细节见 [`05-tvm.md`](./paper-notes/05-tvm.md) §6。

```
ONNX / PyTorch  ──frontend──►  Relay IRModule          ──relay.build──►  GraphModule
                              （静态 shape 友好）

ONNX / …       ──frontend──►  Relax IRModule           ──relax.build──►  VirtualMachine
                              （符号/动态 shape）
```

### 2.2 算子融合：四类规则（重点）

> **这是「子图划分」问题最经典的解法之一，必须吃透。**  
> 融合 = 在单设备、单编译栈内部，把多个图节点合并成一次 kernel 调用；  
> 多后端委托里的 partition = 在设备/EP 边界上切图。二者粒度不同，但都是"哪些算子绑在一起执行"的决策，且都受"下游能否为切分结果生成高效代码"约束。

#### 2.2.1 四类算子

| 类别 | 含义 | 典型算子 | 融合时要注意什么 |
|------|------|---------|----------------|
| **injective** | 输出与输入在索引上大致一对一（element-wise / broadcast 可视为此类） | `add`、`mul`、`relu`、`bias_add`、`sqrt` | 彼此容易融合；中间张量可消掉 |
| **reduction** | 沿某轴归约，输出秩降低 | `sum`、`max`、`mean`（reduce） | 可吃掉**输入侧**的 injective；输出侧再融要谨慎 |
| **complex-out-fusable** | 计算复杂（卷积/GEMM），但**输出侧**可接 element-wise | `conv2d`、`dense`/`matmul` | 经典模式：`conv + bias + bn + relu` 融成一个 kernel |
| **opaque** | 内部有复杂控制流/数据依赖，编译器不擅自融 | `sort`、部分 `nms`、某些自定义 op | **不参与**自动融合；边界即 fusion group 边界 |

#### 2.2.2 融合规则（谁可以和谁融）

用"生产者 → 消费者"方向记三条主规则 + 一条禁止：

```
规则 A  injective ⊕ injective → injective
        e.g.  mul → add → relu   ⇒  一个 element-wise kernel

规则 B  injective → reduction
        e.g.  scale → sum        ⇒  归约前的缩放融进 reduction kernel
        （reduction 吃掉输入侧 injective）

规则 C  complex-out-fusable → injective*
        e.g.  conv2d → bias → relu  ⇒  卷积写回前做完 bias/relu
        （complex 吃掉输出侧一串 injective）

禁止    涉及 opaque 的边不自动跨过
        e.g.  relu → sort → add   ⇒  sort 切开两侧；两侧可各自融合
```

更细的组合直觉：

| 模式 | 能否融合 | 原因 |
|------|---------|------|
| `add → mul → relu` | ✅ 一条 injective 链 | 无归约、无复杂计算，中间值可寄存器/共享内存传递 |
| `conv2d → bias_add → relu` | ✅ complex-out + injective | 卷积每个输出点算出后立刻做 bias/relu，避免全局写读 |
| `mul → sum`（对某轴） | ✅ injective → reduction | 先乘后加可在归约循环里完成 |
| `sum → relu` | ⚠️ 通常**不**按 complex-out 那套路自动融成"归约后 element-wise 大核"的首选形态；实践中视实现与 shape，但**不要**把它记成与规则 C 对称的"reduction-out-fusable" | 归约已改变数据并行结构，输出侧再融收益/复杂度与 conv 后不同 |
| `conv2d → conv2d` | ❌ 一般不融 | 两个 complex 各自有大工作集；融成一个超大 kernel 搜索/寄存器压力爆炸 |
| `relu → sort` | ❌ 被 opaque 切开 | sort 需要全局重排，不能塞进 element-wise 或 conv epilogue |
| `dense → add → softmax` | 部分：`dense+add` 可走规则 C；`softmax` 含 reduce+broadcast，常单独或按专用模式处理 | softmax 不是单纯 injective |

#### 2.2.3 例子：融合前后在执行上差在哪

**未融合（三次 kernel + 两次全局中间写）：**

```
  X ──► [conv2d] ──► T1 ──► [bias_add] ──► T2 ──► [relu] ──► Y
         kernel1      DRAM     kernel2      DRAM    kernel3
```

**融合后（一次 kernel，T1/T2 不落全局）：**

```
  X ──► [conv2d + bias + relu] ──► Y
              一个 fused kernel
         （累加寄存器/共享内存里做完 bias、relu 再写 Y）
```

论文实测融合单独贡献约 **1.2×–2×**（见 [`05-tvm.md`](./paper-notes/05-tvm.md)）；在带宽受限的 GPU/加速器上，省掉的是中间张量的全局内存往返。

#### 2.2.4 为什么这是"子图划分"的经典解

1. **用语义类别代替穷举模式**：不维护 `conv+bn+relu`、`depthwise+bias`……无限模式表，而是维护四类 + 规则——模式爆炸变成分类问题。
2. **划分结果可被下游消化**：融合出的节点交给 TE/TIR 生成**一个**循环程序；不会出现"图上融了但库里没有 fused kernel"的断裂（这正是纯算子库路线的痛点）。
3. **对你做多后端委托的启示**：ORT EP / ExecuTorch Partitioner 在设备边界上切图时，同样需要"可融合/可委托"的类别与规则；TVM 证明了**规则要和代码生成能力闭环**，不能只在图上纸上谈兵。

### 2.3 数据布局变换与传播

#### 2.3.1 机制

每个算子（或每个后端实现）可声明**偏好布局**（preferred layout），例如：

- CPU 向量化友好：`NCHW` 或 `NCHWc`（c 为内层 block，对齐 SIMD）
- 某些 GPU/库：`NHWC`
- 加速器 / TensorCore：`NCHW16c`、`HWNC`、矩阵的 `row-major` vs `col-major`、warp 专用 swizzle 等

当边上 **producer 产出的布局 ≠ consumer 期望的布局** 时，图优化插入 **layout transform** 节点：

```
  优化前（偏好冲突，却没有显式转换）：
    conv(偏好 NCHW) ──► relu(被标注为 NHWC)     ← 语义上非法或极慢

  优化后：
    conv(NCHW) ──► layout_transform(NCHW→NHWC) ──► relu(NHWC)
```

**传播（propagation）**：不是每条边都盲目插转换。常见策略是：

1. 从对布局敏感的算子（conv/dense）出发，把偏好沿图传播；
2. 尽量让整段子图统一到同一布局，把 transform **合并、外推到边界**；
3. 在"统一布局的收益"与"转换次数"之间做权衡（多余的 transform 本身就是开销）。

```
  偏好布局声明          布局传播           冲突处插入 transform
  ─────────────        ──────────         ────────────────────
  conv  → NCHW    →    下游 relu 也改成     仅在与外部/另一后端
  dense → NC            NCHW 偏好           边界处保留转换
```

#### 2.3.2 与研究问题②的关系

根 README 研究问题②：**跨后端 layout / 量化兼容**。

| 场景 | TVM 已给出的 | 算力网还缺的 |
|------|-------------|-------------|
| **单目标硬件** | 偏好声明 + 传播 + 插 transform | —— |
| **多 EP / 多厂商** | 同一套"图上表达布局"的思路 | **跨厂商没有统一协商协议**；边界上可能是 layout + dtype + 量化参数一起转，代价远大于单设备内 `NCHW→NHWC` |
| **和 IREE 对照** | TVM 在图 IR 上传播 | IREE 用 `hal.buffer_view` 携带 element type / 形状等信息；设备侧 layout 更多在 codegen/variant 路径里消化 |

**一句话**：TVM 的 layout 传播是单后端场景下的**标准答案模板**；跨后端时，你要额外设计"边界上何时转、转什么、谁承担精度损失"。

### 2.4 常量折叠与静态内存规划

| 优化 | 做什么 | 为何在 DL 图上特别有效 |
|------|--------|----------------------|
| **常量折叠** | 编译期算死能算死的子图（权重相关的固定变换等） | 推理图大量常量权重 |
| **静态内存规划** | 预知中间张量生命周期，复用 buffer（类似寄存器分配，对象是张量） | 经典 CNN shape 静态 → 可 AOT 规划峰值显存 |

这两项不是本文重点，但要知道它们和图融合、layout 一起，构成"图级四件套"。动态 shape（Relax）会削弱"完全静态规划"的假设——这又是 Relay→Relax 的动机之一。

---

## 第 3 章 TE + schedule 原语：每个都看懂循环变化

> 对应根 README：对着 `tvm.lower(..., simple_mode=True)` 说清每个原语对循环做了什么。  
> 思想源头：[`paper-notes/04-halide.md`](./paper-notes/04-halide.md) 的 algorithm/schedule 分离。

### 3.1 TE compute：只描述"算什么"

```python
import tvm
from tvm import te

M, N, K = 1024, 1024, 1024
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
k = te.reduce_axis((0, K), name="k")
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
```

- `compute(shape, lambda indices: expr)`：**输出每个坐标的值**，没有 for、没有并行、没有缓存。
- 同一份 TE，配不同 schedule → 性能可以差几个数量级，语义不变。

### 3.2 默认（naive）schedule 的循环长什么样

```python
s = te.create_schedule(C.op)
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

TIR 风格伪代码：

```text
for i in 0..1023:
  for j in 0..1023:
    C[i, j] = 0
    for k in 0..1023:
      C[i, j] += A[i, k] * B[k, j]
```

下面每个原语都给出**作用对象 → 循环结构前后对比 → 一句话目的**。

---

### 3.3 `split`：一轴变两轴

把轴 `i`（长度 1024）按 `factor=32` 拆成 `i.outer`（32）× `i.inner`（32）。

```python
i, j = C.op.axis
io, ii = s[C].split(i, factor=32)
```

```text
【前】 for i in 0..1023:          【后】 for io in 0..31:
        for j ...                           for ii in 0..31:      # i = io*32+ii
                                              for j ...
```

**目的**：制造可 tiling / 可 bind 到 block-thread / 可 vectorize 的内层。

---

### 3.4 `tile`：两轴同时拆，形成块

```python
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], x_factor=32, y_factor=32)
# 注意：API 参数名与轴顺序以你安装的 TVM 版本文档为准；思想是「两维一起分块」
```

等价于对两个空间轴各做一次 `split`，并通常配合 `reorder` 把两个 outer 放外、两个 inner 放内：

```text
【前】 for i:                     【后】 for io:        # 块行
        for j:                            for jo:      # 块列
          for k: ...                        for ii:    # 块内行
                                              for ji:  # 块内列
                                                for k: ...
```

**目的**：分块后，一块内的 `A`/`B` 子矩阵可复用（cache / shared memory 的前提）。

---

### 3.5 `reorder`：改嵌套顺序

```python
# 假设轴顺序现为 (io, ii, jo, ji)
s[C].reorder(io, jo, ii, ji)   # 两个 outer 在外，两个 inner 在内
```

```text
【前】 io, ii, jo, ji            【后】 io, jo, ii, ji
      （先扫完一行块的内层再换 j）      （先定位到块 (io,jo)，再扫块内）
```

**目的**：改善局部性（块内连续访问）、满足后续 `bind`/`vectorize` 对最内层的要求。  
**注意**：乱序可能破坏依赖或向量化连续性——合法 schedule 仍须尊重 reduce 依赖等约束。

---

### 3.6 `fuse`：多轴合成一轴

```python
# 把 yo, xo 融成一个轴，便于映射到一维 threadIdx
fused = s[C].fuse(yo, xo)
s[C].bind(fused, te.thread_axis("threadIdx.x"))
```

```text
【前】 for yo:                    【后】 for fused in 0..(yo_n*xo_n-1):
        for xo:                            # yo = fused / xo_n; xo = fused % xo_n
          ...
```

**目的**：GPU 上线程索引常是一维/三维网格；`fuse` 把逻辑多维轴塞进硬件轴。

---

### 3.7 `vectorize`：最内层 → SIMD

```python
s[C].vectorize(ji)   # ji 长度通常取 4/8/16 等对齐 SIMD 宽度
```

```text
【前】 for ji in 0..7:            【后】  // 语义上变成向量操作
        C[i, j0+ji] = ...                  C[i, j0:j0+8] = vector_op(...)
                                           # 编译后 → AVX/NEON 等
```

**目的**：CPU 主战场；要求内层连续、无循环携带依赖（或可识别的归约向量化）。

---

### 3.8 `unroll`：展开循环

```python
s[C].unroll(ki)      # ki 很小，如 4..16
```

```text
【前】 for ki in 0..3:            【后】 body(ki=0);
        body(ki)                           body(ki=1);
                                           body(ki=2);
                                           body(ki=3);
```

**目的**：减少循环开销、暴露 ILP、方便后续指令调度；过大 unroll 会撑爆 I-cache / 寄存器。

---

### 3.9 `parallel`：多核并行（CPU）

```python
s[C].parallel(io)
```

```text
【前】 for io in 0..31:           【后】 parallel for io in 0..31:  # OpenMP 等
        ...                                ...
```

**目的**：外层无依赖的空间轴分给多核。与 GPU 的 `bind` 不同——`parallel` 是 CPU 线程池模型。

---

### 3.10 `bind`：绑到 GPU 硬件轴

```python
bx = te.thread_axis("blockIdx.x")
tx = te.thread_axis("threadIdx.x")
s[C].bind(jo, bx)
s[C].bind(ji, tx)
```

```text
【前】 for jo:                    【后】  // 伪 CUDA
        for ji:                           jo = blockIdx.x;
          ...                             ji = threadIdx.x;
                                          ...
```

**目的**：把逻辑循环映射到 CUDA/OpenCL 的 block/thread（或等价物）。**GPU schedule 的核心原语。**

---

### 3.11 `cache_read`：为读入插缓存 stage

```python
AA = s.cache_read(A, "shared", [C])   # 在 C 读 A 的路径上插 shared 副本
```

```text
【前】 每层循环直接读 A[global]     【后】 在某层（由 compute_at 决定）：
                                         shared AA[...] = A[...]   # 协作搬运
                                         __syncthreads()
                                         计算阶段读 AA
```

**目的**：把全局内存数据提升到 shared/local；配合 `compute_at` 控制**何时**填充缓存。  
GPU 上多线程协作填同一块 shared = **cooperative fetching**（论文 Figure 7 的关键优化）。

---

### 3.12 `cache_write`：为写出插本地累加缓冲

```python
CL = s.cache_write(C, "local")
```

```text
【前】 归约直接读改写 C[global]     【后】 在 local CL 上完成累加
                                         最后再写回 C[global]
```

**目的**：减少对全局输出的反复 RMW；寄存器/本地内存上累加，块结束一次性 store。

---

### 3.13 `compute_at`：把某 stage 沉到另一 stage 的某层循环内

```python
s[CL].compute_at(s[C], jo)
```

```text
【前】（默认 compute_root）
      先完整算完 CL 再算 C         【后】 for io:
                                        for jo:
                                          // 在此计算 CL 的对应 tile
                                          用 CL 写 C 的 tile
```

**目的**：控制**计算粒度 / 生产者-消费者局部性**（Halide 的 call schedule 思想）。  
与 `cache_read`/`cache_write` 几乎总是成对出现：先插入 cache stage，再用 `compute_at` 指定它在哪一层落地。

| `compute_at` 附着点 | 效果直觉 |
|---------------------|---------|
| 很外层 | 一次算一大块，占用存储大，复用可能好 |
| 很内层 | 细粒度融合，存储小，可能重复计算或同步更频 |

---

### 3.14 综合例：matmul naive vs tiled + cache_write + vectorize

与根 README 动手验收一致；论文笔记 [`05-tvm.md`](./paper-notes/05-tvm.md) §4.1 有同构代码。

```python
import tvm
from tvm import te

M = N = K = 1024
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
k = te.reduce_axis((0, K), name="k")
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

# ----- schedule A：naive -----
s0 = te.create_schedule(C.op)
print("=== NAIVE ===")
print(tvm.lower(s0, [A, B, C], simple_mode=True))

# ----- schedule B：tile + cache_write + compute_at + split + vectorize -----
s = te.create_schedule(C.op)
CL = s.cache_write(C, "local")
io, jo, ii, ji = s[C].tile(C.op.axis[0], C.op.axis[1], 32, 32)
s[CL].compute_at(s[C], jo)
ko, ki = s[CL].split(s[CL].op.reduce_axis[0], factor=8)
s[CL].vectorize(ki)   # 示例：对拆出的内层标记向量化；实际轴选择以 lower 结果为准
print("=== TILED ===")
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

**TIR 风格结构对比（示意）：**

```text
======================== NAIVE ========================
for i in 0..1023:
  for j in 0..1023:
    C[i,j] = 0
    for k in 0..1023:
      C[i,j] += A[i,k] * B[k,j]


=========== TILED + cache_write + vectorize ===========
for io in 0..31:
  for jo in 0..31:
    # compute_at：该 32×32 tile 的累加发生在这里
    local CL[32][32] = 0
    for ko in 0..127:              # K 维分块
      for ii in 0..31:
        for ji in 0..31:
          for ki in 0..7:          # vectorize → SIMD
            CL[ii,ji] += A[...] * B[...]
    for ii in 0..31:
      for ji in 0..31:
        C[io*32+ii, jo*32+ji] = CL[ii,ji]   # 整块写回
```

| 原语 | 在本例中的作用 |
|------|---------------|
| `tile` | 形成 32×32 输出块，便于缓存与写回 |
| `cache_write` | 累加打在 `CL` 上，避免对 `C` 反复全局 RMW |
| `compute_at` | 让 `CL` 的生命周期绑在每个 `(io,jo)` 块上 |
| `split`（K） | 为内层向量化/流水露出 `ki` |
| `vectorize` | 最内层变 SIMD |

跑通后请自己用 `tvm.lower` 的**真实输出**核对轴名字与嵌套——不同 TVM 小版本打印细节可能不同，**结构差异**才是验收点。

### 3.15 原语速查表

| 原语 | 循环/结构变化 | 主战场 |
|------|--------------|--------|
| `split` | 1 轴 → outer×inner | 全后端 |
| `tile` | 两轴分块 | 全后端 |
| `reorder` | 改嵌套顺序 | 全后端 |
| `fuse` | 多轴 → 1 轴 | GPU 索引 / 并行 |
| `vectorize` | 内层 → SIMD | CPU |
| `unroll` | 循环展开 | CPU/GPU |
| `parallel` | CPU 多线程 | CPU |
| `bind` | 逻辑轴 → block/thread | GPU |
| `cache_read` | 插入读缓存 stage | GPU/加速器 |
| `cache_write` | 插入写缓存 stage | GPU/加速器/CPU |
| `compute_at` | 改 stage 计算附着点 | 全后端 |
| `tensorize` | 循环块 → 张量指令（下一章） | 异构后端 |

---

## 第 4 章 tensorize：异构后端的钩子

### 4.1 问题：向量化不够用了

- `vectorize`：把**一维、定长**的标量运算打成 SIMD（AVX、NEON）。
- 现代加速器还有 **矩阵/张量指令**：NVIDIA TensorCore（如 m16n16k16 MMA）、AMX、自研 NPU 的 `gemm8×8`、低比特 bit-serial 微内核等。
- 这些指令的操作数是**小块多维张量**，还常常要求特定 layout / 对齐——不是 `vectorize` 能表达的。

### 4.2 机制：用 TE 声明 intrinsic，用 `tensorize` 替换循环块

```
  ① 用 TE 描述「这块小计算在数学上等于什么」
        → tensor_intrin 的 compute 描述（behavior）

  ② 再声明「lowering 时变成哪条硬件调用」
        → TensorIntrin 的实现（调用 TensorCore API / 内联汇编 / 运行时 hook）

  ③ schedule 里：
        s[C].tensorize(yi, gemm_16x16_intrin)
        → 匹配的内层循环 nest 被替换为一次（或一组）张量指令
```

伪代码示意：

```text
【tensorize 前】
for yo: for xo:
  for yi in 0..15:
    for xi in 0..15:
      for ki in 0..15:
        C[…] += A[…] * B[…]

【tensorize 后】
for yo: for xo:
  tensor_core_mma_16x16x16(&C[…], &A[…], &B[…])
  // 内层 16×16×16 循环消失，换成一条（或微码序列）硬件指令
```

### 4.3 为什么这是「异构后端接入」的钩子

| 做法 | 代价 |
|------|------|
| 为每种张量指令改编译器核心 | 不可扩展 |
| **新增一份 `tensor_intrin` 声明 + 在 schedule/MetaSchedule 规则里 `tensorize`** | 论文称接入一类新 FPGA 加速器可低至 ~2k LoC 量级（见 [`05-tvm.md`](./paper-notes/05-tvm.md)） |

厂商或系统集成方需要提供的，往往是：

1. 指令的**语义**（可用 TE 写清）；
2. **lowering** 到驱动/ISA；
3. 可选：MetaSchedule 的 **schedule rule**，让搜索自动尝试 tensorize。

这与 Glow「把节点 lower 到少量线性代数原语、厂商实现原语」是另一条对照路线（见根 README §5 Glow）——TVM 让厂商挂的是**可搜索的张量 intrinsic**，而不是只实现固定 BLAS 列表。

### 4.4 和 layout 的耦合

TensorCore 类指令几乎总是绑定特殊 layout（fragment、swizzle、行/列主序）。因此：

```
图级 layout 传播  →  算子输入已是 MMA 友好布局
        ↓
schedule tile 尺寸 对齐 16×16×16（或硬件要求）
        ↓
tensorize 才能匹配成功
```

layout 决策与 tensorize **不独立**——这又回到第 2.3 节：图级 layout 在塑造算子级搜索空间。

---

## 第 5 章 AutoTVM / Ansor / MetaSchedule：搜索闭环

> **tuning log 复用 = 研究问题⑥「配置组合爆炸」的工程答案。**

### 5.1 为什么必须搜索

对一份 matmul TE，仅 `tile` 因子 × `unroll` × 是否 `vectorize` × cache 位置……就可以到**亿～百亿**级合法配置。人手只能试几个点；穷举实测不可行（每次实测毫秒～秒级）。

### 5.2 统一闭环（三个名字，一套骨架）

```
  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
  │ 搜索空间     │ →  │ ML cost model│ →  │ 选一批候选   │
  │ (template /  │    │ 预测相对快慢  │    │ 上真机 measure │
  │  生成规则)   │    │ (毫秒级)      │    └──────┬──────┘
  └─────────────┘    └──────────────┘           │
         ▲                                       ▼
         │                              ┌────────────────┐
         │                              │ 写入 tuning log │
         │                              │ / database      │
         │                              └────────┬───────┘
         │                                       │
         └──── 用新实测更新模型；部署时 apply_best ─┘
```

| 阶段 | 做什么 | 对应问题 |
|------|--------|---------|
| **定义空间** | template knob，或 Ansor/MetaSchedule 自动生成 schedule 候选 | 组合爆炸的"定义域" |
| **cost model** | 从 TIR/特征预测**排序**而非精确周期 | 避免每次都上真机 |
| **explorer** | 模拟退火 / 进化 / 梯度式采样等 | 在空间里走 |
| **device measure** | RPC 或 LocalRunner 真机跑 | 校正模型偏差 |
| **log reuse** | `apply_history_best` / database 命中 | **别为同一 (硬件,算子,shape) 再调一遍** |

### 5.3 AutoTVM：模板式（论文原版思路）

开发者写 `@autotvm.template`，用 `cfg.define_split` 等声明 knob：

```python
@autotvm.template("tutorial/matmul")
def matmul_template(M, N, K, dtype):
    # ... TE 定义 ...
    s = te.create_schedule(C.op)
    cfg = autotvm.get_config()
    cfg.define_split("tile_y", cfg.axis(M), num_outputs=2)
    cfg.define_split("tile_x", cfg.axis(N), num_outputs=2)
    # apply split → reorder → …
    return s, [A, B, C]

tuner = autotvm.tuner.XGBTuner(task)
tuner.tune(n_trial=200, measure_option=...,
           callbacks=[autotvm.callback.log_to_file("matmul.log")])

with autotvm.apply_history_best("matmul.log"):
    # 编译时直接复用已搜到的最优配置
    func = tvm.build(...)
```

- **优点**：空间可控、与手写 schedule 经验结合紧。  
- **缺点**：每个算子要写/维护 template。  
- **XGBoost 特征工程**：知道"用循环结构特征做 rank 模型"即可，**细节先跳过**（根 README 明确跳过）。

### 5.4 Ansor（auto_scheduler）：模板无关

```python
@auto_scheduler.register_workload
def matmul_add(...):
    # 只注册 TE workload，不写 schedule template
    return [A, B, C]

task = auto_scheduler.SearchTask(func=matmul_add, args=..., target=...)
task.tune(TuningOptions(num_measure_trials=200,
          measure_callbacks=[auto_scheduler.RecordToFile("matmul.json")]))
sch, args = task.apply_best("matmul.json")
```

搜索空间由编译器**自动推导**（sketch + 随机推导等）。代价是空间更大、有时更难 constraining。

### 5.5 MetaSchedule：当前统一框架

- 作用对象转向 **TensorIR / IRModule**（并可挂到 Relax pass：`MetaScheduleTuneIRMod` / `MetaScheduleApplyDatabase`）。
- 用 **Schedule Rule / Postproc / Cost Model / Database** 插件化，吸收 AutoTVM（可约束）与 Ansor（可生成）两派。
- 社区策略：**新调优工作优先 MetaSchedule**；TE-schedule API 不再作为新功能主战场。

对学习路径：先跑通 AutoTVM + log 复用（验收任务），再读官方 MetaSchedule 教程把名词映射过去即可。

### 5.6 Tuning log 复用：怎么回答「配置组合爆炸」

```
  请求：(target=llvm -mcpu=xxx, workload=matmul, shape=1024^3, dtype=fp32)
           │
           ▼
      查 database / matmul.log
           │
     ┌─────┴─────┐
     │ hit       │ miss
     ▼           ▼
  apply_best   触发 tune（昂贵）
  直接 build    → 写回 log → 下次 hit
```

工程含义（算力网视角）：

- 索引键 ≈ `(硬件型号/target 指纹, 算子签名, shape/dtype, …)`
- 集群内**共享** tuning database = 把一次性搜索成本摊到所有部署
- 迁移学习（相关 workload 热启动 cost model）= 进一步降 miss 成本

这与"每个节点每次部署从零搜"相反——**爆炸的是理论空间，不是每次线上请求的实际成本**。

---

## 第 6 章 Runtime：PackedFunc ABI + Graph/VM Executor

### 6.1 PackedFunc：统一调用 ABI

问题：CPU kernel 是 `.so` 里的函数，GPU 是 launch 配置 + 驱动 API，加速器可能是提交指令流——若 runtime 按后端写 `switch`，每加一后端就改核心。

**PackedFunc** 的答案：

```
  任意后端产物
       │  包装
       ▼
  PackedFunc：统一的参数打包 / 调用约定
       │
       ├─ Python 可直接调
       ├─ C++ 可注册/调用
       └─ 可序列进模块，跨进程/RPC 派发
```

直觉：它是 TVM 运行时的"**函数插件接口**"——异构执行差异被藏在 PackedFunc 实现里，调度器只看见"调用一个函数"。

### 6.2 Graph Executor（经典部署路径）

`relay.build` 产物逻辑上是三元组：

| 部件 | 内容 |
|------|------|
| **graph** | 优化后的图（JSON 等），节点指向算子名 |
| **lib** | 各节点对应的编译好的 PackedFunc |
| **params** | 权重 |

```
  GraphModule.run():
      按拓扑序 for node in graph:
          inputs = 物化好的 NDArray
          lib.get_function(node.op)(inputs..., outputs...)
```

特点：简单、开销可预测、极适合静态图推理。动态控制流弱 → 复杂模型可走 **VM**。

### 6.3 VM 路径（Relay VM / Relax VM）

```
  字节码 / 可执行指令流
       │
       ▼
  VirtualMachine：解释或执行指令
       ├─ 分配寄存器式的值槽
       ├─ 调用 PackedFunc 做重计算
       └─ 支持更灵活的控制流与动态 shape（尤其 Relax VM）
```

| | Graph Executor | VM |
|--|----------------|-----|
| 模型 | 静态 DAG | 更通用的程序 |
| 动态 shape | 弱 | Relax VM 更强 |
| 学习要求 | **必懂思想** | 知道有这条路径 |

### 6.4 RPC：调优与部署共用的远程通道

```
  开发机：编译 / 选候选 schedule
       │  RPC
       ▼
  设备池：真机 measure 或直接推理
       │
       ▼
  回传时延数字 / 输出张量
```

同一套 RPC：既服务 AutoTVM 的 device measure，也服务"交叉编译后扔到板子上跑"。嵌入式场景几乎必用。

---

## 第 7 章 与 AI-Infra 目标的对接（含 vs IREE/MLIR）

自学枢纽见 [`docs/README.md`](./README.md)；项目总目标见根 [`README.md`](../README.md)。

### 7.1 三个研究问题在 TVM 里的位置

| 研究问题 | TVM 给出的经典解 | 你还要补的 |
|----------|-----------------|-----------|
| ① **子图划分** | 四类算子融合规则 = 单栈内的划分；融合与 codegen 闭环 | 多 EP/多设备边界上的 partition；代价模型要含传输与转换 |
| ② **跨后端 layout** | 单设备 preference + 传播 + transform 节点 | 跨厂商协商、量化一并转换、边界放置策略 |
| ⑥ **配置组合爆炸** | 搜索空间 + cost model + **tuning log/database 复用** | 集群级共享库、跨设备迁移、命中率与冷启动 |

### 7.2 TVM vs IREE/MLIR（选型直觉）

```
                 图优化 + 算子搜索打极致性能
                    ┌──────── TVM ────────┐
                    │ Relay/Relax + TIR   │
                    │ MetaSchedule        │
                    └──────────┬──────────┘
                               │ 都要回答「异构上怎么跑」
                    ┌──────────┴──────────┐
                    │ IREE + MLIR         │
                    │ linalg→flow→stream→hal │
                    │ 调度进 AOT 产物 / HAL │
                    └─────────────────────┘
                 可插拔 dialect + 设备抽象 + 薄运行时
```

| 维度 | TVM | IREE |
|------|-----|------|
| IR 哲学 | 自建 Relay/Relax + TIR | 可扩展 MLIR dialect 栈 |
| 强项 | 自动搜索出高性能 kernel；图融合与 layout 方法论清晰 | 统一 HAL、timeline 同步、把宿主调度也编译进去 |
| 多后端 | codegen + PackedFunc 插件 | `executable` variant + HAL driver |
| 更贴近算力网「基础设施」的 | 调优/融合/layout **方法论** | **设备抽象与执行模型**（见 [`iree-learning-guide.md`](./iree-learning-guide.md)） |

**结论**：项目底座更靠近 IREE HAL；学 TVM 是为了拿满「融合规则、layout 传播、调度原语、搜索+日志复用」这套**图编译方法论**，并在自研 pass/委托策略里复用，而不是把整个 serving 锁死在 TVM runtime 上。

### 7.3 和 Glow / Halide 的阅读顺序

1. Halide 笔记：只建立 **algorithm ≠ schedule**（[`04-halide.md`](./paper-notes/04-halide.md)）。  
2. 本文 + [`05-tvm.md`](./paper-notes/05-tvm.md)：完整图编译 + 搜索。  
3. Glow：对照「少量原语 + 厂商实现」vs「TE + 搜索 + tensorize」。

---

## 第 8 章 学习路径：最小必要集与动手清单

### 8.1 必须掌握

1. **端到端流水线**：前端 → Relay/Relax → 图优化 → TE/TIR → CodeGen → PackedFunc + executor（能默画第 1 章总图）。  
2. **四类融合 + 规则**：injective / reduction / complex-out-fusable / opaque；能举 `conv+bias+relu` 与 `sort` 切开的例子。  
3. **layout 传播**：偏好 → 传播 → 冲突插 transform；能说到研究问题②。  
4. **全部列出的 schedule 原语**：对每个都能画出循环前后差（第 3 章）。  
5. **matmul 两 schedule**：naive vs tiled+cache_write+vectorize，能对着 `tvm.lower` 讲。  
6. **tensorize**：循环块 → 张量指令；异构接入钩子。  
7. **调优闭环**：空间 → cost model → measure → **log 复用**（问题⑥）。  
8. **PackedFunc + graph executor** 如何派发。  
9. **现代分工**：TE-compute 留、TE-schedule 心智保留但新工作走 TensorIR+MetaSchedule；Relax ≈ 动态 shape。

### 8.2 先跳过（根 README §4.1）

| 跳过项 | 理由 |
|--------|------|
| VTA/FPGA 微架构细节 | 懂 memory scope + tensorize + 显式同步即可 |
| XGBoost 特征工程细节 | 懂 rank cost model 角色即可 |
| Relay pass 全集 | 会用 `opt_level` / 知道融合与 layout 在图级发生即可 |
| TVM 源码目录漫游 | 不助力验收 |
| Relax 深入（符号 shape 全套 API） | 知道动机；动态 shape 深水区以后再挖 |
| virtual thread / DAE 实现细节 | 论文概念扫一眼；非本阶段动手项 |

### 8.3 动手清单（按顺序）

**环境**：按官方文档安装 Apache TVM（带 LLVM 的构建或官方 pip 轮子，视平台而定）。

**推荐入口（仓库动手项目）**：[`../tvm-fatbin-lab/`](../tvm-fatbin-lab/)

```bash
cd tvm-fatbin-lab && pip install -r requirements.txt && bash scripts/run_tvm.sh
# 先读 out/ANALYSIS.md 与 out/tvm/01_READING.md …
```

覆盖：两 schedule lower 对比、Relay 融合、layout、tensorize、AutoTVM + log 复用、PackedFunc。

**若只想手写最小复现**（不跑完整 lab）：

1. TE matmul：naive vs tiled+cache_write+vectorize，`tvm.lower(..., simple_mode=True)` 对照。  
2. AutoTVM 小 template，`n_trial` 不必大；`apply_history_best` 复用 log。  
3. （可选）对照 [`iree-learning-guide.md`](./iree-learning-guide.md) 画「同一 matmul 在 TVM 与 IREE 各经哪几层」。

### 8.4 过关标准（组会能讲清）

- [ ] 四类融合：各举一例能融 / 不能融，并解释为什么。  
- [ ] 指着两份 `lower` 输出讲清五个以上原语。  
- [ ] 说明 tensorize 与 `vectorize` 差在哪，为何能接 TensorCore。  
- [ ] 画出调优闭环，并强调 **tuning log 复用** 如何对抗配置爆炸。  
- [ ] 一句话对比 TVM 与 IREE：谁擅长搜 kernel，谁擅长设备抽象与 AOT 调度。

---

## 附录：一页速查

```
【流水线】
  Frontend → Relay/Relax → 图优化(融合/layout/常量/内存规划)
           → TE compute → (TE schedule | TensorIR+MetaSchedule) → TIR
           → CodeGen → Runtime(PackedFunc + GraphExecutor/VM)

【现代策略】
  TE-compute：保留
  TE-schedule：新工作弃用 → TensorIR + MetaSchedule
  Relay：静态；Relax：动态 shape（深挖可跳过）

【融合四类】
  injective           一对一 / element-wise    ←→ 彼此可融
  reduction           归约                     ←  吃输入侧 injective
  complex-out-fusable conv/gemm                ←  吃输出侧 injective*
  opaque              sort 等                  ✗  不跨融

【layout】
  算子声明 preferred layout → 传播 → 冲突边插入 layout_transform
  → 研究问题② 的单设备原型

【schedule 原语 → 循环】
  split     1轴→outer×inner
  tile      两轴分块
  reorder   改嵌套顺序
  fuse      多轴→1轴
  vectorize 内层→SIMD
  unroll    展开
  parallel  CPU 多核
  bind      → blockIdx/threadIdx
  cache_read  / cache_write   插入缓存 stage
  compute_at  缓存/生产者沉到消费者某层循环
  tensorize   循环块→TensorCore/矩阵指令   ★ 异构钩子

【调优闭环】
  空间(template|Ansor|MetaSchedule) → ML cost model → 真机 measure
  → tuning log/database → apply_best 复用     ★ 问题⑥

【Runtime】
  PackedFunc = 统一 ABI（跨语言/后端）
  GraphExecutor = 拓扑序调 PackedFunc
  VM = 更灵活控制流 / Relax 动态 shape

【vs IREE】
  TVM：融合+layout+搜索方法论
  IREE：MLIR 分层 + HAL 设备/同步/AOT 调度
  算力网底座偏 IREE；方法论必修 TVM

【动手验收】
  ① matmul：naive vs tiled+cache_write+vectorize，对比 lower
  ② AutoTVM 一次 + apply_history_best 复用 log

【交叉链接】
  横切概念  ai-compiler-foundations.md §3 §4
  论文动机  paper-notes/05-tvm.md
  Halide    paper-notes/04-halide.md
  IREE      iree-learning-guide.md
  委托对照  onnx-learning-guide.md · executorch-learning-guide.md
  导航      docs/README.md
```

---

## 维护约定

- 官方 TE / TIR / MetaSchedule 策略变更时，优先同步第 3、5 章与附录速查。  
- 融合分类与「分区边界代价」的表述以 [`ai-compiler-foundations.md`](./ai-compiler-foundations.md) §3 为准，改动时同步 [`onnx-learning-guide.md`](./onnx-learning-guide.md) §7.3 与 [`executorch-learning-guide.md`](./executorch-learning-guide.md) §5。  
- 新增动手脚本入口补到 [`docs/README.md`](./README.md) 阶段 4 与根 [`README.md`](../README.md) §4.1。

---

*文档版本：与仓库 AI-Infra 自学体系同步；API 细节以 https://tvm.apache.org/docs/ 当前稳定文档为准。小版本之间 `tile` 返回轴顺序、`vectorize` 作用轴等打印细节可能变化，以你本地 `tvm.lower` 为准核对结构。*
