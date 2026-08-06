# ExecuTorch 学习文档：委托机制 + Partitioner 深入详解

> **本文档的定位**
> - 基于 ExecuTorch **官方文档**（https://docs.pytorch.org/executorch）整理，不是论文笔记。
> - 目标读者是"要在算力网上搭分布式大模型执行基础设施"的编译器/系统工程师。
> - **重点是 Backend Delegation 与 Partitioner**（约 55% 篇幅）：`to_backend` 三条流、分区边界代价、与 ORT EP / IREE `flow.dispatch` 的对照。
> - 与根目录 [`../README.md`](../README.md) **§4.2**（ONNX + 多后端委托）和 **§6**（六个研究问题）直接对齐；自学路径见 [`README.md`](./README.md) 阶段 4。
> - **先修**：[`ai-compiler-foundations.md`](./ai-compiler-foundations.md) §3.3（子图划分与四类边界代价）；图级融合背景见 [`tvm-learning-guide.md`](./tvm-learning-guide.md) §2.2。
> - **动手项目**：[`../onnx-delegate-lab/`](../onnx-delegate-lab/) —— `bash scripts/run_executorch.sh`（或随 `scripts/run.sh`），先读 `out/ANALYSIS.md`。
>
> **一句话读法**：如果只有一小时，读第 1 章坐标系、第 4 章 Backend Delegation、第 5 章分区边界代价、第 8 章三系统对照。
>
> **主要信息源**
> - 委托与分区：https://docs.pytorch.org/executorch/main/compiler-delegate-and-partitioner.html
> - `to_backend` 三条流：https://docs.pytorch.org/executorch/main/examples-end-to-end-to-lower-model-to-delegate.html
> - EXIR / Edge Dialect：https://docs.pytorch.org/executorch/main/ir-exir.html
> - 后端总览：https://docs.pytorch.org/executorch/main/backends-overview.html
> - 交叉阅读：[`onnx-learning-guide.md`](./onnx-learning-guide.md)、[`iree-learning-guide.md`](./iree-learning-guide.md)、动手 [`../onnx-delegate-lab/`](../onnx-delegate-lab/)

---

## 目录

- [第 1 章 ExecuTorch 是什么：一分钟建立坐标系](#第-1-章-executorch-是什么一分钟建立坐标系)
- [第 2 章 编译流水线：从 PyTorch 到 .pte](#第-2-章-编译流水线从-pytorch-到-pte)
- [第 3 章 Edge Dialect：为什么要单独一层](#第-3-章-edge-dialect为什么要单独一层)
- [**第 4 章 Backend Delegation 详解**](#第-4-章-backend-delegation-详解重点)
  - [4.1 入口：`to_backend` 与双端接口](#41-入口to_backend-与双端接口)
  - [4.2 三条委托流](#42-三条委托流)
  - [4.3 Partitioner 契约：只打标签，不改结构](#43-partitioner-契约只打标签不改结构)
  - [4.4 从 tag 到 blob：融合、preprocess、LoweredBackendModule](#44-从-tag-到-blob融合preprocessloweredbackendmodule)
  - [4.5 运行时：`call_delegate` / init / execute](#45-运行时call_delegate--init--execute)
- [第 5 章 分区边界代价：研究问题①②的物理根源](#第-5-章-分区边界代价研究问题①②的物理根源)
- [第 6 章 多后端委托的两种写法](#第-6-章-多后端委托的两种写法)
- [第 7 章 最小 Partitioner 示例（概念代码）](#第-7-章-最小-partitioner-示例概念代码)
- [第 8 章 对照表：ExecuTorch / ORT EP / IREE flow.dispatch](#第-8-章-对照表executorch--ort-ep--iree-flowdispatch)
- [第 9 章 与 AI-Infra 项目及六个研究问题](#第-9-章-与-ai-infra-项目及六个研究问题)
- [第 10 章 学习路径：必学 / 跳过 / 动手清单](#第-10-章-学习路径必学--跳过--动手清单)
- [附录：一页速查](#附录一页速查)

---

## 第 1 章 ExecuTorch 是什么：一分钟建立坐标系

**ExecuTorch** 是 PyTorch 官方的**端侧 / 边缘部署栈**：把 `torch.nn.Module` 编译成可移植的 `.pte`，在轻量 C++ 运行时上执行；同时通过 **Backend Delegation** 把子图交给 NPU / DSP / GPU 等专用后端。

一句话记住它和别人的区别：

> **ONNX Runtime** 以交换格式 + Execution Provider 分区为核心；  
> **IREE** 把模型整段编译成带调度的程序（`.vmfb`），用 HAL 统一设备；  
> **ExecuTorch** 守着 PyTorch 原生图（Export IR → Edge Dialect），用 `Partitioner` 打标签、用 `to_backend` 把连通子图换成不透明的 delegate blob。

由此推出 ExecuTorch 几乎所有的设计决定：

| 决定 | 原因 |
|------|------|
| 入口是 `torch.export`，不是 ONNX | 与 PyTorch 生态同源，保留 ATen 语义与 debug handle |
| 中间有 Edge Dialect | 端侧需要**可移植、dtype 特化、无 Scalar** 的算子集，又不想过早绑死某一硬件 |
| 委托用 `delegation_tag` 而非改图 | Partitioner 只声明"谁去哪"，结构变换交给框架，避免后端互踩 |
| 产物是 `.pte`（FlatBuffer） | 一张图里混着 portable kernel 调用 + 多个 backend blob |
| 运行时看见 `call_delegate` | 委托子图对 ExecuTorch 不透明，由 backend 的 `init`/`execute` 接管 |

### 三个必须先分清的层次

```
┌────────────────────────────────────────────────────────────┐
│  编译侧（Python / EXIR）                                    │
│  torch.nn.Module                                           │
│       → torch.export          （ATen Dialect / Export IR） │
│       → to_edge               （Edge Dialect）             │
│       → to_backend / Partitioner （可选，Backend Dialect） │
│       → to_executorch         （序列化）                   │
└────────────────────────────────────────────────────────────┘
                          │ 产物 .pte
                          ▼
┌────────────────────────────────────────────────────────────┐
│  ExecuTorch Runtime（轻量宿主）                             │
│  解释 / 派发 portable 算子；遇到 call_delegate 交给 backend │
└────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────┐
│  Backend Delegate（厂商 / 官方）                            │
│  AOT: partition + preprocess → binary blob                 │
│  RT:  is_available / init / execute / (destroy)            │
└────────────────────────────────────────────────────────────┘
```

**关键认知**：对根 README §4.2 而言，ExecuTorch 的价值不在"又一个部署框架"，而在它把**子图划分接口**暴露成干净的 `Partitioner`——你亲手写一次最小 Partitioner，六个研究问题就从纸面句子变成可观察的工程现象。

---

## 第 2 章 编译流水线：从 PyTorch 到 .pte

官方端到端路径可以压成一行：

```
PyTorch model → torch.export → Edge Dialect → (optional) to_backend/Partitioner → .pte → Runtime
```

展开成相位表：

| 阶段 | API / 产物 | 回答的问题 | 核心约束 |
|------|-----------|-----------|---------|
| **Eager 模型** | `torch.nn.Module` | 用户写什么 | 动态 Python，尚无稳定图 |
| **ATen Dialect** | `torch.export(...)` → `ExportedProgram` | 计算图是什么 | functionalize；连续内存；ATen / HOOP / 已注册 custom op |
| **Edge Dialect** | `to_edge(...)` → `EdgeProgramManager` | 端侧可移植子集是什么 | Edge op（dtype 特化）；Scalar→Tensor；仍**硬件无关** |
| **Backend Dialect**（可选） | `to_backend` / backend pass | 哪些子图交给谁 | `delegation_tag` → 子图 → `preprocess` → blob；或 backend-specific op |
| **ExecuTorch Program** | `to_executorch()` → `.pte` | 怎么装进设备 | FlatBuffer：指令流 + 常量 + delegate blob |

最小可跑骨架（无委托）：

```python
import torch
from torch.export import export
from executorch.exir import to_edge

class M(torch.nn.Module):
    def forward(self, x, y):
        return (x + y) * y

ep = export(M(), (torch.randn(2, 3), torch.randn(2, 3)))
edge = to_edge(ep)
prog = edge.to_executorch()
with open("model.pte", "wb") as f:
    f.write(prog.buffer)
```

带委托时，在 `to_edge` 与 `to_executorch` 之间插入：

```python
edge = edge.to_backend(MyPartitioner())   # 或 to_backend(ep, MyPartitioner())
```

新版高层封装（如 `to_edge_transform_and_lower`）把 Edge 变换与 Partitioner 捆在一起，底层仍是同一条流水线——学机制时以 `to_edge` + `to_backend` 为准。

与 [`iree-learning-guide.md`](./iree-learning-guide.md) 对照：IREE 的 `linalg → flow → stream → hal` 在**编译器内部**固化调度；ExecuTorch 的调度更多留在**运行时解释器 + 各 delegate**，编译期只决定"哪段图被哪块 blob 替换"。

---

## 第 3 章 Edge Dialect：为什么要单独一层

### 3.1 官方定位

官方 IR 文档写得很清楚：Edge Dialect 是为 **Edge 设备**引入的特化，但**刻意不进一步绑死某一硬件**——不引入新的硬件相关概念，只强化"可移植端侧子集"。

EXIR 里还有三层 dialect：

```
ATen Dialect  →  Edge Dialect  →  Backend Dialect（可选）  →  .pte
   忠实捕获         端侧可移植           目标感知 / 委托
```

### 3.2 ATen Dialect 留下什么

`torch.export` 得到的 ATen Dialect 已经是合法 Export IR，并额外保证：

1. `call_function` 是 ATen / higher-order op / 已注册 custom op；
2. 每个 op 有 meta kernel（形状推理）；
3. 输入输出 pytree-able；
4. 允许动态 dtype、隐式 broadcast / promotion；
5. 张量内存格式为 `torch.contiguous_format`；
6. **functionalization**：去掉 alias / mutation，便于后续变换。

这一层仍偏"服务器侧通用导出"——算子集大、dtype 规则松，直接当端侧 ABI 会让每个后端都实现过大的算子库。

### 3.3 Edge Dialect 多约束了什么

Edge Dialect 在 Export IR 之上再加两条硬性质：

1. **算子集**：`call_function` 必须是 **Edge Operators**（带 dtype 特化的 ATen）或已注册 custom op。dtype 约束写在 `edge.yaml`，例如 `sigmoid` 对整数输入强制产出 `Float`。这样厂商可以为**少数 dtype** 注册 kernel，缩小二进制体积。
2. **无 Scalar**：图的输入、输出、以及每个节点的入出，都不能是 Python/IR Scalar——`float`/`int` 等一律抬成 Tensor。端侧运行时不必维护一套标量路径。

内存中的载体是 `exir.EdgeProgramManager`（内含一个或多个 `ExportedProgram`）。自定义 pass 通过 `edge.transform(...)` 跑；注意此时 `node.target` 已是 **Edge op**，不是 ATen dialect 里的 `torch.ops.aten.*`。

### 3.4 为什么这层对委托至关重要

Backend Delegation 的契约建立在 **Edge Dialect 图**上：

- Partitioner 看到的是**稳定、portable、dtype 已特化**的图；
- `preprocess` 收到的也是 Edge 子图（lifted：权重常作 placeholder，可用 `get_params` 取常量）；
- 未委托的节点仍走 ExecuTorch portable kernel，与 Edge 算子集对齐。

若跳过 Edge、直接从松散 ATen 委托，每个 backend 都要自己消化 broadcast / promotion / Scalar——分区边界会更脏，研究问题②（layout / 量化 / 内存空间）会更早爆炸。

**Backend Dialect** 是 Edge 之后的可选目标感知层：可插入 backend-specific op、元数据或已 lowered 的 module。它与"整段委托成 blob"是两条互补手段——融合模式可用 pattern→backend op；大块加速仍走 delegate。细节见官方 Backend Dialect 页；本仓库优先级上**先吃透 Partitioner，再碰 backend op 注册**。

---

## 第 4 章 Backend Delegation 详解（重点）

### 4.1 入口：`to_backend` 与双端接口

官方开篇：Backend Delegation 让专用后端吃掉性能关键路径，同时用户体验仍接近 PyTorch runtime。高层上后端只对接两样东西：

1. **IR**：Edge Dialect（经 `to_edge`）；
2. **接口**：
   - **AOT**：`partition` + `preprocess`（编译期变成 blob，写入 `.pte`）；
   - **Runtime**：`is_available` / `init` / `execute` / 可选 `destroy`。

用户侧统一入口是 **`to_backend`**（定义在 `exir/backend/backend_api.py`，也挂在 `EdgeProgramManager.to_backend`）。

AOT 侧两个核心签名（概念）：

```python
def partition(exported_program: ExportedProgram) -> PartitionResult:
    ...

def preprocess(
    edge_program: ExportedProgram,
    compile_specs: List[CompileSpec],
) -> PreprocessResult:
    # 返回 compiled blob（bytes），序列化进 .pte
    ...
```

Runtime 侧（C++）：

```cpp
ET_NODISCARD bool is_available();

ET_NODISCARD virtual Result<DelegateHandle*> init(
    BackendInitContext& context,
    FreeableBuffer* processed,          // preprocess 产出的 blob
    ArrayRef<CompileSpec> compile_specs);

ET_NODISCARD virtual Error execute(
    BackendExecutionContext& context,
    DelegateHandle* handle,
    Span<EValue*> args);

virtual void destroy(ET_UNUSED DelegateHandle* handle);  // 可选
```

后端通过 `register_backend` 注册；常见做法是静态初始化时注册，链接进 runtime 即可用。

### 4.2 三条委托流

官方文档列出三条前端流。**第三条（Partitioner）最重要**——真实模型几乎总是"部分可加速"。

#### Flow 1：整模块 lowering

```python
def to_backend(
    backend_id: str,
    edge_program: ExportedProgram,
    compile_spec: List[CompileSpec],
) -> LoweredBackendModule:
```

- 输入：已是 Edge Dialect 的 `ExportedProgram`；
- 行为：直接调该 `backend_id` 的 `preprocess`，得到 blob，包成 `LoweredBackendModule`；
- 用途：测 backend / 测 preprocess；或准备一块可复用的 lowered 子模块。

```python
from executorch.exir import to_edge
from executorch.exir.backend.backend_api import to_backend
from torch.export import export

class Lowerable(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)

edge_ep = to_edge(export(Lowerable(), (torch.ones(1),))).exported_program()
lowered = to_backend("BackendWithCompilerDemo", edge_ep, [])
# lowered.buffer() 可直接落盘，或嵌回更大模型（Flow 2）
```

#### Flow 2：lowered 模块与普通模块组合

Flow 1 得到的 `LoweredBackendModule` 可以当作 `nn.Module` 嵌进更大模型；未 lowered 的部分继续走 ExecuTorch portable 路径。适合**复用**已导出的加速子模块，或手工划定"这一块永远走某 backend"。

```python
class Composite(torch.nn.Module):
    def __init__(self, lowered):
        super().__init__()
        self.accel = lowered          # 已 to_backend
        self.bias = torch.nn.Parameter(torch.ones(1) * 0.3)

    def forward(self, x):
        a = self.accel(x)
        return a + self.bias

prog = to_edge(export(Composite(lowered), (torch.ones(1),))).to_executorch()
```

#### Flow 3：经 Partitioner 的部分 lowering（主路径）

```python
def to_backend(
    edge_program: ExportedProgram,
    partitioner: Partitioner,
) -> ExportedProgram:
```

或等价地：

```python
edge = to_edge(export(model, inputs))
edge = edge.to_backend(AddMulPartitionerDemo())
prog = edge.to_executorch()
```

流程：

1. Partitioner 给节点打 `delegation_tag`，返回 `PartitionResult`；
2. `to_backend` 按 tag 抽出**连通**被标记节点，形成子图；
3. 每个子图走 Flow 1 的 `preprocess` → blob；
4. 原图中对应区域替换为 `LoweredBackendModule` / `call_delegate`。

官方示例模型（`+ * - / * +`）配合只接管 `add`/`mul` 的 `AddMulPartitionerDemo`，会在 `-`/`/` 处断开，形成**多个** delegate 子图——这正是根 README §4.2 动手验收第 3 条要观察的现象。

### 4.3 Partitioner 契约：只打标签，不改结构

`Partitioner`（`exir/backend/partitioner.py`）是抽象基类。实现者必须满足官方**强硬契约**：

| 规则 | 含义 |
|------|------|
| **只打 tag** | 意图委托的节点设置 `node.meta["delegation_tag"] = "<tag>"` |
| **禁止改程序结构** | partition 阶段**不得**改写 `ExportedProgram` 的计算结构（官方：not allowed to mutate the program；只允许为委托做必要的封装标记） |
| **tag → DelegationSpec** | 每个用过的 tag 必须出现在 `partition_tags: Dict[str, DelegationSpec]` |
| **DelegationSpec** | `(backend_id: str, compile_specs: List[CompileSpec])` |
| **同一 tag = 同一子模块意图** | 每个 tag 代表一个打算整体 lowering 的 distinct submodule，应自洽、可封闭 |

返回类型：

```python
@dataclass
class PartitionResult:
    tagged_exported_program: ExportedProgram
    partition_tags: Dict[str, DelegationSpec]
```

设计意图可以记成一句话：

> **Partitioner = 声明式能力声明 + 粗粒度亲和标注；图重写与 blob 生成是框架的事。**

这与 ONNX Runtime EP 的 `GetCapability` 同构（见第 8 章）：先回答"我能吃哪些节点/子图"，再 `Compile`。差别在于 ExecuTorch 用 **字符串 tag** 显式把"同一子图"钉死，并允许一个 Partitioner 一次打出指向**多个** `backend_id` 的 tag。

### 4.4 从 tag 到 blob：融合、preprocess、LoweredBackendModule

`to_backend` 在拿到 `PartitionResult` 后做的事：

```
图中节点
  │  meta["delegation_tag"] == "t0" 的连通分量
  ▼
子图 S0  ──preprocess(S0, compile_specs)──►  blob0
  │
  ▼
原位置替换为 LoweredBackendModule（持有 blob0 + backend_id）
运行时对应 call_delegate
```

要点：

1. **相同 tag 的连通节点融成一个子图**，送进一次 `preprocess`。不连通但同 tag 的处理以框架当前实现为准——实践中应保证"一个 tag ↔ 一块自洽子图"。
2. **不同 tag → 不同子图 → 多次 preprocess → 多个 blob**。只接管部分算子时，被 portable 算子隔开的片段会变成多个 delegate——边界越多，第 5 章的代价越大。
3. **`preprocess` 输入是 Edge 子图**。官方 demo（`BackendWithCompilerDemo`）会遍历节点，把 `add`/`mul`/`sin` 等编成字符串 blob，运行时再解析执行——教学用；真后端通常输出 TSI / 自定义二进制 / 厂商 compiler 产物。
4. **权重**：preprocess 所见常为 lifted graph，权重作输入；AOT 可用 `torch._export.utils.get_params` 取出并**打进 blob**。若要优化常量，Partitioner 需把对应 placeholder（state）一并打上 tag。

### 4.5 运行时：`call_delegate` / init / execute

`.pte` 加载后：

1. 普通 Edge op → ExecuTorch 内核库；
2. 委托点 → **`call_delegate`** 指令：runtime 按 `backend_id` 找到已注册 backend，把序列化 blob 交给 `init`，之后每次调用走 `execute(handle, args)`；
3. `destroy` 在程序生命周期结束时释放 backend 资源（可选）。

对 Developer Tools 而言，delegate 内部默认**不透明**——调试/profiling 要靠 debug handle map，把 backend 内部 handle 映射回原始子图（见官方 Delegate Debugging）。这不影响分区机制本身，但提醒：分区后"算子级"观测变难，边界选择既影响性能也影响可观测性。

---

## 第 5 章 分区边界代价：研究问题①②的物理根源

每插入一条**分区边界**（portable ↔ delegate，或 backend A ↔ backend B），运行时至少可能付出四类代价：

| 代价 | 发生原因 | 典型场景 |
|------|---------|---------|
| **数据拷贝** | 两侧不共享物理缓冲；或 ABI 要求拥有权转移 | CPU 堆 ↔ NPU SRAM / DMA 缓冲 |
| **Layout 转换** | 两侧默认 layout 不同 | NCHW ↔ NHWC；packed / blocked layout |
| **内存空间切换** | host / device / 加速器私有内存 | 跨 PCIe；跨 chip；跨进程共享失败 |
| **同步点** | 两侧执行队列不同；必须 wait 结果可见 | GPU stream vs NPU cmd queue；CPU fallback 夹心 |

根 README §4.2 原话（与 [`ai-compiler-foundations.md`](./ai-compiler-foundations.md#33-子图划分与委托partition--delegate) §3.3、[`onnx-learning-guide.md`](./onnx-learning-guide.md) §7.3 用同一套四类分类）：

> 每个分区边界都可能引入**四类代价：数据拷贝、layout 转换、内存空间切换、同步点**——**这正是研究问题①②的根源**。

映射到 [`../README.md`](../README.md) §6：

- **问题①（子图划分与跨设备传输最小化）**：Partitioner 的 tag 策略 = 图分割决策。多切一刀就多一条边传输；少切一刀可能把低效算子硬塞进加速器或错过融合。优化目标是 **compute gain − boundary cost**，不是最大化委托节点数。
- **问题②（Layout / 量化 / 内存空间兼容）**：边界上的 cast、layout transpose、dequant/quant 往往比"多跑一个 add"更贵。Campo 类工作指出 FP32↔FP16 cast 有时吃掉低精度收益——委托边界上同样成立。

**观察实验（与动手清单呼应）**：用只 tag `add`/`mul` 的 Partitioner 跑官方 `+ * - / * +` 图，数 `call_delegate` / lowered submodule 个数。每增加一个夹在中间的 `-` 或 `/`，就多一条边界——把"分区边界有代价"从口号变成计数。

与 IREE 对照：IREE 在 Flow 层形成 `flow.dispatch`，边界代价被编译进 Stream/HAL 的 copy 与 barrier；ExecuTorch 把边界留给 runtime + 各 delegate 的 buffer 约定。**问题同构，结算层不同**。

---

## 第 6 章 多后端委托的两种写法

官方 FAQ：**可以**委托给多个 backend。两条路：

### 选项 1：多次 `to_backend`

```python
# 先按 backend_1 的 Partitioner 吃掉一批节点
ep1 = to_backend(exported_program, backend_1_partitioner())
# 剩余节点再给 backend_2
ep2 = to_backend(ep1, backend_2_partitioner())
```

特点：简单、顺序固定（先到先得）；后面的 Partitioner 看不到已被替换成 `LoweredBackendModule` 的区域。适合"优先级明确：NPU > GPU > CPU"的策略。

### 选项 2：一个自定义 Partitioner 打多 backend 的 tag

```python
class Backend_1_2_Partitioner(Partitioner):
    def __init__(self) -> None:
        self.delegation_spec_1 = DelegationSpec("Backend1", [])
        self.delegation_spec_2 = DelegationSpec("Backend2", [])
        self.partition_tags = {}

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # 伪代码：按策略选节点，打不同 tag
        for node in nodes_for_backend_1:
            tag = f"b1_{id(node)}"  # 或按连通分量共用 tag
            node.meta["delegation_tag"] = tag
            self.partition_tags[tag] = self.delegation_spec_1
        for node in nodes_for_backend_2:
            tag = f"b2_{id(node)}"
            node.meta["delegation_tag"] = tag
            self.partition_tags[tag] = self.delegation_spec_2
        return PartitionResult(exported_program, self.partition_tags)
```

特点：一次遍历可做**全局**划分（考虑边界代价、避免两后端抢同一节点）；实现复杂度和 cost model 都更高——也更接近研究问题①的真实形态。

工程建议：先用选项 1 跑通双后端；需要联合优化边界时再上选项 2。

---

## 第 7 章 最小 Partitioner 示例（概念代码）

下面是一份**教学用** Partitioner：只接管 Edge 图上的 `add` / `mul`，其余留给 portable runtime。逻辑对齐官方 `AddMulPartitionerDemo` 的意图。

```python
from typing import Dict
import torch
from torch.export import ExportedProgram, export
from executorch.exir import to_edge
from executorch.exir.backend.partitioner import Partitioner
from executorch.exir.backend.backend_details import DelegationSpec, PartitionResult
from executorch.exir.dialects._ops import ops as exir_ops

# 演示用 backend_id；真实项目换成 XNNPACK / QNN / CoreML 等
_DEMO_BACKEND = "BackendWithCompilerDemo"

_ADD = exir_ops.edge.aten.add.Tensor
_MUL = exir_ops.edge.aten.mul.Tensor


class AddMulOnlyPartitioner(Partitioner):
    """只把 add/mul 打上 delegation_tag；不修改图结构。"""

    def __init__(self) -> None:
        self.delegation_spec = DelegationSpec(_DEMO_BACKEND, [])
        self.partition_tags: Dict[str, DelegationSpec] = {}

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        graph = exported_program.graph_module.graph
        partition_id = 0

        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if node.target not in (_ADD, _MUL):
                continue

            # 简单策略：每个可委托节点独立 tag → 多个小 delegate 子图
            # 更优策略：对连通的 add/mul 分量共用同一 tag，减少边界
            tag = f"addmul_{partition_id}"
            partition_id += 1
            node.meta["delegation_tag"] = tag
            self.partition_tags[tag] = self.delegation_spec

        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags=self.partition_tags,
        )


class Model(torch.nn.Module):
    def forward(self, x, y):
        x = x + y      # 可委托
        x = x * y      # 可委托
        x = x - y      # portable → 断开
        x = x / y      # portable → 断开
        x = x * y      # 可委托
        x = x + y      # 可委托
        return x


def compile_and_inspect():
    m = Model()
    inputs = (torch.randn(1, 3), torch.randn(1, 3))
    edge = to_edge(export(m, inputs))
    edge = edge.to_backend(AddMulOnlyPartitioner())
    prog = edge.to_executorch()

    # 观察点：打印图，数 LoweredBackendModule / call_delegate
    print(edge.exported_program().graph)
    # 期望直觉：- 与 / 把图切成两段 add/mul 区域 → ≥2 个 delegate 子图
    # （若每个 add/mul 独立 tag，子图数会更多——刻意用来观察边界）
    with open("addmul_partial.pte", "wb") as f:
        f.write(prog.buffer)


if __name__ == "__main__":
    compile_and_inspect()
```

**读这段代码时盯住三件事**：

1. `partition` **只写** `node.meta["delegation_tag"]` 和 `partition_tags`；
2. `-` / `/` 未打 tag → 留在 Edge portable 路径，成为**分区边界**；
3. tag 粒度决定子图个数：一节点一 tag vs 连通分量共 tag，直接改变边界数量——这是研究问题①的最小实验室。

官方还提供 helper partitioners（见 custom compiler passes 文档），便于从 decomposed op 反查源算子；写生产 Partitioner 前先扫一眼。

---

## 第 8 章 对照表：ExecuTorch / ORT EP / IREE flow.dispatch

三者都是工业界对**"子图划分 + 后端执行"**的回答，抽象层不同：

| 维度 | **ExecuTorch Partitioner** | **ONNX Runtime EP `GetCapability`** | **IREE `flow.dispatch`** |
|------|----------------------------|-------------------------------------|--------------------------|
| **输入 IR** | Edge Dialect（Export IR 子集） | ONNX Graph | Linalg（及扩展）等，经 IREE 流水线 |
| **谁声明能力** | 后端提供的 `Partitioner.partition` | EP 的 `GetCapability` | 编译器 fusion / dispatch 形成 pass（非厂商回调式） |
| **划分结果形态** | `delegation_tag` + `DelegationSpec` | 支持的子图 / 节点集合（capability） | `flow.dispatch` region（必须在设备上原子执行的代码包） |
| **编译子图** | `preprocess` → blob 写入 `.pte` | `Compile` → EP 内部可执行体 | Codegen → PTX/SPIR-V/ELF，链进 `.vmfb` |
| **运行时派发** | `call_delegate` → `init`/`execute` | Session 按分区调用各 EP | VM 调 HAL：`command_buffer.dispatch` + fence |
| **未覆盖算子** | 留在 ExecuTorch portable kernel | 回退其他 EP（常 CPU EP） | 仍在同一 IREE 程序内，换不同 executable variant / device |
| **多后端** | 多次 `to_backend` 或单 Partitioner 多 tag | 多 EP 按优先级注册 | 多 `hal.device.target` / affinity；偏"统一 HAL 多设备" |
| **边界代价落点** | 用户可见的 tag 边界 + delegate ABI | EP 分区边界 + 拷贝/layout | Stream/HAL 显式 copy、barrier、semaphore |
| **更像什么** | **声明式委托 API**（厂商友好） | **插件式运行时分区** | **编译期把划分编进程序** |

一句话对照：

- **ORT**：运行时图 + 多 EP 竞价式能力声明——见 [`onnx-learning-guide.md`](./onnx-learning-guide.md)。
- **ExecuTorch**：导出期打 tag，blob 进 `.pte`——PyTorch 原生、端侧友好。
- **IREE**：划分是编译相位的一等公民，调度与同步进 HAL——见 [`iree-learning-guide.md`](./iree-learning-guide.md) 第 2、4 章。

对算力网基础设施：三者都会逼你回答同一问题——**边界放哪、代价谁付**。ExecuTorch / ORT 把答案暴露成可插拔 API（适合研究问题①②的实验）；IREE 把答案编进程序（适合要强异步与多设备时间线时对标）。

---

## 第 9 章 与 AI-Infra 项目及六个研究问题

### 9.1 在仓库优先级里的位置

| 材料 | 角色 |
|------|------|
| 根 [`README.md`](../README.md) §4.2 | 规定 ExecuTorch 为 **P1**，与 ONNX/ORT 一起构成"多后端委托"必学块 |
| [`docs/README.md`](./README.md) 阶段 4 | 读本篇 + ONNX 指南；做最小 Partitioner；验"边界为什么在那里" |
| 本篇 | 机制教材（委托 / Partitioner / 边界代价） |
| [`onnx-learning-guide.md`](./onnx-learning-guide.md) | 交换格式 + ORT EP `GetCapability` |
| [`iree-learning-guide.md`](./iree-learning-guide.md) | 统一运行时 / HAL；对照"编译期划分" |

### 9.2 六个研究问题怎么钉在 ExecuTorch 上

| # | 研究问题（根 README §6） | 在 ExecuTorch 里的直接抓手 |
|---|-------------------------|---------------------------|
| **①** | 子图划分与跨设备传输最小化 | `delegation_tag` 策略；连通分量大小；多次 vs 联合 Partitioner；数边界个数 |
| **②** | Layout / 量化 / 内存空间兼容 | 边界上的 transpose/cast/quant；`CompileSpec` 传递 layout 偏好；preprocess 是否把转换吃进 blob |
| **③** | 动态 Shape / KV Cache | Edge/ATen 的动态维；delegate 是否要求静态特化；与 PagedAttention 块状态如何过边界 |
| **④** | 运行状态变化后低成本切后端 | 换 Partitioner / 换 `.pte` variant 的成本；blob 内状态能否块级迁移；对比 Universal Checkpointing 思想 |
| **⑤** | 后端最优图不一致 | 同一 Edge 图、不同 Partitioner → 不同融合；Backend Dialect pattern vs 整段 blob |
| **⑥** | 配置组合爆炸 | `CompileSpec` × shape × dtype × backend；缓存哪一层（Edge / 某 backend 的 preprocess log / 完整 `.pte`） |

**最值得先做的两个实验**（对应①②）：

1. 同一模型，两种 tag 粒度（逐节点 vs 最大连通 add/mul），量边界数与（若可得）拷贝次数；
2. 强制在边界插入 layout 转换节点，看 portable 段与 delegate 段的交接成本是否主导耗时。

### 9.3 和项目目标的一句话

算力网要的是"异构池上可切换的执行"。ExecuTorch 不提供分布式调度，但提供了**工业级、可插拔的子图委托词汇表**——`Partitioner` / `DelegationSpec` / `LoweredBackendModule` / `call_delegate`。把这套词汇和 IREE HAL 的 device/buffer/fence、ORT 的 EP 分区放在同一张图里，六个研究问题才有可实现的接口锚点。

---

## 第 10 章 学习路径：必学 / 跳过 / 动手清单

### 10.1 必须掌握

1. **端到端相位**：`export → to_edge → to_backend → to_executorch → .pte → runtime`。
2. **Edge Dialect 存在的理由**：portable + dtype 特化 + 无 Scalar；仍硬件无关。
3. **`to_backend` 三条流**：整模块 / 组合 / **Partitioner（主）**。
4. **Partitioner 契约**：只打 `delegation_tag`；返回 `PartitionResult`；`partition_tags: Dict[str, DelegationSpec]`；partition 期不改计算结构。
5. **同 tag → 一子图 → 一次 preprocess → 一个 blob → `LoweredBackendModule` / `call_delegate`**。
6. **分区边界四代价**：拷贝、layout、内存空间、同步——以及它们如何长成研究问题①②。
7. **多后端两写法**：串行多次 `to_backend` vs 单 Partitioner 多 `DelegationSpec`。
8. **三系统对照**：ExecuTorch tag / ORT `GetCapability` / IREE `flow.dispatch`。

### 10.2 可以先跳过

- 各厂商 backend 内部（QNN / CoreML / Vulkan / MPS 实现细节）——用到再查 [backends overview](https://docs.pytorch.org/executorch/main/backends-overview.html)。
- `edge.yaml` 全量算子表与逐 op dtype 约束。
- Delegate Debugging / ETRecord 全流程（需要查 crash 再学）。
- Backend Dialect 的 `bind_pattern_to_op` 与自定义 backend op 注册（第二部分再补）。
- 量化 API 全家桶、LLM 专用导出配方（属于负载侧，见 FlashAttention / PagedAttention 笔记）。
- C++ runtime 逐文件实现；会注册 backend、看懂 `init`/`execute` 签名即可。

### 10.3 动手清单（按顺序）

**推荐入口：仓库动手项目**

```bash
cd onnx-delegate-lab && bash scripts/run_executorch.sh
# 未装 executorch 时仍产出 per-node vs connected 边界数模拟报告
# 读 out/executorch/01_READING.md 与 02_THREE_SYSTEMS.md
```

**第一步：无委托走通**（可选加深）

```bash
python -c "import executorch; print(executorch.__file__)"
```

**第二步：最小 Partitioner（根 README §4.2 验收）**——lab 已覆盖

1. 固定模型 `+ * - / * +`（见第 7 章或 lab）；
2. 对比 per-node tag vs connected tag；
3. 记录：delegate 子图数、边界位置、是否同意「边界在 `-`/`/`」。

**第三步：对照 ORT / IREE / TVM**

- ORT：lab ONNX 轨步骤 03，或 [`onnx-learning-guide.md`](./onnx-learning-guide.md)；
- IREE：`--compile-to=flow`——[`iree-learning-guide.md`](./iree-learning-guide.md)；
- TVM 融合（单后端划分原型）：[`../tvm-fatbin-lab/`](../tvm-fatbin-lab/)；
- 写三句：三种切分的**决策时刻**与**边界代价支付方**。

**过关标准（与 [`docs/README.md`](./README.md) 阶段 4 对齐）**：

- [ ] 能默画 `export → Edge → Partitioner → blob → call_delegate`；
- [ ] 能解释为何 Partitioner 禁止在 partition 时乱改图；
- [ ] 能指着一次实验说出"有几个 delegate 子图、边界为何在那里"；
- [ ] 能用一张表对比 ExecuTorch / ORT / IREE 的划分接口；
- [ ] 能把分区边界四代价接到研究问题①②各两句。

---

## 附录：一页速查

```
【流水线】 Module → torch.export(ATen) → to_edge(Edge) → to_backend(?) → to_executorch → .pte
【Edge】   Edge op = dtype 特化 ATen；无 Scalar；硬件无关；委托契约建在这层

【to_backend 三流】
  1) (backend_id, edge_ep, compile_spec) → LoweredBackendModule     整模块
  2) 把 LoweredBackendModule 嵌回 nn.Module 再 export                组合
  3) (edge_ep, Partitioner) → 打 tag → 子图 preprocess → 替换       ★主路径

【Partitioner 契约】
  node.meta["delegation_tag"] = tag
  PartitionResult(tagged_program, partition_tags: Dict[str, DelegationSpec])
  DelegationSpec(backend_id, compile_specs)
  partition 阶段：不改计算结构，只标注

【AOT / RT】
  AOT: partition → preprocess → blob ∈ .pte
  RT:  register_backend; init(blob) → execute(handle, args); call_delegate

【边界代价】 copy · layout · memory-space · sync  → 研究问题 ①②
【多后端】   多次 to_backend  │  单 Partitioner 多 DelegationSpec

【三工业答案】
  ExecuTorch: delegation_tag + preprocess blob
  ORT:        GetCapability → Compile → EP 派发
  IREE:       flow.dispatch → stream/hal → .vmfb
```

**交叉链接**

- 动手项目：[`../onnx-delegate-lab/`](../onnx-delegate-lab/)  
- 自学枢纽：[`docs/README.md`](./README.md) 阶段 4  
- 总规划 §4.2 / §6：[`../README.md`](../README.md)  
- 横切概念（划分与边界代价）：[`ai-compiler-foundations.md`](./ai-compiler-foundations.md) §3.3 · 四栈选型对照 §3.4  
- ONNX / ORT EP：[`onnx-learning-guide.md`](./onnx-learning-guide.md)  
- IREE / HAL / dispatch：[`iree-learning-guide.md`](./iree-learning-guide.md)  
- 图级融合背景：[`tvm-learning-guide.md`](./tvm-learning-guide.md) §2.2 · 动手 [`../tvm-fatbin-lab/`](../tvm-fatbin-lab/)  
- 官方委托页：https://docs.pytorch.org/executorch/main/compiler-delegate-and-partitioner.html  

---

## 维护约定

- 官方 `to_backend` / Partitioner API 变更时，优先同步第 4、7 章与附录。  
- 分区边界代价的分类以 [`ai-compiler-foundations.md`](./ai-compiler-foundations.md) §3.3 的四类为准，改动时同步 [`onnx-learning-guide.md`](./onnx-learning-guide.md) §7.3 与根 [`README.md`](../README.md) §4.2。  
- 新增动手脚本入口补到 [`docs/README.md`](./README.md) 阶段 4。
