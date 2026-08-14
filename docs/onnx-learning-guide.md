# ONNX 学习文档：IR 全景 + ORT EP 子图划分详解

> **本文档的定位**
> - 基于 ONNX **官方规范**（[`IR.md`](https://github.com/onnx/onnx/blob/main/docs/IR.md) / [`Versioning.md`](https://github.com/onnx/onnx/blob/main/docs/Versioning.md) / [`ShapeInference.md`](https://github.com/onnx/onnx/blob/main/docs/ShapeInference.md)）与 ONNX Runtime **Execution Providers** 文档整理，不是论文笔记。
> - 目标读者是"要在算力网上做多后端委托 / 子图划分"的编译器/系统工程师。
> - **前 5 章建立 ONNX IR 坐标系（约 45% 篇幅），第 7 章是重点：ORT EP 机制（约 30% 篇幅）**，第 6、8、9 章是工程手感、研究问题对接与学习路径。
> - ExecuTorch 的 `Partitioner` / `to_backend` 是**另一篇文档**：[executorch-learning-guide.md](./executorch-learning-guide.md)。本文只点明「ORT EP 是工业界子图分区接口；PyTorch 侧委托见 ExecuTorch 文档」，不重复 Partitioner 全章。
> - **先修**：[`ai-compiler-foundations.md`](./ai-compiler-foundations.md) §3（计算图、融合、子图划分与四类边界代价）。
> - **动手项目**：[`../onnx-delegate-lab/`](../onnx-delegate-lab/) —— `bash scripts/run.sh`（ONNX 轨），先读 `out/ANALYSIS.md`。
>
> **一句话读法**：如果只有一小时，读[第 1 章坐标系](#第-1-章-onnx-是什么一分钟建立坐标系)、[2.1 结构总览](#21-总览从文件到节点)、第 7 章 ORT EP 机制、[附录速查](#附录一页速查)。
>
> **主要信息源**
> - IR 规范：https://github.com/onnx/onnx/blob/main/docs/IR.md
> - 版本策略：https://github.com/onnx/onnx/blob/main/docs/Versioning.md
> - Shape 推断：https://github.com/onnx/onnx/blob/main/docs/ShapeInference.md
> - ORT EP 总览：https://onnxruntime.ai/docs/execution-providers/
> - 根 README §4.2（必学清单与动手验收）+ §6（六个研究问题）
> - 融合对照：[paper-notes/05-tvm.md](./paper-notes/05-tvm.md) §3.1（四类算子融合）；动手对照 [`../tvm-fatbin-lab/`](../tvm-fatbin-lab/)
> - 自学枢纽：[docs/README.md](./README.md) 阶段 4

---

## 目录

- [第 1 章 ONNX 是什么：一分钟建立坐标系](#第-1-章-onnx-是什么一分钟建立坐标系)
- [第 2 章 模型结构层次详解](#第-2-章-模型结构层次详解)
  - [2.1 总览：从文件到节点](#21-总览从文件到节点)
  - [2.2 ModelProto：元数据与能力声明](#22-modelproto元数据与能力声明)
  - [2.3 GraphProto：可执行的计算图](#23-graphproto可执行的计算图)
  - [2.4 NodeProto / TensorProto / ValueInfoProto](#24-nodeproto--tensorproto--valueinfoproto)
  - [2.5 Functions：模型本地函数](#25-functions模型本地函数)
- [第 3 章 图语义：拓扑、SSA、属性 vs 输入](#第-3-章-图语义拓扑ssa属性-vs-输入)
- [第 4 章 算子集与版本管理](#第-4-章-算子集与版本管理)
- [第 5 章 Shape Inference：能推什么、推不出什么](#第-5-章-shape-inference能推什么推不出什么)
- [第 6 章 工程手感：用 onnx.helper 读写改图](#第-6-章-工程手感用-onnxhelper-读写改图)
- [**第 7 章 ONNX Runtime EP 机制**](#第-7-章-onnx-runtime-ep-机制重点)
  - [7.1 EP 在 ORT 架构中的位置](#71-ep-在-ort-架构中的位置)
  - [7.2 GetCapability → Compile → 运行时派发](#72-getcapability--compile--运行时派发)
  - [7.3 分区边界代价](#73-分区边界代价)
  - [7.4 与 ExecuTorch / IREE / TVM 的对照](#74-与-executorch--iree--tvm-的对照)
  - [7.5 动手：打印真实 EP 分区](#75-动手打印真实-ep-分区)
- [第 8 章 与 AI-Infra 六个研究问题的对接](#第-8-章-与-ai-infra-六个研究问题的对接)
- [第 9 章 学习路径：最小必要集与动手清单](#第-9-章-学习路径最小必要集与动手清单)
- [附录：一页速查](#附录一页速查)

---

## 第 1 章 ONNX 是什么：一分钟建立坐标系

**ONNX = Open Neural Network Exchange**，一套**运行时无关（runtime-agnostic）的神经网络图中间表示（IR）**。

官方 IR.md 开宗明义：

> ONNX does not pre-suppose or imply any particular method of runtime implementation.

也就是说：

| ONNX 是什么 | ONNX 不是什么 |
|------------|--------------|
| 可序列化的计算图格式（protobuf） | 一个推理引擎 / 运行时 |
| 算子契约（签名 + 语义）的标准库 | 某家框架的内部 IR |
| 框架之间交换模型的「通用语」 | 调度器、内存分配器、kernel 实现 |

一句话记住：

> **ONNX 描述「算什么」；谁来算、怎么调度、内存怎么管，全部留给实现方。**

实现可以是：解释执行的运行时（ORT）、整图代码生成（TVM / IREE）、硬件固定功能单元、或三者的组合。规范本身不偏向任何一种。

### 1.1 两个官方变体：ONNX vs ONNX-ML

| 变体 | 默认算子集 | 典型用途 |
|------|-----------|---------|
| **ONNX** | `ai.onnx`（domain `""`） | 深度神经网络：Conv、MatMul、Attention 等 |
| **ONNX-ML** | `ai.onnx` + `ai.onnx.ml` | 传统机器学习：TreeEnsemble、SVM、LabelEncoder 等 |

区别主要在**默认算子集**，不是两套互不兼容的文件格式。同一套 `ModelProto` / `GraphProto` 结构；模型通过 `opset_import` 声明自己依赖哪些 domain。做深度学习基础设施时，日常接触的几乎全是 **ONNX**（`ai.onnx`）；`ai.onnx.ml` 用到再查。

### 1.2 在 AI-Infra 自学体系里的位置

```
框架导出 / 手工构图
        │
        ▼
┌───────────────────┐
│  ONNX IR（本文）   │  ← 交换格式 + 图级语义契约
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
 ONNX Runtime   其他编译器
 (EP 分区)      (IREE / TVM / …)
    │
    └─→ 工业「子图划分」接口（第 7 章）
          │
          │  PyTorch 原生路径见
          ▼
   ExecuTorch Partitioner
   （executorch-learning-guide.md）
```

根 README §4.2 把它标为 **P1**：六个研究问题**全部从多后端委托机制长出来**；不亲手看一次分区结果，那些问题只是纸上的句子。

---

## 第 2 章 模型结构层次详解

> **速记**：为什么 ONNX 用 Protobuf？Protobuf 与 gRPC 什么关系？→ [docs/notes/protobuf-grpc-explained.md](../notes/protobuf-grpc-explained.md)

### 2.1 总览：从文件到节点

```
*.onnx 文件（protobuf 二进制）
│
└─ ModelProto                          ← 顶层：版本、opset、元数据、图
   ├─ ir_version / opset_import / …
   ├─ graph: GraphProto                ← 可执行主体
   │  ├─ input / output: ValueInfoProto[]
   │  ├─ initializer: TensorProto[]    ← 常量权重（及可选默认输入）
   │  ├─ value_info: ValueInfoProto[]  ← 中间值的类型/形状注解
   │  └─ node: NodeProto[]             ← 拓扑有序的算子调用
   │        ├─ op_type / domain
   │        ├─ input[] / output[]      ← 字符串名字（边）
   │        └─ attribute[]             ← 编译期常量参数
   └─ functions: FunctionProto[]       ← 可选：模型本地函数（可内联）
```

**记忆口诀**：Model 装元数据与能力声明；Graph 装拓扑与常量；Node 装一次算子调用；边不是指针，而是**共享的字符串名字**。

#### 示例精讲：`Gemm → Relu` 从构图到 `print(model)`

最小具体输入：一层线性 + 一个激活，权重全部只走 `initializer`。

```python
import numpy as np
import onnx
from onnx import helper, numpy_helper, checker, TensorProto

W = numpy_helper.from_array(np.zeros((3, 4), np.float32), name="W")   # Gemm 的 B
b = numpy_helper.from_array(np.zeros((4,), np.float32), name="b")     # Gemm 的 C

gemm = helper.make_node("Gemm", ["x", "W", "b"], ["h"], name="gemm0")
relu = helper.make_node("Relu", ["h"], ["y"], name="relu0")

X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])

graph = helper.make_graph([gemm, relu], "gemm_relu", [X], [Y], initializer=[W, b])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)],
                          producer_name="ai-infra-demo")
model.ir_version = onnx.IR_VERSION
checker.check_model(model)
print(model)          # protobuf 文本格式
```

`print(model)` 的关键字段（省略 `raw_data` 与空字段）：

```protobuf
ir_version: 10                      # = onnx.IR_VERSION，随本地 onnx 版本变化
producer_name: "ai-infra-demo"
graph {
  node { input: "x" input: "W" input: "b" output: "h" name: "gemm0" op_type: "Gemm" }
  node { input: "h" output: "y" name: "relu0" op_type: "Relu" }
  name: "gemm_relu"
  initializer { dims: 3 dims: 4 data_type: 1 name: "W" raw_data: "..." }
  initializer { dims: 4 data_type: 1 name: "b" raw_data: "..." }
  input  { name: "x" type { tensor_type { elem_type: 1
           shape { dim { dim_value: 1 } dim { dim_value: 3 } } } } }
  output { name: "y" type { tensor_type { elem_type: 1
           shape { dim { dim_value: 1 } dim { dim_value: 4 } } } } }
}
opset_import { version: 17 }        # 没有 domain 行 = 默认 domain 是空字符串
```

（真实输出每个字段单独一行；这里把 `node` / `shape` 压成一行省篇幅。`data_type: 1` = `TensorProto.FLOAT`。）

内存里是同构的对象树：

```text
ModelProto                                   ← model
├─ ir_version = 10
├─ opset_import[0] = OperatorSetIdProto(domain="", version=17)
├─ producer_name  = "ai-infra-demo"
└─ graph : GraphProto                        ← model.graph
   ├─ name = "gemm_relu"
   ├─ node[0] : NodeProto  op_type="Gemm"  input=["x","W","b"]  output=["h"]
   ├─ node[1] : NodeProto  op_type="Relu"  input=["h"]          output=["y"]
   ├─ initializer[0] : TensorProto  name="W"  dims=[3,4]
   ├─ initializer[1] : TensorProto  name="b"  dims=[4]
   ├─ input[0]  : ValueInfoProto  name="x"
   └─ output[0] : ValueInfoProto  name="y"
```

文本字段 ↔ 访问路径 ↔ 语义：

| 文本里看到 | Python 访问 | 它是「节点」还是「边」 |
|-----------|------------|--------------------|
| `op_type: "Gemm"` | `model.graph.node[0].op_type` | 节点：一次算子调用 |
| `output: "h"` / `input: "h"` | `node[0].output[0]` / `node[1].input[0]` | 边：同名字符串把两个节点接起来 |
| `initializer { name: "W" }` | `model.graph.initializer[0]` | 常量定义；`W` 不出现在 `graph.input` |
| `input { name: "x" }` | `model.graph.input[0]` | 唯一需要运行时喂的值 |

中间值 `h` 在 `graph.value_info` 里**没有条目**——中间值的类型注解不是必须的，要靠[第 5 章](#第-5-章-shape-inference能推什么推不出什么)的 `infer_shapes` 补出来。

> **自测**：把 `W` 也加进 `graph.input` 之后，`sess.get_inputs()` 会多出什么，模型为什么仍然合法？

### 2.2 ModelProto：元数据与能力声明

`ModelProto` 的职责不是「算」，而是让加载方在真正执行前回答：**我能不能跑这个模型？**

| 字段 | 类型 | 你必须理解的语义 |
|------|------|----------------|
| `ir_version` | int64 | **IR 格式版本**（单调整数）。决定 protobuf 字段与图语义的基线。每个模型 **MUST** 设置。 |
| `opset_import` | OperatorSetId[] | 模型依赖的算子集列表，每项是 `(domain, version)`。实现方必须支持所列算子集中用到的算子，否则应拒绝模型。 |
| `producer_name` / `producer_version` | string | 导出工具名与版本（如 `pytorch`、`tf2onnx`）。调试「谁导出的怪图」时有用。 |
| `domain` | string | 模型命名空间，建议 reverse-DNS（如 `org.onnx`）。 |
| `model_version` | int64 | **模型自身**版本（与 IR / opset 无关）。共享模型库时建议按 SemVer 约定演进。 |
| `doc_string` | string | 人类可读说明，允许 Markdown。 |
| `graph` | GraphProto | 推理（及可选训练）要执行的主图。 |
| `metadata_props` | map | 可选键值元数据；标准键有 `model_author`、`model_license`。 |
| `functions` | FunctionProto[] | 模型本地函数（见 [2.5](#25-functions模型本地函数)）。 |
| `training_info` | TrainingInfoProto[] | IR≥7 的训练扩展。**做推理基础设施可先跳过。** |
| `configuration` | DeviceConfigurationProto[] | IR≥11 的多设备配置。知道存在即可。 |

**加载时的第一道检查**（对应 `onnx.checker`）：

1. `ir_version` 是否在本库支持范围内；
2. 每个 `opset_import` 是否有对应实现；
3. 图中每个 node 的 `(domain, op_type)` 能否在已导入 opset 中解析到合法声明。

### 2.3 GraphProto：可执行的计算图

| 字段 | 说明 |
|------|------|
| `name` | 图名；**MUST** 有。 |
| `node` | 节点列表，**MUST 按拓扑序**（见第 3 章）。 |
| `initializer` | 命名常量张量（权重、偏置等）。 |
| `input` / `output` | 图边界的 `ValueInfoProto`；主图 **MUST** 给出类型，且 **MUST 有 shape（至少 rank）**。 |
| `value_info` | 中间值的类型/形状注解（shape inference 常写到这里）。 |
| `doc_string` / `metadata_props` | 文档与可选元数据。 |

#### ★ 关键陷阱：`initializer` vs `graph.input`

这是工程里**最高频的坑**，官方 IR.md 写得很清楚：

| 名字出现位置 | 含义 |
|-------------|------|
| **只在 `initializer`** | 真正的常量；调用方**不应**覆盖。权重几乎都该这样。 |
| **只在 `graph.input`** | 运行时必须由调用方提供的输入。 |
| **同时出现在两者** | initializer 提供**默认值**；运行时 **MAY** 覆盖，也 **MAY** 省略输入而用默认值。 |

```
错误心智：  「initializer = 所有权重，input = 所有输入」
正确心智：  「Value 命名空间里的一个名字可以有定义（input / initializer / node output）
             与使用（node input / graph output）。
             权重名通常只出现在 initializer，不要放进 graph.input。」
```

导出工具若把权重同时放进 `input` 和 `initializer`，推理时会出现「多了一堆假输入」——Session 的 `get_inputs()` 会列出权重名，调用方极易搞错。检查真实模型时，第一件事往往是：

```python
weight_names = {t.name for t in model.graph.initializer}
input_names  = {i.name for i in model.graph.input}
print("真正需要喂的输入:", input_names - weight_names)
print("可覆盖的默认输入:", input_names & weight_names)
print("纯常量权重:", weight_names - input_names)
```

### 2.4 NodeProto / TensorProto / ValueInfoProto

#### NodeProto —— 一次算子调用

| 字段 | 说明 |
|------|------|
| `op_type` | 算子名（大小写敏感），如 `Conv`、`Add`。 |
| `domain` | 算子所属 domain；默认 ONNX 算子为 `""`。 |
| `input` | 输入值名字列表（位置对应算子签名）。空字符串表示省略的 optional 输入。 |
| `output` | 输出值名字列表；**在图内必须唯一**（SSA）。 |
| `attribute` | 命名属性（编译期常量）。 |
| `name` | 可选节点名，仅诊断用。 |
| `overload` | IR≥10，用于区分同名函数的不同实现体。 |

节点**不持有边对象**；边由「某个 node 的 output 名 = 另一个 node 的 input 名」隐式建立。

#### ValueInfoProto —— 类型与形状注解

```
ValueInfoProto
  name: string
  type: TypeProto          ← 含 element type + TensorShapeProto
  doc_string: string
```

主图的 input/output **必须**带类型与 shape（rank 已知；具体 dim 可以是符号或未知）。中间张量的注解放在 `value_info`，不是必须，但对编译器 / 调试极有价值。

#### TensorProto —— 具体张量数据

用于 `initializer`、以及属性里的张量常量。关键点：

- `dims` + `data_type` + 数据字段（`float_data` / `raw_data` / …）；
- 大权重可用 **external data**（另文件 + offset/length），避免把 GB 级权重塞进单个 protobuf。

#### 示例精讲：一个 Conv 节点的三件套 proto

最小具体输入：`img[N,3,32,32] → Conv(3×3, 8 输出通道, pad=1) → feat`。

```python
conv = helper.make_node(
    "Conv", ["img", "conv_w", "conv_b"], ["feat"], name="conv0",
    kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1,
)
```

**NodeProto**（`print(conv)`，属性按名字排序）：

```protobuf
input: "img"
input: "conv_w"
input: "conv_b"
output: "feat"
name: "conv0"
op_type: "Conv"
attribute { name: "dilations"    ints: 1 ints: 1                 type: INTS }
attribute { name: "group"        i: 1                            type: INT  }
attribute { name: "kernel_shape" ints: 3 ints: 3                 type: INTS }
attribute { name: "pads"         ints: 1 ints: 1 ints: 1 ints: 1 type: INTS }
attribute { name: "strides"      ints: 1 ints: 1                 type: INTS }
```

**TensorProto**（`initializer` 里的权重，8×3×3×3）：

```protobuf
initializer {
  dims: 8  dims: 3  dims: 3  dims: 3
  data_type: 1                   # TensorProto.FLOAT
  name: "conv_w"                 # ← 必须与 node.input[1] 同名
  raw_data: "\000\000\200?..."   # 8*3*3*3*4 = 864 字节，小端裸内存
}
```

权重大到不想塞进 protobuf 时换成 external data，数据字段整个消失：

```protobuf
  data_location: EXTERNAL
  external_data { key: "location" value: "weights.bin" }
  external_data { key: "offset"   value: "0" }
  external_data { key: "length"   value: "864" }
```

**ValueInfoProto**（图输入 `img`，第 0 维是符号维）：

```protobuf
input {
  name: "img"                    # ← 必须与 node.input[0] 同名
  type {
    tensor_type {
      elem_type: 1               # 与 TensorProto.data_type 共用同一套枚举
      shape {
        dim { dim_param: "N" }   # 符号维
        dim { dim_value: 3 }
        dim { dim_value: 32 }
        dim { dim_value: 32 }
      }
    }
  }
}
```

三者的引用关系全靠**字符串名字**，proto 里没有任何指针：

| 名字 | 定义在哪 | 使用在哪 | 谁描述它的类型/形状 |
|------|---------|---------|------------------|
| `img` | `graph.input[i]`（ValueInfoProto） | `conv0.input[0]` | 就是那个 ValueInfoProto |
| `conv_w` | `graph.initializer[j]`（TensorProto） | `conv0.input[1]` | TensorProto 自带 `dims` + `data_type` |
| `conv_b` | 同上 | `conv0.input[2]` | 同上 |
| `feat` | `conv0.output[0]`（SSA 定义点） | 下游节点的 `input` | 跑过 shape inference 后才在 `graph.value_info` 里出现 |

一句话记住分工：**TensorProto 带数据、ValueInfoProto 只带类型、NodeProto 两个都不带，只带名字。**

> **自测**：这段 proto 里能读出 `feat` 的形状吗？要拿到它得先调用什么？

### 2.5 Functions：模型本地函数

`FunctionProto` = **算子签名 + 用更原语算子写成的函数体（子图）**。

| 角色 | 说明 |
|------|------|
| 对模型作者 | 把复合算子（如 LayerNorm 的展开）封装成可复用单元 |
| 对运行时 | **MAY** 内联函数体执行；也 **MAY** 用自己的优化 kernel 直接实现同名算子，忽略函数体 |

唯一标识（IR≥10）：`(name, domain, overload)`。函数体有自己的 `opset_import` 与拓扑有序 `node` 列表。

**学习优先级**：知道「函数是可内联的默认实现」即可；深入自定义 function 可后置。

---

## 第 3 章 图语义：拓扑、SSA、属性 vs 输入

### 3.1 拓扑有序、无环

- 图是 **DAG**；节点依赖由数据名字引用建立。
- `node` 列表 **MUST 拓扑有序**：若 K 在列表中位于 N 之后，则 N 的任何输入都不得引用 K 的输出。
- 这让单次正向扫描即可调度，也让许多图改写（插入/删除节点）必须维护拓扑序。

### 3.2 值名字 ≈ SSA 字符串

官方要求：**所有 node 输出名在图内唯一**——即对 node output 做 SSA。

```
graph.input:   "x"
initializer:   "W", "b"
nodes:
  Gemm  →  output "y"          # 定义 y
  Relu  →  input "y", output "z"
graph.output:  "z"
```

- **定义**出现在：graph input、initializer、node output；
- **使用**出现在：node input、graph output；
- 同一名字在 input 与 initializer 同时出现是**唯一常规例外**（默认值语义，见 [2.3](#23-graphproto可执行的计算图)）。
- 嵌套子图（控制流算子的 `body` 等）里**禁止变量遮蔽**：内层名字不得与外层可见名字冲突。

名字空间有多套（Attribute / Value / Node / Graph / Operator / Shape），日常改图主要碰 **Value** 与 **Node**。

### 3.3 属性 vs 输入：两种传参通道

| | **Inputs** | **Attributes** |
|--|-----------|----------------|
| 值何时确定 | 运行时（或来自上游节点） | 构图时已固定 |
| 典型内容 | 激活、权重（作为 initializer 喂入）、动态 shape 参数 | kernel_size、strides、axis、epsilon |
| 关联方式 | **位置**对应算子签名 | **名字**对应算子属性 |
| 对优化的含义 | 可参与数据流、可被 EP 分区切开 | 常量，常被折叠进 kernel 特化 |

官方提醒：这种区分对某些实现的性能至关重要，对另一些实现则无关——但作为 IR 契约必须遵守。

**常见错误**：把本该是 attribute 的常量做成输入（或反过来），导致 opset 校验失败，或让本可常量折叠的路径变成动态。

#### 示例精讲：合法 3 节点图 vs 把 attribute 误做 input

合法版：`x → MatMul(W) → Add(b) → Softmax(axis=-1) → y`。

```python
W = numpy_helper.from_array(np.zeros((3, 4), np.float32), name="W")
b = numpy_helper.from_array(np.zeros((4,), np.float32), name="b")

nodes = [
    helper.make_node("MatMul",  ["x", "W"],  ["h"],  name="mm0"),
    helper.make_node("Add",     ["h", "b"],  ["h2"], name="add0"),
    helper.make_node("Softmax", ["h2"],      ["y"],  name="sm0", axis=-1),  # axis 是属性
]
```

对应文本（只列关键行）：

```protobuf
node { input: "x"  input: "W" output: "h"  name: "mm0"  op_type: "MatMul" }
node { input: "h"  input: "b" output: "h2" name: "add0" op_type: "Add" }
node { input: "h2"            output: "y"  name: "sm0"  op_type: "Softmax"
       attribute { name: "axis" i: -1 type: INT } }
```

逐条对上本章三条规则：

| 规则 | 这张图怎么满足的 |
|------|---------------|
| 拓扑有序（3.1） | `mm0` 定义 `h` 排在 `add0` 使用 `h` 之前；把 `node` 列表倒过来写就非法 |
| 输出 SSA（3.2） | `h` / `h2` / `y` 互不相同，也不与 `x` / `W` / `b` 撞名 |
| 属性 vs 输入（3.3） | `axis` 走属性（构图时定死）；`W` / `b` 走 input + initializer（数据流上的常量） |

反例：把 `axis` 做成第 2 个输入。

```python
axis = numpy_helper.from_array(np.array([-1], np.int64), name="axis")
bad = helper.make_node("Softmax", ["h2", "axis"], ["y"], name="sm_bad")   # 多了一个 input
```

`checker.check_model` 直接拒绝（报错信息示意，措辞随版本变化）：

```text
onnx.checker.ValidationError: Node (sm_bad) has input size 2 not in range [min=1, max=1]
```

因为 opset 17 的 `Softmax` 签名就是「1 输入 1 输出 + `axis` 属性」——**签名由 opset 钉死，不是可选风格**（见 [4.1](#41-算子身份三元组)）。

就算换成本来就允许该参数走输入的算子（例如 opset≥5 的 `Reshape`，`shape` 确实是 input），把编译期常量塞进数据流仍然有代价：

| 影响面 | 属性版 | 输入版 |
|-------|-------|-------|
| 常量折叠 | 值在 schema 校验时已知，kernel 直接特化 | 要先做常量传播确认上游是 initializer，才敢折叠 |
| Shape inference | 推断函数直接读属性，输出形状确定 | 上游若不是常量，推断退化为未知维（[5.3](#53-能力边界必须建立正确预期)） |
| EP 分区 | 节点自洽，EP 一眼判断能否接管 | EP 通常要求该输入必须是 initializer，否则拒接 → 分区在此断开 |
| 图改写 | 改属性一行搞定 | 要同时改 initializer，可能还要修拓扑序 |

> **自测**：`Reshape` 的 `shape` 从 initializer 变成上游算出来的张量后，EP 为什么更可能把这个节点丢回 CPU？

### 3.4 Optional 输入/输出（知道即可）

**Static optional** 用省略尾参或 `""`；**Dynamic optional**（IR≥8）是类型 `optional<T>`。改图遇到空字符串 input 先查算子文档，别当成坏图。

---

## 第 4 章 算子集与版本管理

ONNX 同时版本化三类东西——**彼此独立**：

| 类别 | 版本形态 | 回答的问题 |
|------|---------|-----------|
| **IR version** | 单调整数 | 文件格式与图级语义 |
| **Operator set version** | `(domain, version)` | 「Add / Conv / …」此刻的签名与语义快照 |
| **Model version** | 建议 SemVer | 这个具体模型相对上一版改了什么 |

### 4.1 算子身份三元组

一个算子声明由 `(domain, op_type, since_version)` 标识，书写为 `domain.op_type:since_version`。

- 默认 ONNX 算子集的 domain 是**空字符串** `""`（有时文档写作 `ai.onnx`）。
- 厂商扩展用 reverse-DNS domain（如 `com.microsoft`）。
- `since_version` = 该算子（当前语义）**首次进入**该 domain 的 opset 版本。

模型不直接写每个节点的 since_version；而是声明：

```text
opset_import = [ ("", 17), ("com.microsoft", 1) ]
```

含义：图中 domain `""` 的节点，按 **opset 17** 的算子快照解释；每个节点绑定到「该 opset 中该 `op_type` 的最新稳定声明」。

### 4.2 为什么升 opset 会改变语义

Versioning.md 规定，下列变更 **必须** 以新算子版本引入（即抬高 opset）：

- 增删改属性名（**包括新增带默认值的 optional 属性**——这点反直觉但必须记住）；
- 增删重排输入/输出；
- 增删支持的类型；
- **签名不变但行为变了**（例如某算子开始隐式支持 broadcast）。

因此：**同一个 `op_type` 字符串在不同 opset 下可能是不同契约**。只看节点名不够，必须连同模型的 `opset_import` 一起看。

### 4.3 版本转换的坑

ONNX 提供 `onnx.version_converter.convert_version(model, target_version)`，但工程上要假设：

| 坑 | 说明 |
|----|------|
| **不是总能升/降** | 缺少 adapter 时直接失败；降级往往比升级更脆 |
| **CompatibleAdapter 假象** | 旧图在新 schema 下「仍合法」≠ 数值/边界行为完全一致 |
| **自定义 domain** | 转换器通常只覆盖官方 `ai.onnx`；厂商算子需自己处理 |
| **函数体 / 子图** | 控制流内部、function body 里的节点版本更容易漏转 |
| **导出工具默认 opset** | PyTorch / tf2onnx 的默认 opset 与 ORT / IREE 支持矩阵可能错位——**先对齐 opset，再谈优化** |

**实践建议**：

1. 固定团队使用的目标 opset（例如 17 或 18），导出时显式指定；
2. 升级前对关键模型跑数值对比（同一输入、ORT CPU）；
3. 不要在生产路径上「运行时自动升 opset」而不做校验。

### 4.4 稳健性原则（读规范时的滤镜）

Versioning 引用 Postel 法则：

- **生产者**必须严格遵守 breaking / non-breaking 规则；
- **消费者**在 MAJOR 未变时应尽量接受；MAJOR 变了可以拒绝，也可以自愿兼容。

对你做运行时/编译器的含义：加载路径要清晰报告「IR 不支持 / opset 不支持 / 某节点无 kernel」，而不是静默算错。

---

## 第 5 章 Shape Inference：能推什么、推不出什么

### 5.1 静态 shape vs 运行时 shape

`TypeProto.Tensor.shape` 是**静态**形状描述，与运行时真实 shape 相关但不等同：

| 静态表示 | 含义 |
|---------|------|
| `shape` 字段缺失 | **未知秩**（任意维数） |
| `shape` 存在但某维既无 `dim_value` 也无 `dim_param` | 该维匿名未知 |
| `dim_value = 64` | 该维编译期已知为 64 |
| `dim_param = "N"` | 符号维；**全图同名符号表示同一运行时长度** |

主图 input/output **必须有 shape（已知 rank）**，但具体维可以是符号。

### 5.2 怎么调用

```python
import onnx
from onnx import shape_inference

model = onnx.load("model.onnx")
inferred = shape_inference.infer_shapes(model)
# 就地写回 ValueInfo；也可 infer_shapes(model, strict_mode=True, data_prop=True)
onnx.save(inferred, "model.shaped.onnx")
```

自 ONNX 1.10 起，图级推断还支持**符号生成/传播**与部分**形状数据传播**（见官方 Symbolic Shape 提案）。

### 5.3 能力边界（必须建立正确预期）

官方明确：**shape inference 不保证完备**。

| 场景 | 结果 |
|------|------|
| `Concat` 两个 `(5,2)` 与 `(7,2)` | 可推出 `(12,2)` |
| `Concat` `(5,2)` 与 `(N,2)` | 得到新的未知符号维，**不会**表示成 `N+5` 表达式 |
| `Reshape` 到动态 shape 输入 | 推断在此阻断或退化为未知 |
| 算子未实现推断函数 | 该节点输出可能没有 shape |
| 仅常量与简单变量 | **不支持**含变量的算术表达式形状 |

此外：

- **类型推断**多半由 schema 的 type constraint 自动完成（如共享类型变量 `T`）；
- **形状推断**几乎总要算子自带 `TypeAndShapeInferenceFunction`。

#### 示例精讲：`[None, 3]` 进来，`infer_shapes` 之后拿到什么

最小具体输入：动态 batch 的 `x`，先与一个 `[5,3]` 常量沿 axis=0 拼接，再做 MatMul。

```python
import numpy as np, onnx
from onnx import helper, numpy_helper, shape_inference, TensorProto

C = numpy_helper.from_array(np.zeros((5, 3), np.float32), name="C")
W = numpy_helper.from_array(np.zeros((3, 4), np.float32), name="W")

nodes = [
    helper.make_node("Concat", ["x", "C"],   ["cat"], name="cat0", axis=0),
    helper.make_node("MatMul", ["cat", "W"], ["y"],   name="mm0"),
]
X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3])   # 匿名动态维
Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 4])
g = helper.make_graph(nodes, "dyn", [X], [Y], initializer=[C, W])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = onnx.IR_VERSION

print(len(m.graph.value_info))               # 0：中间值一个注解都没有
m2 = shape_inference.infer_shapes(m)
print(len(m2.graph.value_info))              # 1：多出中间值 cat
```

`[None, 3]` 写成 proto 是这样——注意第 0 维**既没有 `dim_value` 也没有 `dim_param`**：

```protobuf
shape { dim { }  dim { dim_value: 3 } }
```

推断前后对比：

```text
# 推断前
graph.value_info: []

# 推断后（示意）
value_info {
  name: "cat"
  type { tensor_type { elem_type: 1
         shape { dim { dim_param: "unk__0" }   # ← 不是 "N+5"，也不是 5
                 dim { dim_value: 3 } } } }
}
```

> 未知维在推断结果里是留成空 `dim { }` 还是填一个自动生成的符号名（`unk__0` 之类），随 onnx 版本变化；两者含义相同：**一个与输入无关的未知维**。

为什么拿不到 `N+5`：`TensorShapeProto.Dimension` 是个 oneof，只有三种状态——

| 维的状态 | proto 形态 | 能表达什么 |
|---------|-----------|----------|
| 已知常量 | `dim_value: 5` | 具体长度 |
| 符号 | `dim_param: "N"` | 「与别处同名的那个长度相同」 |
| 未知 | 两个字段都不设 | 什么都不知道 |

**没有第四种「表达式」状态**，所以 `? + 5` 只能退化成「未知」或一个新造的符号。`dim_param` 的语义是同名即同长，`unk__0` 与输入的动态维之间没有任何代数关系，只是个占位符。

把 `x` 改成 `["N", 3]`（即 `dim_param: "N"`）结果一样：`Concat` 照样推不出 `N+5`。但换成 `Add` 这类逐元素算子，`N` 会**原样传播**下去——差别在于算子是否需要对维做算术。

> **自测**：把 `[None,3]` 换成 `[8,3]` 后，`cat` 的第 0 维变成什么？EP 的分区决策会因此不同吗？

### 5.4 对多后端委托的含义

- 分区与 kernel 选择经常依赖静态 shape；动态维过多时，EP 可能拒绝接管或退回 CPU。
- 符号维 `"N"`、`"seq"` 在全图一致——这是跨节点约束，改图时不要随意改名符号。
- **研究问题 3（动态 Shape）** 的工程起点，往往是：先看 `infer_shapes` 之后还有多少维仍是 `dim_param` / 匿名未知。

---

## 第 6 章 工程手感：用 onnx.helper 读写改图

根 README §4.2 动手验收第 1 条：手工构造 3 层小图 + 解析真实模型并改节点。下面给出可直接跑的模式。

### 6.1 手工构造三层小图（Linear → ReLU → Add）

```python
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, checker, shape_inference

# 权重
W = numpy_helper.from_array(np.random.randn(4, 3).astype(np.float32), name="W")
b = numpy_helper.from_array(np.random.randn(4).astype(np.float32), name="b")
bias2 = numpy_helper.from_array(np.ones(4, dtype=np.float32), name="bias2")

# 节点：Gemm → Relu → Add
gemm = helper.make_node("Gemm", ["x", "W", "b"], ["h"], name="gemm")
relu = helper.make_node("Relu", ["h"], ["h_act"], name="relu")
add  = helper.make_node("Add", ["h_act", "bias2"], ["y"], name="add")

X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3])  # batch 动态
Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 4])

graph = helper.make_graph(
    [gemm, relu, add],
    "tiny_mlp",
    inputs=[X],
    outputs=[Y],
    initializer=[W, b, bias2],  # 权重只进 initializer，不进 inputs
)
model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid("", 17)],
    producer_name="ai-infra-tutorial",
)
model.ir_version = onnx.IR_VERSION
checker.check_model(model)
model = shape_inference.infer_shapes(model)
onnx.save(model, "tiny_mlp.onnx")
```

用 ORT 跑通：`InferenceSession(..., providers=["CPUExecutionProvider"]).run(None, {"x": x})`，期望 `y.shape == (2, 4)`。

### 6.2 加载真实模型：巡检清单

```python
import onnx
from collections import Counter

model = onnx.load("real.onnx")
g = model.graph
print("IR:", model.ir_version)
print("Opsets:", [(o.domain, o.version) for o in model.opset_import])
print("#nodes:", len(g.node), "#inits:", len(g.initializer))
print(Counter((n.domain, n.op_type) for n in g.node))  # EP 能否吃下，先看直方图
init = {t.name for t in g.initializer}
print("runtime inputs:", [i.name for i in g.input if i.name not in init])
```

### 6.3 插入 / 删除节点

**插入 Identity 探针**：把所有使用 `producer_output` 的边改接到 `new_name`，再 `make_node("Identity", [producer_output], [new_name])` 追加进 `graph.node`（生产代码应拓扑重排后再 `checker.check_model`）。

**删除 Identity**：找到节点，令 `inp, out = node.input[0], node.output[0]`，把下游（含 `graph.output`）对 `out` 的引用改回 `inp`，再 `graph.node.remove(node)`。

### 6.4 修改 initializer

```python
def replace_initializer(graph, name: str, array: np.ndarray):
    for i, t in enumerate(list(graph.initializer)):
        if t.name == name:
            del graph.initializer[i]
            break
    graph.initializer.append(numpy_helper.from_array(array, name=name))
```

量化消融、权重置零都走这条路；改完务必 `checker.check_model` + 数值冒烟。

### 6.5 拓扑合法性抽查

```python
def assert_topo(graph):
    defined = {i.name for i in graph.input} | {t.name for t in graph.initializer}
    for n in graph.node:
        for inp in n.input:
            if inp:
                assert inp in defined, f"{n.name} uses undefined {inp}"
        defined.update(o for o in n.output if o)
```

---

## 第 7 章 ONNX Runtime EP 机制（重点）

> 这一章对应根 README §4.2 的 ★ 条目，也是六个研究问题的**工业接口现场**。

### 7.1 EP 在 ORT 架构中的位置

**Execution Provider（EP）** = ORT 用来对接「某类硬件加速库」的插件抽象。官方原话大意：

- ORT 通过 EP 框架与不同硬件加速库协作；
- 用 `GetCapability()` 把**特定节点或子图**分配给 EP；
- EP 在支持的硬件上编译并执行这些 ONNX 子图；
- 从而把 CPU / GPU / FPGA / NPU 等差异挡在统一 Session API 之后。

```
                    InferenceSession(model, providers=[...])
                                    │
                    ┌───────────────┴───────────────┐
                    │     ONNX Runtime 图分区器      │
                    │  按 providers 优先级询问各 EP   │
                    └───────────────┬───────────────┘
           GetCapability │          │ GetCapability
                         ▼          ▼
                   ┌──────────┐ ┌──────────┐     ┌──────────┐
                   │ CUDA EP  │ │ TensorRT │ …   │ CPU EP   │
                   │ 声明子图  │ │  声明…    │     │ 兜底整图  │
                   └────┬─────┘ └────┬─────┘     └────┬─────┘
                        │ Compile    │ Compile         │
                        ▼            ▼                 ▼
                   编译后的子图执行句柄（fused node / 引擎）
                                    │
                         运行时按分区结果派发 Run()
```

**API 层你看到的**：

```python
sess = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
# providers 列表是优先级顺序：前者能拿的节点/子图优先给前者
```

同一套 `run()` API；换 EP 列表 = 换分区与设备，不换应用代码。

### 7.2 GetCapability → Compile → 运行时派发

这是 EP 生命周期的三拍，也是「子图划分」在 ORT 里的标准接口。

#### 拍 1：`GetCapability` —— EP 声明「我能吃哪些子图」

Session 初始化时，ORT 拿着计算图（往往已经过分融合等优化）给每个 EP 机会回答：

> 在当前图上，哪些**节点集合**（通常是连通子图）我可以负责？

EP 返回的是一组 **capability**（能力声明），每项大致包含：

- 覆盖的节点集合（子图）；
- 可选的融合/元数据，供后续 Compile 使用。

要点：

1. **不是**「这个 op_type 我有 kernel 吗」的简单布尔表——工业 EP（TensorRT、CUDA、QNN…）会做支持性检查、形状/类型过滤、有时还有自身的融合视图。
2. **优先级**：`providers` 列表靠前的 EP 先挑；挑走的节点不再给后面的 EP（CPU 通常垫底兜底）。
3. **结果是图分割**：一张 ONNX 图变成「EP_A 子图 ∪ EP_B 子图 ∪ … ∪ CPU 余图」，子图之间的边成为**分区边界**。

这就是研究问题 1 的工业形态：

> 子图划分 = 带（隐式）传输代价的图分割；`GetCapability` 是各后端的**可行域声明**，ORT 的分配策略是**贪心按优先级**，不是全局最优求解器。

#### 拍 2：`Compile` —— 把拿走的子图编成可执行体

对每个被接受的 capability，ORT 调用该 EP 的 `Compile`：

- 输入：子图（节点、权重、I/O 边界）；
- 输出：EP 内部的执行句柄（CUDA graph / TRT engine / 一串 kernel launch 元数据等）；
- 原图上通常表现为一个（或一组）**融合后的占位节点**，运行时接到这些节点就转入 EP。

Compile 阶段常见工作：

- 子图内再融合 / 常量折叠；
- 选择 layout（NCHW↔NHWC）、精度（FP16/FP8）；
- 分配 EP 私有内存池或绑定外部分配器；
- 失败则回退：该 capability 作废，节点改派其他 EP。

#### 拍 3：运行时按分区派发

`session.run()` 时：

1. 按拓扑（或 ORT 内部调度序）执行；
2. 遇到某 EP 的融合节点 → 调用该 EP 执行编译好的子图；
3. 跨 EP 的边 → 插入必要的数据运动与同步（见下一节）；
4. CPU EP 执行剩余节点。

```
ONNX 逻辑图:   [Conv]→[BN]→[Relu]→[MatMul]→[Softmax]
优先级: CUDA > CPU

一种可能分区:
  CUDA EP:  Conv+BN+Relu+MatMul   （一次 GetCapability 吃下一大块）
  CPU EP:   Softmax               （CUDA 不接或故意不接）

边界: MatMul 的输出要从 GPU 拷到 Host（或保持在统一内存），再进 Softmax
```

#### 示例精讲：3 节点模型上，三拍各改变了图的什么

最小具体输入：`x → Conv → Relu → Softmax → y`（3 个节点，Conv 权重在 initializer）。

```python
import onnxruntime as ort

so = ort.SessionOptions()
so.optimized_model_filepath = "opt.onnx"        # 落盘「优化 + 分区之后」的图
sess = ort.InferenceSession("tiny.onnx", so,
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
```

> API 名称随 ORT 版本变化，跑之前核对本地版本；下面的融合节点名与 domain 同样是示意。

**先分清 EP 有两类**，三拍在它们身上长得不一样：

| EP 类型 | 代表 | `GetCapability` 返回 | `Compile` | 优化后图里看到什么 |
|--------|------|---------------------|-----------|-----------------|
| kernel-based | CPU、CUDA | 一批**单节点** capability | 不走这条路（直接查 kernel registry） | 节点个数不变，只是各节点归属的 provider 变了 |
| compiling | TensorRT、QNN、NNAPI… | 一个**子图** capability | 走：子图 → 引擎 / blob | 整块被换成**一个不透明的融合节点** |

同一个模型，两种 provider 列表，`opt.onnx` 长得完全不同：

```text
# A. providers=["CUDAExecutionProvider","CPUExecutionProvider"]（kernel-based）
node { op_type: "FusedConv" domain: "com.microsoft" input: "x"  output: "h2" }  # Conv+Relu
node { op_type: "Softmax"   domain: ""              input: "h2" output: "y"  }

# B. providers=["TensorrtExecutionProvider","CPUExecutionProvider"]（compiling）
node { op_type: "TRTKernel_graph_tiny_0" domain: "com.microsoft"
       input: "x" output: "y" }         # 三个节点整段被吃进一个引擎
```

三拍分别落在哪一步：

| 时刻 | 发生什么 | 对应关系 |
|------|---------|---------|
| Session 初始化，分区**前** | L1 基础优化：常量折叠、冗余节点消除 | 拍 1 拿到的图已经被动过 |
| 分区 | 按 `providers` 顺序问每个 EP `GetCapability`，先到先得 | **拍 1** |
| 分区**后** | L2 扩展优化（如 Conv+Relu → `FusedConv`）只作用于分给 CPU / CUDA 的节点 | A 里的融合来自这里，**不是**拍 2 |
| 每个 capability | 调该 EP 的 `Compile`，产出引擎/blob，图上留一个占位节点 | **拍 2** |
| `session.run()` | 按分区结果派发；跨 EP 的边补数据搬运 | **拍 3** |

所以「优化后的图里出现融合节点」有两种来源，看 `opt.onnx` 时必须先分清：**ORT 自己的图优化**（`FusedConv`，仍是可读算子、仍能逐节点观测）与 **EP `Compile` 的产物**（`TRTKernel_*`，内部不可见）。

> **自测**：只用 `providers=["CPUExecutionProvider"]` 落盘一次 `opt.onnx`，`FusedConv` 还在吗？为什么一定没有 `TRTKernel_*`？

### 7.3 分区边界代价

**每个分区边界都可能引入四类开销**（与 [`ai-compiler-foundations.md` §3.3](./ai-compiler-foundations.md#33-子图划分与委托partition--delegate)、[`executorch-learning-guide.md`](./executorch-learning-guide.md) §5 用同一套分类）——这正是研究问题 ①② 的根源：

| 代价类型 | 发生原因 | 典型症状 |
|---------|---------|---------|
| **数据拷贝** | 子图产出必须搬到下一后端能读的地方（host↔device、device↔device） | `cudaMemcpy` / PCIe 往返；小子图时拷贝 > 计算 |
| **Layout 转换** | 两端 EP 偏好不同 layout（NCHW vs NHWC vs 打包格式） | 边界出现 Transpose / Reformat；带宽被吃掉 |
| **内存空间切换** | 不同设备指针空间 / 分配器不互认，无法零拷贝共享 | 需要经 host 中转或重新注册内存 |
| **同步点** | 不同队列/流、Host 回调、EP 内部同步模型不一致 | GPU 假忙等；吞吐被气泡打穿 |

```
        EP_A (GPU)                         EP_B (NPU)
   ┌─────────────────┐                ┌─────────────────┐
   │  fused subgraph │─── boundary ──▶│  fused subgraph │
   └─────────────────┘    ▲           └─────────────────┘
                          │
              layout convert?
              device→device copy?
              stream sync / event?
```

**设计直觉**（对照 [paper-notes/05-tvm.md](./paper-notes/05-tvm.md) 的融合）：

- TVM 的融合规则在**单后端内**消灭中间写回；
- ORT EP 分区在**多后端之间**制造边界；
- 好的划分 ≈ 在「后端算得快」与「边界次数/流量小」之间折中；
- 贪心优先级**不会**自动做出这个折中——这就是研究空白。

#### 示例精讲：`Conv`(GPU EP) → `Softmax`(CPU EP) 边界上到底多了什么

构造方式：让 `Conv` 归 CUDA EP、`Softmax` 归 CPU EP（真实做法是给 `Softmax` 用该 EP 不支持的 dtype/属性，或用只注册了 `Conv` 的自定义 EP）。分区之后，ORT 的 memcpy 变换会在跨设备的边上插节点：

```text
分区前（逻辑图）
  x ──▶ Conv ──h──▶ Softmax ──▶ y

分区后（示意，节点名随版本变化）
  x ──▶ MemcpyFromHost ──▶ Conv[CUDA] ──▶ MemcpyToHost ──▶ Softmax[CPU] ──▶ y
       └──── e1 ────┘                   └──── e2 ────┘
```

`MemcpyFromHost` / `MemcpyToHost` 是 ORT 内部算子（不属于 `ai.onnx`），**只有图被切开且两侧不在同一设备时才出现**——它们是四类代价里最可见的那一类。

四类代价各对应哪条边：

| 代价 | 落在哪条边 | 图上的证据 | 什么条件下不出现 |
|------|-----------|-----------|--------------|
| **数据拷贝** | e1（H2D，喂 `x`）、e2（D2H，取 `h`） | 两个 `Memcpy*` 节点本身 | 用 IOBinding 把输入直接放到 device 上，e1 可消掉 |
| **内存空间切换** | e1、e2 | `Memcpy*` 存在即说明两侧分配器不互认 | 统一内存；或两侧同为 CPU |
| **同步点** | e2 | **图上看不见**；CPU 要读 `h` 就必须等 GPU 流跑完 | 下游仍在同一 stream 上 |
| **Layout 转换** | 视 EP 偏好而定，通常也压在 e2 | 边上多出 `Transpose`（偏好 NHWC 的 EP 由 L3 layout 优化插入） | 两侧 layout 偏好一致 |

e2 的代价与 `h` 的大小成正比：`Conv` 输出 `[1,64,56,56]` 的 FP32 ≈ 802 KB，一次 D2H 可能比在 CPU 上跑一次 `Softmax` 还贵——**这就是「多委托一个节点反而更慢」的最小复现**。

反过来看：若 `Softmax` 也能进 CUDA，e2 直接消失，两个节点并成一块，边界从「图中间」退到「图两端」。所以分区的目标不是「委托节点数最大」，而是 **计算收益 − 边界代价** 最大（对照 [`ai-compiler-foundations.md` §3.3](./ai-compiler-foundations.md#33-子图划分与委托partition--delegate)）。

> **自测**：把 Conv 输出通道从 64 降到 4，e2 的拷贝量变成多少？这时把 `Softmax` 也搬上 GPU 还划算吗？

### 7.4 与 ExecuTorch / IREE / TVM 的对照

| 系统 | 「谁声明能吃什么」 | 「怎么编译子图」 | 「边界代价暴露在哪」 |
|------|------------------|----------------|-------------------|
| **ORT EP** | `GetCapability` | `Compile` | 跨 EP 边的 copy/sync；可用 profiling 看到 |
| **ExecuTorch** | `Partitioner` 标记可委托节点 | `to_backend` → `LoweredBackendModule` | delegate 子图边界的 data copy | 
| **IREE** | 编译期 `flow.dispatch` 形成 | Stream/HAL lowering + codegen | `hal.tensor.import/export` + fence | 
| **TVM** | 融合规则（四类算子）决定 kernel 粒度 | TE/schedule 或 MetaSchedule | 融合消除边界；跨设备需另做 |

- **ORT EP** = 工业界最常见的「子图分区接口」——加载期协商、运行时派发。
- **ExecuTorch** = PyTorch 侧对等故事；**完整 Partitioner 章节见** [executorch-learning-guide.md](./executorch-learning-guide.md)，本文不重复。
- **IREE** 把划分决策**前移到编译期**（flow），运行时更「笨」——见 [iree-learning-guide.md](./iree-learning-guide.md)。
- **TVM 融合**是同问题的**单设备经典解**；多设备分区是其粗粒度推广——见 [paper-notes/05-tvm.md](./paper-notes/05-tvm.md) §3.1 与文末「与子图划分」段。

### 7.5 动手：打印真实 EP 分区

根 README §4.2 验收第 2 条。目标：**看见**哪些节点去了哪个 EP，而不是只看延迟数字。

**推荐入口**：仓库动手项目 [`../onnx-delegate-lab/`](../onnx-delegate-lab/)

```bash
cd onnx-delegate-lab && bash scripts/run_onnx.sh
# 读 out/onnx/03_READING.md 与 03_partition_report.json
```

最小手写复现：

```python
import numpy as np
import onnxruntime as ort

so = ort.SessionOptions()
so.enable_profiling = True
so.optimized_model_filepath = "optimized.onnx"  # 落盘优化后图，看 fused / 边界
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]  # 无 CUDA 则换你有的加速 EP

sess = ort.InferenceSession("model.onnx", so, providers=providers)
print("providers:", sess.get_providers())
feeds = {
    i.name: np.zeros([d if isinstance(d, int) else 1 for d in i.shape], np.float32)
    for i in sess.get_inputs()
}
sess.run(None, feeds)
print("profile:", sess.end_profiling())  # Chrome tracing / JSON 里看各 op 的 provider
```

**验收时要能回答**：① 哪些连续节点被同一 EP 吃成一块？② 边界落在哪两个 op 之间、为何（类型 / 动态 shape / 启发式）？③ 边界两侧是否多了 layout / copy？

---

## 第 8 章 与 AI-Infra 六个研究问题的对接

六个问题全文见根 README §6；下表只钉 **ONNX / ORT EP 视角** 的切入点（尤其 1、2、5、6）。

| # | 问题 | ONNX / ORT 里你能直接摸到的现象 | 可做的观察 / 实验 |
|---|------|-------------------------------|------------------|
| **1** | 子图划分与跨设备传输最小化 | `GetCapability` 贪心分区；边界 = 潜在 PCIe/拷贝 | 强制不同 `providers` 顺序；对比延迟与 profile 中 copy 占比 |
| **2** | 跨后端 Layout / 量化 / 内存空间 | 边界插入 Transpose、Cast、device copy | 同一模型 FP32 vs FP16 EP；看 optimized graph 多了哪些转换节点 |
| 3 | 动态 Shape | `dim_param`、Reshape 动态阻断推断；EP 可能拒分 | 固定 shape 导出 vs 动态 batch，对比分区碎片化程度 |
| 4 | 低成本后端切换 | Session 级 `set_providers` 会重建；**状态迁移不是 ORT 的菜** | 理解其边界：权重在 EP 私有格式中时切换成本高——对照研究问题 4 的 checkpoint 思路 |
| **5** | 后端最优图不一致 | TRT 要的融合图 ≠ CUDA EP 要的 ≠ CPU | 分别只开单一 EP，导出 `optimized_model_filepath` 对比 |
| **6** | 配置组合爆炸 | opset × EP × dtype × shape × 图优化级别 | 固定其余，只扫 shape 或只扫 EP，感受笛卡尔积 |

**与 TVM 融合的对照（问题 1 / 5）**：

- TVM 四类算子规则（injective / reduction / complex-out-fusable / opaque）回答「**单设备内**哪些节点应合成一个 kernel」——见 [paper-notes/05-tvm.md](./paper-notes/05-tvm.md) §3.1。
- ORT EP 回答「**多设备/多库之间**哪些节点合成一个委托子图」。
- 两者同构：都是图分割；评价函数都离不开「下游能不能为这块子图生成高效实现」。差异在代价模型：TVM 偏内存层级与 schedule；ORT 偏 EP 能力表 + 拷贝/同步。

**与 IREE 的对照**：IREE 在 `flow` 固化 dispatch、在 `hal` 钉死设备对象——编译期划分；ORT 在 Session 创建期用 EP 协商划分。算力网若要「策略可复述、可复现」，需要比 ORT 贪心更显式的划分 IR（这正是你读 IREE 的理由）。

**ExecuTorch**：PyTorch 导出路径上的对等物；委托边界同样产生 copy。细节不在本文展开 → [executorch-learning-guide.md](./executorch-learning-guide.md)。

---

## 第 9 章 学习路径：最小必要集与动手清单

对应根 README §4.2 的必学表；**ExecuTorch 条目（验收第 3 条）放到 ExecuTorch 专文**，本文覆盖验收 1–2。

### 9.1 必须掌握

1. **ONNX ≠ 运行时**；ONNX vs ONNX-ML 的差别在算子集。
2. **结构层次**：`ModelProto → GraphProto → NodeProto`，外加 `TensorProto` / `ValueInfoProto`；能默画 [2.1](#21-总览从文件到节点) 那张图。
3. **`initializer` vs `graph.input`**——能解释三种名字出现方式，能从真实模型里筛出「真正要喂的输入」。
4. **图语义**：拓扑序、输出 SSA、属性 vs 输入。
5. **`(domain, opset_version)`**：默认 domain `""`；升 opset 可能改语义；version converter 有坑。
6. **`infer_shapes` 的边界**：常量可算、符号不代数化、动态 Reshape 阻断。
7. **`onnx.helper` 工程手感**：构图、插删节点、改 initializer、checker。
8. **ORT EP 三拍** ★：`GetCapability` → `Compile` → 按分区派发；能讲清分区边界四类代价（拷贝 / layout / 内存空间 / 同步）。
9. **能指着一次真实分区结果**说：边界为何在此、可能付出什么拷贝/同步。

### 9.2 可以先跳过

- 全部官方算子语义手册（用到查 [ONNX Operators](https://onnx.ai/onnx/operators/)）；
- ONNX 训练扩展（`training_info`）、IR≥11 多设备 configuration 细节；
- 具体 EP 内部实现（TensorRT builder 参数、OpenVINO 插件源码、QNN 细节）；
- 自定义 OpSchema / C++ shape inference 插件（除非你要给自研算子进 ONNX）；
- Function 多态与 overload 的全部边角（知概念即可）。

### 9.3 动手清单（按顺序）

**推荐：一键跑仓库项目**

```bash
cd onnx-delegate-lab && pip install -r requirements.txt && bash scripts/run.sh
# 先读 out/ANALYSIS.md；ONNX 轨覆盖下方第一步～第三步
```

**第一步：三层小图闭环**（验收 §4.2-1 前半）——或直接跑 lab `01_build_and_infer.py`

1. 跑通 [6.1](#61-手工构造三层小图linear--relu--add) 或 lab 步骤 01；
2. `onnx.checker.check_model` + `infer_shapes`；
3. ORT CPU `run` 出正确 shape。

**第二步：巡检 + 改图**（验收 §4.2-1 后半）——lab `02_mutate_graph.py`

1. 打印 opset、算子直方图、真正输入列表；
2. 插入 `Identity`、改 initializer，确认仍能跑且数值变化。

**第三步：EP 分区可视化**（验收 §4.2-2）——lab `03_ort_ep_partition.py`

1. CPU-only vs（若有）加速 EP + CPU；
2. 读 `out/onnx/03_READING.md`，回答边界三问。

**第四步：带着问题回看**

1. 用 [第 8 章](#第-8-章-与-ai-infra-六个研究问题的对接) 表格，给问题 1 和 2 各写一段证据；
2. 对照 [`../tvm-fatbin-lab/`](../tvm-fatbin-lab/) 融合步骤：同一段算子在 TVM 里融、在 EP 里是否被切开；
3. ExecuTorch 侧：跑 lab ExecuTorch 轨或见 [executorch-learning-guide.md](./executorch-learning-guide.md)。

### 9.4 建议阅读顺序（官方原文）

| 顺序 | 文档 | 读什么 |
|------|------|--------|
| 1 | [IR.md](https://github.com/onnx/onnx/blob/main/docs/IR.md) | Models / Graphs / Nodes / SSA / initializer 规则 / 静态 shape |
| 2 | [Versioning.md](https://github.com/onnx/onnx/blob/main/docs/Versioning.md) | 三类版本、opset 什么算 breaking、Released Versions 表 |
| 3 | [ShapeInference.md](https://github.com/onnx/onnx/blob/main/docs/ShapeInference.md) | Limitations 一节 + 符号维语义 |
| 4 | [ORT EPs](https://onnxruntime.ai/docs/execution-providers/) | 架构段 + `set_providers` 优先级语义 |
| 5 | 按需 | ExternalData.md、具体 EP 的 provider options |

枢纽页阶段地图：[docs/README.md](./README.md) 「阶段 4｜图级编译与多后端委托」。

---

## 附录：一页速查

```
【定位】   ONNX = 运行时无关的 NN 图 IR（protobuf）；不是运行时
【变体】   ONNX（ai.onnx / domain ""） vs ONNX-ML（+ ai.onnx.ml）

【层次】   ModelProto
             ├─ ir_version, opset_import[(domain,ver)], model_version
             ├─ graph: GraphProto
             │    ├─ input/output: ValueInfoProto  （主图必须有类型+rank）
             │    ├─ initializer: TensorProto      （常量；≠ 全部 input）
             │    ├─ value_info: 中间值注解
             │    └─ node[]: 拓扑序 NodeProto
             └─ functions[]: 可内联的默认实现

【initializer 陷阱】
  仅 initializer     = 真常量（权重应如此）
  仅 input           = 运行时必供
  两者都有           = 默认值，运行时可覆盖

【图语义】  DAG + 输出 SSA（值=字符串名）；attr=编译期常量，input=运行时数据流
【版本】    IR ≠ opset ≠ model；opset 升高可改语义；converter 有坑，先对齐再优化
【Shape】   infer_shapes 非完备；无 N+5 代数；动态 Reshape 阻断；符号维全图同名同义

【ORT EP 三拍】★
  GetCapability  → EP 声明可接管的节点/子图（按 providers 优先级贪心）
  Compile        → 子图 → EP 私有可执行体（失败则回退）
  Runtime        → 按分区派发；边界可能 layout / memspace / sync

【对照】
  ORT EP           工业子图分区接口（本文）
  ExecuTorch       PyTorch Partitioner → executorch-learning-guide.md
  IREE flow/hal    编译期划分 + 设备抽象 → iree-learning-guide.md
  TVM fusion       单设备四类算子规则 → paper-notes/05-tvm.md

【改图最小 API】
  helper.make_node / make_graph / make_model
  numpy_helper.from_array / to_array
  checker.check_model ; shape_inference.infer_shapes
  load / save ; 改 graph.node / graph.initializer 后务必 check
```

---

## 维护约定

官方 IR / Versioning / ShapeInference 或 ORT EP 注册方式变更时，优先同步第 2/4/5/7 章与附录。分区边界代价的分类以 [`ai-compiler-foundations.md`](./ai-compiler-foundations.md) §3.3 的四类为准，改动时同步 [`executorch-learning-guide.md`](./executorch-learning-guide.md) §5 与根 README §4.2。新动手脚本入口补到 [docs/README.md](./README.md) 阶段 4 与根 README §4.2。
