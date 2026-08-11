# 检验体系 05｜ONNX + ONNX Runtime

> **对应学习文档**：[`../onnx-learning-guide.md`](../onnx-learning-guide.md)  
> **对应动手项目**：[`onnx-delegate-lab/`](../../onnx-delegate-lab/) 的 ONNX 轨（`onnx_lab/01~03`）  
> **分级与资源标签定义**：[`./README.md`](./README.md)

本册检验的是这些知识点能不能落成代码：**ModelProto→GraphProto→NodeProto 的层次与 `initializer` vs `graph.input`、图语义（拓扑序 / 输出 SSA / 属性 vs 输入）、opset 版本化、shape inference 的能力边界、用 `onnx.helper` 从零构图与改图、自己写一个图重写 pass、ORT 的 `GetCapability`→`Compile`→派发三拍与分区边界代价**。

学习指南里的过关标准偏口述（能不能讲清楚 §9.1 那九条）；本册全部换成动手题——**改代码、跑命令、看产物差异**。

**入门线**：L0 两条全做 + L1 至少两条 + **L2-ONNX-06 与 L2-ONNX-07 必做**（一条是「能不能从零构一张非线性图」，一条是「能不能自己写图重写」，这两条是分水岭）。

**资源总览**：**除最后一条可选的 L3-ONNX-10 外，本册全部条目都能在本地 CPU 上完成**，只需 `pip install onnx onnxruntime`，标签 `本地+工具链`。GPU 只用于 L3 的双 EP 真机观察，非必需。

```bash
# 开工前自查：版本与可用 providers（后面凡是「以本地版本为准」的地方都回来看这行输出）
python -c "import onnx, onnxruntime; print(onnx.__version__, onnxruntime.__version__, onnxruntime.get_available_providers())"
```

> 本册涉及的 ONNX Python API 在不同版本间有细微差异（尤其 `onnx.defs`、`version_converter`、ORT 的 `GraphOptimizationLevel` 枚举名）。凡是文中写「以本地版本为准」的地方，都请先用 `dir()` / `help()` 确认签名再抄命令。

---

## L0 复现

### L0-ONNX-01｜跑通 ONNX 轨，手拆 ModelProto 的三层结构

- **检验什么**：这条通过 = 你真的掌握了「`ModelProto → GraphProto → NodeProto` 的嵌套关系」，以及 `initializer` 与 `graph.input` 的区别——为什么权重放哪边都合法，以及怎么筛出「真正要喂的输入」
- **前置**：装好 `onnx` + `onnxruntime`
- **资源**：本地+工具链
- **预计耗时**：1h

**任务**：跑通 ONNX 轨三步，然后**不看 `01_READING.md`**，自己写一小段 Python 打开 `out/onnx/01_tiny_mlp.onnx`，逐层打印：模型的 `ir_version` / `opset_import`，图名与节点数，每个 `NodeProto` 的 `op_type` / `input` / `output` / `attribute`，以及 `initializer` 的名字与形状。最后回答：这张 Gemm→Relu→Add 的三层 MLP 里，哪些名字是权重、哪些是真正的运行时输入、`graph.input` 里到底列了几项。

**验收命令**：

```bash
cd onnx-delegate-lab
bash scripts/run_onnx.sh        # 依次跑 01 / 02 / 03，产物落在 out/onnx/
python - <<'PY'
import onnx
m = onnx.load("out/onnx/01_tiny_mlp.onnx")
print("ir_version:", m.ir_version, "opset:", [(o.domain, o.version) for o in m.opset_import])
g = m.graph
print("graph:", g.name, "nodes:", len(g.node), "inits:", len(g.initializer))
for n in g.node:
    print(" ", n.op_type, list(n.input), "->", list(n.output), [a.name for a in n.attribute])
init = {t.name for t in g.initializer}
print("initializer:", sorted(init))
print("graph.input:", [i.name for i in g.input])
print("真正 runtime inputs:", [i.name for i in g.input if i.name not in init])
PY
```

**通过标准**：

- `run_onnx.sh` 以 `[OK] ONNX 轨完成` 结束，`out/onnx/` 下出现 `01_tiny_mlp.onnx`、`02_tiny_mlp_mutated.onnx`、`03_partition_demo.onnx`、`03_partition_report.json` 及三份 `*_READING.md`
- 你打印出的节点序列是 `Gemm → Relu → Add`，并能指出脚本里 `assert y.shape == (2, 4)` 断言的是哪个 `ValueInfoProto`
- 你能用一句话说清：`initializer` 里有名字的张量为什么可以**不**出现在 `graph.input` 里，以及若同时出现在两边，`initializer` 扮演的是什么角色（默认值 / 可被覆盖的输入）

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| `run_onnx.sh` 输出 `[SKIP] 需要 onnx + onnxruntime` | 环境没装或用错解释器；脚本靠 `scripts/env.sh` 的 `have_onnx` 探测，先确认 `PYTHON_BIN` 指向的 Python |
| 把 `graph.input` 的条数直接当成「要喂几个张量」 | 没建立「initializer 同名项要减掉」的意识，真实模型上这会让你喂错输入 |
| 说不出 `attribute` 与 `input` 的区别 | 第 3 章的核心分野还没建立，L1-ONNX-03 会正面考这一点 |

---

### L0-ONNX-02｜读分区报告，复述 ORT 一次会话建立的几拍

- **检验什么**：这条通过 = 你真的掌握了「`GetCapability` → `Compile` → 运行时派发」这三拍分别发生在什么时刻，以及 provider 列表的优先级语义
- **前置**：L0-ONNX-01
- **资源**：本地+工具链
- **预计耗时**：0.5h

**任务**：读 `out/onnx/03_partition_report.json` 与 `03_READING.md`，对着 JSON 里的 `available_providers` / `cpu_only.requested` / `cpu_only.active` / `provider_hit_counts` / `graph_nodes` 五个字段，写下：从 `ort.InferenceSession(...)` 这一行开始到 `sess.run()` 返回，ORT 依次做了哪几件事，哪几件只在建会话时做一次、哪几件每次 `run` 都做。再解释：为什么 `providers=[...]` 是一个**有序**列表。

**验收命令**：

```bash
cd onnx-delegate-lab
python - <<'PY'
import json
r = json.load(open("out/onnx/03_partition_report.json", encoding="utf-8"))
print("available:", r["available_providers"])
print("cpu requested:", r["cpu_only"]["requested"], "active:", r["cpu_only"]["active"])
print("hits:", r["cpu_only"].get("provider_hit_counts"))
print("nodes:", [n["op"] for n in r["graph_nodes"]])
print("dual:", "无第二 EP" if r["dual"] is None else r["dual"]["active"])
PY
```

**通过标准**：

- 你能把三拍对上具体现象：`GetCapability` 决定了 `provider_hit_counts` 里每个 provider 拿到多少节点；`Compile` 解释了为什么 `optimized_model_filepath` 导出的图与原图节点不同；派发解释了 profiling 事件里为什么每个节点带 provider 标签
- 你能说清 `active` 列表里为什么**总是**带着 `CPUExecutionProvider`（兜底 EP，任何 EP 不认的节点都得有人接）
- 只有 CPU 时 `dual` 为 `null`，你能指出这不是失败，而是脚本自动探测 `CUDAExecutionProvider` / `DmlExecutionProvider` / `CoreMLExecutionProvider` 都没命中

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 认为 `dual` 为 null 是脚本坏了 | 没读 `03_ort_ep_partition.py` 里的 `accel` 探测逻辑；双 EP 是可选观察项，不是主链路 |
| 把「优化」和「分区」混为一谈 | 图优化（融合）发生在分区之前/之中，两者产物不同：一个看 `03_optimized_cpu.onnx`，一个看 provider 归属 |
| `provider_hit_counts` 为空也不追究 | ORT 各版本 profiling JSON 字段名不同（`traceEvents` / 扁平列表 / `args.provider`），脚本已做兼容；字段名以本地版本为准，必要时直接打开 profile JSON 看原始 key |

---

## L1 改一处

### L1-ONNX-03｜换一种改图方式，撞出「属性 vs 输入」的分界

- **检验什么**：这条通过 = 你真的掌握了「attribute 是编译期常量、input 是数据流边」这条分界，以及改哪一边会被 `checker` / `shape_inference` 拦下
- **前置**：L0-ONNX-01
- **资源**：本地+工具链
- **预计耗时**：1h

**任务**：改 `onnx_lab/02_mutate_graph.py`。当前它做两件事：`insert_identity_after(g, "h_act", "h_act_probe")` 与 `replace_initializer(g, "bias2", zeros)`。请**换成两种新改法**（保留原函数，另写新的）：

1. 插一个 `Transpose`（而不是 `Identity`）——注意它会改变形状，你得决定插在哪、要不要成对插回来；
2. 改一个**属性**而不是 initializer——例如把 `Gemm` 节点的 `transB` / `alpha` 改掉（属性名以本地 opset 的 Gemm 规范为准，先 `print(node.attribute)` 看这张图上实际有哪些）。

两种改法都跑 `checker.check_model` + `shape_inference.infer_shapes` + ORT 数值对照，记录哪一种被拦下、在哪一步被拦下。

**先预测再动手**：

1. 插入 `Transpose` 后不改任何下游节点，你预期 `checker` 会过吗？`infer_shapes` 会过吗？ORT 建会话会过吗？——这三关卡的严格程度谁强谁弱，为什么？
2. 把 `Gemm` 的 `alpha` 从 1.0 改成 2.0：`checker` 会不会报错？输出数值会不会变？为什么这两个答案不一样？
3. 如果你想让「乘 2」这件事变成一条**数据流边**而不是属性，图该怎么改？（提示：加一个 `Mul` 节点 + 一个 initializer）这说明属性和输入在表达能力上差在哪？

**验收命令**：

```bash
cd onnx-delegate-lab
python onnx_lab/01_build_and_infer.py          # 确保基线模型存在
python onnx_lab/02_mutate_graph.py             # 跑你改过的版本
python - <<'PY'
import numpy as np, onnx, onnxruntime as ort
from onnx import checker, shape_inference
m = onnx.load("out/onnx/02_tiny_mlp_mutated.onnx")
checker.check_model(m)
shape_inference.infer_shapes(m)
x = np.ones((2, 3), dtype=np.float32)
a = ort.InferenceSession("out/onnx/01_tiny_mlp.onnx", providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
b = ort.InferenceSession("out/onnx/02_tiny_mlp_mutated.onnx", providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
print("shape:", a.shape, b.shape, "allclose:", np.allclose(a, b))
PY
```

**通过标准**：

- 两种改法各留下一份可复现的记录：**哪一关报错、报错文案的关键词是什么**（`checker` 的类型/形状不匹配 vs ORT 建会话时的 kernel 找不到）
- 改属性的那一路：`checker` 通过但数值改变，你能解释为什么 `checker` 管不着「算得对不对」
- 插 `Transpose` 的那一路：你能说清形状是在哪一步被发现不一致的

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 插完 `Transpose` 什么都没报错，数值也没变 | 你插在了不影响输出的分叉上，或成对插入互相抵消了；没建立「改图必须有可观察差异」的验证习惯 |
| 直接改 `node.attribute[0].f` 报类型错 | 不清楚 `AttributeProto` 是带 type tag 的联合体（`i` / `f` / `s` / `t` / `ints`...），该用 `helper.make_attribute` 重建 |
| 认为 `checker` 过了就等于图是对的 | `checker` 只查结构合法性与类型一致性，不查语义正确性；数值对照才是最后一道关 |

---

### L1-ONNX-04｜造两张非法图，读 checker 的原话

- **检验什么**：这条通过 = 你真的掌握了 ONNX 的两条硬性图语义：**节点必须拓扑有序**、**每个值名字只能被定义一次（输出 SSA）**
- **前置**：L0-ONNX-01
- **资源**：本地+工具链
- **预计耗时**：0.5h

**任务**：写一个一次性脚本（不必并入 lab），加载 `out/onnx/01_tiny_mlp.onnx`，分别制造两种非法：

1. **打乱拓扑序**：把 `graph.node` 的顺序倒过来（消费者排在生产者前面）；
2. **重复定义**：让两个节点输出同一个名字（例如把 `Relu` 的输出名也改成 `Gemm` 的输出名）。

各自调用 `checker.check_model`，把**完整报错文案**抄下来。

**先预测再动手**：

1. ONNX 明确要求节点按拓扑序排列。那么「乱序但依赖关系仍然可解」这件事，`checker` 是会自己排序容忍，还是直接拒绝？为什么规范要选后者？（提示：谁来消费这个文件——运行时想不想做拓扑排序）
2. 两个节点输出同名，报错会指向「重复定义」还是「找不到输入」？你预期报错发生在哪一层检查？
3. 如果只是**打乱 initializer 的顺序**呢？会不会报错？为什么它和 node 的顺序要求不同？

**验收命令**：

```bash
cd onnx-delegate-lab
python - <<'PY'
import onnx
from onnx import checker

def try_check(tag, m):
    try:
        checker.check_model(m)
        print(f"[{tag}] 通过（说明这不是 checker 管的事）")
    except Exception as e:
        print(f"[{tag}] 报错：{type(e).__name__}: {e}")

m = onnx.load("out/onnx/01_tiny_mlp.onnx")
bad = onnx.ModelProto(); bad.CopyFrom(m)
nodes = list(bad.graph.node); del bad.graph.node[:]; bad.graph.node.extend(reversed(nodes))
try_check("topo-reversed", bad)

bad2 = onnx.ModelProto(); bad2.CopyFrom(m)
dup = bad2.graph.node[0].output[0]
bad2.graph.node[1].output[0] = dup
try_check("duplicate-output", bad2)
PY
```

**通过标准**：

- 两种非法都被 `checker` 拒绝，你抄下了报错文案，并能指出报错里提到的具体名字（哪个 node、哪个值）
- 你能用「SSA + DAG」两个词解释这两条约束各自守住了什么：拓扑序让运行时可以**单遍**执行，唯一定义让「名字 → 生产者」是一个函数而不是关系
- 你能说出 initializer 顺序不受此约束的原因（它们是图的常量集合，不参与执行顺序）

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 直接改 `m.graph.node` 后原模型也被改了 | 没意识到 protobuf 是引用语义，要 `CopyFrom` 深拷贝；这个坑在写改图 pass 时会反复出现 |
| 只报了「找不到输入 X」就以为是别的问题 | 没把报错翻译回语义：消费者排在生产者前，对单遍检查而言就等于「该名字还没被定义」 |
| 期待 ORT 会容忍乱序图 | 把「运行时能不能救」和「格式合法不合法」混在一起；ONNX 选择把负担放在生产者一侧 |

---

### L1-ONNX-05｜跨一个 since_version 分界点改 opset

- **检验什么**：这条通过 = 你真的掌握了「算子身份是 `(domain, op_type, since_version)` 三元组」，以及 opset 号不是装饰而是语义契约
- **前置**：L0-ONNX-01
- **资源**：本地+工具链
- **预计耗时**：1h

**任务**：不要凭印象猜版本号。先用本地 `onnx` 查出某个算子的所有 `since_version`，挑一个**真实存在的跨越点**，再动手。

1. 用 `onnx.defs.get_all_schemas_with_history()`（函数名以本地版本为准，也可用 `onnx.defs.get_schema(op_type, max_inclusive_version=N)`）列出候选算子的版本历史。推荐候选：`ReduceSum`（某个版本起 `axes` 从属性改成了输入）、`Squeeze` / `Unsqueeze`（同类改动）、`Softmax`（`axis` 的语义在某版发生过变化）——**具体在哪一版变，以你本地查出来的 `since_version` 为准**。
2. 把 `03_ort_ep_partition.py` 里 `build_partition_demo_model` 的 `helper.make_opsetid("", 17)` 改成跨越点**两侧**的版本各跑一次（例如 `since_version` 是 N，就试 `N` 和 `N-1`），观察 `checker` 与 ORT 建会话的行为差异。
3. 再试一次 `onnx.version_converter.convert_version`，看它能不能自动把图搬过这个分界点。

**先预测再动手**：

1. 如果一个算子在版本 N 把 `axes` 从属性挪成了输入，你把 opset 号从 N 降到 N-1 但**节点写法不动**，谁会先报错：`checker` 还是 ORT？报错会说什么？
2. `checker` 拿到一个「opset 号与节点写法不匹配」的模型，它是按声明的 opset 去找 schema，还是按节点长相猜？这决定了报错文案的形态。
3. `version_converter` 失败时，你预期它是抛异常还是静默产出一张语义已经变了的图？哪种更危险？

**验收命令**：

```bash
cd onnx-delegate-lab
# 第一步：查清版本历史（不要写死版本号，以这里的输出为准）
python - <<'PY'
import onnx.defs as d
for op in ("ReduceSum", "Squeeze", "Softmax"):
    vs = sorted({s.since_version for s in d.get_all_schemas_with_history() if s.name == op and s.domain == ""})
    print(op, "since_versions:", vs)
PY
# 第二步：改 make_opsetid 后重跑，观察 checker / ORT
python onnx_lab/03_ort_ep_partition.py
```

**通过标准**：

- 你写出了具体的三元组结论，形如「本地 onnx 里 `("", ReduceSum)` 的 since_version 列表是 [...]，我选了 N 作为跨越点」
- 跨越点两侧至少有一侧报错，且你能把报错归因到「声明的 opset 决定了用哪份 schema 校验」
- 你能说清 `version_converter` 的适用范围与风险（不是所有算子都有转换规则；转不了时要么报错，要么留下你必须自己核对的语义差异）

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 抄了一个网上看来的版本号，结果本地没这回事 | 没建立「先查 schema 再动手」的习惯；opset 分界点随 onnx 版本演进，硬编码必然过时 |
| 改了 opset 号但什么都没变 | 挑的算子在这两个版本间根本没改过；说明还没真正理解 `since_version` 的含义 |
| ORT 报 `Unsupported model IR version` 或找不到 kernel 就放弃 | 没区分三层限制：ONNX 的 IR version、opset version、ORT 实现覆盖的 opset 范围 |

---

## L2 加组件（主判据）

### L2-ONNX-06｜从零构一张带分支的图，并接进主链路

- **检验什么**：这条通过 = 你掌握了「用 `onnx.helper` 独立构造一张非线性（有分叉与汇合）的合法图」的完整链路：造节点 → 造 value_info → 造 initializer → checker → shape inference → ORT 跑通 → 与 numpy 参考实现对齐
- **前置**：L0-ONNX-01、L1-ONNX-04
- **资源**：本地+工具链
- **预计耗时**：2h

**任务**：新增 `onnx_lab/04_branch_graph.py`，照 `01_build_and_infer.py` 的骨架（`sys.path.insert` + `from lab_common import banner, out_dir, require_onnx, write_text`）写：

- 手工构造一张**带分支**的图：`Split` 分成两路 → 两路走**不同**的算子（例如一路 `Relu`、一路 `Sigmoid`，或一路 `Mul` 常量、一路 `Add` 常量）→ `Concat` 汇合。至少 5 个节点，至少 1 个 initializer。
- 走完全套检查：`checker.check_model` → `shape_inference.infer_shapes`，并**打印推断出来的中间张量形状**（证明 shape inference 穿过了分叉与汇合）。
- 用 `ort.InferenceSession(..., providers=["CPUExecutionProvider"])` 跑一遍，和你手写的 numpy 参考实现 `np.allclose` 对齐。
- 产物：`out/onnx/04_branch_graph.onnx` 与 `out/onnx/04_READING.md`（用 `write_text` 写）。
- 把 `04_branch_graph.py` 加进 `scripts/run_onnx.sh` 的循环列表里。

**先预测再动手**：

1. `Split` 的分法在你选的 opset 里是属性 `split` 还是第二个**输入**？（回到 L1-ONNX-05 的查法，用 `onnx.defs.get_schema("Split")` 确认）选错了会在哪一关被拦下？
2. 如果你不给中间张量声明 `value_info`，`shape_inference` 还能推出它们的形状吗？推出来后存在哪里——`graph.value_info` 还是 `graph.output`？
3. 两路分支的输出 dtype 如果不一致（比如一路被你写成了 int64），`Concat` 会在 `checker`、`infer_shapes`、ORT 三关的哪一关炸？

**验收命令**：

```bash
cd onnx-delegate-lab
python onnx_lab/04_branch_graph.py
bash scripts/run_onnx.sh          # 04 应作为第四步被带起来
python - <<'PY'
import onnx
m = onnx.load("out/onnx/04_branch_graph.onnx")
onnx.checker.check_model(m)
inferred = onnx.shape_inference.infer_shapes(m)
print("nodes:", [n.op_type for n in m.graph.node])
print("推断出的中间 value_info:")
for vi in inferred.graph.value_info:
    print("  ", vi.name, [d.dim_value or d.dim_param for d in vi.type.tensor_type.shape.dim])
PY
```

**通过标准**：

- `04_branch_graph.onnx` 与 `04_READING.md` 生成成功；`run_onnx.sh` 一键跑完四步且不报错
- `checker` 无异常；`infer_shapes` 后 `graph.value_info` 里出现**分叉两路各自的中间张量**，形状是具体数字而非空
- 脚本内断言 ORT 输出与 numpy 参考实现 `np.allclose`（把这个断言写进脚本，让它成为机器判据）

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| `Split` 报输出数量不对 | 没意识到 `Split` 的输出个数由节点的 output 列表长度决定，且要和 split 的分法自洽 |
| `infer_shapes` 后 `value_info` 是空的 | 图的输入没给全形状（`make_tensor_value_info` 的 shape 留了 None），推断从源头就断了 |
| ORT 结果和 numpy 差一点点但不 allclose | 多半是你的参考实现没对齐 axis 或 `Concat` 的拼接维；顺手把 `atol/rtol` 调大来「通过」是自欺 |
| `run_onnx.sh` 里加了文件却没跑 | 没注意脚本是遍历一个**写死的文件名列表**（`for s in 01_... 02_... 03_...`），要把 04 加进去 |

---

### L2-ONNX-07｜写一个自己的图优化 pass：模式匹配 + 重写

- **检验什么**：这条通过 = 你掌握了「pattern 匹配与重写」在非 MLIR 体系里的等价物——手写遍历 `GraphProto`、识别模式、安全地删节点接边，并用数值与节点数双重验证
- **前置**：L2-ONNX-06（你得先能自己造图，才能造出待优化的图）
- **资源**：本地+工具链
- **预计耗时**：2.5h

**任务**：新增 `onnx_lab/05_my_pass.py`（或并入 04 亦可，但要独立成函数），写一个**纯 Python** 的图优化 pass，三选一：

- **Transpose-Transpose 抵消**：连续两个 `Transpose`，若 perm 复合后是恒等，则两个都删；
- **连续 Identity 折叠**：一串 `Identity` 链塌成一条边（这条和 `02_mutate_graph.py` 插入的 `probe_identity` 正好呼应）；
- **`Add(x, 0)` 消除**：另一个输入是全零 initializer 时删掉该节点。

pass 的骨架：遍历 `graph.node` 建「名字 → 生产者 / 消费者」索引 → 匹配模式 → **把下游消费者的输入改接到上游输出** → 删除被消解的节点 →（若被删节点的输出恰好是 `graph.output`，要特殊处理）→ 重新 `checker` + `infer_shapes`。

**优化前后必须给出两组对比**：节点数变化、同一输入下 ORT 输出的 `np.allclose`。

这条与 [`./02-mlir.md`](./02-mlir.md) 的 `RewritePattern` 是同一件事的两种写法——做完请对照着看：MLIR 帮你做了什么（匹配驱动、use-def 链、不动点迭代），在这里你得手写哪些。

**先预测再动手**：

1. 你删掉一个节点时，怎么知道它的输出**没有别的消费者**？如果有第二个消费者你还删不删？——这就是 MLIR 里 `use` 计数替你做的事，你打算用什么数据结构补上？
2. 被消解的节点的输出如果同时是 `graph.output` 的名字，直接改接下游会发生什么？（提示：图的输出名是对外契约，不能随便改）
3. 你的 pass 跑**一遍**够吗？三个连续 `Identity` 折叠一次之后还剩几个？需要循环到不动点吗——怎么判断「不再变化」？
4. 删节点时你是原地 `del graph.node[i]` 还是重建一个列表？在遍历中原地删会踩什么坑？

**验收命令**：

```bash
cd onnx-delegate-lab
python onnx_lab/01_build_and_infer.py && python onnx_lab/02_mutate_graph.py
python onnx_lab/05_my_pass.py                 # 输入取 02 的产物（里面有 probe_identity）
python - <<'PY'
import numpy as np, onnx, onnxruntime as ort
before = onnx.load("out/onnx/02_tiny_mlp_mutated.onnx")
after  = onnx.load("out/onnx/05_optimized.onnx")
onnx.checker.check_model(after)
print("nodes before:", len(before.graph.node), [n.op_type for n in before.graph.node])
print("nodes after :", len(after.graph.node),  [n.op_type for n in after.graph.node])
x = np.ones((2, 3), dtype=np.float32)
a = ort.InferenceSession("out/onnx/02_tiny_mlp_mutated.onnx", providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
b = ort.InferenceSession("out/onnx/05_optimized.onnx", providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
print("allclose:", np.allclose(a, b), "max diff:", float(np.abs(a - b).max()))
PY
```

**通过标准**：

- 优化后节点数**严格减少**，且减少的正是你要消解的那类算子（打印出的 `op_type` 列表里它消失了）
- 优化后 `checker.check_model` 无异常、`infer_shapes` 无异常
- 同一输入下优化前后 `np.allclose` 为 `True`（这是「变换必须保语义」的机器判据）
- 你能说出一个**你的 pass 故意不处理**的情形，以及为什么处理它会出错（这一条比「能删」更重要）

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 删完 `checker` 报某个输入找不到生产者 | 只删了节点没改下游的输入名；不理解删节点的本质是「接边」而不是「移除一行」 |
| 图的最终输出变成了空/改名 | 忽略了 `graph.output` 是对外契约，被删节点若产出图输出，必须把上游输出重命名或保留一个 `Identity` |
| 数值对不上 | 匹配条件写松了（比如两个 `Transpose` 的 perm 复合并非恒等也给删了）；这正是 MLIR 里 pattern 的 match 条件要写严的原因 |
| 遍历中原地删导致漏删 | 边遍历边改容器；应先收集待删索引再倒序删，或整体重建 node 列表 |

---

### L2-ONNX-08｜四档优化级别对拍，指认 ORT 做了哪些融合

- **检验什么**：这条通过 = 你掌握了「ORT 的图优化是分档的、可导出的、发生在分区之前」，并能从两张图的 diff 里读出具体做了什么融合
- **前置**：L0-ONNX-02
- **资源**：本地+工具链
- **预计耗时**：1.5h

**任务**：写一个脚本（可放 `onnx_lab/` 下，或直接一次性脚本），对 `out/onnx/03_partition_demo.onnx`（Gemm→Relu→Softmax→ReduceSum）用 `SessionOptions.graph_optimization_level` 的四档分别建会话，各自导出 `optimized_model_filepath`：

- `ORT_DISABLE_ALL` / `ORT_ENABLE_BASIC` / `ORT_ENABLE_EXTENDED` / `ORT_ENABLE_ALL`（枚举成员名以本地 `onnxruntime` 版本为准，先 `print([x for x in dir(ort.GraphOptimizationLevel) if not x.startswith('_')])`）

然后对四份产物做结构 diff：节点数、`op_type` 直方图、出现的**新算子名**（融合产物往往叫 `FusedGemm` / `FusedMatMul` / `Gelu` 之类，且可能带非空 `domain`，如 `com.microsoft`）。最后回答：哪一档开始出现融合、融合掉了哪几个节点、新算子的 domain 说明了什么。

**先预测再动手**：

1. `ORT_DISABLE_ALL` 导出的图，和你喂进去的原图会**完全一样**吗？（提示：常量折叠、冗余节点消除分别属于哪一档？）
2. Gemm→Relu 这对，你预期在哪一档被融合？融合后的新算子还是标准 ONNX 算子吗——如果不是，这张导出的图还能被别的运行时加载吗？
3. 这些优化发生在 EP 分区**之前**还是之后？如果在之前，融合会不会反过来影响分区边界的位置？

**验收命令**：

```bash
cd onnx-delegate-lab
python - <<'PY'
from collections import Counter
from pathlib import Path
import onnx, onnxruntime as ort

src = "out/onnx/03_partition_demo.onnx"
levels = [n for n in dir(ort.GraphOptimizationLevel) if n.startswith("ORT_")]
print("本地可用档位:", levels)
for name in levels:
    dst = Path(f"out/onnx/08_opt_{name.lower()}.onnx")
    so = ort.SessionOptions()
    so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, name)
    so.optimized_model_filepath = str(dst)
    ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
    g = onnx.load(dst).graph
    hist = Counter((n.domain or "ai.onnx", n.op_type) for n in g.node)
    print(f"{name:22s} nodes={len(g.node):2d}  {dict(hist)}")
PY
```

**通过标准**：

- 四份 `08_opt_*.onnx` 全部生成，你能列出一张「档位 → 节点数 → 算子直方图」的对照表
- 至少指认出**一处**具体的结构变化（节点数减少、某算子消失、出现带 `com.microsoft` 之类 domain 的新算子），并说清它对应哪种优化（常量折叠 / 冗余消除 / 算子融合 / layout 变换）
- 你能解释为什么带非标准 domain 的融合算子会**降低模型的可移植性**，以及这与 L1-ONNX-05 的 `(domain, op_type, version)` 三元组是同一件事

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 四档产物完全一样 | 这张 demo 图太小、没有可融合模式；换一张（如你在 L2-ONNX-06 造的分支图，或加几个 `Add`/`Mul`）再试，别急着下「ORT 没优化」的结论 |
| `AttributeError: ORT_ENABLE_XXX` | 枚举名随版本变化，必须先 `dir()` 查；这就是「以本地版本为准」的典型场景 |
| 认为优化级别越高一定越快 | 高档位可能引入 EP 特定算子，反而让某些节点落不到目标 EP 上；性能与可移植性是一对矛盾 |

---

### L2-ONNX-09｜把一个算子推出 EP 的能力范围，看分区数怎么变

- **检验什么**：这条通过 = 你掌握了 `GetCapability` 的实际含义：EP 声明能吃哪些子图 → ORT 据此切分 → 每个分区各自 `Compile`；以及边界数量为什么直接对应拷贝/同步代价
- **前置**：L0-ONNX-02、L2-ONNX-08
- **资源**：本地+工具链（有第二 EP 更直观，无卡也能完成）
- **预计耗时**：2h

**任务**：改 `onnx_lab/03_ort_ep_partition.py` 的 `build_partition_demo_model`，把当前的 `Gemm → Relu → Softmax → ReduceSum` 改成**故意制造边界**的结构。两种做法任选（都做更好）：

1. **插入一个冷门算子**到链路中间（例如 `NonZero`、`Unique`、`StringNormalizer` 这类实现覆盖少的算子，或用一个非默认 domain 的算子），让它落在 EP 能力之外；
2. **交错排列**：把容易被加速 EP 吃掉的算子（`Gemm` / `MatMul` / `Conv` / `Relu`）与常留 CPU 的算子（归约、`Softmax`、形状类算子）**交替**排布，例如 `Gemm → Softmax → Gemm → Softmax`，制造多个来回。

改完重跑，读 `03_partition_report.json` 的 `provider_hit_counts` 与 profiling 里每个节点的 provider 标签，数出**分区段数**（同一 provider 的连续段算一段，provider 切换一次算一个边界）。

无第二 EP 时的降级做法：脚本里已有 `conceptual_boundary_note()` 按算子类型给出「典型归属直觉」——用它人工推演边界位置与段数，同时明确写下「本机只有 CPU EP，所以这是推演不是实测，拿不到真实拷贝耗时」。

**先预测再动手**：

1. 交错排布 `Gemm → Softmax → Gemm → Softmax`，如果加速 EP 只吃 `Gemm`，会切出几个分区？边界有几个？每个边界意味着几次数据搬运（去 + 回）？
2. 如果某个 EP 只能吃**单个节点**而不是连续子图，总体性能可能比全 CPU 更差——为什么？这解释了为什么 EP 的 `GetCapability` 倾向于返回**尽可能大的连通子图**。
3. 一个 EP 不认识的算子夹在中间，ORT 会报错还是自动 fallback？fallback 到哪个 EP？（回看 L0-ONNX-02 里 `active` 列表末尾总是 CPU 那件事）
4. 你的改动会不会同时影响 L2-ONNX-08 的融合结果？融合和分区谁先谁后，会不会互相搅局？

**验收命令**：

```bash
cd onnx-delegate-lab
python onnx_lab/03_ort_ep_partition.py        # 跑你改过的模型结构
python - <<'PY'
import json
r = json.load(open("out/onnx/03_partition_report.json", encoding="utf-8"))
print("nodes:", [n["op"] for n in r["graph_nodes"]])
print("cpu hits:", r["cpu_only"].get("provider_hit_counts"))
if r["dual"]:
    print("dual active:", r["dual"]["active"])
    print("dual hits:", r["dual"].get("provider_hit_counts"))
    seq = [e["provider"] for e in r["dual"].get("sample_events", []) if "provider" in e]
    segs = [p for i, p in enumerate(seq) if i == 0 or p != seq[i-1]]
    print("provider 切换序列:", segs, "→ 边界数:", max(len(segs) - 1, 0))
else:
    print("无第二 EP：按 03_READING.md 的概念对照表人工推演边界（记得注明这是推演）")
PY
```

**通过标准**：

- 改动后的 `graph_nodes` 列表确实交错/含冷门算子，且你能写出一张「节点 → 预期归属 → 实测（或推演）归属」的表
- 你给出了**具体的分区段数与边界数**，并把每个边界折算成「几次跨设备拷贝 + 几个同步点」
- 你能把结论落回 `03_READING.md` 里的四类代价表（数据拷贝 / layout 转换 / 内存空间切换 / 同步点），每类各举一个本次实验里的具体位置
- 无 GPU 时，报告里明确标注了哪些数字是推演、哪些是实测

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 加了冷门算子后 `checker` 直接失败 | 输入类型/形状没按该算子的 schema 给；先 `onnx.defs.get_schema("NonZero")` 看清签名 |
| ORT 建会话报「找不到 kernel」就以为实验失败 | 恰恰相反，这说明连兜底 CPU EP 都没实现该算子——换一个「CPU 有、加速 EP 没有」的算子才能造出边界 |
| 只看 `provider_hit_counts` 的总数就下结论 | 计数只说「谁吃了多少节点」，说不出**边界在哪**；边界要看 provider 在节点序列上的切换位置 |
| 认为边界代价只有拷贝 | 漏掉 layout 转换与同步点；跨 EP 的隐式 Transpose 常常是真正的性能杀手 |

---

## L3 打通

### L3-ONNX-10｜真机双 EP：从 profiling 里量化跨 EP 边界的代价

- **检验什么**：这条通过 = 你把「分区边界有代价」从一句话变成了带单位的数字——每个节点花了多少微秒、跨边界的拷贝算子占了多少
- **前置**：L2-ONNX-09
- **资源**：单卡GPU（需申请；无卡有降级方案，见下）
- **预计耗时**：2h

**任务**：装 `onnxruntime-gpu`（注意：与 CPU 版 `onnxruntime` **不能共存**，先卸载；CUDA / cuDNN 版本对应关系以本地 ORT 版本的官方矩阵为准），确认 `get_available_providers()` 里出现 `CUDAExecutionProvider`。然后用 L2-ONNX-09 改造过的交错模型，跑双 EP 会话，打开 `enable_profiling`，从生成的 profiling JSON 里逐节点提取 `provider` 与 `dur`（微秒），做三件事：

1. 按 provider 汇总总耗时，算出 CPU 段与 GPU 段各占多少；
2. 找出 profiling 里的**内存拷贝 / 数据传输类事件**（不同版本命名不同，常见含 `Memcpy` / `MemcpyToHost` / `MemcpyFromHost` 字样），统计它们的次数与总耗时；
3. 与纯 CPU EP 的端到端耗时对比——双 EP 到底是快了还是慢了，边界代价占多少比例。

**降级方案（无 GPU 时，不阻塞）**：仍用单 CPU EP 跑完 profiling，把每个节点的 `dur` 拿到手，按 L2-ONNX-09 推演出的边界位置**人工推演**：假设跨边界要搬多大的张量、按典型 PCIe 带宽估一个拷贝耗时上界。产出同样一张表，但**必须在报告里写明「拷贝耗时为估算，非实测；本机无第二 EP，拿不到真实的跨设备搬运数据」**。这条降级版本可以完成分析框架，唯独「真实拷贝开销」这一个数字必须真机才作数。

**先预测再动手**：

1. 你预期跨 EP 的双 EP 版本比纯 CPU **快还是慢**？在这个小模型上（张量只有几十个元素），拷贝固定开销和算子本身的耗时哪个更大？
2. profiling 里除了每个节点的事件，还有 session 级、run 级的事件。你打算怎么区分「算子耗时」和「框架开销」？把它们加在一起会不会重复计数？
3. 如果 GPU 段总耗时明显小于 CPU 段，但端到端反而更慢，差额去了哪里？

**验收命令**：

```bash
# 环境准备（版本对应关系以官方矩阵为准）
pip uninstall -y onnxruntime && pip install onnxruntime-gpu
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"

cd onnx-delegate-lab
python onnx_lab/03_ort_ep_partition.py        # dual 分支这次应该真的跑起来
python - <<'PY'
import glob, json
from collections import defaultdict
prof = max(glob.glob("**/*.json", recursive=True), key=lambda p: ("profile" in p, p))
data = json.load(open(prof, encoding="utf-8"))
evs = data if isinstance(data, list) else data.get("traceEvents", [])
by_prov, copies = defaultdict(float), []
for e in evs:
    args = e.get("args", {}) or {}
    prov = args.get("provider") or args.get("provider_type")
    dur = e.get("dur", 0)
    if prov:
        by_prov[prov] += dur
    if "Memcpy" in (e.get("name") or ""):
        copies.append((e["name"], dur))
print("profile:", prof)
print("各 provider 总耗时(us):", dict(by_prov))
print("拷贝事件数:", len(copies), "总耗时(us):", sum(d for _, d in copies))
PY
```

**通过标准**：

- `get_available_providers()` 中出现 `CUDAExecutionProvider`，且 `03_partition_report.json` 的 `dual.active` 非空、`provider_hit_counts` 里两个 provider 都有命中
- 你产出了一张按 provider 汇总的耗时表，以及拷贝类事件的次数与总耗时
- 你能给出一个带数字的结论，形如「这张图上跨 EP 边界共 N 处，拷贝事件 M 次共 X 微秒，占端到端的 Y%；因此在这个规模下双 EP 不划算，需要算子规模到某个量级才回本」
- 无卡走降级路线时，报告里清楚区分了实测值与估算值

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 装完 `onnxruntime-gpu` 仍只有 CPU provider | CUDA/cuDNN 版本与 ORT 不匹配，或 CPU 版没卸干净（两个包同时装会互相覆盖）；查官方版本矩阵 |
| profiling JSON 里找不到 `provider` 字段 | 字段名随版本变化（`provider` / `provider_type` / 放在 `cat` 里），以本地版本为准，直接打开文件看原始 key |
| 找不到任何 Memcpy 事件却断言「没有拷贝开销」 | 拷贝可能被融进算子内部，或 ORT 把小张量留在了 CPU 上；先确认边界确实存在（回 L2-ONNX-09） |
| 只比总时间，不看首次运行 | 第一次 run 含 kernel 编译/预热，必须跑多次取稳定值再比较 |

---

## 条目 × 资源需求速查

| 编号 | 任务 | 级别 | 资源 | 耗时 |
|------|------|------|------|------|
| L0-ONNX-01 | 跑通 ONNX 轨，手拆三层结构 | L0 | 本地+工具链 | 1h |
| L0-ONNX-02 | 读分区报告，复述会话三拍 | L0 | 本地+工具链 | 0.5h |
| L1-ONNX-03 | 换改图方式，撞属性 vs 输入 | L1 | 本地+工具链 | 1h |
| L1-ONNX-04 | 造非法图，读 checker 原话 | L1 | 本地+工具链 | 0.5h |
| L1-ONNX-05 | 跨 since_version 分界改 opset | L1 | 本地+工具链 | 1h |
| **L2-ONNX-06** | **从零构带分支的图并接进主链路** | L2 | 本地+工具链 | 2h |
| **L2-ONNX-07** | **手写图优化 pass：匹配 + 重写** | L2 | 本地+工具链 | 2.5h |
| L2-ONNX-08 | 四档优化级别对拍看融合 | L2 | 本地+工具链 | 1.5h |
| L2-ONNX-09 | 制造 EP 能力缺口看分区变化 | L2 | 本地+工具链 | 2h |
| L3-ONNX-10 | 真机双 EP 量化边界代价 | L3 | 单卡GPU（可降级） | 2h |

**只有 L3-ONNX-10 需要 GPU，且给了不阻塞的降级方案。** 其余九条全部在个人开发机 CPU 上完成，本地部分约 12 小时，含 L3 共约 14 小时。

**下一步**：完成本册后接 [`./06-executorch.md`](./06-executorch.md)——同一个 lab 的另一条轨，把「谁来吃这段子图」从 ORT 的 EP 换成 ExecuTorch 的 Partitioner，边界账可以直接对照着算。想看编译期（而非运行时）做同一件事，去 [`./03-iree.md`](./03-iree.md) 的 `flow.dispatch`；想看 L2-ONNX-07 手写的模式匹配在成熟基础设施里长什么样，去 [`./02-mlir.md`](./02-mlir.md) 的 `RewritePattern`。
