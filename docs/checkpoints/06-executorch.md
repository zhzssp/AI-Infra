# 检验体系 06｜ExecuTorch

> **对应学习文档**：[`../learning-guides/executorch-learning-guide.md`](../learning-guides/executorch-learning-guide.md)  
> **对应动手项目**：[`onnx-delegate-lab/`](../../onnx-delegate-lab/) 的 ExecuTorch 轨（`executorch_lab/`）  
> **分级与资源标签定义**：[`./README.md`](./README.md)

本册检验的是这些知识点能不能落成代码：**Partitioner 契约（只打 `delegation_tag`，不改图结构）、tag 粒度如何决定 delegate 子图数与分区边界数、Edge Dialect 为什么要单独一层、`to_backend` 对划分合法性的要求、边界代价与 ORT EP / IREE dispatch 的同构关系**。

学习指南里的过关标准是口述题（能不能默画流水线、能不能解释为什么禁止乱改图）。本册全部换成动手题：**每一条都要跑出一个可比较的数字或一段可复现的报错**。

## 本册的实验对象

全仓库共用的主角 `tiny_mlp`（`Gemm→Relu→Add`）**叠两层，中间夹一个 `softmax`**：

```
x ─▶ addmm ─▶ relu ─▶ add ─▶ softmax ─▶ addmm ─▶ relu ─▶ add ─▶ y
     └───── 可委托 ─────┘    portable   └───── 可委托 ─────┘
      索引 0    1     2         3         4      5     6
```

**为什么要叠两层、为什么要夹一个 softmax**：单独一份 `tiny_mlp` 三个算子谁都支持，会被整个后端一口吃掉——**没有边界就量不出边界代价**，本册九条题一条也做不成。夹一个 portable 的 `softmax` 把图切成两段，`per_node` 与 `connected` 才会给出不同的子图数与边界数。同样的手法在 [`05-onnx-ort.md`](./05-onnx-ort.md) 里是给主角接一条 `Softmax + ReduceSum` 的尾巴，动机完全一样。

基线数字（L0-ET-01 会验证）：`per_node` = 6 子图 / 6 边界，`connected` = 2 子图 / 2 边界。**后面每一条题都是在这两组数字上做扰动。**

**入门线**：L0 两条全做 + L1 至少一条 + **L2-ET-05 与 L2-ET-07 必做**（一个是「能不能自己写 Partitioner」，一个是「知不知道 Partitioner 的边界在哪」）。

## 资源总览：先读这一段

**本册全部条目都不需要 GPU。** 核心路径全在 CPU 上。

更重要的一点：**没装 ExecuTorch 也能完成本册大部分条目**。`executorch_lab/01_partitioner_lab.py` 有两条路径——

| 路径 | 触发条件 | 能验证什么 | 不能验证什么 |
|------|---------|-----------|-------------|
| **真实路径** | 装了 `torch` + `executorch` | `torch.export → to_edge → to_backend → to_executorch` 全链路、`.pte` 落盘、`to_backend` 的合法性检查与报错 | —— |
| **模拟路径** | 缺 `torch` 或缺 `executorch` | tag 策略 → 子图数 / 边界数的**因果账**（纯 Python 复现 `simulate_partition`），这是研究问题①②的核心 | Edge Dialect 的真实形态、`to_backend` 的真实报错、`.pte` 内容 |

**关于 ExecuTorch 安装**：ExecuTorch 与 `torch` 版本强绑定（每个 ET release 只对应特定的 torch nightly/stable），跨版本安装失败是常态，不要在这上面耗时间。本册的设计原则是：**每一条依赖真实 ET 的条目，都配一条模拟路径的等价降级**，只有 L2-ET-06 和 L3-ET-09 例外（这两条已明确标注）。装不上就走模拟路径，L0–L2 的绝大部分照样通过。

**API 稳定性警告**：ExecuTorch 的 AOT 入口（`to_edge` / `EdgeProgramManager.to_backend` / `exir.backend.*` 的模块路径 / `PartitionResult` 字段名）在不同版本之间**改过多次**。本册凡涉及具体 API 处，一律以「**以本地版本为准**」标注，并给出自查命令。看到 `ImportError` 先查版本，不要先怀疑自己写错逻辑。

```bash
# 开工前自查：确认你走的是哪条路径
cd onnx-delegate-lab
python - <<'PY'
try:
    import torch; print("torch:", torch.__version__)
except Exception as e:
    print("torch: MISSING —", type(e).__name__)
try:
    import executorch
    v = getattr(executorch, "__version__", None)
    if v is None:
        try:
            from importlib.metadata import version; v = version("executorch")
        except Exception:
            v = "<无 __version__ 属性，用 pip show executorch 查>"
    print("executorch:", v, "@", executorch.__file__)
except Exception as e:
    print("executorch: MISSING —", type(e).__name__)
for path in ("executorch.exir.to_edge",
             "executorch.exir.backend.partitioner.Partitioner",
             "executorch.exir.backend.backend_details.PartitionResult"):
    mod, _, name = path.rpartition(".")
    try:
        __import__(mod); getattr(__import__(mod, fromlist=[name]), name); print("OK  ", path)
    except Exception as e:
        print("MISS", path, "—", type(e).__name__)
PY
```

最后一段输出就是你的「API 以本地版本为准」清单：哪一行 `MISS`，就去本地 `executorch` 包里搜同名符号的新位置。

---

## L0 复现

### L0-ET-01｜跑通 ET 轨，把两种 tag 策略的边界账算平

- **检验什么**：这条通过 = 你真的掌握了「**tag 粒度就是图分割决策**」——同一张图、同一组可委托算子，只因为 tag 打法不同，delegate 子图数和分区边界数就不一样
- **前置**：无
- **资源**：本地（模拟路径）/ 本地+工具链（真实路径，需 torch + executorch）
- **预计耗时**：1h

**任务**：跑 `scripts/run_executorch.sh`（**模拟路径也算通过**），然后**不看 `01_READING.md` 的表格**，只对着 `out/executorch/01_partition_compare.json` 回答三个问题：

1. `per_node` 与 `connected` 各自的 `delegate_subgraph_count` 和 `boundary_count` 是多少？
2. 模型是主角 `tiny_mlp` 叠两层、中间夹一个 `softmax`，算子序列 `addmm, relu, add, softmax, addmm, relu, add`，其中只有 `softmax` 不可委托。这两种策略的边界**分别落在哪两个节点之间**（读 `boundaries[].between`）？
3. 两者的 `boundary_count` 差值从何而来——是「不可委托算子把图切开」造成的，还是「tag 打法自己制造的」？把差值拆成这两部分。

答完再打开 `01_READING.md` 对答案。

**验收命令**：

```bash
cd onnx-delegate-lab
bash scripts/run_executorch.sh
python - <<'PY'
import json
d = json.load(open("out/executorch/01_partition_compare.json", encoding="utf-8"))
for mode, v in d["simulate"].items():
    print(f"{mode:10s} subgraphs={v['delegate_subgraph_count']}  boundaries={v['boundary_count']}")
    for b in v["boundaries"]:
        print("      ", b["between"], b["tags"])
print("real status:", d["real"].get("status"))
PY
```

**通过标准**：

- `out/executorch/01_partition_compare.json`、`01_READING.md`、`02_THREE_SYSTEMS.md` 三个文件存在
- 读出的数字是 `per_node: subgraphs=6, boundaries=6`、`connected: subgraphs=2, boundaries=2`
- 你能把 6 拆成 **2 + 4**：
  - `connected` 的那 2 条边界在 `add#2 → softmax#3` 和 `softmax#3 → addmm#4`，即**由不可委托算子造成，无法通过改 tag 消除**（**结构性边界**）
  - `per_node` 多出的 4 条（`addmm#0 → relu#1`、`relu#1 → add#2`、`addmm#4 → relu#5`、`relu#5 → add#6`）是**tag 打法自己制造的，纯属自伤**（**策略性边界**）
- 你能指出这 4 条自伤边界里最贵的是哪一类：`Gemm → Relu` 被切开，意味着 Relu 无法被吸进 Gemm 的输出循环，`h` 必须写回内存再读一遍——这个融合机会**下游谁都补不回来**（对照[链路总图](../learning-guides/00-end-to-end-pipeline.md)断链表第 ② 行）

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 脚本报「未找到 python」直接放弃 | `scripts/env.sh` 会尝试激活 conda 环境，路径是作者机器的；本机直接用 `python executorch_lab/01_partitioner_lab.py` 跑即可 |
| 看到 `real.status = no_executorch` 以为失败了 | 没读懂脚本设计：ET 是可选依赖，模拟路径是**预期行为**不是降级失败 |
| 说不清 6 条边界分别在哪 | 只看了汇总数字没看 `boundaries[].between`；边界是**边**不是**点**，说不出「哪两个节点之间」就等于没理解 |
| 认为 `connected` 的边界也能靠改 tag 消掉 | 没区分「结构性边界」（不可委托算子造成）与「策略性边界」（tag 粒度造成）——这是研究问题②的分水岭 |

---

### L0-ET-02｜判定你走的是真实路径还是模拟路径

- **检验什么**：这条通过 = 你清楚**每条路径各自能证明什么、不能证明什么**，后面做 L2 时不会拿模拟结果去冒充真实链路的结论
- **前置**：L0-ET-01
- **资源**：本地
- **预计耗时**：0.5h

**任务**：用下面的自查命令判定路径，然后填一张两列表：**「这条路径能验证的结论」/「这条路径无法验证、只能靠读文档补的结论」**。至少各写三条。

参考答案的方向（自己先写，再对）：模拟路径能验证 tag 策略 → 子图数 / 边界数的因果（因为这段逻辑是纯图论，与 ET 实现无关）；不能验证的是 Edge 算子的真实命名与 dtype 特化、`to_backend` 对非法 tag 的检查、`.pte` 里 `call_delegate` 的真实条数、blob 是否真的生成。

**验收命令**：

```bash
cd onnx-delegate-lab
# 1) 依赖探测（脚本自带的那套）
bash -c 'source scripts/env.sh'          # 看 [env] executorch=OK 还是 <optional>
# 2) 产物侧判定（更可靠：真实路径才会写 .pte）
python - <<'PY'
import json, pathlib
d = json.load(open("out/executorch/01_partition_compare.json", encoding="utf-8"))
status = d["real"].get("status")
print("JSON 自报路径:", status)
print("real 段内容:", json.dumps(d["real"], ensure_ascii=False)[:300])
ptes = sorted(pathlib.Path("out/executorch").glob("*.pte"))
print(".pte 产物:", [p.name for p in ptes] or "无（= 模拟路径）")
PY
```

**通过标准**：

- 能说出**三个互相独立的判定依据**：`env.sh` 的 `have_executorch` 探测、JSON 里 `real.status`（取值 `real_executorch` / `no_executorch` / `no_torch`）、`out/executorch/*.pte` 是否存在
- 知道 `real.status == "real_executorch"` **仍可能**每个 `runs[]` 里带 `error` 字段（导入成功但 API 不匹配，被 `try/except` 兜住了）——所以判定真实路径必须同时看 `runs[].error` 为空且 `.pte` 存在
- 两列表各写满三条

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 只看 `status` 就下结论 | 没注意 `try_real_executorch()` 里第二层 `try` 会把每个 mode 的异常记进 `runs[].error`；导入成功 ≠ 跑通 |
| 认为模拟路径「什么都证明不了」 | 边界账那部分是纯图论，模拟与真实同构；否定它等于没看懂 `simulate_partition` 在算什么 |
| 认为模拟路径「和真实一样」 | 反向错误：模拟不经过 Edge Dialect，也不会触发 `to_backend` 的任何合法性校验 |

---

## L1 改一处

### L1-ET-03｜改可委托算子集合，验证「边界数不是由委托算子数量决定的」

- **检验什么**：这条通过 = 你掌握了「**边界位置由未委托算子的分布决定，不由委托算子的多少决定**」，并且知道 `boundary_count` 单独看会骗人；附带一条：**用字符串判断算子身份有多脆**
- **前置**：L0-ET-01
- **资源**：本地
- **预计耗时**：1h

**任务**：改 `executorch_lab/01_partitioner_lab.py` 顶部的 `PORTABLE_HINTS`，做三组独立实验（每组做完还原）。

这一版 lab 里，模拟路径与真实路径**共用同一个 `is_delegatable(op)`**，它对 `PORTABLE_HINTS` 里的每个串做**子串匹配**：

```python
PORTABLE_HINTS = ("softmax",)                                  # ← 只改这一行

def is_delegatable(op: str) -> bool:
    return not any(h in op for h in PORTABLE_HINTS)
```

所以三组实验都只改这一行（真实路径匹配的是 `str(node.target)`，形如 `aten.addmm.default`，子串照样命中）：

- **A 组「缩小委托面」**：`PORTABLE_HINTS = ("softmax", "relu")`
- **B 组「取消所有边界」**：`PORTABLE_HINTS = ()`
- **C 组「看着只动了一个算子」**：`PORTABLE_HINTS = ("softmax", "add")`

每组都重跑并记录两种策略的 `delegate_subgraph_count`、`boundary_count`，以及**被 tag 的节点数**（`node_tags` 里 tag 非 `None` 的个数——这个指标必须自己数）。

**先预测再动手**：动手前把答案写下来——

1. A 组把 `relu` 变成 portable 之后，`connected` 的 `boundary_count` 会**变多、变少，还是不变**？想清楚：原来 `addmm,relu,add` 是一段连续可委托区，抽掉中间的 relu 之后，这一段还连着吗？
2. A 组里 `per_node` 和 `connected` 算出来的数字会不会**完全相同**？什么条件下这两种策略必然等价？
3. B 组把 `softmax` 也纳入委托后，`connected.boundary_count` 是多少？这个数字为什么必然是它——图上还剩几个「接缝」？
4. C 组你以为只把 `add` 排除了。**数一下实际被排除的算子有几个**（提示：`"add" in "addmm"`）。
5. 对比 A 组与 C 组的两个数字：哪一组 `boundary_count` 更小？哪一组 `tagged` 更大？如果只看 `boundary_count` 排名，会得出什么荒谬结论？

**验收命令**：

```bash
cd onnx-delegate-lab
# 改完 PORTABLE_HINTS 后每组各跑一次，并把结果另存以便对比
python executorch_lab/01_partitioner_lab.py
python - <<'PY'
import json
d = json.load(open("out/executorch/01_partition_compare.json", encoding="utf-8"))
for mode, v in d["simulate"].items():
    tagged = sum(1 for _, _, t in v["node_tags"] if t)
    print(f"{mode:10s} subgraphs={v['delegate_subgraph_count']} "
          f"boundaries={v['boundary_count']} tagged_nodes={tagged}")
# 哪些算子实际被判成了 portable —— C 组必须看这一行
print("portable:", [op for _, op, t in d["simulate"]["connected"]["node_tags"] if not t])
# 真实路径：tag_count 是【tag 个数】= 模拟侧的 delegate_subgraph_count，不是 tagged 节点数
for r in d["real"].get("runs", []):
    print(" real", r.get("mode"), "tag_count=", r.get("tag_count", r.get("error")))
PY
```

注意真实路径的 `tag_count` 取的是 `len(pr.partition_tags)`，对应模拟侧的 **`delegate_subgraph_count`**（不是 `tagged`）。两者对不上，说明 `PORTABLE_HINTS` 的子串在 Edge 算子全名上的命中情况和你在短名上的预期不同——这本身就是 C 组要教的东西。

**通过标准**：

四组数字（含基线）全部对上：

| 组 | `PORTABLE_HINTS` | 实际不可委托的算子 | `connected` 子图/边界 | `per_node` 子图/边界 | `tagged` |
|----|------------------|-------------------|----------------------|---------------------|----------|
| 基线 | `("softmax",)` | `softmax` | 2 / **2** | 6 / 6 | 6 |
| A | `("softmax","relu")` | `softmax`, `relu` ×2 | 4 / **6** | 4 / 6 | 4 |
| B | `()` | 无 | 1 / **0** | 7 / 6 | 7 |
| C | `("softmax","add")` | `softmax`, `add` ×2, **`addmm` ×2** | 2 / **4** | 2 / 4 | 2 |

- 你能用一句话解释这四行：**`connected` 的边界数 = 不可委托算子在链上切出的接缝数**，与被委托算子的多少无关。B 组一个接缝都没有所以是 0；A 组的 relu 卡在每段中间，把两段切成四段，接缝反而从 2 涨到 6
- A 组和 C 组你都能指出 `per_node` 与 `connected` **两列数字完全相同**，并说清条件：**没有任何两个可委托算子相邻**时，「连通分量」退化成「单节点」，两种策略必然等价
- C 组你能报出：`"add"` 这一个 hint 实际命中了 **4** 个算子（2 个 `add` + 2 个 `addmm`），不是你以为的 2 个；连同 `softmax`，全图 7 个算子里有 **5** 个退回了 CPU。原因是 `is_delegatable` 做的是子串匹配，`"add" in "addmm"` 成立
  （自查：`print([op for _, op, t in ...["connected"]["node_tags"] if not t])`）
- **对比 A 与 C**：C 的 `boundary_count`（4）比 A（6）更小，但 C 只委托了 2 个算子而 A 委托了 4 个。只看 `boundary_count` 会得出「C 比 A 好」的结论，而 C 其实把七成算子都退回了 CPU
- 你能说出必须补的第二个指标：**被委托的节点数 / 覆盖率**

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 预测「A 组委托算子变少，边界也会变少」 | 把「委托面积」和「边界数」当成同向指标；实际 portable 算子插在段**中间**时会把一段劈成两段，面积减少而边界增加 |
| C 组只数出 2 个被排除的算子 | 没读 `is_delegatable` 的实现就下结论；`any(h in op ...)` 是子串匹配不是相等判断 |
| 认为 C 组的子串命中「是 lab 的 bug」 | 方向可以讨论，但要先想清楚：改成精确匹配后，真实路径上 `aten.add.Tensor` 这种带重载后缀的名字全都匹配不上了。**算子识别靠字符串本身就是脆的**，这正是 Edge Dialect 要收敛算子集合的动机（接 L2-ET-06） |
| 拿 `boundary_count` 单指标下「C 组最优」的结论 | 正中陷阱——这条题目的全部意义就在这里 |
| 改完只跑了模拟路径就收工（装了 ET 的情况下） | `PORTABLE_HINTS` 两条路径共用，但真实路径匹配的是 Edge 算子全名；**务必确认 `runs[].tag_count` 与模拟侧的 `tagged` 一致**，不一致说明子串在真实算子名上没按你以为的方式命中 |

---

### L1-ET-04｜改模型排布，量化「连通子图」相对「逐节点」的真实收益

- **检验什么**：这条通过 = 你理解了「connected 策略的收益**不是恒定的**，它完全取决于可委托算子在图上是不是扎堆」——也就是划分收益的**数据依赖性**
- **前置**：L1-ET-03
- **资源**：本地
- **预计耗时**：1h

**任务**：改模型结构，构造两种排布并各跑一次（基线是「两段各 3 个」）：

- **交替排布**：`addmm, softmax, relu, softmax, add, softmax, addmm`（可委托算子被 softmax 完全打散）
- **连续排布**：`addmm, relu, add, addmm, relu, add, softmax`（可委托算子全扎堆在前，softmax 挪到末尾）

这一组要改**两处**，缺一不可：

1. `ToyModel.forward_ops()` 的返回列表（模拟路径）
2. 真实路径 `TinyMlpx2.forward()` 里的运算顺序（把 `torch.softmax` 挪到对应位置）

`PORTABLE_HINTS` 保持基线的 `("softmax",)` 不变——**这一条实验的自变量是「排布」，不是「算子集合」**，两个变量一起动就说明不了问题。

定义**收益** `gain = per_node.boundary_count − connected.boundary_count`，对三种排布（基线 + 两种新排布）各算一个 `gain`。

**先预测再动手**：

1. 交替排布下，`connected` 策略还能把任何两个节点合进同一个 tag 吗？如果一个 tag 里只有一个节点，`connected` 和 `per_node` 有区别吗？
2. 连续排布下，`per_node` 会把那 6 个连续可委托算子切成几段？`connected` 呢？中间那 5 条「本可以不存在」的边界，物理上对应什么代价（回看学习指南第 5 章的边界四代价）？
3. 基线的 `gain` 是 4。你能不能在**不看结果**的前提下，只数「每段的长度」就把三种排布的 `gain` 全算出来？把你的公式写下来再跑。
4. 如果一个真实模型全是「conv-relu-conv-relu」这种可委托算子连片的结构，Partitioner 该优先优化哪一项——**扩大算子覆盖面**，还是**把已覆盖的算子合并成更少的段**？

**验收命令**：

```bash
cd onnx-delegate-lab
# 每改一种排布跑一次，把两次输出并排记下来
python executorch_lab/01_partitioner_lab.py
python - <<'PY'
import json
d = json.load(open("out/executorch/01_partition_compare.json", encoding="utf-8"))
s = d["simulate"]
print("ops:", [op for _, op, _ in s["per_node"]["node_tags"]])
pn, cc = s["per_node"]["boundary_count"], s["connected"]["boundary_count"]
print(f"per_node={pn}  connected={cc}  gain={pn - cc}")
PY
```

**通过标准**：

三组数字全部对上：

| 排布 | 可委托段长 | `per_node` 边界 | `connected` 边界 | `gain` | `connected` 子图数 |
|------|-----------|----------------|-----------------|--------|-------------------|
| 基线 `addmm,relu,add,softmax,addmm,relu,add` | 3 + 3 | 6 | 2 | **4** | 2 |
| 交替 | 1+1+1+1 | 6 | 6 | **0** | 4 |
| 连续 | 6 | 6 | 1 | **5** | 1 |

- 交替排布：`gain` 恰好为 **0**，`connected` 与 `per_node` 的每一个数字都相同
- 连续排布：`gain` 为 **5**，`connected` 的子图数降到 **1**
- **三种排布的 `gain` 严格递增（0 < 4 < 5），且三者的 `per_node.boundary_count` 全是 6**——`per_node` 的边界数只和可委托算子总数有关，与排布无关，这是对照组成立的前提
- 你能写出结论并**用它反推出上表的 gain 列**：`connected` 的收益 = `Σ(每个连续可委托段长度 − 1)`。段越长收益越大；段长全为 1 时收益为 0

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 三种排布 `gain` 算出来一样 | 排布没改对（可委托算子仍然扎堆或仍然全打散），先 `print` 一遍 `node_tags` 确认 |
| 只改了 `forward_ops()` 没改 `TinyMlpx2.forward()`，真实路径结果对不上模拟 | 没意识到这是**两套并行实现**：`PORTABLE_HINTS` 是共用的，但**模型结构不是**，排布必须两边同步改 |
| 顺手把 `PORTABLE_HINTS` 也改了 | 单次实验动了两个自变量，`gain` 的变化归因不到排布上 |
| 说不出收益公式 | 只做了三组数据点没有归纳；这条题的产物是**公式**不是三个数字 |
| 认为 `connected` 永远优于 `per_node` | 方向对但太粗——交替排布上两者**完全等价**（gain=0），「永远更优」和「有时白做」是两种理解深度 |

---

## L2 加组件（主判据）

### L2-ET-05｜写一个全新的 Partitioner：按段长阈值划分

- **检验什么**：这条通过 = 你掌握了「**Partitioner 是一段可以自由设计的策略代码**」——它不止能回答「这个算子我支不支持」，还能回答「这一段值不值得委托」，也就是把边界代价纳入决策
- **前置**：L1-ET-04
- **资源**：本地（模拟路径）/ 本地+工具链（真实路径统计 `call_delegate`）
- **预计耗时**：2.5h

**任务**：新增第三种策略 `threshold`，规则是：**只 delegate 连续长度 ≥ N 的可委托算子段（默认 N=3），长度不足的短段整段留给 CPU**。

必须实现**两套**（这是本条的硬性要求，模拟侧是不依赖 ET 的等价降级）：

1. **模拟侧**：给 `simulate_partition(ops, mode)` 加一个 `mode == "threshold"` 分支（连同一个 `min_len: int = 3` 参数）。实现方式建议：先扫一遍求出所有连续可委托段的 `(起点, 长度)`，再只给长度达标的段打 tag。**不要改 `boundaries` 的计算逻辑**——边界统计是公共判据，改了就没法和前两种策略比。在 `main()` 里加一次 `simulate_partition(ops, "threshold")` 并写进 `01_partition_compare.json`。
2. **真实侧**：给 `TinyMlpPartitioner.__init__` 加参数（如 `mode: str` 或 `min_len: Optional[int]`），在 `partition()` 里实现同一套阈值逻辑。**注意 `partition()` 目前是单遍流式扫描，无法在遇到段首时就知道段长**——必须先把 `graph.nodes` 收集成 list 做两遍扫描。这正是这条题目的核心工程点。

基线模型的两段长度都是 3，所以 **N=3 时 `threshold` 与 `connected` 完全相同**，N=4 时则一个段都不留。这两个退化情形都要跑一次并记录，但**它们都不足以证明你的实现是对的**——真正的判据是下面那条长短段共存的序列。

**必须自己构造一条长短段共存的序列**：在 `ToyModel.forward_ops()` 里改成

```
addmm, relu, add, softmax, addmm, softmax, addmm, relu, add
         └─ 段长 3 ─┘          └段长 1┘      └─ 段长 3 ─┘
```

中间那个孤零零的 `addmm` 就是要被阈值筛掉的短段。

**先预测再动手**：

1. 在基线模型上跑 N=4，`delegate_subgraph_count` 和 `boundary_count` 各是多少？`boundary_count = 0` 是不是意味着「零边界代价、最优划分」？（想清楚这个数字为什么会骗人）
2. 在基线模型上跑 N=3，`threshold` 和 `connected` 的结果**完全相同**。为什么？这说明「结果相同」不能用来证明实现正确——你还需要什么样的测试用例？
3. 在上面那条长短段序列上跑 N=3，`threshold` 与 `connected` 的 `delegate_subgraph_count` / `boundary_count` / `tagged` 三个数字**分别**会怎么变？三个都会变小吗？
4. 那个被筛掉的短段（单个 `addmm`）退回 CPU 之后，边界数是变多还是变少？画一下它前后两条边：原来是「delegate → delegate」两条边，现在是什么？
5. 真实路径下，一个 delegate 段在 Edge 图里会变成几个 `call_delegate`？如果 `threshold` 比 `connected` 少委托了一段，`.pte` 里 `call_delegate` 的数量会怎么变？

**验收命令**：

```bash
cd onnx-delegate-lab
python executorch_lab/01_partitioner_lab.py
# 三策略横向对比（threshold 段落需你自己加进 JSON）
python - <<'PY'
import json
d = json.load(open("out/executorch/01_partition_compare.json", encoding="utf-8"))
s = d["simulate"]
print(f"{'mode':12s} {'subgraphs':>9s} {'boundaries':>10s} {'tagged':>7s}")
for mode in ("per_node", "connected", "threshold"):
    v = s.get(mode)
    if v is None:
        print(f"{mode:12s}  <未实现>"); continue
    tagged = sum(1 for _, _, t in v["node_tags"] if t)
    print(f"{mode:12s} {v['delegate_subgraph_count']:9d} "
          f"{v['boundary_count']:10d} {tagged:7d}")
PY

# 真实路径追加：统计 .pte 里的 delegate 痕迹（有 .pte 才做）
python - <<'PY'
import pathlib
for p in sorted(pathlib.Path("out/executorch").glob("*.pte")):
    b = p.read_bytes()
    print(p.name, "size=", len(b),
          "backend_id 出现次数=", b.count(b"BackendWithCompilerDemo"))
PY
```

真实路径还可以直接读 JSON 里 `real.runs[].delegate_markers_approx` 字段（脚本已在 `edge2.exported_program().graph` 的文本里数过 `call_delegate` / `lowered_module`），以及 `runs[].tag_count`。**这两个字段的语义以本地 ExecuTorch 版本打印出的图文本为准**——不同版本 `call_delegate` 的打印形态不一样，数不准时以 `tag_count` 为主、文本计数为辅。

**通过标准**：

- `01_partition_compare.json` 的 `simulate` 段里出现第三个 key `threshold`，字段结构与前两个完全一致
- 基线模型 + N=3：三个数字与 `connected` **逐一相同**（2 / 2 / 6），且你知道这不构成正确性证据
- 基线模型 + N=4：`delegate_subgraph_count == 0`、`boundary_count == 0`、`tagged == 0`；且你能解释这个 0 是「全部退回 CPU」而不是「完美划分」
- 在长短段共存序列 `addmm,relu,add,softmax,addmm,softmax,addmm,relu,add` 上跑 N=3，三个数字**全部严格小于** `connected`：

  | 策略 | 子图数 | 边界数 | tagged |
  |------|--------|--------|--------|
  | `connected` | 3 | 4 | 7 |
  | `threshold` (N=3) | **2** | **2** | **6** |

  边界数从 4 掉到 2，是因为那个孤立的 `addmm` 退回 CPU 后，它两侧的 `softmax` 与它同为「无 tag」，两条边界**合并消失**了
- 真实路径（若可用）：`threshold` 模式下 `.pte` 仍能生成，且 delegate 痕迹计数不高于 `connected` 模式
- 模拟路径（降级）：上述前四条全部满足即算通过，在笔记里注明「真实侧未验证」

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| `threshold` 与 `connected` 结果永远相同 | 测试序列里没有短段，实验设计失败；或者阈值逻辑写成了「段内逐节点判断」而不是「整段判断」 |
| 把 `boundary_count == 0` 当成最优 | 没建立「边界数必须和覆盖率一起看」的双指标意识（L1-ET-03 已经埋过一次） |
| 真实侧 `partition()` 里想单遍流式实现阈值 | 没意识到「段长」是段扫完才知道的信息；流式扫描做不了回溯打标，必须两遍 |
| 真实侧改完 `to_backend` 报 tag 不一致 | `partition_tags` 里登记了某个 tag，但没有任何节点的 `node.meta["delegation_tag"]` 用它（短段被跳过时忘了同步删除登记）——**这就是 Partitioner 契约的一致性要求** |
| 只实现了真实侧，装不上 ET 就交白卷 | 违反本条硬性要求：模拟侧的等价实现才是这条题的保底判据 |

---

### L2-ET-06｜对比 `to_edge` 前后：ATen Dialect vs Edge Dialect

- **检验什么**：这条通过 = 你理解了「**Edge 这一层不是多余的转译，它是委托契约的地基**」——没有 dtype 特化和算子命名空间收敛，Partitioner 根本没法可靠地判断「这个算子我支不支持」
- **前置**：L0-ET-02
- **资源**：本地+工具链（**本条需要真实 torch + executorch**）
- **预计耗时**：1.5h

**⚠ 真实环境要求**：这是本册**唯一无法用 lab 模拟路径等价替代**的 L2 条目——`simulate_partition` 用的是字符串 op 名，压根不经过 dialect 转换。降级方案见下方，但降级版**只算补课，不算 L2 通过**，请在笔记里如实标注。

**先看 lab 已经给你的东西**：真实路径跑通后，`01_partitioner_lab.py` 会落两份文本 dump——

```bash
cd onnx-delegate-lab
diff out/executorch/01_aten.graph.txt out/executorch/01_edge.graph.txt
```

重点看 `self.fc1(x)` 这一句在两层各变成了什么：`nn.Linear` 在 Edge 层会被拆成 `permute_copy`（或 `t_copy`）+ `addmm`。**这条 diff 就是 L1-ET-03 里「为什么 Partitioner 只能用排除法」的证据**：一个 `Linear` 落到几个 edge op、叫什么名字，是随版本变的。

**但文本 dump 只能看到名字**，看不到 `node.target` 的 Python 对象类型——那才是 ATen 与 Edge 的真正分界。所以本条还要写一个独立小脚本（建议 `executorch_lab/03_dialect_diff.py`，或直接用下面的 heredoc），**找出至少两处差异**并解释每一处存在的理由。

至少要覆盖这两个方向：

1. **算子命名空间**：ATen 侧是 `torch.ops.aten.add.Tensor`（打印为 `aten.add.Tensor`），Edge 侧是 `executorch.exir.dialects.edge...`（打印为 `aten.add.Tensor` 但 `node.target` 的 Python 对象类型不同）。**光看打印文本可能看不出差别，必须比较 `type(node.target)` 与 `node.target.__module__`。**
2. **dtype / shape 是否显式化**：看 `node.meta["val"]` 里的 `FakeTensor` 是否带完整的 dtype 与 shape，以及 Edge 侧是否消除了 Scalar 重载（学习指南 §3.3）。

**先预测再动手**：

1. 如果 Partitioner 拿到的是 ATen 图（有 Scalar 重载、dtype 可能未特化），你写 `if node.target == aten.add.Tensor` 这种判断会漏掉什么情况？后端拿到这个节点时还需要额外做什么才能生成代码？
2. Edge Dialect 声称「仍然硬件无关」。既然硬件无关，为什么不干脆在 ATen 层做委托？（提示：ATen 算子有两千多个且带各种重载，Edge 是收敛过的子集）
3. 你预计 `node.meta` 里除了 `val`，还会有哪些键是 Partitioner 依赖的？（回想 `TinyMlpPartitioner` 往里写了什么）

**验收命令**：

```bash
cd onnx-delegate-lab
python - <<'PY'
import torch
from torch.export import export

# 与 executorch_lab/01_partitioner_lab.py 的 TinyMlpx2 同构（主角叠两层夹 softmax）
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = torch.nn.Linear(3, 4), torch.nn.Linear(4, 4)
        self.register_buffer("bias2", torch.full((4,), 1.0))
    def forward(self, x):
        h = torch.relu(self.fc1(x)) + self.bias2
        p = torch.softmax(h, dim=-1)
        return torch.relu(self.fc2(p)) + self.bias2

m, inputs = M().eval(), (torch.tensor([[1.0, 2.0, 3.0]]),)
ep = export(m, inputs)

def dump(tag, gm):
    print("=" * 20, tag, "=" * 20)
    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue
        val = n.meta.get("val")
        print(f"  target={n.target}  type={type(n.target).__name__}  "
              f"module={getattr(n.target, '__module__', '?')}")
        print(f"    meta.val: dtype={getattr(val,'dtype',None)} shape={getattr(val,'shape',None)}")
        print(f"    meta keys: {sorted(n.meta.keys())}")

dump("ATen Dialect (torch.export)", ep.graph_module)
try:
    from executorch.exir import to_edge          # 入口以本地版本为准
    edge = to_edge(ep)
    dump("Edge Dialect (to_edge)", edge.exported_program().graph_module)
except Exception as e:
    print("!! to_edge 不可用:", type(e).__name__, e)
    print("   本地 executorch 版本的入口可能已改名，去包里搜 'def to_edge'")
PY
```

**通过标准**：

- 两段输出都成功打印
- 你能列出**至少两处**具体差异，每处都指到字段级别（例如：`type(node.target)` 从 `OpOverload` 变成 `EdgeOpOverload`、`__module__` 从 `torch._ops` 变成 `executorch.exir.dialects.edge.*`、`meta` 键集合的增减、Scalar 重载被替换为 Tensor 重载）
- 每处差异配一句「为什么委托机制需要它」
- **额外一处必须来自 `diff 01_aten.graph.txt 01_edge.graph.txt`**：写出 `self.fc1(x)` 在两层分别对应哪几个节点，并解释这对 `PORTABLE_HINTS` 那种字符串匹配意味着什么风险（接 L1-ET-03 的 C 组）

**降级方案（无 torch / 无 executorch）**：读 [`../learning-guides/executorch-learning-guide.md`](../learning-guides/executorch-learning-guide.md) 第 3 章的 §3.2 与 §3.3，对着文中给出的两段 IR 做同样的差异清单，产物形式一致（至少两处差异 + 每处的理由）。**在笔记里标注「纸面完成，未跑通真实 IR」**，这条按补课计，不计入 L2 完成数。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| `to_edge` 报 `ImportError` 就卡住 | 没执行开工自查；ET 各版本入口位置改过多次，先去本地包里搜符号再说 |
| 两边打印的 `target` 字符串一样就断言「没区别」 | 只看了 `__str__`；Edge op 刻意保留了可读名，差别在**对象类型和所属模块**上 |
| 说不出 Edge 层对委托的作用 | 只把它当成一次格式转换；实际上「算子集合收敛 + dtype 显式」正是后端能声明「我支持哪些算子」的前提 |
| `torch.export` 报动态 shape 相关错误 | 与本条无关的环境问题，把模型改成最简单的两输入逐元素运算即可 |

---

### L2-ET-07｜故意制造非法划分，逼出 `to_backend` 的契约检查

- **检验什么**：这条通过 = 你真正理解「**Partitioner 契约不是风格建议，是可执行的约束**」——tag 必须切出一个**能被独立抽成子图**的区域，否则整个程序无法调度
- **前置**：L2-ET-05
- **资源**：本地（模拟路径实现检查器）/ 本地+工具链（真实路径观察报错）
- **预计耗时**：1.5h

**任务**：构造一个**非法 tag 分组**，让本该合法的划分变成环状依赖，观察 ExecuTorch 怎么拒绝它。

推荐做法（最容易复现）：在 `TinyMlpPartitioner.partition()` 里，把**第一段**（`addmm#0, relu#1, add#2`）和**第二段**（`addmm#4, relu#5, add#6`）打成**同一个 tag**，中间的 `softmax#3` 保持不打 tag。这样：

```
第一段 ─┬─► [同一个 tag，会被融成一个子图] ◄─┬─ 第二段
        └─► softmax#3 ────────────────────┘
```

子图（第一段）的输出流进 `softmax`，`softmax` 的输出又流回子图（第二段）——**子图与外部节点互为上下游，形成环**，无法拓扑排序。

注意这个非法分组恰好是 `connected` 策略「差一点」就会做出来的事：`connected` 在遇到 `softmax` 时执行了 `current = None`，正是这一行把两段隔开。**把那一行删掉就得到非法版本**——契约的边界就在这一行代码上。

**同样必须做模拟侧的等价降级**：在 `01_partitioner_lab.py` 里新增一个 `validate_tags(node_tags) -> list[str]` 函数，把 tag 分组按「同 tag 的节点集合 + 节点间的链式依赖」建图，检测**同一 tag 的节点之间是否夹着未打该 tag 的节点**，是则返回一条违规说明。用它在你的非法分组上跑出报错文本。

**先预测再动手**：

1. 这个错误会在什么时候被发现——`partition()` 返回时、`to_backend()` 里、`to_executorch()` 里，还是运行时？为什么？
2. 假如 ExecuTorch **不检查**，直接把两个 add 融成一个 blob，运行时会发生什么？（想清楚：`call_delegate` 是一次原子调用，它必须一次性拿齐所有输入）
3. 换一种非法方式：只 tag 一个节点，但**在 `partition_tags` 里登记两个 tag**（其中一个没有任何节点使用）。这会报错还是被静默忽略？两种非法的严重程度一样吗？

**验收命令**：

```bash
cd onnx-delegate-lab
# 改完 partition() 后跑真实路径（有 ET 才有报错可看）
python executorch_lab/01_partitioner_lab.py 2>&1 | tee /tmp/et_illegal.log
python - <<'PY'
import json
d = json.load(open("out/executorch/01_partition_compare.json", encoding="utf-8"))
for r in d["real"].get("runs", []):
    print(r.get("mode"), "->", r.get("error", "<无报错>"))
PY
# 模拟侧：你的 validate_tags 应当独立报出同一个问题
python -c "
import sys; sys.path.insert(0,'executorch_lab')
import importlib.util as u
spec = u.spec_from_file_location('lab','executorch_lab/01_partitioner_lab.py')
lab = u.module_from_spec(spec); spec.loader.exec_module(lab)
# 非法：t0 的两个节点中间夹着无 tag 的 softmax → 应报违规
print('illegal:', lab.validate_tags([(0,'addmm','t0'),(1,'relu','t0'),(2,'add','t0'),
                                     (3,'softmax',None),
                                     (4,'addmm','t0'),(5,'relu','t0'),(6,'add','t0')]))
# 合法：同一份图按 connected 打法 → 应返回空列表（不能误报）
print('legal  :', lab.validate_tags(lab.simulate_partition(
    lab.ToyModel().forward_ops(), 'connected')['node_tags']))
"
```

注意 `01_partitioner_lab.py` 的 `try_real_executorch()` 用 `try/except` 兜住了每个 mode 的异常并写进 `runs[].error`，**报错不会让脚本崩溃**——所以要去 JSON 里捞，别只看终端。

**通过标准**：

- 真实路径：`runs[].error` 里出现一条与「循环 / 依赖 / 分区不合法」相关的异常（**具体异常类型与文案以本地 ExecuTorch 版本为准**，常见于图拆分或拓扑排序阶段；把原文抄进笔记）
- 模拟路径：`validate_tags` 对非法分组返回非空违规列表，对 L0 基线的 `per_node` / `connected` 分组返回空列表（**不能误报**）
- 你能用一句话说清违反了契约的哪一条：**同一个 tag 的节点集合，在图上必须构成一个「凸」区域**（任意两个同 tag 节点之间的所有路径都不能穿出该区域）

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 改完什么错都没报，`.pte` 照常生成 | tag 没真的打成同一个（检查 `node.meta["delegation_tag"]` 实际值），或本地版本把不连通组自动拆开了——如果是后者，去看它拆成了几个子图，结论同样成立 |
| 报错了但说不出错在哪 | 只复制了 traceback；这条题的产物是**用自己的话讲清环怎么形成的** |
| `validate_tags` 对合法分组也报错 | 判定条件写成了「同 tag 节点必须索引连续」——对链式模型碰巧成立，但那不是契约的本质（本质是凸性 / 无穿出路径） |
| 认为「不连通的 tag 只是效率差一点」 | 没理解 `call_delegate` 是原子调用；这不是效率问题，是**不可调度** |

---

### L2-ET-08｜三方对照：ET Partitioner / ORT EP / IREE dispatch

- **检验什么**：这条通过 = 你能把三套系统的划分机制**归约到同一组维度**上，而不是记住三段各自的名词——这是研究问题①②能不能跨栈成立的检验
- **前置**：L2-ET-05、L2-ET-07
- **资源**：本地
- **预计耗时**：1.5h

**任务**：分两步，**顺序不能反**。

1. **先自己闭卷写一遍**。新建一份自己的对照表（写在 `docs/notes/` 下你的实验记录里），行是三套系统，列**至少**覆盖这四个维度：
   - **划分谁做**：是用户写的回调 / 厂商插件，还是编译器 pass 自己决定？
   - **决策时刻**：导出期 / Session 构建期 / 编译相位？
   - **边界代价体现在哪**：具体是什么运行时开销，落在哪个对象上（`call_delegate` 调用、EP 之间的 tensor 拷贝、HAL buffer 与 dispatch 同步）？
   - **划分失败时怎么表现**：报错、静默回退到 CPU，还是根本不算失败？
   
   这一列最容易漏，重点想：ORT 的 EP 声明了能力但编译失败会怎样？ET 的非法 tag 会怎样（你在 L2-ET-07 已经亲手试过）？IREE 的 dispatch 形成失败又是什么后果？
   
2. **再打开 `out/executorch/02_THREE_SYSTEMS.md` 对照**，列出**你漏掉的维度**和**你写错的格子**。lab 那张表有八行（输入 IR / 谁声明能力 / 划分结果 / 编译子图 / 运行时派发 / 未覆盖算子 / 决策时刻 / 更像什么），你自己那张至少要能覆盖其中五行的内容。

**先预测再动手**：

1. 三者里哪一个的划分决策**最难改**（改一次要付出最大代价）？哪一个最容易在运行时动态调整？
2. 「未覆盖的算子怎么办」这一行，三者的答案是同一类机制吗？ET 的 portable kernel、ORT 的 EP 回退、IREE 的 device variant，本质上是同一件事的三种实现，还是三件不同的事？
3. 你在 L0-ET-01 里区分过「结构性边界」与「策略性边界」。这个区分在 ORT 和 IREE 上还成立吗？如果成立，各自对应什么？

**验收命令**：

```bash
cd onnx-delegate-lab
python executorch_lab/02_compare_three_systems.py     # 确保对照表已落盘
# 闭卷写完之后再看这个文件
cat out/executorch/02_THREE_SYSTEMS.md
```

延伸阅读：[`./05-onnx-ort.md`](./05-onnx-ort.md)（ORT EP `GetCapability` 与分区边界实验）、[`./03-iree.md`](./03-iree.md)（`--compile-to=flow` 看 dispatch region 形成）。**做过那两册的对应条目再做本条，收益翻倍**。

**通过标准**：

- 你的表先于 lab 的表写完（时间戳或 git 提交顺序可自证）
- 四个维度全部填满，三套系统一个不缺
- 与 lab 表对照后，明确列出「我漏了 N 项 / 我写错了 M 处」，并逐条说明漏掉的原因
- 「划分失败时怎么表现」这一行你能给出三个**不同**的答案，且 ET 那一格能引用 L2-ET-07 的真实报错

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 先看了 lab 的表再写 | 这条题检验的是**你脑子里有没有这张表**，抄一遍等于零 |
| 三行「边界代价」写得一模一样 | 停留在「都要拷贝」的抽象层；要说到具体载体（`call_delegate` 的输入打包 / EP 边界的 tensor 转换 / HAL buffer 与 barrier） |
| 「划分失败」三行都写「报错」 | 没意识到 ORT 的常态是**静默回退**，这与 ET 的硬报错是完全不同的工程后果——调 ORT 时性能莫名其妙掉一半，根因常在这里 |
| 认为 IREE 与另两者「不是一类东西」 | 方向有道理（IREE 的划分是编译器自己做的，不是厂商回调），但要把这个差异写成表格里的一格，而不是拒绝比较 |

---

## L3 打通

### L3-ET-09｜把 `.pte` 真正跑起来并验证数值

- **检验什么**：这条通过 = 你走完了「AOT 编译 → 运行时加载 → backend 注册 → `call_delegate` 派发 → 数值正确」的全链路，确认前面所有的 tag 实验**真的产出了能执行的东西**
- **前置**：L2-ET-05（且必须走通过真实路径，有 `.pte` 产物）
- **资源**：本地+工具链（**要求最高的一条**：需要 ExecuTorch C++ runtime 构建环境，CMake + 编译器工具链；仍**不需要 GPU**）
- **预计耗时**：3h（降级版 1h）

**⚠ 环境警告**：ExecuTorch runtime 需要从源码构建（`cmake` + 拉子模块），构建时间以小时计，且与 AOT 侧的 ET 版本必须一致。**这条是加深项，不在入门线内**，做不了直接走降级方案。

**任务（完整版）**：构建 ExecuTorch runtime，用它加载 L0/L2 产出的 `.pte`，喂同样的输入，与 PyTorch eager 的输出做数值对比。

需要注意：`01_partitioner_lab.py` 用的 backend id 是 `BackendWithCompilerDemo`，这是 ExecuTorch 自带的**示例 backend**，runtime 侧要确保它被注册进去（相关 CMake 选项与 target 名**以本地版本为准**，去本地 ET 源码里搜 `BackendWithCompilerDemo`）。若示例 backend 在你的版本里不可用，改用 XNNPACK backend 重跑一遍 AOT（Partitioner 换成 ET 自带的 XNNPACK partitioner），再跑 runtime。

**先预测再动手**：

1. 同一个模型，`per_node` 与 `connected` 两份 `.pte` 的**数值结果**应该相同还是不同？如果不同，说明什么严重问题？
2. 两份 `.pte` 的**文件体积**呢？哪一份更大，为什么（提示：每个 tag 会走一次 `preprocess`，产出一个独立 blob，blob 有固定头部开销）？
3. runtime 加载时如果找不到 `BackendWithCompilerDemo` 的注册，会在什么阶段失败——加载 `.pte` 时，还是第一次执行到 `call_delegate` 时？

**验收命令（完整版，路径以本地 ET 源码树为准）**：

```bash
# 1) 构建 runtime（选项名与 target 名以本地版本为准，先看 ET 源码的 CMakeLists）
#    典型形态：cmake -B cmake-out -DEXECUTORCH_BUILD_<...>=ON && cmake --build cmake-out
# 2) 用自带的 executor_runner 加载 .pte
#    ./cmake-out/executor_runner --model_path <repo>/onnx-delegate-lab/out/executorch/01_tiny_mlp_connected.pte
# 3) 与 eager 结果对比（AOT 侧参考值）
cd onnx-delegate-lab
python - <<'PY'
import json
# lab 已经把第一段的手算参考值写进了 JSON，先对这一段
d = json.load(open("out/executorch/01_partition_compare.json", encoding="utf-8"))
print("stage1 实际:", d["real"].get("stage1_output"))
print("stage1 手算:", d["real"].get("stage1_expected"))   # [[2.5, 3.5, 4.5, 7.5]]
PY
python - <<'PY'
import torch
# 全图参考值：与 01_partitioner_lab.py 的 TinyMlpx2 同一组权重
W1 = torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]])
x, bias2 = torch.tensor([[1., 2., 3.]]), torch.full((4,), 1.0)
h = torch.relu(x @ W1.T + 0.5) + bias2          # 第一份 tiny_mlp
p = torch.softmax(h, dim=-1)                     # portable 边界
ref = torch.relu(p @ torch.eye(4).T) + bias2     # 第二份 tiny_mlp（fc2 = I，b = 0）
print("input:", x)
print("stage1:", h)          # 应为 [[2.5, 3.5, 4.5, 7.5]]
print("eager ref:", ref)
PY
```

**这条题在本册里独一份的价值**：前面所有条目量的都是「编译期的数字」（子图数、边界数、tag 数），只有这一条回答「这些数字**变了之后，算出来的结果还一样吗**」。`stage1` 那个 `[2.5, 3.5, 4.5, 7.5]` 与 [`iree-lab`](../../iree-lab/)、[`tvm_lab/02`](../../tvm-fatbin-lab/tvm_lab/02_fusion_relay.py)、[`onnx_lab/01`](../../onnx-delegate-lab/onnx_lab/01_build_and_infer.py) 是同一组权重下的同一个答案——**四个系统、四套划分策略，必须给出同一个数**。

**降级方案（不构建 C++ runtime，只做 AOT 侧验证）** —— 这是推荐的默认路径：

```bash
cd onnx-delegate-lab
python - <<'PY'
import pathlib
d = pathlib.Path("out/executorch")
ptes = sorted(d.glob("*.pte"))
if not ptes:
    raise SystemExit("无 .pte —— 你走的是模拟路径，本条只能读文档补课")
for p in ptes:
    b = p.read_bytes()
    print(f"{p.name:32s} size={len(b):>8d}  "
          f"backend_id={b.count(b'BackendWithCompilerDemo'):>3d}  "
          f"header={b[:8]!r}")
print("\n体积差：per_node 与 connected 的字节数之差 = "
      "每多一个 delegate 段所付的固定开销 × 段数差")
PY
```

降级版还可以再进一步：如果本地装了 ET 的 Python 序列化工具（模块路径**以本地版本为准**，可在包里搜 `_serialize` / `deserialize`），把 `.pte` 反序列化成 program 对象，直接数 `execution_plan` 里的 delegate 条目与 chain 指令数——比数字节串精确得多。

**通过标准**：

- **完整版**：runtime 成功执行，输出与 eager 参考值在 `1e-5` 内一致；`per_node` 与 `connected` 两份 `.pte` 的数值结果**必须相同**（划分策略不改变语义）
- **降级版**：两份 `.pte` 都存在且非空；能报出两者的**体积差**并解释这个差来自 blob 个数；能说出降级版**没有验证**的东西——运行时是否真的调用了 delegate、数值是否正确
- 无论哪一版，在笔记里写清「本条完成到哪一层」

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| runtime 加载报 backend 未注册 | 没理解 delegate blob 是**按 backend_id 查表**派发的；AOT 侧写进去的 id 和 runtime 侧注册的 id 必须字面一致 |
| 两份 `.pte` 数值结果不同 | 划分策略改变了语义 —— 这是**严重 bug**，回到 L2-ET-05 检查你的 Partitioner 是不是在 `partition()` 里动了图（违反契约第 2 条） |
| runtime 与 AOT 版本不一致导致 `.pte` schema 报错 | 没建立「`.pte` 是有版本的序列化格式」的概念 |
| 降级版做完就宣称「端到端打通了」 | AOT 侧生成成功 ≠ 能跑对；这两件事的距离正是这条 L3 的全部意义 |

---

## 条目 × 资源需求速查

| 编号 | 任务 | 级别 | 资源 | 需要真实 ET | 耗时 |
|------|------|------|------|------------|------|
| L0-ET-01 | 跑通 ET 轨，算平两种 tag 策略的边界账 | L0 | 本地 | 否（模拟即可） | 1h |
| L0-ET-02 | 判定真实路径 vs 模拟路径 | L0 | 本地 | 否 | 0.5h |
| L1-ET-03 | 改可委托算子集合，验证边界数的真正来源 | L1 | 本地 | 否（模拟即可） | 1h |
| L1-ET-04 | 改模型排布，量化 connected 的真实收益 | L1 | 本地 | 否（模拟即可） | 1h |
| **L2-ET-05** | **写新 Partitioner：按段长阈值划分** | L2 | 本地 / 本地+工具链 | 否（模拟侧为保底判据） | 2.5h |
| L2-ET-06 | ATen vs Edge Dialect 差异对比 | L2 | 本地+工具链 | **是**（降级只算补课） | 1.5h |
| **L2-ET-07** | **制造非法划分，逼出契约检查** | L2 | 本地 / 本地+工具链 | 否（模拟侧写检查器） | 1.5h |
| L2-ET-08 | 三方对照：ET / ORT EP / IREE dispatch | L2 | 本地 | 否 | 1.5h |
| L3-ET-09 | `.pte` 真正执行并验证数值 | L3 | 本地+工具链 | **是**（降级只做 AOT 侧） | 3h（降级 1h） |

**合计约 13.5 小时**；只做入门线（L0 两条 + L1 一条 + L2-ET-05 + L2-ET-07）约 6.5 小时；不做 L3 共约 10.5 小时。

**本册无任何条目需要 GPU 或集群资源。** 九条里**七条**在完全没有 ExecuTorch 的机器上可以按模拟路径完成；只有 L2-ET-06（Edge Dialect 对比）与 L3-ET-09（runtime 执行）依赖真实环境，且都给了降级说明。

**下一步**：本册与 [`./05-onnx-ort.md`](./05-onnx-ort.md)（ORT EP 分区）、[`./03-iree.md`](./03-iree.md)（IREE dispatch region）是同一个问题的三种答案。三册都做完，再回头重写一遍 L2-ET-08 的对照表——那一版才是你真正拥有的知识。
