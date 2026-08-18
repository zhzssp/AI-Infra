# 端到端链路：一个模型、一次优化，穿过七个知识点

> **本文档的定位**
> - 其余七份 `*-learning-guide.md` 各自讲清**一个系统怎么工作**；这一份只回答一件事：**它们怎么接力完成同一次优化**。
> - 做法是全仓库共用一个**主角**：一个 tiny MLP 和它内部的一个算子（另有一级放大版留给 `dist-train-lab`，见 §1.3）。每份指南的「示例精讲」讲的都是**这同一个东西在本层长什么样**，而不是各讲各的玩具。
> - 主角不是新编的——它已经躺在 [`onnx-delegate-lab/onnx_lab/01_build_and_infer.py`](../../onnx-delegate-lab/onnx_lab/01_build_and_infer.py) 和 [`llvm-hello-compile/src/kernel.c`](../../llvm-hello-compile/src/kernel.c) 里了。本文只是把这条已经长了一半的线点明并接通。
>
> **怎么用这份文档**
> - **第一次读**：只看[第 1 章主角](#第-1-章-主角一个模型和它的内核)、[第 2 章七站总图](#第-2-章-七站总图)、[第 3 章那次优化](#第-3-章-一次优化的接力)。约 30 分钟，之后再读任何一份专题指南都有坐标。
> - **每读完一份专题指南**：回到[第 5 章断链表](#第-5-章-断链表哪一环断了会怎样)，确认自己能说出「这一层断了，下游会看到什么症状」。
> - **收尾**：按[第 6 章](#第-6-章-按链路顺序跑一遍)把五个编译 lab 按链路顺序跑一遍，产物对照着读；
>   之后再用 `dist-train-lab` 把「划分」从算子推到设备维度。

---

## 目录

- [第 1 章 主角：一个模型和它的内核](#第-1-章-主角一个模型和它的内核)
- [第 2 章 七站总图](#第-2-章-七站总图)
- [第 3 章 一次优化的接力](#第-3-章-一次优化的接力)
- [第 4 章 逐站详解](#第-4-章-逐站详解)
- [第 5 章 断链表：哪一环断了会怎样](#第-5-章-断链表哪一环断了会怎样)
- [第 6 章 按链路顺序跑一遍](#第-6-章-按链路顺序跑一遍)
- [第 7 章 自测](#第-7-章-自测)

---

## 第 1 章 主角：一个模型和它的内核

### 1.1 图级主角：`tiny_mlp`

```text
x:[N,3] ──Gemm(W:[4,3], b:[4], transB=1)──▶ h:[N,4] ──Relu──▶ h_act:[N,4] ──Add(bias2:[4])──▶ y:[N,4]
```

三个算子，刚好覆盖三类典型：**Gemm 是计算密集的 reduction**、**Relu 是纯逐元素（injective）**、**Add 是逐元素 + 广播**。后面讲融合、划分、边界代价时，都要靠这三类的区别。

权重故意用 `W:[4,3]` 而不是 `[3,4]`，因为 `[out_features, in_features]` 是三个系统的共同约定：

| 系统 | 写法 | 权重形状 |
|------|------|---------|
| PyTorch | `nn.Linear(3, 4).weight` | `[4, 3]` |
| ONNX | `Gemm(transB=1)` | `[4, 3]` |
| TVM Relay | `nn.dense(x, w)` | `[4, 3]` |

**同一份权重可以直接在三个系统之间搬**——这不是巧合，`transB` 这个属性存在的理由就是它。

主角就是 [`onnx-delegate-lab/onnx_lab/01_build_and_infer.py`](../../onnx-delegate-lab/onnx_lab/01_build_and_infer.py) 建的那张图，图名字面上就叫 `tiny_mlp`。

### 1.2 算子级主角：Gemm 的最内层

把 Gemm 摊开写成循环：

```c
for (int i = 0; i < N; ++i)          // batch
  for (int j = 0; j < 4; ++j) {      // 输出通道
    float acc = b[j];
    for (int k = 0; k < 3; ++k)      // 归约维
      acc += x[i][k] * W[j][k];      // ← 乘加
    h[i][j] = acc;
  }
```

最内层那句 `acc += x[i][k] * W[j][k]`，把它按 `j` 展开、`k` 固定，就是一句 **axpy**：`y[i] = a * x[i] + y[i]`。紧跟着的 Relu 就是 `v > 0 ? v : 0`。

这两样都在 [`llvm-hello-compile/src/kernel.c`](../../llvm-hello-compile/src/kernel.c) 里，而且是该文件存在的全部理由：

```c
void axpy(int n, float a, const float *restrict x, float *restrict y);  // ← Gemm 的最内层
float relu_sum(const Tensor *t);                                        // ← Relu + 归约
```

**所以 `kernel.c` 不是"一个随便的 C 文件"，它是 `tiny_mlp` 里那个 Gemm 降到 C 层之后的样子。** LLVM 指南里所有关于别名、FMA、向量化、寄存器分配的讨论，讨论的都是这张图里那一个算子。

### 1.3 为什么要分层主角

图级和算子级回答的是**两个不相干的问题**，混在一起讲是绝大多数材料读起来割裂的根因：

| | 图级（`tiny_mlp`） | 算子级（`axpy`） |
|--|-------------------|-----------------|
| 问的问题 | **谁来算、算几次、中间结果落不落 DRAM** | **这一次算，算得多快** |
| 决策单位 | 子图 / kernel 边界 | 循环序、向量宽度、寄存器 |
| 主要 lab | `onnx-delegate-lab`、`tvm-fatbin-lab` T02 | `llvm-hello-compile`、`tvm-fatbin-lab` T01 |
| 做错的后果 | 多几次 DRAM 往返（**几百倍**） | 少几倍 IPC（**几倍**） |

**量级差这么多，所以顺序不能反**：先把图级边界定对，再去抠算子级。第 3 章那次优化就是按这个顺序接力的。

> 全仓库其实有**三级**主角，上表只列了七站用得到的两级。第三级是**规模级 `FFNBlock`**
> （`dist-train-lab/model.py`，`tiny_mlp` 那根骨架放大到 GPU 尺寸），它问的是「装不下时切到多卡
> 要付多少通信代价」。这个问题**不在七站之内**，所以本章不展开——见[第 6 章](#第-6-章-按链路顺序跑一遍)末尾。

---

## 第 2 章 七站总图

```text
                        ┌────────────────────────────────────────────────┐
   tiny_mlp 的一生       │  每一站「固化」一批决定，下游再也改不回来        │
                        └────────────────────────────────────────────────┘

  ①  表示          Gemm/Relu/Add 三个节点 + 权重 + 动态 batch
      交换 IR      ──▶ 固化：算子集、shape 是静态还是符号、权重放哪
                        │                             onnx-delegate-lab O01
                        ▼
  ②  划分          谁来算哪一段（ORT EP / ET Partitioner / IREE dispatch）
      子图边界     ──▶ 固化：kernel 边界 → 边界上的四类代价
                        │                             onnx-delegate-lab O03 / E01
                        ▼
  ③  融合          同一后端内部，哪几个 op 合成一个 kernel
      执行规划     ──▶ 固化：中间张量落不落 DRAM
                        │                             tvm-fatbin-lab T02
                        ▼
  ④  调度          这一个 kernel 内部怎么算：循环序 / tile / 向量宽度
      算法≠调度    ──▶ 固化：访存模式、并行度、搜索空间
                        │                             tvm-fatbin-lab T01 / T04 / T05
                        ▼
  ⑤  多层降低      高层语义一层层掉到能表达移位、能表达 buffer 的层
      渐进 lowering ─▶ 固化：每一层能做什么优化、丢掉什么信息
                        │                             mlir-toy-dialect / iree-lab
                        ▼
  ⑥  指令生成      循环 → SSA → 别名分析 → 向量化 → 寄存器分配 → 机器码
      后端         ──▶ 固化：真实指令、真实寄存器
                        │                             llvm-hello-compile
                        ▼
  ⑦  打包与运行时  多架构实现打进一个产物；运行时选路 + 异步调度
      variant/HAL  ──▶ 固化：换硬件还跑不跑得起来、能不能重叠
                                                      tvm-fatbin-lab F01–F03 / iree-lab
```

> 站 ② 和站 ③ 在 IREE 里是同一相位（`flow`）做的两件事：先切 dispatch，再在
> dispatch 内部融合。`iree-lab/scripts/run_phases.sh` 数一下 `flow.dispatch` 的个数，
> 就能同时看到这两个决定的结果。

一句话记住整张图：**上面四站决定"算什么、算几次"，下面三站决定"算得多快、在哪能跑"。**

各站的锚点：

| 站 | 主问题 | 动手项目 | 主教材 | 横切概念 |
|----|--------|---------|--------|---------|
| ① 表示 | 模型怎么被写下来 | [`onnx-delegate-lab/`](../../onnx-delegate-lab/) O01–O02 | [onnx](./onnx-learning-guide.md) §2–4 | foundations §2 |
| ② 划分 | 谁来算哪一段 | [`onnx-delegate-lab/`](../../onnx-delegate-lab/) O03 / E01 | [onnx](./onnx-learning-guide.md) §7 · [executorch](./executorch-learning-guide.md) §4–7 | foundations §3.3 |
| ③ 融合 | 哪几个 op 一个 kernel | [`tvm-fatbin-lab/`](../../tvm-fatbin-lab/) T02 | [tvm](./tvm-learning-guide.md) §2.2 | foundations §3.2 |
| ④ 调度 | 这个 kernel 怎么算 | [`tvm-fatbin-lab/`](../../tvm-fatbin-lab/) T01/T04/T05 | [tvm](./tvm-learning-guide.md) §3–5 | foundations §4 |
| ⑤ 降低 | 语义怎么一层层掉 | [`mlir-toy-dialect/`](../../mlir-toy-dialect/)（手工挡）· [`iree-lab/`](../../iree-lab/)（自动挡） | [mlir](./mlir-learning-guide.md) §6–8 · [iree](./iree-learning-guide.md) §1–2 | foundations §6.2 |
| ⑥ 指令 | 循环怎么变机器码 | [`llvm-hello-compile/`](../../llvm-hello-compile/) | [llvm](./llvm-learning-guide.md) §2/§4/§5 | foundations §6.1 |
| ⑦ 打包 | 换硬件还跑不跑 | [`tvm-fatbin-lab/`](../../tvm-fatbin-lab/) F01–F03 · [`iree-lab/`](../../iree-lab/) · [`onnx-delegate-lab/`](../../onnx-delegate-lab/) E01 | [fatbin](./cuda-fatbin-learning-guide.md) · [iree](./iree-learning-guide.md) §3–4 · [executorch](./executorch-learning-guide.md) §2、§4.5 | foundations §7.3 · §8 |

> **站 ⑤ 为什么有两个项目**：`mlir-toy-dialect` 是手工挡 —— 你自己调 `mlir-opt`，
> 一个 pass 一个 pass 地降，每一步都看得见。`iree-lab` 是自动挡 —— 工业编译器把那套流程
> 封装成相位，代价是中间多了 `flow / stream / hal` 三层。**先手工后自动**，
> 否则那三层就只是三个名词。

---

## 第 3 章 一次优化的接力

到这里为止都还是地图。下面换成你要真正干的一件事。

### 3.1 目标

> **「让 `tiny_mlp` 的 Gemm→Relu→Add 只落一次 DRAM，并且最内层真的走 8 宽向量。」**

这句话里有两个诉求，分别属于图级和算子级，**它们各自需要一条独立的接力链，而且任何一环断掉，后面全部白做**。

### 3.2 诉求 A：只落一次 DRAM

先算清楚账。设 batch `N=1024`，中间张量 `h` / `h_act` 各是 `[1024,4]` float32 = 16 KiB。

| 方案 | kernel 数 | 中间张量落 DRAM | 全局访存 |
|------|----------|----------------|---------|
| 三个算子各自成 kernel | 3 | `h` 写+读、`h_act` 写+读 | 4 × 16 KiB |
| 三个融进一个 kernel | 1 | 都留在寄存器/cache | 0 |

要拿到第二行，需要**三个知识点依次成立**：

```text
② 划分：Gemm / Relu / Add 三个节点必须落在【同一个后端】
        └─ 若 Relu 被判给另一个 EP，中间张量被【强制物化】，融合窗口当场关闭
              ▼
③ 融合：同一后端内，FuseOps 判定 reduction(Gemm) + injective(Relu) + injective(Add) 可以合并
        └─ 产出一个带 Primitive=1 标记的内联函数 = 一个 kernel
              ▼
④ 调度：融合后的这一个 kernel 内部，Relu 与 Add 必须 compute_at 到 Gemm 的输出循环里
        └─ 否则融合只是"打了个包"，内部仍然先算完整个 h 再遍历一遍
```

**注意 ② 是 ③ 的前提，而 ② 属于另一个系统（ORT / ExecuTorch），甚至另一个团队。** 这就是为什么"划分"这个看起来很行政的决定，会直接决定你的 kernel 优化有没有意义——这一条在只读单份指南时几乎看不出来。

### 3.3 诉求 B：最内层走 8 宽向量

```text
④ 调度：schedule 里 vectorize 内层轴，声明"我要向量化"
              ▼
⑤ 降低：降到 llvm dialect 时，必须把"这两块 buffer 不重叠"翻译成 IR 上的 noalias
        └─ MLIR 侧信息丢在这一步，下游【无法补救】
              ▼
⑥ 指令：LLVM 拿到 noalias 才不插运行时指针检查；
        TTI 报告目标有 256-bit 向量寄存器，向量化器才敢选 8 宽而不是 4 宽
              ▼
⑦ 打包：换一张算力不同的卡还要能跑 → 多变体打包 + 运行时选择
```

第 ⑤→⑥ 这一跳最容易被忽略，也最能说明"知识点必须串起来"：

**`noalias` 不是 LLVM 自己推出来的，是上游 lowering 主动写上去的。** 上游不写，LLVM 就只能给出 MayAlias，向量化器要么插一堆运行时检查跑双版本循环，要么直接放弃。你在 ④ 精心调的 schedule，会在 ⑥ 悄无声息地退化——而且不报错。

亲眼看这一条（[`llvm-hello-compile/`](../../llvm-hello-compile/) 里 `axpy` 与 `axpy_may_alias` 就是为此准备的一对）：

```bash
cd llvm-hello-compile && bash scripts/tour.sh
# 第 9 站：aa-eval —— 有无 restrict 时别名分析的回答
# 第 11 站：axpy 的向量体 vs axpy_may_alias 里的 memcheck / found.conflict
```

两个函数**源码逻辑一模一样**，只差参数上的 `restrict`（降到 IR 就是 `noalias`），产出的机器码完全是两个东西。

### 3.4 两条链合起来

```text
                  诉求 A：少落 DRAM              诉求 B：内层向量化
                  ─────────────────              ──────────────────
  ② 划分          三个节点同后端  ────┐
  ③ 融合          合成一个 kernel  ───┤
  ④ 调度          compute_at 内联  ───┴──────▶  vectorize 内层轴  ──┐
  ⑤ 降低                                        传 noalias/align  ──┤
  ⑥ 指令                                        AA + TTI 决定宽度 ──┤
  ⑦ 打包                                        多变体，换卡能跑  ──┘

           量级：几百倍                          量级：几倍
```

**先做左边，再做右边。** 左边没做对时，右边调得再好也是在优化一个本来不该存在的 kernel。

---

## 第 4 章 逐站详解

每一站固定四问：**上游交给我什么 / 我固化什么 / 我交给下游什么 / 亲手看一次**。

### 站 ① 表示：模型怎么被写下来

| | |
|--|--|
| **上游给我** | 训练框架里的一个 `nn.Module` |
| **我固化** | 用哪个算子集（opset）、shape 是静态还是符号、权重进 `initializer` 还是进 `input` |
| **交给下游** | 一张有向无环图 + 一份权重 |
| **亲手看** | `cd onnx-delegate-lab && bash scripts/run_onnx.sh` → `out/onnx/01_READING.md` |

这一站最容易被当成"格式转换"而轻视。实际上它固化了两个后面改不动的决定：

1. **动态维一旦写成符号（`[None, 3]`），下游所有依赖具体尺寸的优化都要退化。** tile 因子、向量宽度、内存规划都想要编译期常量。见 [tvm §2.1](./tvm-learning-guide.md) 里 Relay 静态 shape 与 Relax 符号 shape 的对照。
2. **`initializer` 与 `graph.input` 的区别决定了什么能被常量折叠。** 权重进了 `initializer`，编译器才敢在编译期把它揉进 kernel。

深入：[onnx §2–4](./onnx-learning-guide.md)。

### 站 ② 划分：谁来算哪一段

| | |
|--|--|
| **上游给我** | 一张图 |
| **我固化** | 子图边界 —— 也就是下游能看到的**最大优化窗口** |
| **交给下游** | 若干段子图，每段归一个后端 |
| **亲手看** | `bash scripts/run_onnx.sh`（ORT EP 分区）与 `bash scripts/run_executorch.sh`（ET tag 粒度） |

三个系统在**不同时刻**做同一件事，这是整条链路上最值得横向对照的一点：

| 系统 | 决策时刻 | 机制 |
|------|---------|------|
| ONNX Runtime | **Session 构建期** | EP 轮流 `GetCapability` → `Compile` |
| ExecuTorch | **导出期** | Partitioner 给节点打 `delegation_tag` |
| IREE | **编译期** | 形成 `flow.dispatch` region |

边界不是免费的。四类代价（拷贝 / layout 转换 / 内存驻留 / 同步）见 [foundations §3.3](./ai-compiler-foundations-learning-guide.md)；在 lab 里数得出来的是**边界张量个数**和 **delegate 子图个数**：

```bash
# 主角叠两层、中间夹一个 portable 的 softmax，per_node vs connected 两种 tag 粒度
python onnx-delegate-lab/executorch_lab/01_partitioner_lab.py
#   per_node : 6 个 delegate 子图 / 6 条边界
#   connected: 2 个            / 2 条
# 差值 4 全部是「tag 打法自己制造的」——其中 Gemm→Relu 那一刀最贵，见断链表第 ② 行
```

深入：[onnx §7](./onnx-learning-guide.md)、[executorch §5–7](./executorch-learning-guide.md)。

### 站 ③ 融合：哪几个 op 合成一个 kernel

| | |
|--|--|
| **上游给我** | 归我这个后端的一段子图 |
| **我固化** | 中间张量落不落 DRAM |
| **交给下游** | 若干个"一个 kernel 该算什么"的定义 |
| **亲手看** | `cd tvm-fatbin-lab && python tvm_lab/02_fusion_relay.py` → 对比 `out/tvm/02_relay_before_fuse.ir.txt` 与 `after` |

融合的判据是**算子类别**，不是算子名字：

| 类别 | 例子 | 能不能当融合的头 |
|------|------|----------------|
| injective（逐元素） | `relu`、`add` | 能被吸进去 |
| reduction | `sum`、`dense` 的归约维 | 能当头，后面可以挂 injective |
| complex-out-fusable | `conv2d`、`dense` | 能当头 |
| opaque | `sort`、`argsort` | **切开**，两边不能跨 |

lab 里跑的就是主角本身：`nn.dense → add(b) → relu → add(bias2)`，形状 `x:[2,3] @ W:[4,3]`，
权重与 [`iree-lab/models/tiny_mlp.mlir`](../../iree-lab/models/tiny_mlp.mlir)、`onnx_lab/01` **逐位相同**。
四个 op 恰好是「一个 complex-out-fusable + 三个 injective」，按上表应当融成一个 `fn`。
后面再挂一个显式屏障（`stop_fusion`）加一个 `relu`——于是"能融的"和"被切开的"在同一份 IR 里并排出现。

> **这一站的数字可以横着比**：同一个主角，四个系统各数一个量——
> TVM 数 `FuseOps` 前后的 `fn` 个数、IREE 数 `flow.dispatch` 个数、
> ORT 数 optimized 图剩几个节点、ExecuTorch 数 delegate 子图数。
> 源码里写了 4 个算子，各自最后剩几个 kernel，差值就是各家融合能力的量化结果。

深入：[tvm §2.2](./tvm-learning-guide.md)、[foundations §3.2](./ai-compiler-foundations-learning-guide.md)。

### 站 ④ 调度：这一个 kernel 内部怎么算

| | |
|--|--|
| **上游给我** | 一个 kernel 该算什么（算法） |
| **我固化** | 循环序、tile、并行、向量宽度（调度） |
| **交给下游** | 一个具体的循环嵌套 |
| **亲手看** | `cd tvm-fatbin-lab && python tvm_lab/01_te_matmul_schedules.py` → `out/tvm/01_matmul_{naive,tiled}.lower.txt` 两份并排 |

这一站的全部内容浓缩成一句话：**算法说"算什么"，调度说"怎么算"，改调度不改结果。** lab 里 `N=64` 的 matmul，算法四行一个字不改，只换 schedule，`lower` 出来的循环嵌套完全不同。

需要注意它**向上依赖站 ③**：融合决定了 schedule 能覆盖多大范围。没融合时，Relu 是一个独立 kernel，你根本没有"把 Relu 塞进 Gemm 输出循环"这个选项。

深入：[tvm §3–5](./tvm-learning-guide.md)、[foundations §4](./ai-compiler-foundations-learning-guide.md)。

### 站 ⑤ 多层降低：语义怎么一层层掉

| | |
|--|--|
| **上游给我** | 一个带高层语义的循环嵌套 |
| **我固化** | 每一层能表达什么、因而能做什么优化 |
| **交给下游** | `llvm` dialect（外加一批必须主动带上的属性） |
| **亲手看** | 手工挡：`cd mlir-toy-dialect && bash scripts/all.sh` → 演示 5-a / 5-b<br>自动挡：`cd iree-lab && bash scripts/run_phases.sh` → `out/PHASES.md` |

toy 项目用一个刻意极小的例子讲清了整件事的核心：同一段 `x * 4`，在 `toy` 层跑化简**纹丝不动**，降到 `low` 层才能变成 `x << 2`。

**不是高层"没做"这个优化，是高层根本无法表达它**（`toy` 层没有"移位"这个概念）。这就是"多层"的全部理由，也是为什么不能一步降到 LLVM IR：每种优化都有它最合适的那一层，早降就丢信息，晚降就没法落地。

同一个道理在真实栈里的样子：`linalg` 层知道"这是一次矩阵乘"，降到 `scf` 就只剩循环，降到 `llvm` 就只剩指针算术。**依赖"这是一次卷积"的优化，必须在丢掉这个信息之前做完。**

这一站也是诉求 B 的断点所在：`noalias` / `align` 必须在这里主动写进 IR。

**同一件事的工业版本**：`iree-lab` 把主角模型停在 `flow / stream / hal / vm` 各相位上，
`run_phases.sh` 会打出每相位的行数。行数本身就是结论 —— 越往下走，同一份语义要写的字越多，
**多出来的字全是"原来隐含、现在必须说清"的东西**（谁分配、谁同步、谁等谁）。
toy 项目告诉你"为什么要分层"，iree-lab 告诉你"工业上分成了哪几层、各自补了什么"。

深入：[mlir §6–8](./mlir-learning-guide.md)、[iree §1–2](./iree-learning-guide.md)、[foundations §6.2](./ai-compiler-foundations-learning-guide.md)。

### 站 ⑥ 指令生成：循环怎么变机器码

| | |
|--|--|
| **上游给我** | LLVM IR（属性齐全或不齐全） |
| **我固化** | 真实指令、真实寄存器 |
| **交给下游** | 一段机器码 |
| **亲手看** | `cd llvm-hello-compile && bash scripts/tour.sh` → `out/tour/TOUR.md` |

这一站是主角算子 `axpy` 的主场。四件事决定它跑多快，且**前两件的输入全部来自上游**：

| 决定 | 依据 | 上游不给会怎样 |
|------|------|--------------|
| 能不能向量化 | 别名分析（`noalias`） | 只能 MayAlias → 运行时检查 + 双版本循环 |
| 向量选多宽 | TTI 的目标信息 | 按基线 128-bit 选 4 宽而不是 8 宽 |
| 乘加能不能合成 FMA | `contract` 标志 | 老老实实两条指令、两次舍入 |
| 寄存器够不够 | 活跃区间分析 | 循环体内插 spill/reload |

`tour.sh` 里第一行要看**两站**（第 9 站 `aa-eval` 给出别名结论，第 11 站才看到结论如何变成向量体或 memcheck），
后三行各对一站：第 12 站（TTI 宽度）、第 4 站（`contract` 与 FMA）、第 13 站（寄存器分配与 spill）。

深入：[llvm §2.6](./llvm-learning-guide.md)（属性）、§4.2（别名）、§4.5–4.6（向量化与 TTI）、§5（后端七阶段）。

### 站 ⑦ 打包与运行时：换硬件还跑不跑

| | |
|--|--|
| **上游给我** | 针对某个具体架构的机器码 |
| **我固化** | 一个产物里带几种实现、运行时怎么选、能不能重叠执行 |
| **交给下游** | 一个能分发的文件 |
| **亲手看** | CUDA 侧：`bash tvm-fatbin-lab/scripts/run_fatbin.sh`（需 nvcc）<br>IREE 侧：`cd iree-lab && bash scripts/run_execute.sh` → `out/execute/tiny_mlp.vmfb` |

同一个问题的两条实现路径，**结构完全同构**：

| | CUDA fatbin | IREE |
|--|------------|------|
| 容器 | `.fatbin` | `.vmfb` |
| 变体 | 每个 `sm_*` 一份 SASS | 每个 target 一个 `ExecutableVariant` |
| 退路 | 存一份 `compute_*` PTX，运行时 JIT | 多 target 编译时全带上 |
| 选择者 | CUDA driver 按设备算力 | HAL 按 device 能力 |

看到这个同构，"多后端"就从一个口号变成了一个有固定形状的工程问题：**编译期生成多份 → 打进一个产物 → 运行期按能力选路并保留退路**。

两边都能亲手数：`cuobjdump -lelf` 数 fatbin 里的 image，`grep hal.executable.variant`
数 IREE 里的变体。**同一个数字，两套系统。**

> **ExecuTorch 的 `.pte` 也是这个形状，但多份沿的是另一根轴**，别混：
> fatbin / vmfb 里的多份是**同一段代码的多个架构实现**（运行时挑**一个**执行）；
> `.pte` 里的多个 delegate blob 是**不同段代码各归一个后端**（运行时**依次全部**执行）。
> 前者是"选路"，后者是"分工"。真实的 `.pte` 当然可以两根轴同时用——
> 那时才是完整的站 ⑦。见 [executorch §4.5](./executorch-learning-guide.md)。

IREE 还多做了一件 fatbin 不做的事：把宿主侧的调度逻辑（分配 buffer、下发 dispatch、
等 semaphore）也编成了 VM 字节码打进同一个 `.vmfb`。所以它的部署产物是**单文件**，
运行时不需要再链接一个调度器 —— `run_execute.sh` 第 ③ 步换 `--device` 不改任何宿主代码，
就是这个设计的直接结果。

深入：[cuda-fatbin](./cuda-fatbin-learning-guide.md)、[iree §4.7](./iree-learning-guide.md#47-设备代码executable--variant--export)。

---

## 第 5 章 断链表：哪一环断了会怎样

**这张表是本文档的核心。** 知识点之间是否真的"协作"，唯一的检验方式就是：断掉一环，能不能说出下游的症状，以及在哪能亲眼看到。

| 断在哪 | 直接症状 | 下游能补救吗 | 在哪亲眼看到 |
|--------|---------|------------|------------|
| ① 把 batch 写成动态维 | tile 因子、向量宽度只能保守取 | 不能（要么重编、要么运行时特化） | `out/onnx/01_READING.md` 的 `infer_shapes` 结果<br>`iree-lab` 实验 A：静态 vs 动态两份 stream IR |
| ② 划分把 Relu 判给另一个后端 | 中间张量被强制物化，**融合窗口关闭** | **不能** | `out/onnx/03_*`、`out/executorch/01_*` 的边界张量计数 |
| ③ 没融合 | 3 个 kernel、2 张中间张量往返 DRAM | 不能（kernel 边界已定死） | `out/tvm/02_relay_before_fuse.ir.txt` vs `after`<br>`iree-lab`：数 `out/phases/tiny_mlp.flow.mlir` 里的 `flow.dispatch` |
| ④ 没 vectorize / 没 compute_at | 标量循环；融合只是打了个包 | 能，但要回到这一层重调 | `out/tvm/01_matmul_*.lower.txt` 两份并排 |
| ⑤ 降低时没传 `noalias` | LLVM 只能 MayAlias | **不能**（信息已丢失） | `tour.sh` 第 9 站 `aa-eval` 输出 |
| ⑥ TTI 没报告 AVX2 | 向量化器按 128-bit 选 4 宽 | 不能（中端唯一的目标信息入口） | `tour.sh` 第 12 站，默认 vs `+avx2` 两份<br>`iree-lab` 实验 B：`<4/8/16 x float>` 计数表 |
| ⑦ 只打了一个 `sm_*` | 换一张卡直接**加载失败** | 不能 | `cuobjdump -lelf` 的 image 列表<br>`iree-lab`：`grep hal.executable.variant` |

读法：**"不能补救"的那几行，就是这条链路上的单向阀。** 七站里占了六个，唯一标「能」的 ④ 还带着前提：
**回到这一层重调**，而不是下游替你补上。

所以这张表真正的结论是狠话一句：**没有任何一站的错误能靠下游消化掉，区别只在重做的代价。**
重调一次 schedule，与"重新导出模型 + 把后面六站全跑一遍"，完全不是一个量级——
这正是[第 1 章](#13-为什么要分层主角)那句"顺序不能反"的实际含义，也是要先把整条链看一遍再动手的原因。

> ④ 那一格值得多看一眼：它写的是「没 vectorize / 没 compute_at」，但这两半的下场并不一样。
> **没 vectorize**，下游 LLVM 的自动向量化器还有机会替你补一部分（前提是站 ⑤ 把 `noalias` 传下来了——
> 又绕回单向阀 ⑤）；**没 compute_at**，融合就只是打了个包，下游谁也补不回来。
> 一格里藏着两种命运，这本身就是"必须落到具体机制上、不能停在名词层面"的例子。

反过来，这张表也是**调试路径**：性能没达到预期时，从 ⑦ 往 ① 倒着查，第一个对不上的就是断点。

---

## 第 6 章 按链路顺序跑一遍

五个编译 lab 的推荐顺序**不是难度顺序，而是链路顺序**。每一步跑完先读它的报告，再往下走。
最后追加的 `dist-train-lab` 不在七站之内——它是站 ② 在**设备维度**上的展开，理由见本章末尾。

```bash
# 依赖一次装齐（仓库根目录，不需要 root）：bash setup.sh

# 站 ①②  表示与划分
cd onnx-delegate-lab && bash scripts/run.sh
#        → out/ANALYSIS.md，重点看 onnx/01_READING.md（结构层次）与 onnx/03（EP 分区边界）
#        → executorch/01_READING.md：同一张图换 ExecuTorch 划分，per_node=6 vs connected=2

# 站 ③④  融合与调度
cd ../tvm-fatbin-lab && bash scripts/run_tvm.sh
#        → out/tvm/02_relay_*fuse*.txt（融合前后）、01_matmul_*.lower.txt（两套 schedule）

# 站 ⑤    多层降低（手工挡：自己一个 pass 一个 pass 地降）
cd ../mlir-toy-dialect && bash scripts/all.sh && bash scripts/run_upstream.sh
#        → 演示 5-a / 5-b：同一段 x*4 在两层的不同命运
#        → examples/upstream：linalg → bufferize → memref → llvm 全程

# 站 ⑥    指令生成
cd ../llvm-hello-compile && bash scripts/run.sh && bash scripts/tour.sh
#        → out/ANALYSIS.md 建立链路感；out/tour/TOUR.md 逐个要点看

# 站 ⑤⑦   多层降低（自动挡）+ 打包与运行时
cd ../iree-lab && bash scripts/run.sh
#        → out/PHASES.md：每相位新增了什么信息
#        → out/execute/：.vmfb 跑出来的数值必须和手算一致
#        → out/variants/：静态 vs 动态、baseline vs avx2 vs avx512

# 站 ⑦    打包与运行时（CUDA 侧）
cd ../tvm-fatbin-lab && bash scripts/run_fatbin.sh     # 需要 nvcc
#        → out/cuda/dump_*.txt：cuobjdump -lelf 看到两个 sm_*；-lptx 非空

# 站 ②的设备维度展开（模型大到单卡装不下之后）
cd ../dist-train-lab && bash scripts/run_cpu_smoke.sh          # 无卡即可
#        → param_diff==0 / grad_diff<1e-6：all_reduce 的两条语义
#        → comm_*.csv：延迟/带宽拐点，即 DDP 梯度分桶的理由
bash scripts/run_8gpu_wall.sh                                  # 有多卡时
#        → out/table_scaling.md：DDP 扩展效率，缺口用通信带宽解释
#        → out/table_zero.md：ZeRO 阶梯的显存实测 vs 手算
```

> 顺序上有一处刻意的安排：**站 ⑥ 排在 iree-lab 之前**。
> `iree-lab` 实验 C 会挖出 kernel 的 `.ll` 和 `.s`，那时你需要已经认得
> `<8 x float>`、`vfmadd`、`vmaxps` 是什么 —— 这些是 `llvm-hello-compile` 教的。

> **`dist-train-lab` 为什么排在最后而不是站 ② 那里**：前面七站讲的是「一张图在**一个设备**上
> 怎么被编译」，划分的对象是**算子**；`dist-train-lab` 讲的是「一张图在**多个设备**上怎么被切」，
> 划分的对象是**张量与模型状态**。两者是同一个问题的两个维度——都是
> [第 2 章](#第-2-章-七站总图)说的「带通信代价的图分割」——
> 但设备维度多了一个前面完全没有的量：**卡间带宽**。先把单设备链路走通，
> 再引入这个量，才不会把「融合省下的访存」和「通信省下的传输」混成一笔账。
>
> 它的排期可以提前：**核心的一半在没有 GPU 的笔记本上就能做完**，见该 lab 的 README。

> Windows 下请在 **Git Bash / WSL** 里跑；缺依赖的轨会自行标记并跳过，不阻塞其它站。

跑完之后，把七份产物报告按上面的顺序摆在一起读一遍——**这是这套材料唯一能替代"在真实项目里干一年"的地方**。

---

## 第 7 章 自测

能连着答完下面七问，说明链路真的通了（而不是七个孤立知识点）：

1. `tiny_mlp` 的 `W` 为什么是 `[4,3]` 而不是 `[3,4]`？这个选择让哪三个系统能共用同一份权重？
2. `kernel.c` 里的 `axpy` 对应主角模型的**哪一部分**？为什么说它不是随便挑的一个 C 函数？
3. 站 ② 的划分决定，为什么会直接影响站 ③ 的融合**有没有机会发生**？
4. 站 ④ 精心调的 `vectorize`，可能在站 ⑥ 悄无声息地退化——是哪个信息在站 ⑤ 丢掉了？
5. 断链表里标"不能补救"的有几站？唯一标"能"的那一站，它的"能"附了什么前提？
6. 说出 CUDA fatbin 与 IREE `ExecutableVariant` 在**四个位置**上的一一对应。
7. 性能没达到预期时，为什么要**从 ⑦ 倒着往 ① 查**，而不是顺着查？

答不上来的那一问，回第 4 章对应站，或点进该站的主教材。

---

> **维护约定**：本文档是链路总纲，**不重复任何一份专题指南的内容**，只负责说清"接力关系"。
> 各专题指南开头的「本篇在链路中的位置」小节应与[第 2 章总图](#第-2-章-七站总图)保持一致；改动一处要同步另一处。
