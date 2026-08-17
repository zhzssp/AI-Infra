# iree-lab —— 用一个模型走完 IREE 的全部编译相位

不编译 IREE 源码、不需要 GPU。`pip install` 两个 wheel，就能把**同一个模型**
停在 `flow / stream / hal / vm` 每一个高度上，看它一层层长出什么。

## 在自学体系中的位置

| | |
|--|--|
| **角色** | IREE 的动手主战场：相位分解 · dispatch 划分 · 资源生命周期 · 单文件部署 |
| **配套教材** | [`../docs/learning-guides/iree-learning-guide.md`](../docs/learning-guides/iree-learning-guide.md)（机制主教材）· [`../docs/paper-notes/07-tinyiree.md`](../docs/paper-notes/07-tinyiree.md)（论文动机） |
| **链路总图** | [`../docs/learning-guides/00-end-to-end-pipeline.md`](../docs/learning-guides/00-end-to-end-pipeline.md) 站 ⑤ → ⑦ |
| **上一站** | [`../mlir-toy-dialect/examples/upstream/`](../mlir-toy-dialect/examples/upstream/) —— 手工跑 `linalg → bufferize → llvm`，看清每一步 |
| **下一站** | [`../tvm-fatbin-lab/`](../tvm-fatbin-lab/) —— 换一套编译器做同一件事，对照两种设计取舍 |

> 上一站是**手工挡**：自己调 `mlir-opt` 一个 pass 一个 pass 地降。
> 这一站是**自动挡**：工业编译器把那套流程封装成相位，代价是中间多了 flow/stream/hal 三层
> ——本项目就是把这三层重新拆开给你看。

---

## 一、主角就是全仓库那个 tiny_mlp

```
x:[2,3] ──Gemm(W:[4,3], transB)──▶ h:[2,4] ──Relu──▶ ──Add(bias)──▶ y:[2,4]
```

同一个模型，在五个 lab 里各有一副面孔（最后一行是**算子级主角**：同一张图的最内层）：

| 项目 | 它长这样 | 你在那里学到 |
|------|---------|-------------|
| `onnx-delegate-lab/onnx_lab/01` | ONNX GraphProto（`Gemm/Relu/Add` 三个节点） | 图的序列化形态、`transB` 这类属性 |
| `onnx-delegate-lab/executorch_lab/01` | `torch.nn.Module`，叠两层夹一个 softmax | 谁来划分、tag 粒度怎么变成边界数 |
| `mlir-toy-dialect/examples/upstream/01` | `linalg.generic` + affine_map | 循环语义如何被结构化地表达 |
| **`iree-lab/models/tiny_mlp.mlir`** | **同上，但要真编真跑** | **相位、dispatch、资源、部署** |
| `tvm-fatbin-lab/tvm_lab/02` | Relay `nn.dense/relu/add` | 另一套编译器的融合决策 |
| `llvm-hello-compile/src/kernel.c` | 摊成循环的 `axpy`（Gemm 最内层） | 标量层的 SSA、向量化、FMA |

**为什么这件事重要**：换成五个不相干的示例，你学到的是五段孤立知识；
换成同一个模型，你每读一份 IR 都是在给同一个对象叠加理解 ——
「哦，ONNX 里那个 `transB=1`，到 linalg 就是 `affine_map<(m,n,k)->(n,k)>`，
到 flow 就被吸进了一个 dispatch，到 hal 就变成一块 buffer 的读法」。

---

## 二、四个模型，各有分工

| 文件 | 规模 | 存在的理由 |
|------|------|-----------|
| `models/abs.mlir` | 1 个 op | **逐相位精读**。每一相 dump 出来都能整篇看完，适合回答「这一相新增了什么」 |
| `models/tiny_mlp.mlir` | 4 个 `linalg.generic` | **看融合与 dispatch 划分**。单个 op 演示不了「几个 op 合成一个 kernel」 |
| `models/tiny_mlp_dynamic.mlir` | 同上，batch 改 `?` | **对照组**。和上一个 diff，量出「静态形状」到底买到了什么 |

> `tiny_mlp.mlir` 文件末尾有**手算推导**，`run_execute.sh` 会自动跟实际输出比对。
> 这一步不是仪式感：前面所有相位都只是文本，只有它能证明那些变换没改语义。

---

## 三、三个脚本，三个问题

| 脚本 | 回答的问题 | 主要产物 |
|------|-----------|---------|
| `scripts/run_phases.sh` | 编译器**做了什么**？每一相位新增哪一类信息？ | `out/phases/*.mlir`、`out/PHASES.md` |
| `scripts/run_execute.sh` | 这些变换**没改语义**吗？部署产物长什么样？ | `out/execute/*.vmfb`、比对结果 |
| `scripts/run_variants.sh` | **我能改什么**？开关一拧，IR 和机器码怎么变？ | `out/variants/` |

### run_phases.sh：把抽象层变成可 diff 的文本

核心就一个标志：

```bash
iree-compile --iree-hal-target-device=local \
             --iree-hal-local-target-device-backends=llvm-cpu \
             --compile-to=flow models/tiny_mlp.mlir -o out/tiny_mlp.flow.mlir
```

`--compile-to=<phase>` 让编译器**停在半路**把 IR 吐出来。于是：

- **flow**：数一下 `flow.dispatch` 有几个。源码写了 4 个 `linalg.generic`，
  这里若少于 4 个 → **融合发生了**，Relu 和 Add 被吸进 Gemm 的 dispatch，中间张量不落 DRAM。
- **stream**：看 `!stream.resource<transient>` 有几个。flow 层只说了「算什么」，
  stream 层第一次说清**谁占内存、占多久、谁等谁**。
- **hal**：看 `hal.executable.variant`。一个 kernel 多份 variant，
  和 CUDA fatbin 里一个 kernel 多份 cubin 是**同一个问题的同构答案**。
- **vm**：宿主侧的「先分配、再 dispatch、再等 semaphore」这套逻辑本身被编成了字节码
  ——所以部署产物是单个 `.vmfb`，不用再链接一个调度器。

脚本还会 diff 相邻两相，直观展示「多出来的行全是内存与时序」。

### run_variants.sh：三组对照，各扣一条断链

| 实验 | 对照 | 对应 [`00-end-to-end-pipeline.md`](../docs/learning-guides/00-end-to-end-pipeline.md) |
|------|------|------|
| A | 静态 `[2,3]` vs 动态 `[?,3]` | 站 ①：形状丢了，后面全链路补不回来 |
| B | 默认 CPU vs `+avx2` vs `+avx512f` | 站 ⑥：向量宽度由目标特性钉死 |
| C | 挖出 kernel 的 `.ll` / `.s` | 站 ⑥⑦：IREE 的叶子就是 LLVM 的入口 |

实验 B 会打出这样一张表：

```
配置       <4 x float>    <8 x float>    <16 x float>   vmfb字节
baseline   ...            ...            ...            ...
avx2       ...            ...            ...            ...
avx512     ...            ...            ...            ...
```

**这张表就是链路总图那句优化目标「内层循环走 8 宽向量」的最终落点。**
关键在于它不是任何单个 pass 的功劳：linalg 层保住了并行语义、flow 层没把循环切碎、
LLVM 后端才有机会按 AVX 宽度打包。**缺一环就退回 `<4 x float>`。**

实验 C 挖出的 `.ll`，正是 [`llvm-learning-guide`](../docs/learning-guides/llvm-learning-guide.md)
第 5 章的输入 —— IREE 负责「切到什么粒度」，LLVM 负责「这一块怎么编」，两份指南在这个文件上握手。

---

## 四、一键启动

```bash
pip install -r requirements.txt      # iree-base-compiler + iree-base-runtime

bash scripts/run.sh 2>&1 | tee out/all.log
# 或分步：
bash scripts/run_phases.sh
bash scripts/run_execute.sh
bash scripts/run_variants.sh
bash scripts/clean.sh                # 清空 out/
```

不装依赖时脚本会打印安装提示并以退出码 2 跳过，不会报错中断。

`scripts/env.sh` 会**实际试一次**来确定当前版本用哪套目标标志
（新版 `--iree-hal-target-device=local` + `--iree-hal-local-target-device-backends=llvm-cpu`，
旧版 `--iree-hal-target-backends=llvm-cpu`），所以跨版本不用改脚本。

---

## 五、目录结构

```
iree-lab/
├── README.md
├── requirements.txt
├── models/
│   ├── abs.mlir                 # 1 op：逐相位精读
│   ├── tiny_mlp.mlir            # 主角：Gemm→Relu→Add（文末附手算答案）
│   └── tiny_mlp_dynamic.mlir    # 对照组：batch 改 ?
├── scripts/
│   ├── env.sh                   # 定位工具 + 探测标志写法
│   ├── run_phases.sh            # ① 相位分解
│   ├── run_execute.sh           # ② 编译执行与数值校验
│   ├── run_variants.sh          # ③ 三组对照实验
│   ├── run.sh                   # 一键跑完
│   └── clean.sh
└── out/                         # 产物（git 忽略）
    ├── PHASES.md                # 自动生成的相位速查表
    ├── phases/  execute/  variants/
```

---

## 六、过关标准

跑完之后，你应当能不看文档回答：

1. `flow` 相位固化了什么？为什么这个决定之后再也改不了？
2. `!stream.resource` 的 `external` / `transient` / `variable` 分别意味着什么约束？
3. 一个 kernel 为什么会有多个 `hal.executable.variant`？这和 fatbin 是什么关系？
4. `.vmfb` 里装了哪三样东西？为什么运行时不需要再带一个编译器？
5. 静态 shape 相比动态 shape，在 stream 层省掉了哪一类指令？为什么这类信息**丢了补不回来**？

答不上来的条目，回 [`iree-learning-guide.md`](../docs/learning-guides/iree-learning-guide.md)
对应章节 —— 那里的每个示例都指回本项目的具体产物文件。
