# 检验体系 04｜TVM

> **对应学习文档**：[`../tvm-learning-guide.md`](../tvm-learning-guide.md)  
> **对应动手项目**：[`tvm-fatbin-lab/`](../../tvm-fatbin-lab/) 的 **TVM 轨**（`scripts/run_tvm.sh`）；CUDA fatbin 轨在 [`./07-cuda-fatbin.md`](./07-cuda-fatbin.md)  
> **分级与资源标签定义**：[`./README.md`](./README.md)

本册检验的是这些知识点能不能落成代码：**算法与调度分离、schedule 原语对循环结构的实际影响、Relay 四类算子的融合规则、layout 变换的语义保持与代价、AutoTVM 的搜索空间定义与 tuning log 复用、Relay pass 的顺序敏感性、CPU/GPU 两种 target 下 lower 结果的差别**。

学习指南 §8.4 的过关标准考的是「组会能不能讲清」（口述题）。本册全部换成动手题：改代码、跑命令、比产物、给出机器可判定的差异。

**入门线**：L0 两条全做 + L1 至少两条 + **L2-TVM-06 与 L2-TVM-07 必做**（自己写一个 TE 算子、自己写一个 AutoTVM 模板，这两条是「能不能自己加东西」的分水岭）。

**资源总览**：本册**除最后一条可选的 L3-TVM-11 外，全部是纯 CPU，不需要 GPU 卡**。lab 的六步一律 `target="llvm"` + `tvm.cpu(0)`。唯一环境要求是本地 Python 能 `import tvm`，即标签 `本地+工具链`。数值验证手段统一用 `np.allclose`（03/05/06 步里已有现成写法），本 lab 没有 pytest，不要去找测试框架。

```bash
# 开工前自查：版本号记下来，后面凡是「以本地版本为准」的地方都靠它
cd tvm-fatbin-lab
python -c "import tvm; print(tvm.__version__)"
python -c "import tvm; print(tvm.target.Target('llvm'))"
bash scripts/env.sh          # 打印 PYTHON / TVM / NVCC 的探测结果
```

> **版本漂移警告（全册通用）**：TVM 的 Relay 与 Relax、AutoTVM 与 MetaSchedule 在不同版本之间 API 变化很大（模块位置、函数签名、IR 打印格式都改过）。本册凡涉及不稳定 API 的地方都标了「以本地版本为准」并附了自查命令。遇到 `AttributeError` / `ModuleNotFoundError` 先用 `help()` 查本地签名，别怀疑思路。

---

## L0 复现

### L0-TVM-01｜跑通六步，把 lower IR 的循环对回 schedule 原语

- **检验什么**：这条通过 = 你真的掌握了「`te.compute` 只描述算什么，schedule 决定怎么算」，并能把 `tile` / `cache_write` / `compute_at` / `split` / `vectorize` 各自在 TIR 里留下的痕迹指出来
- **前置**：本地能 `import tvm`
- **资源**：本地+工具链
- **预计耗时**：1h

**任务**：跑 `bash scripts/run_tvm.sh`。然后**先不看 `out/tvm/01_READING.md`**，只对着两份 lower 文本，把 `tvm_lab/01_te_matmul_schedules.py` 里 `schedule_tiled()` 的五个调用（`s.cache_write` / `s[C].tile` / `s[CL].compute_at` / `s[CL].split` / `s[CL].vectorize`）逐个指到 IR 的具体行号上，写成五行「原语 → 行号 → 这行说明什么」。写完再打开 READING 对答案。

**验收命令**：

```bash
cd tvm-fatbin-lab
pip install -r requirements.txt
bash scripts/run_tvm.sh
# 重点读这三个产物
#   out/tvm/01_matmul_naive.lower.txt   两层空间循环 + 一层 reduce
#   out/tvm/01_matmul_tiled.lower.txt   分块后的多层嵌套 + 临时缓冲
#   out/tvm/01_READING.md               对答案用，自己写完再打开
wc -l out/tvm/01_matmul_naive.lower.txt out/tvm/01_matmul_tiled.lower.txt
ls out/tvm/
```

**通过标准**：

- `run_tvm.sh` 六步全部跑完，末尾打印 `[OK] 全部 TVM 步骤完成`
- `out/tvm/` 下同时存在 `01_matmul_naive.lower.txt`、`01_matmul_tiled.lower.txt`、`02_relay_before_fuse.ir.txt`、`02_relay_after_fuse.ir.txt`、`03_READING.md`、`05_autotvm.log`、`06_graph.json`
- tiled 版的循环嵌套层数**多于** naive 版，且你能指出 naive 里没有的那块临时缓冲（`cache_write` 的产物，名字通常形如 `C.global`，具体命名以本地版本为准）
- 你能说清：两份 IR 来自**同一个** `build_matmul()`，`te.compute` 一个字都没改

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 脚本直接 `[SKIP] TVM 不可用` 并以退出码 2 结束 | `import tvm` 不通；`scripts/env.sh` 走的是 `CONDA_HOME` / `TVM_ENV` 约定，要么装 TVM 要么把这两个变量指对 |
| 认为 tiled 版不同是因为「算法换了」 | 没建立算法 / 调度分离的心智模型——这是本册第一块基石，后面每条都靠它 |
| 找不到 `vectorize` 的痕迹 | 只看循环头没看循环体；向量化在 IR 里表现为最内层被折成向量宽度的访存与运算（`ramp` / `x8` 之类形式），具体写法以本地版本为准 |

---

### L0-TVM-02｜数出融合后的分组，把四类规则对回去

- **检验什么**：这条通过 = 你真的掌握了「injective / reduction / complex-out-fusable / opaque 四类算子 + 融合规则」，以及「融合 = 决定哪些节点落进同一个可执行单元」这个子图划分原型
- **前置**：L0-TVM-01
- **资源**：本地+工具链
- **预计耗时**：1h

**任务**：读 `out/tvm/02_relay_before_fuse.ir.txt` 与 `02_relay_after_fuse.ir.txt`。数出 after 里带 `Primitive=1` 属性的函数有几个、每个装了哪些 op。再对着 `tvm_lab/02_fusion_relay.py` 的建图段（`dense → add → relu → stop_fusion → relu`），逐个说清：`nn.dense` 属于哪一类、`add`/`relu` 属于哪一类、`stop_fusion` 在这里扮演什么角色，以及分组边界是被哪条规则划出来的。

**验收命令**：

```bash
cd tvm-fatbin-lab
grep -c 'Primitive=1' out/tvm/02_relay_after_fuse.ir.txt     # 期望 >= 2
grep -c 'Primitive=1' out/tvm/02_relay_before_fuse.ir.txt    # 期望 0
grep -n 'Primitive'   out/tvm/02_relay_after_fuse.ir.txt     # 看属性实际怎么打印
grep -oE 'nn\.[a-z_0-9]+' out/tvm/02_relay_after_fuse.ir.txt | sort | uniq -c
```

**通过标准**：

- before 的 `Primitive=1` 计数为 0，after 的计数 ≥ 2（**属性打印格式随版本略有差异**；若 grep 不到先看 `grep -n 'Primitive'` 的实际输出再定 pattern，别直接判定「没融合」）
- 你能说出 `dense → add → relu` 落在同一个融合函数里，理由是「complex-out-fusable 可以吸收输出侧的 injective」
- 你能说出屏障之后的 `relu` 单独成组，理由是 opaque 边界切断了融合链
- 你能回答：如果把 `dense` 换成 `sort` 这类真正的 opaque 算子，分组数会往哪个方向变

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| grep 不到 `Primitive=1` 就认为融合没发生 | 版本的 IR 打印格式不同；应先读 IR 原文再定判据 |
| 数出的分组数与预期对不上 | 没注意 `FoldConstant` 先跑过一轮，常量权重可能已经改变了图的形状 |
| 说「融合就是为了少写几行 IR」 | 没抓住实质：融合减少的是**中间张量的物化**与 kernel 启动次数 |

---

## L1 改一处

### L1-TVM-03｜改 tile 大小，看 IR 变而算法不变

- **检验什么**：这条通过 = 你真的掌握了「算法与调度分离」——同一份 `te.compute` 能 lower 出结构完全不同的循环
- **前置**：L0-TVM-01
- **资源**：本地+工具链
- **预计耗时**：1h

**任务**：改 `tvm_lab/01_te_matmul_schedules.py`，做两组改动，每组各跑一次第 01 步并把 `01_matmul_tiled.lower.txt` 另存对比：

1. `main()` 里 `schedule_tiled(C2, tile=16)` 的 tile 依次改成 8、32、64。
2. `schedule_tiled()` 里 `s[CL].split(s[CL].op.axis[1], factor=8)` 的 factor 改成 4 与 16 各试一次。

全程**不许动 `build_matmul()`**。

**先预测再动手**：

1. tile 从 16 改到 64（等于矩阵边长 N=64），外层块循环的 extent 会变成多少？这层循环会不会被直接消掉？
2. `split(factor=8)` 改成 16 之后，`vectorize` 作用的那条轴宽度变成多少？如果 factor 不能整除该轴长度，IR 里会多出什么结构（提示：边界处理会引入条件判断）？
3. 这五次改动里 `C` 的定义一次都没变。既然算法没变，为什么 lower 出来的 IR 差别这么大？把答案写成一句话——这句话就是「算法 / 调度分离」的定义。

**验收命令**：

```bash
cd tvm-fatbin-lab
cp out/tvm/01_matmul_tiled.lower.txt /tmp/tiled_16.txt
cp out/tvm/01_matmul_naive.lower.txt /tmp/naive_base.txt
# 编辑 tvm_lab/01_te_matmul_schedules.py，把 tile 改成 8
python tvm_lab/01_te_matmul_schedules.py
cp out/tvm/01_matmul_tiled.lower.txt /tmp/tiled_8.txt
diff /tmp/tiled_16.txt /tmp/tiled_8.txt          # 应有差异
diff /tmp/naive_base.txt out/tvm/01_matmul_naive.lower.txt   # 应为空
# 再把 tile 改成 64、把 split factor 改成 16，各重复一次
```

**通过标准**：

- 各版本 tiled lower 两两 `diff` 非空，差异集中在循环 extent 与嵌套层数上
- naive 版 lower 在每次改动后重跑都与基线**完全相同**（`diff` 为空）——这是「你只改了 schedule」的机器证据
- 你能对至少一次改动说清：哪条循环的 extent 从多少变成了多少，为什么

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 改完 tile 后 lower 直接报错 | factor 与轴长度不整除，或 `compute_at` 挂到了已失效的轴上；说明不清楚 `tile`/`split` 之后原轴对象就不能再用了 |
| naive 版 lower 也变了 | 你改到了 `build_matmul()`（算法）而不是 schedule——这恰好说明本条检验的价值 |
| 只 `diff` 不读，说不清 extent 变化 | lower 文本里每层 `for` 都写着范围，逐行对就能读出来 |

---

### L1-TVM-04｜增删融合屏障，先预测分组数再验证

- **检验什么**：这条通过 = 你掌握了「融合边界由算子类别 + 显式屏障共同决定」，能提前推断分组结果而不是事后解释
- **前置**：L0-TVM-02
- **资源**：本地+工具链
- **预计耗时**：1h

**任务**：在 `tvm_lab/02_fusion_relay.py` 里做两组改动，每组跑一次并记录 `Primitive=1` 的计数：

- **A 组**：删掉 `relay.annotation.stop_fusion` 那一段（连同 `except` 里的 cast 回退），让 `dense → add → relu → relu` 一路到底。
- **B 组**：恢复屏障，在末尾再接 `relay.nn.softmax`（reduction）+ `relay.add`（injective），观察 reduction 的融合方向。

**先预测再动手**：

1. A 组删掉屏障后，四个 op 会融成 1 组还是 2 组？依据是四类规则里的哪一条？
2. B 组里 `softmax` 后面那个 `add`，会被吸进 softmax 所在的组，还是另开一组？（提示：reduction 融的是**输入侧**的 injective）
3. 如果把 `relay.nn.dense` 换成 `relay.nn.conv2d`（同属 complex-out-fusable），分组数会不会变？为什么输入 shape 与常量权重形状必须跟着改？

**验收命令**：

```bash
cd tvm-fatbin-lab
python tvm_lab/02_fusion_relay.py
grep -c 'Primitive=1' out/tvm/02_relay_after_fuse.ir.txt
cp out/tvm/02_relay_after_fuse.ir.txt /tmp/fuse_base.txt
# 改成 A 组后重跑
python tvm_lab/02_fusion_relay.py
grep -c 'Primitive=1' out/tvm/02_relay_after_fuse.ir.txt
cp out/tvm/02_relay_after_fuse.ir.txt /tmp/fuse_A.txt
diff /tmp/fuse_base.txt /tmp/fuse_A.txt
```

**通过标准**：

- A 组的分组数**严格小于**基线（屏障去掉 → 融合链更长）
- B 组能指出 softmax 所在组的边界在哪一条边上，并说清是哪条规则划的
- 每组动手前写下的预测数字与实际 grep 结果并排记录；**不一致的地方必须给出新的解释**，不能含糊过去

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 删掉 `stop_fusion` 后分组数没变 | `except` 分支里的 cast 往返还在，它同样是屏障；说明没把建图代码读全 |
| 预测和实际差一组 | 多半把「输入侧 / 输出侧」记反了；回学习指南第 2.2.2 节的规则表 |
| 换 op 后报 shape 错误 | 没同步改 `relay.var` 的 shape 与常量权重形状；说明只记住了融合规则，没建立「图是有类型的」概念 |

---

### L1-TVM-05｜调 AUTOTVM_TRIALS，验证 log 复用不再重搜

- **检验什么**：这条通过 = 你掌握了「搜索空间 → 实测 → 最优配置」的闭环，以及 `apply_history_best` 为什么是对抗配置组合爆炸的关键
- **前置**：L0-TVM-01
- **资源**：本地+工具链
- **预计耗时**：1h

**任务**：用环境变量 `AUTOTVM_TRIALS` 跑三次 `tvm_lab/05_autotvm_tune.py`（如 4、12、以及一个远大于空间大小的数），每次记录 `out/tvm/05_READING.md` 里的「搜索耗时」与「复用 log 再 build 耗时」。再做一次对照：**不删 log** 直接重跑，看这两个耗时的比值。

**先预测再动手**：

1. 脚本里有 `trials = min(n_trial, space_size)`。当 `AUTOTVM_TRIALS=1000` 时实际测量次数是多少？搜索耗时会不会跟着线性增长？
2. 搜索耗时与 `apply_history_best` 之后 build 的耗时，量级差多少？为什么后者完全不需要再上机测量？
3. `05_autotvm.log` 是追加写还是覆盖写？跑三次之后，`apply_history_best` 取的是最后一次的结果，还是全部记录里最好的那条？

**验收命令**：

```bash
cd tvm-fatbin-lab
python -c "import tvm; print(tvm.__version__)"     # AutoTVM API 以本地版本为准
AUTOTVM_TRIALS=4  python tvm_lab/05_autotvm_tune.py
grep -E '搜索耗时|复用 log|config space|数值正确' out/tvm/05_READING.md
AUTOTVM_TRIALS=12 python tvm_lab/05_autotvm_tune.py
grep -E '搜索耗时|复用 log|config space|数值正确' out/tvm/05_READING.md
wc -l out/tvm/05_autotvm.log
```

**通过标准**：

- 「搜索耗时」随 trials 增加而增加，撞上 config space 上限后不再增长
- 「复用 log 再 build 耗时」在每一轮都**显著小于**同轮搜索耗时（通常低一到两个数量级）
- 每轮都打印 `数值正确: True`（脚本内用 `np.allclose` 对着 `a @ b` 验）
- 你能说出 config space 的大小是多少，以及它由哪两个 `cfg.define_split` 的什么 `filter` 决定

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 搜索耗时几乎不随 trials 变化 | trials 已被 `min()` 截到 `space_size`；说明没看脚本打印的空间大小 |
| 复用耗时与搜索耗时同量级 | log 路径没对上或 log 为空，`apply_history_best` 实际没拿到记录，悄悄退回了 fallback 配置 |
| `autotvm.tuner.RandomTuner` 报 `AttributeError` | AutoTVM 在部分版本里模块位置有变；以本地版本为准，先 `python -c "from tvm import autotvm; print(dir(autotvm.tuner))"` |

---

## L2 加组件（主判据）

### L2-TVM-06｜新增一个 TE 算子，写两套 schedule 并量化对比

- **检验什么**：这条通过 = 你能独立走完「TE 描述算法 → 写 schedule → lower 看 IR → 计时 → `np.allclose` 验数值 → 接进流水线」这条最小闭环，而不只是读别人写好的脚本
- **前置**：L1-TVM-03
- **资源**：本地+工具链
- **预计耗时**：3h

**任务**：新建 `tvm_lab/07_te_custom_op.py`，照 `01_te_matmul_schedules.py` 的结构起手（`sys.path.insert(0, ...)` + `from tvm_lab.common import banner, out_dir, require_tvm, write_text`），完成六件事：

1. 用 `te.compute` 写一个新算子，二选一：**matmul + bias + relu**（三个 `te.compute` 串起来）或 **depthwise conv**（`te.compute` + `te.reduce_axis`）。
2. 写两套 schedule：`schedule_baseline()` 只做 `te.create_schedule`；`schedule_opt()` 至少用上 `tile` / `split` / `reorder` / `vectorize` 中的三个（可再加 `cache_write` + `compute_at`）。
3. 两套都 `tvm.lower(..., simple_mode=True)`，用 `write_text` 写出 `out/tvm/07_custom_baseline.lower.txt` 与 `out/tvm/07_custom_opt.lower.txt`。
4. 两套都 `tvm.build(s, args, target="llvm")`，在 `tvm.cpu(0)` 上各跑若干次计时（`time.time()` 循环取均值即可；若用 `func.time_evaluator`，签名以本地版本为准），把耗时写进 `out/tvm/07_READING.md`。
5. 两套输出都用 `np.allclose` 对着 numpy 参考实现验证，并把结果打印出来。
6. 把 `07_te_custom_op.py` 加进 `scripts/run_tvm.sh` 的 `STEPS` 数组，并把回显里的 `TVM step ${idx}/6` 改成 `/7`。

**先预测再动手**：

1. matmul+bias+relu 用三个 `te.compute` 写，默认 schedule 下会物化几个中间张量？如果对 bias 与 relu 两个 stage 调 `compute_inline()`，物化数会变成几个？（这就是融合在 TE 层的对应物）
2. 你的 `schedule_opt` 里 `vectorize` 作用的轴长度是多少？它和机器 SIMD 宽度（float32 下 AVX2 是 8 个）对得上吗？对不上会发生什么？
3. 动手前先猜一个 baseline / opt 的耗时比值。如果跑完发现 opt 反而更慢，你会按什么顺序排查（提示：规模太小被启动开销淹没、reduce 轴没处理、向量化宽度不匹配）？

**验收命令**：

```bash
cd tvm-fatbin-lab
python tvm_lab/07_te_custom_op.py
ls -l out/tvm/07_custom_baseline.lower.txt out/tvm/07_custom_opt.lower.txt out/tvm/07_READING.md
diff out/tvm/07_custom_baseline.lower.txt out/tvm/07_custom_opt.lower.txt
grep -E '数值正确|耗时' out/tvm/07_READING.md
bash scripts/run_tvm.sh          # 七步应全部跑完并以 [OK] 结束
```

**通过标准**：

- 两份 lower 文本都存在且 `diff` 非空；你能在 opt 版里指出 `tile` 与 `vectorize` 留下的痕迹
- 脚本打印的两套 schedule 的 `np.allclose` 结果**都为 True**
- `07_READING.md` 里有两套的耗时数字与比值
- `bash scripts/run_tvm.sh` 顺序执行到第 7 步并以 `[OK] 全部 TVM 步骤完成` 结束

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| opt 版的 `np.allclose` 为 False | schedule 改变了计算语义（多半是 reduce 轴被错误 split 或 reorder）；schedule 本应只改「怎么算」不改「算什么」——这个 False 就是没掌握的直接证据 |
| lower 阶段报「找不到 axis」 | `tile`/`split` 之后必须用返回的新轴，原轴对象已失效；说明不清楚原语的返回值含义 |
| opt 明显慢于 baseline | 规模太小被启动开销淹没，或向量化宽度与硬件不匹配；把 N 提到 512 以上再测一遍 |
| `run_tvm.sh` 跑不到新脚本 | 只加了文件没加进 `STEPS`，或数组里的文件名拼写与实际不一致 |

---

### L2-TVM-07｜给新算子写 AutoTVM template 并复现最优配置

- **检验什么**：这条通过 = 你掌握了「搜索空间不是自动来的，是你用 `cfg.define_*` 在模板里显式定义出来的」，以及 log 复用如何保证结果可复现
- **前置**：L2-TVM-06
- **资源**：本地+工具链
- **预计耗时**：2.5h

**任务**：新建 `tvm_lab/08_custom_autotvm.py`，照 `05_autotvm_tune.py` 的骨架，为 L2-TVM-06 的算子写调优模板：

1. 用 `@autotvm.template("ai_infra/<你的算子名>")` 装饰一个返回 `(s, [args...])` 的函数。
2. 至少写两个 `cfg.define_split`（如 `tile_i` / `tile_j`），并用 `filter=` 限制候选，避免空间过大。
3. `autotvm.task.create(..., target="llvm")` → `RandomTuner` → `autotvm.callback.log_to_file` 写 `out/tvm/08_custom_autotvm.log`；trial 数读 `AUTOTVM_TRIALS`，**并加一个分支：为 0 时跳过搜索，直接复用已有 log**。
4. 搜索完用 `autotvm.apply_history_best(log)` 包住 `tvm.build`，跑一次 `np.allclose` 数值验证。
5. 把 config space 大小、搜索耗时、复用耗时、数值结果写进 `out/tvm/08_READING.md`。
6. 加进 `scripts/run_tvm.sh` 的 `STEPS`。

AutoTVM 的装饰器名、`get_config()` 的位置、`apply_history_best` 的上下文语义在不同版本有差异，**以本地 tvm 版本的文档为准**。自查：

```bash
python -c "import tvm; print(tvm.__version__)"
python -c "from tvm import autotvm; print([n for n in dir(autotvm) if 'templ' in n or 'apply' in n])"
python -c "from tvm import autotvm; help(autotvm.template)"
```

**先预测再动手**：

1. 两个 `define_split`、`num_outputs=2`、filter 限定内层为 {4, 8, 16}，在 N=64 时 config space 的理论大小是多少？跑出来的 `len(task.config_space)` 和你的估算差多少，差在哪（提示：filter 之外还有整除约束）？
2. 把 `filter` 整个删掉，空间会涨到什么量级？RandomTuner 在 8 次 trial 下命中最优的概率会怎么变？
3. 在 `apply_history_best` 上下文**之外**再 build 一次，拿到的会是最优配置还是 fallback 配置？你打算从耗时还是从 lower IR 上区分这两者？

**验收命令**：

```bash
cd tvm-fatbin-lab
AUTOTVM_TRIALS=8 python tvm_lab/08_custom_autotvm.py
wc -l out/tvm/08_custom_autotvm.log            # 每条 trial 一行
grep -E 'config space|搜索耗时|复用|数值正确' out/tvm/08_READING.md
# 复现性检查：不重新搜索，只复用 log 再 build 一次
AUTOTVM_TRIALS=0 python tvm_lab/08_custom_autotvm.py
wc -l out/tvm/08_custom_autotvm.log            # 行数应保持不变
```

**通过标准**：

- `08_custom_autotvm.log` 非空，行数与实际 trial 数一致
- `apply_history_best` 之后 build 出来的函数，`np.allclose` 为 True
- 以 `AUTOTVM_TRIALS=0` 第二次运行时不产生新的 trial 行（log 行数不变），但仍能 build 出与上一轮相同的配置
- 你能从 log 最后一行里读出被选中的 split 因子，并在 lower IR 里找到对应的循环 extent

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| `len(task.config_space)` 是 1 | `define_split` 写在 `get_config()` 之前或作用域之外，cfg 没生效；说明不理解模板会在「定义空间」和「应用配置」两种模式下各跑一次 |
| 所有 trial 都失败（错误计数拉满） | schedule 里的轴顺序与 cfg 切分不兼容；先把模板当普通 schedule 单跑一次确认能 lower |
| log 有内容但 `apply_history_best` 不生效 | 模板名字符串与 `task.create` 的第一个参数不一致 |
| 换个 TVM 版本后装饰器直接报错 | AutoTVM 在部分版本已进入维护状态；以本地文档为准，必要时改走 MetaSchedule（见 L3-TVM-10） |

---

### L2-TVM-08｜换一种 layout 变换，验证语义保持

- **检验什么**：这条通过 = 你真的理解「逻辑 shape ≠ 物理 layout」，layout 变换必须保持语义，而它本身是一笔要付带宽的真实成本
- **前置**：L0-TVM-01
- **资源**：本地+工具链
- **预计耗时**：2h

**任务**：新建 `tvm_lab/09_layout_variant.py`，仿 `03_layout_transform.py`（它已经给了 `relay.build` + `graph_executor` + `np.allclose` 的完整写法），做**其中之一**：

- **A 路线**：NCHW → 打包 layout（如 `dst_layout="NCHW4c"` 或 `NCHW8c`；**可用的 layout 字符串以本地版本为准**）。
- **B 路线**：对含 `relay.nn.dense` 的小图做权重侧 layout 变换，或用 `relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]})` 对一张含 conv2d 的小图整体换 layout。

无论选哪条，都要：`relay.build(..., target="llvm")`、在 `tvm.cpu(0)` 上执行、用 `np.allclose` 对着 numpy 参考结果验证、把变换前后的 IR 分别写出到 `out/tvm/09_layout_before.ir.txt` 与 `09_layout_after.ir.txt`。

**先预测再动手**：

1. NCHW → NCHW4c 之后，输出张量的**维数**是 4 还是 5？C 维被拆成了哪两段？如果 C 不能被 4 整除会发生什么？
2. 走 B 路线用 `ConvertLayout` 时，除了 conv2d 本身，图里还有哪些 op 的 layout 会被连带改掉？`layout_transform` 会被插在哪几条边上？
3. 变换前后 `np.allclose` 应该为 True。如果为 False，更可能是你的 numpy 参考实现轴序写错了，还是 TVM 变换错了？你打算用什么办法区分这两种情况？

**验收命令**：

```bash
cd tvm-fatbin-lab
python -c "import tvm; from tvm import relay; print(relay.transform.ConvertLayout.__doc__[:400])"   # 可用性与签名以本地版本为准
python tvm_lab/09_layout_variant.py
grep -n 'layout_transform' out/tvm/09_layout_after.ir.txt
diff out/tvm/09_layout_before.ir.txt out/tvm/09_layout_after.ir.txt
```

**通过标准**：

- 脚本打印 `数值正确: True`（`np.allclose` 对着 numpy 参考结果）
- `09_layout_after.ir.txt` 里能 grep 到 `layout_transform`（或本地版本的等价 op 名），且 before 里没有或数量更少
- 你能说出这次变换在**元素总数完全不变**的前提下改变了什么（步长、访存连续性），以及它在真实部署里为什么算一笔成本

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| `np.allclose` 为 False | 参考实现的 `np.transpose` 轴序与 layout 字符串对不上；先在 numpy 里把四维索引映射手推一遍 |
| 报未知 layout | 该 layout 字符串在本地版本或后端不受支持；先用 NHWC 跑通，再挑战打包 layout |
| 认为「layout 变换只是换个名字」 | 没意识到它是真实的数据搬运；对着 IR 里 transform 节点的输入输出 shape 看一遍就明白了 |

---

### L2-TVM-09｜自己组 pass 序列，验证顺序敏感性

- **检验什么**：这条通过 = 你理解 `opt_level` 背后是一串**有序**的 pass，pass 之间存在依赖与破坏关系（和 LLVM 的 phase ordering 是同一个问题）
- **前置**：L0-TVM-02
- **资源**：本地+工具链
- **预计耗时**：1.5h

**任务**：新建 `tvm_lab/10_pass_pipeline.py`，用与 `02_fusion_relay.py` 相同的那张图（import 它的建图逻辑或复制过来），跑五条 pipeline，各写出一份 IR 到 `out/tvm/10_pipeline_<名字>.ir.txt`：

1. 什么 pass 都不跑（baseline）
2. 只 `relay.transform.FoldConstant()`
3. 只 `relay.transform.FuseOps()`
4. `tvm.transform.Sequential([FoldConstant(), FuseOps()])`
5. `tvm.transform.Sequential([FuseOps(), FoldConstant()])`

再在 `out/tvm/10_READING.md` 里列一张表：pipeline 名 → `Primitive=1` 计数 → 常量节点数 → 一句话结论。

**先预测再动手**：

1. 只跑 `FuseOps` 不跑 `FoldConstant`，分组数会和「两个都跑」一样吗？如果不一样，是更多还是更少？
2. `FuseOps` 之后再跑 `FoldConstant`，还能折出东西吗？（提示：常量此时可能已经被包进 `Primitive=1` 的函数体里了）
3. `tvm.transform.PassContext(opt_level=N)` 里的 `N` 与你手写的 `Sequential` 是什么关系？把 `opt_level` 从 3 降到 0，你手动调用的 pass 还会执行吗？

**验收命令**：

```bash
cd tvm-fatbin-lab
python tvm_lab/10_pass_pipeline.py
for f in out/tvm/10_pipeline_*.ir.txt; do echo "== $f"; grep -c 'Primitive=1' "$f" || true; done
diff out/tvm/10_pipeline_fold_then_fuse.ir.txt out/tvm/10_pipeline_fuse_then_fold.ir.txt
```

**通过标准**：

- 五份 IR 全部产出，`10_READING.md` 的表格填满
- 至少有**一对** pipeline 的 IR `diff` 非空，且你能解释差异从哪来
- 若两种顺序的产物恰好完全相同，你必须能说清为什么这张图对顺序不敏感，并**构造一个对顺序敏感的图**（提示：让常量参与到会被融合的算子里）再验一次

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 所有 pipeline 产出一模一样 | 图太简单，既没有可折叠的常量也没有可融合的链；换成带 const 权重的 dense + add 链 |
| `opt_level=0` 之后手动调的 pass 也不生效 | 部分 pass 有 `opt_level` 门槛，低于门槛会被跳过；说明没理解 `PassContext` 是 pass 的运行条件而不只是个开关 |
| 把 `Sequential` 当成「顺序无关的集合」 | 没建立 phase ordering 的概念——这正是本条要打的点 |

---

## L3 打通

### L3-TVM-10｜用 MetaSchedule 跑同一算子，对比空间定义方式

- **检验什么**：这条通过 = 你说得清「模板式（AutoTVM）」与「模板无关 / 自动生成（MetaSchedule）」在**搜索空间从哪来**这件事上的根本差别
- **前置**：L2-TVM-07
- **资源**：本地+工具链（搜索耗时长，核数越多越舒服）
- **预计耗时**：2h

**任务**：用 `tvm.meta_schedule` 对 L2-TVM-06 的算子（或直接用 matmul）跑一次**小规模**调优，产物写到 `out/tvm/11_ms_work_dir/`；然后在 `out/tvm/11_READING.md` 里写一张两栏对照表，四行分别是：**空间从哪来 / 谁写 schedule / 调优记录长什么样 / 怎么复用**。AutoTVM 那一栏必须用你在 L2-TVM-07 里的实际体验来填，不是抄文档。

**MetaSchedule 的 API 在不同 TVM 版本差异极大**（`tune_tir` / `tune_te` / `TuneContext` / `database` 的名字与签名都改过），**一切以本地 tvm 版本的文档为准**。先自查：

```bash
python -c "import tvm; print(tvm.__version__)"
python -c "import tvm.meta_schedule as ms; print([n for n in dir(ms) if n.startswith('tune')])"
python -c "import tvm.meta_schedule as ms; help(ms.tune_tir)"    # 若不存在，按上一行列出的名字改
python -c "import tvm.meta_schedule as ms; print(ms.__file__)"   # 直接去这个目录读源码里的真实签名
```

若本地连 `tvm.meta_schedule` 都 import 不了，说明该构建未启用，本条**降级**为：只读官方文档 + 填对照表，不要求跑出 log。

**先预测再动手**：

1. AutoTVM 要你写 `cfg.define_split` 才有空间；MetaSchedule 不写模板也能搜，它的候选从哪来（提示：schedule rule / 自动 tiling 规则作用在 TensorIR 的 block 上）？
2. 同一个 matmul，MetaSchedule 的空间比你手写的 AutoTVM 空间大几个量级？大空间分别对调优时间和 cost model 提出了什么要求？
3. 两者的调优记录（AutoTVM 的一行 JSON vs MetaSchedule 的 database）在「可复现」这件事上，哪个约束更强？为什么？

**验收命令**：

```bash
cd tvm-fatbin-lab
python -c "import tvm.meta_schedule as ms; print('ok')"    # 先确认模块存在
python tvm_lab/11_metaschedule.py                          # 你自己写的脚本，API 以本地版本为准
ls out/tvm/11_ms_work_dir/
grep -E '对照|空间|复用' out/tvm/11_READING.md
```

**通过标准**：

- 要么产出 MetaSchedule 的 work dir（含 database / tuning record），并跑出一次 `np.allclose` 为 True 的 build；
- 要么在模块不可用时，把上面自查命令的实际输出与降级说明写进 `11_READING.md`；
- 两种情况都要求对照表四行填满，且 AutoTVM 一栏来自你自己跑过的实验。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| `ModuleNotFoundError: tvm.meta_schedule` | 本地构建未启用该组件；按降级方案走，别硬凑 |
| 照抄网上教程直接报 `TypeError` 签名不符 | 版本漂移；用 `help()` 读本地签名——这本身就是本条要练的能力 |
| 搜索跑几个小时不结束 | `max_trials_global` 之类的上限没设小；小规模验证不需要真的搜到最优 |

---

### L3-TVM-11｜（可选，需 GPU）target 换成 cuda，看 thread binding

- **检验什么**：这条通过 = 你理解 `bind` 与 GPU 硬件轴的对应关系，以及同一份 TE 算法在不同 target 下 lower 出的循环结构差异
- **前置**：L2-TVM-06
- **资源**：**单卡GPU**（无卡时按下面的降级方案做，仍能完成主要观察）
- **预计耗时**：1.5h

> **本册唯一需要申请 GPU 的条目。** 申请话术见 [`./README.md`](./README.md) §6.2。前面十条全部在 CPU 上完成。

**任务**：新建 `tvm_lab/12_cuda_matmul.py`，把 matmul 的 schedule 改成 GPU 版：用 `s[C].bind(axis, te.thread_axis("blockIdx.x"))` / `threadIdx.x` 之类把轴绑到硬件维度，target 设为 `"cuda"`，写出 `out/tvm/12_cuda_matmul.lower.txt`；有卡时在 `tvm.cuda(0)` 上执行并用 `np.allclose` 验数值。

**无卡时的降级方案**（按顺序尝试，走到哪步算哪步）：

1. **只做 `tvm.lower(s, args, simple_mode=True)`**——lower 不需要设备，你依然能看到绑定后的轴变成 `blockIdx.x` / `threadIdx.x`。这一步在任何机器上都能完成，也是本条的主要观察目标。
2. 再**尝试** `tvm.build(s, args, target="cuda")`，成功的话打印 `mod.imported_modules[0].get_source()` 看生成的 CUDA C 源码。**这一步能否成功取决于本地 TVM 是否带 CUDA 支持、以及是否装了 CUDA Toolkit**；若报缺少设备 / 找不到 `nvcc` / `libcuda` 之类的错误，不要纠结，**退回第 1 步只看 lower IR**。
3. 真机执行（`tvm.cuda(0)` + `np.allclose`）只在有卡时做。

**先预测再动手**：

1. `bind` 之后，lower IR 里原来那层 `for` 会变成什么？循环还在吗，还是被换成了「每个线程执行一份」的隐式并行？
2. CPU 版的 `vectorize` 与 GPU 版的 `bind(threadIdx.x)` 都在利用并行，二者在 IR 上的表现差在哪？
3. 如果 tile 大小让 `threadIdx.x` 的 extent 超过 1024，报错会发生在 lower、build 还是运行时？

**验收命令**：

```bash
cd tvm-fatbin-lab
python -c "import tvm; print(tvm.cuda(0).exist)"     # False = 无可用 GPU，走降级方案
nvidia-smi || echo "无 GPU，按降级方案只看 lower IR"
python tvm_lab/12_cuda_matmul.py
grep -nE 'blockIdx|threadIdx' out/tvm/12_cuda_matmul.lower.txt
```

**通过标准**：

- **有卡**：`np.allclose` 为 True，lower IR 里 grep 得到 `blockIdx` / `threadIdx`，并能贴出一段生成的 CUDA 源码
- **无卡**：lower IR 里 grep 得到 `blockIdx` / `threadIdx`，并在 READING 里记录 `tvm.build(target="cuda")` 那一步的实际报错原文（这条报错本身就是「编译期依赖 vs 运行期依赖」的好素材）
- 两种情况都要能回答：为什么 lower 不需要设备，而 build / run 可能需要

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| lower 里找不到 `threadIdx` | `bind` 绑到了错误的轴，或 schedule 根本没走到 bind 那一行 |
| 有卡但 `np.allclose` 为 False | 线程划分与 reduce 轴处理冲突（常见于跨线程归约没用 `rfactor` 或没做原子处理） |
| 把 build 失败当成本条失败 | 降级方案第 2 步本来就是「尽力而为」；本条的核心观察在 lower IR |

> CUDA 工具链本身那一轨（fatbin 结构、多架构编译、`cuobjdump` 验证）在 [`./07-cuda-fatbin.md`](./07-cuda-fatbin.md)，用的是同一个 lab 的 `scripts/run_fatbin.sh`，本册不涉及。

---

## 条目 × 资源需求速查

| 编号 | 任务 | 级别 | 资源 | 耗时 |
|------|------|------|------|------|
| L0-TVM-01 | 跑通六步，lower IR 对回 schedule 原语 | L0 | 本地+工具链 | 1h |
| L0-TVM-02 | 数融合分组，对回四类规则 | L0 | 本地+工具链 | 1h |
| L1-TVM-03 | 改 tile 大小，IR 变而算法不变 | L1 | 本地+工具链 | 1h |
| L1-TVM-04 | 增删融合屏障，预测分组数 | L1 | 本地+工具链 | 1h |
| L1-TVM-05 | 调 AUTOTVM_TRIALS，验证 log 复用 | L1 | 本地+工具链 | 1h |
| **L2-TVM-06** | **新增 TE 算子 + 两套 schedule + 量化对比** | L2 | 本地+工具链 | 3h |
| **L2-TVM-07** | **给新算子写 AutoTVM template 并复现最优配置** | L2 | 本地+工具链 | 2.5h |
| L2-TVM-08 | 换一种 layout 变换，验证语义保持 | L2 | 本地+工具链 | 2h |
| L2-TVM-09 | 自组 pass 序列，验证顺序敏感性 | L2 | 本地+工具链 | 1.5h |
| L3-TVM-10 | MetaSchedule 对比 AutoTVM 的空间定义 | L3 | 本地+工具链 | 2h |
| L3-TVM-11 | target 换 cuda，看 thread binding | L3 | **单卡GPU**（可降级） | 1.5h |

**合计约 17.5 小时**：L0 + L1 共 5h，L2 共 9h（主判据），L3 共 3.5h。

**本册只有 L3-TVM-11 需要 GPU，且它是可选项并给了无卡降级方案。** 其余十条全部在个人开发机的 CPU 上完成，target 一律 `llvm`、设备一律 `tvm.cpu(0)`。

**下一步**：完成本册后接 [`./05-onnx-ort.md`](./05-onnx-ort.md)——TVM 的 `FuseOps` 与 ORT 的 EP 分区是同一个问题（哪些节点归到同一个可执行单元）在不同系统里的两种答案，对照着做感受最深。想走硬件那一轨就接 [`./07-cuda-fatbin.md`](./07-cuda-fatbin.md)，同一个 lab 换 `scripts/run_fatbin.sh`。
