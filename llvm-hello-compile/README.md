# llvm-hello-compile —— 一个极简的 LLVM 编译全流程学习项目

**两个入口，各跑一条命令即可：**

| 脚本 | 回答什么问题 | 场景源码 |
|------|-------------|---------|
| `bash scripts/run.sh` | **链路**：一份源码怎么一步步变成可执行文件 | `src/sum.c` |
| `bash scripts/tour.sh` | **要点**：[学习指南](../docs/llvm-learning-guide.md) §9.1 那 11 条必学点分别长什么样 | `src/kernel.c` + `src/poison_demo.ll` + `src/mini.td` |

`run.sh` 先跑（建立整体感），`tour.sh` 后跑（把每个概念都亲眼看一遍）。
两者都会边跑边讲解，并把所有中间产物落盘。

本项目还**手写了 5 个自定义 Pass**（`passes/`），用 New PassManager 插件方式挂进同一套管线，
覆盖「Module / Function 两个层级」与「四种 `PreservedAnalyses` 写法」。

## 在自学体系中的位置

| | |
|--|--|
| **角色** | 编译器地基（P2）：建立「单层 IR + Pass + CodeGen」手感 |
| **总规划** | [`../README.md`](../README.md) §5.1 |
| **配套教材** | [`../docs/llvm-learning-guide.md`](../docs/llvm-learning-guide.md)（机制）· [`../docs/paper-notes/02-llvm.md`](../docs/paper-notes/02-llvm.md)（动机） |
| **阶段导航** | [`../docs/README.md` 阶段 0](../docs/README.md#阶段-0编译器地基半天1-天可与阶段-1-并行) |
| **下一站** | [`../mlir-toy-dialect/`](../mlir-toy-dialect/) —— 同一套心智模型升到多层 dialect |

> 建议在啃 MLIR 之前先跑通本项目（半天）；卡在 `llvm` dialect / `poison` / 向量化时再回 [`llvm-learning-guide`](../docs/llvm-learning-guide.md) 深挖。

---

## 一、你将学到什么

| 阶段 | 工具 | 核心概念 | 在本项目中的体现 |
|------|------|---------|-----------------|
| **前端 Frontend** | `clang -ast-dump` | 词法/语法/语义分析、AST | `01_ast.txt`：编译器如何把源码理解成一棵树 |
| **生成 IR** | `clang -emit-llvm` | LLVM IR、alloca/load/store | `02_sum_O0.ll`：未优化 IR，局部变量都是内存 |
| **优化 Pass（单个）** | `opt -passes=mem2reg` | **SSA、phi 节点、Pass** | `03a_sum_mem2reg.ll`：内存变量→寄存器，出现 φ |
| **优化 Pass（-O2）** | `opt -O2` | 内联、常量传播、循环优化 | `03b_sum_O2.ll`：`square()` 被内联，IR 变紧凑 |
| **自定义 Pass（插件）** | `opt -load-pass-plugin` | **手写 Pass、分析 vs 变换、插件注册** | `passes/` + `03c_sum_logged.ll`：把自己的 Pass 挂进管线 |
| **IR 表示** | `llvm-as` / `llvm-dis` | 文本 `.ll` ↔ 位码 `.bc` | `04_*`：两种等价形式互转 |
| **后端 CodeGen** | `llc` | 指令选择/寄存器分配/发射汇编 | `05_sum_O2.s`：真实的 x86-64 汇编 |
| **汇编+链接** | `clang`(`as`+`ld`) | 目标文件、链接、可执行 | `06_sum`：能跑的可执行文件 |
| **JIT 执行** | `lli` | 直接解释/JIT 执行 IR | 不落地机器码，直接跑 `.ll` 验证语义 |

一句话概括你要建立的认知：

```
C 源码  ──①clang前端──▶  AST  ──②emit-llvm──▶  LLVM IR(.ll)
                                                   │
                    ③opt 优化                      │  ← SSA / Pass / 内联
                      ├ ③a mem2reg（内置：进入 SSA）│
                      ├ ③b -O2   （内置：内联等）   │
                      └ ③c 自定义 Pass（插件：count-ir / inject-log）
                                                   ▼
                              优化后 IR  ──⑤llc──▶  汇编(.s)  ──⑥as+ld──▶  可执行文件
```

**核心心智模型**：LLVM 把编译拆成「前端各写各的 → 统一到 LLVM IR → 在 IR 上做与语言无关的优化 → 后端各自生成机器码」。
IR 是整个体系的枢纽，这也正是 MLIR「多层 IR」思想的源头。

---

## 一之二、核心要点覆盖表（`tour.sh` 的 17 个站点）

学习指南 [§9.1「必须掌握」](../docs/llvm-learning-guide.md#91-必须掌握) 的 11 条，
在 `bash scripts/tour.sh` 里都有一个能亲眼看到的站点：

| 站 | 学习指南 | 要点 | 在项目里怎么看 |
|----|---------|------|--------------|
| 0 | §2.1 | IR 三态：`.ll` ↔ `.bc` 无损互转 | `llvm-as` / `llvm-dis` 往返 + 体积对比 |
| 1 | §2.2 | 类型系统：`iN` / `float` / **opaque `ptr`** / `<N x T>` / 结构体 | `kernel.c` 里的 `Tensor` 与向量类型 |
| 2 | §2.3 | **SSA + 显式 CFG + terminator + `phi`** | `mem2reg` 前后：`alloca` 归零、`phi` 出现 |
| 3 | §2.5 | **`getelementptr` 与 `inbounds` 承诺** | `relu_sum` 里 `t->data[i]` 展开的多条 GEP |
| 4 | §2.6 | **属性与标志**：`noalias` / `align` / `dereferenceable`、`contract`→FMA | `restrict` 参数；`-ffp-contract=off` vs `fast` |
| 5 | §2.7 | **poison / undef / freeze**、`select` 毒性屏障 | 手写 `src/poison_demo.ll` 过一遍 `-O2` |
| 6 | §2.8 | Metadata：`!tbaa` / `!llvm.loop` | `-O2` IR 里直接 grep |
| 7 | §2.9 | Intrinsics：`llvm.fmuladd` / `llvm.vector.reduce.*` | `-O2` IR 里直接 grep |
| 8 | §3 | **New PM 四层嵌套 + 四种 `PreservedAnalyses`** | `-debug-pass-manager` + 5 个自定义 pass |
| 9 | §4.1–4.2 | 六个 Analysis + **别名分析四种回答** | `aa-eval`；`axpy` vs `axpy_may_alias` |
| 10 | §4.3–4.4 | 默认 pipeline 骨架 | `opt -print-changed` 只看真正改了 IR 的 pass |
| 11 | §4.5 | 两个向量化器 + **为什么没向量化** | `-Rpass-missed=loop-vectorize` |
| 12 | §4.6 | **TTI 是中端唯一的目标信息入口** | 自写 `tti-info` pass：默认 vs AVX2 的宽度/代价 |
| 13 | §5 | **后端七阶段 + MIR + 寄存器分配链条** | `-stop-after=finalize-isel` vs `virtregrewriter` 两份 `.mir` |
| 14 | §5.7 | MC 层：指令编码 | `llvm-mc -show-encoding` + `llvm-objdump -d` |
| 15 | §6 | **TableGen 在 LLVM 与 MLIR 是同一套语言** | `llvm-tblgen --print-records src/mini.td` |
| 16 | §7 | 与 MLIR 的接缝、三个杠杆 | 命令链说明 + 下一站指路 |

> 场景为什么是 `src/kernel.c`？因为一个「axpy + ReLU 求和」的迷你算子，天然同时含有
> **循环 / 结构体 / 数组下标 / 浮点乘加 / 三元表达式 / 指针别名**——上表大半要点都能从它一份 IR 里看到。

---

## 二、目录结构

```
llvm-hello-compile/
├── README.md            # 本文件
├── .gitignore           # 忽略 out/ 和 build/
├── src/
│   ├── sum.c            # 【run.sh 的场景】平方和：小函数 + 循环 + printf
│   ├── kernel.c         # 【tour.sh 的场景】迷你 AI 算子：axpy + ReLU 求和
│   ├── poison_demo.ll   # 手写 IR：poison / freeze / select 屏障 / 何时变成 UB
│   └── mini.td          # 30 行 TableGen：class / def / multiclass / let
├── passes/              # 【自定义 Pass 源码】编译成插件 MyPasses.so
│   ├── Passes.h              # 5 个 Pass 的类声明（附层级与返回值对照表）
│   ├── CountIR.cpp           # Module 级 · 分析型（count-ir）
│   ├── CfgInfo.cpp           # Function 级 · 分析型：消费 DominatorTree / LoopInfo
│   ├── TTIInfo.cpp           # Function 级 · 分析型：消费 TargetTransformInfo
│   ├── StrengthReduce.cpp    # Function 级 · 变换型：mul→shl，preserveSet<CFGAnalyses>
│   ├── InjectLogging.cpp     # Module 级 · 变换型：入口注入 printf（inject-log）
│   ├── PluginRegistration.cpp# 把管线名注册给 opt（Module + Function 两个回调）
│   └── CMakeLists.txt        # 编译为可被 opt 加载的 MyPasses.so
├── tests/               # lit / FileCheck 风格测试（和 LLVM、MLIR 官方写法一致）
│   ├── strength-reduce.ll
│   └── cfg-info.ll
├── scripts/
│   ├── env.sh           # 定位 clang/opt/llc/lli/llvm-tblgen/llvm-mc/FileCheck ...
│   ├── run.sh           # 【入口一】编译链路：源码 → AST → IR → 汇编 → 可执行
│   ├── tour.sh          # 【入口二】核心要点巡礼：17 个站点覆盖学习指南必学点
│   ├── build_passes.sh  # 构建自定义 Pass 插件（幂等；两个入口都会自动调用）
│   ├── run_tests.sh     # 跑 tests/ 下的 FileCheck 测试
│   └── clean.sh         # 清理 out/
├── build/               # 运行后自动生成（cmake 构建目录，含 passes/MyPasses.so）
└── out/                 # 运行后自动生成
    ├── (run.sh 产物)     # 01_ast.txt / 02_*.ll / ... / ANALYSIS.md
    └── tour/            # (tour.sh 产物) IR / MIR / 汇编 / TOUR.md
```

---

## 三、一键运行

```bash
cd ~/AI-Infra/llvm-hello-compile
bash scripts/run.sh          # 入口一：走完 ①~⑦ 全流程，产物存到 out/
bash scripts/tour.sh         # 入口二：走完 17 个要点站点，产物存到 out/tour/
bash scripts/run_tests.sh    # （可选）跑自定义 pass 的 FileCheck 测试
```

> 建议顺序：`run.sh` → 读 `out/ANALYSIS.md` → `tour.sh` → 读 `out/tour/TOUR.md`。
> 前者建立「一条链路」的整体感，后者把链路上每个概念单独拎出来看一遍。

> 工具链：脚本会优先激活 conda 环境 `mlir-env`（内含 LLVM 17.0.6 的 `opt/llc/lli/...`），
> `clang` 用系统同版本 17.0.6，两者匹配可无缝衔接。
> 覆盖默认路径：`CONDA_HOME=/path/to/conda LLVM_ENV=my-env bash scripts/run.sh`。

脚本会**边跑边讲解**：每一步都打印 `[链路]（输入→工具→输出）`、`[命令]`、`[结果]`，
并把输出以 **3 种形式全部落盘**，方便你事后反复回看：

- **全量日志** `out/full_run.log`：本次运行的完整终端流水账；
- **分步日志** `out/steps/*.log`：每个步骤的解说 + 结果单独存一份；
- **汇总报告** `out/ANALYSIS.md`：流程图 + 各阶段指令统计对比 + 阅读指引（**建议先看这份**）。

运行后 `out/` 里会得到（**建议按顺序对比阅读**）：

| 文件 | 说明 |
|------|------|
| `ANALYSIS.md` | **汇总报告**：流程图 + 指令统计对比表 + 产物清单 + 阅读顺序 |
| `01_ast.txt` | ① 前端 Clang AST |
| `02_sum_O0.ll` | ② 未优化 IR（充满 `alloca`/`load`/`store`） |
| `03a_sum_mem2reg.ll` | ③a mem2reg 后（`alloca` 消失、出现 `phi`，**SSA 精髓**） |
| `03b_sum_O2.ll` | ③b `-O2` 完整优化后 |
| `03c_sum_logged.ll` / `03c_sum_logged` | ③c **自定义 Pass** inject-log 注入 `printf` 后的 IR 及其可执行（运行可见 trace） |
| `diff_02_to_03a.txt` | ②→③a 的 IR 差异（内存操作如何变成 SSA） |
| `diff_03a_to_03b.txt` | ③a→③b 的 IR 差异（`-O2` 到底改了什么） |
| `04_sum_O0.bc` / `04_roundtrip.ll` | ④ 位码及其转回的文本 |
| `05_sum_O2.s` | ⑤ 目标汇编 |
| `06_sum` | ⑥ 最终可执行文件 |
| `steps/*.log` | 每一步的解说+结果（分步日志） |
| `full_run.log` | 完整运行日志 |

---

## 四、每一步在讲什么（重点解读）

### 步骤 ② vs ③a：理解 SSA 与 mem2reg（**全项目最关键的一步**）

`clang -O0` 生成的 IR，为每个局部变量分配一块栈内存（`alloca`），读写靠 `load`/`store`：

```llvm
; 02_sum_O0.ll （节选）—— acc 是一块内存，每次 += 都要 load 再 store
%acc = alloca i32
store i32 0, ptr %acc
...
%0 = load i32, ptr %acc
%add = add i32 %0, %sq
store i32 %add, ptr %acc
```

跑一个 `mem2reg` Pass 后，内存变量被「提升」为纯寄存器值。由于循环里 `acc` 在
「循环入口」和「上一轮迭代」两个前驱块都有定值，就必须用 **`phi` 节点**来汇合：

```llvm
; 03a_sum_mem2reg.ll （节选）—— 不再有 alloca/load/store，改用 phi
%acc.0 = phi i32 [ 0, %entry ], [ %add, %loop ]   ; ← φ：SSA 的标志
```

> **为什么这重要？** SSA（静态单赋值：每个变量只被赋值一次）是几乎所有现代编译器
> 优化的基础表示。`mem2reg` 是「进入 SSA 世界」的门票，理解了它，就理解了 LLVM 优化的地基。

### 步骤 ③b：`-O2` 做了什么

- **内联（inline）**：`square()` 被直接展开进循环，`@square` 定义随后被删除；
- **常量传播/折叠、循环优化**：IR 进一步简化。
- 脚本会自动生成差异文件 `out/diff_03a_to_03b.txt`（等价于 `diff out/03a_sum_mem2reg.ll out/03b_sum_O2.ll`），直接看优化改了哪些行。

### 步骤 ③c：自定义 Pass（本项目的重点扩展）

前面 `mem2reg`/`-O2` 都是 LLVM **内置** Pass。这一步演示如何把「**自己写的 Pass**」挂进
同一套管线：用 New PassManager 的**插件**方式——`opt` 加载 `.so` 后，按名字调用其中的 Pass。

源码在 `passes/`，编译产物是 `build/passes/MyPasses.so`（两个入口脚本都会自动调 `build_passes.sh` 构建）。
5 个 Pass 刻意分布在**两个层级**上，并把 `PreservedAnalyses` 的**四种写法**用全：

| Pass | 管线名 | 层级 | 类型 | 做什么 | 返回值（对 PassManager 的承诺） |
|------|--------|------|------|--------|-------------------------------|
| `CountIR` | `count-ir` | Module | 分析型 | 统计每个函数的基本块/指令/`call` 数 | `all()`（没动 IR，已有分析全部有效） |
| `CfgInfo` | `cfg-info` | Function | 分析型 | **向 `FunctionAnalysisManager` 索取** DominatorTree / LoopInfo，打印 terminator 与循环深度 | `all()` |
| `TTIInfo` | `tti-info` | Function | 分析型 | 向 **TTI** 询问向量寄存器位宽、`fmul` 各宽度的代价 | `all()` |
| `StrengthReduce` | `strength-reduce` | Function | 变换型 | 把 `mul x, 2^k` 改写成 `shl x, k` | **`preserveSet<CFGAnalyses>()`**（只改指令没动控制流） |
| `InjectLogging` | `inject-log` | Module | 变换型 | 在每个函数入口插一句 `printf("[trace] enter <fn>")` | `none()`（保守：分析全部作废重算） |

> 第四种写法 `PA.preserve<DominatorTreeAnalysis>()`（只保留某一个分析）在
> `StrengthReduce.cpp` 的注释里给了对照——它和 `preserveSet` 的区别是「点名保留」vs「按集合保留」。

三个值得注意的教学点，都写在对应源码的文件头：

1. `CfgInfo.cpp`：分析**不是自己算的**，是向 `AnalysisManager` 要的；命中缓存就不重算。
2. `StrengthReduce.cpp`：改写时**故意不复制原来的 `nsw`/`nuw`**——保守是安全的，
   随手复制标志正是最常见的 poison/UB 来源（对应 [`notes/llvm-poison-ub.md`](../docs/notes/llvm-poison-ub.md)）。
3. `TTIInfo.cpp`：同一段 IR，带不带 `target-features=+avx2`，TTI 的回答完全不同——
   这就是「接新硬件时 TTI 决定优化质量」的最小演示。

手动运行两个 Pass：

```bash
PLUGIN=build/passes/MyPasses.so
# 分析型：只打印统计，无产物
opt -load-pass-plugin=$PLUGIN -passes=count-ir  -disable-output out/02_sum_O0.ll
# 变换型：真的改 IR，注入 printf
opt -load-pass-plugin=$PLUGIN -passes=inject-log -S out/03a_sum_mem2reg.ll -o out/03c_sum_logged.ll
```

把注入后的 IR 编译成可执行并运行，每进一个函数都会打印一行 trace，最后仍算出 `55`
（证明 Pass 只加了日志、没破坏原逻辑）：

```
[trace] enter main
[trace] enter sum_of_squares
[trace] enter square      ← 打印 5 次，暴露真实调用关系 main → sum_of_squares → square×5
...
sum_of_squares(5) = 55
```

> **为什么这重要？** `-O2` 只是内置 Pass 的一种组合，而 `-load-pass-plugin` 让你把手写 Pass
> 插进**同一套 PassManager**。这揭示了 LLVM 的核心设计：**编译优化 = 一串可插拔的 Pass 在 IR 上反复变换**，
> 「谁改不改 IR、改什么」完全由 Pass 自己决定——这也是理解 MLIR Pass/Lowering 的地基。

想改 Pass 逻辑，编辑 `passes/*.cpp` 后重跑 `bash scripts/build_passes.sh`（幂等，会按需重建）即可。

### 步骤 ⑤：后端 CodeGen

`llc` 把与平台无关的 IR 翻译成具体的 x86-64 汇编，内部经历：
**指令选择(ISel) → 寄存器分配 → 指令调度 → 发射汇编**。这是「IR → 机器」的最后一跳。

### 步骤 ⑥/⑦：链接执行 与 JIT

- `clang 05_sum_O2.s -o 06_sum`：clang 顺带调用汇编器 `as` 和链接器 `ld`，生成能跑的可执行文件；
- `lli 03b_sum_O2.ll`：不落地任何机器码文件，直接 JIT 执行 IR——用来快速验证 IR 语义。

---

## 四之二、`tour.sh` 跑完后最该看的 5 组对比

跑完 `bash scripts/tour.sh` 后，产物都在 `out/tour/`。按这个顺序对读，收益最高：

| # | 对比 | 你应该看到什么 |
|---|------|--------------|
| 1 | `00_O0.ll` → `03_mem2reg.ll` | `alloca` 归零、`phi` 出现——**进入 SSA 的那一刻** |
| 2 | `04_contract_off.ll` → `05_contract_fast.ll` | `fmul`+`fadd` 两条 → 一条 `llvm.fmuladd`：**一次舍入的 FMA 是被「许可」出来的** |
| 3 | `06_poison_O2.ll` | `and poison, 0` 仍折叠成 `poison`；`select` 未选中的臂不传染 |
| 4 | `01_O2.ll` 里 `@axpy` vs `@axpy_may_alias` | 有无 `restrict`（→`noalias`）导致向量化结果完全不同 |
| 5 | `21_after_isel.mir` → `22_after_regalloc.mir` | 虚拟寄存器 `%0/%1` 与 `PHI` 如何变成 `$xmm0` 和 copy |

第 5 组是「寄存器分配 + SSA 解构」从抽象变具体的关键一步，值得多花几分钟。

---

## 五、几个值得动手玩的小实验

1. **改优化级别看差异**：把 `run.sh` 里步骤 ③b 的 `-O2` 换成 `-O1`/`-O3`，对比 IR。
2. **只跑指定 Pass**：`opt -S -passes='mem2reg,instcombine,simplifycfg' out/02_sum_O0.ll`，
   体会 Pass 是可自由组合的流水线。
3. **看 Pass 到底改了什么**：`opt -S -passes=mem2reg -print-before-all -print-after-all out/02_sum_O0.ll`。
4. **换目标架构**：`llc -march=aarch64 out/03b_sum_O2.ll -o out/sum_arm64.s`，
   同一份 IR 生成 ARM 汇编——直观体会「IR 与硬件无关」。
5. **分析机器码吞吐**：`llvm-mca out/05_sum_O2.s`，看后端如何评估指令级并行。
6. **加一个函数**：在 `sum.c` 里加个 `int cube(int x){return x*x*x;}` 并调用，重跑，观察内联行为。
7. **改自己的 Pass**：编辑 `passes/InjectLogging.cpp`，把注入的 `printf` 内容改成带参数（如打印函数入参），
   `bash scripts/build_passes.sh` 重建插件后重跑，看 trace 变化。
7b. **把 `restrict` 删掉重跑 tour**：改 `src/kernel.c` 里 `axpy` 的参数，重跑 `tour.sh`，
   看第 9/11 站的别名结论与向量化 remark 怎么变——这是"别名信息传不下来就跑不快"的实感。
7c. **给 `strength-reduce` 加一条规则**：比如把 `sdiv x, 2^k` 变成 `ashr`（注意负数语义！），
   再在 `tests/strength-reduce.ll` 里补一条 `CHECK`，用 `bash scripts/run_tests.sh` 验证。
7d. **故意加错标志**：在 `src/poison_demo.ll` 里给某条 `add` 加上 `nsw` 但让它溢出，
   跑 `opt -passes='default<O2>'`，观察 poison 怎么传播、什么时候才变成 UB。
8. **写第三个 Pass**：模仿 `CountIR.cpp` 新增一个强度削减 Pass（把 `mul x, 2` 换成 `shl x, 1`），
   在 `PluginRegistration.cpp` 里注册一个新管线名，体会窥孔优化如何落地。

---

## 六、常见问题

- **`opt` 对 `-O0` 的 IR 不生效？** clang 的 `-O0` 会给函数打上 `optnone` 属性，
  阻止后续优化。本项目已用 `-Xclang -disable-O0-optnone` 去掉它（见 `run.sh` 步骤 ②）。
- **IR 里变量是 `%1 %2` 而不是 `%acc`？** 加 `-fno-discard-value-names`（本项目已加）保留可读名字。
- **找不到 `clang`/`opt`？** 确认 conda 环境 `mlir-env` 已建好，或用 `CONDA_HOME`/`LLVM_ENV` 指定；
  `clang` 需在系统 PATH 中且版本与 `opt` 一致（本机均为 17.0.6）。

---

## 七、参考资料

| 主题 | 链接 | 对应本项目 |
|------|------|-----------|
| **本仓库 LLVM 学习文档（优先）** | [`../docs/llvm-learning-guide.md`](../docs/llvm-learning-guide.md) | 跑完本项目后对照读：四层 IR、New PM、CodeGen |
| **自学体系枢纽** | [`../docs/README.md`](../docs/README.md) | 本项目在整条学习路径中的位置 |
| **LLVM LangRef** | https://llvm.org/docs/LangRef.html | 读懂 `02_*.ll` / `03_*.ll` 的语法 |
| **Writing an LLVM Pass** | https://llvm.org/docs/WritingAnLLVMNewPMPass.html | 理解步骤 ③ 的 Pass 机制 |
| **Mem2Reg / SSA** | https://llvm.org/docs/Passes.html#mem2reg-promote-memory-to-register | 步骤 ③a 的原理 |
| **Kaleidoscope 教程** | https://llvm.org/docs/tutorial/ | 想进一步「用 C++ API 亲手造 IR」时的下一站 |

学完这个项目，再去隔壁 [`../mlir-toy-dialect`](../mlir-toy-dialect/)，你会发现 MLIR 的
Operation / BasicBlock / SSA / Pass / Lowering 全都源自这里——只是把「单层 LLVM IR」
扩展成了「多层可扩展 IR」。

祝学习愉快！从 `bash scripts/run.sh` 跑一遍开始，再对着 `out/` 逐个文件读。
