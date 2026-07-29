# llvm-hello-compile —— 一个极简的 LLVM 编译全流程学习项目

用一段 30 行的 C 代码 `src/sum.c`，配合一个脚本 `scripts/run.sh`，**一条命令**走完
LLVM 从「源码」到「可执行文件」的完整编译链路，每一步都落盘中间产物、肉眼可见。

除了跑内置 Pass（`mem2reg`/`-O2`），本项目还**手写了两个自定义 Pass**（`passes/`），
用 New PassManager 插件方式挂进同一套管线，让你亲眼看到「优化 = 一串可插拔的 Pass 在 IR 上变换」。

它对应学习路线第二阶段的 **「LLVM：建立编译器 IR 的系统认知（SSA、Pass Manager、CodeGen 流程）」**，
是读 [LLVM LangRef](https://llvm.org/docs/LangRef.html) 前后最好的动手实践。

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

## 二、目录结构

```
llvm-hello-compile/
├── README.md            # 本文件
├── .gitignore           # 忽略 out/ 和 build/
├── src/
│   └── sum.c            # 唯一的源码：平方和（含小函数+循环+printf）
├── passes/              # 【自定义 Pass 源码】编译成插件 MyPasses.so
│   ├── Passes.h              # 两个 Pass 的类声明
│   ├── CountIR.cpp           # 分析型 Pass：只统计不改 IR（count-ir）
│   ├── InjectLogging.cpp     # 变换型 Pass：入口注入 printf（inject-log）
│   ├── PluginRegistration.cpp# 把管线名注册给 opt（New PassManager 插件入口）
│   └── CMakeLists.txt        # 编译为可被 opt 加载的 MyPasses.so
├── scripts/
│   ├── env.sh           # 定位 clang/opt/llc/lli...（优先用 conda 的 LLVM 17）
│   ├── run.sh           # 【主入口】分步走完全流程，边讲解边把输出落盘
│   ├── build_passes.sh  # 单独构建自定义 Pass 插件（幂等；run.sh 会自动调用）
│   └── clean.sh         # 清理 out/
├── build/               # 运行后自动生成（cmake 构建目录，含 passes/MyPasses.so）
└── out/                 # 运行后自动生成（中间产物 + steps/分步日志 + ANALYSIS.md 报告）
```

---

## 三、一键运行

```bash
cd ~/AI-Infra/llvm-hello-compile
bash scripts/run.sh          # 走完 ①~⑦ 全流程，边跑边讲解，产物存到 out/
```

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

源码在 `passes/`，编译产物是 `build/passes/MyPasses.so`（`run.sh` 会自动调 `build_passes.sh` 构建）。
两个 Pass 一读一写，正好对比出「分析型」与「变换型」的本质区别：

| Pass | 管线名 | 类型 | 做什么 | 返回值（对 PassManager 的承诺） |
|------|--------|------|--------|-------------------------------|
| `CountIR` | `count-ir` | **分析型** | 遍历每个函数，统计基本块/指令/`call` 数并打印 | `PreservedAnalyses::all()`（我没动 IR，已有分析全部有效） |
| `InjectLogging` | `inject-log` | **变换型** | 在每个函数入口插一句 `printf("[trace] enter <fn>")` | `PreservedAnalyses::none()`（IR 变了，分析全部作废重算） |

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
| **LLVM LangRef（IR 语言参考）** | https://llvm.org/docs/LangRef.html | 读懂 `02_*.ll` / `03_*.ll` 的语法 |
| **LLVM 编译流程总览** | https://llvm.org/docs/CommandGuide/ | `clang`/`opt`/`llc`/`lli` 各工具手册 |
| **Writing an LLVM Pass** | https://llvm.org/docs/WritingAnLLVMNewPMPass.html | 理解步骤 ③ 的 Pass 机制 |
| **Mem2Reg / SSA** | https://llvm.org/docs/Passes.html#mem2reg-promote-memory-to-register | 步骤 ③a 的原理 |
| **Kaleidoscope 教程** | https://llvm.org/docs/tutorial/ | 想进一步"用 C++ API 亲手造 IR"时的下一站 |

学完这个项目，再回头看隔壁 `../mlir-toy-dialect`，你会发现 MLIR 的
Operation / BasicBlock / SSA / Pass / Lowering 全都源自这里——只是把「单层 LLVM IR」
扩展成了「多层可扩展 IR」。

祝学习愉快！从 `bash scripts/run.sh` 跑一遍开始，再对着 `out/` 逐个文件读。
