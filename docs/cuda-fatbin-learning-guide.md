# CUDA Fatbin 学习文档：多架构打包与运行时选择

> **本文档的定位**
> - 基于 **NVIDIA CUDA / NVCC 官方文档**里关于编译流水线、虚拟/真实架构、fatbinary 与 `cuobjdump` 的概念整理，不是 CUDA 编程教程。
> - 目标读者是在搭「异构算力网 / 多后端执行基础设施」的编译器与系统工程师——你真正要带走的是**设计模式**，不是 GPU 指令细节。
> - **篇幅刻意压短（半天量）**：根 README §5.4 只要求三点；本文把这三点讲透，其余全部标成跳过。
> - 与 [`iree-learning-guide.md`](./iree-learning-guide.md) **§4.7 ExecutableVariant** 是同一思想的两条实现路径；读完应对着那一节做同构对照。
> - 阶段导航见 [`docs/README.md`](./README.md) 阶段 6（回填与收束）。
> - **先修**：[`ai-compiler-foundations.md`](./ai-compiler-foundations.md) §7.2–7.3（虚拟 ISA vs 真实 ISA、多变体打包）；上游机器码怎么来见 [`llvm-learning-guide.md`](./llvm-learning-guide.md) 第 5 章。
> - **动手项目**：[`../tvm-fatbin-lab/`](../tvm-fatbin-lab/) CUDA 轨 —— `bash scripts/run_fatbin.sh`（或随 `scripts/run.sh`）。
> - **邻接**：多后端**划分**（不是 ISA 打包）见 [`../onnx-delegate-lab/`](../onnx-delegate-lab/)。
>
> **主要信息源**
> - NVCC 用户手册：*CUDA Compiler Driver NVCC*（`-gencode` / `-arch` / `-code` / `-fatbin`）
> - CUDA 编程指南：*Programming Guide* 中 Virtual Architectures / Binary Compatibility / JIT Compilation
> - 二进制工具：`cuobjdump` / `nvdisasm` 手册
>
> **一句话读法**：如果只有一小时，读[第 1 章问题动机](#第-1-章-为什么需要-fatbin)、[第 3 章 compute vs sm](#第-3-章-compute_xx-vs-sm_xx重点)、[第 6 章设计同构](#第-6-章-设计模式同构fatbin--executablevariant)，再做[第 8 章动手](#第-8-章-动手清单半天做完)。

---

## 目录

- [第 1 章 为什么需要 fatbin](#第-1-章-为什么需要-fatbin)
- [第 2 章 CUDA 编译流水线与 fatbin 结构](#第-2-章-cuda-编译流水线与-fatbin-结构)
- [**第 3 章 compute_XX vs sm_XX（重点）**](#第-3-章-compute_xx-vs-sm_xx重点)
- [第 4 章 关键 NVCC 标志](#第-4-章-关键-nvcc-标志)
- [第 5 章 检视工具：cuobjdump / nvdisasm](#第-5-章-检视工具cuobjdump--nvdisasm)
- [第 6 章 设计模式同构：fatbin ↔ ExecutableVariant](#第-6-章-设计模式同构fatbin--executablevariant)
- [第 7 章 与 AI-Infra 的对接点](#第-7-章-与-ai-infra-的对接点)
- [第 8 章 动手清单（半天做完）](#第-8-章-动手清单半天做完)
- [第 9 章 必学三点 vs 全部跳过](#第-9-章-必学三点-vs-全部跳过)
- [附录：一页速查](#附录一页速查)

---

## 第 1 章 为什么需要 fatbin

### 1.1 问题：N×M 发货爆炸

NVIDIA GPU 的**真实指令集代际**用 `sm_XX`（streaming multiprocessor architecture）标识：`sm_70`（Volta）、`sm_75`（Turing）、`sm_80`/`sm_86`（Ampere）、`sm_89`（Ada）、`sm_90`（Hopper）……每一代的 SASS（Shader Assembly）**不保证二进制兼容**——为 `sm_80` 编出来的机器码，不能直接当 `sm_90` 的镜像用。

与此同时，你的软件还要：

- 覆盖用户手里**多种 GPU**（笔记本 75、数据中心 80/90、推理卡 89……）；
- 往往还要覆盖**多个 CUDA Toolkit / 驱动版本**组合；
- 交付物最好是**一个**库 / 一个插件 / 一个 wheel，而不是「按卡型拆十个包」。

若朴素地「每个目标架构各编一份、各发一份」：

```
软件版本 × GPU 架构 × OS/CUDA 组合  →  发货矩阵爆炸（N×M）
```

运维、测试、下载体积、版本对齐全部变差。这就是 fatbin 要消灭的问题。

### 1.2 解法：一个容器，多份镜像

**fatbinary（俗称 fatbin）** 是 NVIDIA 的「胖二进制」容器：在**同一个**设备代码产物里，并行存放**多份**面向不同架构的镜像。运行时由 CUDA driver：

1. 读当前 GPU 的 compute capability；
2. 在 fatbin 里挑一份**最合适**的镜像；
3. 加载进设备（必要时对 PTX 做 JIT）。

宿主侧（host）链接进来的仍然是「一个符号、一次 `<<<>>>` 启动」——调用方不需要知道里面塞了几代架构。

> **一句话**：fatbin = **编译期打包多变体 + 运行时按能力选择**。发一份二进制，跑在多代 GPU 上。

### 1.3 和你真正关心的问题的关系

在 AI 编译 / 异构运行时语境里，这不是「CUDA 奇技」，而是一个反复出现的模式：

| 场景 | 组合爆炸从哪来 | 打包/选择机制 |
|------|----------------|---------------|
| CUDA 部署 | 多代 `sm_XX` | fatbin + driver 选镜像 |
| IREE | 多 backend / 多 target / 多 tile 配置 | `hal.executable` ⊃ 多个 `variant`（见 [§6](#第-6-章-设计模式同构fatbin--executablevariant)） |
| 算力网 | 多厂商加速器 × 多精度 × 多并行配置 | 需要你自己做「多变体产物 + 条件选择」 |

学 fatbin，是为了把这个模式变成肌肉记忆，而不是为了手写 SASS。

---

## 第 2 章 CUDA 编译流水线与 fatbin 结构

### 2.1 总图：`.cu` 如何变成「宿主 + 设备」

NVCC 不是单一编译器，而是一个**驱动（compiler driver）**：它拆分源码、调用宿主编译器（MSVC / gcc / clang）与设备编译器（`cicc` 等），再把结果粘回去。

```
*.cu
  │
  ▼
 NVCC（compiler driver：拆分 / 调度 / 嵌入）
  ├─ Host 路径  →  普通 C++ 目标码（CPU）
  └─ Device 路径
        CUDA C++  →  PTX (compute_XX，虚拟 ISA)
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
         嵌入 PTX（供 JIT）      ptxas → SASS (sm_XX，真实 ISA)
              │                     │
              └──────────┬──────────┘
                         ▼
                  ★ fatbinary（多份 PTX 和/或 SASS）
                         │
                         ▼
            嵌进宿主 .o / .obj → 可执行文件 / .so / .dll
```

要点：① Host / Device 两条路，设备码以 blob 嵌进宿主对象；② Device 上先 PTX 再 SASS（第 3 章重点）；③ fatbinary 是设备侧多镜像容器。

### 2.2 三种常见设备产物

| 产物 | 大致是什么 | 典型怎么得到 |
|------|-----------|--------------|
| **`.cubin`** | 单一（或少数）真实架构的 SASS 镜像 | `nvcc -cubin …` |
| **`.ptx`** | 文本或嵌入的虚拟 ISA | `nvcc -ptx …`，或 `-gencode …,code=compute_XX` |
| **`.fatbin` / 嵌入的 fatbinary** | **容器**：内含多份 PTX 和/或 cubin | 默认编译进 `.o`；或 `nvcc -fatbin …` |

日常链接进可执行文件时，你往往**看不到独立的 `.fatbin` 文件**——它已经作为节区/数据嵌在 `.o` 里。用 `cuobjdump` 打开的正是这份嵌进去的容器（见[第 5 章](#第-5-章-检视工具cuobjdump--nvdisasm)）。

### 2.3 Fatbin 里到底有什么（概念结构）

不必背二进制布局，记住逻辑结构即可：

```
fatbinary
├── image[0]:  kind=PTX , arch=compute_80 , bytes=…
├── image[1]:  kind=ELF/cubin , arch=sm_80  , bytes=…
├── image[2]:  kind=ELF/cubin , arch=sm_86  , bytes=…
└── image[3]:  kind=PTX , arch=compute_90 , bytes=…   ← 可选：给更新卡留 JIT 退路
```

运行时选择规则（直觉版，细节由驱动实现）：

1. 若有与当前 GPU **完全匹配**的 SASS（`sm_XX`）→ **直接加载**，最快；
2. 否则若有**兼容的**较旧 SASS（在二进制兼容规则允许时）→ 可能复用；
3. 否则若有合适的 **PTX**（`compute_XX`）→ **驱动 JIT** 成当前 GPU 的 SASS，再加载（有首次开销，可缓存）；
4. 都没有 → 加载失败。

> 官方文档把「带 PTX 以兼容未来 GPU」称为一种 **forward compatibility** 策略；「只带 SASS」则偏向 **load 快、体积可控、但不面向未知架构**。

---

## 第 3 章 compute_XX vs sm_XX（重点）

> 根 README §5.4 的第 2 点。这是半天学习里**唯一需要反复咀嚼**的概念对。搞混了后面所有 `-gencode` 都会用错。

### 3.1 一张对照表钉死

| | **`compute_XX`** | **`sm_XX`** |
|--|------------------|-------------|
| 官方称呼 | **Virtual architecture**（虚拟架构） | **Real architecture** / GPU architecture |
| 产物形态 | **PTX**（Parallel Thread Execution），虚拟指令集 | **SASS**，真实机器码（装在 cubin 里） |
| 谁生成 | 设备前端 / 优化器（如 `cicc`） | **`ptxas`** 把 PTX 汇编成 SASS |
| 是否绑死某代硅片 | 否——描述的是**能力集与特性级别** | 是——对应某一代（或某一具体）SM 微架构 |
| 运行时角色 | 可被 **driver JIT** 成当前 GPU 的 SASS | 匹配则**直接加载**，无需再编译 |
| 典型 NVCC 写法 | `arch=compute_80` 或 `code=compute_80` | `code=sm_80` |

记忆口诀：

```
compute_XX  =  "我保证用到的 GPU 特性不超过这一档"  →  输出 PTX
sm_XX       =  "请为这一代真实芯片生成机器码"      →  输出 SASS
```

数字往往对齐（`compute_80` ↔ Ampere 能力档，`sm_80` ↔ A100 一类），但**语义不同**：一个是虚拟 ISA 的目标，一个是真实 ISA 的目标。

### 3.2 流水线上的位置（再钉一次）

```
CUDA C++  ──►  PTX (compute_XX)  ──►  SASS (sm_XX)
              ▲                      ▲
              │                      │
         虚拟架构目标            真实架构目标
         可嵌入 fatbin            可嵌入 fatbin
         可被驱动 JIT ───────────┘
```

两条合法终点：

- **只停在 PTX**：fatbin 里只有 `compute_*` 镜像 → 每台机器首次加载可能 JIT；
- **继续到 SASS**：fatbin 里有 `sm_*` 镜像 → 匹配 GPU 上零 JIT。

也可以**两者都带**：常见生产配置是「为目标卡带 SASS + 再带一份较高档的 PTX 做前向兼容」。

### 3.3 运行时 JIT：没有匹配 SASS 时会发生什么

当进程在某块 GPU 上 `cuModuleLoad*` / 启动 kernel 时，驱动大致做：

```
查询 GPU → compute capability = X.Y（对应某 sm_XY）
    │
    ├─ fatbin 内有 sm_XY（或兼容）的 SASS？ ──是──► 直接映射/加载
    │
    └─ 否 → 有可用的 PTX（compute_XX，且 XX 不超过当前能力）？
              │
              ├─ 是 → JIT PTX → 当前 sm 的 SASS → 加载
              │         （结果可被驱动缓存，避免每次冷启动都付费）
              └─ 否 → 失败（常见报错与 "no kernel image is available" 一类相关）
```

因此：

- **只发 SASS、不发 PTX**：在**清单内的卡**上启动快、行为确定；遇到**更新一代、你没编过的卡**可能直接挂。
- **带 PTX**：新卡有机会靠 JIT「跑起来」——这是 NVIDIA 宣传的 **future-proof / forward compatible** 路径；代价是体积、首次延迟、以及 JIT 路径上偶发的边角差异（生产环境通常仍会为主要 SKU 预编 SASS）。

### 3.4 取舍：什么时候偏 PTX，什么时候偏 SASS

| 策略 | 做法 | 优点 | 代价 |
|------|------|------|------|
| **偏 SASS** | `-gencode arch=compute_80,code=sm_80`（可多条，覆盖 sm_75/80/90…） | 加载快；指令调度按该架构做死 | 体积随架构数线性涨；未知新架构可能无镜像 |
| **偏 PTX** | `code=compute_80`（或再加更高 `compute_*`） | 一份 PTX 覆盖「能力 ≥ 该档」的多代卡（经 JIT） | 首次 JIT 成本；极致性能/新指令可能吃不到 |
| **混合（常用）** | 对主力卡 `code=sm_XX`，再加 `code=compute_YY` | 主力零 JIT + 长尾可跑 | 体积与维护略复杂，但仍是一个二进制 |

工程经验（与官方建议一致的方向）：

1. **为你承诺支持的每一档主要 GPU 编一份 SASS**；
2. **再嵌入至少一份 PTX**（通常取你支持的最高或次高 virtual arch），给「清单外但能力兼容」的卡留退路；
3. 不要迷信「只留一份很老的 PTX 包打天下」——太老的 virtual arch 表达不了新特性，太新的又可能在旧驱动上不可用。

### 3.5 兼容性直觉 + 收束

- 同一 major 内偶有「旧 SASS 跑在新卡」的有限 binary compatibility——**不要拿它替代正确的 `-gencode` 列表**。
- 跨代发货更可靠的杠杆是 **嵌入 PTX + 够新的驱动 JIT**。
- `arch=compute_XX` 决定按哪档虚拟架构生成/检查 PTX；`code=` 决定往 fatbin **塞 PTX、某 sm 的 SASS，还是两者**。

**三句收束**：`compute_*`→PTX，`sm_*`→SASS；无匹配 SASS 可 JIT PTX；发货 = 在体积 / 冷启动 / 架构覆盖之间选混合策略。

---

## 第 4 章 关键 NVCC 标志

### 4.1 `-gencode`：你真正应该用的旋钮

现代推荐写法是显式 `-gencode`（可重复多次）：

```bash
-gencode arch=compute_80,code=sm_80
```

含义拆开：

| 字段 | 含义 |
|------|------|
| `arch=compute_80` | 前端/优化按 **virtual arch 8.0** 生成 PTX（特性上限） |
| `code=sm_80` | 把该 PTX **汇编成 sm_80 的 SASS** 并打进 fatbin |

只要 SASS、不要嵌入 PTX 时，上面这一条就够（对这一档而言）。

**同时嵌入 PTX（供 JIT）**：

```bash
-gencode arch=compute_80,code=compute_80
```

或一条里写多个 `code`：

```bash
-gencode arch=compute_80,code=[sm_80,compute_80]
```

**多架构示例**（两代 SASS + 一份 PTX 兜底）：

```bash
nvcc kernel.cu -c -o kernel.o \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_80,code=compute_80
```

读法：为 Turing / Ampere 各留一份真码；再留 compute_80 的 PTX，让更新的卡有机会 JIT。

### 4.2 简写 `-arch` / `-code`（知道即可）

```bash
nvcc -arch=sm_80 ...           # 常见简写：按 8.0 虚拟架构编，并生成 sm_80 代码
nvcc -arch=compute_80 -code=sm_80,compute_80 ...
```

简写在简单项目里方便，**多架构发货时一律改用多条 `-gencode`**，避免隐式默认害你只带了一种镜像。

### 4.3 直接产出 fatbin / cubin / ptx

```bash
# 只要胖二进制容器（设备码），不链接宿主可执行文件
nvcc -fatbin kernel.cu -o kernel.fatbin \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80

# 只要某一真实架构的 cubin
nvcc -cubin kernel.cu -o kernel.cubin -gencode arch=compute_80,code=sm_80

# 只要 PTX 文本（看虚拟 ISA、或留给自己加载）
nvcc -ptx kernel.cu -o kernel.ptx -gencode arch=compute_80,code=compute_80
```

`-fatbin` 在「只研究设备镜像、不写完整 CUDA 程序」时特别好用：产物小、直接给 `cuobjdump` 看。

### 4.4 链接（半分钟）

默认多 TU 编译时，各翻译单元的设备码各自嵌入 fatbin 段再会合。学习目标只需：**会写 `-gencode` + 会 `cuobjdump`**；`-rdc` / nvlink 属跳过项。

---

## 第 5 章 检视工具：cuobjdump / nvdisasm

> 根 README §5.4 的第 3 点。验收标准不是「会反汇编」，而是**打开任意 `.o` / `.cubin` / `.fatbin`，说出里面有哪些架构镜像**。

### 5.1 角色分工

| 工具 | 干什么 | 你什么时候用 |
|------|--------|--------------|
| **`cuobjdump`** | 列出 / 抽出 fatbin 内的 PTX、ELF/cubin、信息表 | **首选**。看「有哪些 arch」 |
| **`nvdisasm`** | 把 cubin 反汇编成可读 SASS | 可选。只在需要确认「真的是某 sm 的机器码」时 |

本学习路径里，`cuobjdump` 占 90% 时间。

### 5.2 必会命令（复制即用）

假设你有 `kernel.o` 或 `kernel.fatbin`：

```bash
# 列出嵌入的 ELF（cubin）架构 —— 最常用
cuobjdump -lelf kernel.o

# 列出嵌入的 PTX 架构
cuobjdump -lptx kernel.o

# 一把梭：看全部相关信息
cuobjdump -all kernel.o

# 把某架构的 PTX 抽出来（文本）
cuobjdump -ptx kernel.o

# 把 SASS 反汇编出来（需对应镜像存在）
cuobjdump -sass kernel.o
```

对独立 `.fatbin` / `.cubin` 同样适用：

```bash
cuobjdump -lelf kernel.fatbin
cuobjdump -lptx kernel.fatbin
cuobjdump -lelf kernel.cubin
```

**你要在输出里找到的信号**：

- `ELF file ... sm_75` / `sm_80` —— 有对应 SASS；
- `PTX file ... compute_80` —— 有可 JIT 的虚拟镜像；
- 若你写了两条 `-gencode` 却只看到一个 arch → 编译命令没按你想的生效（查是否被简写 `-arch` 覆盖、或文件不是你以为的那份）。

### 5.3 `nvdisasm`（可选一眼）

```bash
# 先抽出或直接针对 cubin
nvdisasm kernel.cubin
# 或
cuobjdump -xelf all kernel.o    # 视版本选项，抽出 ELF 后再 nvdisasm
```

看到函数符号与指令即可停——**不要钻进 SASS 助记符**（根 README 明确跳过）。

### 5.4 和「验收」对齐的一句话

> 打开产物，能指着 `cuobjdump -lelf` / `-lptx` 的输出说：  
> 「这里有 sm_A 和 sm_B 两份 SASS；还有没有 compute_C 的 PTX；运行时会怎么选。」  
> 这一条过关，§5.4 第三点就过关。

---

## 第 6 章 设计模式同构：fatbin ↔ ExecutableVariant

> 这是花半天学 fatbin 的**真正理由**（根 README 原话）：  
> 「它和 IREE 的 `ExecutableVariant`、和『配置组合爆炸』是同一个问题的三种说法。」  
> 精读对照：[`iree-learning-guide.md` §4.7](./iree-learning-guide.md#47-设备代码executable--variant--export)。

### 6.1 并排看结构

**CUDA fatbin（设备镜像层）**

```
一个 CUDA 模块 / 一个嵌入的 fatbinary
  ├─ 镜像：sm_75  SASS
  ├─ 镜像：sm_80  SASS
  └─ 镜像：compute_80 PTX   ← 条件不满足 SASS 时的退路
           │
           ▼
    驱动按 GPU capability 选一份加载（或 JIT）
```

**IREE `hal.executable`（运行时可执行体层）**

```mlir
hal.executable @my_kernel {
  hal.executable.variant @cuda_sm80 target(#hal.executable.target<"cuda", …>) { … }
  hal.executable.variant @cuda_sm90 target(…) { … }
  // 还可按 condition 表达「这台 device 能不能用这份」
}
```

官方对 variant 的定义要点（IREE 学习文档已引）：*多个 target-specific variant 编译期独立降低，运行时表现为**一个** executable（**类似 fat binary**）*。

### 6.2 同构映射表

| 维度 | CUDA fatbin | IREE ExecutableVariant |
|------|-------------|------------------------|
| **问题** | 多 `sm` / 多发货组合 | 多 backend、多 target、多调参实现 |
| **编译期** | 多次 `-gencode` / 多镜像写入容器 | 每个 variant **独立 lowering / codegen** |
| **产物单元** | 一个模块里的多 image | 一个 `hal.executable` 里的多 `variant` |
| **选择键** | GPU compute capability（+ 驱动策略） | `target` + 可选 `hal.executable.condition` |
| **退路** | PTX JIT | condition / export fallback 链 |
| **调用方感知** | 一次 launch，不知有几份镜像 | 一次 lookup/dispatch，不知有几个 variant |
| **失败模式** | 无可用 image | 无可用 variant → 加载失败 |

### 6.3 「配置组合爆炸」是同一个敌人

把问题写抽象一点：

```
配置空间 = 后端 × 架构 × 精度 × tile/schedule × …
```

若「一种配置 = 一份要单独发货的产物」，矩阵不可维护。工业界反复用的解法是：

1. **编译期**：把有限个**高价值变体**做出来；
2. **打包期**：放进**同一个逻辑模块**（fatbin / executable / 插件包）；
3. **运行期**：用**廉价谓词**（capability、device query、condition op）选一个；
4. **可选**：留一条**更可移植、更慢或更通用**的退路（PTX JIT、fallback export、CPU 路径）。

CUDA fatbin 是这个模式在 **NVIDIA GPU ISA 层**的实例；IREE variant 是它在 **ML 运行时可执行体层**的实例。算力网上你要做的，往往是**第三层实例**：跨厂商、跨节点、跨并行策略的变体选择——机制可以抄，对象模型要自己钉。

**一处容易搞混的差异（读 IREE 文档时会撞上）**：结构同构不等于产物相同。

| | NVCC 产出的 fatbin | IREE CUDA 后端（`cuda-nvptx-fb`） |
|--|--------------------|-----------------------------------|
| 容器 | `.o` / 可执行文件里嵌 fatbinary | `.vmfb` 里嵌 FlatBuffer |
| 通常携带 | **SASS（多档）+ PTX 退路** | **PTX 为主**（不预生成多档 SASS） |
| 谁在选 | CUDA 驱动按 compute capability 选镜像 | HAL 按 target/condition 选 variant，PTX 交给驱动 |
| 首次运行代价 | 命中 SASS 则无 JIT | 走驱动 JIT（结果被驱动缓存） |

也就是说：**IREE 的「多变体」变在 backend/target 这一维，NVCC 的「多镜像」变在 `sm_*` 这一维。** IREE 相当于始终走 fatbin 的「PTX 退路」那条路径。理解这点，你才知道在算力网上想做「热配置预编译 + 冷配置 JIT」时，**这两层的变体轴可以叠加**：外层按后端选 variant，内层按架构选镜像。上游 PTX 是怎么从 LLVM 出来的，见 [`llvm-learning-guide.md`](./llvm-learning-guide.md) 第 5 章（NVPTX 后端）。

### 6.4 对照题（答案见附录）

1. fatbin 里一份 `sm_80` 镜像 ≈ IREE 的 ________？  
2. 驱动按 capability 选镜像 ≈ IREE 按 ________ 选 variant？  
3. 嵌入 PTX 供 JIT ≈ IREE 的哪类退路？  
4. 「调用方只看到一个 kernel / executable」对应哪条工程约束？

---

## 第 7 章 与 AI-Infra 的对接点

位置：根 [`README.md`](../README.md) **P2 / 半天 / §5.4**；阶段入口 [`docs/README.md`](./README.md) **阶段 6**。卡住「多后端怎么低成本共存」时，fatbin 是最短的实物教具。

### 7.1 低成本后端切换 + 多变体打包

目标不是为每个后端 fork 运行时，而是：**编译期**为同一算子生成多份实现 → **产物**仍是一个模块（`.vmfb` / so / 服务镜像）→ **运行期**按能力选路并保留退路。IREE 用 variant 已走通（机制见 [`iree-learning-guide.md` §4.7](./iree-learning-guide.md#47-设备代码executable--variant--export)，与算力网目标的对接见该文档第 6 章）；fatbin 让你在更硬的 ISA 层亲手看到同一结构。

| 变体轴 | 例子 | 类似 fatbin 的哪一维 |
|--------|------|----------------------|
| 硬件 ISA | sm_80 vs sm_90；CUDA vs ROCm | `sm_*` 多镜像 |
| 可移植表示 | PTX / SPIR-V | `compute_*` / 中间 ISA |
| 调参实现 | 不同 tile / warp 特化 | 多 SASS 或多 variant |
| 部署约束 | 包体积 vs 冷启动 | 嵌多少档 SASS / 是否留 PTX |

评审时可问：承诺 SKU 列表内是否必须 AOT 真码？列表外是否允许 JIT？选择谓词是否可测可日志？调用方 API 是否稳定到看不见变体数？

### 7.2 串回主线

```
分布式（切什么）→ MLIR（怎么表达）→ IREE variant（设备多实现）
    → ★ fatbin（多 ISA 镜像实物）→ 你的跨厂商打包层
```

---

## 第 8 章 动手清单（半天做完）

> 环境：已安装 CUDA Toolkit，`nvcc`、`cuobjdump` 在 `PATH` 中。架构号可按你机器改；**关键是两条 `-gencode` 能在 dump 里看见两个 arch**。

### 8.0 推荐：仓库动手项目

```bash
cd tvm-fatbin-lab
bash scripts/run_fatbin.sh          # 或 bash scripts/run.sh 两轨一起
# 先读 out/cuda/READING.md 与 out/ANALYSIS.md
```

源码在 [`../tvm-fatbin-lab/cuda/add.cu`](../tvm-fatbin-lab/cuda/add.cu)；脚本会生成 SASS-only 与「SASS + PTX 退路」两份 fatbin，并落盘 `cuobjdump` 输出。

### 8.1 最小复现（不跑 lab 时）

```bash
nvcc -fatbin tvm-fatbin-lab/cuda/add.cu -o add.fatbin \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80
cuobjdump -lelf add.fatbin
cuobjdump -lptx add.fatbin
```

再编一份带 PTX 退路的，对比 `-lptx`：

```bash
nvcc -fatbin tvm-fatbin-lab/cuda/add.cu -o add_with_ptx.fatbin \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=[sm_80,compute_80]
```

**过关标准**：

- [ ] `-lelf` 能看到 **两个**不同的 `sm_*`；  
- [ ] SASS-only 的 `-lptx` 为空或无对应档；with_ptx 能看到 `compute_*`；  
- [ ] 能解释：无 PTX 时在更新 GPU 上可能失败；有 `compute_*` 时可 JIT。

### 8.2（可选）IREE 对照

对小模型 `--compile-to=hal`，找 `hal.executable` / `variant` / `export`，对照 fatbin 多 image；问 `target` ≈ 哪条 `-gencode`，`condition` ≈ 更丰富的选择谓词。

**时间盒**：0–20 min 读 §1–3 → 20–40 min 跑 lab → 40–60 min 对照 IREE §4.7 写三句「问题 / 打包 / 选择」。

---

## 第 9 章 必学三点 vs 全部跳过

与根 README **§5.4** 严格对齐。

### 9.1 必须掌握（只有这三点）

1. **fatbin 的结构**：一个二进制（容器）里如何同时携带多个架构的代码（多份 PTX 和/或 SASS image）。  
2. **`compute_XX`（PTX，虚拟架构）vs `sm_XX`（SASS，真实架构）**，以及无匹配 SASS 时驱动 **JIT PTX→SASS** 的兜底与取舍。  
3. 用 **`cuobjdump` / `nvdisasm`** 查看 `.o` / `.cubin` / `.fatbin` 里到底有哪些架构——以 `cuobjdump -lelf` / `-lptx` 为主。

外加一条**模式层**验收（README「为什么值得花这半天」）：能口述 fatbin 与 IREE `ExecutableVariant` 的同构，以及二者如何缓解配置组合爆炸。

### 9.2 明确跳过

| 跳过项 | 原因 |
|--------|------|
| SASS 指令集细节、流水线冒险、寄存器压力调优 | 那是 GPU 微架构/性能工程师主业，不是多变体打包课 |
| 手写 PTX 编程、内联 PTX 方言 | 你消费 PTX 作为可移植镜像即可 |
| 完整 CUDA Runtime / Driver API 教程 | 只需知道「加载模块时驱动会选镜像 / JIT」 |
| `nvlink`、深度 RDC、CUDA 图、统一内存细节 | 与 fatbin 学习目标无关 |
| 每一代 `sm` 的特性表背诵 | 需要时查附录/官方 arch 表 |

### 9.3 过关自问

1. 为什么「一卡一包」不可扩展？fatbin 消灭的是哪一维矩阵？  
2. 解释 ` -gencode arch=compute_80,code=sm_80` 与 `code=compute_80` 差在产物的哪一层。  
3. 打开一份陌生的 `.so`，你用哪两条命令证明它带了哪些 GPU 镜像？  
4. 用一张四行表对照 fatbin image 与 `hal.executable.variant`（问题、编译、选择、退路）。

答得出来，半天结束，回到主线。

---

## 附录：一页速查

```
【问题】   多 sm × 多软件版本 → N×M 发货爆炸
【解法】   fatbinary：一容器多镜像；运行时按 GPU 能力选择（或缺 SASS 则 JIT PTX）

【两层 ISA】
  compute_XX → PTX（虚拟架构，可 JIT）
  sm_XX      → SASS（真实架构，直接加载）

【流水线】
  .cu → NVCC → host 目标码
            ↘ device: CUDA C++ → PTX (compute_*) → SASS (sm_*) → 打进 fatbin

【关键 flag】
  -gencode arch=compute_80,code=sm_80          嵌入 SASS
  -gencode arch=compute_80,code=compute_80     嵌入 PTX
  -gencode arch=compute_80,code=[sm_80,compute_80]
  -fatbin / -cubin / -ptx

【检视】
  cuobjdump -lelf  file.o|.fatbin|.cubin     有哪些 SASS
  cuobjdump -lptx  file…                     有哪些 PTX
  cuobjdump -sass / nvdisasm                 可选，看真码即可停

【同构】
  CUDA fatbin image     ↔  IREE hal.executable.variant
  driver 按 capability 选 ↔  target + condition 选
  PTX JIT 退路          ↔  fallback / 可移植 variant
  详见 iree-learning-guide.md §4.7

【必学三点】结构 · compute vs sm + JIT · cuobjdump
【跳过】    SASS 细节 · 手写 PTX
```

**§6.4 填空参考答案**

1. `hal.executable.variant`  
2. `target` + 可选 `condition`（第一个可用者）  
3. 更可移植的 variant / export fallback（或保留中间表示类产物）  
4. 宿主/调用方 API 稳定：一次 launch 或一次 executable lookup，变体数量是实现细节  

---

*文档版本随仓库自学体系维护；优先级与「只学三点」以根 README §5.4 为准。阶段入口：[`docs/README.md`](./README.md)。*
