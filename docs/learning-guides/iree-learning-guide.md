# IREE 学习文档：核心概念全景 + HAL 深入详解

> **本文档的定位**
> - 基于 IREE **官方文档**（https://iree.dev）与 **主干源码**（`iree-org/iree`）整理，不是论文笔记。
> - 目标读者是"要在算力网上搭分布式大模型执行基础设施"的编译器/系统工程师。
> - **前 3 章建立坐标系（约 30% 篇幅），第 4 章是重点：IREE HAL 详解（约 55% 篇幅）**，第 5–7 章是落地与学习路径。
> - TinyIREE 论文的简版印象见 [`paper-notes/07-tinyiree.md`](../paper-notes/07-tinyiree.md)；那篇讲得很浅，本文才是正式材料。
>
> **先修与邻接材料**
> - **先修**：[`ai-compiler-foundations.md`](./ai-compiler-foundations-learning-guide.md) §3.3（子图划分）、§7.1（kernel）、§8（设备抽象机与同步）。
> - **上游**：`linalg → flow` 之前的 MLIR 机制见 [`mlir-learning-guide.md`](./mlir-learning-guide.md)。
> - **下游**：variant 里的机器码怎么来见 [`llvm-learning-guide.md`](./llvm-learning-guide.md) 第 5 章；多变体打包的 ISA 层同构物见 [`cuda-fatbin-learning-guide.md`](./cuda-fatbin-learning-guide.md)。
> - **目标场景**：分布式训练侧的需求来自 [`paper-notes/01-efficient-training-distributed-infra.md`](../paper-notes/01-efficient-training-distributed-infra.md)。
> - **划分对照（导出期 / Session 期）**：[`../../onnx-delegate-lab/`](../../onnx-delegate-lab/) —— 与 IREE 编译期 `flow.dispatch` 对照「决策时刻」差在哪。
>
> **一句话读法**：如果只有两小时，读[第 1 章总图](#第-1-章-iree-是什么一分钟建立坐标系)、[第 2 章 dialect 流水线](#第-2-章-编译侧dialect-流水线)、[4.2 对象模型总览](#42-对象模型总览)、[4.6 timeline semaphore](#46-同步timeline-semaphore-与-fencehal-的灵魂)、[4.7 variant](#47-设备代码executable--variant--export)、[附录速查](#附录一页速查)。
>
> **主要信息源**
> - HAL 设计意图：`runtime/src/iree/hal/README.md`
> - 执行模型：`developers/design-docs/invocation-execution-model.md`
> - HAL 抽象与硬件的落差：`developers/design-docs/cuda-hal-driver.md`（**最值得精读的一篇**）
> - HAL dialect 全量 op/attr/enum：https://iree.dev/reference/mlir-dialects/HAL/
> - 部署配置：`guides/deployment-configurations/`
> - HAL 运行时 C API：`runtime/src/iree/hal/*.h`

---

### 本篇在链路中的位置

> 全局链路见 [`00-end-to-end-pipeline.md`](./00-end-to-end-pipeline.md)。本篇横跨**第 ⑤ 站（多层降低）到第 ⑦ 站（打包与运行时）**，是链路上唯一一个"从 linalg 一直管到真跑起来"的系统。

```text
③ 融合 ──▶ ④ 调度 ──▶ ⑤ 多层降低 ──▶ ⑥ 指令生成 ──▶ ⑦ 打包与运行时
                        └────────── 本篇 ──────────┘        └── 本篇 ──┘
                         flow / stream / hal 三层         .vmfb + HAL 运行时
```

| | |
|--|--|
| **上游交给我** | 一份 `linalg` 层的模型（知道"这是矩阵乘"、"这两维可并行"） |
| **我固化** | kernel 边界（`flow`）、内存生命周期与时序（`stream`）、设备对象与多变体（`hal`）、宿主调度指令（`vm`） |
| **我交给下游** | 一个 `.vmfb` —— 设备码 + 宿主调度字节码 + 常量，**单文件即可部署** |
| **本篇的主角** | [`../../iree-lab/models/tiny_mlp.mlir`](../../iree-lab/models/tiny_mlp.mlir)：全仓库同一个 `Gemm→Relu→Add`，这次要真编真跑 |

**本篇示例的来源约定**：凡是标了「**可跑**」的代码块，都能在 [`../../iree-lab/`](../../iree-lab/) 里复现，产物路径写在示例旁边。

| 章节 | 示例来自 | 怎么跑 |
|------|---------|--------|
| 第 1–2 章（相位） | `iree-lab/models/{abs,tiny_mlp}.mlir` | `bash scripts/run_phases.sh` → `out/phases/`、`out/PHASES.md` |
| 第 3、5 章（执行与部署） | 同上 | `bash scripts/run_execute.sh` → `out/execute/*.vmfb` |
| 第 4 章（HAL 对象模型） | 编译产物 + 运行时 C API | `run_phases.sh` 的 hal 相位 + 本篇的 C 侧伪码 |
| 第 4.7 / 5 章（变体与开关） | 三组对照实验 | `bash scripts/run_variants.sh` → `out/variants/` |

> 第 4 章讲 HAL 运行时 C API 的部分**没有 lab 代码可指** —— 那是 C 语言的宿主侧接口，
> 要写就得引入一个 CMake 工程，与本仓库"pip 装完即可跑"的取向冲突。
> 所以那部分保持伪码，但**每个对象都会指出它在 `out/phases/tiny_mlp.hal.mlir` 里对应的那一行**，
> 让你至少能看到编译器为它生成了什么。

---

## 目录

- [第 1 章 IREE 是什么：一分钟建立坐标系](#第-1-章-iree-是什么一分钟建立坐标系)
- [第 2 章 编译侧：dialect 流水线](#第-2-章-编译侧dialect-流水线)
- [第 3 章 运行侧：执行模型（HAL 的上层语境）](#第-3-章-运行侧执行模型hal-的上层语境)
- [**第 4 章 IREE HAL 详解**](#第-4-章-iree-hal-详解重点)
  - [4.1 HAL 是什么：五项职责与设计取向](#41-hal-是什么五项职责与设计取向)
  - [4.2 对象模型总览](#42-对象模型总览)
  - [4.3 设备与内存：driver / device / allocator / buffer / buffer_view](#43-设备与内存driver--device--allocator--buffer--buffer_view)
  - [4.4 工作表达：command_buffer](#44-工作表达command_buffer)
  - [4.5 工作提交：device queue API](#45-工作提交device-queue-api)
  - [4.6 同步：timeline semaphore 与 fence（HAL 的灵魂）](#46-同步timeline-semaphore-与-fencehal-的灵魂)
  - [4.7 设备代码：executable / variant / export](#47-设备代码executable--variant--export)
  - [4.8 集合通信与多设备：channel / topology](#48-集合通信与多设备channel--topology)
  - [4.9 编译器侧的 HAL dialect](#49-编译器侧的-hal-dialect)
  - [4.10 HAL 的三种形态：Full / Inline / Loader](#410-hal-的三种形态full--inline--loader)
  - [4.11 实现一个新 HAL driver 的清单](#411-实现一个新-hal-driver-的清单)
- [第 5 章 部署配置与命令行](#第-5-章-部署配置与命令行)
- [第 6 章 与"算力网分布式基础设施"的对接点](#第-6-章-与算力网分布式基础设施的对接点)
- [第 7 章 学习路径：最小必要集与动手清单](#第-7-章-学习路径最小必要集与动手清单)
- [附录：一页速查](#附录一页速查)

---

## 第 1 章 IREE 是什么：一分钟建立坐标系

**IREE = IREE's an Retargetable Execution Environment**，一套基于 MLIR 的**端到端 ML 编译器 + 运行时**。

一句话记住它和别人的区别：

> 传统 ML 运行时把模型当作**数据**（图 / 算子序列），运行时去解释它；
> **IREE 把模型当作程序去编译**——产物里既有"算什么"（设备侧 kernel），也有"怎么调度"（宿主侧的分配、拷贝、派发、同步指令序列）。

由此推出 IREE 几乎所有的设计决定：

| 决定 | 原因 |
|------|------|
| AOT 编译为主，产物是 `.vmfb` | 调度逻辑已经编译死了，运行时不需要图解释器 |
| 运行时刻意做得"薄而笨" | 智能全部前移到编译期。运行时只是把编译器算好的指令翻译给驱动 |
| 需要一层 HAL | 编译器要能对"设备"做统一的假设，才能把调度编译出来 |
| HAL 长得像 Vulkan | 因为要**显式控制**（显式分配、显式命令录制、显式同步），才能把编译期的调度决定原样表达出来 |
| 能缩到几十 KB，也能扩到数据中心 | 同一条编译流水线，末端换不同的运行时组装方式 |

### 三个必须先分清的层次

```
┌───────────────────────────────────────────────────────────┐
│  编译器（iree-compile）                                    │
│  输入 dialect → Linalg → Flow → Stream → HAL → VM          │
│                                    └─ codegen → PTX/SPIR-V/ELF │
└───────────────────────────────────────────────────────────┘
                          │ 产物 .vmfb（FlatBuffer）
                          ▼
┌───────────────────────────────────────────────────────────┐
│  VM（宿主侧虚拟机）                                        │
│  执行编译出来的调度指令：调用 HAL 模块的导出函数            │
└───────────────────────────────────────────────────────────┘
                          │ HAL 是 VM 的一个内置 module
                          ▼
┌───────────────────────────────────────────────────────────┐
│  HAL（硬件抽象层）                                         │
│  driver / device / buffer / command_buffer / semaphore ... │
│  └─ 具体实现：local-sync / local-task / cuda / hip / vulkan / metal │
└───────────────────────────────────────────────────────────┘
```

**关键认知**：在 IREE 里 **HAL 本身就是一个 VM module**（就像 `math` 模块一样）。编译出来的程序调用 `hal.device.queue.execute` 这类函数，和调用一个普通函数没有区别。这样做的直接好处是：**用户可以替换掉 HAL 模块而不用改 IREE 核心代码**。官方文档原话是"在 IREE 里一切都是 module，包括 HAL 这样的运行时子系统"。

#### 示例精讲：一个 `abs` 模型穿过三层

**可跑** · 源码 [`iree-lab/models/abs.mlir`](../../iree-lab/models/abs.mlir)

```mlir
func.func @abs(%input : tensor<f32>) -> tensor<f32> {
  %result = math.absf %input : tensor<f32>
  return %result : tensor<f32>
}
```

**为什么用一个只有一个 op 的模型开场**：它小到每一相位 dump 出来都能整篇看完，
所以适合回答"这一相**新增**了什么"。等到第 2.2 节要看**融合**时会换成
[`tiny_mlp.mlir`](../../iree-lab/models/tiny_mlp.mlir) —— 单个 op 演示不了"几个 op 合成一个 kernel"。
lab 里两个模型并存，就是这个分工。

```bash
cd iree-lab && bash scripts/run_execute.sh
# 内部执行的就是这两条：
#   iree-compile --iree-hal-target-device=local \
#                --iree-hal-local-target-device-backends=llvm-cpu \
#                models/abs.mlir -o out/execute/abs.vmfb
#   iree-run-module --device=local-sync --module=out/execute/abs.vmfb \
#                   --function=abs --input=f32=-2.0
```

> **标志写法会变**：`--iree-hal-target-device=` 是新版写法，老版本是
> `--iree-hal-target-backends=llvm-cpu`。`iree-lab/scripts/env.sh` 会**实际试一次**再决定用哪套，
> 所以脚本跨版本不用改；你手敲时报 unknown option，换另一套即可。

**① 编译器产物 `abs.vmfb` 里有什么**

```text
abs.vmfb（FlatBuffer）
├─ 依赖表   需要哪些 VM module 才能链接 —— 这里是内置的 hal
├─ 导出表   @abs                        ← --function=abs 查的就是这张表
├─ 函数体   宿主侧调度指令的 bytecode：建命令缓冲 / 派发 / 等 fence
└─ rodata   设备侧代码：llvm-cpu → ELF，cuda → PTX
```

一句话：**「怎么调度」进了 bytecode，「算什么」进了 rodata**。

`run_execute.sh` 第 ④ 步会 `strings` 一遍 `.vmfb`，你能在里面看到 `llvm-cpu`、
目标三元组、`tiny_mlp_dispatch_0` 这类字符串 —— 那就是 rodata 段里设备码的痕迹。

**② VM 层：bytecode 去调 HAL 模块的导出函数**

```mlir
vm.module @module {
  // HAL 是一个普通 module，它的每个 op 就是一个可调用的导入符号
  vm.import private @hal.command_buffer.create(...) -> !vm.ref<!hal.command_buffer>
  vm.import private @hal.device.queue.execute(...)
  vm.import private @hal.fence.await(...) -> i32

  vm.rodata private @abs_dispatch_0_binary ...        // 设备代码就躺在这

  vm.func private @abs(%view: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view> {
    %cmd = vm.call @hal.command_buffer.create(...) : (...) -> !vm.ref<!hal.command_buffer>
    vm.call @hal.command_buffer.dispatch(...)
    vm.call @hal.device.queue.execute(...)
    %status = vm.call @hal.fence.await(...) : (...) -> i32
    vm.return ...
  }
}
```

> 形态示意：`vm.import` 的参数列表很长，具体拼写以你本地 `--compile-to=vm` 的 dump 为准。这里要看的是结构：**调 HAL 和调普通函数没有区别**。

**③ 运行时：`iree-run-module` 的执行路径**

```text
iree-run-module --device=local-sync --module=abs.vmfb --function=abs
└─ 建 VM instance
   └─ driver_registry 按 "local-sync" 找 driver → 建 device        【HAL】
      └─ 把内置 hal module（已绑定该 device）与 abs.vmfb 链接成 context
         └─ 查导出表拿到 @abs，把 --input 解析成 buffer_view，invoke
            └─ VM 逐条解释 bytecode
               ├─ vm.call @hal.command_buffer.create   → iree_hal_command_buffer_create()
               ├─ vm.call @hal.command_buffer.dispatch → iree_hal_command_buffer_dispatch()
               ├─ vm.call @hal.device.queue.execute    → iree_hal_device_queue_execute()
               └─ vm.call @hal.fence.await             → 等到时间线推进
                  └─ local-sync：当前线程直接跳进 rodata 里的 ELF kernel
```

| | 编译器 | VM | HAL |
|--|--------|----|-----|
| 输入 | `.mlir` | `.vmfb` 里的 bytecode | VM 传来的对象与参数 |
| 干的事 | **决定**调度 | **复述**调度 | **翻译**给驱动 |
| 智能程度 | 全部智能在这 | 无优化 | 薄，无调度决策 |
| 换后端要改什么 | target backend | 不用改 | 写一个新 driver |

> **自测**：设备侧 kernel 存在 `.vmfb` 的哪一部分，宿主侧的派发顺序又存在哪一部分？

---

## 第 2 章 编译侧：dialect 流水线

`iree-compile` 的相位（phase）在源码 `compiler/src/iree/compiler/Pipelines/Pipelines.h` 的 `IREEVMPipelinePhase` 里定义，可以用 `--compile-to=<phase>` 在任意一相停下来看 IR：

```
start → input → abi → preprocessing → global-optimization → dispatch-creation
      → flow → stream
      → executable-sources → executable-configurations → executable-targets
      → hal → vm → end
```

### 2.1 各层职责表

| 层 | 回答的问题 | 核心产物 |
|----|-----------|---------|
| **输入 dialect**<br>StableHLO / TOSA / Torch / ONNX | 模型是什么 | 框架语义的张量算子 |
| **Linalg**（+ LinalgExt / TensorExt / Encoding） | 计算的循环结构是什么 | `linalg.generic`：`iterator_types` + `indexing_maps` + payload |
| **Flow** | 哪些计算打包成一次设备调用；张量在程序里怎么流动 | **dispatch region**：必须在设备上原子执行的一段代码 |
| **Stream** | 这些 dispatch 按什么时序执行；资源什么时候分配/释放；跑在哪个设备上 | 异步执行时间线 + 资源生命周期 |
| **HAL** | 用哪个设备对象、哪块 buffer、哪个 executable、哪个 fence 来完成它 | 具体的设备命令序列 |
| **VM** | 宿主侧的控制流长什么样 | 可序列化为 bytecode 或 EmitC C 源 |
| **Codegen**（并行分支）<br>Linalg → Vector → LLVM / SPIR-V | dispatch region 内部怎么算 | PTX / HSACO / SPIR-V / ELF |

`iree.dev` 上还列了一批"内部 dialect"，值得知道名字：`Util`（跨 dialect 通用类型与 op）、`IREECodegen` / `IREEGPU` / `IREECPU` / `IREEVectorExt`（codegen 侧）、`VMVX`（可移植解释后端）、`HAL/Inline` 与 `HAL/Loader`（HAL 的精简变体，见 [4.10](#410-hal-的三种形态full--inline--loader)）、`IO/Parameters`（外部参数/权重加载）、`PCF`（并行控制流建模）。

#### 示例精讲：用 `--compile-to` 逐相位盯 op

**可跑** · 脚本 [`iree-lab/scripts/run_phases.sh`](../../iree-lab/scripts/run_phases.sh) · 产物 `iree-lab/out/phases/`

```bash
cd iree-lab && bash scripts/run_phases.sh
# 对 abs.mlir 和 tiny_mlp.mlir 各跑一遍下面这个循环：
#   for phase in input abi preprocessing global-optimization dispatch-creation \
#                flow stream executable-sources executable-targets hal vm; do
#     iree-compile <target-flags> --compile-to=$phase models/$m.mlir -o out/phases/$m.$phase.mlir
#   done
```

脚本第一件事是打一张**行数表**：

```text
phase                    abs行数    mlp行数    abs状态    mlp状态
------------------------ --------  --------  --------  --------
input                    ...       ...       ok        ok
flow                     ...       ...       ok        ok
stream                   ...       ...       ok        ok
hal                      ...       ...       ok        ok
vm                       ...       ...       ok        ok
```

**行数本身就是结论**：越往下走，同一份语义要写的字越多。多出来的字全是
**「原来隐含、现在必须说清」**的东西 —— 谁分配内存、谁等谁、跑在哪个设备上。
这也是理解"分层"最省力的入口：每一层不是包装，是**补信息**。

> 脚本会自动跳过当前版本不支持的相位名（相位表历年有增删），跳过的记 `skip` 而不是报错。

打开每个文件时**只找下表里的东西**：找到了，就说明这一相位该干的活落地了。

| 相位 | 该出现的关键 op / 类型 | 读法 |
|------|---------------------|------|
| `input` | 输入方言原样：`stablehlo.*` / `tosa.*` / `torch.*`，纯 `func.func` | 还没有任何 IREE 概念 |
| `abi` | `hal.tensor.import` / `hal.tensor.export` | **边界建立**：外面是 `!hal.buffer_view`，里面是 `tensor` |
| `global-optimization` | `util.global`、融合后的 `linalg.generic` | 权重变 global；算子按结构融合 |
| `dispatch-creation` | `flow.dispatch.region` | 「哪几段打包成一次设备调用」已定，但还没外提 |
| `flow` | `flow.executable` / `flow.executable.export` / `flow.dispatch` / `!flow.dispatch.tensor` | dispatch 被**外提成独立单元** |
| `stream` | `stream.cmd.execute` / `stream.cmd.dispatch` / `stream.resource.alloca` / `!stream.resource<transient>` / `!stream.timepoint` | 出现**时序与生命周期** |
| `executable-sources` | `hal.executable` + `hal.executable.variant`，内部还是设备无关 IR | executable 骨架成型，未选实现策略 |
| `executable-configurations` | variant 内部多出 codegen 配置属性（tile size、translation 策略等） | 「怎么算」的决定 |
| `executable-targets` | variant 内部已降到 LLVM / SPIR-V dialect | 设备代码定型 |
| `hal` | `hal.command_buffer.create` / `.dispatch` / `.finalize`、`hal.device.queue.execute`、`hal.fence.*` | 只剩驱动调用 |
| `vm` | `vm.module` / `vm.func` / `vm.call @hal.*` / `vm.rodata` | 宿主程序的最终形态 |

> 形态示意：`executable-configurations` 里的配置属性名各版本差异较大，以本地 dump 为准。表里要记的是「**这一相位新增了哪一类信息**」。

三个最值得直接 `diff` 的跃迁：`flow → stream`（多出时序）、`stream → hal`（抽象资源变设备对象）、`hal → vm`（op 变函数调用）。

`run_phases.sh` 第 ⑥ 步会替你做第一个 diff，只留新增行：

```bash
diff out/phases/abs.flow.mlir out/phases/abs.stream.mlir | grep '^>'
```

一个 `math.absf` 而已，stream 层却多出十几行 —— **多出来的全是内存与时序**。
这是"抽象层不是包装，是补信息"最短的一份证据。

> **自测**：如果 `abs.stream.mlir` 里找不到 `stream.resource.alloca`，说明这个程序的中间结果处于什么状态？

### 2.2 三个最需要理解的转折点

**转折点 1：Linalg → Flow，形成 dispatch region**

Linalg 的价值在于用 `iterator_types`（parallel / reduction）和 `indexing_maps`（仿射映射）**描述计算的结构而不是语义**。这带来两个能力：

- **融合不需要理解算子**。判断两个生产者-消费者算子能否融合，只看这两组元信息。若在 TOSA/StableHLO 层做，就会陷入"N 个算子两两组合"的排列爆炸。
- **tiling 也只看这两组元信息**。切出来的 tile 被封装成 dispatch region。

**dispatch region 是后面所有调度的最小单位**——记住这一点，Stream 和 HAL 两层都是围绕它做文章。

**这句话可以直接数出来** · 模型 [`iree-lab/models/tiny_mlp.mlir`](../../iree-lab/models/tiny_mlp.mlir)

主角模型里写了 **4 个 `linalg.generic`**（广播 `b` / Gemm / Relu / Add bias2）：

```bash
cd iree-lab && bash scripts/run_phases.sh
grep -c '= flow.dispatch ' out/phases/tiny_mlp.flow.mlir
```

**数出来少于 4，就说明融合发生了。** Relu 和 Add 的 `iterator_types` 全是 `parallel`、
`indexing_maps` 是恒等映射 —— 编译器据此判定它们可以并进上游 Gemm 的输出循环，
于是 `h` 和 `r` 这两张中间张量**根本不需要落 DRAM**。

注意编译器**从头到尾没有"认识" Relu 是什么**。它看的只是那两组元信息。
这正是 [`00-end-to-end-pipeline.md`](./00-end-to-end-pipeline.md) 站 ③ 那条优化目标
（"Gemm→Relu→Add 只碰一次 DRAM"）在 IREE 里的兑现方式 ——
而在 TVM 里同一个决定是靠 [`OpPatternKind` 分类](./tvm-learning-guide.md)做的，
两套系统的答案不同，问题却是同一个。

**转折点 2：Flow → Stream，从"数据流"到"执行流"**

Stream 是 IREE 最独特、也最容易被忽略的一层。它做四件事：

1. **分区（partitioning）**：把 dispatch 划分成一批可并发执行的分组。`--iree-stream-partitioning-favor=` 可以在 `max-concurrency`（默认，追吞吐）和 `min-peak-memory`（追内存，单线程/嵌入式场景用）之间取舍。
2. **异步时间线建模**：显式表达"这批工作等待哪个时间点、完成后推进到哪个时间点"。
3. **资源生命周期分类**：每个资源被标注 `Lifetime`——`constant`（常量）/ `variable`（跨调用状态）/ `transient`（单次调用内的临时量）/ `staging`（host-device 中转）/ `external`（用户传入）。这个分类直接决定了后面 HAL 层怎么分配内存。
4. **设备归属（affinity）**：多设备场景下决定每段工作跑在哪个设备/队列上。

**Stream 决定了 IREE 的峰值内存表现**：绝大部分内存要么是常量、要么是 `transient`。`transient` 内存从一个"感知流式行为"的池子里预留，**只为当前真正并发的那部分工作提交物理内存**。于是一个已加载但空闲的程序，静止内存占用可以低到几 KB（常量另算，且常量常可映射为可丢弃内存）。这就是所谓 **stream-ordered allocation**，第 3 章会展开。

**转折点 3：Stream → HAL，从抽象资源到具体设备对象**

Stream 里的"某个资源"变成 `!hal.buffer`；"某批工作"变成 `!hal.command_buffer` 里的一串命令；"某个时间点"变成 `!hal.fence`；"某段 kernel"变成 `hal.executable` 里的一个 export。这一层往下就没有抽象了，只剩下驱动调用。

#### 示例精讲：`abs` 在 linalg / flow / stream / hal 的四张面孔

**可跑** · 产物 `iree-lab/out/phases/abs.{flow,stream,hal}.mlir`（`bash scripts/run_phases.sh` 生成）

同一个函数、四段骨架 IR，每段只看**新增了什么、消失了什么**。

**① Linalg：只有计算结构**

```mlir
%init = tensor.empty() : tensor<f32>
%0 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>],
                     iterator_types = []}
     ins(%arg0 : tensor<f32>) outs(%init : tensor<f32>) {
^bb0(%in: f32, %out: f32):
  %1 = math.absf %in : f32
  linalg.yield %1 : f32
} -> tensor<f32>
```

新增 `iterator_types` + `indexing_maps`（融合与 tiling 的全部依据）；设备、内存、时序一概没有。

**② Flow：计算被打包成 dispatch**

```mlir
flow.executable private @abs_dispatch_0 {
  flow.executable.export public @abs_dispatch_0_generic
  builtin.module {
    func.func @abs_dispatch_0_generic(%in : !flow.dispatch.tensor<readonly:tensor<f32>>,
                                      %out : !flow.dispatch.tensor<writeonly:tensor<f32>>) {
      // load → linalg.generic → store
    }
  }
}
// 主函数里只剩一次调用
%0 = flow.dispatch @abs_dispatch_0::@abs_dispatch_0_generic(%arg0) : (tensor<f32>) -> tensor<f32>
```

新增 `flow.executable` / `flow.dispatch` / `!flow.dispatch.tensor`（读写方向已声明）；算子体**搬离主函数**，主函数降级为「数据怎么流」。

**③ Stream：出现时间线与生命周期**

```mlir
%alloc, %t0 = stream.resource.alloca uninitialized : !stream.resource<transient>{%sz} => !stream.timepoint
%t1 = stream.cmd.execute await(%t0) => with(%in as %i : !stream.resource<external>{%sz},
                                            %alloc as %o : !stream.resource<transient>{%sz}) {
  stream.cmd.dispatch @abs_dispatch_0::@abs_dispatch_0_generic {
    ro %i[%c0 for %sz] : !stream.resource<external>{%sz},
    wo %o[%c0 for %sz] : !stream.resource<transient>{%sz}
  }
} => !stream.timepoint
```

新增 `await(...) => !stream.timepoint`（时序）、`<external>` / `<transient>`（生命周期）、`ro` / `wo`（访问权限）；`tensor` 消失，变成带字节长度的 `!stream.resource`。

**④ HAL：全是设备对象**

```mlir
%cmd = hal.command_buffer.create device(%device : !hal.device)
           mode("OneShot") categories("Transfer|Dispatch") : !hal.command_buffer
hal.command_buffer.dispatch<%cmd : !hal.command_buffer>
    target(%exe : !hal.executable)[%c0]
    workgroups([%c1, %c1, %c1])
    bindings([(%in_buf : !hal.buffer)[%c0, %sz], (%out_buf : !hal.buffer)[%c0, %sz]])
    flags(None)
hal.command_buffer.finalize<%cmd : !hal.command_buffer>
hal.device.queue.execute<%device : !hal.device>
    affinity(%affinity) wait(%wait) signal(%signal) commands(%cmd) flags(None)
```

新增 `!hal.device` / `!hal.buffer` / `!hal.executable` / `!hal.fence` 与 3D workgroup 数；`!stream.resource` 的生命周期语义消失——它已在编译期兑现成具体 buffer 与具体的分配位置。

> 形态示意：四段都做了删减，operand / attribute 的准确拼写以本地 `--compile-to=` dump 为准。

| 层 | 数据长什么样 | 时序怎么表达 | 谁来执行 |
|----|-------------|-------------|---------|
| Linalg | `tensor` | 无（SSA 依赖即顺序） | 未定 |
| Flow | `tensor` + `!flow.dispatch.tensor` | 无（仍是数据流） | 未定 |
| Stream | `!stream.resource<lifetime>{字节数}` | `!stream.timepoint` 的 await / produce | 某个 affinity |
| HAL | `!hal.buffer` / `!hal.buffer_view` | `!hal.fence` 的 wait / signal | 具体 `!hal.device` |

**在主角模型上把生命周期数一遍**：

```bash
grep -o '!stream.resource<[a-z]*>' out/phases/tiny_mlp.stream.mlir | sort | uniq -c
```

`external` 是跨边界、宿主看得见的（不能随便复用），`transient` 是一次执行内的中间量
（**可以和别人共享同一块内存**），`variable` 跨执行存活。
`transient` 数量越多，说明编译器识别出的可复用内存越多 —— 这个数字直接决定峰值内存，
[3.4 节](#34-stream-ordered-allocation峰值内存的关键)会展开。

> **自测**：从 flow 到 stream，`tensor<f32>` 为什么必须变成带显式字节长度的 `!stream.resource`？

---

## 第 3 章 运行侧：执行模型（HAL 的上层语境）

官方设计文档 `invocation-execution-model.md` 的第一句话很关键：

> 整个 IREE 栈遵循**同一个执行模型**，本质上就是**乱序执行（out-of-order execution）**算法的一次实例化——和 1960 年代的 Tomasulo 算法同源。

也就是说，用户看到的 API 语义、Stream dialect 的调度语义、HAL 的队列语义，**是同一套东西在三个层次上的复述**。理解了这一节，HAL 的同步模型就是自然结果。

### 3.1 六个基本概念

| 概念 | 定义 | 传统 ML 运行时的对应物 |
|------|------|---------------------|
| **Program** | 一组 module 在一个 context 里实例化后的整体 | —— |
| **Module** | 可加载、可链接、可运行的代码与数据单元，类比 ELF 共享库。只读，可跨 context 复用 | 一个 model / graph |
| **Context** | 一组 module 链接并实例化的结果，持有各 module 的**可变状态**。创建成本极低（微秒级、约 100B + 程序状态） | session |
| **Invocation** | 对 module 导出函数的一次调用。**同步还是异步由调用方逐次决定**，与被调函数无关 | 一次 `run()` |
| **Timeline** | 可观测的执行顺序。由用户自己定义，通过 fence 告知 IREE | —— |
| **Fence** | 一个或多个 timeline 上的某个进度点，充当 barrier / fork / join | —— |

三个反直觉但重要的点：

1. **在 IREE 里"一切都是 module"**，包括 HAL 这样的运行时子系统、包括用户自定义的 C 代码。传统运行时把模型当作特殊对象，IREE 不区分。
2. **invocation 之间默认没有顺序**。不加 fence 时，多次 invocation 的执行顺序是任意的、可以并发的——"就像 C 里没有 barrier 的多个线程"。顺序**只**来自 fence。
3. **fence 的等待是"wait-until"语义**：要求 timeline 至少到达某个点。这给了执行器重排和延迟的自由（比如把相似的工作攒在一起跑）。

### 3.2 用户看到的样子

```python
# fence 本质就是 (timeline, 整数) 二元组，持有成本极低
wait_fence   = my_timeline.at(t)
signal_fence = my_timeline.at(t + 1)

# 这是"向时间线上排布工作"，不是"执行工作"
async_invoke(@some_fn, wait_fence, signal_fence)
# 调用可能已经立即返回了，此刻可能一条指令都还没真正执行

signal_fence.wait()   # 只有这里才是真正的同步点
```

多个 invocation 可以直接拼成任意 DAG，**不需要等前一个完成**：

```python
fence_a, fence_b, fence_c = tl.at(t), tl.at(t+1), tl.at(t+2)
async_invoke(@fn0, fence_a, fence_b)
async_invoke(@fn1, fence_b, fence_c)
async_invoke(@fn2, fence_b, fence_c)   # 与 fn1 并发
async_invoke(@fn3, None,    fence_c)   # 无前置依赖
async_invoke(@fn4, fence_a, fence_c)
fence_c.wait()
```

**设计原则**：*尽可能少地声明约束*。用户声明的约束越少，下层的调度自由度越大。

### 3.3 输入输出与 fence 的绑定

fence 本身只管执行顺序，不管资源。但用户可以用 fence **给资源加上时间约束**：

```python
# wait_fence_a 到达前不能读 buffer_a / buffer_b
# wait_fence_b 到达前不能读 buffer_c
# signal_fence_a 到达后 buffer_a 可读（原地写回）
async_invoke(@fn,
             (wait_fence_a, buffer_a, buffer_b),
             42,                              # 值类型无需排序
             (wait_fence_b, buffer_c),
             (signal_fence_a, buffer_a))
```

这个模式会**原样落到 HAL 层**：`hal.tensor.import` 带一个 `wait` fence，`hal.tensor.barrier` / `hal.tensor.export` 产出 signal fence。见 [4.9](#49-编译器侧的-hal-dialect)。

### 3.4 Stream-ordered allocation（峰值内存的关键）

一次 invocation 本身开销只有几 KB，但**存储 buffer 可以轻松到几百 MB**。IREE 同时支持两种分配：

- **host-ordered allocation**（类似 malloc/free）：用于长期存活的常量、用户自管的 ringbuffer。
- **stream-ordered allocation**：分配和释放**本身也带 wait/signal fence**，作为工作被排到时间线上，可以在设备上远程执行，**不需要 host 参与**。

效果：

```
fence0 ─┬─ alloca0 ─┐
        └─ alloca1 ─┴─ fence0b ─ @fn0 ─ fence1a ─ dealloc0 ─ fence1b ─┬─ @fn1 ─┐
                                                                       └─ @fn2 ─┴─ dealloc1
```

`@fn0` 的临时内存在执行前一刻才提交（commit）、执行后立刻退还；`@fn0 → @fn1/@fn2` 的传递数据活到 fn1/fn2 都跑完。**顺序执行 N 次 invocation，只需要一次 invocation 的内存**。

两个额外好处：
- 原生支持 stream-ordered 分配的设备（CUDA）甚至可以**跨进程共享内存池**。
- 内存不足时不会失败而是**自动串行化**：512MB 的系统可以跑两个各需 400MB 临时内存的独立 invocation，用户只感知到延迟变高。

> 对算力网场景：这一条直接决定了"单卡能塞下多大的模型/多少并发请求"。

**可跑** · 这套机制的输入就是 stream 层给每个资源打的 `Lifetime` 标签：

```bash
cd iree-lab && bash scripts/run_phases.sh
grep -o '!stream.resource<[a-z]*>' out/phases/tiny_mlp.stream.mlir | sort | uniq -c
grep -c 'stream.resource.alloca' out/phases/tiny_mlp.stream.mlir
```

只有被判为 `transient` 的资源才进得了上面那个"用完立刻退还"的流程 ——
`external` 是用户传进来的、`variable` 要跨调用活着，两者都不能这么处理。
所以**峰值内存的上限，在 stream 相位打标签那一刻就定了**，运行时再聪明也改不了。
主角模型里如果 Relu/Add 被融进了 Gemm，中间张量连 `transient` 都不用做 —— 压根不存在。

---

## 第 4 章 IREE HAL 详解（重点）

### 4.1 HAL 是什么：五项职责与设计取向

`runtime/src/iree/hal/README.md` 的原文定义：

> IREE HAL 表达了对现代计算 API（如 Vulkan，CPU 也算）的一层底层抽象。每一个 HAL 实现都能够：
> 1. **枚举并查询设备及其能力**
> 2. **定义在设备上运行的可执行代码**
> 3. **分配统一内存或离散内存，并提供缓存控制**
> 4. **把工作组织成序列以便延迟提交**
> 5. **提供显式的同步原语来给提交排序**

把这五条和第 3 章的执行模型对照，会发现它们一一对应：

| HAL 职责 | 对应的对象 | 对应的执行模型概念 |
|---------|-----------|------------------|
| 枚举查询设备 | `driver` / `device` | —— |
| 定义设备代码 | `executable` / `executable_cache` | dispatch region 的落地 |
| 分配内存与缓存控制 | `allocator` / `buffer` / `buffer_view` | Stream 的 Lifetime 分类 |
| 组织工作、延迟提交 | `command_buffer` + device queue API | invocation "排布工作而非执行工作" |
| 显式同步原语 | `semaphore` / `fence` / `event` | timeline / fence |

### 4.1.1 三条设计取向（理解 HAL 的钥匙）

**(1) 向现代 GPU API 看齐，显式控制一切**

CUDA HAL driver 设计文档说得非常直白：

> IREE HAL 的设计从现代 GPU API 汲取灵感——它提供对底层 GPU 对象的**显式控制**。**编译器**被期望去规划对象生命周期、以优化的方式安排工作负载与同步；**HAL 实现和底层 GPU 驱动栈被期望是一层薄薄的、没有太多聪明劲和魔法的东西**。

所以实现 CUDA HAL driver 时 IREE 用的是 **driver API 而非 runtime API**，运行时动态加载 `libcuda.so` / `nvcuda.dll` 并通过 `cuGetProcAddress()` 只取用到的那个子集。

这条取向的推论：**如果你要给自研加速器写 HAL driver，不要在 driver 里做调度优化**——那是编译器的活。driver 越薄，编译器的决定就越能原样落地。

**(2) compiler target backend 与 runtime HAL device 是 N:M 关系**

官方 `#hal.executable.target` 的文档原话：

> 同一个编译后端可以为好几种不同的运行时设备翻译 executable；同样，同一个运行时设备也可以使用多种 executable target 之一。**一律假设二者之间是 N:M 映射**。

具体对照（来自部署配置文档）：

| 编译 target backend | 产物格式 | 兼容的 HAL device |
|--------------------|---------|------------------|
| `llvm-cpu` | ELF / 静态库 | `local-sync`, `local-task` |
| `vmvx` | 微内核库解释器 | `local-sync`, `local-task` |
| `vulkan-spirv` | SPIR-V | `vulkan` |
| `cuda` | PTX | `cuda` |
| `rocm` | HSACO | `hip` |
| `metal-spirv` | MSL | `metal` |
| `webgpu-spirv`（实验） | WGSL | `webgpu` |

**这个 N:M 关系是"低成本切换后端"的结构基础**——它让"编译目标"和"运行时设备"解耦。

**(3) HAL 是可选的、可替换的**

设计文档原话："HAL 是 IREE 的一个**可选特性**……它被 IREE 程序内部用来定义和提交工作、跨设备发信号，但**也可以被用户直接使用**来以兼容的方式对接硬件。"

这意味着两个高级用法：
- 用户直接构造 HAL buffer 喂给 IREE，**零 marshalling 开销**。
- 把 IREE **插进用户已有的设备 context**（比如已有的 CUDA context / Vulkan device），共享调度和资源；或者把用户自己的代码插进 IREE 的流水线。

> 对算力网场景，第二条是关键：IREE 不要求独占设备，可以嵌进已有的分布式训练/推理框架里。

### 4.2 对象模型总览

`runtime/src/iree/hal/api.h` 是一个伞形头文件，它 include 的列表就是 HAL 的完整对象清单：

```
allocator.h        buffer.h            buffer_transfer.h   buffer_view.h
buffer_view_util.h channel.h           channel_provider.h  command_buffer.h
device.h           device_group.h      driver.h            driver_registry.h
event.h            executable.h        executable_cache.h  fence.h
file.h             pool.h              pool_set.h          queue.h
resource.h         semaphore.h         string_util.h       topology.h
topology_builder.h profile_*.h
```

按职责分组后的关系图：

```
                      driver_registry
                            │ 枚举
                            ▼
                         driver ──────────► device ◄──── topology / device_group
                                              │            （多设备拓扑，较新）
            ┌─────────────────────────────────┼─────────────────────────────┐
            │                                 │                             │
       【内存】                          【工作表达与提交】              【同步】
            │                                 │                             │
       allocator                        command_buffer                 semaphore
            │  分配                            │ 录制命令                    │ 64 位单调时间线
            ▼                                 │  ├ dispatch                  ▼
         buffer  ◄── subspan/import           │  ├ copy / fill / update    fence
            │                                 │  ├ collective ──► channel   （一组 timepoint）
            ▼ + shape/dtype                   │  └ execution_barrier          │
       buffer_view                            │                             event
                                              ▼ 提交                        （设备内细粒度）
                                    device queue API
                                    (alloca/execute/read/write/...)
                                              │
                                              ▼
                                     executable_cache ──► executable
                                     （准备/缓存设备代码）  （fat binary，多 variant）
                                              ▲
                                              │ 加载
                                       file / pool（参数与内存池，较新）
```

一句话记忆：**allocator 造内存、command_buffer 攒工作、queue 提交工作、semaphore 排时序、executable 装代码、channel 做集合通信。**

#### 示例精讲：一次 invocation 里对象按什么顺序被造出来

**部分可跑** · 左栏的 C 侧伪码**无 lab 对应**（要写就得引一套 CMake，与"pip 装完即跑"冲突）；
右栏可跑 · 命令 `cd iree-lab && bash scripts/run_phases.sh` · 产物 `out/phases/tiny_mlp.hal.mlir`

**每一步在编译产物里都留了痕**——右栏给出它在 `tiny_mlp.hal.mlir` 里的对应物，
先跑上面的命令生成该文件，然后照着 grep：

| 伪码这一步 | 在 `tiny_mlp.hal.mlir` 里 grep 什么 |
|-----------|-----------------------------------|
| ② 分配 buffer | `hal.allocator.allocate` / `hal.device.queue.alloca` |
| ③ 准备 executable | `hal.executable` / `hal.executable_cache` |
| ④ 录制命令 | `hal.command_buffer.create` / `.dispatch` / `.finalize` |
| ⑤⑥ 时序与提交 | `hal.fence.create` / `hal.device.queue.execute` / `hal.fence.await` |

**为什么值得对照着看**：伪码是"你手写会怎么写"，`hal.mlir` 是"编译器替你写成了什么"。
两边一一对得上，说明 IREE 的运行时 API 不是另一套东西，而是**编译器生成的调度代码的目标语言**。

host 侧一次完整调用的骨架（伪代码，参数省略）：

```c
// ① 接入点：driver 只负责枚举与创建
iree_hal_driver_registry_try_create(registry, IREE_SV("local-sync"), &driver);
iree_hal_driver_create_device_by_id(driver, device_id, ..., &device);

// ② 内存：allocator 挂在 device 上，不单独创建
iree_hal_allocator_t* allocator = iree_hal_device_allocator(device);
iree_hal_allocator_allocate_buffer(allocator, params, /*size=*/..., &buffer);
iree_hal_buffer_view_create(buffer, shape_rank, shape, elem_type, enc_type, ..., &view);

// ③ 设备代码：cache 把二进制「准备」成可执行对象并缓存
iree_hal_executable_cache_create(device, IREE_SV("cache"), ..., &cache);
iree_hal_executable_cache_prepare_executable(cache, &params, &executable);

// ④ 工作：录制一段命令（此刻设备上什么都没发生）
iree_hal_command_buffer_create(device, mode, categories, queue_affinity, ..., &cmd);
iree_hal_command_buffer_begin(cmd);
iree_hal_command_buffer_dispatch(cmd, executable, /*ordinal=*/0, ...);
iree_hal_command_buffer_end(cmd);

// ⑤ 时序：semaphore 是时间线本身
iree_hal_semaphore_create(device, /*initial_value=*/0ull, ..., &semaphore);

// ⑥ 提交：wait / signal 都是「(semaphore, value)」列表
iree_hal_device_queue_execute(device, queue_affinity,
                              /*wait=*/{semaphore, 0}, /*signal=*/{semaphore, 1}, cmd, ...);
iree_hal_semaphore_wait(semaphore, /*value=*/1ull, timeout);
```

> 伪代码：各版本的参数列表有增删（例如 `queue_execute` 是否带 binding table），以本地 `runtime/src/iree/hal/*.h` 为准。要记住的是**顺序与依赖**。

持有关系（实线箭头 = 创建 / 派生，虚线 = 引用计数持有）：

```text
driver_registry
└─ driver                        ← 只持有动态加载的符号与设备枚举能力
   └─ device                     ← 一切的中心，下面全由它派生
      ├─ allocator（device 自带，非独立创建）
      │  └─ buffer
      │     └─ buffer_view ····> buffer（+ shape / dtype / encoding）
      ├─ executable_cache
      │  └─ executable           ← fat binary，内含多 variant
      ├─ command_buffer
      │  └─ 命令列表 [dispatch] ····> executable（target + ordinal）
      │                         ····> buffer（bindings）
      ├─ semaphore               ← 时间线本身，由 device 创建
      ├─ fence ················> 一组 (semaphore, value)，只是时间点的打包
      └─ queue：不是对象，是 device 上的一组 API + queue_affinity 位掩码
```

三个容易搞错的点：

- **allocator 不是独立造出来的**，它由 device 提供（`iree_hal_device_allocator()`）。
- **queue 不是对象**：`hal.device.queue.*` 全部挂在 device 上，用 `affinity` 位掩码选队列。
- **HAL 对象都是引用计数的**：命令缓冲录进去的 buffer / executable 会被保活到不再需要为止；但「什么时候该分配、什么时候能还」是**编译器**算好的，不是运行时临场决定的（[4.1.1](#411-三条设计取向理解-hal-的钥匙)）。

> **自测**：`command_buffer` 创建出来时，`executable` 必须已经准备好了吗？为什么？

### 4.3 设备与内存：driver / device / allocator / buffer / buffer_view

#### driver 与 driver_registry

`iree_hal_driver_t` 负责**枚举与创建设备**。注意 CUDA HAL driver 文档的说明：

> **没有任何 CUDA 构造直接对应 `iree_hal_driver_t`**。我们用它来持有为所有设备动态加载的符号，以及做设备枚举和创建。

也就是说 driver 是一个纯 IREE 侧的概念，是"某类硬件的接入点"。运行时可用的 driver 列表：

| HAL device | 说明 |
|-----------|------|
| `local-sync` | 同步本地 CPU 设备，内联执行（裸机首选） |
| `local-task` | 多线程本地 CPU 设备，用 task executor |
| `cuda` | NVIDIA GPU |
| `hip` | AMD GPU（HIP） |
| `amdgpu`（实验） | AMD GPU（HSA） |
| `vulkan` | 跨平台 GPU |
| `metal` | Apple 平台 GPU |
| `webgpu`（实验） | Web |

**外部 driver**：CMake 选项 `IREE_EXTERNAL_HAL_DRIVERS` 允许在树外定义 driver——这是接入自研加速器的官方口子。

#### device

`iree_hal_device_t` 是核心句柄。关键概念是 **queue affinity**：一个 `uint64_t` 位掩码，指明操作可以落在设备的哪些队列上。几乎所有队列操作和 channel 创建都带这个参数。

CUDA 实现的细节很能说明问题：每个设备创建**两个 `CUstream`**——一个负责按程序指令发射内存分配和 kernel launch，另一个专门用来在命令缓冲完成后发射 host 回调函数（原因见 [4.6](#46-同步timeline-semaphore-与-fencehal-的灵魂)）。

`iree_hal_device_query_*` 用来查询设备能力，编译器生成的代码会用 `hal.device.query` 在运行时做能力分支。

#### allocator 与内存类型

`iree_hal_allocator_t` 负责分配。分配请求由两组位域描述：

**MemoryType**（内存住在哪里、谁能看见）：

| 位 | 值 | 含义 |
|----|----|------|
| `Optimal` | 1 | 让实现自选最优 |
| `HostVisible` | 2 | host 可映射访问 |
| `HostCoherent` | 4 | host 与 device 视图自动一致，无需显式 flush |
| `HostCached` | 8 | host 侧走缓存 |
| **`HostLocal`** | 70 | = `HostVisible｜HostCoherent｜Host…` 组合，内存主体在 host |
| `DeviceVisible` | 16 | device 可访问 |
| **`DeviceLocal`** | 48 | = `DeviceVisible｜Optimal…` 组合，内存主体在 device |

**BufferUsage**（这块内存要拿来干什么）：

| 组 | 位 | 用途 |
|----|----|------|
| 传输 | `TransferSource` / `TransferTarget` / `Transfer` | 作为拷贝的源/目标 |
| 派发 | `DispatchStorageRead` / `DispatchStorageWrite` / `DispatchStorage`<br>`DispatchUniformRead` / `DispatchIndirectParameters`<br>`DispatchImageRead` / `DispatchImageWrite` | kernel 怎么用它 |
| 共享 | `SharingExport` / `SharingReplicate` / `SharingConcurrent` / `SharingImmutable` | 跨设备/跨队列共享语义 |
| 映射 | `MappingScoped` / `MappingPersistent` / `MappingOptional`<br>`MappingAccessRandom` / `MappingAccessSequentialWrite` | host 侧映射方式 |

还有 **MemoryAccess**（`Read` / `Write` / `Discard` / `MayAlias` / `Unaligned` / `Any`）用于 subspan 和 mapping 时的权限收窄，以及 **MemoryModel**（`Unified` / `Discrete`）描述设备的整体内存模型。

> **这套位域就是 TinyIREE 论文里"buffer 可见性与权限控制"的真身**：一块内存可以在 host 分配、对 device 只开放 `Read`。

CUDA 的实现映射非常直接，可以用来记忆语义：

| 请求 | CUDA 调用 |
|------|----------|
| host-local 内存 | `cuMemHostAlloc()` |
| device-local 且 host 不可见 | `cuMemAlloc()` |
| device-local 且 host 可见 | `cuMemAllocManaged()` |

#### buffer 与 buffer_view

- **`iree_hal_buffer_t`**：一段裸内存。在 CUDA 上就是一个 host 指针或一个 `CUdeviceptr`。支持 `subspan`（切子区间，可同时收窄访问权限）、`load`/`store`（host 侧直接读写小值）、`length`。
- **`iree_hal_buffer_view_t`**：**buffer + shape + element type + encoding type**。这是"张量"在运行时的形态，也是 ABI 边界上传进传出的东西。

对应的 HAL dialect op 一览（这组 op 在读 IR 时天天见）：

```mlir
%buffer  = hal.buffer_view.buffer<%view : !hal.buffer_view> : !hal.buffer
%rank    = hal.buffer_view.rank<%view : !hal.buffer_view> : index
%dim0    = hal.buffer_view.dim<%view : !hal.buffer_view>[0] : index
%etype   = hal.buffer_view.element_type<%view : !hal.buffer_view> : i32
%sub     = hal.buffer.subspan<%buffer : !hal.buffer>[%offset, %length] : !hal.buffer
```

还有一组 `hal.buffer.allocation.{discard, preserve, is_terminal}`，用于配合 stream-ordered allocation 表达"这块内存的内容还需不需要保留"。

#### 示例精讲：同一块 buffer 的三副面孔

**部分可跑** · ② 和 ③ 在 `iree-lab` 的 `out/phases/tiny_mlp.hal.mlir` 里能直接找到
（`grep -n 'buffer_view\|command_buffer.dispatch'`）；① 那种显式 `hal.allocator.allocate`
在编译产物里通常已被 stream 层的资源分配吸收掉，是 runtime API 侧的写法。

一块内存从分配到被 kernel 读，中间换了三种「说法」，但**自始至终是同一个 `!hal.buffer`**：

```mlir
// ① 分配：说清「住哪」(MemoryType) 与「干什么用」(BufferUsage)
%buffer = hal.allocator.allocate<%allocator : !hal.allocator>
              affinity(%affinity)
              type("DeviceVisible|DeviceLocal")
              usage("TransferTarget|DispatchStorageRead")
              : !hal.buffer{%length}

// ② 加上 shape / dtype，变成 ABI 边界上的「张量」
%view = hal.buffer_view.create buffer(%buffer : !hal.buffer)[%c0, %length]
              shape([%c4, %c8])
              type(%element_type)
              encoding(%encoding_type) : !hal.buffer_view

// ③ 派发时又退回裸 buffer + 字节区间：kernel 只认偏移和长度
hal.command_buffer.dispatch<%cmd : !hal.command_buffer>
    target(%exe : !hal.executable)[%c0]
    workgroups([%c4, %c1, %c1])
    bindings([(%buffer : !hal.buffer)[%c0, %length]])
    flags(None)
```

> 形态示意：`type(...)` / `usage(...)` 的枚举串与 `encoding` 的取值以本地 dump 为准。

**读法**：`buffer_view` 是给 **ABI 和人**看的（带 shape），`dispatch` 的 binding 是给 **kernel** 看的（只有 offset + length）。kernel 侧再用 [4.7.4](#474-pipeline-layouthost-与-device-的-abi-契约) 的 `hal.interface.binding.subspan` 把这段字节重新解释成 `memref<4x8xf32>`——**静态 shape 是编译期编死在 kernel 里的，运行时不传**。

MemoryType / BufferUsage 怎么配，与 CUDA 的对照：

| 场景 | MemoryType | BufferUsage 至少要有 | CUDA 上的落地 |
|------|-----------|--------------------|--------------|
| 输入 / 输出，host 要读写 | `HostLocal` | `Transfer` + 映射类（`MappingScoped` 等） | `cuMemHostAlloc()` |
| 中间结果，只有 kernel 碰 | `DeviceLocal` | `DispatchStorage` | `cuMemAlloc()` |
| device 为主但 host 也要看 | `DeviceLocal｜HostVisible` | `Transfer｜DispatchStorage` | `cuMemAllocManaged()` |
| 只读常量 / 权重 | `DeviceLocal` | `DispatchStorageRead`（+ `SharingImmutable`） | `cuMemAlloc()`，只开读权限 |

对照记忆：**MemoryType ≈ CUDA 里你挑哪个 `cuMemAlloc*`；BufferUsage 则是 CUDA 没有的一维**——CUDA 靠约定，IREE 要求把用途显式声明出来，编译器才敢做别名分析与内存复用。

> **自测**：kernel 侧拿到的 `memref<4x8xf32>`，它的 `4x8` 是运行时从 `buffer_view` 读出来的吗？

### 4.4 工作表达：command_buffer

`iree_hal_command_buffer_t` 是**一段命令的录制**；提交给设备后才在 GPU 上异步执行。这是"延迟提交"这条职责的载体。

#### 可录制的命令

| 命令 | HAL dialect op | 作用 |
|------|---------------|------|
| 派发 kernel | `hal.command_buffer.dispatch` | 核心命令 |
| 间接派发 | `hal.command_buffer.dispatch.indirect` | workgroup 数来自设备内存 |
| 拷贝 | `hal.command_buffer.copy_buffer` | buffer 间拷贝 |
| 填充 | `hal.command_buffer.fill_buffer` | 重复 pattern 填充 |
| 更新 | `hal.command_buffer.update_buffer` | 把 host 侧小数据塞进去 |
| 集合通信 | `hal.command_buffer.collective` | all_reduce 等，见 [4.8](#48-集合通信与多设备channel--topology) |
| 屏障 | `hal.command_buffer.execution_barrier` | 命令缓冲**内部**的执行/内存依赖 |
| 调试 | `hal.command_buffer.begin_debug_group` / `end_debug_group` | Tracy / Nsight 里的分组标签 |
| 收尾 | `hal.command_buffer.finalize` | 录制结束，之后不可再改 |

`dispatch` 的完整形态（这是全 HAL 最该读懂的一条 op）：

```mlir
hal.command_buffer.dispatch<%cmd : !hal.command_buffer>
    target(%executable : !hal.executable)[%entry_point_ordinal]
    workgroups([%count_x, %count_y, %count_z])
    constants([%c1, %c2])
    bindings([
      (%buffer0 : !hal.buffer)[%offset0, %length0],
      (%buffer1 : !hal.buffer)[%offset1, %length1]
    ])
    flags(None)
```

三个要点：

1. **workgroup 是 3D grid**——GPU 风格。CPU 后端也照此建模，只是把三重循环在 host 侧展开（TinyIREE 论文里 `worker.cnt.{x,y,z}` 三重循环调 `dispatch_ptr` 就是这个）。
2. **constants 恒为 4 字节值且语义不透明**——可以是 bit-cast 的 float、位打包的 bool。这就是 Vulkan 的 push constant。
3. **bindings 里的 buffer 可以是真实 HAL buffer，也可以是"命令缓冲绑定表"的间接引用**（`index or buffer`）。后者让同一份录制好的命令缓冲可以换一批 buffer 重放。

`CommandBufferMode` 位域只有两个有效值：`OneShot`（一次性，录完就提交一次）和 `AllowInlineExecution`（允许录制时直接执行）。`CommandCategory` 用 `Transfer` / `Dispatch` 声明这个命令缓冲会用到哪类队列能力。

#### 实现层的经典难题：录制语义 vs 硬件语义

CUDA HAL driver 提供了两套 `command_buffer` 实现，对比非常有教学价值：

| 实现 | 后端 | 特点 |
|------|------|------|
| **graph_command_buffer** | `CUgraph` | 概念上最贴合——`CUgraph` 本来就是"命令的录制"。还能轻松表达 dispatch 之间的细粒度依赖，不用开多条 stream。**但 `CUgraph` 是为"录一次、放多次"设计的，一次性命令缓冲用它有性能惩罚** |
| **stream_command_buffer** | `CUstream` | 录制即发射，没有录制/回放分离。为了对上 HAL 的录制语义，**必须先录进内存里的 `iree_hal_deferred_command_buffer_t`，等到"应用"命令缓冲时再新建一个 `CUstream` 版本重放** |

由 `iree_hal_cuda_device_params_t::command_buffer_mode` 选择。

> **`iree_hal_deferred_command_buffer_t` 是写 HAL driver 时的通用救生圈**：如果目标 API 没有"录制"概念，就用它把命令先攒在内存里。

### 4.5 工作提交：device queue API

命令缓冲录好之后，通过 device 上的**队列 API** 提交。**所有队列操作都是异步的，都由 `(wait_fence, signal_fence)` 包裹**：

| 队列操作 | HAL dialect op | 作用 |
|---------|---------------|------|
| 执行命令缓冲 | `hal.device.queue.execute` | 主力 |
| 间接执行 | `hal.device.queue.execute.indirect` | 带绑定表的执行 |
| **流序分配** | `hal.device.queue.alloca` | 见 [3.4](#34-stream-ordered-allocation峰值内存的关键) |
| **流序释放** | `hal.device.queue.dealloca` | |
| 队列内屏障 | `hal.device.queue.barrier` | 纯同步，不带命令 |
| 拷贝 / 填充 / 更新 | `hal.device.queue.copy` / `.fill` / `.update` | 不用建命令缓冲的单发传输 |
| 文件读写 | `hal.device.queue.read` / `.write` | 与 `iree_hal_file_t` 配合，直接从文件加载参数到设备 |
| 冲刷 | `hal.device.queue.flush` | 强制把攒着的工作推给设备 |

`hal.device.queue.execute` 的签名完整体现了这个模式：

```mlir
hal.device.queue.execute<%device : !hal.device>
    affinity(%queue_affinity)     // i64 位掩码：允许落在哪些队列
    wait(%wait_fence)             // 到达前一条命令都不执行
    signal(%signal_fence)         // 全部完成后 signal
    commands(%cmd)
    flags(None)
```

官方描述："在 wait fence 到达之前不会执行任何命令，所有命令完成后 signal fence 被置位。"

> **注意 `alloca` / `dealloca` 也在这张表里**。这正是 stream-ordered allocation 的实现方式：内存分配本身就是一条排在时间线上的队列操作，可以在设备上远程完成，不需要 host 往返。

#### 示例精讲：从建命令缓冲到等到结果的六步

**可跑（对着 dump 认）** · 跑 [`iree-lab/scripts/run_phases.sh`](../../iree-lab/scripts/run_phases.sh) 后，
在 `out/phases/tiny_mlp.hal.mlir` 里按顺序 `grep` 这六个词，你会发现**它们的出现顺序和下面完全一致**：

```bash
grep -n 'command_buffer.create\|command_buffer.dispatch\|command_buffer.finalize\|queue.execute\|fence.await' \
     iree-lab/out/phases/tiny_mlp.hal.mlir
```

把 [4.4](#44-工作表达command_buffer) 和本节串起来，一次 dispatch 的完整链条只有六步（第 5 步发生在设备侧，host 看不到）：

```mlir
// 1. create：声明模式与要用到的队列能力
%cmd = hal.command_buffer.create device(%device : !hal.device)
           mode("OneShot") categories("Transfer|Dispatch") : !hal.command_buffer

// 2. dispatch：只是录制
hal.command_buffer.dispatch<%cmd : !hal.command_buffer>
    target(%exe : !hal.executable)[%c0]
    workgroups([%cx, %cy, %cz])
    constants([%c8])
    bindings([(%in : !hal.buffer)[%c0, %len], (%out : !hal.buffer)[%c0, %len]])
    flags(None)

// 3. finalize：封口，之后不可再改
hal.command_buffer.finalize<%cmd : !hal.command_buffer>

// 4. queue.execute：排到时间线上，立即返回
hal.device.queue.execute<%device : !hal.device>
    affinity(%affinity)
    wait(%wait_fence)      // 到达前一条命令都不执行
    signal(%signal_fence)  // 全部完成后置位
    commands(%cmd)
    flags(None)

// 6. await：这里才是真正的同步点
%status = hal.fence.await until([%signal_fence]) timeout_millis(%timeout) : i32
```

六步里「谁阻塞、谁不阻塞」是最该记牢的：

| 步 | 动作 | 阻塞吗 | 此刻设备上发生了什么 |
|----|------|-------|-------------------|
| 1 | `create` | 否 | 造一个录制器 |
| 2 | `dispatch` | 否 | **什么都没有**，只往缓冲里写一条命令 |
| 3 | `finalize` | 否 | 命令缓冲转为只读，可提交 |
| 4 | `queue.execute` | **否** | 交给驱动；`wait` 未到达时实现可以先压着不发 |
| 5 | wait 到达 | —— | 命令按录制顺序执行 |
| 6 | `fence.await` | **是** | host 挂起，直到 signal 值被推进 |

此刻内存里的持有关系：

```text
device
├─ executable            ← executable_cache 准备好，可被多次 dispatch 复用
├─ buffer %in / %out     ← allocator 分配，或 queue.alloca 流序分配
├─ command_buffer %cmd
│  └─ 命令列表 [dispatch] ──► executable（target + ordinal=0）
│                        ──► buffer %in / %out（bindings 里的 offset+length）
└─ semaphore
   ├─ %wait_fence   = {(semaphore, N)}
   └─ %signal_fence = {(semaphore, N+1)}
```

关键：**第 4 步返回时，第 2 步录的 kernel 很可能一条都还没跑**。命令缓冲、buffer、executable 三者都必须活到执行真正完成——这就是 HAL 对象全都带引用计数的原因。

> **自测**：如果把第 6 步删掉，计算结果会算错吗？会出什么问题？

### 4.6 同步：timeline semaphore 与 fence（HAL 的灵魂）

这一节是整个 HAL 里最需要花时间的部分。

#### 4.6.1 语义定义

CUDA HAL driver 文档给出了最完整的定义：

> IREE HAL 用 semaphore 在 host CPU 线程与 device GPU 流之间同步。它是一个**统一原语，覆盖全部四个方向**——host→host、host→device、device→host、device→device，并且**允许灵活的信号/等待顺序**——先 signal 后 wait、或先 wait 后 signal 都可以。**对同一个值的等待次数也没有限制。**
>
> HAL semaphore 的核心状态是一个**单调递增的 64 位整数**，它构成一条时间线——把 semaphore signal 到更大的值就推进了时间线，并解除那些等待更早值的工作。语义紧密对应 **Vulkan timeline semaphore**。

拆成四条能力，逐条都要能背出来：

| 能力 | 要求 |
|------|------|
| **状态** | 64 位单调递增整数（不是二值信号量） |
| **顺序** | signal-before-wait 和 wait-before-signal **都必须支持** |
| **方向** | host↔host、host↔device、device↔host、device↔device **全支持** |
| **多等待** | 同一个值可以被任意多次等待 |

**Fence** 则是"一组 semaphore timepoint"的封装——`iree_hal_fence_t`。用户层看到的是 fence，HAL 内部展开成 semaphore 列表。HAL dialect 里对应的 op：

```mlir
%fence = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
%joined = hal.fence.join at([%fence_a, %fence_b]) -> !hal.fence   // 汇合多条时间线
hal.fence.signal<%fence : !hal.fence>
hal.fence.fail<%fence : !hal.fence> status(%status)               // 传播错误
%status = hal.fence.await until([%fence]) timeout_millis(%t) : i32
%reached = hal.fence.query<%fence : !hal.fence> : i32
```

`hal.fence.fail` 值得注意：**错误也沿着时间线传播**。异步执行体系里，错误不能靠返回值传，只能靠 fence 带回来。

#### 示例精讲：一个 semaphore 从 0 走到 3

**无 lab 对应（C runtime API 伪码）**——lab 走的是 `iree-run-module`，
时序被封在 runtime 里，看不到显式的 `signal/wait` 数值。
编译产物侧的等价物是 `hal.fence.create` / `hal.fence.await`，
在 `iree-lab/out/phases/tiny_mlp.hal.mlir` 里可以 grep 到（见上一节「从建命令缓冲到等到结果的六步」）。

场景：同一条时间线上排两批工作，`cmd0` 完成后 `cmd1` 才能开始，host 最后取结果。

```c
// 伪代码：创建时给初值 0
iree_hal_semaphore_create(device, /*initial_value=*/0ull, flags, &sem);

// 第一批：等 1，完成后 signal 2
iree_hal_device_queue_execute(device, affinity,
    /*wait=*/  {{sem, 1ull}},
    /*signal=*/{{sem, 2ull}}, cmd0, ...);

// 第二批：等 2，完成后 signal 3   ← 与上一批的依赖只靠这两个数值
iree_hal_device_queue_execute(device, affinity,
    /*wait=*/  {{sem, 2ull}},
    /*signal=*/{{sem, 3ull}}, cmd1, ...);

iree_hal_semaphore_signal(sem, 1ull);                        // host 开闸
iree_hal_semaphore_wait(sem, 3ull, iree_infinite_timeout()); // host 等结果
```

数值时间线（横轴是 semaphore 的值，不是墙钟时间）：

```text
sem   0 ──────► 1 ──────► 2 ──────► 3
                ▲         ▲         ▲
host  signal(1) ┘         │         │   host 开闸，cmd0 的等待条件才满足
cmd0     wait(1) ═════════╝         │   放行 → 设备执行 → 完成时 signal(2)
cmd1               wait(2) ═════════╝   等到 2 才放行 → 完成时 signal(3)
host                       wait(3) ─┘   值到 3，host 返回，输出可读
```

逐步走读：

| 时刻 | 值 | 发生了什么 |
|------|----|-----------|
| t0 | 0 | 两次 `queue_execute` 都已返回，但 `wait(1)` 不满足，**一条命令都没发给设备** |
| t1 | 0→1 | host `signal(1)`：cmd0 条件满足，实现把它放行给驱动 |
| t2 | 1 | cmd0 在设备上执行；cmd1 仍压着（等 2） |
| t3 | 1→2 | cmd0 完成并 signal 2 → cmd1 被放行 |
| t4 | 2→3 | cmd1 完成并 signal 3 |
| t5 | 3 | host 的 `wait(3)` 返回，此时才能安全读输出 buffer |

三个反直觉点，全都是「单调 64 位 + 多等待」直接推出来的：

- **两批工作的先后不是靠提交顺序建立的**，是靠 `signal 2` / `wait 2` 这对数值。把两次 `queue_execute` 的调用顺序对调，结果一样。
- **wait-before-signal 合法**：t0 时 `wait(1)` 先提交、`signal(1)` 后到，实现必须把工作压住——这正是 [4.6.2](#462-为什么这个抽象是难的cuda-上的落地) 里「待决动作队列」存在的理由。
- **同一个值可以被任意多个等待者等**：再排一批 `wait(2)` 的工作，它就与 cmd1 并发。

与 CUDA 的并排：

| | IREE semaphore | CUDA `CUevent` |
|--|----------------|----------------|
| 状态 | 64 位单调递增值 | 二值（未发生 / 已发生） |
| 一个同步点 = | 一个数值 | 一个 event 对象 |
| 表达上面这条线 | 一个 semaphore，值 1 / 2 / 3 | 要 2~3 个 event 对象 |
| wait 先于 signal | 允许 | **不允许**，必须先 record |
| host 侧 signal | `iree_hal_semaphore_signal()` | **没有**，host 无法 signal 一个 event |
| host 侧等待 | `iree_hal_semaphore_wait(值)` | `cuEventSynchronize()` |
| 方向覆盖 | 四个方向全支持 | 只有 device→device / device→host |

> **自测**：如果 host 始终不调 `signal(1)`，设备上会发生什么？两批工作各自处在什么状态？

#### 4.6.2 为什么这个抽象是"难"的：CUDA 上的落地

这是理解"HAL 抽象 vs 硬件能力落差"的最佳案例，也是我认为**最值得完整读一遍源码的一段**。

CUDA 里**没有任何一个原语能满足上述四条**：

| 候选方案 | 为什么不行 |
|---------|-----------|
| `cuStreamWriteValue64()` / `cuStreamWaitValue64()` | 能做 64 位值的 signal/wait，但**要求 device 指针、不接受 managed memory 指针 → host 侧用不了**。而且规范明说"通过这些 API 建立的同步顺序对 CUDA 不可见"，无法与其他 CUDA 组件集成 |
| `cuSignalExternalSemaphoresAsync()` / `cuWaitExternalSemaphoresAsync()` | 能直接对应 Vulkan timeline semaphore，但**只能通过 `cuImportExternalSemaphore()` 导入**，没法凭空创建 |
| `CUevent` | 能力最基础，但**状态是二值的、必须 signal 在 wait 之前、方向只有 device→device 和 device→host** |

IREE 的做法是以 `CUevent` 为基线，**用多个原语拼出缺失的能力**：

| 缺什么 | 怎么补 |
|-------|-------|
| 需要 64 位时间线，但 `CUevent` 是二值 | **每个 `CUevent` 当作时间线上的一个"时间点（timepoint）"** |
| 需要 wait-before-signal | **把工作压住不发给 GPU**，直到 host 侧的 semaphore 被 signal 过、或者已经有一个可以等的 `CUevent` |
| 缺 host→host 与 host→device | 用 CPU 侧的通知原语（`iree_event_t`）补 |

四个方向的具体处理：

- **CPU signal + CPU wait**：纯 host 侧，用 `iree_event_t`（包装 OS 原语）。时间线上记录所有 CPU wait timepoint，新值 signal 后遍历唤醒所有等待更早值的。
- **CPU signal + GPU wait**：CPU 没法 signal 一个 `CUevent`。**只能把提交批次缓存下来延迟发射**，等 CPU signal 过了目标值再放行 → 需要一个**待决动作队列（pending actions queue）**。
- **GPU signal**：只能通过 `CUevent`（二值）。用 `cuLaunchHostFunc()` 从 CPU 侧推进时间线，顺便复用 CPU signal 的逻辑去唤醒 CPU 等待者。
- **GPU signal + GPU wait**：可以走上面那条 host 中转的路，但**让 CPU 卷入 GPU 内部同步是性能灾难**。优化做法是记录时间线上所有 GPU signal，看到 GPU wait 请求时扫描时间线找一个能越过目标值的 GPU signal，直接用那个 `CUevent` 等。若还没见到对应的 signal，就把 wait 也记下来，等 signal 出现时再回填——这同时保证了 `CUevent` "先 record 后 wait" 的硬性要求。

两个实现上的坑：

1. `cuLaunchHostFunc()` 的文档规定"**host 函数内不得调用任何 CUDA API**"。所以不能在回调里直接推更多工作给 GPU，必须**通知另一个专门的线程**去调 CUDA API → 待决动作队列必须自带一个线程。
2. `cuLaunchHostFunc()` "会在当前已入队的工作之后调用，并**阻塞其后加入的工作**"。为了不让 host 阻塞主流水线，要**用一条专门的 `CUstream` 发射 host 函数**，并让它等原 stream 上的 `CUevent`。资源回收也一并在这里做。

最终 CUDA HAL driver 为此引入的数据结构（这套结构在别的 driver 里也高度可复用）：

| 结构 | 作用 |
|------|------|
| `iree_event_t` / `iree_event_pool_t` | CPU 通知原语及其对象池 |
| `iree_hal_cuda_event_t` / `iree_hal_cuda_event_pool_t` | 带引用计数的 `CUevent` 包装及对象池，引用归零自动回池 |
| `iree_hal_cuda_timepoint_t` / `..._pool_t` | 时间线上的一个 wait/signal 时间点，内部包一个 CPU event 或 GPU event |
| `iree_hal_cuda_timeline_semaphore_t` | 持有 CPU wait 与 GPU wait/signal 时间点列表 |
| `iree_hal_cuda_queue_action_t` | 一个待决队列动作（kernel launch 或 流序分配） |
| `iree_hal_cuda_pending_queue_actions_t` | 待决动作管理器，条件满足时把动作放行给 GPU |

> **写给要实现自研加速器 HAL driver 的你**：如果你的硬件只有二值 event，上面这一整套就是标准答案，照抄结构即可。如果你的硬件原生支持 64 位 timeline（如 Vulkan timeline semaphore、Level Zero event pool），实现会简单一个数量级。

#### 4.6.3 event

`iree_hal_event_t` 是命令缓冲**内部**的细粒度同步原语（对应 Vulkan 的 `VkEvent`）。注意 CUDA driver 文档的说明：

> `iree_hal_event_t` 目前编译器还没用到，所以 CUDA HAL driver 里还没实现。

**这是一个重要信号**：写新 driver 时，`event` 可以先不实现。

### 4.7 设备代码：executable / variant / export

这是"低成本切换/共存多后端"的核心机制，也是编译期与运行期的接缝。

**可跑** · 在主角模型上把这四层数出来（先 `bash scripts/run_phases.sh`）：

```bash
cd iree-lab
grep -nE 'hal.executable |hal.executable.variant|hal.executable.export' \
     out/phases/tiny_mlp.hal.mlir
```

> **和 fatbin 是同一个问题的同构答案**：CUDA 在一个 `.fatbin` 里按 `sm_*` 存多份 SASS，
> IREE 在一个 `hal.executable` 里按 target 存多份 variant。
> 两边都能亲手数：`cuobjdump -lelf` 数 image，`grep hal.executable.variant` 数 variant。
> 对照见 [`cuda-fatbin-learning-guide.md`](./cuda-fatbin-learning-guide.md)。

#### 4.7.1 四层结构

```mlir
hal.executable public @my_kernel {                          // ① 编译单元，运行时加载一次并缓存
  hal.executable.variant public @cuda_sm80                  // ② 目标特化的一份实现
      target(#hal.executable.target<"cuda", "cuda-nvptx-fb", {…}>) {
    hal.executable.export public @matmul ordinal(0)         // ③ 一个入口点
        layout(#hal.pipeline.layout<
            bindings = [#hal.pipeline.binding<storage_buffer>,
                        #hal.pipeline.binding<storage_buffer>,
                        #hal.pipeline.binding<storage_buffer>],
            constants = 2>)
        count(%device: !hal.device, %m: index, %n: index) -> (index, index, index) {
      // workgroup count region：由 workload 算出 3D grid 大小
      %x = affine.apply …
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module { /* 设备侧 IR，最终 codegen 成 PTX */ }
  }
  hal.executable.variant public @cuda_sm90 target(…) { … }  // 同一 executable 里的另一份
}
```

| 层 | 官方定义要点 |
|----|------------|
| **`hal.executable`** | "target-specific 的可执行模块"。**被当作独立的编译单元**，可含多个 export 并在内部共享代码。运行时在模块初始化时加载并缓存整个生命周期；带 `lazy` 属性则**推迟到首次使用才加载** |
| **`hal.executable.variant`** | "为支持多目标，每个 executable 可以有一个或多个 target-specific variant，它们在编译时**独立降低**，但在运行时表现为**一个 executable**（**类似 fat binary**）" |
| **`hal.executable.export`** | 一个入口点声明，携带静态可知的 IO 接口信息与派发元数据 |
| **`hal.executable.binary`** | 最终序列化的二进制（PTX / SPIR-V / ELF 字节流） |

还有 `hal.executable.source`：**未指定目标的源码形态**，用于手写 executable。这是"手工插入自定义 kernel"的入口（配合 `hal.dispatch.extern`）。

#### 4.7.2 variant 是怎么选中的

官方原文：

> Variant 基于它的 **target** 以及一个**可选的 condition op**（该 op 在给定运行时 `!hal.device` 时返回是否可用）来选择。若 executable 内没有任何 variant 可用，运行时加载会失败。**若多个 variant 可用，取找到的第一个可用者。**

对应 `hal.executable.condition` 区域。而 `hal.executable.export` 上还有一个 `condition` 区域 + `condition_fallback` 符号引用，可以把多个 export **串成 fallback 链**：

> Fallback 的 export 必须**完全匹配 layout 和 workload**，但可以改变任何其他属性（比如 workgroup size 或 translation 配置）。

> **这是 tuning 与异构适配的机制基础**：为同一个 kernel 编几份不同 tile size 的实现，运行时按设备特性选。

#### 4.7.3 两个关键属性

**`#hal.executable.target<backend, format, configuration>`** —— 怎么编：

```mlir
#hal.executable.target<"llvm-cpu", "system-elf-x86_64", {
  triple = "x86_64-unknown-linux-elf",
  cpu = "host",
  cpu_features = "host",
  abi = "lp32"
}>
```

**`cpu_features` 这一栏不是装饰，它直接决定向量宽度。**
[`run_variants.sh`](../../iree-lab/scripts/run_variants.sh) 实验 B 就是在拧它：

```bash
cd iree-lab && bash scripts/run_variants.sh
# 三份编译，只差 cpu_features：baseline / +avx2 / +avx512f,+avx512vl
# 然后数各自 kernel LLVM IR 里的向量类型
```

打出来是这样一张表：

```text
配置       <4 x float>    <8 x float>    <16 x float>   vmfb字节
baseline   ...            ...            ...            ...
avx2       ...            ...            ...            ...
avx512     ...            ...            ...            ...
```

这张表是 [`00-end-to-end-pipeline.md`](./00-end-to-end-pipeline.md) 那条优化目标
「内层循环走 8 宽向量」的**最终落点**。要紧的是：它不是任何单个 pass 的功劳 ——
linalg 层保住了并行语义、flow 层没把循环切碎、后端才有机会按 AVX 宽度打包，
**缺一环就退回 `<4 x float>`**。

**`#hal.device.target<deviceID, configuration, [executable_targets]>`** —— 跑在哪：

```mlir
#hal.device.target<"local", {
  device_configuration = …
}, [
  #hal.executable.target<"llvm-cpu", "embedded-elf-arm_32">,
  #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64">
]> : !hal.device
```

用它初始化一个 device global 时，"返回**第一个满足目标要求的设备**，没有匹配则返回 null。可选的 `ordinal` 索引用来在多个同构设备里选第 N 个。"

围绕设备选择，HAL 还有一整族属性：`#hal.device.select`（候选集）、`#hal.device.fallback`（回退）、`#hal.device.optimal`（择优）、`#hal.device.affinity`（亲和性）、`#hal.device.ordinal`、`#hal.device.alias`、`#hal.device.promise`（先声明后绑定）。**多设备编译的表达力主要来自这一族属性。**

#### 4.7.4 pipeline layout：host 与 device 的 ABI 契约

```mlir
#hal.pipeline.layout<
  bindings = [
    #hal.pipeline.binding<storage_buffer>,
    #hal.pipeline.binding<storage_buffer, ReadOnly>
  ],
  constants = 4
>
```

官方定义："指定与 executable 函数交互所用的布局信息，**让 host 代码能正确地把参数映射到底层目标特定的传参行为上**。"

设备侧 kernel 通过下面这组 op 看到这份契约（这组 op 出现在 `hal.executable.variant` 内部的 `builtin.module` 里）：

```mlir
%arg0 = hal.interface.binding.subspan layout(#layout) binding(0)
            offset(%c0) : memref<128x256xf32>
%c    = hal.interface.constant.load layout(#layout) ordinal(0) : i32
%id_x = hal.interface.workgroup.id[0]    : index
%cnt_x= hal.interface.workgroup.count[0] : index
%sz_x = hal.interface.workgroup.size[0]  : index
```

> **这五个 op 就是 IREE 的 kernel ABI**。任何后端的 codegen 最终都要把它们翻译成目标语言里的等价物（CUDA 的 `blockIdx.x`、SPIR-V 的 `WorkgroupId`、CPU 上的循环变量）。**接自研后端时，这是必须打通的第一件事。**

#### 4.7.5 executable_cache

`iree_hal_executable_cache_t` 负责把编译产物"准备"成设备可执行对象并缓存。CUDA 上的映射：

> `iree_hal_executable_t` 自然地映射到 `CUmodule`。编译器生成一个 FlatBuffer，内含 PTX 镜像以及入口点函数列表与关联元数据（名字、workgroup size、动态共享内存大小等）。运行时 CUDA HAL driver 加载 PTX 镜像，为各入口点创建 `CUfunction`。

`hal.executable.create` 是编译器发射的创建调用，`hal.executable.lookup` / `hal.executable.lookup.function` 用于按符号查找。

### 4.8 集合通信与多设备：channel / topology

**这一块 2022 年的 TinyIREE 论文完全没有，但对算力网分布式场景是最直接相关的部分。**

#### iree_hal_channel_t

头文件 `runtime/src/iree/hal/channel.h` 的注释直接给出了对标物：

```c
// A collective communication channel representing a single rank.
// Equivalent to:
//   MPI_Comm
//   ncclComm_t
//   ccl::communicator
typedef struct iree_hal_channel_t iree_hal_channel_t;
```

创建参数：

```c
typedef struct {
  iree_hal_channel_flags_t flags;
  iree_const_byte_span_t   id;     // 等价于 ncclUniqueId；留空则从环境变量取
  iree_string_view_t       group;  // 用户自定义的分组 key，用来区分多个 channel 组
  int32_t                  rank;   // 本 rank 序号；RANK_DEFAULT(-1) 表示从环境取
  int32_t                  count;  // 组内总参与者数；COUNT_DEFAULT(-1) 表示从环境取
} iree_hal_channel_params_t;
```

核心 API：

```c
iree_hal_channel_create(device, queue_affinity, params, &channel);
iree_hal_channel_split(base_channel, color, key, flags, &split);  // == MPI_Comm_split / ncclCommSplit
iree_hal_channel_query_rank_and_count(channel, &rank, &count);
```

`split` 的语义值得留意：`color == IREE_HAL_CHANNEL_NO_COLOR (-1)` 时返回 NULL channel，表示该 rank 不参与任何子组。**这正是表达 TP/PP/DP 多维并行组切分的原语**——用不同的 color 从全局 channel 切出张量并行组、流水并行组、数据并行组。

注意 `iree_hal_channel_create` 带 `queue_affinity`：**channel 绑定在设备的一组队列上**，与执行队列共享调度。

#### 编译器侧

```mlir
%channel = hal.channel.create device(%device : !hal.device)
               affinity(%affinity) flags("None")
               id(%id) group(%group) rank(%rank) count(%count) : !hal.channel
%sub = hal.channel.split<%channel : !hal.channel> color(%color) key(%key) flags("None") : !hal.channel
%rank, %count = hal.channel.rank_and_count<%channel : !hal.channel> : i32, i32

hal.command_buffer.collective<%cmd : !hal.command_buffer>
    channel(%channel : !hal.channel)
    op(<all_reduce with sum : f32>)
    send(%send : !hal.buffer)[%so, %sl]
    recv(%recv : !hal.buffer)[%ro, %rl]
    count(%elements)
```

支持的 `CollectiveKind`：`all_gather`、`all_reduce`、`all_to_all`、`broadcast`、`reduce`、`reduce_scatter`、`send`、`recv`、`send_recv`。
`CollectiveReductionOp`：`sum`、`product`、`minimum`、`maximum`、`average`。

**这基本覆盖了 NCCL 的全集**，`send`/`recv`/`send_recv` 的存在意味着流水并行的点对点通信也能表达。

> 关键点：**collective 是作为一条命令录进 command buffer 的**，和 dispatch 平级。这意味着通信与计算在同一套时间线里调度，编译器可以做 **计算-通信 overlap** 的重排——这是分布式训练里最重要的优化之一。

#### device topology

较新加入的 `#hal.device.topology` 属性，以 `stream.topology` 挂在顶层 module 上，**告知 Stream dialect 的 pass 设备间的连通性**：

```mlir
module attributes {
  stream.topology = #hal.device.topology<links = [
    (@device_gpu_0 -> @device_cpu   = {transparent_access = true, unified_memory = true}),
    (@device_gpu_1 -> @device_cpu   = {transparent_access = true}),
    (@device_cpu   -> @device_gpu_0 = {unified_memory = true}),
    (@device_cpu   -> @device_gpu_1 = {})
  ]>
}
```

它描述"设备之间的连接拓扑，包括内存共享与访问特性"。配合 `iree_hal_topology_t` / `iree_hal_topology_builder_t` / `iree_hal_device_group_t` 这几个运行时对象使用。

> **对算力网直接相关**：异构算力节点之间的带宽/延迟/是否统一内存差异巨大，这套拓扑属性是把这些信息喂给编译器做放置决策的官方入口。目前 IREE 主要面向**单进程多设备**，跨节点还需要自己在外层做——这是你的项目可能要补的地方。

### 4.9 编译器侧的 HAL dialect

HAL dialect 的 op 按用途分为几组，读 IR 时按组认即可（完整清单见 https://iree.dev/reference/mlir-dialects/HAL/）：

| 组 | 代表 op |
|----|--------|
| Allocator | `hal.allocator.allocate` / `.import` / `.select` / `.resolve_memory_properties` |
| Buffer | `hal.buffer.subspan` / `.load` / `.store` / `.length` / `.assert` / `.allocation.*` |
| Buffer view | `hal.buffer_view.create` / `.buffer` / `.dim` / `.rank` / `.element_type` / `.trace` |
| Channel | `hal.channel.create` / `.split` / `.rank_and_count` |
| Command buffer | `hal.command_buffer.create` / `.dispatch` / `.copy_buffer` / `.fill_buffer` / `.collective` / `.execution_barrier` / `.finalize` |
| Device | `hal.device.query` / `hal.device.queue.*` / `hal.device.resolve` / `hal.devices.count` / `hal.devices.get` |
| Executable | `hal.executable` / `.variant` / `.export` / `.binary` / `.create` / `.lookup` / `.condition` / `.constant.block` / `.calculate_workgroups` |
| Fence | `hal.fence.create` / `.join` / `.signal` / `.fail` / `.await` / `.query` |
| Interface（设备侧 ABI） | `hal.interface.binding.subspan` / `.constant.load` / `.workgroup.id` / `.count` / `.size` |
| Instrument（调试） | `hal.instrument.print` / `.value` / `.workgroup` / `.memory.load` / `.memory.store` |
| Pseudo / 边界 | `hal.tensor.import` / `.export` / `.barrier` / `.alias` / `.transients`、`hal.dispatch.extern`、`hal.device.memoize` |

#### 最该重点看的：tensor 边界 op

`hal.tensor.import` / `export` 是**张量世界与 buffer 世界的接缝**，它把第 3 章的"fence 绑定资源"落到了 IR 上：

```mlir
// 导入：外部 buffer_view → SSA tensor，可挂一个 wait fence
%t = hal.tensor.import wait(%wait_fence) => %view
       : !hal.buffer_view -> tensor<4x?xf32>{%dim}

// 屏障：给 tensor 打一个时间点，产出 signal fence
%ready = hal.tensor.barrier join(%result : tensor<…>) => %signal_fence : !hal.fence

// 导出：SSA tensor → 外部 buffer_view
%view2 = hal.tensor.export %ready : tensor<4x?xf32>{%dim} -> !hal.buffer_view
```

三个细节：

- **`wait` fence 可选**，不给就表示 buffer view 立即可用。
- **`consume` 标记表示所有权转移**：即使被导入的值仍有外部引用，资源也会"原子地从原持有者释放、被导入者持有"。这对应 [3.4](#34-stream-ordered-allocation峰值内存的关键) 里"用户把输入 buffer 的所有权转给程序，让程序可以提前复用这块存储"。
- **`source_encoding` / `target_encoding` 允许 ABI 面向的类型与内部表示不同**（须可 bitcast、动态维数量一致），用来做 rank-0 与 rank-N、不同 element type 之间的转换。
- **`on(%affinity)`** 指定这次导入/导出发生在哪个设备上。

#### 示例精讲：一个完整的 `@main` 边界

**可跑** · 在主角模型上看真实版本：

```bash
cd iree-lab && bash scripts/run_phases.sh
grep -nE 'hal.tensor.import|hal.tensor.barrier|hal.tensor.export|!hal.fence' \
     out/phases/tiny_mlp.abi.mlir out/phases/tiny_mlp.hal.mlir
```

`abi` 相位就是**专门做这件事的那一相** —— 之前 `input` 相位里 `@tiny_mlp` 还是个
收 `tensor<2x3xf32>` 的普通函数，`abi` 之后它的签名被改写成上面这个样子。
`run_phases.sh` 的行数表里，`input → abi` 那一跳增加的行，绝大部分就是这几行边界 op。

编译器为**带 fence 的异步 ABI** 生成的入口函数大致长这样：外面是 buffer_view 加一对 fence，里面是纯 tensor 世界。

```mlir
func.func @main(%input: !hal.buffer_view,
                %wait: !hal.fence,     // 用户：这个到达前别读我的输入
                %signal: !hal.fence)   // 用户：这个到达后你可以读输出
    -> !hal.buffer_view {
  // ① 进门：buffer_view → tensor，挂上 wait fence
  %t0 = hal.tensor.import wait(%wait) => %input
          : !hal.buffer_view -> tensor<4x8xf32>

  // ② 中间：普通 tensor 计算，看不到任何 fence
  %t1 = linalg.generic ... ins(%t0 : tensor<4x8xf32>) ... -> tensor<4x8xf32>

  // ③ 打时间点：把「%t1 算完了」绑到 signal fence 上
  %ready = hal.tensor.barrier join(%t1 : tensor<4x8xf32>) => %signal : !hal.fence

  // ④ 出门：tensor → buffer_view
  %out = hal.tensor.export %ready : tensor<4x8xf32> -> !hal.buffer_view
  return %out : !hal.buffer_view
}
```

> 形态示意：真实 dump 里外层通常还有 `iree.abi.*` 属性、参数名字符串与 `util.func` 包装，这里只留主干。

与 [3.3](#33-输入输出与-fence-的绑定) 里用户侧写法的对照：

| 用户侧 | IR 里的落点 | 含义 |
|--------|-----------|------|
| `(wait_fence, buffer_a)` 元组 | `hal.tensor.import wait(%wait) => %input` | fence 绑的是**这一个资源**，不是全局 barrier |
| 值类型参数 `42` | 普通 SSA 值传入，**不挂 fence** | 值不需要排序 |
| `(signal_fence, buffer_out)` | `hal.tensor.barrier ... => %signal` | 「算到这里」这个时间点被交回用户 |
| `signal_fence.wait()` | 不在这个函数里，在**调用方** | 函数本身立即返回 |

四个要点：

1. **函数是立即返回的**。`@main` 返回时 kernel 可能一条都没跑，`%out` 要等 `%signal` 到达才可读。
2. **`barrier` 必须在 `export` 之前**：先确定「什么时候算完」，才谈「把结果交出去」。
3. **多个输出共用一个 signal fence** 时，`join` 可以接多个 tensor——这就是它叫 join 的原因。
4. **`wait` 可选**：省掉表示输入立即可用（同步 ABI 就是这种形态，签名里连 fence 参数都没有）。

再往下一层，这三个 op 会落成：`import` 的 wait fence 接到 [4.5](#45-工作提交device-queue-api) 里 `hal.device.queue.execute` 的 `wait`，`barrier` 的 signal fence 接到它的 `signal`。**所以「边界三 op」本质是把用户的时间线接进程序内部的时间线。**

> **自测**：`@main` 返回之后、`%signal` 到达之前，用户去读 `%out` 对应的内存会读到什么？

另外两个值得知道的 pseudo op：
- **`hal.dispatch.extern`**：直接派发一个外部提供的 executable，**这是插入手写 kernel 的官方口子**。
- **`hal.device.memoize`**：把一段设备相关的构造过程记忆化（比如只建一次命令缓冲然后反复用）。

### 4.10 HAL 的三种形态：Full / Inline / Loader

这是很容易被忽略、但对嵌入式与极简部署至关重要的一点。IREE 有**三个 HAL 相关的 dialect / 运行时模块**：

| 形态 | dialect | 特点 | 适用 |
|------|---------|------|------|
| **Full HAL** | `hal` | 完整对象模型：命令缓冲、异步队列、semaphore、多设备 | 服务器、GPU、需要并发与流水 |
| **HAL/Inline** | `hal_inline` | "Inline HAL interop runtime module"。**同步内联执行**，没有命令缓冲、没有异步时间线 | 单线程、无 OS、极简同步场景 |
| **HAL/Loader** | `hal_loader` | "HAL inline executable loader"。**只保留 executable 的加载与直接调用**，不要完整设备模型 | 只需要"加载一段 kernel 然后调它" |

TinyIREE 论文里那四个"缩小开关"，其中三个落在这里：

| TinyIREE 开关 | 对应机制 |
|--------------|---------|
| 同步调度器 vs 异步任务调度器 | `local-sync` vs `local-task` HAL device |
| 精简 HAL | `hal_inline` / `hal_loader` 替代 full HAL |
| 静态库 vs 动态库 | executable 的打包形式（`--iree-llvmcpu-static-library-output-path`） |
| bytecode vs EmitC C 源 | VM 层，不在 HAL |

裸机构建时对应的 CMake 开关（官方 bare-metal 指南原文）：

```cmake
set(IREE_HAL_DRIVER_DEFAULTS OFF)
set(IREE_HAL_DRIVER_LOCAL_SYNC ON)              # 只启用本地同步 driver

set(IREE_HAL_EXECUTABLE_LOADER_DEFAULTS OFF)
set(IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF ON) # 只启用嵌入式 ELF 加载器
set(IREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE ON)

set(IREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS OFF)
set(IREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF ON)
```

> 从这组开关能读出 HAL 的可裁剪粒度：**driver、executable loader、executable plugin 三个维度各自独立可选**。

### 4.11 实现一个新 HAL driver 的清单

如果要给自研加速器接入 IREE，需要做的事：

**运行时侧**——每类对象实现一张 vtable（`iree_hal_*_vtable_t`，见各头文件末尾的 "implementation details" 段）：

| 对象 | 必要性 | 备注 |
|------|-------|------|
| `driver` | 必须 | 加载符号、枚举设备 |
| `device` | 必须 | 队列 API 的实现主体 |
| `allocator` | 必须 | 映射 MemoryType / BufferUsage 到你的分配接口 |
| `buffer` | 必须 | 裸指针包装 |
| `command_buffer` | 必须 | 若硬件无"录制"概念，用 `iree_hal_deferred_command_buffer_t` |
| `executable_cache` + `executable` | 必须 | 加载二进制、解析入口点元数据 |
| `semaphore` | 必须，且**最难** | 参照 [4.6.2](#462-为什么这个抽象是难的cuda-上的落地) 的 CUDA 方案 |
| `event` | **可先跳过** | 编译器目前不生成 |
| `channel` | 分布式才需要 | 对接你的通信库 |
| `file` / `pool` | 可选 | 参数加载与内存池优化 |

**编译器侧**：
1. 注册一个 **target backend**（产出某种 executable format），实现 codegen 到你的目标语言。
2. 定义 **device target**（`#hal.device.target<"your-device", …>`）与它接受的 executable target 列表。
3. 打通 kernel ABI：把 `hal.interface.binding.subspan` / `constant.load` / `workgroup.id/count/size` 翻译到你的目标语言。

**验证**：`runtime/src/iree/hal/cts/` 是 **HAL Conformance Test Suite**，新 driver 应当跑通它。

**树外接入**：CMake 选项 `IREE_EXTERNAL_HAL_DRIVERS` 允许不改 IREE 主干。

---

## 第 5 章 部署配置与命令行

### 5.1 编译

```bash
# 通用 CPU
iree-compile model.mlir \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  -o model_cpu.vmfb

# CUDA
iree-compile model.mlir \
  --iree-hal-target-device=cuda \
  -o model_cuda.vmfb

# 裸机（官方 bare-metal 指南原文示例）
iree-compile \
  --iree-stream-partitioning-favor=min-peak-memory \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=x86_64-pc-linux-elf \
  --iree-llvmcpu-debug-symbols=false \
  samples/models/simple_abs.mlir -o simple_abs.vmfb
```

常用 flag 速查：

| flag | 作用 |
|------|------|
| `--iree-hal-target-device=` | 指定运行时 HAL 设备（可多次指定做多设备编译） |
| `--iree-hal-local-target-device-backends=` | `local` 设备下用哪个编译后端 |
| `--iree-hal-list-target-backends` | 列出所有可用编译后端 |
| `--iree-stream-partitioning-favor=` | `max-concurrency`（默认）/ `min-peak-memory` |
| `--compile-to=<phase>` | **在指定相位停下输出 IR，学习 IREE 的头号工具** |
| `--mlir-print-ir-after-all` | 打印每个 pass 之后的 IR（配合 `--mlir-elide-elementsattrs-if-larger=8`） |
| `--iree-llvmcpu-target-cpu-features=` | 指定 ISA（`+avx2` / `+avx512f`），**直接决定向量宽度** |
| `--iree-hal-dump-executable-files-to=` | 把每个 kernel 的 `.ll` / `.s` / `.o` 全 dump 出来 |
| `--iree-llvmcpu-static-library-output-path=` | 产出静态库供裸机链接 |

> **标志名会随版本变**：`--iree-hal-target-device=` 曾经是 `--iree-hal-target-backends=`，
> `--iree-llvmcpu-target-cpu-features=` 曾经是 `--iree-llvmaot-...`，
> `--iree-hal-dump-executable-files-to=` 也有 `-intermediates-to` 的旧写法。
> `iree-lab/scripts/` 的做法是**先拿一个空模型试一次**再决定用哪个，
> 报 unknown option 时照 `iree-compile --help | grep -i <关键字>` 查即可。

最后两个 flag 是 [`run_variants.sh`](../../iree-lab/scripts/run_variants.sh) 实验 B、C 的全部内容：
dump 出来的 `.ll` **正是 [`llvm-learning-guide.md`](./llvm-learning-guide.md) 第 5 章的输入**。
IREE 负责"切到什么粒度"，LLVM 负责"这一块怎么编"——两份指南在这个文件上握手。

### 5.2 运行与查看

```bash
iree-run-module --list_drivers            # 列出可用 driver
iree-run-module --list_devices            # 列出可用设备
iree-run-module --dump_devices            # 打印设备详细能力
iree-run-module --device=cuda --module=model.vmfb \
                --function=main --input=1x224x224x3xf32=0
```

**可跑** · 主角模型的完整一轮（[`run_execute.sh`](../../iree-lab/scripts/run_execute.sh) 做的就是这件事）：

```bash
cd iree-lab
iree-compile --iree-hal-target-device=local \
             --iree-hal-local-target-device-backends=llvm-cpu \
             models/tiny_mlp.mlir -o out/execute/tiny_mlp.vmfb
iree-run-module --device=local-sync --module=out/execute/tiny_mlp.vmfb \
                --function=tiny_mlp --input="2x3xf32=1 2 3 -4 -5 -6"
```

期望输出 `[[2.5 3.5 4.5 7.5] [1 1 1 1]]`，推导写在
[`models/tiny_mlp.mlir`](../../iree-lab/models/tiny_mlp.mlir) 文件末尾，脚本会自动比对。

**这一步不是仪式感。** 前面所有相位都只是文本，只有这行"数值一致"能证明
flow 融了 op、stream 复用了内存、hal 换了 buffer 布局、vm 重排了调度之后，
**语义确实没变**。第二行输入全负，`relu` 把整行钳成 0 —— 挑这组数就是为了让 Relu 不是摆设。

Python 侧：

```python
import iree.compiler as ireec
import iree.runtime as ireert
ireec.query_available_targets()
ireert.system_setup.query_available_drivers()
```

---

## 第 6 章 与"算力网分布式基础设施"的对接点

把 HAL 的知识对回你的项目目标（参考 `paper-notes/01-efficient-training-distributed-infra.md`）：

| 你的需求 | IREE 里现成的机制 | 缺口 / 需要自己补的 |
|---------|-----------------|-------------------|
| **异构后端低成本切换** | `hal.executable` 的 **fat binary + variant + condition** 机制；compiler target 与 runtime device 的 N:M 解耦 | 自研加速器需要写 HAL driver + target backend（[4.11](#411-实现一个新-hal-driver-的清单)） |
| **分布式通信** | `iree_hal_channel_t`（对标 `ncclComm_t`/`MPI_Comm`）+ `hal.command_buffer.collective`，覆盖 NCCL 全集 | IREE 当前主要面向**单进程多设备**，**跨节点编排要在外层自己做** |
| **计算-通信 overlap** | collective 与 dispatch **在同一个命令缓冲、同一条时间线上**，编译器可统一重排 | 需要验证 Stream 层的分区策略对通信的处理是否够好 |
| **异构拓扑感知放置** | `#hal.device.topology`（unified_memory / transparent_access 等链路属性）喂给 Stream pass | 算力网的带宽/延迟/故障率等维度目前没建模 |
| **显存峰值控制** | **stream-ordered allocation** + `--iree-stream-partitioning-favor=min-peak-memory`；内存不足时自动串行化而非 OOM | 与 ZeRO / offload 这类策略的结合需要自己设计 |
| **异步流水编排** | timeline semaphore + fence，覆盖 host↔device 四方向；fence 可任意组 DAG；`hal.fence.fail` 传播错误 | —— |
| **容错** | fence 携带错误状态 | **重启/重算/checkpoint 全部没有**，需要在 IREE 之上建 |
| **动态 shape** | Flow/Stream/HAL 全链路有 `{%dim}` 动态维支持 | LLM 的变长序列 + KV cache 场景要重点验证成熟度 |

**结论性的判断**：IREE 的 HAL 给了你一套**已经工业级验证的"设备抽象 + 异步执行 + 集合通信"底座**，尤其是 executable variant 机制和 timeline semaphore 语义，直接照搬即可；但**跨节点调度、弹性容错、大模型特有的并行策略**这三块 IREE 基本是空白，是你的项目真正要做的增量。

---

## 第 7 章 学习路径：最小必要集与动手清单

### 7.1 必须掌握（会反复用到）

1. **执行模型六概念**：program / module / context / invocation / timeline / fence，以及"invocation 是排布工作而非执行工作"。
2. **HAL 对象模型**：能默画出 [4.2](#42-对象模型总览) 的那张图，说清每个对象负责什么。
3. **timeline semaphore 的四条能力**（64 位单调 / 双向顺序 / 四方向 / 多等待），以及为什么 `CUevent` 满足不了。**这是 HAL 里最硬的知识点。**
4. **executable / variant / export / binary 四层结构**与 variant 选择机制。
5. **kernel ABI 五个 op**：`hal.interface.binding.subspan`、`constant.load`、`workgroup.id/count/size`。
6. **`hal.tensor.import/export/barrier`** 这三个边界 op 怎么把 fence 绑到资源上。
7. **stream-ordered allocation** 为什么能压住峰值内存。
8. **`hal.command_buffer.dispatch`** 的完整签名（3D workgroup + 4 字节 constants + bindings）。

### 7.2 可以先跳过

- HAL 的 instrument / profile 系列（调试用，需要时再看）。
- `pool` / `pool_set` / `file` / `IO/Parameters`（参数加载优化）。
- Codegen 内部（`IREECodegen` / `IREEGPU` / LLVMGPU / SPIRV pass 细节）——这是另一个完整领域，除非你要做 kernel 生成。
- VM bytecode 的指令格式、EmitC 路径。
- `webgpu` / `metal` 等你用不到的后端。

### 7.3 动手清单（按顺序做）

**第一步（前三小步已经封装成 lab，直接跑）**

```bash
cd iree-lab
pip install -r requirements.txt        # iree-base-compiler + iree-base-runtime
bash scripts/run.sh 2>&1 | tee out/all.log
```

| 脚本 | 回答的问题 | 先读哪个产物 |
|------|-----------|-------------|
| `run_phases.sh` | 编译器**做了什么**？每相位新增哪类信息？ | `out/PHASES.md` 的行数表 |
| `run_execute.sh` | 这些变换**没改语义**吗？ | 手算 vs 实际输出的比对结果 |
| `run_variants.sh` | **我能改什么**？ | 实验 B 的 `<4/8/16 x float>` 计数表 |

**跑完先回答这五问**（答不上来就回对应章节，不要往下走）：

1. `flow` 相位固化了什么？为什么这个决定之后再也改不了？（→ [2.2](#22-三个最需要理解的转折点)）
2. `!stream.resource` 的 `external` / `transient` / `variable` 各自意味着什么约束？（→ [3.4](#34-stream-ordered-allocation峰值内存的关键)）
3. 一个 kernel 为什么会有多个 variant？和 fatbin 什么关系？（→ [4.7](#47-设备代码executable--variant--export)）
4. `.vmfb` 里装了哪三样东西？为什么运行时不用再带编译器？（→ [1 章示例](#示例精讲一个-abs-模型穿过三层)）
5. 静态 shape 比动态 shape 在 stream 层省掉了哪一类指令？为什么**丢了补不回来**？（→ `run_variants.sh` 实验 A）

`--compile-to=<phase>` 会在该相位结束后停下并输出**完整 IR**（不是产物），这是学 IREE 最高效的一个 flag。若还想看单个 pass 的效果，再叠加 `--mlir-print-ir-after-all --mlir-elide-elementsattrs-if-larger=8`。

**要在 `--compile-to=hal` 的输出里找到并读懂这些**：
- `hal.executable` / `variant` / `export` 的嵌套结构，以及 export 上的 `layout` 和 `count` region
- `hal.command_buffer.create` → 一串命令 → `hal.command_buffer.finalize`
- `hal.device.queue.execute` 上的 `wait` / `signal` fence
- `hal.tensor.import` / `export` 在函数边界的位置
- variant 内部 `builtin.module` 里的 `hal.interface.*` op

**第二步：观察设备，并确认换 device 不用改任何东西**

```bash
iree-run-module --dump_devices          # 打印本机设备详细能力
# run_execute.sh 第 ③ 步已经用同一个 .vmfb 跑了 local-sync 和 local-task 两遍：
iree-run-module --device=local-sync --module=out/execute/tiny_mlp.vmfb \
                --function=tiny_mlp --input="2x3xf32=1 2 3 -4 -5 -6"
iree-run-module --device=local-task --module=out/execute/tiny_mlp.vmfb \
                --function=tiny_mlp --input="2x3xf32=1 2 3 -4 -5 -6"
```

**调度策略换了，结果不变。** 能做到这点靠的是 stream 相位已经把依赖关系**显式写进 IR**，
而不是靠宿主代码里的调用顺序隐式表达 —— 这是 [4.1.1](#411-三条设计取向理解-hal-的钥匙)
那三条设计取向的直接回报。

**第三步：精读三份源码**（按优先级）

1. `docs/website/docs/developers/design-docs/cuda-hal-driver.md` —— **最高优先级**。它是唯一一份把"HAL 抽象与真实硬件的落差"讲透的文档。
2. `runtime/src/iree/hal/semaphore.h` + `runtime/src/iree/hal/drivers/cuda/` 下的 semaphore / event pool / pending queue actions 实现。
3. `runtime/src/iree/hal/command_buffer.h` + `runtime/src/iree/hal/utils/deferred_command_buffer.h`。

**第四步：读一个最简单的完整 driver**

`runtime/src/iree/hal/drivers/local_sync/` —— 同步本地 driver，是所有 driver 里最短的，适合当作"实现 HAL 需要写多少东西"的标尺。然后对照 `runtime/src/iree/hal/cts/` 看一致性测试要求什么。

**第五步：多设备与集合通信**

编译一个带两个 device target 的程序，观察 `#hal.device.affinity` / `#hal.device.select` 怎么出现在 IR 里；再看 `hal.command_buffer.collective` 在多设备程序里的形态。

---

## 附录：一页速查

```
【执行模型】 program ⊃ context ⊃ (module × N)；invocation 排布工作到 timeline，fence 定序
【流水线】   input → abi → preprocessing → global-optimization → dispatch-creation
             → flow → stream → executable-{sources,configurations,targets} → hal → vm
【HAL 五职责】枚举设备 / 定义设备代码 / 分配内存 / 组织延迟提交 / 显式同步

【HAL 对象】
  driver_registry → driver → device
  device ─ allocator → buffer → buffer_view
         ─ command_buffer（dispatch/copy/fill/update/collective/barrier）
         ─ queue API（alloca/dealloca/execute/read/write/copy/fill/barrier/flush）
         ─ semaphore（64 位单调时间线）→ fence（timepoint 集合）
         ─ executable_cache → executable ⊃ variant ⊃ export ⊃ binary
         ─ channel（NCCL/MPI 对标）、topology / device_group

【关键属性】
  #hal.executable.target<backend, format, config>    ── 怎么编
  #hal.device.target<id, config, [exec_targets]>     ── 跑在哪
  #hal.pipeline.layout<bindings=[...], constants=N>  ── host↔device ABI
  #hal.device.topology<links=[...]>                  ── 多设备拓扑

【kernel ABI 五 op】
  hal.interface.binding.subspan / constant.load / workgroup.id / workgroup.count / workgroup.size

【边界三 op】
  hal.tensor.import (wait fence) / hal.tensor.barrier (signal fence) / hal.tensor.export

【三种 HAL 形态】 hal（完整） / hal_inline（同步内联） / hal_loader（仅加载执行）
```
