# LLVM 学习文档：核心链路 + 关键概念蒸馏

> **本文档的定位**
> - 基于 **LLVM 官方文档主干版本**（`llvm/docs/` 下的 LangRef、CodeGenerator、NewPassManager、Passes、AliasAnalysis、Vectorizers、TableGen 等）蒸馏，讲的是**今天的 LLVM 实际长什么样**。
> - 与 [`paper-notes/02-llvm.md`](../paper-notes/02-llvm.md) 分工明确：那篇是 **2004 年 CGO 论文**的笔记，讲的是"当年为什么要这样设计"（lifelong compilation、IR 三态等价、link-time 优化的动机）。本文讲**结构与机制**，尤其是论文里完全没有的**后端 CodeGen 七阶段**和**现代 New Pass Manager**。
> - 服务目标：MLIR / IREE / TVM 这些栈最终都落到 LLVM，你需要知道"交给 LLVM 之后发生了什么"以及"在哪些地方可以插手"。
>
> **先修与邻接材料**
> - **先修**：[`ai-compiler-foundations.md`](./ai-compiler-foundations-learning-guide.md) §6（SSA / Pass / 渐进 lowering）、§7（kernel 与 ISA 层级）。
> - **动手**：[`../../llvm-hello-compile/`](../../llvm-hello-compile/)。
> - **上游**：谁在给 LLVM 喂 IR —— [`mlir-learning-guide.md`](./mlir-learning-guide.md)（`llvm` dialect）与 [`iree-learning-guide.md`](./iree-learning-guide.md)（LLVM-CPU / NVPTX 后端）。
> - **下游**：GPU 侧 `NVPTX → PTX` 之后如何打包分发 —— [`cuda-fatbin-learning-guide.md`](./cuda-fatbin-learning-guide.md) · [`../../tvm-fatbin-lab/`](../../tvm-fatbin-lab/) CUDA 轨。
> - **图级委托（另一层问题）**：[`../../onnx-delegate-lab/`](../../onnx-delegate-lab/)。
>
> **一句话读法**：如果只有一小时，读[第 1 章的总图](#第-1-章-一张总图从源码到机器码)、[2.7 poison 语义](#27-undef--poison--freeze现代-llvm-最容易踩的坑)、[第 5 章后端七阶段](#第-5-章-后端-codegen七个阶段重点)。

---

### 本篇在链路中的位置

> 全局链路见 [`00-end-to-end-pipeline.md`](./00-end-to-end-pipeline.md)。本篇是**第 ⑥ 站：指令生成**。

```text
… ④ 调度 ──▶ ⑤ 多层降低 ──▶ 【⑥ 指令生成 ← 本篇】──▶ ⑦ 打包与运行时
                (MLIR)          (LLVM)                  (fatbin / IREE)
```

| | |
|--|--|
| **上游交给我** | LLVM IR —— 以及一批**必须由上游主动写上**的属性（`noalias` / `align` / `contract`）。上游不写，这里无从补救 |
| **我固化** | 真实指令、真实寄存器。到这一步「这是一次矩阵乘」已经彻底看不见了 |
| **我交给下游** | 一段针对某个具体架构的机器码 |
| **本篇的主角** | 图级主角 `tiny_mlp` 里 Gemm 的最内层，也就是 [`../../llvm-hello-compile/src/kernel.c`](../../llvm-hello-compile/src/kernel.c) 里的 `axpy`。下面所有「示例精讲」都取自这个文件，可直接 `bash scripts/tour.sh` 跑出来对照 |

**本篇讨论的四个决定，前两个的输入全部来自上游**——这是整条链路上最容易断的一节：

| 决定 | 依据 | 上游不给会怎样 | 本篇章节 |
|------|------|--------------|---------|
| 能不能向量化 | 别名分析（`noalias`） | 只能 MayAlias → 运行时检查 + 双版本循环 | [§4.2](#42-别名分析四种回答与链式结构) · [§4.5](#45-向量化两个正交的向量化器) |
| 向量选多宽 | TTI 的目标信息 | 按基线 128-bit 选 4 宽而不是 8 宽 | [§4.6](#46-targettransforminfo中端如何询问目标) |
| 乘加合不合成 FMA | `contract` 标志 | 两条指令、两次舍入 | [§2.6](#26-属性与标志告诉优化器你可以假设什么) |
| 寄存器够不够 | 活跃区间分析 | 循环体内插 spill/reload | [§5.6](#56-寄存器分配) |

---

## 目录

- [第 1 章 一张总图：从源码到机器码](#第-1-章-一张总图从源码到机器码)
- [第 2 章 LLVM IR：语言参考的蒸馏](#第-2-章-llvm-ir语言参考的蒸馏)
- [第 3 章 Pass 基础设施（New Pass Manager）](#第-3-章-pass-基础设施new-pass-manager)
- [第 4 章 中端：核心分析与变换](#第-4-章-中端核心分析与变换)
- [**第 5 章 后端 CodeGen：七个阶段**](#第-5-章-后端-codegen七个阶段重点)
- [第 6 章 TableGen：把目标描述变成代码](#第-6-章-tablegen把目标描述变成代码)
- [第 7 章 与 MLIR / AI 编译器的接缝](#第-7-章-与-mlir--ai-编译器的接缝)
- [第 8 章 工具链与观察手段速查](#第-8-章-工具链与观察手段速查)
- [第 9 章 学习路径：最小必要集与动手清单](#第-9-章-学习路径最小必要集与动手清单)
- [附录：一页速查](#附录一页速查)

---

## 第 1 章 一张总图：从源码到机器码

**最重要的一个认知先说在前面**：很多人以为"LLVM = LLVM IR"，这是错的。从 `.c` 到 `.o`，程序会依次经过**四种不同的中间表示**，每一种有自己的数据结构、自己的优化 pass、自己的调试手段：

```
                    ┌───────────────── 前端（Clang / MLIR / Rust / Swift ...）
                    ▼
   ①  LLVM IR       Module / Function / BasicBlock / Instruction
       .ll / .bc    类型化 SSA，无限虚拟寄存器，目标无关
                    │
                    │  ◄── 中端优化（opt，New Pass Manager）
                    │      mem2reg / instcombine / GVN / LICM / inline / 向量化 ...
                    ▼
                    │  ◄── CodeGenPrepare（IR 级，但已属后端）
                    ▼
   ②  SelectionDAG  SDNode / SDValue，每个基本块一张 DAG
       （或 GlobalISel 直接走 ③）
                    │  ◄── build → combine → legalize types → combine
                    │      → legalize ops → combine → select → schedule
                    ▼
   ③  Machine IR    MachineFunction / MachineBasicBlock / MachineInstr
       .mir         目标指令 + 虚拟寄存器，初期是 SSA 形态
                    │  ◄── machine SSA opts → 寄存器分配（含 PHI 消解）
                    │      → prologue/epilogue → late opts
                    ▼
   ④  MC 层         MCInst / MCStreamer / MCSymbol / MCSection
                    只有标签、指令、节区，没有"全局变量""跳转表"这些高层概念
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   MCAsmStreamer            MCObjectStreamer
      .s 汇编                  .o 目标文件
```

对应的工具与观察方式：

| 阶段 | 数据结构 | 主要工具 | 怎么看 |
|------|---------|---------|--------|
| ① LLVM IR | `Module`/`Function`/`BasicBlock`/`Instruction` | `clang -emit-llvm`、`opt`、`llvm-as`/`llvm-dis` | `-S` 直接看文本 |
| ② SelectionDAG | `SDNode`/`SDValue` | `llc` 内部 | `llc -debug-only=isel-dump`、`-view-isel-dags` |
| ③ Machine IR | `MachineInstr` | `llc` 内部 | `llc -stop-after=<pass>` 输出 `.mir` |
| ④ MC | `MCInst` | `llc`、`llvm-mc` | `llc -show-mc-encoding`、`llvm-objdump -d` |

> **为什么要分这么多层**：每一层丢掉一部分目标无关性、换来一部分目标信息。LLVM IR 完全不知道有多少个寄存器；SelectionDAG 知道哪些类型/操作合法；MachineIR 知道具体指令但还用虚拟寄存器；MC 层连"这是个全局变量"都不知道了，只剩符号和字节。**这就是渐进式 lowering 在 LLVM 内部的体现**——和 MLIR 的多 dialect 是同一个思想，只是 LLVM 的层数固定、不可扩展（这正是 MLIR 要解决的问题）。

#### 示例精讲：一个函数走完四层 IR

**可跑** · 源码 [`llvm-hello-compile/src/kernel.c`](../../llvm-hello-compile/src/kernel.c) 的 `scale8` ·
命令 `cd llvm-hello-compile && bash scripts/tour.sh` · 产物 `out/tour/`

拿动手项目里现成的函数走一遍，每层看一眼：

```c
int scale8(int x) { return x * 8; }
```

**① LLVM IR**

```bash
clang -O2 -S -emit-llvm -fno-discard-value-names kernel.c -o k.ll
```

```llvm
define i32 @scale8(i32 %x) {
entry:
  %mul = shl i32 %x, 3        ; instcombine 已把 mul 8 规范成 shl 3
  ret i32 %mul
}
```

这一层的内存对象树，就是第 3 章里 Pass 拿到手的东西：

```text
Module "kernel.c"
├─ target datalayout / triple
└─ Function "scale8"  : i32 (i32)
   ├─ Argument  %x
   └─ BasicBlock "entry"
      ├─ BinaryOperator  shl (%x, 3)     ← 这条指令本身就是值 %mul
      └─ ReturnInst      ret (%mul)
```

**这里有个容易滑过去的点**：`%mul` 不是"变量名"，而是**那条指令对象的名字**。SSA 里"值"和"定义它的指令"是同一个对象，`ret` 的操作数指针直接指向那条 `shl`——所以遍历 use-def 不需要查符号表。

**② SelectionDAG → ③ 指令选择刚结束的 MIR**

```bash
llc -mtriple=x86_64-- -O2 -stop-after=finalize-isel k.ll -o k.isel.mir
```

```text
bb.0.entry:
  %0:gr32 = COPY $edi                        ; 参数按调用约定从物理寄存器进来
  %1:gr32 = SHL32ri %0:gr32(tied-def 0), 3, implicit-def dead $eflags
  $eax    = COPY %1:gr32
  RET64 implicit $eax
```

> 形态示意：实际 dump 以本地 llc 版本为准。

已经是**目标指令**了（`SHL32ri` 是 X86 的 opcode，不再是 IR 的 `shl`），但寄存器还是 `%0` / `%1` 这样的**虚拟寄存器**，只有 ABI 强制的 `$edi` / `$eax` 是物理的——这正是"第 1 阶段产出 SSA 虚拟寄存器 + 少量物理寄存器"的字面意思。

**③ 寄存器分配之后的 MIR**

```bash
llc -mtriple=x86_64-- -O2 -stop-after=virtregrewriter k.ll -o k.ra.mir
```

```text
bb.0.entry:
  renamable $eax = COPY $edi
  renamable $eax = SHL32ri renamable $eax(tied-def 0), 3, implicit-def dead $eflags
  RET64 implicit $eax
```

> 形态示意：`SHL32ri` 的目标与源必须同寄存器（two-address），所以要先 copy；x86 后端也常改用一条 `LEA` 来省掉这次 copy。以本地 llc 输出为准。

**④ 汇编 / MC**

```bash
llc -mtriple=x86_64-- -O2 k.ll -o k.s
llc -mtriple=x86_64-- -O2 -show-mc-encoding k.ll -o -
```

到这一层，"函数""参数""类型"全都没了，只剩标签、指令和字节。`-show-mc-encoding` 会在每条指令后面追加 `# encoding: [...]`，那串字节就是 `MCInst` 经编码器出来的最终产物；`llvm-objdump -d` 是同一层的反方向。

**四层并排**：

| 层 | 寄存器 | 类型信息 | 控制流的载体 | 还知道什么 |
|----|--------|---------|-------------|-----------|
| ① LLVM IR | 无限虚拟 SSA 值 | IR 类型（`i32`） | BasicBlock + terminator | 函数、全局变量、属性、metadata |
| ② SelectionDAG | 边即值，无寄存器概念 | MVT（机器类型） | **每块一张图**，块间不相连 | 哪些类型/操作对目标合法 |
| ③ MIR | 虚拟寄存器 → 物理寄存器 | 寄存器类（`gr32`） | MachineBasicBlock | 目标指令、栈帧、调用约定 |
| ④ MC | 只有物理寄存器编号 | 无 | 只有标签 | 符号、节区、字节编码 |

> **自测**：`-stop-after=finalize-isel` 的输出里 `$edi` 是物理寄存器，为什么它的存在不违反"这一阶段用虚拟寄存器"？

---

## 第 2 章 LLVM IR：语言参考的蒸馏

### 2.1 三态与顶层结构

LLVM IR 有三种**完全等价、可无损互转**的形态：

| 形态 | 用途 | 转换工具 |
|------|------|---------|
| 文本 `.ll` | 人读、写测试 | `llvm-dis` |
| 位码 `.bc` | 序列化、LTO 传递 | `llvm-as` |
| 内存 `Module` | 编译器内部 | —— |

一个 Module 的顶层构造：

```llvm
; 目标描述——决定了指针宽度、对齐、字节序
target datalayout = "e-m:e-p270:32:32-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Point = type { i32, i32 }          ; 具名结构体类型

@g = global i32 42, align 4                ; 全局变量
@.str = private unnamed_addr constant [6 x i8] c"hello\00"
@alias_g = alias i32, ptr @g               ; 别名
@llvm.global_ctors = appending global ...  ; 特殊的 intrinsic 全局变量

declare i32 @external_fn(i32)              ; 只声明不定义
define i32 @f(i32 %x) #0 { ... }           ; 定义，#0 是属性组

attributes #0 = { nounwind willreturn memory(none) }   ; 属性组
!llvm.module.flags = !{!0}                 ; 具名 metadata
!0 = !{i32 1, !"wchar_size", i32 4}
```

三个容易忽略但很关键的东西：

- **`datalayout` 是唯一必须的目标描述**。官方原话：`DataLayout` 是"唯一必需的目标描述类，也是唯一不可扩展的类"。它规定结构体布局、各类型对齐、指针宽度、大小端。**IR 的语义依赖它**——同一份 IR 配不同 datalayout 含义会变。
- **linkage 类型**决定符号在链接时怎么处理：`private` / `internal`（模块内）、`external`、`available_externally`、`linkonce` / `weak`（可被丢弃/覆盖）、`linkonce_odr` / `weak_odr`（C++ inline 函数用）、`appending`（只用于 `@llvm.global_ctors` 这类数组）、`common`。
- **属性组 `#0`** 把重复的函数属性提出来复用，是 IR 体积优化，读 IR 时要记得往文件末尾找。

### 2.2 类型系统

LangRef 把类型分成 **first class**（可以作为指令结果）和其他：

**Single value types**（CodeGen 视角下"能放进寄存器"的类型）：

| 类型 | 语法 | 说明 |
|------|------|------|
| 整数 | `iN` | 任意位宽，1 到 2²³（约 800 万位）。`i1` 就是布尔 |
| 字节 | `bN` | **较新加入**。表示"运行时无法确定是指针还是别的东西"的原始内存数据，每一位可以是整数位、指针值的一部分、或 poison。用于 `memcpy` lowering、union、未初始化数据。**只在中端有意义**——到 IR→MIR 边界就被降成同宽度的 `iN` |
| 浮点 | `half` `bfloat` `float` `double` `fp128` `x86_fp80` `ppc_fp128` | —— |
| 指针 | `ptr` / `ptr addrspace(N)` | **opaque pointer**：不再携带"指向什么类型"。类型信息移到访问指令上 |
| 向量 | `<N x T>` / `<vscale x N x T>` | 见下 |
| 目标扩展类型 | `target("...")` | 目标特有的不透明类型（如 AMX tile、RISC-V vector tuple） |

**向量类型**值得单独说，因为它是 AI 编译栈落到 LLVM 时的主要载体：

```llvm
<4 x i32>            ; 定长向量：正好 4 个 i32
<8 x float>
<4 x ptr>            ; 指针向量（gather/scatter 用）
<vscale x 4 x i32>   ; 可伸缩向量：元素数是 4 的 vscale 倍
```

`vscale` 是一个**编译期未知、运行时对所有可伸缩向量都相同的正的 2 的幂**。这是为 ARM SVE / RISC-V RVV 这类"向量长度不定"的 ISA 设计的。官方措辞很精确："一个特定可伸缩向量类型的大小在 IR 内是常量，即使它的确切字节数直到运行时才能确定。"

**Aggregate types**（不是 single value，不能直接进寄存器）：

- 数组 `[4 x i32]`、`[2 x [3 x float]]`
- 结构体 `{ i32, float }`（对齐按 datalayout）、`<{ i8, i32 }>`（packed，无填充）

**其他**：`label`（基本块地址）、`token`（不能被检查/复制的一次性令牌）、`metadata`、`void`、函数类型 `i32 (i32, ...)`。

### 2.3 SSA、基本块与 phi

- 每个函数由若干**基本块**组成，第一个块是 entry（不能有前驱）。
- **每个基本块必须以且仅以一条 terminator 结尾**：`ret` / `br` / `switch` / `indirectbr` / `invoke` / `callbr` / `resume` / `catchswitch` / `catchret` / `cleanupret` / `unreachable`。
- terminator 显式列出后继，所以 **CFG 是显式的**，不需要额外数据结构去推断。
- 每个值只被定义一次（SSA），使用点必须被定义点**支配（dominate）**。
- 汇合点用 `phi` 表达：

```llvm
merge:
  %r = phi i32 [ %a, %then ], [ %b, %else ]
```

`phi` 必须放在基本块的最开头（在所有非 phi 指令之前），且**每个前驱恰好一个入口**。

> **速记**：[../notes/llvm-phi.md](../notes/llvm-phi.md) —— 汇合选值、两条规则、与 MLIR block argument 的对应。

> **和 MLIR 的对比**：MLIR 用**基本块参数（block argument）**代替 `phi`——`^bb2(%r: i32):`，语义等价但避免了"phi 必须在块首""前驱顺序必须匹配"这些结构约束。这是 MLIR 相对 LLVM 的一个有意改进。

> **速记**：[../notes/mlir-block-arg-ssa.md](../notes/mlir-block-arg-ssa.md) —— 二者表达到达定值的汇合；不是分支、≠ 活跃变量；SSA 值如何对应源变量。

#### 示例精讲：`clamp0` 的三个基本块与一个 phi

**可跑** · 源码 [`llvm-hello-compile/src/kernel.c`](../../llvm-hello-compile/src/kernel.c) 的 `clamp0` ·
命令 `cd llvm-hello-compile && bash scripts/tour.sh`（第 2 站）· 产物 `out/tour/`

第 2 站会把下面三份 IR 并排打出来。

```c
int clamp0(int a, int b) {
  int y = a + b;
  int r;
  if (y > 0) r = y;
  else       r = 0;
  return r;
}
```

源码里 `r` 被赋值两次；SSA 不允许，所以汇合点必须选值。**同一个函数在三份 IR 里长三个样，把 SSA 的来龙去脉一次讲完。**

**① `-O0`：还不是 SSA，`r` 是一块内存**

```bash
clang -O0 -S -emit-llvm -Xclang -disable-O0-optnone -fno-discard-value-names \
      src/kernel.c -o 00_O0.ll      # tour.sh 第 0 站产物
```

```llvm
%r = alloca i32                       ; r 是栈上一块内存
br i1 %cmp, label %if.then, label %if.else
if.then:  store i32 %add, ptr %r      ; 两条路径各写一次 —— 内存允许被多次赋值
if.else:  store i32 0,    ptr %r
if.end:   %0 = load i32, ptr %r
```

**② `mem2reg` 之后：`alloca` 归零，`phi` 出现**

```bash
opt -S -passes=mem2reg 00_O0.ll -o 03_mem2reg.ll
```

```llvm
define i32 @clamp0(i32 %a, i32 %b) {
entry:
  %add = add nsw i32 %a, %b
  %cmp = icmp sgt i32 %add, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  br label %if.end

if.else:
  br label %if.end

if.end:
  %r.0 = phi i32 [ %add, %if.then ], [ 0, %if.else ]   ; ← 汇合点选值
  ret i32 %r.0
}
```

> 形态示意：块名与 SSA 编号以本地 clang/opt 版本为准；关键是 `alloca` 消失、`phi` 出现、入口数等于前驱数。

读法：**从 `%if.then` 进来 `%r.0` 就是 `%add`，从 `%if.else` 进来就是 `0`**。`phi` 不做任何计算、不产生任何机器指令，它只是把"这条边上该带哪个值"写下来。

**③ `-O2` 之后：`phi` 又不见了，变成 `select`**

```llvm
define i32 @clamp0(i32 %a, i32 %b) {
entry:
  %add  = add nsw i32 %a, %b
  %cmp  = icmp sgt i32 %add, 0
  %r.0  = select i1 %cmp, i32 %add, i32 0     ; 分支被抹平，CFG 只剩一个块
  ret i32 %r.0
}
```

**这两跳是两件不同的事，非常容易混**：

| 跳 | 谁干的 | 干了什么 | 变的是 |
|----|-------|---------|-------|
| ① → ② | `mem2reg` | 把内存变量提升成寄存器值 | **数据表示**（进入 SSA） |
| ② → ③ | `simplifycfg` 的 if-conversion | 两条短路径合并成一条无分支路径 | **控制流**（消除分支） |

`select` 还是 [2.7 节](#27-undef--poison--freeze现代-llvm-最容易踩的坑)毒性屏障那条规则的主角：**没被选中的那一臂即使是 poison 也不传染**——正因为有这个保证，if-conversion 才是合法的。少了它，把两条分支都无条件求值就可能引入原本不会发生的 UB。

对应的内存对象（Function Pass 在第 ② 步拿到的就是这棵树）：

```text
Function "clamp0"
├─ Argument %a, %b
├─ BasicBlock "entry"      preds: —                 succs: if.then, if.else
│    ├─ BinaryOperator  add nsw
│    ├─ ICmpInst        sgt
│    └─ BranchInst      br i1
├─ BasicBlock "if.then"    preds: entry             succs: if.end
│    └─ BranchInst      br
├─ BasicBlock "if.else"    preds: entry             succs: if.end
│    └─ BranchInst      br
└─ BasicBlock "if.end"     preds: if.then,if.else   succs: —
     ├─ PHINode         [%add, if.then], [0, if.else]   ← 必须是块里第一条
     └─ ReturnInst      ret %r.0
```

**注意 `phi` 的入口顺序要和 `if.end` 的前驱列表对得上**：前驱有两个，入口就必须恰好有两个。如果某个前驱块被 pass 删掉或复制了，所有以它为前驱的 `phi` 都要同步改——这正是"改 CFG 的 pass 不能声明 `preserveSet<CFGAnalyses>()`"的原因之一。

顺带解释了第 ③ 步为什么 `phi` 会消失：`simplifycfg` 发现 `if.then` / `if.else` 都是空块，把它们连同 `if.end` 一起折进 `entry`——前驱没了，`phi` 自然也就没了存在的理由。**`phi` 的生命周期完全绑在 CFG 形状上**，这是它和普通指令最不一样的地方。

同一段语义在 MLIR 里用 **block argument**：

```mlir
func.func @clamp0(%a: i32, %b: i32) -> i32 {
  %c0  = arith.constant 0 : i32
  %y   = arith.addi %a, %b : i32
  %cmp = arith.cmpi sgt, %y, %c0 : i32
  cf.cond_br %cmp, ^then, ^merge(%c0 : i32)   // 值挂在「边」上
^then:
  cf.br ^merge(%y : i32)
^merge(%r: i32):                              // 块声明自己要收什么
  return %r : i32
}
```

| | LLVM `phi` | MLIR block argument |
|--|-----------|---------------------|
| 值写在哪 | **汇合块内**，列出 `[值, 前驱]` | **分支指令上**，作为后继的实参 |
| 谁负责对齐 | phi 的入口列表要匹配前驱列表 | 分支自带实参，天然一一对应 |
| 位置约束 | 必须在块首、在所有非 phi 指令之前 | 无（就是块签名的一部分） |
| 同一前驱出现两次 | 必须重复写两个相同入口 | 两条边各自带参数，不存在这个问题 |
| 改 CFG 时 | 要手动修所有相关 phi | 改分支指令即可 |

> 展开阅读：[../notes/llvm-phi.md](../notes/llvm-phi.md)（汇合选值的两条规则）与 [../notes/llvm-mlir-pass-ir-unit.md](../notes/llvm-mlir-pass-ir-unit.md)（同一个 `clamp0` 在两边 Pass 里的对象树）。

亲手看一次（在 [`../../llvm-hello-compile/`](../../llvm-hello-compile/) 目录下）：

```bash
bash scripts/tour.sh                                     # 第 2 站直接并排打出三份形态
awk '/define.*@clamp0\(/,/^}/' out/tour/00_O0.ll         # ① alloca + 两次 store
awk '/define.*@clamp0\(/,/^}/' out/tour/03_mem2reg.ll    # ② phi
awk '/define.*@clamp0\(/,/^}/' out/tour/01_O2.ll         # ③ select
```

> **自测**：第 ③ 步 `phi` 消失了，但函数语义没变——那么 `phi` 到底是"语义的一部分"还是"表示的一部分"？
> 如果把 `else` 那条路径改成有实际计算（比如 `r = -y`），第 ③ 步还会变成 `select` 吗？

### 2.4 指令的九大类

LangRef 的 Instruction Reference 就是按这九类组织的，记住分类比记住每条指令更有用：

| 类别 | 代表指令 |
|------|---------|
| **Terminator** | `ret` `br` `switch` `indirectbr` `invoke` `callbr` `resume` `unreachable` |
| **Unary** | `fneg` |
| **Binary** | `add` `fadd` `sub` `mul` `udiv` `sdiv` `fdiv` `urem` `srem` `frem` |
| **Bitwise Binary** | `shl` `lshr` `ashr` `and` `or` `xor` |
| **Vector** | `extractelement` `insertelement` `shufflevector` |
| **Aggregate** | `extractvalue` `insertvalue` |
| **Memory Access & Addressing** | `alloca` `load` `store` `fence` `cmpxchg` `atomicrmw` **`getelementptr`** |
| **Conversion** | `trunc` `zext` `sext` `fptrunc` `fpext` `fptoui` `sitofp` `ptrtoint` `inttoptr` `bitcast` `addrspacecast` |
| **Other** | `icmp` `fcmp` `phi` `select` `freeze` `call` `va_arg` `landingpad` `catchpad` |

> **速记**：[../notes/conversion-llvm-vs-mlir.md](../notes/conversion-llvm-vs-mlir.md) —— LLVM 的 Conversion 是 cast 指令类；MLIR Dialect Conversion 是跨 dialect 的 lowering 框架，勿混。

两个观察：

1. **没有隐式内存访问**。除了 `load`/`store`/原子指令，没有任何指令碰内存。这是别名分析能做得好的根本前提。
2. **`shufflevector` 是向量世界的万能刀**。CodeGen 文档明确要求：目标如果有合法向量类型，就必须为常见的 shufflevector 形式（vector select / insert subvector / extract subvector / splat）生成高效代码。

### 2.5 getelementptr

`getelementptr`（GEP）**只计算地址，不访问内存**：

```llvm
%struct.Tensor = type { ptr, i32 }        ; kernel.c 里的 { float *data; int len; }

; t->len 的地址
%len_ptr = getelementptr inbounds %struct.Tensor, ptr %t, i32 0, i32 1
;                                                 ^^^^^^^^  ^^^^^^  ^^^^^^
;                                                 基址      第 0 个   第 1 个
;                                                          Tensor   字段
```

**第一个索引最容易搞错**：它是"从 `%t` 往后跳几个完整的 `%struct.Tensor`"，而不是字段号。所以取字段时第一个索引恒为 0。

`inbounds` 是一个**语义承诺**：结果指针必须落在同一个分配对象内部，否则结果是 poison。优化器严重依赖这个承诺去做别名分析——去掉 `inbounds` 会让很多优化失效。

#### 示例精讲：`relu_sum` 里的两种 GEP

**可跑** · 源码 [`llvm-hello-compile/src/kernel.c`](../../llvm-hello-compile/src/kernel.c) 的 `Tensor` 与 `relu_sum` ·
命令 `bash scripts/tour.sh`（第 3 站）或 `grep getelementptr out/tour/00_O0.ll`

```c
typedef struct { float *data; int len; } Tensor;   // datalayout 下：data 在 +0（8 字节），
                                                   // len 在 +8（4 字节），整体 16 字节（含尾部填充）

float relu_sum(const Tensor *t) {
    float s = 0.0f;
    for (int i = 0; i < t->len; ++i) {
        float v = t->data[i];                      // ← 这一行里有两次地址计算
        s += v > 0.0f ? v : 0.0f;
    }
    return s;
}
```

**一个函数里同时出现了 GEP 的两种典型用法**：

```llvm
%struct.Tensor = type { ptr, i32 }

; ① 取结构体字段：第一个索引恒为 0
%len.ptr  = getelementptr inbounds %struct.Tensor, ptr %t, i64 0, i32 1
%len      = load i32, ptr %len.ptr, align 8        ;  → 字节偏移 +8

%data.ptr = getelementptr inbounds %struct.Tensor, ptr %t, i64 0, i32 0
%data     = load ptr, ptr %data.ptr, align 8       ;  → 字节偏移 +0，装的是另一个指针

; ② 取数组元素：源元素类型直接就是 float，只有一个索引
%idx      = sext i32 %i to i64
%elt.ptr  = getelementptr inbounds float, ptr %data, i64 %idx
%v        = load float, ptr %elt.ptr, align 4      ;  → 字节偏移 4*i
```

> 形态示意：SSA 名字与 `align` 以本地 clang 输出为准；关键是索引个数与源元素类型。

**注意 `t->data[i]` 为什么是两次 GEP 而不是一次**：`data` 是一个**存在结构体里的指针**，必须先 `load` 出来才知道它指向哪。GEP 只算地址、从不访存，所以它没法"顺着指针走"——**每跨越一次间接引用，就必须有一条 `load` 夹在两条 GEP 中间**。这条规律在读任何降低产物时都成立，包括 MLIR 的 memref descriptor（见[第 7 章](#第-7-章-与-mlir--ai-编译器的接缝)）。

**"第一个索引恒为 0"到底在说什么**：现代 LLVM 的指针是 opaque 的（只有 `ptr`），GEP 自己带着**源元素类型** `%struct.Tensor`。它把这个指针看成"一个 `Tensor` 数组的首地址"——

- **第一个索引**：在这个假想数组上走几步，步长 = `sizeof(Tensor)` = 16。
- **后续索引**：钻进类型内部选字段，步长由字段布局决定。

所以 `t->len` 这种"就地取字段"必须写 `0` 起手。对照第 ② 组：源元素类型是 `float` 时没有"内部结构"可钻，索引就只有一个，且它**就是**元素下标。

**一个高频误写**：

```llvm
; 想写 t->len，却漏了那个 0
%bad = getelementptr inbounds %struct.Tensor, ptr %t, i32 1
```

它的含义是 `&t[1]`——**指向下一个 `Tensor` 的 `data`，偏移 +16**，类型上完全合法，编译器不会报错。后果分两种：

| 情况 | 后果 |
|------|------|
| `%t` 确实指向一个 `Tensor` 数组的第 0 个元素 | 静默读到相邻元素，结果错但不崩 |
| `%t` 只指向单个 `Tensor` | 越出了分配对象，**违反 `inbounds` 承诺 → 结果是 poison**；随后 `load %bad` 就是 UB |

**这就是 `inbounds` 和 poison 的接头处**：`inbounds` 不做检查、不产生代码，它只是你对优化器的一句承诺——"这次地址计算不会走出所在的那个分配对象"。承诺成立时，优化器可以据此断定"不同对象的 GEP 结果永不别名""地址计算不会回绕"，从而放心做 GVN / LICM / 向量化；承诺不成立时，结果指针立刻变成 poison，把它交给 `load`/`store` 就踩进 UB（见 [2.7 节](#27-undef--poison--freeze现代-llvm-最容易踩的坑)）。

反过来说：**拿不准就别写 `inbounds`**。少写只是少一点优化，写错是引入 UB。

看一眼真实输出：

```bash
cd llvm-hello-compile
clang -O0 -S -emit-llvm -Xclang -disable-O0-optnone src/kernel.c -o k.O0.ll
grep getelementptr k.O0.ll         # relu_sum 的 t->len / t->data / data[i] 全在这里
```

> **自测**：`getelementptr inbounds %struct.Tensor, ptr %t, i64 0, i32 1` 和
> `getelementptr inbounds i32, ptr %t, i64 2` 算出的地址一样吗？两者对别名分析提供的信息一样吗？

### 2.6 属性与标志：告诉优化器"你可以假设什么"

这一层是 LLVM 优化能力的真正来源，写前端/写 lowering 时必须理解。

**指令级标志**（poison-generating flags）：

| 标志 | 含义 | 违反时 |
|------|------|--------|
| `nsw` / `nuw` | 有符号/无符号加减乘不溢出 | 结果 poison |
| `exact` | `udiv`/`sdiv`/`lshr`/`ashr` 整除、无位丢失 | 结果 poison |
| `inbounds` | GEP 不越出对象 | 结果 poison |
| `nnan` `ninf` `nsz` `arcp` `contract` `afn` `reassoc` `fast` | 浮点快速数学标志 | 允许非 IEEE 变换 |

其中 **`contract`** 对 AI 编译器很重要：它是允许把 `a*b+c` 融合成 FMA 的开关。

> **速记**：[../notes/llvm-fma-contract.md](../notes/llvm-fma-contract.md) —— FMA 一次舍入；`contract` 是软件许可；高效真 FMA 通常靠硬件（也可软实现）；与图级算子融合同思路、不同层级。

**参数/返回值属性**：`noalias`（等价于 C 的 `restrict`）、`nocapture`、`readonly` / `writeonly`、`nonnull`、`dereferenceable(N)`、`align N`、`noundef`、`byval` / `sret`（ABI 相关）。

**函数属性**：`memory(none)` / `memory(read)` / `memory(argmem: readwrite)`（取代了老的 `readnone`/`readonly`）、`nounwind`、`willreturn`、`speculatable`、`alwaysinline` / `noinline`、`optnone`、`"target-cpu"` / `"target-features"`。

> **`noalias` + `dereferenceable` + `align` 这三个属性组合，是把张量 buffer 传给 LLVM 时能不能向量化的关键。** MLIR 的 `memref` lowering、IREE 的 `hal.interface.binding.subspan` 最终都要落到带这些属性的指针参数上。

#### 示例精讲：同一个 axpy，属性齐全与否的两份 IR

**可跑** · 源码 [`llvm-hello-compile/src/kernel.c`](../../llvm-hello-compile/src/kernel.c) 的 `axpy`（算子层主角）·
命令 `bash scripts/tour.sh` · 产物 `out/tour/00_O0.ll` 与 `out/tour/01_O2.ll`

```c
void axpy(int n, float a, const float *restrict x, float *restrict y) {
    for (int i = 0; i < n; ++i)
        y[i] = a * x[i] + y[i];
}
```

**A. 什么都没说**（手写 lowering 常见的样子）：

```llvm
define void @axpy(i32 %n, float %a, ptr %x, ptr %y) {
  ; ... 循环体：load x[i] / load y[i] / fmul / fadd / store y[i]
}
```

**B. 前端/lowering 把知道的事都说出来**（`restrict` + `-O2` 推导后的形态）：

```llvm
define void @axpy(i32 noundef %n, float noundef %a,
                  ptr noalias nocapture readonly align 16 dereferenceable(64) %x,
                  ptr noalias nocapture           align 16 dereferenceable(64) %y) #0 {
  ; ... 同样的循环体，但优化器能做的事完全不同
}

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) }
```

> 较新的 LLVM 打印 `nocapture` 时会写成 `captures(none)`，含义相同。

每个属性到底买到了什么：

| 属性 | 告诉优化器 | 换来的能力 |
|------|-----------|-----------|
| `noalias`（= C 的 `restrict`） | 这个指针可达的内存，不会被其他参数指针访问 | AA 直接回答 **NoAlias** → `store y[i]` 不会打断 `load x[i]` 的重排 → **不用插运行时指针检查，不用生成双版本循环** |
| `align 16` | 基址按 16 字节对齐 | 用对齐的向量访存，且**不必生成 peeling 前导循环**去把地址凑齐 |
| `dereferenceable(N)` | 从这里起至少 N 字节可安全读 | 允许**投机加载**：整块 load 出来再谓词化，而不是逐元素判断 |
| `readonly` | 只读不写 | `x` 上的 load 可以外提、可以合并 |
| `nocapture` | 指针不会被存到别处 | 调用之后这块内存的别名集合不会变大 |
| `memory(argmem: readwrite)` | 函数只碰参数指向的内存 | 调用点上的全局变量/其他 buffer 不会被这次调用作废 |

**为什么这直接决定能不能向量化**：向量化器要把 `y[i] = a*x[i] + y[i]` 一次做 4 个元素，前提是**这 4 次迭代之间没有内存依赖**。少了 `noalias`，`x` 和 `y` 可能重叠（比如 `y = x + 1`），第 1 次迭代写的 `y[0]` 可能就是第 2 次要读的 `x[1]`——顺序不能换。编译器这时有三条路，代价依次递增：

```text
有 noalias        →  一份向量循环，干净
无 noalias、边界可算 →  vector.memcheck 块 + found.conflict 判断
                      + 向量版 & 标量版两份循环（代码膨胀，短循环反而更慢）
无 noalias、边界不可算 →  彻底放弃，退回标量
```

自己验证一遍（`kernel.c` 里 `axpy` 与 `axpy_may_alias` 就是这一对）：

```bash
clang -O2 -S -emit-llvm kernel.c -o k.O2.ll
grep 'define.*@axpy' k.O2.ll                 # 看两个函数的参数属性差在哪

clang -O2 -Rpass=loop-vectorize \
         -Rpass-missed=loop-vectorize \
         -Rpass-analysis=loop-vectorize -c kernel.c -o /dev/null
```

`-Rpass-analysis` 会直接把"因为可能别名，所以加了运行时检查"这类原因打出来，是调 kernel 性能时的第一手工具。

> **自测**：`align 16` 让向量化器省掉了哪一段代码？如果实际传进来的指针只对齐到 4 字节，会发生什么？

### 2.7 undef / poison / freeze（现代 LLVM 最容易踩的坑）

这是当代 LLVM IR 语义里最微妙、也最必须搞清楚的一块。

**poison** 是"错误操作的结果"。设计动机是**便于投机执行**：很多指令拿到非法操作数时不立即触发 UB，而是返回 poison，让优化器可以放心地把它提到分支外面。

> **速记**：[../notes/llvm-poison-ub.md](../notes/llvm-poison-ub.md) —— poison≠空/≠异常；违反标志先 poison、用到禁位才 UB；`select` 只认选中臂；「坏路径不用」是优化器对承诺的假设。

三条规则：

1. **大多数指令的操作数是 poison，结果就是 poison**（`select` 是著名的例外）。
2. **poison 可以被替换成 undef 或该类型的任意值**——这正是优化器敢做激进变换的依据。
3. **把 poison 用在"会触发 UB 的位置"就立刻是 UB**。官方列举的位置：`load`/`store` 的指针操作数、除法的除数、`br` 的条件、`call` 的被调用者、标了 `noundef` 的参数/返回值。

官方的例子最能说明问题：

```llvm
%poison = sub nuw i32 0, 1           ; nuw 被违反 → poison
%still_poison = and i32 %poison, 0   ; 值是 0，但仍然是 poison
%p2 = getelementptr i32, ptr @h, i32 %still_poison
store i32 0, ptr %p2                 ; UB：向 poison 指针写

%cmp = icmp slt i32 %poison, 0       ; poison
br i1 %cmp, label %end, label %end   ; UB：即使两个目标一样
```

注意 `and %poison, 0` —— **数值上必然是 0，但依然是 poison**。这条最反直觉。

**`freeze`** 是 poison 的终结者：`%y = freeze i32 %x`。如果 `%x` 是 poison 或 undef，`freeze` 返回该类型的**某个任意但固定的值**；否则原样返回。它的作用是把"可以是任何值"收敛成"是某个具体值"，从而阻止 poison 继续传播。

**undef vs poison**：`undef` 是老机制（"任意值，且每次读都可以不同"），`poison` 更强也更容易推理。LLVM 一直在往用 poison 替代 undef 的方向走。写新代码时优先用 poison。

> **实践含义**：如果你在写 MLIR → LLVM 的 lowering，随手加 `nsw`/`nuw`/`inbounds` 能让下游优化好很多，但**加错了就是引入 UB，而且症状会是"某个不相关的循环被优化没了"这种极难调的 bug**。宁可不加。

### 2.8 Metadata：可丢弃的额外信息

Metadata 的定义性质是：**丢掉它程序语义不变**。所以它承载的都是"优化提示"。

| Metadata | 作用 |
|----------|------|
| `!dbg` | 调试位置，也是 remark / 诊断的来源 |
| `!tbaa` | 基于类型的别名分析信息（前端提供，中端消费） |
| `!alias.scope` / `!noalias` | 细粒度的别名作用域，**内联后保持 `restrict` 语义的关键** |
| `!range` | 值域，如"这个 load 结果在 [0, 16)" |
| `!llvm.loop` | 循环元数据：`llvm.loop.vectorize.enable`、`unroll.count`、`llvm.loop.parallel_accesses` 等 |
| `!prof` | 分支权重 / 函数热度（PGO） |
| `!nonnull` `!align` `!dereferenceable` | 加在 load 上的指针性质 |

> **`!llvm.loop.parallel_accesses` 是 MLIR/OpenMP 这类前端告诉 LLVM "这个循环的这些访存没有循环携带依赖"的官方通道**，直接影响能不能向量化。

### 2.9 Intrinsics：跨越抽象层的官方逃生舱

Intrinsic 是形如 `@llvm.xxx` 的"伪函数"。它们**不是库调用**——优化器认识它们的语义，后端会把它们直接翻成指令。LangRef 的 intrinsic 分类：

| 组 | 例子 | 对 AI 编译器的意义 |
|----|------|------------------|
| 标准库 | `llvm.memcpy` `llvm.memset` `llvm.fma` `llvm.sqrt` | 基础 |
| 位操作 | `llvm.ctlz` `llvm.ctpop` `llvm.bswap` `llvm.fshl` | —— |
| 溢出算术 | `llvm.sadd.with.overflow` | —— |
| 饱和 / 定点算术 | `llvm.sadd.sat` `llvm.smul.fix` | **量化推理** |
| **向量规约** | `llvm.vector.reduce.add/fadd/mul/and/smax/...` | **reduction 落地的标准形式** |
| **向量部分规约** | `llvm.vector.partial.reduce.add` | 点积/矩阵乘的加速原语 |
| **向量操作** | `llvm.vector.insert/extract` `llvm.vector.splice` `llvm.experimental.stepvector` | 可伸缩向量的必需品 |
| **向量掩码** | `llvm.masked.load/store/gather/scatter` | **不规则访存与谓词化** |
| **矩阵** | `llvm.matrix.multiply` `llvm.matrix.transpose` `llvm.matrix.column.major.load/store` | 直接表达小矩阵乘 |
| 目标特定 | `llvm.nvvm.*` `llvm.amdgcn.*` `llvm.x86.*` | 访问 tensor core、LDS 等硬件特性 |

> **这是理解"AI 编译器怎么接入 LLVM"的关键**：当你有一个 LLVM IR 层面没有一等表达的操作（比如 warp shuffle、tensor core MMA），标准做法不是扩展 IR，而是**用一个 target intrinsic**。MLIR 的 `nvvm` / `rocdl` dialect 本质上就是这些 intrinsic 的 MLIR 包装。

---

## 第 3 章 Pass 基础设施（New Pass Manager）

### 3.1 一个必须先知道的事实

> 官方原话："LLVM 当前包含两个 pass manager，legacy PM 和 new PM。**优化流水线（即中端）使用 new PM，而后端目标相关的代码生成使用 legacy PM。**"

所以：**读中端代码用 new PM 的心智模型，读后端 codegen 代码要切回 legacy PM 的心智模型**（`TargetPassConfig::addXxx()` 那一套）。目前社区正在推进 codegen 迁移到 new PM，但还没完成。

### 3.2 IR 层级与嵌套

```
Module ──(可选)──▶ CGSCC ──▶ Function ──▶ Loop
```

CGSCC = Call Graph Strongly Connected Component，调用图的强连通分量。**内联器就是一个 CGSCC pass**——它需要按调用图的后序处理，先优化被调用者再决定要不要内联。

每一层有自己的 `PassManager` 和 `AnalysisManager`。层与层之间用 **adaptor** 连接：

```cpp
FunctionPassManager FPM;
FPM.addPass(createFunctionToLoopPassAdaptor(LoopRotatePass()));

ModulePassManager MPM;
MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(
    createCGSCCToFunctionPassAdaptor(FunctionFooPass())));
```

**一个 PassManager 本身就是它那一层的 pass**，所以可以嵌套。

> **速记**：[../notes/llvm-pass-manager.md](../notes/llvm-pass-manager.md) —— New/Legacy 两套引擎；Module/CGSCC/Function/Loop 四层调度；默认 `default<O2>` 已含大量内置 pass；PM 对外也是 pass 故可嵌套。
> **速记**：[../notes/llvm-mlir-pass-ir-unit.md](../notes/llvm-mlir-pass-ir-unit.md) —— 与 MLIR 对照：传入的是内存 IR 根；四层是调度粒度不是黑盒；细匹配在 Pass 内，MLIR 则按 Op 嵌套锚定。

**一条重要的实践建议**（官方明确写了）：应该把同层的 pass**打包进一个 PassManager**，而不是给每个 pass 各套一个 adaptor。区别是：

```cpp
// 写法 A：对所有函数跑 Pass1，再对所有函数跑 Pass2
MPM.addPass(createModuleToFunctionPassAdaptor(FunctionPass1()));
MPM.addPass(createModuleToFunctionPassAdaptor(FunctionPass2()));

// 写法 B（推荐）：对第一个函数跑 Pass1+Pass2，再处理第二个函数
FunctionPassManager FPM;
FPM.addPass(FunctionPass1());
FPM.addPass(FunctionPass2());
MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
```

写法 B 更好，官方给了两条理由：**LLVM 数据结构的缓存局部性**，以及**优化质量**——"在一个循环上跑完所有循环 pass，可能让后面的循环获得比逐 pass 跑更多的优化机会"。

### 3.3 概念式多态：写一个 pass

New PM **没有基类接口**，用的是 concept-based polymorphism：只要你的类有一个签名正确的 `run()` 方法就行。

```cpp
#include "llvm/IR/PassManager.h"

namespace llvm {
class HelloWorldPass : public OptionalPassInfoMixin<HelloWorldPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};
} // namespace llvm
```

```cpp
PreservedAnalyses HelloWorldPass::run(Function &F, FunctionAnalysisManager &AM) {
  errs() << F.getName() << "\n";
  return PreservedAnalyses::all();
}
```

两个 mixin 的区别：
- **`OptionalPassInfoMixin`**：可以被跳过（比如函数带 `optnone` 属性时）。优化 pass 用这个。
- **`RequiredPassInfoMixin`**：不可跳过。`AlwaysInlinerPass` 用这个，因为 `alwaysinline` 语义必须保证。PassManager 本身也是 required（因为它可能装着 required pass）。

注册：在 `llvm/lib/Passes/PassRegistry.def` 的 `FUNCTION_PASS` 段加一行

```cpp
FUNCTION_PASS("helloworld", HelloWorldPass())
```

或者走**插件**路径（不用重编 LLVM）：实现 `llvmGetPassPluginInfo()`，用 `opt -load-pass-plugin=...` 加载。

测试用 lit + FileCheck：

```llvm
; RUN: opt -disable-output -passes=helloworld %s 2>&1 | FileCheck %s
; CHECK: {{^}}foo{{$}}
define i32 @foo() { ret i32 0 }
```

> 这套 `RUN:` + `CHECK:` 的写法和你 `mlir-toy-dialect` 里的 lit 测试完全一样——**MLIR 的测试基础设施就是从 LLVM 直接继承的**。

### 3.4 Analysis 的缓存与失效

这是 New PM 设计里最精巧的部分。

**基本流程**：pass 向 AnalysisManager 请求某个 analysis → AM 检查缓存 → 命中且有效就返回，否则调 analysis 的 `run()` 算一遍、缓存、返回。

**失效**：pass 通过返回值声明自己保留了什么：

```cpp
return PreservedAnalyses::all();       // 什么都没改
return PreservedAnalyses::none();      // 改了，全部作废

PreservedAnalyses PA;
PA.preserve<DominatorTreeAnalysis>();  // 我自己顺手更新了支配树
return PA;

PreservedAnalyses PA;
PA.preserveSet<CFGAnalyses>();         // 没动 CFG，只关心 CFG 的分析都还有效
return PA;
```

`preserveSet<CFGAnalyses>()` 是最常用也最容易被忘的一个——很多变换只改指令不改控制流，声明这个能省下大量重算。

想精确控制的话，analysis 自己可以实现 `invalidate()`：

```cpp
bool FooAnalysisResult::invalidate(Function &F, const PreservedAnalyses &PA,
                                   FunctionAnalysisManager::Invalidator &Inv) {
  auto PAC = PA.getChecker<FooAnalysis>();
  if (!PAC.preserved() && !PAC.preservedSet<AllAnalysesOn<Function>>())
    return true;
  // 传递依赖：我依赖的分析失效了，我也失效
  return Inv.invalidate<BarAnalysis>(F, PA);
}
```

### 3.5 Proxy：为什么内层不能算外层的分析

一个 function pass 拿到的 `FunctionAnalysisManager` **只提供 function 级分析**。想跨层访问要走 proxy：

```cpp
// CGSCC pass 里拿内层的 FunctionAnalysisManager（可以算）
FunctionAnalysisManager &FAM =
    AM.getResult<FunctionAnalysisManagerCGSCCProxy>(InitialC, CG).getManager();

// CGSCC pass 里拿外层的 module 分析（只能取缓存，不能算）
const auto &MAMProxy = AM.getResult<ModuleAnalysisManagerCGSCCProxy>(InitialC, CG);
FooAnalysisResult *AR = MAMProxy.getCachedResult<FooAnalysis>(M);
```

**内层只能 `getCachedResult()` 取外层分析，不能触发计算。** 官方给了两条理由，都值得记住：

1. **避免二次复杂度**：module 分析常常要扫每个函数。如果允许 function pass 触发 module 分析，函数就会被扫平方次。
2. **为未来的 pass 并发留路**：如果允许内层触发外层计算，并行跑 function pass 时结果会不确定。

**唯一的例外是 loop pass 访问 function 分析**（支配树等）。因为循环和它所在的函数耦合太紧，loop pass 必然要改函数。这些分析通过 `LoopStandardAnalysisResults` 参数传进来，并且由 loop pass **手动维护**以避免失效。代价是放弃了"同一函数内不同循环并行"的可能性——官方把这个明确说成一个 tradeoff。

### 3.6 pipeline 文本语法

```bash
opt -passes='pass1,pass2' a.ll -S
opt -p pass1,pass2 a.ll -S              # -p 是 -passes 的别名
```

嵌套必须显式（有两条便利规则）：

```bash
# 显式嵌套
opt -passes='no-op-module,cgscc(no-op-cgscc,function(no-op-function,loop(no-op-loop)))' a.ll -S

# 规则一：第一个 pass 不是 module pass 时，自动建对应层级的 PM
opt -passes='no-op-function,no-op-function'          # ≡ function(no-op-function,no-op-function)

# 规则二：能自动插 adaptor 时自动插
opt -passes='no-op-function,no-op-loop'              # ≡ no-op-function,loop(no-op-loop)

# 嵌套错了会直接报错
opt -passes='no-op-function,no-op-module'
# opt: unknown function pass 'no-op-module'
```

三个高频调试手段：

```bash
opt --print-passes                        # 列出所有 pass 及其 IR 层级
opt -passes=... -debug-pass-manager       # 打印实际执行顺序
opt -passes='function(require<my-analysis>),my-module-pass'   # 强制先算某个分析
```

---

## 第 4 章 中端：核心分析与变换

### 4.1 六个必须认识的 Analysis

| Analysis | 提供什么 | 谁在用 |
|----------|---------|--------|
| **DominatorTree / PostDominatorTree** | 支配关系 | 几乎所有变换。SSA 构造、代码提升的正确性前提 |
| **LoopInfo** | 自然循环的嵌套结构 | 所有循环 pass |
| **ScalarEvolution (SCEV)** | 标量在循环中如何演化（"这个值是 `4*i + base`"） | 循环强度削减、向量化、依赖分析、`scev-aa` |
| **AliasAnalysis** | 两个指针会不会指向同一块内存 | GVN、LICM、DSE、向量化 |
| **MemorySSA / MemoryDependenceAnalysis** | 内存操作之间的依赖 | GVN、DSE、LICM |
| **TargetTransformInfo (TTI)** | **目标代价模型**：某个操作在这个目标上多贵、向量宽度是多少、有没有某条指令 | 向量化、展开、内联的决策 |

### 4.2 别名分析：四种回答与链式结构

`alias()` 返回四种结果之一：

| 回答 | 官方定义 |
|------|---------|
| **NoAlias** | 基于这两个指针的任何内存引用之间**永远不存在直接依赖** |
| **MayAlias** | 两个指针**可能**指向同一对象（保守的默认答案） |
| **PartialAlias** | 两块内存**已知有重叠**，无论起始地址是否相同 |
| **MustAlias** | 两块内存**保证起始于完全相同的位置**。注意：MustAlias **不意味着指针比较相等** |

另一组接口是 `getModRefInfo()`，回答"这条指令会不会读/写某个位置"，返回 `NoModRef` / `Ref` / `Mod` / `ModRef`。**注意这个关系对两个 call site 不满足交换律。**

LLVM 的 AA 是**链式（chaining）**的：多个实现串起来，一个说不知道就问下一个。核心实现：

- **`basic-aa`**：无状态的局部分析，但知道很多硬事实——不同的全局/栈/堆分配永不别名、结构体不同字段不别名、静态下标不同的数组元素不别名、不逃逸的栈对象不会被函数调用改动。**日常 90% 的别名结论来自它**。
- **`globals-aa`**：针对"没有被取地址"的内部全局变量做上下文敏感的 mod/ref 分析。很快，能让 GVN 直接消掉整个 call。
- **`scev-aa`**：把别名查询翻译成 ScalarEvolution 查询，因此**对 `getelementptr` 和循环归纳变量的理解比其他 AA 都好**——循环里的数组访问要靠它。
- **`tbaa`**：消费前端通过 `!tbaa` metadata 提供的类型信息。

> **对 AI 编译器的直接含义**：从 MLIR 降下来的 IR，如果 buffer 指针没有 `noalias`、循环访问没有 `!alias.scope`，`basic-aa` 只能给出 MayAlias，向量化和 LICM 全部失效。**"生成的 LLVM IR 跑不快"最常见的原因就是别名信息没传下来**，而不是 LLVM 不行。

#### 示例精讲：两个指针参数的循环，AA 的回答决定一切

**可跑** · 源码 [`llvm-hello-compile/src/kernel.c`](../../llvm-hello-compile/src/kernel.c) 的 `axpy` 与 `axpy_may_alias` ·
命令 `bash scripts/tour.sh` · 产物 `out/tour/01_O2.ll`

lab 里这两个函数是**故意成对写的**：循环体一模一样，唯一的差别是参数上有没有 `restrict`。

```c
// ① 带 restrict：告诉编译器 x 和 y 不重叠
void axpy(int n, float a, const float *restrict x, float *restrict y) {
    for (int i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}

// ② 一模一样的循环，但没有 restrict
void axpy_may_alias(int n, float a, const float *x, float *y) {
    for (int i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}
```

**版本 ①**：参数带 `noalias`，`basic-aa` 对 `(x[i], y[i])` 直接回答 **NoAlias**，向量化器不需要任何保护，产出一份干净的向量循环：

```llvm
vector.body:
  %vx = load <4 x float>, ptr %xi, align 4
  %vy = load <4 x float>, ptr %yi, align 4
  %vm = fmul <4 x float> %vx, %va.splat
  %vs = fadd <4 x float> %vm, %vy
  store <4 x float> %vs, ptr %yi, align 4
  ; ... 归纳变量递增 + 回边
```

**版本 ②**：`x` 和 `y` 可能是同一块内存（调用方完全可以传 `axpy_may_alias(n, a, p, p+1)`），AA 只能回答 **MayAlias**。但两个指针的**访问范围在运行时是可算的**（基址 + `4*n` 字节），所以向量化器不放弃，而是插入运行时检查 + 双版本循环：

```llvm
entry:
  br label %vector.memcheck

vector.memcheck:                                  ; 编译期证不了，就运行时证
  %x.end = getelementptr float, ptr %x, i64 %n
  %y.end = getelementptr float, ptr %y, i64 %n
  %c1    = icmp ult ptr %y, %x.end
  %c2    = icmp ult ptr %x, %y.end
  %found.conflict = and i1 %c1, %c2               ; 两区间真的重叠？
  br i1 %found.conflict, label %scalar.ph, label %vector.ph

vector.ph:                                        ; 不重叠 → 走 <4 x float> 版本
  ...
scalar.ph:                                        ; 重叠 → 退回一次一个元素
  ...
```

> 形态示意：块名与比较方式以本地 LLVM 版本为准，关键是认出 `vector.memcheck` / `found.conflict` 这两个名字。

**同一个循环，AA 给的答案不同，结局差三档**：

| AA 对 `(x[i], y[i])` 的回答 | 向量化器的动作 | 代价 |
|---------------------------|--------------|------|
| **NoAlias** | 直接向量化 | 最优，一份循环 |
| **MayAlias**，但访问区间运行时可算 | `vector.memcheck` + 双版本循环 | 代码膨胀约 2 倍；短循环里检查开销可能吃掉全部收益 |
| **MayAlias**，且区间算不出来（步长不定 / 指针来自 call / 间接下标） | 放弃向量化 | 全标量 |
| 存在**循环携带依赖**（如 `y[i] += y[i-1]`） | 放弃（或只能做部分展开） | 全标量 |

**谁给出了这些答案**：`basic-aa` 认得"两个不同的分配对象不别名"这类硬事实，但 `x` 和 `y` 是**外部传进来的参数**，它无从判断——除非参数上写着 `noalias`。循环里 `y[i]` 随 `i` 变化的部分则要靠 `scev-aa`（它把 GEP 和归纳变量翻译成 SCEV 表达式再比较）。这就是为什么**属性必须由前端/lowering 提供**：中端没有能力凭空推出跨函数的不重叠。

亲手看两版的差别：

```bash
cd llvm-hello-compile
clang -O2 -S -emit-llvm src/kernel.c -o k.O2.ll
grep -nE 'memcheck|found\.conflict' k.O2.ll   # 只出现在 axpy_may_alias 里，axpy 一条都没有

opt -aa-pipeline=basic-aa -passes=aa-eval -disable-output k.O2.ll
```

`aa-eval` 会统计整份 IR 里 NoAlias / MayAlias / MustAlias 各占多少——**MayAlias 比例高，通常就是性能问题的根**。

> **接到链路**：这一对函数是[链路总图](./00-end-to-end-pipeline.md)站 ⑥ 的落点。上游 MLIR 的 `memref` 如果没把不重叠信息带下来（`llvm.noalias`），这里看到的就是 `axpy_may_alias` 那一版——**LLVM 不是不会向量化，是没人告诉它可以**。

> **自测**：给 `axpy_may_alias` 的两个指针都加上 `noalias`，但调用方偏偏传了重叠的指针，会发生什么？这是编译器的 bug 还是调用方的 bug？

### 4.3 变换 pass 分组

| 组 | 代表 pass | 作用 |
|----|----------|------|
| **规范化（canonicalization）** | `mem2reg` / `sroa`、`simplifycfg`、`instcombine`、`reassociate`、`loop-simplify`、`lcssa` | 把 IR 变成"标准形状"，后续 pass 才好匹配 |
| **标量简化** | `gvn`（全局值编号）、`sccp` / `ipsccp`（稀疏条件常量传播）、`dce` / `adce`、`dse`（死存储消除）、`jump-threading`、`memcpyopt`、`sink` | —— |
| **循环** | `loop-rotate`、`licm`（循环不变量外提）、`indvars`、`loop-unroll` / `unroll-and-jam`、`simple-loop-unswitch`、`loop-deletion`、`loop-reduce`（强度削减）、`loop-fusion` | —— |
| **过程间（IPO）** | `inline`（CGSCC）、`always-inline`、`function-attrs`（推导属性）、`argpromotion`、`deadargelim`、`globalopt`、`globaldce`、`constmerge`、`internalize` | —— |
| **向量化** | Loop Vectorizer、SLP Vectorizer | —— |
| **CodeGen 准备** | `codegenprepare` | IR 级但属于后端 |

**几个特别值得知道的**：

- **`mem2reg` / `sroa`**：把 `alloca` 提升成 SSA 寄存器。这就是论文里说的"前端不用自己构造 SSA"——前端可以把所有局部变量放栈上，`sroa` 负责变成 SSA。**写 MLIR → LLVM lowering 时同理**：先老老实实生成 `alloca`+`load`/`store`，交给 SROA。
- **`instcombine`**：LLVM 里最大的 pass，成千上万条局部窥孔规则。它是**规范化器**而不只是优化器——它把等价写法收敛到一个标准形式，后续 pass 才能用简单的模式匹配。
- **`licm`**：把循环不变的计算和访存提到循环外。**它高度依赖别名分析**。
- **`loop-rotate`**：把 `while` 形状转成 `do-while` 形状。这是几乎所有后续循环优化的前置条件。

### 4.4 默认 pipeline 的骨架

`PassBuilder::buildPerModuleDefaultPipeline(OptimizationLevel::O2)` 大致是这个顺序：

```
1. 早期规范化      simplifycfg, sroa, early-cse ...
2. 模块级简化      globalopt, ipsccp, function-attrs ...
3. CGSCC 循环      ┌─ inline
                   └─ 对每个函数：sroa → early-cse → instcombine
                      → simplifycfg → 循环 pass 组 → gvn → memcpyopt
                      → dse → adce → instcombine ...   （反复迭代）
4. 模块级优化      循环向量化 → SLP 向量化 → 循环展开
                   → instcombine 收尾 → alignment 推导
5. 后端准备        codegenprepare（由 llc/clang 添加，不在 opt 的 -O2 里）
```

**顺序的逻辑**：先把 IR 规范化，让内联决策基于"已经简化过"的函数体；内联后再次简化（内联暴露的常量传播、死代码是主要收益来源）；**向量化放在最后**，因为它会破坏 IR 的规范形状，后面就不好再做标量优化了。

用 `opt -passes='default<O2>' -debug-pass-manager` 可以打出完整的真实顺序。

> **速记**：[../notes/llvm-pass-scheduling.md](../notes/llvm-pass-scheduling.md) —— LLVM 的调度是预先构造的 pipeline；analysis 按需计算并缓存，transform 通过 `PreservedAnalyses` 声明失效，二者在同一条流水线里交错执行。

### 4.5 向量化：两个正交的向量化器

| | Loop Vectorizer | SLP Vectorizer |
|---|---|---|
| 对象 | 循环 | 直线代码中相邻的标量运算 |
| 做法 | 一次迭代处理多个元素 | 把多条独立标量指令打包成一条向量指令 |
| 关键依赖 | 循环依赖分析、别名分析、TTI 的向量宽度与代价 | TTI 代价模型 |
| 用户控制 | `#pragma clang loop vectorize(enable)`、`!llvm.loop.vectorize.*` | `-fno-slp-vectorize` |

Loop Vectorizer 的能力（官方 Features 一节列了很多）：处理带 if 的循环（谓词化）、reduction、归纳变量、指针归纳、reverse 迭代、混合类型、运行时指针检查（编译期不能证明不别名时，插入运行时检查 + 双版本循环）、可伸缩向量。

> **诊断很重要**：`-Rpass=loop-vectorize`、`-Rpass-missed=loop-vectorize`、`-Rpass-analysis=loop-vectorize` 会告诉你**为什么某个循环没被向量化**。调 AI kernel 性能时这是第一手工具。

### 4.6 TargetTransformInfo：中端如何询问目标

TTI 是中端唯一被允许知道的"目标信息"接口，回答这类问题：

- 这个目标的最大向量寄存器位宽是多少？
- 一次 `<8 x float>` 的 `fmul` 大概多少代价？
- 有没有硬件 gather/scatter？谓词化便宜吗？
- 展开这个循环划算吗？内联这个函数划算吗？

**为什么这对 AI 编译器重要**：如果你接一个新硬件后端，**TTI 是"让 LLVM 中端替你做正确决策"的唯一入口**。TTI 填得潦草，向量化器就会做出糟糕的选择——不是它不聪明，是你没告诉它代价。这个道理和 MLIR 里给 op 加 cost interface（比如你在 `mlir-toy-dialect` 里写的 `ToyCostOpInterface`）是完全同构的。

> **速记**：[../notes/llvm-tti.md](../notes/llvm-tti.md) —— TTI 是中端询问目标机器能力与代价的接口集合；它本质上不是优化，而是给向量化、展开、内联等 pass 提供决策依据。

---

## 第 5 章 后端 CodeGen：七个阶段（重点）

**这一章是 2004 年论文完全没有覆盖、但工程上最需要理解的部分。**

### 5.1 官方定义的七个阶段

CodeGenerator 文档明确把代码生成分成七步：

| # | 阶段 | 做什么 |
|---|------|--------|
| 1 | **Instruction Selection** | 把 LLVM IR 表达成目标指令集。产出使用 **SSA 形式的虚拟寄存器** + 少量因目标约束/调用约定而必须的物理寄存器。这一步把 LLVM 代码变成一张**目标指令的 DAG** |
| 2 | **Scheduling and Formation** | 给上一步的目标指令 DAG 定一个线性顺序，按这个顺序发射成 `MachineInstr` |
| 3 | **SSA-based Machine Code Optimizations** | *可选*。在指令选择产出的 SSA 形态上做机器码级优化（模调度、窥孔） |
| 4 | **Register Allocation** | 从**无限虚拟寄存器 SSA** 变成目标的**具体寄存器文件**。引入 spill 代码，消除所有虚拟寄存器引用 |
| 5 | **Prolog/Epilog Code Insertion** | 栈空间大小已知后插入函数序言/尾声，消除"抽象栈位置引用"。帧指针消除、栈打包在这里做 |
| 6 | **Late Machine Code Optimizations** | 在"最终"机器码上做的优化：spill 代码调度、窥孔 |
| 7 | **Code Emission** | 输出汇编或机器码 |

目标还可以在这个流程里**插入任意目标特定的 pass**（X86 用一个专门的 pass 处理 80x87 浮点栈架构）。

> 官方对这个设计的说明值得记：整个 codegen 建立在"指令选择器会用**最优模式匹配**生成高质量原生指令序列"这个假设上。基于"模式展开 + 激进迭代窥孔"的替代设计**慢得多**。这个设计让"高效编译（JIT 场景）"和"激进优化（离线场景）"能共用一套框架——每一步都可以换成不同复杂度的组件。

### 5.2 目标描述类：接新硬件要填的表

所有目标描述类在 `include/llvm/Target/`，除 `DataLayout` 外都设计为被具体目标继承：

| 类 | 职责 |
|----|------|
| **`TargetMachine`** | 总入口。通过 `getInstrInfo()` / `getRegisterInfo()` / `getFrameInfo()` 等访问器暴露其他描述类 |
| **`DataLayout`** | **唯一必需、且不可继承扩展**。结构体布局、类型对齐、指针宽度、大小端 |
| **`TargetLowering`** | 供 SelectionDAG ISel 使用：各 ValueType 用哪个寄存器类、哪些操作原生支持、`setcc` 的返回类型、移位量类型、"除以常数是否值得换成乘法"这类高层特性 |
| **`TargetRegisterInfo`** | 寄存器文件描述。**寄存器用无符号整数表示**：物理寄存器是小整数（通常 1~1023），虚拟寄存器是大整数，**`#0` 保留作标志值**。还定义 **register class**（一组性质相同的寄存器，如"所有 32 位整数寄存器"）——每个虚拟寄存器都属于某个 register class，分配器就在这个集合里挑 |
| **`TargetInstrInfo`** | 指令集描述 |
| **`TargetFrameLowering`** | 栈帧布局 |
| **`TargetSubtargetInfo`** | 同一架构下的具体型号差异（`-mcpu`） |

其中 `TargetRegisterInfo` / `TargetInstrInfo` 的具体实现**是从 TableGen 描述自动生成的**。

### 5.3 SelectionDAG 指令选择

#### 数据结构

- **`SDNode`**：核心载荷是 opcode（定义在 `include/llvm/CodeGen/ISDOpcodes.h`）+ 操作数。**一个节点可以定义多个值**（比如 div/rem 合并节点同时产出商和余数）。
- **`SDValue`**：`<SDNode, unsigned>` 二元组，指明"哪个节点的第几个结果"。边就是 SDValue。
- **`MVT`**（Machine Value Type）：每个产出值的类型。
- **两类边**：
  - **数据边**：普通的整数/浮点值。
  - **chain 边**（类型 `MVT::Other`）：给有副作用的节点（load/store/call/return）排序用。**约定：chain 输入永远是操作数 #0，chain 结果永远是最后一个产出值**（指令选择后 machine node 的 chain 位置会变到操作数之后，还可能跟 glue 节点）。
- **Entry 节点**：opcode 为 `ISD::EntryToken` 的标记节点。**Root 节点**：token chain 上最后一个有副作用的节点（单块函数里就是 return）。

#### legal vs illegal

> "一张对某目标 **legal** 的 DAG，是只使用该目标支持的操作和支持的类型的 DAG。比如在 32 位 PowerPC 上，含有 i1/i8/i16/i64 值的 DAG 是非法的，用了 SREM 或 UREM 的 DAG 也是非法的。"

把非法 DAG 变合法，正是 legalize 阶段的工作。

#### 八个步骤

```
1. Build initial DAG        从 LLVM IR 朴素展开成一张（非法的）DAG
2. Optimize (DAG Combine)   清理，识别 rotate、div/rem 对这类元指令
3. Legalize Types           消除目标不支持的类型
4. Optimize (DAG Combine)   清理类型合法化暴露的冗余
5. Legalize Ops             消除目标不支持的操作
6. Optimize (DAG Combine)   清理操作合法化引入的低效
7. Select                   模式匹配成目标指令 DAG
8. Schedule and Formation   线性化，发射成 MachineInstr
```

**Legalize Types 的手段**（记这四个词）：

| 标量 | 向量 |
|------|------|
| **promote**（小类型升到大类型，如 i1/i8/i16 → i32） | **split**（拆成两半，必要时反复拆） |
| **expand**（大整数拆成小的，如 i64 → 两个 i32） | **widen**（补元素凑到合法长度） |
| | **scalarize**（拆到单元素还找不到合法向量类型，就退化成标量） |

目标通过在 `TargetLowering` 构造函数里调 `addRegisterClass()` 告诉 legalizer 支持哪些类型。

**Legalize Ops 的手段**（三个词）：

- **expand**：用其他操作序列开码模拟。
- **promote**：升到支持该操作的更大类型。
- **custom**：调目标自己的钩子。

目标通过 `setOperationAction()` 声明。

**DAG Combiner 跑三次**的原因很有意思：官方说这让 **Legalize 可以写得非常简单**——"它可以专注于把代码变合法，而不用同时操心生成*好*的合法代码"，烂摊子由后面的 combine 收拾。**这是一条很值得迁移到自己编译器设计里的经验**。

#### Select 阶段

拿一个例子（官方原文）：

```llvm
%t1 = fadd float %W, %X
%t2 = fmul float %t1, %Y
%t3 = fadd float %t2, %Z
```

对应 DAG：`(fadd:f32 (fmul:f32 (fadd:f32 W, X), Y), Z)`。如果目标支持 FMA，其中一个 add 可以和 mul 合并——这就是模式匹配要干的事。**匹配规则大部分从 `.td` 文件生成**（见第 6 章）。

#### 示例精讲：`axpy` 的一次乘加，从 DAG 到 FMA

**可跑** · 源码 [`llvm-hello-compile/src/kernel.c`](../../llvm-hello-compile/src/kernel.c) 的 `axpy`（链路主角 `tiny_mlp` 中 Gemm 的最内层）·
命令 `bash scripts/tour.sh`（第 4 站）· 产物 `out/tour/04_contract_off.ll` 与 `05_contract_fast.ll`

```c
void axpy(int n, float a, const float *restrict x, float *restrict y) {
    for (int i = 0; i < n; ++i)
        y[i] = a * x[i] + y[i];      // ← 一次乘加，FMA 的天然候选
}
```

**`contract` 标志决定这一句是两条指令还是一条**：

```bash
clang -O1 -S -emit-llvm -ffp-contract=off  src/kernel.c -o 04_contract_off.ll
clang -O1 -S -emit-llvm -ffp-contract=fast src/kernel.c -o 05_contract_fast.ll
```

```llvm
; ── contract=off：老老实实两条指令，两次舍入
%mul = fmul float %a, %xi
%add = fadd float %mul, %yi

; ── contract=fast：允许收缩成一次舍入的融合乘加
%r = call float @llvm.fmuladd.f32(float %a, float %xi, float %yi)
```

**这是一个语义选择，不是优化选择**：`a*x+y` 分两步算要舍入两次，融合成 FMA 只舍入一次——**两者结果可以不同**。所以编译器不能自作主张，必须由 `contract` 标志（或 `-ffp-contract`）**授权**。`contract` 是「许可」，不是「要求」。

> **速记**：[../notes/llvm-fma-contract.md](../notes/llvm-fma-contract.md)。

**建完 DAG 之后**（`llc` 内部，一个基本块一张图；取循环体内那次乘加）：

```text
  t0: ch  = EntryToken                       ← chain 的起点
  t5: f32 = CopyFromReg t0, Register:f32 %0  ; a（循环不变，提到循环外）
  t8: f32,ch = load<(load 4)> t0, x.addr     ; x[i]      ← 访存，挂在 chain 上
  t11:f32,ch = load<(load 4)> t8:1, y.addr   ; y[i]      ← 排在上一次 load 之后
  t13:f32 = fmul contract t5, t8
  t14:f32 = fadd contract t13, t11
  t16:ch  = store<(store 4)> t11:1, t14, y.addr
```

> 形态示意：实际 dump 以本地 llc 版本为准（`llc -debug-only=isel` 可看全过程）。

画成图，把两类边分开看：

```text
  数据边（实线）                    chain 边（虚线，只管副作用的先后）
        fadd contract                 EntryToken
        /          \                       ┆
  fmul contract   load y[i]           load x[i]
    /       \                              ┆
   a      load x[i]                   load y[i]
                                           ┆
                                      store y[i]
```

**chain 边的意义在这个例子里才看得出来**：三次访存（读 `x[i]`、读 `y[i]`、写 `y[i]`）**必须保持相对顺序**，否则写回可能跑到读之前。DAG 本身不保证求值顺序，副作用的顺序全靠这条链——**chain 输入永远是操作数 #0，chain 结果永远是最后一个产出值**。Root 就是链上最后一个节点。

顺带说明了为什么别名信息值钱：如果 `x` 和 `y` 可能重叠，chain 就必须把两次 `load` 和那次 `store` 严格串起来；证明了不重叠（`restrict` → `noalias`），调度器才敢把不同迭代的访存交错重排。这条线索一直通到 [4.5 节](#45-向量化两个正交的向量化器)的向量化。

**legalize 前后**：

| 阶段 | x86-64 + SSE 上发生什么 |
|------|------------------------|
| Legalize Types | `f32` 在 SSE 上有寄存器类（`fr32`），**什么都不做**。若把类型换成 `<8 x float>` 而目标只有 128 位向量，这里就会 **split** 成两个 `<4 x float>`；换成 `half` 则会 **promote** 到 `f32` |
| Legalize Ops | `fmul` / `fadd` 的 f32 版本是 Legal，**什么都不做**。若目标没有硬件 FMA 而 IR 里写了 `@llvm.fma.f32`，这里会 **expand** 成对 `fmaf` 的 libcall |
| Select | 真正的分岔点，见下表 |

**Select 阶段选不选 FMA**——三个条件缺一不可：

| IR 上有 `contract`？ | 目标有 FMA 硬件（`-mattr=+fma`）？ | 选出的指令 | 舍入次数 |
|---|---|---|---|
| 否 | 有 | `vmulss` + `vaddss` | 2 次（不许合并，合并会改变结果） |
| 是 | 有 | `vfmadd213ss` | 1 次 |
| 是 | 无 | `mulss` + `addss` | 2 次（许可了，但硬件没有） |

**这张表就是"`contract` 是软件许可、FMA 是硬件能力"的全部含义**：前端负责发许可证，后端负责看有没有那条指令，两者都齐才会融合。这也解释了为什么同一份 C 代码换个 `-march` 结果会有 ULP 级差异。

观察命令：

```bash
clang -S -emit-llvm -O1 -ffp-contract=off  mad.c -o off.ll     # 看不到 contract
clang -S -emit-llvm -O1 -ffp-contract=fast mad.c -o fast.ll    # fmul/fadd 带 contract

llc -mtriple=x86_64-- -mattr=+fma -debug-only=isel fast.ll -o /dev/null
llc -mtriple=x86_64-- -mattr=+fma -debug-only=isel-dump -filter-print-funcs=mad fast.ll -o /dev/null
llc -mtriple=x86_64-- -mattr=+fma -view-isel-dags fast.ll -o /dev/null   # 需要 graphviz
```

`-debug-only=` 需要 assertion 构建的 LLVM；没有的话用 `-view-isel-dags` 看图，或者直接对比 `off.ll` / `fast.ll` 生成的汇编。

> **自测**：`-ffp-contract=off` 编出来的 IR 喂给一个有 FMA 的目标，后端能不能自作主张选 `vfmadd`？为什么？

### 5.4 GlobalISel：另一条路

GlobalISel 是 SelectionDAG 和 FastISel 的**替代方案**（尚未完全取代）。官方给了三个动机：

| 问题 | SelectionDAG 的毛病 | GlobalISel 的做法 |
|------|-------------------|------------------|
| **性能** | 引入了一个专门的中间表示，有编译期开销 | 直接操作 codegen 后续都在用的 MIR（扩展成 **gMIR** 以支持任意输入 IR） |
| **粒度** | SelectionDAG 和 FastISel 都**按基本块**工作，丢失全局优化机会 | **在整个函数上工作** |
| **模块化** | SelectionDAG 和 FastISel 差异巨大、几乎不共享代码 | 优化版和快速版共享同一条流水线，目标可以自行配置 |

**GlobalISel 的四个 pass**（记住这四个名字就够了）：

```
IRTranslator      LLVM IR  →  gMIR（通用 MachineInstr）
Legalizer         消除目标不支持的通用操作/类型
RegBankSelect     决定每个虚拟寄存器落在哪个寄存器 bank（GPR? FPR? VGPR?）
InstructionSelect gMIR  →  目标 MachineInstr
```

`RegBankSelect` 是 SelectionDAG 里没有的独立步骤——把"值放在哪类寄存器"从指令选择里拆了出来，这对 GPU（标量寄存器 vs 向量寄存器）尤其重要。

**现状**：AArch64 上已经用 GlobalISel 替代 FastISel（`-O0`），AMDGPU 也大量使用；替代 SelectionDAG 作为优化 ISel 是长期目标。选择失败时会 fallback 到 SelectionDAG。

### 5.5 Machine IR

| 类 | 说明 |
|----|------|
| `MachineFunction` | 对应一个 `Function` |
| `MachineBasicBlock` | 对应一个 `BasicBlock` |
| `MachineInstr` | 一条目标指令：opcode + 一组 `MachineOperand` |
| `MachineOperand` | 寄存器 / 立即数 / 基本块地址 / 全局地址 / 帧索引 / ... |

关键点：

- **指令选择后到寄存器分配前，MachineIR 是 SSA 形态**（虚拟寄存器 + `PHI` 伪指令）。
- **物理寄存器在寄存器分配前只在单个基本块内活跃**——这个假设让活跃变量分析可以在块内做纯局部分析，非常快。
- **MachineInstr Bundle**：把几条必须一起调度/发射的指令捆成一束（VLIW 目标必需）。
- MIR 可以用 `.mir` 文本序列化，配合 `llc -stop-after=` / `-start-before=` 单独测试某个 codegen pass。

### 5.6 寄存器分配

**问题定义**：把使用无限虚拟寄存器的程序 P_v 映射到只有有限物理寄存器的 P_p。装不下的虚拟寄存器要放到内存里，叫 **spilled virtuals**。

**前置分析**：

1. **Live Variable Analysis**：算出每条指令之后立即 dead 的寄存器、以及被这条指令 kill 的寄存器。因为虚拟寄存器是 SSA 形态，这个分析可以做得**稀疏**（sparse），很便宜。`PHI` 需要特殊处理——从当前块末尾模拟一次赋值，然后向后继块传播。
2. **Live Interval Analysis**：给基本块和指令编号，把活跃性表示成区间 `[i, j)`。两个需要同一物理寄存器的虚拟寄存器如果区间重叠就冲突，必须 spill 一个。

**SSA 解构（PHI Elimination）**：真实指令集没有 `PHI` 指令，必须消掉。LLVM 采用最传统的做法——**把 `PHI` 换成 copy 指令**（`lib/CodeGen/PHIElimination.cpp`）。

**两地址指令处理**：x86 这类 `add %eax, %ebx`（目标同时是源）的指令，需要 two-address 转换插入 copy。

**四个内建分配器**：

| 分配器 | 说明 |
|--------|------|
| **Fast** | debug 构建的默认。**基本块级别**分配，尽量把值留在寄存器里并复用 |
| **Basic** | 增量式，按启发式顺序逐个分配活跃区间。**不是生产级分配器**，用作调 bug 和性能基线，也是开发新分配器的框架 |
| **Greedy** | **默认分配器**。Basic 的高度调优版，加入了**全局活跃区间分裂（live range splitting）**，努力最小化 spill 代码的代价 |
| **PBQP** | 把寄存器分配建模成 Partitioned Boolean Quadratic Programming 问题求解 |

用 `llc -regalloc=greedy|fast|basic|pbqp` 切换。

**指令折叠（instruction folding）**：分配过程中消除多余的 copy，比如

```
%EBX = LOAD %mem_address
%EAX = COPY %EBX
```
折叠成
```
%EAX = LOAD %mem_address
```

#### 示例精讲：`relu_sum` 的循环 MIR before / after

**可跑** · 源码 [`llvm-hello-compile/src/kernel.c`](../../llvm-hello-compile/src/kernel.c) 的 `relu_sum` ·
命令 `bash scripts/tour.sh`（第 13 站）· 产物 `out/tour/21_after_isel.mir` 与 `22_after_regalloc.mir`

```c
float relu_sum(const Tensor *t) {
    float s = 0.0f;
    for (int i = 0; i < t->len; ++i) {
        float v = t->data[i];
        s += v > 0.0f ? v : 0.0f;
    }
    return s;
}
```

> **选它的理由**：浮点加法不满足结合律，所以在没开 `-ffast-math` 时 LLVM **不会**把这个归约向量化。
> 循环因此保持标量形态，寄存器分配的三件事看得最干净。同一份 `kernel.c` 里的 `axpy` 则会被向量化——
> 两个函数在同一份 `.mir` 里并排，正好对照。

**指令选择刚结束**（`llc -mtriple=x86_64-- -O2 -stop-after=finalize-isel out/tour/01_O2.ll`）：

```text
bb.1.loop:
  successors: %bb.1, %bb.2
  %10:gr64 = PHI %4,  %bb.0, %14, %bb.1   ; i    ← 还是 SSA：PHI 伪指令 + 虚拟寄存器
  %11:fr32 = PHI %5,  %bb.0, %13, %bb.1   ; s
  %12:fr32 = MOVSSrm %2, 4, %10, 0, $noreg      ; load t->data[i]
  %13:fr32 = ADDSSrr %11(tied-def 0), %15       ; s += relu(v)
  %14:gr64 = ADD64ri8 %10(tied-def 0), 1, implicit-def dead $eflags
  JCC_1 %bb.1, 5, implicit $eflags
```

> 形态示意：实际 dump 以本地 llc 版本为准。

**寄存器分配之后**（`llc ... -stop-after=virtregrewriter`）：

```text
bb.1.loop:
  renamable $xmm1 = MOVSSrm renamable $rsi, 4, renamable $rax, 0, $noreg
  renamable $xmm0 = ADDSSrr renamable $xmm0(tied-def 0), killed renamable $xmm1
  renamable $rax  = ADD64ri8 renamable $rax(tied-def 0), 1, implicit-def dead $eflags
  JCC_1 %bb.1, 5, implicit $eflags
```

三件事同时发生了：

| 变化 | 谁干的 | 现象 |
|------|-------|------|
| `%10 %11 %12 ...` → `$rax $xmm0 $xmm1 ...` | Greedy 分配器 | 虚拟寄存器全部消失 |
| `PHI` 不见了 | **PHIElimination**（SSA 解构） | 变成前驱块末尾的 `COPY`，很多 copy 随后被合并掉 |
| `tied-def 0` 依然在 | two-address 转换 | x86 的 `add` 目标即源，必要时插 copy 保证这一点 |

`tour.sh` 第 13 站会直接把这三项数出来给你看：

```text
指标                       ISel后     分配后
PHI 指令                       2          0      ← SSA 解构
虚拟寄存器 %0/%1...           27          0      ← 全部映射到物理寄存器
物理寄存器 $xmm/$e..           6         31
```

**为什么这个例子不需要 spill**：活跃区间少，寄存器够用。把它想象成时间轴上的线段：

```text
指令编号        0      16      32      48      64      80
$rsi (data)     ├───────────────────────────────────────┤   横跨整个循环，占死一个寄存器
$rax (i)                ├───────────────────────────────┤
$xmm0 (s)               ├───────────────────────────────┤
$xmm1 (v)                       ├───┤                       只活 2 条指令，随便复用
                                ↑
                       区间重叠 = 冲突，不能共用同一个物理寄存器
```

`data` / `i` / `s` 三条长区间两两重叠，必须各占一个寄存器；`v` 的区间极短，可以在每次迭代里反复复用同一个 `$xmm1`。注意 `i` 和 `s` **分属两个寄存器类**（通用 vs 向量），所以它们其实不竞争——寄存器分配是按寄存器类分别做的。**当同一类里同时重叠的长区间数量超过可用物理寄存器时，才需要 spill**：

```text
  MOVSSmr %stack.0, 1, $noreg, 0, $noreg, killed renamable $xmm0   ; spill：写回栈槽
  ...                                                              ; 中间这段把 $xmm0 让给别人
  renamable $xmm0 = MOVSSrm %stack.0, 1, $noreg, 0, $noreg         ; reload：用之前读回来
```

> 形态示意：spill 槽名与寻址模式以本地 llc 输出为准。

**Greedy 的核心价值就在 spill 的插入点上**：Basic 分配器发现装不下就把整条区间 spill 掉（循环里每次迭代都要 reload）；Greedy 会先做**活跃区间分裂**——把长区间从"用不到"的那一段剪断，只 spill 冷的部分，尽量不在循环体内插访存。看差别最直接的办法是换分配器对比（在 `llvm-hello-compile/` 下）：

```bash
llc -mtriple=x86_64-- -O2 -regalloc=greedy out/tour/01_O2.ll -o greedy.s
llc -mtriple=x86_64-- -O2 -regalloc=basic  out/tour/01_O2.ll -o basic.s
diff greedy.s basic.s                                    # 差异集中在循环体内的访存
llc -mtriple=x86_64-- -O2 -print-after=greedy out/tour/01_O2.ll -o /dev/null
```

> **自测**：`PHI` 消失后变成了 copy，为什么这些 copy 大多不会真的出现在最终汇编里？
> 同一份 `.mir` 里 `@axpy` 被向量化而 `@relu_sum` 没有——差别的根源是什么？（提示：不是别名，是浮点结合律。）

### 5.7 MC 层

> "MC 层用来表示和处理最原始的机器码级别的代码，**剥离了'常量池''跳转表''全局变量'这类高层信息**。在这一层 LLVM 处理的是标签名、机器指令、目标文件里的节区。"

四个核心类：

| 类 | 说明 |
|----|------|
| **`MCInst`** | 目标无关的指令表示，比 `MachineInstr` 简单得多：一个目标特定 opcode + 一组 `MCOperand`。`MCOperand` 是三选一的判别联合：立即数 / 目标寄存器 ID / 符号表达式（`MCExpr`，如 `Lfoo-Lbar+42`） |
| **`MCStreamer`** | **本质上就是一个汇编器 API**。每条汇编指示对应一个方法：`emitLabel`、`emitSymbolAttribute`、`switchSection`、`emitValue`（对应 `.byte`/`.word`）、`emitInstruction` |
| **`MCContext`** | 各种 uniqued 数据结构的持有者（符号、节区）。不可继承 |
| **`MCSymbol`** / **`MCSection`** | 符号和节区。由 `MCContext` 创建并 unique，所以**可以用指针相等判断是不是同一个符号** |

`MCStreamer` 有两个主要实现：

- **`MCAsmStreamer`**：每个方法打印一条指示（`emitValue` → `.byte`），产出 `.s`。
- **`MCObjectStreamer`**：**实现了一个完整的汇编器**，直接产出 `.o`。

`MCInst` 是"MC 层的通用货币"：指令编码器、指令打印器、汇编解析器、反汇编器都用它。`llvm-mc` 这个独立汇编器/反汇编器就是建在这层上的。

支持的目标文件格式：ELF（多数目标）、MachO（Darwin）、COFF、XCOFF（AIX）、以及 DirectX / SPIR-V / WebAssembly 这类自有格式。

### 5.8 怎么观察后端

```bash
# 各 codegen pass 之后的 MIR
llc -print-after-all foo.ll -o /dev/null
llc -print-after=greedy foo.ll -o /dev/null

# 在某个 pass 后停下，输出 .mir（可以再喂回 llc 单独测试）
llc -stop-after=finalize-isel foo.ll -o foo.mir
llc -start-before=greedy foo.mir -o foo.s

# 指令选择的详细过程
llc -debug-only=isel foo.ll -o /dev/null
llc -debug-only=isel-dump -filter-print-funcs=my_kernel foo.ll -o /dev/null

# 可视化 DAG（需要 graphviz）
llc -view-dag-combine1-dags   # 建完 DAG、第一次 combine 之前
llc -view-legalize-dags       # legalize 之前
llc -view-dag-combine2-dags   # 第二次 combine 之前
llc -view-isel-dags           # Select 之前
llc -view-sched-dags          # 调度之前
llc -view-sunit-dags          # 调度器的依赖图
llc -filter-view-dags=<bb名>  # 只看某个基本块

# 换寄存器分配器 / 看机器码编码
llc -regalloc=fast foo.ll
llc -show-mc-encoding foo.ll
```

---

## 第 6 章 TableGen：把目标描述变成代码

### 为什么需要

> "目标描述类需要对目标架构的详尽描述。这些描述往往包含大量共同信息（例如 `add` 指令几乎和 `sub` 指令一模一样）。为了最大程度地把共性提取出来，LLVM 代码生成器用 TableGen 来描述目标机器的大部分内容。"

好处有两条（官方原话）：**减少要写的 C++ 代码量、降低理解门槛**；以及**只要改 `tblgen` 一处就能把所有目标更新到新接口**。

### 核心概念

TableGen 的语义模型极简：**class**（可继承的模板）、**def**（具体记录）、**multiclass**（一次定义多个 def）、**let**（覆盖字段）、**dag**（有向无环图字面量，用于模式）。

前端解析 `.td` 并实例化所有声明，然后把结果交给一个**领域特定的 backend** 处理。`llvm-tblgen -help` 能列出所有 backend。

### 在 LLVM 后端里长什么样

一条 X86 指令展开后的记录（官方示例）：

```
def ADD32rr {   // Instruction X86Inst I
  string Namespace = "X86";
  dag OutOperandList = (outs GR32:$dst);
  dag InOperandList = (ins GR32:$src1, GR32:$src2);
  string AsmString = "add{l}\t{$src2, $dst|$dst, $src2}";
  list<dag> Pattern = [(set GR32:$dst, (add GR32:$src1, GR32:$src2))];
  list<Register> Defs = [EFLAGS];
  bit mayLoad = 0;
  bit mayStore = 0;
  ...
}
```

注意 **`Pattern` 字段**：`[(set GR32:$dst, (add GR32:$src1, GR32:$src2))]` 就是喂给指令选择器的匹配模式。**SelectionDAG 的 Select 阶段的大部分逻辑就是从这些 Pattern 生成的。**

从 `.td` 生成的东西包括：`TargetRegisterInfo` 实现、`TargetInstrInfo` 实现、指令选择器的匹配表、汇编器/反汇编器、指令打印器、调度模型（SchedModel）。

查看方式：

```bash
llvm-tblgen X86.td -print-enums -class=Instruction   # 列出所有指令名
llvm-tblgen X86.td -dump-json                        # 全部记录导成 JSON
llvm-tblgen X86.td                                   # 展开所有 class 和 def
```

### 与 MLIR ODS 的关系

**MLIR 的 ODS（Operation Definition Specification）就是 TableGen 的另一个 backend。** 你在 `mlir-toy-dialect/include/Toy/ToyOps.td` 里写的

```tablegen
def Toy_MulOp : Toy_Op<"mul", [Pure, ToyCostOpInterface]> { ... }
```

和上面 X86 的 `def ADD32rr` 是**同一套语言、同一个解析器**，只是交给了不同的 backend（`mlir-tblgen -gen-op-defs` vs `llvm-tblgen -gen-instr-info`）。

理解这一点的价值在于：**TableGen 在 LLVM 里解决的问题（目标描述的组合爆炸）和它在 MLIR 里解决的问题（op 定义的样板代码爆炸）是同一个问题。** 学会读 X86 的 `.td` 会直接提升你读/写 MLIR ODS 的能力，反之亦然。

> **速记**：[../notes/tablegen-llvm-mlir.md](../notes/tablegen-llvm-mlir.md) —— TableGen 是声明式描述 + 代码生成基础设施；LLVM 用它描述目标机器，MLIR 用同一套语言的 ODS backend 描述 dialect / op。

---

## 第 7 章 与 MLIR / AI 编译器的接缝

### 7.1 `llvm` dialect ↔ LLVM IR

MLIR 里有一个和 LLVM IR **几乎一一对应**的 `llvm` dialect。所有高层 dialect（`linalg` / `affine` / `vector` / `memref`，以及 IREE 的 `hal` / `stream`）最终都降到它，然后用

```bash
mlir-translate --mlir-to-llvmir input.mlir -o output.ll
```

翻成真正的 LLVM IR。**注意这是 translate 而不是 lowering**——它是两个不同 IR 系统之间的机械转换，不是 MLIR 内部的 pass。

**接缝上的关键映射**：

| MLIR | LLVM IR |
|------|---------|
| `memref<?x?xf32>` | 一个 struct（allocated ptr、aligned ptr、offset、sizes[]、strides[]）—— **MemRef descriptor** |
| `vector<8xf32>` | `<8 x float>` |
| `vector<4x8xf32>` | `[4 x <8 x float>]`（多维向量降成数组套向量） |
| block argument | `phi` |
| `llvm.func` / `llvm.call` | `define` / `call` |
| `nvvm.*` / `rocdl.*` op | 对应的 target intrinsic |

#### 示例精讲：一个 memref 函数翻过接缝

**可跑** · 源码 [`mlir-toy-dialect/examples/upstream/03-memref-to-llvm.mlir`](../../mlir-toy-dialect/examples/upstream/03-memref-to-llvm.mlir) ·
命令 `cd mlir-toy-dialect && bash scripts/run_upstream.sh` · 产物 `out/upstream/`

最小的 MLIR：一个 `func.func`，参数是动态 shape 的 `memref`，体内只有一次读-算-写。算的正是链路主角 `tiny_mlp` 里的 Relu：

```mlir
func.func @relu_first(%buf: memref<?xf32>) {
  %c0   = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %v = memref.load %buf[%c0] : memref<?xf32>
  %p = arith.cmpf ogt, %v, %zero : f32
  %r = arith.select %p, %v, %zero : f32          // ← 与 relu_sum 降出来的 select 同形
  memref.store %r, %buf[%c0] : memref<?xf32>
  return
}
```

先在 MLIR 内部降到 `llvm` dialect，再翻译：

```bash
mlir-opt --convert-arith-to-llvm \
         --finalize-memref-to-llvm \
         --convert-func-to-llvm \
         --reconcile-unrealized-casts 03-memref-to-llvm.mlir -o x.llvm.mlir
mlir-translate --mlir-to-llvmir x.llvm.mlir -o x.ll
```

（老一些的 MLIR 里这个 pass 叫 `--convert-memref-to-llvm`；`run_upstream.sh` 会自动试两个名字。）

降完之后，函数签名变了样——**一个 `memref` 参数被摊平成了 5 个标量参数**：

```mlir
llvm.func @relu_first(%arg0: !llvm.ptr,   // allocated ptr
                      %arg1: !llvm.ptr,   // aligned ptr
                      %arg2: i64,         // offset
                      %arg3: i64,         // size[0]
                      %arg4: i64)         // stride[0]
```

翻成 LLVM IR 就是：

```llvm
define void @relu_first(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4) {
  %6 = getelementptr float, ptr %1, i64 %2      ; aligned + offset（索引是 0，被折掉了）
  %7 = load float, ptr %6, align 4
  %8 = fcmp ogt float %7, 0.000000e+00
  %9 = select i1 %8, float %7, float 0.000000e+00
  store float %9, ptr %6, align 4
  ret void
}
```

同一个文件里还放了静态 shape 与二维两个对照函数：**参数个数恒等于 `3 + 2 × rank`**，跑之前先自己数一遍。

> 形态示意：SSA 编号与临时值个数以本地 mlir-translate 版本为准；关键是参数个数与地址计算的形状。

**MemRef descriptor 的字段对照**（`memref<?xf32>` 是 rank 1，所以 sizes/strides 各一项）：

| descriptor 字段 | LLVM 类型 | 含义 | 摊平后的参数 |
|----------------|----------|------|------------|
| allocated ptr | `ptr` | `malloc` 返回的原始指针，**只用于 free** | `%0` |
| aligned ptr | `ptr` | 对齐后的数据基址，**所有访存都从它算起** | `%1` |
| offset | `i64` | 起始偏移，**以元素为单位**（不是字节） | `%2` |
| sizes[0..rank-1] | `i64 × rank` | 各维大小 | `%3` |
| strides[0..rank-1] | `i64 × rank` | 各维步长，同样以元素为单位 | `%4` |

不摊平时它就是一个真结构体：`{ ptr, ptr, i64, [1 x i64], [1 x i64] }`。摊平（unpacked）是默认的 ABI，因为标量参数比传结构体更容易让 LLVM 做优化。

**地址公式**——这是理解所有 `memref` lowering 结果的钥匙：

```text
element_addr = aligned_ptr + (offset + Σ_d  index_d * stride_d) * sizeof(elem)
```

`memref.load %buf[%i, %j]` 降下来就是这条公式展开成的一串 `mul`/`add`/`getelementptr`。

**接缝上最值得注意的一点**：上面那份 `.ll` 里，`%0` 和 `%1` 这两个指针参数**什么属性都没有**——没有 `noalias`、没有 `align`、没有 `dereferenceable`。如果这个函数里是个循环，LLVM 只能给 MayAlias，向量化立刻退化成 [4.2 节](#42-别名分析四种回答与链式结构)里说的"运行时检查 + 双版本"甚至彻底放弃。**这些属性必须由 lowering 主动加上**（MLIR 侧可以通过 `llvm.noalias` 等参数属性传递），这正是 [7.3 节](#73-让-llvm-替你干活的三个杠杆)三个杠杆里的第一个。

一路推到汇编，把整条缝看完整（在 `mlir-toy-dialect/` 下）：

```bash
mlir-opt --convert-arith-to-llvm --finalize-memref-to-llvm \
         --convert-func-to-llvm --reconcile-unrealized-casts \
         examples/upstream/03-memref-to-llvm.mlir \
  | mlir-translate --mlir-to-llvmir \
  | opt -passes='default<O2>' -S \
  | llc -mtriple=x86_64-- -o -
```

**这条命令就是链路上站 ⑤ 交给站 ⑥ 的那一刻**（见 [`00-end-to-end-pipeline.md`](./00-end-to-end-pipeline.md)）：左边是 MLIR 的地盘，右边是本篇的地盘，属性传不传得过来全在这一根管道上。

> **自测**：既然 `aligned ptr` 才是访存用的基址，为什么 descriptor 还要单独保留 `allocated ptr`？
> 把 `memref<?x?xf32>` 摊平会得到几个参数？（文件里 `@relu_first_2d` 就是它，先猜再跑。）

### 7.2 为什么 AI 编译器只用 LLVM 的最后一段

TVM、IREE、Triton、XLA 都是"图级/张量级自己做，机器码生成交给 LLVM"。原因很实际：

- **指令选择、寄存器分配、指令调度是几十年积累 + 高度依赖具体 ISA 细节的体力活**，LLVM 已经为几十种架构维护了成熟实现。
- AI 编译器的差异化价值在**融合、tiling、内存规划、并行策略**这些更高层的决定上。

但要清楚**边界在哪里**：一旦降到 LLVM IR，"这是一次卷积"这个信息就永久丢失了。所以**所有依赖高层结构的优化必须在进入 LLVM 之前做完**。LLVM 之后能拿到的，只有循环级和指令级的收益。

### 7.3 让 LLVM 替你干活的三个杠杆

从 MLIR 降下来的 IR 质量，决定了 LLVM 能帮你多少。三个最有效的杠杆：

1. **别名信息**：给 buffer 指针加 `noalias`，给循环访存加 `!alias.scope` / `!noalias`，给并行循环加 `!llvm.loop.parallel_accesses`。**没有这些，LICM 和向量化基本瘫痪。**
2. **对齐与解引用信息**：`align N`、`dereferenceable(N)`、`nonnull`。决定能不能用对齐的向量 load/store、能不能投机加载。
3. **TTI 填充质量**（如果你接了新后端）：向量宽度、各操作代价、是否支持 gather/scatter/谓词化。**中端的所有代价决策都问它。**

### 7.4 GPU 路径

```
MLIR nvvm dialect  ──▶  LLVM IR + llvm.nvvm.* intrinsics  ──▶  NVPTX 后端  ──▶  PTX
MLIR rocdl dialect ──▶  LLVM IR + llvm.amdgcn.* intrinsics ──▶ AMDGPU 后端 ──▶  HSACO
```

要点：

- **NVPTX 后端产出的是 PTX 文本，不是最终机器码**——还要经过 `ptxas` 或驱动的 JIT。所以 IREE 的 CUDA target 打包的是 PTX 镜像。
- AMDGPU 后端直接产出 HSACO（真正的机器码）。
- **地址空间是 GPU 路径上最容易出错的地方**：NVPTX 里 generic/global/shared/const/local 是不同的 `addrspace(N)`，`addrspacecast` 不是免费的。MLIR 侧的 `memref` memory space 必须正确映射过来。

### 7.5 LTO / ThinLTO

- **Full LTO**：链接时把所有模块的 IR 合并成一个大 Module，跑完整的跨模块优化。效果最好，但内存和时间开销大、无法并行。
- **ThinLTO**：每个模块只产出一份**摘要（summary）**，链接器用摘要做全局的内联/导入决策，然后各模块**并行**做优化。这正是 2004 年论文里说的"编译期算好过程间摘要、链接期直接消费摘要"的现代实现。

对 AI 编译栈的相关性：IREE 的 executable 链接、TVM 的 host+device 模块合并，本质上都是同类问题——**在"全局视野"和"编译并行度"之间取舍**。

---

## 第 8 章 工具链与观察手段速查

### 工具

| 工具 | 用途 |
|------|------|
| `clang -S -emit-llvm` | 源码 → `.ll` |
| `opt` | 对 IR 跑 pass。中端的唯一驱动 |
| `llc` | IR → 汇编/目标文件。后端的唯一驱动 |
| `lli` | 解释/JIT 执行 IR |
| `llvm-as` / `llvm-dis` | `.ll` ↔ `.bc` |
| `llvm-link` | 合并多个 `.bc` |
| `llvm-extract` | 从模块里抠出某个函数（**调试时极有用**） |
| `llvm-reduce` | 自动缩小触发 bug 的测试用例 |
| `llvm-mc` | 独立汇编器/反汇编器（MC 层） |
| `llvm-objdump` | 反汇编目标文件 |
| `llvm-mca` | 静态机器码性能分析（吞吐、端口压力） |
| `llvm-tblgen` / `mlir-tblgen` | TableGen 驱动 |
| `FileCheck` / `llvm-lit` | 测试基础设施（MLIR 直接复用） |

### 调试 flag

```bash
# ── 中端 ──
opt -passes='default<O2>' -debug-pass-manager        # 打印真实执行顺序
opt -passes=... -print-after-all                     # 每个 pass 后打印 IR
opt -passes=... -print-changed                       # 只打印真正改变了 IR 的 pass ★
opt -passes=... -filter-print-funcs=foo              # 只看某个函数
opt -passes=... -debug-only=instcombine              # 某个 pass 的内部日志（需 assertion 构建）
opt -passes=... -stats                               # 各 pass 的计数器
opt -passes=... -time-passes                         # 各 pass 耗时
opt --print-passes                                   # 列出所有可用 pass

# ── 后端 ──
llc -mtriple=aarch64-linux-gnu -mcpu=neoverse-v1 -mattr=+sve
llc -print-after-all / -stop-after=<pass> / -start-before=<pass>
llc -debug-only=isel-dump -filter-print-funcs=<fn>
llc -regalloc=greedy|fast|basic|pbqp

# ── 从 clang 透传给 LLVM ──
clang -mllvm -print-after-all foo.c
clang -Rpass=loop-vectorize -Rpass-missed=loop-vectorize \
      -Rpass-analysis=loop-vectorize foo.c            # 为什么没向量化 ★
clang -fsave-optimization-record foo.c                # 产出 .opt.yaml
```

`-print-changed` 和 `-Rpass-missed=loop-vectorize` 这两个是日常最高频的，值得单独记住。

---

## 第 9 章 学习路径：最小必要集与动手清单

### 9.1 必须掌握

1. **四层 IR 的总图**（[第 1 章](#第-1-章-一张总图从源码到机器码)）——LLVM IR / SelectionDAG / MachineIR / MCInst 各自解决什么，能默画出来。
2. **SSA + 显式 CFG + terminator 规则**，以及 MLIR 用 block argument 替代 `phi` 的原因。
3. **`getelementptr` 的第一个索引语义**和 `inbounds` 的承诺。
4. **poison / undef / freeze 三者的关系**，以及"`and poison, 0` 仍然是 poison"这类反直觉规则。**写 lowering 时随手加 `nsw`/`inbounds` 是最常见的 UB 来源。**
5. **属性与 metadata 是优化能力的来源**：`noalias`、`align`、`dereferenceable`、`!alias.scope`、`!llvm.loop.parallel_accesses`。
6. **New PM 的四层嵌套与 PreservedAnalyses 四种写法**，以及"内层不能算外层分析"的两条理由。
7. **别名分析的四种回答**与 `basic-aa` 知道哪些硬事实。
8. **后端七阶段**，尤其是 SelectionDAG 的八步、legalize types 的五个手段（promote/expand/split/widen/scalarize）、legalize ops 的三个手段（expand/promote/custom）。
9. **寄存器分配链条**：LiveVariables → LiveIntervals → PHIElimination → two-address → Greedy 分配 → 折叠。
10. **TableGen 在 LLVM 和 MLIR 里是同一套语言**，Pattern 字段驱动指令选择。
11. **TTI 是中端唯一的目标信息入口**——接新硬件时它决定了向量化/展开/内联的质量。

### 9.2 可以先跳过

- LangRef 里 intrinsic 的逐条细节（当字典查）。
- 调试信息（DWARF、`DICompositeType` 那一整套 metadata）。
- 异常处理的底层实现（landingpad / personality / compact unwind）。
- 具体目标后端的细节（X86 寻址模式编码、PowerPC 帧布局、eBPF 指令编码）。
- 内联汇编约束字符串。
- GC 相关的 intrinsic 与 statepoint。
- `llvm-mca` 的调度模型细节（除非要做微架构级调优）。
- Legacy Pass Manager 的具体 API（除非要改 codegen pipeline）。

### 9.3 动手清单（按顺序）

**第一步：看穿中端**

```bash
cat > loop.c <<'EOF'
void axpy(float a, float *restrict x, float *restrict y, int n) {
  for (int i = 0; i < n; i++) y[i] += a * x[i];
}
EOF

clang -O0 -S -emit-llvm loop.c -o loop.O0.ll     # 全是 alloca/load/store
opt -passes='mem2reg' loop.O0.ll -S              # 观察 SSA 是怎么被构造出来的 ★
opt -passes='default<O2>' loop.O0.ll -S -o loop.O2.ll
```

**要在 O0 → mem2reg → O2 的三份 IR 里确认**：`alloca` 消失、`phi` 出现、循环被 rotate、最终出现 `<4 x float>` 或 `<vscale x 4 x float>`。然后把 `restrict` 去掉重跑一次，**观察向量化怎么变成了"运行时指针检查 + 双版本循环"甚至彻底失败**——这一步能把第 4.2 节的别名分析讲的东西变成手感。

**第二步：看穿后端**

```bash
llc -mtriple=x86_64-- -O2 loop.O2.ll -o loop.s
llc -mtriple=x86_64-- -stop-after=finalize-isel loop.O2.ll -o loop.mir   # 指令选择刚结束
llc -mtriple=x86_64-- -stop-after=greedy        loop.O2.ll -o loop.ra.mir # 寄存器分配后
```

对比两份 `.mir`：前者全是 `%0`、`%1` 这样的虚拟寄存器和 `PHI`，后者只剩 `$xmm0` 这样的物理寄存器且 `PHI` 已经变成 copy。**这一步做完，"寄存器分配 + SSA 解构"就不再抽象。**

**第三步：观察 pipeline 本身**

```bash
opt -passes='default<O2>' -debug-pass-manager loop.O0.ll -o /dev/null 2>&1 | head -100
opt -passes='default<O2>' -print-changed loop.O0.ll -S -o /dev/null
```

第二条只打印真正改变了 IR 的 pass，是理解"O2 到底做了什么"最高效的方式。

**第四步：写一个 pass**

按 [3.3 节](#33-概念式多态写一个-pass)写一个 function pass（统计各类指令数量之类），用 pass plugin 方式编译，`opt -load-pass-plugin` 加载运行，再补一个 lit 测试。**可与 `mlir-toy-dialect` 里的 MLIR Pass 对照，体会两套 Pass 基础设施的同构性。**

**第五步：跨到 MLIR 边界**

```bash
mlir-opt --convert-vector-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts x.mlir \
  | mlir-translate --mlir-to-llvmir \
  | opt -passes='default<O2>' -S \
  | llc -mtriple=x86_64-- -o -
```

把一段 `vector` dialect 的代码一路推到汇编，观察 `vector<8xf32>` 怎么变成 `<8 x float>` 再变成 `vmulps`。**这条命令链是把本文和 MLIR 知识缝起来的那一针。**

---

## 附录：一页速查

```
【四层 IR】
  LLVM IR (.ll/.bc)  Module/Function/BasicBlock/Instruction   ← opt
  SelectionDAG       SDNode/SDValue/MVT，chain 边排副作用      ← llc 内部
  Machine IR (.mir)  MachineFunction/MBB/MI/MO                ← llc -stop-after
  MC                 MCInst/MCStreamer/MCSymbol/MCSection     ← llvm-mc

【IR 类型】 iN  bN(新)  half/bfloat/float/double  ptr(opaque)+addrspace
           <N x T> / <vscale x N x T>  [N x T]  {T,...}  label/token/metadata

【指令九类】terminator / unary / binary / bitwise / vector / aggregate
            / memory&addressing / conversion / other

【三个语义陷阱】
  poison 传染（select 例外）；用在 UB 位置立刻 UB；`and poison,0` 仍是 poison
  freeze 终结 poison
  GEP 第一个索引是"跳几个整对象"，不是字段号

【New PM】
  层级  Module →(CGSCC)→ Function → Loop
  返回  PreservedAnalyses::all() / none() / preserve<X>() / preserveSet<CFGAnalyses>()
  规则  内层只能 getCachedResult 外层分析（防二次复杂度 + 保并发确定性）
  例外  loop pass 可用 LoopStandardAnalysisResults 里的 function 分析
  语法  opt -passes='module(cgscc(function(loop(...))))'
  注意  中端用 New PM，后端 codegen 仍用 Legacy PM

【AA 四答】 NoAlias / MayAlias / PartialAlias / MustAlias（不含"指针相等"）

【后端七阶段】
  1 指令选择 → 2 调度与成形 → 3 机器 SSA 优化 → 4 寄存器分配
  → 5 序言/尾声插入 → 6 后期优化 → 7 代码发射

【SelectionDAG 八步】
  build → combine → legalize types → combine → legalize ops → combine
  → select → schedule
  legalize types: promote / expand / split / widen / scalarize
  legalize ops  : expand / promote / custom

【GlobalISel 四 pass】 IRTranslator → Legalizer → RegBankSelect → InstructionSelect

【寄存器分配】 LiveVariables → LiveIntervals → PHIElimination(SSA解构)
              → two-address → Greedy(默认) → 指令折叠

【接 MLIR 的三个杠杆】 noalias/!alias.scope · align/dereferenceable · TTI 代价模型

【最常用的两个 flag】
  opt  -print-changed
  clang -Rpass-missed=loop-vectorize
```
