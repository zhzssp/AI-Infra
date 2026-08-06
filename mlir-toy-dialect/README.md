# mlir-toy-dialect —— 一个极小的 MLIR Dialect 学习项目

这是一个**树外（out-of-tree）** 的最小 MLIR Dialect 示例，专为学习 MLIR 的核心概念而设计。

## 在自学体系中的位置

| | |
|--|--|
| **角色** | MLIR 深化的动手主战场（P0）：双 dialect · Region · Interface · Dialect Conversion |
| **总规划** | [`../README.md`](../README.md) §3.2（必学清单 + 端到端验收） |
| **配套教材** | [`../docs/mlir-learning-guide.md`](../docs/mlir-learning-guide.md)（机制主教材）· [`../docs/paper-notes/03-mlir.md`](../docs/paper-notes/03-mlir.md)（论文动机） |
| **阶段导航** | [`../docs/README.md` 阶段 2](../docs/README.md#阶段-2mlir-深化p0约-2-周) |
| **上一站** | [`../llvm-hello-compile/`](../llvm-hello-compile/) —— SSA / Pass / lit 的同构预习 |
| **下一站** | [`../docs/iree-learning-guide.md`](../docs/iree-learning-guide.md) —— 把多层 lowering 接到工业运行时 |

> 本项目的核心机制（对照表见下）已经齐；按总规划验收，下一步是把 Toy 接到 `linalg → scf → llvm` 端到端链路上。

---

## 一、你将学到什么

| 概念 | 在本项目中的体现 |
|------|-----------------|
| **Dialect（方言）** | `toy`（高层，数学语义）与 `low`（低层，贴近硬件）**两个** dialect，MLIR 中扩展 IR 的基本单位 |
| **Operation（操作）** | `toy.constant/add/mul/box/unbox/repeat/yield`；`low.constant/add/mul/shl`，即 IR 中的指令 |
| **Type（自定义类型）** | `!toy.num`（`ToyTypes.td`）——可扩展性的第二根支柱，降低时由 TypeConverter 换成 `i32` |
| **Attribute（属性）** | `I32Attr` 承载常量值与移位量；`--toy-print-cost` 动态挂上 `toy.cost` 这类可丢弃属性 |
| **Region / Block / Terminator** | `toy.repeat` 带一个 Region，`toy.yield` 是终结符——**MLIR 相对 LLVM IR 最本质的区别：IR 是可嵌套的树** |
| **ODS / TableGen** | 用 `ToyOps.td` / `LowOps.td` / `ToyTypes.td` / `ToyInterfaces.td` 声明式地定义一切 |
| **Trait（特征）** | `Pure`、`Commutative`、`ConstantLike`、`Terminator`、`SingleBlock` |
| **OpInterface（操作接口）** | `ToyCostOpInterface`：`toy` 与 `low` 两个 dialect 的 op 都实现它，于是 `--toy-print-cost` 这个 Pass **不认识任何具体 dialect 也能工作** |
| **Verifier（验证器）** | ODS 自动生成的类型/结构检查 + 手写的 `RepeatOp::verify()`（`count > 0`、交还类型匹配） |
| **Diagnostics（诊断）** | `emitOpError()` + `-verify-diagnostics` 测试（`test/verify.mlir`） |
| **Folder / Canonicalization** | 两种 fold 都有：`add/mul` 返回 **Attribute**（算出新常量）、`unbox` 返回 **Value**（复用已有值） |
| **Constant Materializer** | 把 fold 算出的"值"重新物化成 `toy.constant` |
| **RewritePattern（模式改写）** | `toy-simplify` Pass 里用 `OpRewritePattern` 实现 `x*1=x`、`x+0=x` |
| **自定义 Pass** | 手写 `ToySimplifyPass`（`PassWrapper` + 贪心驱动器），亲手写一个 Pass |
| **Lowering（降低）—— 两种写法** | ① `--toy-to-low`：贪心驱动器版；② `--toy-to-low-convert`：**Dialect Conversion 版**（`ConversionTarget` + `TypeConverter` + `OpConversionPattern`）。真实编译器一律用后者 |
| **Strength Reduction（强度削减）** | `--low-strength-reduce` 把 `low.mul %x, 2^k` 改写成更廉价的 `low.shl %x, k` |
| **Cost Model（代价模型）** | 借接口给每个 op 标价，量化强度削减的收益：`low.mul(5)` → `low.shl(1)` |
| **多层 IR 分工（Multi-Level）** | 同一段 `x*4`：在 `toy` 层无从优化，降到 `low` 层才削减为 `x<<2`——直观演示"不同层级看到不同信息" |
| **Pass（编译趟）** | `--canonicalize` / `--toy-simplify` / `--toy-to-low` / `--toy-to-low-convert` / `--low-strength-reduce` / `--toy-print-cost` |
| **mlir-opt 风格工具** | 自己的 `toy-opt`，可读入 `.mlir` 文件并跑 Pass |
| **lit + FileCheck 测试** | 9 个用例，含 `-split-input-file`、`-verify-diagnostics`、`--check-prefix` 等常用姿势 |

> 想知道**还有哪些 MLIR 特性本项目故意没做、以及为什么**，见下面的
> 「一点五、MLIR 核心特性覆盖对照表」。

---

## 一点五、MLIR 核心特性覆盖对照表

MLIR 的核心特性可以归成 **五组**。下表说明每一组在本项目里落在哪个文件，
以及**哪些刻意留白**——留白的判断标准是："它是不是理解 MLIR 心智模型的必需品？"
不是的话，就属于"遇到再学"。

### ✅ 已覆盖（构成 MLIR 的完整心智模型）

| # | 特性组 | 具体机制 | 代码位置 |
|---|--------|---------|---------|
| 1 | **IR 数据结构** | Operation / Value / Type / Attribute | `ToyOps.td`、`ToyTypes.td` |
| | | **Region / Block / Terminator（嵌套 IR）** | `toy.repeat`、`toy.yield` |
| 2 | **可扩展性三支柱** | 自定义 Operation | `ToyOps.td` / `LowOps.td` |
| | | 自定义 **Type**（`!toy.num`） | `ToyTypes.td` + `ToyDialect.cpp` 的 `addTypes` |
| | | Attribute（内置 + 动态挂载 `toy.cost`） | `ToyCostPass.cpp` |
| 3 | **抽象与复用** | Trait（是/否的标记） | 各 `.td` 的 `[Pure, Commutative, ...]` |
| | | **OpInterface（能力查询）** | `ToyInterfaces.td` + `ToyCostPass.cpp` |
| 4 | **变换机制** | `fold()` → Attribute（算常量） | `ToyOps.cpp` 的 `AddOp/MulOp::fold` |
| | | `fold()` → Value（复用已有值） | `ToyOps.cpp` 的 `UnboxOp::fold` |
| | | `RewritePattern` + 贪心驱动器 | `ToyPasses.cpp` |
| | | **Dialect Conversion**（Target/TypeConverter） | `ConvertToyToLow.cpp` |
| | | 自定义 Pass、Pass 注册与调度 | 各 `*Passes.cpp`、`toy-opt.cpp` |
| 5 | **正确性与工程化** | Verifier + `emitOpError` 诊断 | `ToyOps.cpp` 的 `RepeatOp::verify` |
| | | ODS/TableGen 全链路（op/type/interface） | `include/*/CMakeLists.txt` |
| | | lit + FileCheck 测试 | `test/` |

### ⏸ 刻意留白（遇到再学，不影响理解 MLIR）

| 特性 | 为什么现在不学 | 什么时候会撞上 |
|------|--------------|--------------|
| **DRR（`Pat<>` 声明式重写）** | 只是 `RewritePattern` 的 TableGen 糖衣，机制完全一样。先看懂 C++ 版，DRR 半小时就能上手 | 写大量简单一对一改写规则时 |
| **TableGen 定义 Pass（`Passes.td` + `GEN_PASS_DEF`）** | 只是 `PassWrapper` 的工程化包装，多了 Pass 选项/统计/依赖声明 | 给 Pass 加命令行选项时 |
| **`--pass-pipeline` 文本管线、PassManager 嵌套** | 属于调度层，理解"Pass 是什么"之后自然会用 | 组织真实编译流水线时 |
| **Symbol / SymbolTable / 模块间引用** | 本项目直接复用内置 `func.func` | 做跨函数分析、内联时 |
| **Bufferization（tensor → memref）** | 是一整套独立子系统，需要先有张量语义 | 做真实的 AI 编译器降低时 |
| **降低到 LLVM Dialect + JIT 执行** | 只是再多一次 lowering，机制与 `toy→low` 完全同构 | 需要真的把 IR 跑起来时 |
| **数据流分析框架、DataLayout、位置追踪** | 高级话题，用得着的场景很窄 | 写复杂分析型 Pass 时 |

**一句话结论**：跑通本项目的 9 组演示 + 读完 10 个测试用例，
你就已经掌握了阅读任意一个 MLIR 项目（IREE / Triton / Torch-MLIR）所需的全部基础概念，
剩下的都是"某个具体 dialect 的领域知识"，而不是"MLIR 本身的机制"。

---

## 二、目录结构

```
mlir-toy-dialect/
├── CMakeLists.txt              # 顶层构建配置（find_package MLIR）
├── README.md                   # 项目说明 / 启动流程 / 参考资料
├── MLIR-运行流程与关键组件.md    # MLIR 运行流程与关键组件的详细讲解
├── .gitignore
├── include/Toy/                # 【高层】toy dialect（数学语义）
│   ├── CMakeLists.txt          # 调用 mlir-tblgen 生成 .inc（op / type / interface 三套）
│   ├── ToyDialect.td           # Dialect 定义（TableGen/ODS）
│   ├── ToyOps.td               # Operation 定义：算术 / 装箱拆箱 / 带 Region 的 repeat
│   ├── ToyTypes.td             # ★ 自定义类型 !toy.num
│   ├── ToyInterfaces.td        # ★ 操作接口 ToyCostOpInterface
│   ├── ToyDialect.h            # Dialect C++ 声明
│   ├── ToyOps.h                # Operation C++ 声明
│   ├── ToyTypes.h              # ★ 自定义类型 C++ 声明
│   ├── ToyInterfaces.h         # ★ 接口 C++ 声明
│   └── ToyPasses.h             # 自定义 Pass 声明
├── include/Low/                # 【低层】low dialect（贴近硬件，含 shl 移位）
│   ├── CMakeLists.txt          # 调用 mlir-tblgen 生成 .inc 文件
│   ├── LowDialect.td           # Dialect 定义
│   ├── LowOps.td               # low.constant/add/mul/shl 定义（也实现了代价接口）
│   ├── LowDialect.h            # Dialect C++ 声明
│   ├── LowOps.h                # Operation C++ 声明
│   └── LowPasses.h             # 降低 / 强度削减 / 转换 Pass 声明
├── lib/
│   ├── CMakeLists.txt
│   ├── ToyDialect.cpp          # Dialect 注册 + 类型注册 + materializeConstant
│   ├── ToyOps.cpp              # fold（两种）/ verify / 接口方法 getCost
│   ├── ToyInterfaces.cpp       # ★ 接口生成代码的落地点
│   ├── ToyPasses.cpp           # 自定义 Pass + RewritePattern（x*1=x, x+0=x）
│   ├── ToyCostPass.cpp         # ★ 只依赖接口、不认识任何 dialect 的通用 Pass
│   ├── LowDialect.cpp          # low dialect 注册
│   ├── LowOps.cpp              # low.constant 的 fold() + low.mul 的 getCost()
│   ├── LowPasses.cpp           # 【核心】降低 Pass(toy→low) + 强度削减 Pass(mul→shl)
│   └── ConvertToyToLow.cpp     # ★ 同一个降低的 Dialect Conversion 版（对照阅读）
├── tools/toy-opt/
│   ├── CMakeLists.txt
│   └── toy-opt.cpp             # mlir-opt 风格的命令行工具（注册两个 dialect + 五个 Pass）
├── scripts/                    # 一键启动/构建/测试脚本（Linux + conda）
│   ├── env.sh                  # 公共环境（激活 conda、导出路径）
│   ├── setup.sh                # 安装依赖（幂等）
│   ├── build.sh                # 配置 + 编译
│   ├── run.sh                  # 手动运行九个演示
│   ├── test.sh                 # lit 自动化测试
│   └── all.sh                  # 一键跑通全流程
└── test/
    ├── CMakeLists.txt          # lit 测试目标（check-toy）
    ├── lit.cfg.py              # lit 配置
    ├── lit.site.cfg.py.in      # lit site 配置模板
    ├── ops.mlir                # 基本语法解析测试
    ├── canonicalize.mlir       # 常量折叠优化测试
    ├── simplify.mlir           # 代数化简（RewritePattern）测试
    ├── lowering.mlir           # 降低测试：toy.* → low.*
    ├── strength.mlir           # 强度削减测试：x*4 → x<<2（多层分工对比）
    ├── region.mlir             # ★ Region 嵌套：Pass 自动递归进区域
    ├── verify.mlir             # ★ Verifier + 诊断（-verify-diagnostics）
    ├── types.mlir              # ★ 自定义类型 !toy.num + fold 返回 Value
    ├── convert.mlir            # ★ Dialect Conversion 版降低 + 类型转换
    └── cost.mlir               # ★ OpInterface 驱动的跨 dialect 代价统计
```

> 打 ★ 的是为了补齐 MLIR 核心特性而后加的部分。

---

## 三、Linux + conda 一键启动（推荐，本机已实测）

> 本机已用 **conda-forge 的预编译 MLIR 17.0.6** 搭好环境（无需自己从源码编译 LLVM，
> 省去数小时）。所有启动/构建/测试步骤都固化在 `scripts/` 目录，一条命令即可跑通。

### 3.0 一键全流程

```bash
cd ~/AI-Infra/mlir-toy-dialect
bash scripts/all.sh          # 装依赖 → 构建 → 自动化测试 → 手动演示
```

### 3.1 分步执行（理解每一步）

```bash
bash scripts/setup.sh        # ① 安装依赖：conda 环境 mlir-env（mlir/llvm/cmake/ninja/编译器/lit），幂等
bash scripts/build.sh        # ② 配置+编译，产出 build/bin/toy-opt（加 clean 可全新构建）
bash scripts/test.sh         # ③ lit 自动化测试（check-toy，共 10 个用例）
bash scripts/run.sh          # ④ 手动运行九组演示，肉眼观察输出
```

### 3.2 环境要点（脚本已自动处理，了解即可）

| 项 | 值 / 说明 |
|----|-----------|
| conda 环境 | `mlir-env`（前缀 `~/miniforge3/envs/mlir-env`） |
| MLIR/LLVM | conda-forge `mlir=17.0.6` / `llvm=17.0.6`（含头文件、静态库、CMake 配置、`mlir-tblgen`、`FileCheck`） |
| 编译器 | conda-forge `cxx-compiler`（GCC 14，与 conda 二进制 ABI 匹配） |
| 测试驱动 | `lit`（conda 安装），`FileCheck` 软链到 `<env>/bin` |
| 为何用 conda | 本机系统 GLIBC 偏旧，官方 Ubuntu 预编译包跑不起来；conda-forge 包兼容性最好 |

> 覆盖默认路径：`CONDA_HOME=/path/to/conda TOY_ENV=my-env bash scripts/all.sh`

---

## 四、完整启动流程（Windows 从零到跑通，备查）

> ⚠️ 本项目**不能独立编译**，它依赖一份已经构建好的 LLVM + MLIR。
> 因此启动分四个阶段：装工具 → 编译 MLIR → 编译本项目 → 运行测试。
> 下面是 Windows/VS 自行编译 LLVM 的完整流程，作为不使用 conda 时的备选参考。

### 阶段 0：安装工具链（Windows 当前缺失，必须先装）

本机需要以下工具。请在普通（非 VS）环境下安装：

| 工具 | 说明 / 下载 |
|------|------------|
| **Visual Studio 2022 Community** | 安装时勾选 **"使用 C++ 的桌面开发"**（提供 MSVC 编译器 `cl.exe`）。https://visualstudio.microsoft.com/ |
| **Git** | https://git-scm.com/download/win |
| **CMake** | 安装时勾选 "Add to PATH"。https://cmake.org/download/ |
| **Ninja** | 解压后把 `ninja.exe` 放入 PATH。https://github.com/ninja-build/ninja/releases |
| **Python 3** | https://www.python.org/downloads/ |

> **重要**：之后所有命令都应在 **"x64 Native Tools Command Prompt for VS 2022"**（开始菜单搜索）里运行，它已配好 MSVC 环境。
>
> **更省心的替代方案**：装 WSL2（`wsl --install`），在 Ubuntu 里 `sudo apt install cmake ninja-build clang git python3`，MLIR 在 Linux 下构建更顺。下面的命令以 Windows+VS 为例，WSL 下基本一致（把路径换成 Linux 路径、去掉 `^` 行续接符）。

---

### 阶段 1：获取并编译 LLVM + MLIR（最耗时，约 30 分钟～数小时）

在 **x64 Native Tools Command Prompt** 中：

```bat
:: 选磁盘空间充足的位置（编译产物约 20~50GB）
cd C:\dev

:: 1. 克隆 llvm-project（很大，耐心等）
git clone --depth 1 https://github.com/llvm/llvm-project.git
cd llvm-project

:: 2. 配置（开启 MLIR，Release，安装工具）
cmake -G Ninja -S llvm -B build ^
  -DLLVM_ENABLE_PROJECTS=mlir ^
  -DLLVM_TARGETS_TO_BUILD=host ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DLLVM_ENABLE_ASSERTIONS=ON ^
  -DLLVM_INSTALL_UTILS=ON

:: 3. 编译（耗时，取决于 CPU 核数，可加 -j 控制并发）
cmake --build build
```

编译成功后，记下这两个路径（**下一步要用**）：

```
C:\dev\llvm-project\build\lib\cmake\mlir
C:\dev\llvm-project\build\lib\cmake\llvm
```

---

### 阶段 2：编译本项目 `mlir-toy-dialect`

仍在 **x64 Native Tools Command Prompt** 中：

```bat
cd C:\Users\hanishzheng\Desktop\AI-Infra\mlir-toy-dialect

:: 配置：把 MLIR_DIR / LLVM_DIR 指向阶段 1 的 build 目录
cmake -G Ninja -S . -B build ^
  -DMLIR_DIR=C:\dev\llvm-project\build\lib\cmake\mlir ^
  -DLLVM_DIR=C:\dev\llvm-project\build\lib\cmake\llvm ^
  -DLLVM_EXTERNAL_LIT=C:\dev\llvm-project\build\bin\llvm-lit.py

:: 编译
cmake --build build
```

成功后会生成：

```
build\bin\toy-opt.exe
```

---

### 阶段 3：运行测试（你最关心的部分）

#### 方式 A：手动运行，肉眼观察（最直观，推荐先做）

```bat
cd C:\Users\hanishzheng\Desktop\AI-Infra\mlir-toy-dialect

:: 测试 1：解析并回显 —— 能正确打印 toy.* 即说明 dialect 解析器 OK
build\bin\toy-opt.exe test\ops.mlir

:: 测试 2：常量折叠优化 —— 这是重点！
build\bin\toy-opt.exe test\canonicalize.mlir --canonicalize
```

**测试 2 的预期效果**：输入里的 `toy.add %0, %1`（即 `10 + 20`）会被折叠成
`toy.constant 30 : i32`；`(10 + 20) * 2` 会被折叠成 `toy.constant 60 : i32`。
当你看到 `toy.add` / `toy.mul` **消失、只剩常量**，就说明你亲手实现的 `fold()` 优化生效了。

#### 方式 B：用 `lit` 做自动化测试（工程化，一键验证）

项目里已内置 `lit` 测试配置。配置本项目时只要指定了 `LLVM_EXTERNAL_LIT`（阶段 2 已写），即可：

```bat
:: 编译测试目标（会顺带确保 toy-opt 是最新的）
cmake --build build --target check-toy
```

`check-toy` 会自动：
1. 用 `toy-opt` 跑 `test/*.mlir`；
2. 用 `FileCheck` 对照 `// CHECK:` 断言；
3. 汇总 PASS / FAIL。

例如 `test/canonicalize.mlir` 头部包含：

```mlir
// RUN: toy-opt %s --canonicalize | FileCheck %s
// CHECK-NOT: toy.add
// CHECK: toy.constant 30 : i32
```

`lit` 会解析 `RUN:` 执行命令、`CHECK:` 校验输出，全部通过才会报 PASS。

---

## 五、预期输出示例

`test/canonicalize.mlir --canonicalize` 的预期输出（节选）：

```mlir
func.func @fold_add() -> i32 {
  %0 = toy.constant 30 : i32
  return %0 : i32
}
func.func @fold_add_mul() -> i32 {
  %0 = toy.constant 60 : i32
  return %0 : i32
}
```

可以看到：原来的 `toy.add` / `toy.mul` 被编译期折叠消除，只留下结果常量。

---

## 五点五、多层 Dialect 分工协作（本项目的核心亮点）

> 单个 dialect 只能演示"封装语义 + 单层优化"。为了直观展示 MLIR 名字里 **"Multi-Level"（多层）** 的真正价值，
> 本项目在高层 `toy` dialect 之外，额外新增了一个**低层 `low` dialect**，并用两个 Pass 串起一条
> 完整的 **"逐层降低 + 分层优化"** 链路。

### 5.5.1 两层 dialect 的定位对比

| | **`toy` 层（高层）** | **`low` 层（低层）** |
|---|---|---|
| **关注点** | 数学 / 代数语义 | 硬件指令代价 |
| **操作** | `constant` / `add` / `mul` | `constant` / `add` / `mul` / **`shl`（左移）** |
| **独有概念** | — | **`shl` 移位——高层根本没有"移位"这个概念** |
| **典型优化** | `x*1→x`、`x+0→x`、常量折叠 | **强度削减** `x*2^k → x<<k`（移位比乘法便宜） |

### 5.5.2 完整链路（两个 Pass 串联）

```
toy.mul %x, 4                          ← 高层：一次普通乘法
      │  --toy-to-low   (降低 / lowering)
      ▼
low.mul %x, 4                          ← 低层：语义等价，只是换了层
      │  --low-strength-reduce  (强度削减)
      ▼
low.shl %x, 2                          ← 低层：x*4 = x<<2，用更廉价的移位替代乘法
```

### 5.5.3 一眼看懂"分层"的意义（对比 `scripts/run.sh` 演示 5-a / 5-b）

同一段 `x * 4`：

```mlir
// 演示 5-a：留在【高层 toy 层】跑 --toy-simplify —— 纹丝不动
%0 = toy.constant 4 : i32
%1 = toy.mul %arg0, %0 : i32     // 没变！高层只懂 x*1=x，且没有"移位"概念

// 演示 5-b：降到【低层 low 层】跑 --low-strength-reduce —— 成功优化
%0 = low.shl %arg0, 2 : i32      // x*4 变成了 x<<2！
```

而 `x * 3`（非 2 的幂）在 low 层保持 `low.mul` 不变——说明强度削减是**有条件的、依赖低层特有信息的优化**。

**核心结论**：`x*4→x<<2` 这个优化机会在高层不是"没做"，而是**根本无法表达**（高层没有 `shl` 操作）。
必须先 lowering 到能表达"移位"的 `low` 层，优化窗口才打开。这正是 MLIR "Multi-Level" 的精髓——
**在信息最合适的那一层，做最合适的优化**。

### 5.5.4 代码入口

| 关注点 | 文件 |
|--------|------|
| 低层操作定义（尤其是高层没有的 `low.shl`） | `include/Low/LowOps.td` |
| 降低 Pass（`toy.*` → `low.*`，`replaceOpWithNewOp`） | `lib/LowPasses.cpp` 第一部分 |
| 强度削减 Pass（`isPowerOfTwoConst` + `low.mul` → `low.shl`） | `lib/LowPasses.cpp` 第二部分 |
| 降低测试 | `test/lowering.mlir` |
| 分层对比测试 | `test/strength.mlir` |

---

## 六、建议的学习顺序

按下面的顺序读，每一步只解决一个问题，大约 3～4 小时能把 MLIR 的心智模型建起来。

**第 1 步：IR 长什么样（30 分钟）**
1. `include/Toy/ToyOps.td` —— 声明式地定义 Operation：输入、输出、属性、trait、装配格式。
2. 构建后看 `build/include/Toy/ToyOps.h.inc` —— 亲眼确认 TableGen 到底替你生成了多少样板。
3. `test/ops.mlir` + `scripts/run.sh` 演示 1 —— 解析/打印往返。

**第 2 步：IR 是一棵树，不是一条指令流（20 分钟）**

4. `ToyOps.td` 第三部分的 `toy.repeat` / `toy.yield` + `test/region.mlir`。
   这是**MLIR 区别于 LLVM IR 的最本质一点**：Operation 里可以嵌 Region，
   Region 里嵌 Block，Block 里再嵌 Operation。`scf.for`、`gpu.launch`、
   `linalg.generic`、`func.func` 全是这么来的。
   同时注意：你写 Pass 时**不用管嵌套**，框架会自动递归。

**第 3 步：怎么改 IR —— 三条主线（60 分钟）**

5. `lib/ToyOps.cpp` 的 `fold()` —— 最窄的一条：就地算值。
   看两种返回值的区别：`AddOp::fold` 返回 Attribute（新常量，配合 `materializeConstant`），
   `UnboxOp::fold` 返回 Value（复用已有值）。
6. `lib/ToyPasses.cpp` —— 第二条：`RewritePattern` + 贪心驱动器，通用的子图改写。
7. `lib/ConvertToyToLow.cpp` —— 第三条：**Dialect Conversion**。
   一定要和 `lib/LowPasses.cpp` 的贪心版**并排对比着读**，文件开头有一张对照表。
   记住结论：真实编译器的 lowering 一律用 Dialect Conversion，因为它把
   "我到底降完了没有"变成框架强制检查的契约，还顺带解决类型转换。

**第 4 步：多层分工与可扩展性（60 分钟）**

8. `lib/LowPasses.cpp` 的强度削减 + 第「五点五」节 + `run.sh` 演示 5-a/5-b ——
   体会"优化机会在高层不是没做，而是根本无法表达"。
9. `include/Toy/ToyTypes.td` + `test/types.mlir` —— 类型也是可扩展的，
   而且降低时会被 `TypeConverter` 换掉。
10. `include/Toy/ToyInterfaces.td` + `lib/ToyCostPass.cpp` + `test/cost.mlir` ——
    **本项目观念密度最高的一处**：一个 Pass 不认识任何 dialect，
    只问 op "你实现接口了吗"，就能同时服务 toy 层和 low 层。
    这就是 MLIR 能容纳上百个 dialect 却共用一套通用 Pass 的原因。

**第 5 步：正确性（20 分钟）**

11. `RepeatOp::verify()` + `test/verify.mlir` —— 验证器在每个 Pass 之后自动跑，
    是 MLIR 的安全网。注意"结构约束交给 trait，语义约束才手写 verifier"的分工。

**第 6 步：动手改（越久越好）**

- 加一个新 Op（比如 `toy.sub`），走一遍 `.td` → `.cpp` → 测试的完整流程；
- 给 `toy.mul` 加代数化简：`x * 1 = x`、`x * 0 = 0`；
- 把 `ConvertToyToLow.cpp` 里的 `MulOpLowering` 从 `patterns.add<>` 删掉再跑，
  观察 `failed to legalize operation 'toy.mul'` 这条报错——然后换成贪心版
  `--toy-to-low` 跑同样的输入，你会看到它**一声不吭**地留下了 `toy.mul`。
  这一个实验就足以说明为什么真实编译器不用贪心版做 lowering；
- 给 `toy.repeat` 加上 `IsolatedFromAbove` trait，看看会报什么错，
  想清楚"隔离区域"为什么是函数和 GPU kernel 的必需品；
- 给 `low.shl` 也实现一次 `getCost()` 返回 0，再跑 `--toy-print-cost`，
  体会 cost model 是怎么影响"优化到底值不值"这个判断的。

---

## 七、常见问题

- **`find_package(MLIR)` 失败**：确认 `-DMLIR_DIR` 指向的是构建目录里的 `lib/cmake/mlir`，而不是源码目录。
- **`check-toy` 找不到 `llvm-lit`**：确认 `LLVM_EXTERNAL_LIT` 指向 `build/bin/llvm-lit.py` 或 `llvm-lit`；该文件在阶段 1 编译后存在。
- **链接报错找不到符号**：检查 `lib/CMakeLists.txt` 里的 `LINK_LIBS` 是否漏了对应 MLIR 库。
- **Windows 下 `cl.exe` 找不到**：确保是在 **x64 Native Tools Command Prompt** 里运行，而不是普通 PowerShell。

---

## 八、参考资料（外部文档与论文）

> 以下为学习本项目时最值得查阅的官方文档与论文，按推荐阅读顺序排列。

### 官方文档（https://mlir.llvm.org/）

| 主题 | 链接 | 与本项目的对应关系 |
|------|------|-------------------|
| **MLIR 主页 / 文档总入口** | https://mlir.llvm.org/ | 所有文档的索引 |
| **Getting Started（构建 MLIR）** | https://mlir.llvm.org/getting_started/ | 对应启动流程「阶段 1」编译 LLVM+MLIR |
| **Creating a Dialect 教程** | https://mlir.llvm.org/docs/Tutorials/CreatingADialect/ | **最重要**——本项目即按此教程实现，可逐段对照 `ToyOps.td` / `ToyOps.cpp` |
| **OpDefinitions（ODS / TableGen 参考）** | https://mlir.llvm.org/docs/OpDefinitions/ | 查 `assemblyFormat`、`hasFolder`、`Trait` 等字段的权威含义 |
| **Canonicalization（规范化）** | https://mlir.llvm.org/docs/Canonicalization/ | 理解 `fold()` 与 `--canonicalize` 的底层机制，以及为何 `materializeConstant` 不可或缺 |
| **Pass Management（Pass 基础设施）** | https://mlir.llvm.org/docs/PassManagement/ | 理解 Pass 如何调度、遍历 IR |
| **LangRef（MLIR 规范）** | https://mlir.llvm.org/docs/LangRef/ | `.mlir` 文本语法的完整定义 |
| **Rationale（设计动机）** | https://mlir.llvm.org/docs/Rationale/ | 解释 MLIR 为何这样设计 |

### 官方最小示例（本项目的"亲兄弟"）

- **`mlir/examples/standalone`**（llvm-project 源码树内）
  GitHub：https://github.com/llvm/llvm-project/tree/main/mlir/examples/standalone
  说明：官方版的 out-of-tree 极小 dialect，同样有 `standalone-opt`。本项目在其基础上增加了中文教学注释、显式的 `fold`/`materializeConstant` 演示以及 lit 测试。建议对照阅读其 `README.md`。

### 论文

- **MLIR: A Compiler Infrastructure for the End of Moore's Law**（Lattner, Amini, et al., 2020 IEEE HCS）
  arXiv：https://arxiv.org/abs/2002.11054
  说明：MLIR 的设计哲学必读，解释了"为什么需要 dialect / 多层 IR"。对应学习计划第二阶段。

祝学习愉快！从 `include/Toy/ToyOps.td` 开始读起。
