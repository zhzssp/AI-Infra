# mlir-toy-dialect —— 一个极小的 MLIR Dialect 学习项目

这是一个**树外（out-of-tree）** 的最小 MLIR Dialect 示例，专为学习 MLIR 的核心概念而设计。
它对应学习路线第二阶段中的 **"MLIR：实现一个非常小的 MLIR Dialect"** 这一动手实践环节。

---

## 一、你将学到什么

| 概念 | 在本项目中的体现 |
|------|-----------------|
| **Dialect（方言）** | `toy` dialect，MLIR 中扩展 IR 的基本单位 |
| **Operation（操作）** | `toy.constant`、`toy.add`、`toy.mul`，即 IR 中的指令 |
| **ODS / TableGen** | 用 `ToyOps.td` 声明式地定义 Dialect 和 Operation |
| **Trait（特征）** | `Pure`、`Commutative`、`ConstantLike` 等 Op 属性标记 |
| **Verifier / Builder** | 由 ODS 自动生成的解析、打印、构造样板 |
| **Folder / Canonicalization** | 常量折叠优化，理解 MLIR 的 rewrite 机制 |
| **Constant Materializer** | 把 fold 算出的"值"重新物化成 `toy.constant` |
| **Pass（编译趟）** | 通过 `toy-opt --canonicalize` 触发优化 |
| **mlir-opt 风格工具** | 自己的 `toy-opt`，可读入 `.mlir` 文件并跑 Pass |

---

## 二、目录结构

```
mlir-toy-dialect/
├── CMakeLists.txt              # 顶层构建配置（find_package MLIR）
├── README.md                   # 项目说明 / 启动流程 / 参考资料
├── MLIR-运行流程与关键组件.md    # MLIR 运行流程与关键组件的详细讲解
├── .gitignore
├── include/Toy/
│   ├── CMakeLists.txt          # 调用 mlir-tblgen 生成 .inc 文件
│   ├── ToyDialect.td           # Dialect 定义（TableGen/ODS）
│   ├── ToyOps.td               # Operation 定义（TableGen/ODS）
│   ├── ToyDialect.h            # Dialect C++ 声明
│   └── ToyOps.h                # Operation C++ 声明
├── lib/
│   ├── CMakeLists.txt
│   ├── ToyDialect.cpp          # Dialect 注册 + materializeConstant
│   └── ToyOps.cpp              # Operation 实现（含 fold/canonicalize）
├── tools/toy-opt/
│   ├── CMakeLists.txt
│   └── toy-opt.cpp             # mlir-opt 风格的命令行工具
└── test/
    ├── CMakeLists.txt          # lit 测试目标（check-toy）
    ├── lit.cfg.py              # lit 配置
    ├── lit.site.cfg.py.in      # lit site 配置模板
    ├── ops.mlir                # 基本语法解析测试
    └── canonicalize.mlir       # 常量折叠优化测试
```

---

## 三、完整启动流程（从零到跑通）

> ⚠️ 本项目**不能独立编译**，它依赖一份已经构建好的 LLVM + MLIR。
> 因此启动分四个阶段：装工具 → 编译 MLIR → 编译本项目 → 运行测试。

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

## 四、预期输出示例

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

## 五、建议的学习顺序

1. **先看 `include/Toy/ToyOps.td`** —— 理解如何用声明式的 TableGen 定义一个 Operation（输入/输出/trait）。
2. **看生成的 `.inc` 文件**（构建后在 `build/include/Toy/` 下）—— 理解 TableGen 到底生成了什么 C++ 代码。
3. **看 `lib/ToyOps.cpp` 中的 `fold()`** —— 理解 MLIR 优化是怎么写的；注意它与 `materializeConstant` 的配合。
4. **动手改**：
   - 加一个新 Op（比如 `toy.sub`），走一遍完整流程；
   - 给 `toy.mul` 加代数化简：`x * 1 = x`、`x * 0 = 0`；
   - 写一个自定义 Pass，把 `toy.add` 换成 `toy.mul`。

---

## 六、常见问题

- **`find_package(MLIR)` 失败**：确认 `-DMLIR_DIR` 指向的是构建目录里的 `lib/cmake/mlir`，而不是源码目录。
- **`check-toy` 找不到 `llvm-lit`**：确认 `LLVM_EXTERNAL_LIT` 指向 `build/bin/llvm-lit.py` 或 `llvm-lit`；该文件在阶段 1 编译后存在。
- **链接报错找不到符号**：检查 `lib/CMakeLists.txt` 里的 `LINK_LIBS` 是否漏了对应 MLIR 库。
- **Windows 下 `cl.exe` 找不到**：确保是在 **x64 Native Tools Command Prompt** 里运行，而不是普通 PowerShell。

---

## 七、参考资料（外部文档与论文）

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
