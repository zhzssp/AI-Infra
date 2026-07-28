# MLIR 运行流程与关键组件详解（以 toy dialect 为例）

> 本文以本仓库 `mlir-toy-dialect` 项目为标本，把 MLIR 的**运行流程**和**关键组件**串起来讲一遍。
> 配合 [`README.md`](./README.md) 的"启动流程"和"参考资料"一起阅读效果最佳。
>
> 核心一句话：**MLIR 的"代码"分两层——你在 `.td` 里声明式地描述 IR，TableGen 把它翻译成 C++，再和你的手写 C++ 一起编译成工具。**

---

## 一、总览：两条时间线

```
┌─────────────────────── 编译期 (build time) ───────────────────────┐
│                                                                    │
│   ToyOps.td / ToyDialect.td  (你写的"声明")                         │
│            │  mlir-tblgen  (代码生成器)                            │
│            ▼                                                       │
│   ToyOps.h.inc / ToyOps.cpp.inc / ToyOpsDialect.*.inc  (生成的C++)  │
│            │  再和你的手写代码合并                                  │
│            ▼                                                       │
│   ToyOps.cpp / ToyDialect.cpp  (fold、materializeConstant 等)      │
│            │  cmake + 编译器                                        │
│            ▼                                                       │
│   libMLIRToy.a  +  toy-opt.exe                                     │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼  (工具产出后，进入运行期)
┌─────────────────────── 运行期 (runtime) ──────────────────────────┐
│                                                                    │
│   test/canonicalize.mlir  (文本形式的 IR)                          │
│        │  toy-opt 读入 → Parser                                    │
│        ▼                                                           │
│   IR 对象 (in-memory: ModuleOp 内含 toy.add / toy.constant...)     │
│        │  运行 Pass: --canonicalize                                │
│        ▼                                                           │
│   每个 Op 的 fold() 被调用 → 常量折叠 → materializeConstant 物化    │
│        │  Printer                                                  │
│        ▼                                                           │
│   输出文本（toy.add 没了，只剩 toy.constant 30）                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 二、编译期详解（你写的 → 机器能用的）

### 1) `.td` 文件 = 声明式定义（"我要什么样的 IR"）

- `include/Toy/ToyDialect.td`：声明 dialect 叫 `toy`，命名空间 `mlir::toy`，并声明 `hasConstantMaterializer`。
- `include/Toy/ToyOps.td`：声明三个 Op（`toy.constant` / `toy.add` / `toy.mul`），每个 Op 用一行行像这样描述：

```tablegen
def Toy_AddOp : Toy_Op<"add", [Pure, Commutative]> {
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasFolder = 1;
}
```

你在这里**只是描述**：这个 Op 有几个输入、几个输出、长什么样、有什么 trait、要不要 fold。**没有写任何 C++ 解析/打印逻辑**——它们由下面的代码生成器自动产生。

### 2) `mlir-tblgen` = 代码生成器（`.td → .inc`）

由 `include/Toy/CMakeLists.txt` 里的 `add_mlir_dialect(ToyOps toy)` 触发。它读 `ToyOps.td`，生成：

| 生成文件 | 内容 | 被谁 include |
|---------|------|-------------|
| `ToyOps.h.inc` | 三个 Op 的**类声明**（含 parse/print/verify 声明） | `ToyOps.h` |
| `ToyOps.cpp.inc` | 三个 Op 的**类定义**（含自动生成的 parse/print/verify 实现） | `ToyOps.cpp` |
| `ToyOpsDialect.h.inc / .cpp.inc` | `ToyDialect` 类的声明/定义 | `ToyDialect.h / .cpp` |

> **学习重点**：这就是 MLIR 最妙的机制——你不用手写 `toy.add` 的解析器（比如怎么从文本 `toy.add %0, %1 : i32` 读进来），`mlir-tblgen` 根据你写的 `assemblyFormat` 自动生成。你只写"语义"。

### 3) 手写 C++ = 实现"生成器写不了"的部分

- `lib/ToyOps.cpp`：实现 `fold()`（常量折叠逻辑），这是**算法**，生成器写不了，必须手写。
- `lib/ToyDialect.cpp`：实现 `materializeConstant()`（把折叠出的"值"重新变成一个 `toy.constant` 节点），以及 `initialize()`（把 Op 注册进 dialect，解析器才认识它们）。

### 4) 链接成库和工具

- `lib/CMakeLists.txt` → 产出 `MLIRToy` 库。
- `tools/toy-opt/CMakeLists.txt` → 产出 `toy-opt.exe`。
- `tools/toy-opt/toy-opt.cpp`：只有几十行，核心是 `MlirOptMain(..., registry)`——复用 MLIR 自带的 opt 主循环，把你的 `ToyDialect` 注册进去。

---

## 三、运行期详解（文本 → IR → 优化 → 文本）

以 `toy-opt test/canonicalize.mlir --canonicalize` 为例，走一遍：

**① 解析（Parser）**
`toy-opt` 读入文本，遇到 `toy.add %0, %1 : i32`，调用 `mlir-tblgen` 自动生成的 `AddOp::parse`，在内存里造出一个 `AddOp` 对象。整份文件变成一个 `ModuleOp`（IR 的根），里面是一棵操作树。

**② 运行 Pass（`--canonicalize`）**
Canonicalizer（规范化 Pass）遍历 IR，对**每个 Op 调用它的 `fold()`**：
- `toy.add(10, 20)` → `AddOp::fold` 发现两个操作数都是常量 → 算出 `30` → 返回 `IntegerAttr(30)`。
- 框架拿到这个"值"，调用 `ToyDialect::materializeConstant(30)` → 造出一个新的 `toy.constant 30` 节点，**替换**掉原来的 `toy.add`。
- 于是 `toy.add` 被消除，只剩常量。

**③ 打印（Printer）**
把优化后的 IR 对象重新序列化成文本输出。你看到的就是折叠结果。

> 关键理解：`fold()` 只负责"算出一个值"，`materializeConstant()` 负责"把值变回一个 IR 节点"。两者配合，才是 MLIR 做常量折叠的标准姿势。这也是初学者最容易卡住的地方。

---

## 三·五、MLIR 改写 IR 的【两条主线】（fold vs RewritePattern）

上面的 `fold()` 只是改写 IR 的第一条主线。MLIR 里更通用、更常用的是第二条主线
——**RewritePattern（模式改写）**。本项目用 `--toy-simplify` 这个自定义 Pass 演示它。

| | **fold()** | **RewritePattern** |
|---|-----------|--------------------|
| 思路 | "把这个 op 算成一个常量值" | "匹配一个 IR 子图，重写成另一个子图" |
| 能力 | 窄：主要用于常量折叠 | 宽：任意结构改写（化简、lowering、conversion 都靠它） |
| 写在哪 | Op 的 `fold()`（`ToyOps.cpp`） | `OpRewritePattern::matchAndRewrite`（`ToyPasses.cpp`） |
| 本项目例子 | `10 + 20 → 30`（常量+常量） | `x * 1 → x`、`x + 0 → x`（含非常量操作数） |

`--toy-simplify` 的运行链路：

```
toy.mul %x, (toy.constant 1)
      │  Pass: runOnOperation()
      ▼
RewritePatternSet{ SimplifyMulByOne, SimplifyAddZero }
      │  applyPatternsAndFoldGreedily 反复遍历，命中即重写，直到不动点
      ▼
SimplifyMulByOne::matchAndRewrite 命中 → rewriter.replaceOp(mul, %x)
      │  贪心驱动器顺带 DCE，清理变成死代码的 toy.constant 1
      ▼
只剩 %x（toy.mul 消失）
```

> 一句话：**`fold()` 是"算值"，`RewritePattern` 是"换结构"。** 真正写编译器
> Pass、做多层 IR lowering 时，用得最多的是 RewritePattern 这条主线。

---

## 四、关键组件清单（对照本项目文件）

| 组件 | 作用 | 本项目位置 |
|------|------|-----------|
| **Dialect** | IR 的命名空间/扩展单位 | `ToyDialect.td`、`ToyDialect.cpp` |
| **Operation** | IR 中的"指令" | `ToyOps.td`、`ToyOps.cpp` |
| **ODS / TableGen** | 声明式定义 IR 的 DSL | `*.td` 文件 |
| **mlir-tblgen** | 把 `.td` 翻译成 C++ 的代码生成器 | 由 CMake `add_mlir_dialect` 触发 |
| **Trait** | Op 的属性标记（如 `Pure`=无副作用，`Commutative`=可交换） | `ToyOps.td` 里 `[Pure, Commutative]` |
| **assemblyFormat** | 声明 Op 的文本语法，自动生成 parse/print | `ToyOps.td` 各 Op 内 |
| **fold()** | 常量折叠等编译期化简的入口（"算值"式改写） | `ToyOps.cpp` |
| **materializeConstant** | 把折叠值物化成常量 Op | `ToyDialect.cpp` |
| **RewritePattern** | 结构化"匹配子图 → 重写子图"的通用改写（"换结构"式改写） | `ToyPasses.cpp` |
| **自定义 Pass** | 手写 `PassWrapper` + 贪心驱动器，实现 `x*1=x`、`x+0=x` | `ToyPasses.cpp` |
| **Pass / canonicalize** | 优化趟，触发 fold | 命令行 `--canonicalize` |
| **Pass / toy-simplify** | 我们自己的 Pass，触发 RewritePattern | 命令行 `--toy-simplify` |
| **MlirOptMain** | `toy-opt` 复用的通用主循环 | `tools/toy-opt/toy-opt.cpp` |
| **.mlir 文本** | IR 的人类可读序列化形式 | `test/*.mlir` |
| **lit + FileCheck** | 自动化测试：跑命令 + 断言输出 | `test/lit*.py`、`// CHECK:` |

---

## 五、关键命令清单

```bat
:: 1. 配置（告诉 CMake 去哪找已编译好的 MLIR）
cmake -G Ninja -S . -B build ^
  -DMLIR_DIR=C:\dev\llvm-project\build\lib\cmake\mlir ^
  -DLLVM_DIR=C:\dev\llvm-project\build\lib\cmake\llvm

:: 2. 编译出 toy-opt.exe
cmake --build build

:: 3. 运行：解析回显
build\bin\toy-opt.exe test\ops.mlir

:: 4. 运行：常量折叠（fold 机制，看 toy.add 变常量）
build\bin\toy-opt.exe test\canonicalize.mlir --canonicalize

:: 5. 运行：代数化简（RewritePattern 机制，看 x*1、x+0 消失）
build\bin\toy-opt.exe test\simplify.mlir --toy-simplify

:: 6. 自动化测试（lit 一键跑全部用例）
cmake --build build --target check-toy
```

> Linux + conda 环境下更省事，直接用脚本（见 README「三、Linux + conda 一键启动」）：
> `bash scripts/all.sh`（装依赖→构建→测试→演示），或分步 `build.sh` / `test.sh` / `run.sh`。

---

## 六、一句话收尾

**MLIR 的本质 = 用 TableGen 声明 IR 形状 + 用手写 C++ 填 IR 行为 + 用 Pass 遍历改写 IR。**
`*.td` 里的"声明"和 `*.cpp` 里的"实现"这两半合起来，才是完整的 Op。这个 `toy` 项目虽小，但已经把 MLIR 的全部主干机制走了一遍。
