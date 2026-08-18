# 检验体系 02｜MLIR

> **对应学习文档**：[`../learning-guides/mlir-learning-guide.md`](../learning-guides/mlir-learning-guide.md)、[`../paper-notes/03-mlir.md`](../paper-notes/03-mlir.md)、[`../notes/llvm-mlir-pass-ir-unit.md`](../notes/llvm-mlir-pass-ir-unit.md)  
> **对应动手项目**：[`mlir-toy-dialect/`](../../mlir-toy-dialect/)  
> **分级与资源标签定义**：[`./README.md`](./README.md)

本册检验的是这些知识点能不能落成代码：**ODS/TableGen 声明式定义 op、fold 与 canonicalize、Trait 与 Interface、Pass 锚定在哪个 Op 上、贪心改写 vs Dialect Conversion、TypeConverter 与 materialization、lit/FileCheck 测试**。

**入门线**：L0 两条全做 + L1 至少两条 + **L2-MLIR-06 与 L2-MLIR-07 必做**（「新增一个 op 并把它一路降下去」是 MLIR 的核心工作流，做完这两条你就能读懂真实 dialect 的代码结构了）。

**资源总览**：本册**全部条目都不需要 GPU**，标签统一为 `本地+工具链`（需 conda 环境 `mlir-env`，内含 LLVM/MLIR 17）。

```bash
# 开工前自查
cd mlir-toy-dialect
bash scripts/setup.sh          # 幂等，已建过会跳过
source scripts/env.sh && echo "$MLIR_DIR"
```

> **与 LLVM 分册的对照建议**：做完 [`./01-llvm.md`](./01-llvm.md) 的 L2-LLVM-07（写一个 LLVM Pass）再来做本册的 L2-MLIR-09（写一个 MLIR Pass），两边并排看，「固定层级 vs 任意 Op 锚定」的差别一次就懂了。

---

## L0 复现

### L0-MLIR-01｜跑通全流程，九组演示各能说出一句话

- **检验什么**：这条通过 = 你知道这个 dialect 里有哪些 op、哪些 pass、以及每个演示在验证什么
- **前置**：无
- **资源**：本地+工具链
- **预计耗时**：1.5h

**任务**：跑 `scripts/all.sh`（setup → build → test → run）。然后对着 `out/` 里九组产物，给每组写一句话：**这一组证明了什么**。特别注意演示 5-a 与 5-b（同一份 `strength.mlir` 在 toy 层与 low 层分别优化）说明了什么——这是「progressive lowering 中每层解决自己该解决的问题」的直接证据。

**验收命令**：

```bash
cd mlir-toy-dialect
bash scripts/all.sh
# 或分步
bash scripts/build.sh && bash scripts/test.sh && bash scripts/run.sh
```

**通过标准**：`check-toy` 全部用例 PASS；`build/bin/toy-opt` 存在；你能说出 `--toy-to-low` 与 `--toy-to-low-convert` 做的是同一件事但用了不同框架。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| `find_package(MLIR)` 失败 | 没 `source scripts/env.sh`，或 conda 环境未建好 |
| lit 找不到 | `-DLLVM_EXTERNAL_LIT` 没传对；不了解 lit 是外部 Python 工具而非 CMake 内置 |

---

### L0-MLIR-02｜找到 TableGen 生成的 C++，看清 ODS 到底替你写了什么

- **检验什么**：这条通过 = 你不再把 `.td` 当黑魔法，知道 `assemblyFormat` 一行换来了多少手写代码
- **前置**：L0-MLIR-01
- **资源**：本地+工具链
- **预计耗时**：1h

**任务**：在 `build/` 里找到 `ToyOps.h.inc` / `ToyOps.cpp.inc`，定位三样东西：

1. `MulOp` 的 `getLhs()` / `getRhs()` 访问器是怎么生成的
2. `assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)"` 生成出的 `parse()` 与 `print()` 函数体
3. `Toy_MulOp` 的 `extraClassDeclaration` 拷进 `.inc` 的 `getCost()` **声明**（注意只有声明，定义在 `lib/ToyOps.cpp` 里手写）

**验收命令**：

```bash
cd mlir-toy-dialect
find build -name 'ToyOps*.inc'
# 也可以直接看 tblgen 的输出（不落盘）
mlir-tblgen -gen-op-defs -I include -I "$MLIR_DIR/../../../include" include/Toy/ToyOps.td | head -80
```

**通过标准**：你能指出 `.inc` 里哪一段对应 `.td` 里的哪一行；能解释为什么 `Toy_AddOp` 没有 `getCost()` 声明而 `Toy_MulOp` 有（前者用接口默认实现，后者靠 `extraClassDeclaration` 覆盖）。

**常见失败 → 说明你哪里没懂**：找不到 `.inc` → 没意识到它是构建期产物，`clean` 之后要重新 build。

---

## L1 改一处

### L1-MLIR-03｜给 toy.mul 加代数化简：x*1 与 x*0

- **检验什么**：这条通过 = 你掌握了 `fold()` 的返回值语义（`OpFoldResult` = Attribute 或 Value）以及它与 `--canonicalize` 的关系
- **前置**：L0-MLIR-02
- **资源**：本地+工具链
- **预计耗时**：1.5h

**任务**：`ToyOps.td` 第 96 行的注释已经把题目写好了。在 `lib/ToyOps.cpp` 的 `MulOp::fold()` 里加两条规则：

- `x * 1 → x`（返回一个**已有的 Value**）
- `x * 0 → 0`（返回一个**新的 Attribute**）

然后在 `test/canonicalize.mlir` 加用例与 `CHECK` / `CHECK-NOT`。

**先预测再动手**：

1. 返回 Value 与返回 Attribute，框架后续的处理有什么不同？（提示：返回 Attribute 时，谁负责把它变成一个真正的 `toy.constant`？`ConstantLike` trait 和 dialect 的 `materializeConstant` 在这里起什么作用？）
2. `toy.mul` 有 `Commutative` trait。如果用户写的是 `toy.mul %one, %x`（常量在左边），你的 fold 还能命中吗？框架会不会先帮你把常量挪到右边？
3. 不加 `--canonicalize` 直接跑 `toy-opt`，fold 会生效吗？

**验收命令**：

```bash
cd mlir-toy-dialect
bash scripts/build.sh
cat > /tmp/mulfold.mlir <<'EOF'
func.func @f(%x: i32) -> (i32, i32) {
  %one  = toy.constant 1 : i32
  %zero = toy.constant 0 : i32
  %a = toy.mul %x, %one  : i32
  %b = toy.mul %x, %zero : i32
  return %a, %b : i32, i32
}
EOF
./build/bin/toy-opt /tmp/mulfold.mlir --canonicalize
bash scripts/test.sh
```

**通过标准**：输出中 `%a` 直接变成 `%x`（`toy.mul` 消失），`%b` 变成 `toy.constant 0`；`check-toy` 全绿。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| fold 写了但不生效 | 没跑 `--canonicalize`，或 `hasFolder` 忘了保持为 1 |
| 返回 Attribute 后 IR 里出现空洞/崩溃 | 不理解 Attribute 需要被 materialize 成常量 op，dialect 必须提供 `materializeConstant` |
| 常量在左边时不命中 | 没意识到 `Commutative` 只是**允许**优化器交换，规范化顺序由 canonicalizer 决定，不能想当然 |

---

### L1-MLIR-04｜删掉一条 lowering 规则，对比两种驱动器的失败方式

- **检验什么**：这条通过 = 你彻底理解「贪心改写描述过程，Dialect Conversion 描述终点」这句话的工程后果
- **前置**：L0-MLIR-01
- **资源**：本地+工具链
- **预计耗时**：0.5h

**任务**：`ConvertToyToLow.cpp` 第 230 行的注释就是题目。把 `patterns.add<...>` 里的 `MulOpLowering` 删掉，重新编译，对同一份 `test/convert.mlir` 分别跑 `--toy-to-low`（贪心版）与 `--toy-to-low-convert`（Conversion 版），对比两者的失败表现。

**先预测再动手**：

1. 贪心版会报错吗？如果不报错，输出的 IR 里会剩下什么？这种「静默残留」在真实编译器里会在哪一步才暴露？
2. Conversion 版的报错信息里会出现哪个 op 的名字？它是怎么知道「这个 op 不合法」的？（回看 `ConversionTarget` 那几行）

**验收命令**：

```bash
cd mlir-toy-dialect
# 注释掉 MulOpLowering 后重新构建
bash scripts/build.sh
./build/bin/toy-opt test/convert.mlir --toy-to-low          # 观察：静默留下 toy.mul
./build/bin/toy-opt test/convert.mlir --toy-to-low-convert  # 观察：failed to legalize
```

**通过标准**：贪心版输出的 IR 中仍含 `toy.mul` 且退出码为 0；Conversion 版报 `error: failed to legalize operation 'toy.mul'` 且退出码非 0。恢复代码后 `check-toy` 重新全绿。

**常见失败 → 说明你哪里没懂**：认为两者「都会报错，只是信息不同」→ 没抓住核心区别：**贪心版根本不知道什么叫「做完了」**。

---

### L1-MLIR-05｜给 toy.repeat 加上 IsolatedFromAbove，看约束如何反噬

- **检验什么**：这条通过 = 你理解 Trait 不是标签而是**框架会强制验证的结构契约**，以及 Region 捕获外层值的意义
- **前置**：L0-MLIR-01
- **资源**：本地+工具链
- **预计耗时**：0.5h

**任务**：在 `ToyOps.td` 给 `Toy_RepeatOp` 的 trait 列表加上 `IsolatedFromAbove`，重新编译，跑 `test/region.mlir`。

**先预测再动手**：`test/region.mlir` 的 region 内部直接引用了外层的 `%arg0`。加上 `IsolatedFromAbove` 后，这会在**解析期**报错、**验证期**报错，还是要等到某个 pass 运行时才报错？报错文案大概会说什么？

**验收命令**：

```bash
cd mlir-toy-dialect
bash scripts/build.sh
./build/bin/toy-opt test/region.mlir --toy-simplify
```

**通过标准**：出现明确的验证错误（提示区域内使用了定义在区域外的值）。能说清 `func.func` 和 GPU kernel 为什么需要这个 trait，而 `scf.for` 为什么不需要。做完恢复 `.td`。

**常见失败 → 说明你哪里没懂**：改完 `.td` 没重新构建就跑 → 忘了 `.td` 是**编译期**输入，改了必须重新 tblgen + 编译。

---

## L2 加组件（主判据）

### L2-MLIR-06｜从零新增一个 op：toy.sub

- **检验什么**：这条通过 = 你掌握了新增 op 的完整链路：ODS 定义 → assemblyFormat → fold → verifier → lit 测试
- **前置**：L1-MLIR-03
- **资源**：本地+工具链
- **预计耗时**：2.5h

**任务**：新增 `toy.sub`，要求覆盖到这几个知识点，缺一不可：

1. **ODS 定义**（`include/Toy/ToyOps.td`）：两个 `I32` 操作数、一个 `I32` 结果、`Pure` trait。**注意不要加 `Commutative`**——减法不可交换，加了就是语义错误，这一点要在你的提交说明里点出来。
2. **接口**：挂上 `ToyCostOpInterface`（用默认代价 1，即不加 `DeclareOpInterfaceMethods`）。
3. **assemblyFormat**：写成 `toy.sub %a, %b : i32` 的形式。
4. **fold**（`lib/ToyOps.cpp`）：常量折叠 + `x - x → 0` + `x - 0 → x`。
5. **测试**：新建 `test/sub.mlir`，含解析回显、canonicalize 折叠、以及一条 `CHECK-NOT` 守住不该发生的化简。

**先预测再动手**：

1. 如果你手滑给 `toy.sub` 加了 `Commutative`，什么时候会出问题？canonicalizer 会怎么坑你？（提示：它可能把 `%a - %b` 的操作数顺序换掉）
2. `x - x → 0` 这条规则，在操作数是**同一个 Value** 与**两个不同但值相等的 Value** 时，分别能不能命中？fold 看的是什么？
3. 新增 op 后，`--toy-to-low` 和 `--toy-to-low-convert` 会怎么对待它？（这是下一条 L2-MLIR-07 的引子）

**验收命令**：

```bash
cd mlir-toy-dialect
bash scripts/build.sh
cat > /tmp/sub.mlir <<'EOF'
func.func @f(%x: i32) -> (i32, i32, i32) {
  %c7 = toy.constant 7 : i32
  %c3 = toy.constant 3 : i32
  %a = toy.sub %c7, %c3 : i32     // 应折叠成 toy.constant 4
  %b = toy.sub %x, %x   : i32     // 应折叠成 toy.constant 0
  %c = toy.sub %x, %c3  : i32     // 不应被折叠
  return %a, %b, %c : i32, i32, i32
}
EOF
./build/bin/toy-opt /tmp/sub.mlir --canonicalize
./build/bin/toy-opt /tmp/sub.mlir --toy-print-cost
bash scripts/test.sh
```

**通过标准**：三条折叠行为符合预期；`--toy-print-cost` 能统计到 `toy.sub` 的代价（证明接口挂对了）；`test/sub.mlir` 被 `check-toy` 收录并 PASS。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 解析报 `expected ':'` 之类 | assemblyFormat 与你写的测试文本不一致；不理解格式串**同时**决定了 parser 和 printer |
| `toy-opt` 说 unregistered op | 新 op 加在了 `.td` 但没重新 build，或 dialect 的 `initialize()` 未包含（本项目由 tblgen 的 `GET_OP_LIST` 自动处理，需确认 `.inc` 已重新生成） |
| `--toy-print-cost` 统计不到 | 没挂接口；说明还没建立「Interface 是 pass 与 op 之间的契约」的认识 |
| fold 时 `x - x` 不生效 | 用了值比较而非 Value 身份比较，或没处理 `getLhs() == getRhs()` |

---

### L2-MLIR-07｜把 toy.sub 一路降下去：贪心版与 Conversion 版各写一遍

- **检验什么**：这条通过 = 你掌握了 `OpRewritePattern` 与 `OpConversionPattern` 的实际差别，特别是 `adaptor` 的存在意义
- **前置**：L2-MLIR-06
- **资源**：本地+工具链
- **预计耗时**：2.5h

**任务**：

1. 在 `include/Low/LowOps.td` 新增 `low.sub`（与 `low.add` 平行，挂 `ToyCostOpInterface`）。
2. 在 `lib/LowPasses.cpp`（贪心版 `--toy-to-low`）加一条 `OpRewritePattern<toy::SubOp>`。
3. 在 `lib/ConvertToyToLow.cpp`（Conversion 版）加一条 `OpConversionPattern<toy::SubOp>`，**操作数必须用 `adaptor.getLhs()` / `adaptor.getRhs()`**。
4. 在 `test/lowering.mlir` 与 `test/convert.mlir` 补 `CHECK`。

**先预测再动手**：

1. 两条 pattern 的函数签名差在哪一个参数上？那个参数是干什么的？
2. 如果你在 Conversion 版里**写成 `op.getLhs()`**，会发生什么？（先预测，下一条 L2-MLIR-08 会让你真的去踩）
3. 只加了 Conversion 版而忘了加贪心版，`--toy-to-low` 会报错还是静默残留？（回看 L1-MLIR-04 的结论）

**验收命令**：

```bash
cd mlir-toy-dialect
bash scripts/build.sh
./build/bin/toy-opt /tmp/sub.mlir --toy-to-low
./build/bin/toy-opt /tmp/sub.mlir --toy-to-low-convert
bash scripts/test.sh
```

**通过标准**：两个 pass 都能把 `toy.sub` 变成 `low.sub`，且 Conversion 版输出里**没有残留** `builtin.unrealized_conversion_cast`；`check-toy` 全绿。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| Conversion 版报 `failed to legalize 'toy.sub'` | pattern 没注册进 `patterns.add<...>`，或 target 没把 `low.sub` 标为合法 |
| 输出里剩下 `unrealized_conversion_cast` | 类型转换没做干净——这是框架给你的明确信号，别忽略它 |
| 贪心版把 op 换了但类型对不上 | 印证了「贪心驱动器不能换类型」这条限制 |

---

### L2-MLIR-08｜故意用 op 代替 adaptor，让 unrealized_conversion_cast 现形

- **检验什么**：这条通过 = 你真正理解 Dialect Conversion 期间「旧值/新值并存」的中间状态，以及 materialization 的作用
- **前置**：L2-MLIR-07
- **资源**：本地+工具链
- **预计耗时**：1h

**任务**：把 `ConvertToyToLow.cpp` 里某条 pattern（建议用 `AddOpLowering` 或你新写的 `SubOpLowering`）的 `adaptor.getLhs()` 改成 `op.getLhs()`，重新编译，对一份**含 `toy.box` / `toy.unbox`（即涉及 `!toy.num` 类型转换）**的输入跑转换，观察输出。

**先预测再动手**：

1. 转换过程中，`%x` 的类型已经从 `!toy.num` 变成了 `i32`。此时 `op.getLhs()` 拿到的是哪一个值——旧的还是新的？
2. 框架发现类型对不上时，会直接报错，还是先插一个「转接头」？那个转接头叫什么名字？
3. 如果最终输出里还留着转接头，说明了什么？

**验收命令**：

```bash
cd mlir-toy-dialect
bash scripts/build.sh
./build/bin/toy-opt test/convert.mlir --toy-to-low-convert 2>&1 | grep -E 'unrealized_conversion_cast|error'
```

**通过标准**：能观察到 `builtin.unrealized_conversion_cast` 残留或合法化失败；改回 `adaptor` 后现象消失。你能用一句话说清 adaptor 存在的理由。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 什么都没变 | 你选的那条 pattern 不涉及类型转换（比如纯 i32 的路径），换一条涉及 `!toy.num` 的再试 |
| 看到 cast 就以为是 bug 输出 | 转换**过程中**出现 cast 是正常的，关键在于**结束后**还剩不剩 |

---

### L2-MLIR-09｜写一个新 Pass，并亲手验证「Pass 锚定在哪个 Op 上」

- **检验什么**：这条通过 = 你掌握了 MLIR 与 LLVM 最本质的机制差异：**Pass 的作用域不是固定的 Module/Function，而是任意 Op 类型**
- **前置**：L0-MLIR-02
- **资源**：本地+工具链
- **预计耗时**：2h

**任务**：新增一个 pass，命令行名 `--toy-op-histogram`，统计并打印 IR 里每种 op 出现的次数。**要写两个版本**：

- 版本 A：`PassWrapper<..., OperationPass<>>`（锚在任意 op 上，与项目里现有 pass 一致）
- 版本 B：`PassWrapper<..., OperationPass<func::FuncOp>>`（只锚在 `func.func` 上），命令行名 `--toy-op-histogram-func`

实现里用 `getOperation()->walk([&](Operation *op) { ... })` 遍历。注册到 `lib/ToyPasses.cpp` 的 `registerToyPasses()`。

**先预测再动手**：

1. 对同一份含两个函数的 `.mlir`，版本 A 打印几份直方图？版本 B 呢？
2. 版本 B 用 `--pass-pipeline='builtin.module(func.func(toy-op-histogram-func))'` 显式指定管线时，与直接写 `--toy-op-histogram-func` 有区别吗？
3. 如果你把版本 B 用在一个**顶层不是 module** 的输入上会怎样？
4. 对照 LLVM：`OperationPass<func::FuncOp>` 相当于 LLVM 的什么？`OperationPass<>` 在 LLVM 里有对应物吗？

**验收命令**：

```bash
cd mlir-toy-dialect
bash scripts/build.sh
cat > /tmp/two_funcs.mlir <<'EOF'
func.func @f(%x: i32) -> i32 {
  %c = toy.constant 2 : i32
  %m = toy.mul %x, %c : i32
  return %m : i32
}
func.func @g(%x: i32) -> i32 {
  %c = toy.constant 3 : i32
  %a = toy.add %x, %c : i32
  return %a : i32
}
EOF
./build/bin/toy-opt /tmp/two_funcs.mlir --toy-op-histogram
./build/bin/toy-opt /tmp/two_funcs.mlir --toy-op-histogram-func
./build/bin/toy-opt /tmp/two_funcs.mlir \
  --pass-pipeline='builtin.module(func.func(toy-op-histogram-func))'
```

**通过标准**：版本 A 打印一份全局直方图；版本 B 打印**两份**（每个函数一份），且 `walk` 只看到本函数内的 op。你能解释这个差异完全来自 `OperationPass<T>` 的模板参数。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 版本 B 编译报错找不到 `func::FuncOp` | 少 include `mlir/Dialect/Func/IR/FuncOps.h`；也说明不清楚 `func` 只是一个普通 dialect |
| 版本 B 也只打印一份 | 模板参数写成了 `OperationPass<>`；没抓住锚定机制 |
| `--pass-pipeline` 写法报错 | 不熟悉管线字符串的嵌套语法，这正是 MLIR 表达「pass 层级」的方式 |

---

### L2-MLIR-10｜改一个接口实现，让代价模型给出不同答案

- **检验什么**：这条通过 = 你理解 Interface 如何让一个**完全不认识具体 dialect** 的 pass 也能工作
- **前置**：L0-MLIR-01
- **资源**：本地+工具链
- **预计耗时**：1h

**任务**：`low.shl` 目前直接挂 `ToyCostOpInterface`（用默认代价 1）。给它加上与 `low.mul` 一样的 `extraClassDeclaration { unsigned getCost(); }`，在 `lib/LowOps.cpp` 实现 `getCost()` 返回 0（移位视为免费），然后跑演示 9-a / 9-b（`test/cost.mlir` 上强度削减前后的代价对比）。改完 `.td` 后若增量构建没重跑 tblgen，用 `bash scripts/build.sh clean`。

**先预测再动手**：

1. `--toy-print-cost` 这个 pass 的代码里有没有出现 `low::ShlOp` 这个类型？它是怎么统计到 shl 的代价的？
2. 改完之后，强度削减（`mul → shl`）带来的代价下降会变大还是变小？变多少？
3. `toy.constant` 刻意没有实现这个接口。`--toy-print-cost` 遇到它会崩溃、报错，还是跳过？为什么？

**验收命令**：

```bash
cd mlir-toy-dialect
bash scripts/build.sh
./build/bin/toy-opt test/cost.mlir --toy-print-cost
./build/bin/toy-opt test/cost.mlir --toy-to-low --low-strength-reduce --toy-print-cost
bash scripts/test.sh    # test/cost.mlir 有 BEFORE/AFTER 两组 CHECK，需同步更新
```

**通过标准**：强度削减后的总代价数值按你的预测变化；`test/cost.mlir` 的 `AFTER` CHECK 更新后全绿。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 代价没变 | 只改了 `.td` 没写 C++ 定义（或反之），链接期/运行期用的还是默认实现 |
| 编译报 `no declaration matches ... getCost()` | `.inc` 里没有声明：要么漏了 `extraClassDeclaration`，要么改了 `.td` 但没重跑 tblgen（`bash scripts/build.sh clean`） |
| 链接报未定义符号 `getCost` | `DeclareOpInterfaceMethods` 只生成**声明**，定义必须手写——这正是 L0-MLIR-02 要你看清的东西 |

---

## L3 打通

### L3-MLIR-11｜给 !toy.num 加参数，做一次真正的类型转换

- **检验什么**：这条通过 = 你掌握了参数化类型（parametric type）的定义，以及 TypeConverter 在类型带参数时的写法
- **前置**：L2-MLIR-07
- **资源**：本地+工具链
- **预计耗时**：半天

**任务**：把 `!toy.num` 从无参类型改成带位宽参数的 `!toy.num<32>` / `!toy.num<64>`：

1. `include/Toy/ToyTypes.td`：加 `parameters = (ins "unsigned":$width)`，`assemblyFormat` 改成 `` `<` $width `>` ``。
2. `ConvertToyToLow.cpp` 的 `populateToyTypeConversions`：把 `!toy.num<W>` 映射到 `IntegerType::get(ctx, W)`。
3. 更新 `test/types.mlir` 与 `test/convert.mlir`。

**先预测再动手**：

1. 类型带参数后，`toy.box` 的 ODS 约束 `Toy_NumType:$result` 还能用吗？它现在匹配任意 width 还是某个特定 width？
2. `toy.box %x : i32 -> !toy.num<64>` 应该合法吗？如果不该，这个约束写在 ODS 里还是 verifier 里？
3. TypeConverter 的兜底规则（`addConversion([](Type t){ return t; })`）为什么必须写在最前面？

**验收命令**：

```bash
cd mlir-toy-dialect
bash scripts/build.sh
cat > /tmp/ptype.mlir <<'EOF'
func.func @f(%x: i32) -> i32 {
  %n = toy.box %x : i32 -> !toy.num<32>
  %y = toy.unbox %n : !toy.num<32> -> i32
  return %y : i32
}
EOF
./build/bin/toy-opt /tmp/ptype.mlir
./build/bin/toy-opt /tmp/ptype.mlir --toy-to-low-convert
bash scripts/test.sh
```

**通过标准**：参数化类型能正确解析/打印；转换后 `!toy.num<32>` 变成 `i32` 且无 cast 残留；宽度不匹配时有明确报错。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 类型解析报错 | `assemblyFormat` 与文本不符；参数化类型的打印格式需要显式声明 |
| 转换后剩下 cast | TypeConverter 的 lambda 参数类型写死成了无参版本，没取到 width |

---

### L3-MLIR-12｜端到端：把 toy 一路降到 LLVM IR 并真正跑起来

- **检验什么**：这条通过 = 你走完了「自定义 dialect → 内置 dialect → LLVM dialect → LLVM IR → 可执行」的完整 progressive lowering 链路
- **前置**：L2-MLIR-07
- **资源**：本地+工具链
- **预计耗时**：半天

**任务**：新增一个 pass `--low-to-arith`，把 `low.constant/add/mul/shl` 降到内置的 `arith` dialect（`arith.constant` / `arith.addi` / `arith.muli` / `arith.shli`），然后用 MLIR 自带的转换管线一路降到 `llvm` dialect，再用 `mlir-translate` 出 `.ll`，最后用 `lli` 或 `clang` 跑出结果。

**先预测再动手**：

1. 为什么要先降到 `arith` 而不是直接降到 `llvm` dialect？（提示：复用生态——`arith` 到 `llvm` 的转换 MLIR 已经写好了）
2. `func.func` 需要被转换吗？函数签名的类型转换由谁负责？
3. `mlir-translate --mlir-to-llvmir` 要求输入 IR 处于什么状态？如果还剩一个 `low.add` 会怎样？

**验收命令**：

```bash
cd mlir-toy-dialect
bash scripts/build.sh
# 具体 pass 名以本地 MLIR 版本为准，先自查：
mlir-opt --help | grep -E 'convert-arith-to-llvm|convert-func-to-llvm|reconcile-unrealized'

./build/bin/toy-opt /tmp/sub.mlir --toy-to-low-convert --low-to-arith -o /tmp/arith.mlir
mlir-opt /tmp/arith.mlir \
  --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts \
  -o /tmp/llvmdialect.mlir
mlir-translate /tmp/llvmdialect.mlir --mlir-to-llvmir -o /tmp/out.ll
lli /tmp/out.ll ; echo "exit=$?"
```

**通过标准**：`/tmp/out.ll` 是合法 LLVM IR（`opt -passes=verify` 通过），且执行结果与你在 toy 层用 `--canonicalize` 折叠出的常量一致。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| `mlir-translate` 报还有非 llvm dialect 的 op | 降低没做干净；`--reconcile-unrealized-casts` 漏了 |
| pass 名不存在 | MLIR 版本差异，以本地 `mlir-opt --help` 为准（17 与更新版本的 flag 名有出入） |
| 函数签名类型没转 | 只转了 body 没转 signature，说明还不理解 `TypeConverter` 需要配合 func 转换 pattern |

---

## 条目 × 资源需求速查

| 编号 | 任务 | 级别 | 资源 | 耗时 |
|------|------|------|------|------|
| L0-MLIR-01 | 跑通全流程，九组演示 | L0 | 本地+工具链 | 1.5h |
| L0-MLIR-02 | 读 TableGen 生成的 C++ | L0 | 本地+工具链 | 1h |
| L1-MLIR-03 | toy.mul 加 x*1 / x*0 折叠 | L1 | 本地+工具链 | 1.5h |
| L1-MLIR-04 | 删 lowering 规则对比两种驱动器 | L1 | 本地+工具链 | 0.5h |
| L1-MLIR-05 | 加 IsolatedFromAbove 看约束反噬 | L1 | 本地+工具链 | 0.5h |
| **L2-MLIR-06** | **从零新增 toy.sub** | L2 | 本地+工具链 | 2.5h |
| **L2-MLIR-07** | **两种 pattern 各写一遍 lowering** | L2 | 本地+工具链 | 2.5h |
| L2-MLIR-08 | 用 op 代替 adaptor 踩坑 | L2 | 本地+工具链 | 1h |
| L2-MLIR-09 | 写新 Pass 并验证锚定 Op | L2 | 本地+工具链 | 2h |
| L2-MLIR-10 | 改接口实现看代价模型 | L2 | 本地+工具链 | 1h |
| L3-MLIR-11 | 参数化类型 + TypeConverter | L3 | 本地+工具链 | 半天 |
| L3-MLIR-12 | 端到端降到 LLVM IR 并运行 | L3 | 本地+工具链 | 半天 |

**本册无任何条目需要 GPU 或集群资源。** 全部可在个人开发机完成，总计约 22 小时。

**下一步**：本册练的是「自己造一个 dialect」；[`./03-iree.md`](./03-iree.md) 练的是「读懂一个真实的、由几十个 dialect 组成的编译器把模型逐层降下去」。先做完本册的 L2，再看 IREE 的相位 dump 会顺畅很多。
