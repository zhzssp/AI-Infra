# 检验体系 01｜LLVM

> **对应学习文档**：[`../llvm-learning-guide.md`](../llvm-learning-guide.md)、[`../notes/llvm-mlir-pass-ir-unit.md`](../notes/llvm-mlir-pass-ir-unit.md)  
> **对应动手项目**：[`llvm-hello-compile/`](../../llvm-hello-compile/)  
> **分级与资源标签定义**：[`./README.md`](./README.md)

本册检验的是这些知识点能不能落成代码：**New PM 的层级与注册机制、AnalysisManager 的按需索取与失效、变换 Pass 的正确性边界（poison/nsw/有符号语义）、FileCheck 测试写法、IR→汇编的后端接缝**。

**入门线**：L0 两条全做 + L1 至少两条 + **L2-LLVM-06 与 L2-LLVM-07 必做**（这两条是「能不能自己加 Pass」的分水岭）。

**资源总览**：本册**全部条目都不需要 GPU**。唯一要求是本地有 LLVM 工具链（`clang` / `opt` / `llc` / `llvm-config` / `FileCheck`），即标签 `本地+工具链`。

```bash
# 开工前自查：这三个必须有，FileCheck 缺了只影响测试条目
which clang opt llc llvm-config FileCheck
llvm-config --version
```

---

## L0 复现

### L0-LLVM-01｜跑通主链路，指认 mem2reg 干了什么

- **检验什么**：这条通过 = 你真的掌握了「C → IR → 优化 → 汇编 → 可执行」每一步的产物形态，以及 SSA 构造（alloca/load/store → phi）
- **前置**：无
- **资源**：本地+工具链
- **预计耗时**：1h

**任务**：跑 `scripts/run.sh`，然后**不看解释**，只对着 `out/steps/diff_02_to_03a.txt` 回答：mem2reg 消掉了哪几类指令、新增了什么、`%retval` 这类槽位去哪了。再用 `out/08_summary` 里的指令计数印证。

**验收命令**：

```bash
cd llvm-hello-compile
bash scripts/run.sh
# 重点读这三个产物
#   out/02_sum_O0.ll          alloca/load/store 满天飞
#   out/03a_sum_mem2reg.ll    alloca 消失，phi 出现
#   out/steps/diff_02_to_03a.txt
```

**通过标准**：

- `out/06_sum` 运行输出 `55`
- 你能在 `03a_sum_mem2reg.ll` 里**手指出至少一条 `phi`**，并说清它的每个 incoming 值来自哪个前驱块
- 你能解释为什么 `-O0` 编译时要加 `-disable-O0-optnone`（否则后续 `opt` 的 pass 会被 `optnone` 属性挡掉）

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 找不到 phi，`03a` 里还是 alloca | 没加 `-disable-O0-optnone`，或读错了文件；说明不清楚 `optnone` 如何阻断 pass |
| 说不清 phi 的 incoming 从哪来 | 没建立「基本块 + 前驱边」的心智模型，回看 llvm 指南的 SSA 示例 |

---

### L0-LLVM-02｜巡礼 17 站 + 测试全绿

- **检验什么**：这条通过 = 你知道这个项目里**有哪些可动的零件**（5 个自定义 Pass、别名分析、TTI、MIR、MC、TableGen），后面才知道该改哪
- **前置**：L0-LLVM-01
- **资源**：本地+工具链
- **预计耗时**：1.5h

**任务**：跑 `tour.sh` 与 `run_tests.sh`。读 `out/tour/TOUR.md`，把 17 站中**你完全看不懂的站点**列出来（这份清单就是你的学习待办）。

**验收命令**：

```bash
cd llvm-hello-compile
bash scripts/build_passes.sh     # 产出 build/passes/MyPasses.so
bash scripts/tour.sh
bash scripts/run_tests.sh        # 期望 2 个测试全 PASS
```

**通过标准**：`run_tests.sh` 输出 `strength-reduce.ll` 与 `cfg-info.ll` 均 PASS；`build/passes/MyPasses.so` 存在。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| `find_package(LLVM)` 失败 | 不清楚插件是靠 `llvm-config --cmakedir` 找到 LLVM 的 CMake 包 |
| 测试跳过（skip） | 环境缺 `FileCheck`；`run_tests.sh` 会跳过而不是报错，注意别把 skip 当 pass |

---

## L1 改一处

### L1-LLVM-03｜让 inject-log 打印出参数值

- **检验什么**：这条通过 = 你掌握了「用 IRBuilder 在指定插入点构造指令、声明外部函数、处理类型匹配」这套 IR 构造基本功
- **前置**：L0-LLVM-02
- **资源**：本地+工具链
- **预计耗时**：1.5h

**任务**：改 `passes/InjectLogging.cpp`。当前它在每个函数入口插一句无参 `printf`；改成打印**函数名 + 第一个整型参数的运行时值**（形如 `[enter] sum(n=10)`）。要点：格式字符串要用 `IRBuilder::CreateGlobalStringPtr` 造，`printf` 是变参函数，声明时 `FunctionType::get(..., /*isVarArg=*/true)`。

**先预测再动手**：动手前写下你的答案——

1. 如果函数的第一个参数是指针或浮点，直接塞进 `%d` 的 `printf` 会发生什么？你打算怎么跳过这类函数？
2. 插入点选在 `F.getEntryBlock().getFirstInsertionPt()` 而不是 `begin()`，差别是什么？（提示：entry 块开头可能有一串 alloca / `llvm.dbg` 之类）

**验收命令**：

```bash
cd llvm-hello-compile
bash scripts/build_passes.sh --force
opt -load-pass-plugin=build/passes/MyPasses.so -passes=inject-log \
    -S out/03b_sum_O2.ll -o out/logged.ll
llc -filetype=obj out/logged.ll -o out/logged.o && clang -no-pie out/logged.o -o out/logged
./out/logged
```

**通过标准**：可执行文件运行时打印出带**实际参数值**的日志行（不是格式串本身），且原有计算结果 `55` 不变。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 链接期报 `printf` 类型冲突 | 没意识到模块里可能已存在 `printf` 声明，应先 `M.getOrInsertFunction` 而不是新建 |
| 运行崩溃 / 打印乱码 | 参数类型没检查就传；说明对「IR 是强类型的，变参调用不做隐式转换」没概念 |
| 插入的指令跑到了 alloca 前面导致 verifier 报错 | 不理解 entry 块的结构约定 |

---

### L1-LLVM-04｜删掉 restrict，看别名分析与向量化一起塌方

- **检验什么**：这条通过 = 你真的理解「别名信息是优化的前提，`noalias` 不是装饰」
- **前置**：L0-LLVM-02
- **资源**：本地+工具链
- **预计耗时**：1h

**任务**：把 `src/kernel.c` 里指针参数的 `restrict` 去掉，重跑 `tour.sh`，对比第 9 站（别名分析）与第 11 站（向量化 remark）的输出差异。

**先预测再动手**：

1. `restrict` 在 IR 层面变成了什么？（找到那个属性名）
2. 去掉之后，编译器还能不能证明两次访存不重叠？它会保守到什么程度——完全放弃向量化，还是插入运行时别名检查？

**验收命令**：

```bash
cd llvm-hello-compile
cp src/kernel.c /tmp/kernel.c.bak
# 编辑 src/kernel.c 删掉 restrict
bash scripts/tour.sh
# 对比 out/tour/ 下别名分析与向量化 remark 的产物
```

**通过标准**：能指出 IR 中 `noalias` 属性的消失，并从 remark 中找到向量化决策发生变化的证据（未向量化，或改为带运行时检查的版本）。做完记得还原源文件。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 输出「看起来没差」 | 没找对产物文件，或函数太简单未触发向量化；应先确认基线版本确实向量化了 |
| 说不出运行时别名检查 | 只知道「不能向量化」，不知道编译器还有「猜 + 运行时验证」这条路 |

---

### L1-LLVM-05｜把 strength-reduce 放进不同的 pass 顺序里

- **检验什么**：这条通过 = 你掌握了「pass 顺序敏感 / phase ordering」，以及自定义 pass 如何与内置 pass 组合
- **前置**：L0-LLVM-02
- **资源**：本地+工具链
- **预计耗时**：0.5h

**任务**：对同一份 `.ll` 分别跑三种管线，对比结果：

1. `-passes=strength-reduce`
2. `-passes='strength-reduce,instcombine'`
3. `-passes='instcombine,strength-reduce'`

**先预测再动手**：`instcombine` 本身就会做 `mul 2^k → shl`。那么第 3 种顺序下，你的 pass 还会打印出 `[strength-reduce]` 日志吗？如果不会，说明了什么？

**验收命令**：

```bash
cd llvm-hello-compile
for P in 'strength-reduce' 'strength-reduce,instcombine' 'instcombine,strength-reduce'; do
  echo "=== $P ==="
  opt -load-pass-plugin=build/passes/MyPasses.so -passes="$P" \
      -S tests/strength-reduce.ll -o /dev/null
done
```

**通过标准**：能根据 `[strength-reduce]` 日志出现与否，说清哪一轮是谁先把 `mul` 改掉的。

**常见失败 → 说明你哪里没懂**：把三种输出当成一样 → 没注意 pass 的**副作用日志**才是判断「谁干的活」的证据。

---

## L2 加组件（主判据）

### L2-LLVM-06｜给 strength-reduce 加除法规则，并踩中有符号语义的坑

- **检验什么**：这条通过 = 你掌握了「变换的正确性边界」——不是所有代数恒等式在机器语义下都成立
- **前置**：L1-LLVM-05
- **资源**：本地+工具链
- **预计耗时**：2h

**任务**：分两步做，第二步是重点。

1. 在 `passes/StrengthReduce.cpp` 里加规则：`udiv %x, 2^k` → `lshr %x, k`（无符号，安全）。
2. 再尝试加 `sdiv %x, 2^k` → `ashr %x, k`。**先找一个反例**证明它是错的，然后决定：要么放弃这条规则，要么只在原指令带 `exact` 标志时才改写。把你的结论写成代码注释以外的东西——写进测试用例。

同步更新 `tests/strength-reduce.ll`，为新规则加 `CHECK` 行，并为**不该被改写的情形**加 `CHECK-NOT`。

**先预测再动手**：

1. `sdiv i32 -8, 8` 的结果是多少？`ashr i32 -8, 3` 呢？换成 `-7 / 8` 与 `ashr -7, 3` 再算一遍——差在哪？（提示：一个向零取整，一个向负无穷取整）
2. `udiv` 改 `lshr` 时，原指令的 `exact` 标志该不该带过去？不带会不会错？带了会不会错？
3. 参考文件头「看点三」：为什么现有代码**故意不复制** `nsw`/`nuw`？

**验收命令**：

```bash
cd llvm-hello-compile
bash scripts/build_passes.sh --force
bash scripts/run_tests.sh        # 新旧 CHECK 必须全绿
# 用具体数值验证语义没被改坏
cat > /tmp/div.ll <<'EOF'
define i32 @sdiv8(i32 %x) { %d = sdiv i32 %x, 8  ret i32 %d }
define i32 @udiv8(i32 %x) { %d = udiv i32 %x, 8  ret i32 %d }
EOF
opt -load-pass-plugin=build/passes/MyPasses.so -passes=strength-reduce -S /tmp/div.ll
```

**通过标准**：

- `udiv8` 被改写成 `lshr`
- `sdiv8` **保持不变**（或仅在带 `exact` 时改写），且测试里有一条 `CHECK-NOT` 守住这个行为
- 你能写出一组具体数值（如 `x = -7`）说明盲目改写会产生什么错误结果

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 把 `sdiv` 直接改成 `ashr` 且测试还绿了 | 测试没覆盖负数路径；说明不会设计「能抓住 bug」的测试 |
| 认为「除以正数不会有问题」 | 没区分被除数符号与除数符号；机器整数除法的取整方向是关键 |
| 改完 `run_tests.sh` 报 verifier 错误 | 替换指令后没处理 use 关系或类型不匹配 |

---

### L2-LLVM-07｜写一个全新的 Function 分析 Pass 并注册

- **检验什么**：这条通过 = 你掌握了「New PM 里新增 Pass 的完整链路：声明 → 实现 → 注册到正确层级 → 向 AnalysisManager 索取分析 → 写测试」
- **前置**：L0-LLVM-02
- **资源**：本地+工具链
- **预计耗时**：2.5h

**任务**：新增一个 Function 级分析 Pass，管线名 `mem-info`，做这几件事：

- 统计函数内 `alloca` / `load` / `store` / `call` 数量
- 通过 `AM.getResult<AAManager>(F)` 拿到别名分析结果，对**任意两条访存指令**做一次 `alias()` 查询并打印结论（`MustAlias` / `MayAlias` / `NoAlias`）
- 返回 `PreservedAnalyses::all()`

涉及文件：`passes/Passes.h`（加类声明）、新建 `passes/MemInfo.cpp`、`passes/CMakeLists.txt`（加源文件）、`passes/PluginRegistration.cpp`（**注册进 FunctionPassManager 那个回调**）。参照 `passes/CfgInfo.cpp` 的写法，它已经演示了如何索取 `DominatorTreeAnalysis` 与 `LoopAnalysis`。

再新增 `tests/mem-info.ll`，用 `; RUN:` + `CHECK` 断言输出里的计数。

**先预测再动手**：

1. 你把 `mem-info` 注册进 Function 回调，那 `opt -passes=mem-info` 能直接跑吗？还是必须写成 `-passes='function(mem-info)'`？为什么（回看 `PluginRegistration.cpp` 头部注释里的 adaptor 说明）？
2. 分析型 pass 返回 `all()`，如果你**谎报**成 `none()`，会有什么后果？反过来呢（该报 `none()` 却报了 `all()`）？
3. 你的 pass 打印到 `errs()`，而测试是 `opt ... | FileCheck`——管道走的是 stdout。这会不会导致 CHECK 匹配不到？该怎么办？

**验收命令**：

```bash
cd llvm-hello-compile
bash scripts/build_passes.sh --force
opt -load-pass-plugin=build/passes/MyPasses.so -passes=mem-info \
    -disable-output out/02_sum_O0.ll
bash scripts/run_tests.sh        # 应出现第 3 个测试并 PASS
```

**通过标准**：`mem-info` 可通过 `-passes=mem-info` 直接调用；输出包含四类计数与至少一条别名查询结论；`tests/mem-info.ll` 通过。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| `opt` 报 `unknown pass name 'mem-info'` | 注册回调写错层级，或 `.so` 没重新编译（记得 `--force`） |
| FileCheck 匹配不到 | 没处理 stderr/stdout 的区别，`; RUN:` 行需要 `2>&1 \|` 重定向 |
| `AAManager` 拿不到结果 | 不清楚别名分析在 New PM 里是通过 `AAManager` 聚合多个 AA 实现的 |

---

### L2-LLVM-08｜写一个 Module 级变换 Pass：运行时调用计数

- **检验什么**：这条通过 = 你掌握了「Module 级 pass 能做而 Function 级做不了的事：新建全局变量、跨函数改写、在 main 出口插代码」
- **前置**：L1-LLVM-03
- **资源**：本地+工具链
- **预计耗时**：2.5h

**任务**：新增 Module 级变换 Pass，管线名 `count-calls-rt`：

- 在模块里新建一个全局变量 `@__call_count`（`i64`，初值 0）
- 在**每个非声明函数的入口**插入 `load / add 1 / store`
- 在 `main` 的每个 `ret` 前插入一句 `printf("[count] %lld\n", @__call_count)`
- 返回 `PreservedAnalyses::none()`

注册到 `PluginRegistration.cpp` 的 **ModulePassManager** 回调。

**先预测再动手**：

1. 为什么这个 pass 不能写成 Function 级？（提示：新建全局变量属于修改 Module；Function pass 在 New PM 下修改所属 Module 是被禁止的）
2. `main` 可能有多个 `ret`，也可能通过 `exit()` 直接退出——你的计数打印会漏掉哪种情况？
3. 全局变量的 linkage 该选 `InternalLinkage` 还是 `ExternalLinkage`？选错会怎样？

**验收命令**：

```bash
cd llvm-hello-compile
bash scripts/build_passes.sh --force
opt -load-pass-plugin=build/passes/MyPasses.so -passes=count-calls-rt \
    -S out/03b_sum_O2.ll -o out/counted.ll
llc -filetype=obj out/counted.ll -o out/counted.o && clang -no-pie out/counted.o -o out/counted
./out/counted            # 应打印 55 与 [count] N
opt -passes=verify -S out/counted.ll -o /dev/null   # IR 必须合法
```

**通过标准**：可执行文件同时输出原结果与计数值，且计数值与你手工数出的函数调用次数一致；`verify` 无报错。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 计数值永远是 0 | 插桩插在了声明（declaration）上，或插入点在 `ret` 之后（不可达） |
| verifier 报「terminator 不是最后一条指令」 | 在 `ret` **之后**插了指令；不理解基本块结构约束 |
| 链接报未定义符号 | 全局变量 linkage 或 `printf` 声明处理不当 |

---

### L2-LLVM-09｜故意注册到错误的 PM 层级

- **检验什么**：这条通过 = 你彻底搞清了「Pass 的层级不是标签，而是由 `run()` 的第一个参数类型决定的契约」
- **前置**：L2-LLVM-07
- **资源**：本地+工具链
- **预计耗时**：0.5h

**任务**：把 `mem-info`（Function 级）的注册代码搬到 ModulePassManager 那个回调里，重新编译，记录**编译期**的报错信息；再恢复。然后反过来想：如果一个 Module pass 被注册到 Function 回调，为什么同样编不过？

**先预测再动手**：这会是编译期错误还是运行期错误？编译器会在哪一行报什么？

**验收命令**：

```bash
cd llvm-hello-compile
# 改完注册位置后
bash scripts/build_passes.sh --force 2>&1 | tail -30
```

**通过标准**：能复述报错的本质——`MPM.addPass(X)` 要求 `X` 有 `run(Module&, ModuleAnalysisManager&)`，类型不匹配在模板实例化时就被拒了。恢复后重新编译通过。

**常见失败 → 说明你哪里没懂**：以为「层级只是分类，运行时才检查」→ 没建立「Pass 概念 = 一个能对特定 IR 单元 `run` 的类型」这个认识。

---

## L3 打通

### L3-LLVM-10｜让自定义 Pass 混进 clang 的默认 -O2 管线

- **检验什么**：这条通过 = 你掌握了「扩展点（Extension Point）」机制——如何不改 clang 源码就把 pass 插进默认流水线
- **前置**：L2-LLVM-06
- **资源**：本地+工具链
- **预计耗时**：1.5h

**任务**：在 `PluginRegistration.cpp` 里除了现有的 `registerPipelineParsingCallback`，再加一个扩展点回调（如 `PB.registerOptimizerLastEPCallback` 或 `registerPipelineStartEPCallback`），把 `strength-reduce` 自动挂上。然后用 `clang -fpass-plugin=...` 直接编译 C 文件，无需手动 `opt`。

**先预测再动手**：`PipelineStart` 与 `OptimizerLast` 两个位置，你的 pass 分别会看到什么形态的 IR？在哪个位置它更可能「白干」（因为 instcombine 已经做过同样的事）？

**验收命令**：

```bash
cd llvm-hello-compile
bash scripts/build_passes.sh --force
clang -O2 -fpass-plugin=build/passes/MyPasses.so src/sum.c -o /tmp/sum_ep 2>&1 | grep strength-reduce
/tmp/sum_ep
```

**通过标准**：编译过程中出现 `[strength-reduce]` 日志（证明 pass 在默认管线里被执行），且程序输出仍为 `55`。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 没有任何日志 | 扩展点回调注册的是 ModulePassManager，需要用 adaptor 包一层才能塞 Function pass |
| 日志出现但结果错了 | 你的改写规则在真实代码上不成立——回到 L2-LLVM-06 重新检查边界条件 |

---

### L3-LLVM-11｜让分析「说谎」，观察 stale analysis 的后果

- **检验什么**：这条通过 = 你真正理解 `PreservedAnalyses` 不是形式主义，而是缓存失效协议
- **前置**：L2-LLVM-07
- **资源**：本地+工具链
- **预计耗时**：1.5h

**任务**：写一个临时的 Function 变换 Pass（可以叫 `break-cfg`），做一件**确实改变 CFG** 的事（例如把某个条件分支改成无条件分支，或删掉一个不可达块），但 `run()` **故意返回 `PreservedAnalyses::all()`**。然后跑 `-passes='break-cfg,cfg-info'`，观察 `cfg-info` 拿到的支配树是否与实际 CFG 不符。

**先预测再动手**：

1. `cfg-info` 是向 `AM.getResult<DominatorTreeAnalysis>(F)` 要结果的。上一个 pass 谎报 `all()` 后，AnalysisManager 会重算还是返回缓存？
2. 你预期看到的是崩溃、断言失败，还是「安静地给出错误答案」？哪一种更可怕？

**验收命令**：

```bash
cd llvm-hello-compile
bash scripts/build_passes.sh --force
opt -load-pass-plugin=build/passes/MyPasses.so \
    -passes='break-cfg,cfg-info' -disable-output out/03a_sum_mem2reg.ll
# 对照组：把返回值改成 PreservedAnalyses::none() 再跑一次
```

**通过标准**：两次运行的 `cfg-info` 输出（基本块数 / 支配树根 / 循环深度）出现可观察的差异，且你能说清差异来自缓存未失效。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 两次输出一模一样 | 你的 `break-cfg` 其实没真正改变 CFG，或 opt 在 pass 之间做了额外的验证/重建 |
| 直接崩溃就以为失败了 | 崩溃恰恰是**好结果**（问题被暴露了）；真正危险的是安静地算错 |

---

### L3-LLVM-12｜换个目标架构看后端

- **检验什么**：这条通过 = 你理解「同一份 IR → 不同 target 的指令选择与寄存器分配结果不同」，IR 是目标无关的但不是目标无知的
- **前置**：L0-LLVM-01
- **资源**：本地+工具链（需 LLVM 编译时包含 AArch64 后端，绝大多数发行版都有）
- **预计耗时**：1h

**任务**：把 `out/03b_sum_O2.ll` 分别用 x86-64 与 aarch64 跑 `llc`，对比：指令条数、调用约定（参数寄存器）、栈帧布局。再用 `llc -march=aarch64 -stop-after=finalize-isel` 看 MIR，找出虚拟寄存器还未分配的证据。

**先预测再动手**：同一条 `mul` 在两个架构上会选出几条机器指令？函数第一个整型参数分别放在哪个寄存器？

**验收命令**：

```bash
cd llvm-hello-compile
llc -O2 out/03b_sum_O2.ll -o /tmp/sum_x86.s
llc -O2 -march=aarch64 out/03b_sum_O2.ll -o /tmp/sum_arm.s
llc -O2 -march=aarch64 -stop-after=finalize-isel out/03b_sum_O2.ll -o /tmp/sum_arm.mir
llc --version | grep -A20 'Registered Targets'   # 确认 aarch64 可用
```

**通过标准**：能指出两份汇编中**至少三处**架构差异（参数寄存器、返回寄存器、栈操作指令），并在 MIR 里指出形如 `%0:gpr32` 的虚拟寄存器。

**常见失败 → 说明你哪里没懂**：`llc` 报 `invalid target` → 本地 LLVM 未编译 AArch64 后端，可换成 `-march=riscv64` 或任何 `Registered Targets` 里列出的目标。

---

## 条目 × 资源需求速查

| 编号 | 任务 | 级别 | 资源 | 耗时 |
|------|------|------|------|------|
| L0-LLVM-01 | 跑通主链路，指认 mem2reg | L0 | 本地+工具链 | 1h |
| L0-LLVM-02 | 巡礼 17 站 + 测试全绿 | L0 | 本地+工具链 | 1.5h |
| L1-LLVM-03 | inject-log 打印参数值 | L1 | 本地+工具链 | 1.5h |
| L1-LLVM-04 | 删 restrict 看别名与向量化 | L1 | 本地+工具链 | 1h |
| L1-LLVM-05 | pass 顺序敏感性 | L1 | 本地+工具链 | 0.5h |
| **L2-LLVM-06** | **加除法规则，踩有符号语义坑** | L2 | 本地+工具链 | 2h |
| **L2-LLVM-07** | **写新的 Function 分析 Pass** | L2 | 本地+工具链 | 2.5h |
| L2-LLVM-08 | 写 Module 级插桩 Pass | L2 | 本地+工具链 | 2.5h |
| L2-LLVM-09 | 故意注册错层级 | L2 | 本地+工具链 | 0.5h |
| L3-LLVM-10 | 挂进默认 -O2 管线 | L3 | 本地+工具链 | 1.5h |
| L3-LLVM-11 | 让分析说谎，观察 stale | L3 | 本地+工具链 | 1.5h |
| L3-LLVM-12 | 换 target 看后端 | L3 | 本地+工具链 | 1h |

**本册无任何条目需要 GPU 或集群资源。** 全部可在个人开发机完成，总计约 17 小时。

**下一步**：完成本册后接 [`./02-mlir.md`](./02-mlir.md)——同样是「加一个 pass」，但你会看到 MLIR 把「层级」从固定的 Module/Function 换成了任意 Op，对照着做感受最深。
