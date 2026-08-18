# 检验体系 03｜IREE

> 分级定义、资源标签、条目写法见总纲 [`./README.md`](./README.md)。
> 知识本体见 [`../learning-guides/iree-learning-guide.md`](../learning-guides/iree-learning-guide.md)（本册每条都标了对应章节）。

---

## 0. 这一册在检验什么

学习指南里 IREE 的「过关标准」是口述题——能不能画出 `linalg → flow → stream → hal`、能不能说清 timeline semaphore 为什么比 `CUevent` 强。
**本册把同样的知识点换成动手题**：改一个模型 / 加一条 pass / 换一个后端，然后用 `grep` 和退出码判定你有没有真的看懂。两边一一呼应，不重复。

### 知识点 → 由哪条检验兜住

| 知识点（指南章节） | 本册条目 |
|------|---------|
| 五层 dialect 各固化了什么决定（第 2 章） | L0-IREE-03 |
| dispatch region 的形成与融合规则（2.2 转折点 1） | L1-IREE-04、L1-IREE-05 |
| 动态 shape 全链路（第 2 章 + 4.3） | L1-IREE-06 |
| 编译流水线的可插入点 / preprocessing（第 5 章） | L2-IREE-07 |
| executable / variant / export 四层与 fat binary（4.7） | L2-IREE-08 |
| `buffer_view` 与 ABI 边界契约（4.3、4.9） | L2-IREE-09 |
| compiler target 与 runtime device 的 N:M 关系（4.1.1） | L2-IREE-08、L3-IREE-10 |
| stream 时间线 → HAL fence / command buffer（3.1、4.5、4.6） | L3-IREE-11 |
| 端到端跑通与数值验证（第 5 章） | L0-IREE-02、L3-IREE-10 |

### 入门线（按总纲 §2）

**全部 L0（01、02、03）+ L1 至少两条（推荐 04、05）+ L2 至少两条（推荐 08、09）**，共 7 条，约一天。
L2-IREE-07 的做法 B 需要从源码构建 IREE，L3-IREE-10 需要 GPU，两者都不计入入门线。

### 环境与工作目录

本册是**动手改东西**的，产物一律**写在仓库外的临时目录**（`~/iree-check/`），不要提交进仓库。

```bash
mkdir -p ~/iree-check && cd ~/iree-check
pip install iree-base-compiler iree-base-runtime   # 较老版本的包名是 iree-compiler / iree-runtime
```

> **和仓库里的 [`iree-lab/`](../../iree-lab/) 是什么关系**：
> `iree-lab/` 是**只读的教学产物** —— 跑完 `bash scripts/run.sh` 就能看到各相位 IR、
> 数值校验、三组对照实验，用来配合指南**建立认知**。
> 本册要你**改模型、加 pass、换后端**，那些改动不该污染 lab，所以另开 `~/iree-check/`。
>
> 建议顺序：**先跑一遍 `iree-lab/`，再做本册**。本册很多条目要求"先预测再动手"，
> 而 `iree-lab/out/PHASES.md` 里已经有对照答案。
> 模型也可以直接拷过去改：`cp <repo>/iree-lab/models/tiny_mlp.mlir ~/iree-check/`。

**命令一律按 bash 写**。Windows 下建议用 Git Bash 或 WSL；若坚持用 PowerShell，把 `grep -c "X" f.mlir` 换成 `(Select-String -Pattern "X" f.mlir | Measure-Object).Count`。

### 版本纪律（每条都适用，只说一次）

IREE 的命令行 flag 版本差异很大，尤其是「指定目标后端」这一组。本册**统一按下面两种写法给命令，你本地用哪种以 `--help` 为准**：

```bash
# 新写法（近两年版本）
--iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu
# 旧写法（老版本；部分新版仍兼容，可能提示 deprecated）
--iree-hal-target-backends=llvm-cpu
```

开工前先跑一遍自查（这也是 L0-IREE-01 的内容）：

```bash
iree-compile --version
iree-compile --help | grep -i "compile-to"
iree-compile --help | grep -iE "target-device|target-backends|preprocessing-pass-pipeline"
iree-compile --iree-hal-list-target-backends
iree-run-module --help | grep -iE "input|expected_output|device"
```

**凡是本册写了「以 `--help` 为准」的地方，都是版本间拼写确有差异的地方**——照着自查命令找到本地的正确写法再往下走，不要硬套本文档。

---

## L0 组：复现

### L0-IREE-01｜工具链自检：搞清你手上是哪五个二进制、哪套 flag

- **检验什么**：这条通过 = 你真的掌握了「IREE 工具链的分工：`iree-compile` 编、`iree-run-module` 跑、`iree-opt` 单独跑 pass、`iree-dump-module` 看产物、`iree-benchmark-module` 测速」，以及「flag 拼写必须以本地版本为准」这条纪律
- **前置**：装好 `iree-base-compiler` + `iree-base-runtime`
- **资源**：本地+工具链
- **预计耗时**：0.5h

**任务**：在 `~/iree-check/` 下建一个 `env.md`，把下面四样东西记进去：

1. 五个工具各自是否存在、版本号（`command -v iree-compile iree-run-module iree-opt iree-dump-module iree-benchmark-module`）。缺哪个记下来，后面用到时才知道要换方案。
2. 本地 `--compile-to` 接受的**全部相位名**（从 `--help` 抄下来，不要抄本文档的）。
3. 本地是 `--iree-hal-target-device=` 还是 `--iree-hal-target-backends=`，或两者都收。
4. `iree-compile --iree-hal-list-target-backends` 与 `iree-run-module --list_drivers` 各列出了什么——**这两张表不是一回事**，前者是编译后端，后者是运行时 driver。

**先预测再动手**：先写下你的答案再去查。

- `llvm-cpu` 是编译后端还是运行时 driver？`local-sync` 呢？两张表为什么长度不一样？
- 一个编译后端能对应几个运行时 driver？（指南 [4.1.1](../learning-guides/iree-learning-guide.md#411-三条设计取向理解-hal-的钥匙) 的 N:M 那张表）
- 如果 `iree-dump-module` 不存在，本册哪些条目会受影响？

**验收命令**：

```bash
cd ~/iree-check
command -v iree-compile iree-run-module iree-opt iree-dump-module iree-benchmark-module
iree-compile --version
iree-compile --help | grep -A 30 -i "compile-to"
iree-compile --iree-hal-list-target-backends
iree-run-module --list_drivers
iree-run-module --list_devices
```

**通过标准**：`iree-compile` / `iree-run-module` / `iree-opt` 三个必须能打印版本或帮助（缺任何一个，本册大部分条目做不了）；`env.md` 里写下了本地的相位名列表、目标后端列表、driver 列表，以及本机采用的是新写法还是旧写法。

**常见失败 → 说明你哪里没懂**：

| 现象 | 你哪里没懂 |
|------|-----------|
| 把 `--iree-hal-target-backends=local-sync` 写进编译命令 | 混淆了「编译后端」与「运行时 driver」；`local-sync` 是 `--device=` 的取值 |
| `--help` 里找不到 `--compile-to` | 装的可能是只带 runtime 的包，或版本过老；先解决工具链再继续 |
| 抄本文档的相位名而不是抄本地 `--help` | 没接受「flag 随版本漂移」这个前提，后面每条都会被同一个坑绊住 |

---

### L0-IREE-02｜最小 linalg 模型：编到 CPU 并做数值验证

- **检验什么**：这条通过 = 你真的掌握了「`.mlir` → `.vmfb` → 在 HAL device 上 invoke」这条最短闭环，以及 `--input` / `--function` 分别对应产物里的什么
- **前置**：L0-IREE-01；会读 `linalg.generic`（不会就先补 [`./02-mlir.md`](./02-mlir.md) 与指南 [2.2](../learning-guides/iree-learning-guide.md#22-三个最需要理解的转折点)）
- **资源**：本地+工具链
- **预计耗时**：0.5h

**任务**：手写 `abs.mlir`（**不要用指南里 `math.absf` 直接作用于 `tensor<f32>` 的那份**，那份省掉了 linalg 层，看不到 dispatch 的结构来源）：

```mlir
func.func @abs(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %e0 = tensor.empty() : tensor<4xf32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%arg0 : tensor<4xf32>) outs(%e0 : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = math.absf %in : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
```

编到 `llvm-cpu`，在 `local-sync` 上跑，并用 `--expected_output` 让**机器**判定数值对不对（不要用肉眼）。

**先预测再动手**：

- `--function=abs` 这个名字，运行时是从 `.vmfb` 的哪一部分查出来的？（指南 [第 1 章](../learning-guides/iree-learning-guide.md#第-1-章-iree-是什么一分钟建立坐标系)的产物结构图）
- 编译时指定的是 `llvm-cpu`，运行时指定的是 `local-sync`，为什么两个名字对不上也能跑通？
- 如果把 `--expected_output` 故意写错一个数，你预期进程退出码是 0 还是非 0？

**验收命令**：

```bash
cd ~/iree-check
iree-compile abs.mlir \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  -o abs.vmfb
# 旧版本改用：iree-compile abs.mlir --iree-hal-target-backends=llvm-cpu -o abs.vmfb

# 正例：splat 输入，全 -2.0 → 全 2.0
iree-run-module --device=local-sync --module=abs.vmfb --function=abs \
  --input=4xf32=-2.0 --expected_output=4xf32=2.0
echo "exit=$?"

# 反例：故意把期望值写错，确认这个校验真的在校验
iree-run-module --device=local-sync --module=abs.vmfb --function=abs \
  --input=4xf32=-2.0 --expected_output=4xf32=3.0
echo "exit=$?"

# 换一份逐元素输入（括号/分隔符写法各版本略有差异，以 --help 的示例为准）
iree-run-module --device=local-sync --module=abs.vmfb --function=abs \
  --input="4xf32=[-1.0 2.0 -3.0 4.0]"
```

**通过标准**：正例退出码为 0，反例退出码非 0（**两个都要看到**，只跑正例不算通过——不能证明校验生效）；`abs.vmfb` 文件存在且非空；把 `--device` 换成 `local-task` 再跑一次，结果完全相同。

**常见失败 → 说明你哪里没懂**：

| 现象 | 你哪里没懂 |
|------|-----------|
| `--function` 找不到 | 函数名与 `.mlir` 里的 `func.func @name` 不一致；或函数没被导出（不是 public） |
| 报 element type / shape 不匹配 | `--input` 的 `4xf32` 必须与签名逐字对上，这是 ABI 契约，见 L2-IREE-09 |
| 用 `--device=llvm-cpu` | 又一次混淆编译后端与 runtime driver（L0-IREE-01 没做扎实） |
| 反例也返回 0 | 你的版本没有 `--expected_output` 或拼写不同；查 `--help`，否则本册所有数值判定都失效 |

---

### L0-IREE-03｜`--compile-to` 逐相位 dump：看 tensor 怎么一路变成设备命令

- **检验什么**：这条通过 = 你真的掌握了「每一相位新增了哪一类信息」——flow 加 dispatch 划分、stream 加时序与生命周期、hal 加设备对象、vm 把 op 变函数调用
- **前置**：L0-IREE-02
- **资源**：本地+工具链
- **预计耗时**：2h

**任务**：把 `abs.mlir` 在各相位停下来 dump 成一组文件，然后用 grep 逐相位核对特征 op。**相位名以你 `env.md` 里抄下来的本地 `--help` 为准**，下面这串是常见版本的取值：

```
input → abi → preprocessing → global-optimization → dispatch-creation
      → flow → stream
      → executable-sources → executable-configurations → executable-targets
      → hal → vm
```

dump 完之后，逐个文件回答「这一相位比上一相位多了什么」，写进 `env.md` 或一份 `phases.md`。

**先预测再动手**：

- `tensor` 这个类型最后一次出现在哪个相位？为什么再往下它必须消失？（提示：设备只认字节和偏移）
- `hal.executable.variant` 在 `flow` 相位里能找到吗？如果找不到，那时候「这段 kernel 编给谁」这个决定还没做，谁在做？
- `abi` 相位新增的 `hal.tensor.import` / `export` 是给谁看的边界？它两侧的类型分别是什么？

**验收命令**：

```bash
cd ~/iree-check
for phase in input abi global-optimization dispatch-creation \
             flow stream executable-sources executable-targets hal vm; do
  iree-compile abs.mlir \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --compile-to=$phase -o abs.$phase.mlir || echo "PHASE-NOT-SUPPORTED: $phase"
done

# 逐相位核对特征 op：每一行都必须打印 OK
check() { grep -q "$2" "$1" && echo "OK   $1 : $2" || echo "FAIL $1 : $2"; }
check abs.abi.mlir                 "hal.tensor.import"
check abs.flow.mlir                "flow.executable"
check abs.flow.mlir                "flow.dispatch"
check abs.stream.mlir              "stream.cmd.execute"
check abs.stream.mlir              "stream.timepoint"
check abs.executable-sources.mlir  "hal.executable.variant"
check abs.hal.mlir                 "hal.command_buffer.create"
check abs.hal.mlir                 "hal.device.queue.execute"
check abs.vm.mlir                  "vm.rodata"
check abs.vm.mlir                  "vm.call @hal"
```

**通过标准**：上面 10 行 check **全部打印 OK**（某个相位名你的版本不支持就换成 `--help` 里的对应名字，不要跳过该相位对应的 check）；另外 `grep -c "tensor<" abs.hal.mlir` 的结果应显著小于 `abs.flow.mlir`（tensor 世界在 hal 相位基本退场，函数边界上的 `!hal.buffer_view` 除外）。

**常见失败 → 说明你哪里没懂**：

| 现象 | 你哪里没懂 |
|------|-----------|
| `--compile-to=xxx` 报错 | 没照 L0-IREE-01 抄本地相位名 |
| `abs.stream.mlir` 里找不到 `stream.resource.alloca` | 这个程序可能没有需要临时分配的中间结果——想清楚 `transient` 是什么才会出现（指南 [3.4](../learning-guides/iree-learning-guide.md#34-stream-ordered-allocation峰值内存的关键)） |
| 说不出 hal 相位比 stream 相位多了什么 | 只在「看到了 op 名」层面通过，没建立「每层固化一个决定」的模型；回看指南 [2.1](../learning-guides/iree-learning-guide.md#第-2-章-编译侧dialect-流水线) 的职责表 |
| 认为 `--compile-to` 输出的是产物 | 它输出的是**该相位结束时的完整 IR**，不是 `.vmfb` |

---

## L1 组：改一处

### L1-IREE-04｜把两个 elementwise 串起来：数 dispatch，看融合

- **检验什么**：这条通过 = 你真的掌握了「dispatch region 是设备调用的最小单位」，以及「生产者-消费者 elementwise 默认会被融进同一个 dispatch」
- **前置**：L0-IREE-03
- **资源**：本地+工具链
- **预计耗时**：1h

**任务**：写 `chain.mlir`——`abs` 之后接一个 `x*x`，两个都是逐元素、同 shape、同 iterator：

```mlir
func.func @chain(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  %e0 = tensor.empty() : tensor<64xf32>
  %abs = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%arg0 : tensor<64xf32>) outs(%e0 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = math.absf %in : f32
      linalg.yield %v : f32
  } -> tensor<64xf32>
  %e1 = tensor.empty() : tensor<64xf32>
  %sq = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%abs : tensor<64xf32>) outs(%e1 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.mulf %in, %in : f32
      linalg.yield %v : f32
  } -> tensor<64xf32>
  return %sq : tensor<64xf32>
}
```

dump 到 `flow` 相位，数 `flow.executable` 与 `flow.dispatch` 的个数，并**打开那个 executable，确认两个算子的 payload 真的在同一个函数体里**。

**先预测再动手**：动手前把答案写在纸上。

- `flow.executable` 会是 1 个还是 2 个？依据是什么——两个 op 的 `iterator_types` 和 `indexing_maps` 有什么关系？
- 中间结果 `%abs` 那块 `tensor<64xf32>` 在最终产物里还需要一块内存吗？如果融合了，它去哪了？
- 如果把第二个 op 的 `indexing_maps` 输入侧改成 `affine_map<(d0) -> (63 - d0)>` 这类反向映射，你预期融合还成立吗？

**验收命令**：

```bash
cd ~/iree-check
iree-compile chain.mlir \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --compile-to=flow -o chain.flow.mlir

grep -c "flow.executable "  chain.flow.mlir   # dispatch 单元数
grep -c "= flow.dispatch @" chain.flow.mlir   # 派发点数
grep -c "math.absf"  chain.flow.mlir
grep -c "arith.mulf" chain.flow.mlir
```

**通过标准**：`flow.executable ` 计数为 **1**；`math.absf` 与 `arith.mulf` 都出现在**同一个** `flow.executable` 的 `builtin.module` 内（用 `sed -n` 或编辑器打开确认，不是靠总数猜）；把这个基线数字 1 记下来，L1-IREE-05 要跟它比。

**常见失败 → 说明你哪里没懂**：

| 现象 | 你哪里没懂 |
|------|-----------|
| 计数是 2 | 两个 op 的 shape / iterator_types / indexing_maps 没写成可融合的形态；回看指南 [2.2 转折点 1](../learning-guides/iree-learning-guide.md#22-三个最需要理解的转折点)——融合只看这两组元信息 |
| `flow.executable` 数与 `flow.dispatch` 数不相等 | 正常：同一个 executable 可以被派发多次。想清楚「编译单元」与「调用点」的区别 |
| 只数数字、没打开 executable 看 payload | 计数是代理指标，真正的判据是「两个算子在同一个 kernel 体内」 |

---

### L1-IREE-05｜插一个算子打断融合：找出真正的 dispatch 边界在哪

- **检验什么**：这条通过 = 你真的掌握了「什么样的算子会成为 dispatch 边界」——不是「op 多了就分开」，而是「访问模式不再对齐 / 需要全量结果」才分开
- **前置**：L1-IREE-04（要用它的基线计数 1）
- **资源**：本地+工具链
- **预计耗时**：2h

**任务**：在 `chain.mlir` 的两个 elementwise 之间**分别**插入三种算子，各存一份，各自 dump 到 flow 相位并计数。

变体 A `chain_reshape.mlir`（形状变换）：

```mlir
func.func @chain_reshape(%arg0: tensor<8x8xf32>) -> tensor<64xf32> {
  %e0 = tensor.empty() : tensor<8x8xf32>
  %abs = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<8x8xf32>) outs(%e0 : tensor<8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = math.absf %in : f32
      linalg.yield %v : f32
  } -> tensor<8x8xf32>
  %flat = tensor.collapse_shape %abs [[0, 1]] : tensor<8x8xf32> into tensor<64xf32>
  %e1 = tensor.empty() : tensor<64xf32>
  %sq = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%flat : tensor<64xf32>) outs(%e1 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.mulf %in, %in : f32
      linalg.yield %v : f32
  } -> tensor<64xf32>
  return %sq : tensor<64xf32>
}
```

变体 B `chain_transpose.mlir`：把变体 A 的 `collapse_shape` 换成一个转置——即再写一个 2D `linalg.generic`，输入侧 map 用 `affine_map<(d0, d1) -> (d1, d0)>`、输出侧用 `affine_map<(d0, d1) -> (d0, d1)>`，形状 `8x16 → 16x8`。

变体 C `chain_reduce.mlir`（归约 + 广播回来，softmax 的骨架）：

```mlir
func.func @chain_reduce(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %e0 = tensor.empty() : tensor<64xf32>
  %abs = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%arg0 : tensor<64xf32>) outs(%e0 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = math.absf %in : f32
      linalg.yield %v : f32
  } -> tensor<64xf32>
  %init = tensor.empty() : tensor<f32>
  %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<f32>) -> tensor<f32>
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]}
      ins(%abs : tensor<64xf32>) outs(%filled : tensor<f32>) {
    ^bb0(%in: f32, %acc: f32):
      %a = arith.addf %in, %acc : f32
      linalg.yield %a : f32
  } -> tensor<f32>
  %e1 = tensor.empty() : tensor<64xf32>
  %norm = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%abs, %sum : tensor<64xf32>, tensor<f32>) outs(%e1 : tensor<64xf32>) {
    ^bb0(%in: f32, %s: f32, %out: f32):
      %d = arith.divf %in, %s : f32
      linalg.yield %d : f32
  } -> tensor<64xf32>
  return %norm : tensor<64xf32>
}
```

**先预测再动手**：三个变体各写一个预测数字，再跑。

- 三种插入里，哪种会让 dispatch 数量变多、哪种不会？**reshape 为什么可能根本不成为边界**（提示：它不搬数据，只改索引解释，编译器可以把它往两边传播甚至折叠掉）。
- 变体 C 的第二个 elementwise 需要 `%sum` 的**完整**结果才能算第一个元素——这个「需要全量」的依赖和 elementwise 的「逐点」依赖，在 `indexing_maps` 上体现为什么差别？
- 变体 C 里 `%abs` 被两个消费者用了（reduce 和 divide）。如果它被融进了两个 dispatch，是不是意味着 `abs` 被算了两遍？重算和多存一块内存，编译器凭什么选？

**验收命令**：

```bash
cd ~/iree-check
for m in chain chain_reshape chain_transpose chain_reduce; do
  iree-compile $m.mlir \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --compile-to=flow -o $m.flow.mlir
  printf "%-16s executables=%s dispatches=%s\n" "$m" \
    "$(grep -c 'flow.executable ' $m.flow.mlir)" \
    "$(grep -c '= flow.dispatch @' $m.flow.mlir)"
done

# 没有打断融合的变体，去确认那个 op 到底怎么了：还在？被传播了？被折叠没了？
grep -n "collapse_shape" chain_reshape.flow.mlir
```

**通过标准**：四行计数表全部记下来；**三个变体里至少有一个的 `flow.executable ` 计数严格大于基线 1**；对于计数**没有**变化的那些变体，你能用 grep 证明插入的那个 op 在 flow 相位的下场（三选一：被折叠消失、被传播到 dispatch 边界外、被融进了某个 dispatch 内部），并说出为什么它不构成边界。

**常见失败 → 说明你哪里没懂**：

| 现象 | 你哪里没懂 |
|------|-----------|
| 预测「加一个 op 就多一个 dispatch」 | 把 dispatch 当成算子的容器了。dispatch 边界由**访问模式**决定，不是由 op 个数决定 |
| reshape 版计数没变就认为实验失败 | 这恰恰是本条最值钱的结论：形状变换通常不搬数据，因此不天然成为边界 |
| 说不清变体 C 为什么分开 | 没抓住「归约的消费者需要完整结果」这个本质；这也是 softmax / layernorm 在所有编译器里都要特殊处理的原因 |
| 三个变体的计数都等于 1 | 你的版本开了更激进的融合。改用 `--mlir-print-ir-after-all` 找到融合发生在哪一相位，把那一相位的 pass 名记下来——结论仍成立，只是边界被推后了 |

---

### L1-IREE-06｜把 shape 改成动态：看 `?` 一路怎么传下去

- **检验什么**：这条通过 = 你真的掌握了「静态 shape 是编译期编死在 kernel 里的，动态 shape 必须显式作为 SSA 值一路传到 workgroup count 与 binding 长度」
- **前置**：L1-IREE-04
- **资源**：本地+工具链
- **预计耗时**：1h

**任务**：把 `chain.mlir` 的 `tensor<64xf32>` 全改成 `tensor<?xf32>`，存为 `chain_dyn.mlir`。`tensor.empty` 需要显式给出动态维：

```mlir
  %c0 = arith.constant 0 : index
  %d  = tensor.dim %arg0, %c0 : tensor<?xf32>
  %e0 = tensor.empty(%d) : tensor<?xf32>
```

然后 dump `flow` / `hal` 两相位，与静态版逐处对比；最后用两种不同长度的输入各跑一次，确认同一个 `.vmfb` 都能算对。

**先预测再动手**：

- dispatch 数量会变吗？动态 shape 影响的是「怎么分组」还是「每组算多大」？
- `flow.dispatch` 的调用点会多出什么操作数？`!flow.dispatch.tensor<readonly:tensor<?xf32>>` 后面那个 `{%dim}` 是给谁用的？
- kernel 侧原来 `memref<64xf32>` 的 `64` 是编译期常量。改成动态后，这个长度从哪里来——运行时读 `buffer_view` 的 dim，还是 host 侧作为 constant 推给 kernel？（指南 [4.7.4](../learning-guides/iree-learning-guide.md#474-pipeline-layouthost-与-device-的-abi-契约)）

**验收命令**：

```bash
cd ~/iree-check
iree-compile chain_dyn.mlir \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --compile-to=flow -o chain_dyn.flow.mlir
iree-compile chain_dyn.mlir \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --compile-to=hal -o chain_dyn.hal.mlir
iree-compile chain_dyn.mlir \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  -o chain_dyn.vmfb

grep -c "flow.executable " chain_dyn.flow.mlir      # 与 L1-IREE-04 的 1 对比
grep -n  "tensor<?xf32>"   chain_dyn.flow.mlir | head
grep -nE "buffer_view.dim|hal.buffer_view.assert" chain_dyn.hal.mlir | head

# 同一个 vmfb，两种长度
iree-run-module --device=local-sync --module=chain_dyn.vmfb --function=chain \
  --input=64xf32=-2.0 --expected_output=64xf32=4.0
iree-run-module --device=local-sync --module=chain_dyn.vmfb --function=chain \
  --input=7xf32=-3.0  --expected_output=7xf32=9.0
```

**通过标准**：`flow.executable ` 计数与静态版一致（**融合结构不因动态 shape 改变**）；`chain_dyn.flow.mlir` 里能找到 `tensor<?xf32>` 与随之传递的 `index` 类型操作数；`chain_dyn.hal.mlir` 里能找到从 `buffer_view` 取维度的 op；两次不同长度的运行退出码都为 0。

**常见失败 → 说明你哪里没懂**：

| 现象 | 你哪里没懂 |
|------|-----------|
| `tensor.empty()` 报 operand 数不对 | 动态维必须显式给尺寸——这正是「动态 shape 要当值传」的第一课 |
| 认为动态版会多出很多 dispatch | 划分依据是访问模式，与具体尺寸无关 |
| 静态版换个长度也能跑 | 不可能。静态版换长度必然报 shape 错——如果你没试，去试，这是 L2-IREE-09 的预演 |
| 说不出 workgroup 数量在动态下怎么算 | 去看 `hal.executable.export` 上的 `count` region：它是一段**运行时执行的 IR**，由 workload 算出 3D grid |

---

## L2 组：加组件

### L2-IREE-07｜把自己的 pass 塞进流水线：两条路，一条不用编源码

- **检验什么**：这条通过 = 你真的掌握了「IREE 的流水线是可插入的」，并且分得清「在前端把 IR 改好」与「在 IREE 内部的 preprocessing 相位插 pass」这两个入口的差别
- **前置**：L1-IREE-04（用它的 `chain.mlir`）；做法 B 另需从源码构建 IREE
- **资源**：做法 A 为 `本地+工具链`；做法 B 为 `本地+工具链`（但需要一次完整源码构建，通常 1～3 小时机时 + 数十 GB 磁盘）
- **预计耗时**：做法 A 半天；做法 B 另加半天到一天

**任务**：目标是同一个——**让一个 pass 作用在 IREE 编译流程里，并观察它对最终 dispatch 划分的影响**。按你的条件二选一（做法 A 是入门线内的推荐路径）。

**做法 A（不需要构建 IREE 源码）**

1. 先用 `iree-opt` 在**编译流程之外**手工做一次融合，确认 pass 真的改了 IR：

```bash
iree-opt --help | grep -i "fuse-elementwise"   # 先确认本地注册了这条 pass，没有就在 --help 里找等价的
iree-opt chain.mlir \
  --pass-pipeline='builtin.module(func.func(linalg-fuse-elementwise-ops,canonicalize,cse))' \
  -o chain_fused.mlir
grep -c "linalg.generic" chain.mlir chain_fused.mlir
```

2. 把手工融合后的 IR 再喂给 `iree-compile`，与原版对比 flow 相位的 dispatch 计数。
3. 用 IREE 自带的插入点，把同一条 pass 注入到编译流程内部（**flag 拼写以 `iree-compile --help | grep -i preprocessing` 为准**，常见形态是 `--iree-preprocessing-pass-pipeline=`），免去中间文件：

```bash
iree-compile chain.mlir \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-preprocessing-pass-pipeline='builtin.module(func.func(linalg-fuse-elementwise-ops))' \
  --compile-to=flow -o chain_pp.flow.mlir
```

**做法 B（需要从源码构建 IREE，诚实说：这条成本高）**

在源码树里新增一个自己的 pass（放在 `compiler/src/iree/compiler/Preprocessing/` 下，或写成独立的 MLIR pass plugin），注册后用 `--iree-preprocessing-pass-pipeline=` 按名字调用；若 `iree-opt --help | grep load-pass-plugin` 有输出，也可以用 `--load-pass-plugin=./libMyPass.so` 免去改主干。**做这条之前先把做法 A 做完**——否则你连「插进去之后该看什么差异」都不知道。

**先预测再动手**：

- 在前端先融合好，和让 IREE 自己在 dispatch-creation 相位融合，最终 dispatch 数量会一样吗？如果一样，说明什么？
- `linalg-fuse-elementwise-ops` 改的是 `linalg.generic` 的个数。**op 个数变了，dispatch 个数就一定变吗？**
- 如果想让插进去的 pass 真的改变 dispatch 划分，它必须改动 IR 的哪一类信息？（回到 `iterator_types` / `indexing_maps`）

**验收命令**：

```bash
cd ~/iree-check
# ① pass 确实改了 IR
grep -c "linalg.generic" chain.mlir chain_fused.mlir

# ② 三条路线的 flow 相位 dispatch 计数放在一起比
for m in chain chain_fused; do
  iree-compile $m.mlir --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --compile-to=flow -o $m.flow.mlir
  printf "%-14s executables=%s\n" "$m" "$(grep -c 'flow.executable ' $m.flow.mlir)"
done
# ③ 注入 preprocessing 的那条：chain_pp.flow.mlir 由上面【任务】第 3 步生成，
#    没跑那步就跳过，不影响前两条路线的对比
[ -f chain_pp.flow.mlir ] && \
  printf "%-14s executables=%s\n" "chain_pp" "$(grep -c 'flow.executable ' chain_pp.flow.mlir)"
```

**通过标准**：`linalg.generic` 计数从 **2 降到 1**（证明这条 pass 真的生效了）；三条路线（原版 / 手工预融合 / 注入 preprocessing）的 `flow.executable ` 计数**全部相同**；你能用一句话解释这个「相同」意味着什么——IREE 自己的融合已经把这活干了，前置 pass 只是提前做了同一件事。做法 B 额外要求：`--iree-preprocessing-pass-pipeline` 里写上你自己的 pass 名字能被识别（写错名字时报 unknown pass），且 pass 生效前后 IR 有可 `diff` 出的差异。

**常见失败 → 说明你哪里没懂**：

| 现象 | 你哪里没懂 |
|------|-----------|
| `--pass-pipeline` 报 anchor 不匹配 | pipeline 字符串要写清作用的 op：`builtin.module(func.func(...))`，pass 只能挂在它声明的 op 类型上 |
| 找不到 `linalg-fuse-elementwise-ops` | 本地 `iree-opt` 注册的 pass 集合不同，去 `--help` 里挑一条同类的；这本身就是「pass 名不能背，要查」的一课 |
| 注入后 dispatch 数变了却说不出原因 | 说明这条 pass 改到了访问模式；去 diff 两份 flow IR，定位到具体哪个 `indexing_maps` 变了 |
| 直接上手做法 B | 没有做法 A 建立的观察基线，构建两小时之后你不知道该看什么 |

---

### L2-IREE-08｜一个 vmfb 装两个后端：executable variant 就是 IREE 的 fatbin

- **检验什么**：这条通过 = 你真的掌握了「`hal.executable` 是编译单元、`variant` 是同一段 kernel 的多份目标特化实现、运行时按 target/condition 选第一个可用者」，以及它与 CUDA fatbin 是同一个设计模式
- **前置**：L0-IREE-03
- **资源**：本地+工具链
- **预计耗时**：2h

**任务**：同一个 `chain.mlir` 编两次——单后端 `llvm-cpu`，双后端 `llvm-cpu` + `vmvx`（`vmvx` 是可移植的字节码解释后端，纯 CPU，不需要任何额外硬件）。对比：

1. `--compile-to=hal` 的 dump 里 `hal.executable.variant` 的个数与各自的 `#hal.executable.target<...>` 字符串。
2. 两个 `.vmfb` 的字节大小。
3. 用 `iree-dump-module` 看产物结构（若这个工具你本地没有，用第 1、2 条判定即可，不影响通过）。
4. 双后端的 `.vmfb` 在 `local-sync` 上跑，数值与单后端一致。

**与 CUDA fatbin 的同构对照（这条必须写进你的记录）**：

| | CUDA fatbin | IREE executable |
|--|-------------|-----------------|
| 容器 | 一个 fatbin | 一个 `hal.executable` |
| 里面的多份实现 | 每个架构一份 cubin（`sm_70` / `sm_80` …）+ 可选 PTX | 每个 target 一个 `hal.executable.variant` |
| 选择时机 | 运行时按当前 GPU 架构挑，挑不到就 JIT PTX | 加载时按 target 与 `hal.executable.condition` 挑，**多个可用取第一个** |
| 挑不到会怎样 | 加载失败 | executable 加载失败 |
| 代价 | 产物膨胀（架构数 × 体积） | 产物膨胀（variant 数 × 体积） |

详见 [`./07-cuda-fatbin.md`](./07-cuda-fatbin.md) 与指南 [4.7](../learning-guides/iree-learning-guide.md#47-设备代码executable--variant--export)、[4.7.2](../learning-guides/iree-learning-guide.md#472-variant-是怎么选中的)。

**先预测再动手**：

- 双后端的 `hal.executable` 会变成 2 个，还是仍是 1 个但里面有 2 个 variant？这两种结果分别对应什么设计？
- `.vmfb` 体积会大约翻倍，还是只多一点？设备代码在产物的哪一部分（指南第 1 章的 `rodata`）？
- 运行时你只给了 `--device=local-sync`，它凭什么知道该用 `llvm-cpu` 那份而不是 `vmvx` 那份？

**验收命令**：

```bash
cd ~/iree-check
# 单后端
iree-compile chain.mlir --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --compile-to=hal -o chain.cpu.hal.mlir
iree-compile chain.mlir --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu -o chain.cpu.vmfb

# 双后端（旧写法：--iree-hal-target-backends=llvm-cpu --iree-hal-target-backends=vmvx）
iree-compile chain.mlir --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu,vmvx \
  --compile-to=hal -o chain.duo.hal.mlir
iree-compile chain.mlir --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu,vmvx -o chain.duo.vmfb

grep -c "hal.executable.variant" chain.cpu.hal.mlir chain.duo.hal.mlir
grep -o '#hal.executable.target<"[a-z-]*"' chain.duo.hal.mlir | sort -u
ls -l chain.cpu.vmfb chain.duo.vmfb
iree-dump-module chain.duo.vmfb | head -40      # 没有这个工具就跳过

iree-run-module --device=local-sync --module=chain.duo.vmfb --function=chain \
  --input=64xf32=-2.0 --expected_output=64xf32=4.0
```

**通过标准**：`chain.duo.hal.mlir` 的 `hal.executable.variant` 计数**严格大于**单后端版本；`#hal.executable.target<"llvm-cpu"` 与 `#hal.executable.target<"vmvx"` **两个字符串都能 grep 到**；`chain.duo.vmfb` 体积大于 `chain.cpu.vmfb`；双后端产物在 `local-sync` 上运行退出码为 0 且数值与单后端相同。
另外记录：你的版本是把双后端表达成「1 个 device、2 个 executable target」还是「2 个 `#hal.device.target`」——两种都合法，但含义不同（前者是一个设备的多份代码，后者是两个设备），在记录里写清你看到的是哪种。

**常见失败 → 说明你哪里没懂**：

| 现象 | 你哪里没懂 |
|------|-----------|
| `llvm-cpu,vmvx` 逗号写法不被接受 | 换成重复传两次 flag；以 `--help` 为准（这就是版本纪律） |
| 认为 variant 多了就要在运行时挑「最快的那个」 | 官方语义是**取第一个可用者**，不是择优。择优要靠 `condition` 与 export 的 fallback 链 |
| 说不出与 fatbin 的差别 | fatbin 的选择维度只有 GPU 架构；IREE 的 variant 维度是「后端 + 格式 + 任意配置属性」，表达力更宽 |
| 只看 `.vmfb` 大小不看 IR | 体积是间接证据，`hal.executable.variant` 的结构才是直接证据 |

---

### L2-IREE-09｜故意喂错 shape：从报错反推 ABI 契约

- **检验什么**：这条通过 = 你真的掌握了「`!hal.buffer_view` = buffer + shape + element type + encoding」，以及「ABI 边界上的检查是编译器插进去的 op，不是运行时的猜测」
- **前置**：L0-IREE-02、L0-IREE-03
- **资源**：本地+工具链
- **预计耗时**：1h

**任务**：用 `chain.vmfb`（签名 `tensor<64xf32>`）制造三种不同的错误输入，逐条读懂报错，然后回到 IR 里找出**是哪条 op 报的**。

| 错误输入 | 错在哪一维信息 |
|---------|--------------|
| `--input=32xf32=1.0` | shape 不匹配（元素数也不同） |
| `--input=8x8xf32=1.0` | **总字节数相同、rank 与 shape 不同**——这条最关键 |
| `--input=64xi32=1` | element type 不匹配 |

**先预测再动手**：三条都先写预测。

- `8x8xf32` 和 `64xf32` 的总字节数完全一样。IREE 会不会放行？如果不放行，说明 ABI 检查的是字节数还是结构？
- 这三条错误分别在什么时候被发现——host 侧解析 `--input` 时、进入函数后的检查 op、还是 kernel 真正读内存时？
- 报错信息里如果只说「buffer view 不匹配」而不说是哪个参数，你要靠什么定位？（提示：编译器在 abi 相位给每个参数插的检查 op 带 message）

**验收命令**：

```bash
cd ~/iree-check
iree-compile chain.mlir --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu -o chain.vmfb

for bad in "64xf32=-2.0" "32xf32=1.0" "8x8xf32=1.0" "64xi32=1"; do
  echo "=== --input=$bad"
  iree-run-module --device=local-sync --module=chain.vmfb --function=chain --input=$bad
  echo "exit=$?"
done

# 找出这些检查是谁插进去的（op 名以本地 dump 为准，通常是 hal.buffer_view.assert / hal.buffer.assert）
iree-compile chain.mlir --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --compile-to=abi -o chain.abi.mlir
grep -n "assert" chain.abi.mlir chain.hal.mlir 2>/dev/null
```

**通过标准**：第一条（正确输入）退出码 0，后三条退出码全部非 0；**三条错误信息互不相同**，且你能把每条映射到 shape / rank / element type 三者之一；`grep -n "assert"` 在 abi 或 hal 相位的 dump 里**至少命中一处**，并且你能指出这条检查 op 携带了哪几样信息（形状、元素类型、编码）。

**常见失败 → 说明你哪里没懂**：

| 现象 | 你哪里没懂 |
|------|-----------|
| 预测 `8x8xf32` 能跑通 | 把 buffer_view 当成裸内存了。它带 shape 与 dtype，**ABI 校验的是结构不是字节数** |
| 三条错误信息看起来一样 | 没读全输出。往上翻，检查 op 的 message 通常带参数序号 |
| 找不到 assert 类 op | 换 grep 关键字（`buffer_view` / `assert` / `expected`），或看更靠后的相位；也可能你的版本把检查放在 runtime 侧——那就记录下来，这是版本差异不是失败 |
| 认为报错是 kernel 抛的 | kernel 只认偏移和长度，它没有 shape 概念。检查发生在进入 kernel 之前的边界上 |

---

## L3 组：打通

### L3-IREE-10｜换到 GPU 后端：同一个模型，两套 target

- **检验什么**：这条通过 = 你真的掌握了「compiler target backend 与 runtime HAL device 的 N:M 关系」，以及「换后端改变的是 codegen 与 workgroup 划分，不改变 flow 层的 dispatch 分组逻辑」
- **前置**：L0-IREE-02、L2-IREE-08
- **资源**：**单卡GPU**（降级方案见下，降级后为 `本地+工具链`）
- **预计耗时**：半天

**任务**：把 `chain.mlir` 编到 CUDA（或 ROCm/HIP），在真卡上跑，与 CPU 版逐项对比：flow 相位的 dispatch 计数、hal 相位的 `#hal.executable.target` 字符串与 `count` region（workgroup 划分）、以及运行结果的数值。

架构 flag 版本差异大：新版通常是 `--iree-cuda-target=sm_80`，老版是 `--iree-hal-cuda-llvm-target-arch=sm_80`；ROCm 侧对应 `--iree-hip-target=gfx942` 一类。**先 `iree-compile --help | grep -iE "cuda|hip|rocm"` 查本地拼写**。

**先预测再动手**：

- `flow` 相位的 `flow.executable` 计数在 CPU 与 GPU 上会不会不同？dispatch 分组是在选定后端**之前**还是**之后**决定的？
- `hal.executable.export` 上的 `count` region（算 3D workgroup 数的那段）在两个后端上会不会不同？为什么 CPU 后端也要有 3D grid？
- 纯 f32 elementwise 在 CPU 与 GPU 上应该逐位相等还是只是接近？如果换成 matmul 或含归约的 `chain_reduce.mlir`，答案会变吗——为什么？

**验收命令（有卡）**：

```bash
cd ~/iree-check
iree-run-module --list_devices          # 先确认 cuda 这个 driver 真的在
iree-compile chain.mlir --iree-hal-target-device=cuda --iree-cuda-target=sm_80 \
  --compile-to=hal -o chain.cuda.hal.mlir
iree-compile chain.mlir --iree-hal-target-device=cuda --iree-cuda-target=sm_80 \
  -o chain.cuda.vmfb

grep -o '#hal.executable.target<"[a-z-]*"' chain.cuda.hal.mlir | sort -u
grep -c "flow.executable " chain.flow.mlir     # CPU 基线（L1-IREE-04）

iree-run-module --device=cuda --module=chain.cuda.vmfb --function=chain \
  --input=64xf32=-2.0 --expected_output=64xf32=4.0
echo "exit=$?"
# 可选：与 CPU 版做一次耗时对照（这么小的模型只能看到调用开销，别用它下性能结论）
iree-benchmark-module --device=cuda --module=chain.cuda.vmfb --function=chain --input=64xf32=-2.0
```

**通过标准（有卡）**：CUDA 版编译成功且 `--device=cuda` 运行退出码为 0，`--expected_output` 校验通过；`#hal.executable.target<"cuda"` 能 grep 到；CPU 与 GPU 两版的 `flow.executable ` 计数**相同**（证明划分与后端无关）；你能列出 hal 相位 IR 的至少三处差异（target 字符串、variant 内部 dialect、workgroup count 的取值）。

**降级方案（无卡，仍可完成本条的编译期部分）**：CUDA target 的**编译**不需要真卡，把上面命令里所有 `iree-run-module` / `iree-benchmark-module` 步骤去掉，只保留 `iree-compile` 与两份 hal dump 的对比。

- **降级下的通过标准**：`chain.cuda.vmfb` 生成成功；`#hal.executable.target<"cuda">` 存在；CPU 与 GPU 两份 hal dump 的差异清单写出来（至少三处）；`flow` 计数相同。
- **降级下拿不到的结论（必须在记录里写明）**：① 生成的 PTX 能否被真实 driver 加载；② GPU 上的数值是否真的与 CPU 一致；③ 任何性能与占用率结论；④ runtime 侧 CUDA HAL 的 semaphore / pending action 行为（指南 [4.6.2](../learning-guides/iree-learning-guide.md#462-为什么这个抽象是难的cuda-上的落地)）——这一条只能在真卡上观察。

**常见失败 → 说明你哪里没懂**：

| 现象 | 你哪里没懂 |
|------|-----------|
| `--device=cuda` 报找不到 driver | 编译成功不等于运行时可用；driver 需要 `libcuda.so`。这正好印证「compiler target 与 runtime device 是两件事」 |
| 架构 flag 报 unknown option | 没查本地 `--help`（版本纪律） |
| 预测 GPU 上 dispatch 会变多 | 把「怎么分组」和「怎么并行执行」混为一谈。分组在 flow 定，并行度在 codegen 与 workgroup count 定 |
| 用 64 元素的模型下性能结论 | 这个规模测的全是调用开销。要谈性能得换到有意义的规模 |

---

### L3-IREE-11｜从 `stream.timepoint` 到 `hal.fence`：把异步执行模型读出来

- **检验什么**：这条通过 = 你真的掌握了「invocation 是**排布**工作而不是执行工作」，以及编译期的时间线（timepoint）如何在 HAL 层兑现成 fence / semaphore 与 command buffer 的 wait/signal
- **前置**：L1-IREE-05（要一个至少两个 dispatch 的程序，用 `chain_reduce.mlir`）、L0-IREE-03
- **资源**：本地+工具链（**不需要 GPU**）
- **预计耗时**：半天

**任务**：对 `chain_reduce.mlir` dump `stream` / `hal` / `vm` 三个相位，画一张**时间线依赖表**：每个 `stream.cmd.execute` 等待哪个 timepoint、产出哪个 timepoint；这些 timepoint 在 hal 相位分别变成了哪个 fence 的 wait / signal；host 侧**唯一**真正阻塞的是哪条 op。

加分项（都以 `--help` 为准，不确定就跳过，不影响通过）：

- `--iree-stream-partitioning-favor=min-peak-memory` 与默认的 `max-concurrency` 各编一份，`diff` 两份 stream IR，看并发结构与 `stream.resource.alloca` 的位置有没有变化（指南 [3.4](../learning-guides/iree-learning-guide.md#34-stream-ordered-allocation峰值内存的关键)）。
- 若本地支持 `--iree-execution-model=async-external` 一类的 flag，对比函数签名：同步 ABI 与异步 ABI 的入口参数差在哪（异步版把 fence 直接暴露给调用方）。

**先预测再动手**：

- `chain_reduce` 有两个 dispatch 且第二个依赖第一个。这个依赖在 stream 相位是靠 SSA 数据流表达的，还是靠 timepoint 表达的？两者同时存在意味着什么？
- 到了 hal 相位，两个 dispatch 会被录进**一个** command buffer 还是两个？如果是一个，它们之间靠什么保证顺序（提示：`hal.command_buffer.execution_barrier` 是缓冲**内部**的依赖，`fence` 是提交之间的依赖）。
- `hal.device.queue.execute` 返回时，kernel 跑完了吗？那这个函数返回给用户时呢？用户读输出前必须先做什么？

**验收命令**：

```bash
cd ~/iree-check
for p in stream hal vm; do
  iree-compile chain_reduce.mlir --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --compile-to=$p -o chain_reduce.$p.mlir
done

grep -nE "stream.timepoint|await\(|stream.cmd.execute|stream.resource.alloca" chain_reduce.stream.mlir
grep -nE "hal.fence.create|hal.fence.await|hal.device.queue.execute|execution_barrier" chain_reduce.hal.mlir
grep -nE "vm.call @hal.fence|vm.call @hal.device.queue" chain_reduce.vm.mlir

# 加分项：分区策略对照
iree-compile chain_reduce.mlir --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-stream-partitioning-favor=min-peak-memory \
  --compile-to=stream -o chain_reduce.minmem.stream.mlir
diff chain_reduce.stream.mlir chain_reduce.minmem.stream.mlir | head -40
```

**通过标准**：三次 grep **各自至少命中一处**（stream 侧有 timepoint 与 await，hal 侧有 fence 与 queue.execute，vm 侧有 `vm.call @hal.*`）；你交出一张依赖表，行是每个 `stream.cmd.execute` / `hal.device.queue.execute`，列是 `wait` 与 `signal`，并且**能指出 host 唯一的阻塞点是哪一行**（通常是 `hal.fence.await`）；能用一句话说清「函数返回」与「计算完成」为什么不是同一件事。

**常见失败 → 说明你哪里没懂**：

| 现象 | 你哪里没懂 |
|------|-----------|
| 认为两个 dispatch 之间一定要各自 signal 一个 fence | 同一个 command buffer 内部用 barrier 就够；fence 是**提交批次之间**的定序原语。混淆这两级 = 没读懂 [4.4](../learning-guides/iree-learning-guide.md#44-工作表达command_buffer)/[4.5](../learning-guides/iree-learning-guide.md#45-工作提交device-queue-api) |
| 找不到 `hal.fence.await` | 可能你的程序用的是同步 ABI 且 await 被内联/改名。往下 dump 到 vm 相位再找 `vm.call @hal.fence.await` |
| 把 semaphore 当二值信号量 | 它是 64 位单调时间线，同一个值可被多次等待、wait 可以先于 signal 提交。这是 HAL 最硬的知识点（[4.6](../learning-guides/iree-learning-guide.md#46-同步timeline-semaphore-与-fencehal-的灵魂)） |
| `diff` 两份分区结果为空就认为 flag 没用 | 小程序对分区策略不敏感很正常。要么换一个含两条独立分支的模型再试，要么在记录里论证为什么这个程序不可能不同 |

---

## 本册条目 × 资源需求速查

| 条目 | 任务 | 级别 | 资源 | 耗时 | 需要申请机器？ |
|------|------|------|------|------|--------------|
| L0-IREE-01 | 工具链与 flag 自查 | L0 | 本地+工具链 | 0.5h | 否 |
| L0-IREE-02 | 最小 linalg 模型跑通并数值验证 | L0 | 本地+工具链 | 0.5h | 否 |
| L0-IREE-03 | `--compile-to` 逐相位 dump | L0 | 本地+工具链 | 2h | 否 |
| L1-IREE-04 | 两个 elementwise 串起来数 dispatch | L1 | 本地+工具链 | 1h | 否 |
| L1-IREE-05 | 插 reshape / transpose / reduction 打断融合 | L1 | 本地+工具链 | 2h | 否 |
| L1-IREE-06 | 改成动态 shape 看全链路传递 | L1 | 本地+工具链 | 1h | 否 |
| L2-IREE-07 | 把 pass 插进流水线（A 免构建 / B 需源码） | L2 | 本地+工具链（B 需一次完整源码构建） | 半天～1 天 | 否（B 需较大磁盘与构建机时） |
| L2-IREE-08 | 双后端 vmfb 与 executable variant | L2 | 本地+工具链 | 2h | 否 |
| L2-IREE-09 | 喂错 shape 反推 ABI 契约 | L2 | 本地+工具链 | 1h | 否 |
| L3-IREE-10 | CUDA/ROCm 后端编译并真卡运行 | L3 | **单卡GPU**（降级为 本地+工具链） | 半天 | **是**，1×NVIDIA GPU + CUDA 12.x |
| L3-IREE-11 | timepoint → fence，读异步执行模型 | L3 | 本地+工具链 | 半天 | 否 |

**只有 L3-IREE-10 需要 GPU**，且它给了「只编译不运行」的降级方案；降级后拿不到的四类结论已在该条目里列明。申请话术见总纲 [`./README.md`](./README.md) §6.2。

完成记录写在 `docs/notes/` 下，格式见总纲 §6.1：预测 vs 实际、卡在哪一步、验收命令与输出片段。
