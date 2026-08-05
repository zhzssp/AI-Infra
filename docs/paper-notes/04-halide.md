# Halide：算法与调度分离范式的奠基之作

> **论文元信息**
> - 标题：*Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines*
> - 作者/机构：Jonathan Ragan-Kelley, Andrew Adams, Saman Amarasinghe（MIT CSAIL）；Connelly Barnes, Sylvain Paris（Adobe）；Frédo Durand（MIT CSAIL）
> - 发表：PLDI 2013（ACM SIGPLAN Conference on Programming Language Design and Implementation）
> - 项目主页：https://halide-lang.org ；DOI：10.1145/2491956.2462176

## 1. 它解决什么问题

图像处理流水线（image processing pipeline）本质上是大量 stencil（模板/邻域）计算拼成的图：每个 stage 只对上一 stage 的输出做一次局部邻域访问（不像科学计算里的 iterated stencil 要在同一个 grid 上迭代成百上千次），但 stage 数量可以多达几十到上百个（论文里 local Laplacian filters 有 99 个 stage）。这类流水线的性能特点是：**逐 stage 的计算强度（arithmetic intensity）很低，但 stage 图又深又宽**，因此朴素实现和手工优化实现之间往往有一个数量级以上的性能差。

手工榨取性能需要同时做三件相互冲突的事：

1. **局部性（locality）**：让生产者（producer）尽快把结果交给消费者（consumer），避免中间结果在内存里来回搬运（带宽瓶颈）。
2. **并行度（parallelism）**：多线程 + SIMD 向量化，要求各次迭代之间尽量没有依赖。
3. **避免冗余重计算（redundant recomputation）**：stencil 的邻域访问会让多个消费者共享同一个生产者的值，若不缓存就会被重复计算。

这三者互相拉扯：越激进地融合（fusion）以提升局部性，越容易引入跨迭代依赖（丧失并行度）或重复计算共享值；越追求并行度、把每个 stage 完整算完再交给下一个（breadth-first），局部性就越差。**手写代码把"算什么"（算法逻辑）和"怎么算/何时算/存哪"（并行化、tiling、局部性、重计算的取舍）耦合在同一份 C/CUDA/intrinsics 代码里**，导致：

- 代码不可移植：换一个架构（x86→ARM→GPU）几乎要重写整个循环嵌套；
- 代码不可组合：把两个已经调优好的 stage 拼接起来，融合优化（跨 stage 的 loop fusion）几乎必须推翻重写；
- 调优空间不可搜索：因为策略和逻辑写在一起，"试一种新的 tiling 策略"意味着手动重写几百行嵌套循环，人类只能探索空间中极少的几个点。

Halide 的核心解法：把 **算法定义（algorithm）** 和 **调度决策（schedule）** 在语言层面彻底解耦——算法用纯函数式 DSL 描述"每个坐标点的值是什么"，调度用一套独立的原语描述"这些值何时何地被计算、存储、复用"，二者通过编译器组合，从同一份算法定义合成出覆盖整个权衡空间的、任意一点的高性能实现。这正是后来 TVM 的 Tensor Expression + schedule primitive 直接继承的思想源头。

## 2. 整体运行框架

```
        Halide DSL (嵌入 C++ 的前端)
              |
     +--------+--------+
     |                 |
  Algorithm          Schedule
  (Func/Var/Expr,     (domain order: split/reorder/
   RDom 定义)          tile/vectorize/parallel/unroll;
     |                 call schedule: compute_at/store_at)
     |                 |
     +--------+--------+
              |
        [编译器 Lowering 流水线]
              |
  4.1 Loop Synthesis        —— 从 output 递归向上，按 schedule 的
      (loop nest 综合)          domain order 和 call schedule 生成
                                符号化的循环嵌套骨架
              |
  4.2 Bounds Inference      —— 区间分析（interval analysis），从
      (边界推断)                下游反推每个 stage 每个维度的
                                min/extent，回填符号边界
              |
  4.3 Sliding Window /      —— 用区间分析识别"存储粒度 > 计算粒度"
      Storage Folding           的场景，收缩重复计算区间、折叠
                                循环缓冲区
              |
  4.4 Flattening            —— 多维坐标 -> 线性 buffer 下标
              |
  4.5 Vectorize / Unroll    —— 常量 extent 的向量/unroll 循环替换
      为 ramp/broadcast IR      为单条向量表达式或展开的代码
              |
  4.6 CodeGen (LLVM)        —— CPU: x86 SSE/AVX、ARM NEON
                                GPU: 生成 CUDA kernel + host 管理代码
              |
        可执行 pipeline（C ABI 函数，可 JIT 或 AOT）
```

各组件输入/输出与职责：

- **Algorithm（算法层）**：程序员只写"每个输出像素/张量元素的值是什么函数"，不写循环、不写内存布局。产物是一张 `Func` 之间的调用图（DAG，可以有 `RDom` 归约边）。
- **Schedule（调度层）**：程序员（或 autotuner）对图中每个 `Func` 独立标注：它的循环该怎么拆分/重排/并行/向量化（**domain order**），以及它相对调用者的循环嵌套，在哪一层被计算、在哪一层被存储（**call schedule**，即 `compute_at`/`store_at` 的语义来源）。Schedule 不改变语义，只改变性能。
- **Lowering（综合）**：编译器**从输出 Func 开始，沿调用图反向递归**：先按 output 的 domain order 生成一层循环骨架，再把每个被调用的 `Func` 的计算代码插入到 call schedule 指定的那层循环体的开头；同时在 call schedule 指定的层级插入其分配语句。**这一步决定了最终 loop nest 的形状完全是 schedule 的函数**——同一个算法在不同 schedule 下会展开出结构完全不同的循环嵌套。
- **Bounds Inference（边界推断）**：Lowering 阶段的循环边界和分配大小先留作符号变量，之后从 output 所需的区域开始，用**区间分析**沿调用图反向传播："消费者在某个循环层访问生产者哪个下标表达式" → 反推生产者在该层需要覆盖的最小区间。这一步保证所有分配和循环都"刚好够大、可终止"，且完全由 schedule 决定注入点，编译器本身不做启发式判断。
- **CodeGen（代码生成）**：把综合出的、已经过 flatten/向量化的 IR 一一映射到 LLVM IR，CPU 后端产出多线程 + SIMD 的目标码，GPU 后端把标注了 block/thread 维度的循环抽取成 CUDA kernel，外面套上 host 侧的 launch、拷贝、同步代码。

## 3. 核心概念与核心特性逐条拆解

### 3.1 算法（algorithm）：纯函数式、无副作用的 Func 定义

**是什么**：Halide 里图像/张量不是可变数组，而是从整数坐标域到值的**纯函数**（`Func`），定义域是无限整数域上的某个逻辑区域。一个 pipeline 就是一条 `Func` 调用链/DAG。函数体（`Expr`）只能是算术/逻辑运算、条件表达式、对输入图像的采样、对其他 `Func` 的调用、`let` 绑定，全部无副作用。

**为什么这样设计**：如果算法定义里混入了"先算 A 再算 B""把结果存到这个数组里"这类顺序/存储信息，调度和算法就无法真正解耦——调度器要改变计算顺序或存储粒度时，就必须重写算法本身。而纯函数式定义只描述"值是什么"，完全不描述"何时/何地被算出"，这就为调度层留出了自由变换的空间：只要某个坐标点的依赖已经就位，编译器可以在保证正确性的前提下，把它的计算挪到调用图中的任意合法位置。

**带来什么能力**：
- 天然数据并行：函数定义域内任意两点互不依赖（除非通过归约显式引入顺序），调度器可以随意选择并行化维度；
- 无限域 + 编译器自动裁剪访问区域，使边界条件（guard band）可以被安全、高效地处理，不需要程序员手写边界特判；
- 算法本身可复用/可组合：把两个 `Func` 拼接、插入新 stage，都不会波及调度决策。

### 3.2 调度（schedule）的三个正交维度

论文把调度分解为两大类原语，本质上覆盖三个正交的决策维度：

**(a) 存储粒度 / 计算粒度（call schedule：compute_at / store_at）**

**是什么**：对调用图里的每一条"调用边"（某 Func 被谁调用），需要回答两个问题——这个 Func 的值在调用者的哪一层循环被**计算**（compute granularity），又在哪一层被**分配存储**（storage granularity）。存储粒度必须等于或粗于计算粒度（值总得先有地方放）。这正是真实 Halide API 里 `compute_root()` / `compute_at(caller, loop_var)` / `store_root()` / `store_at(caller, loop_var)` 的语义来源。

**为什么这样设计**：如果只能"存储粒度=计算粒度"（像传统 loop fusion 那样"用即算、算完即弃"），就无法表达"提前算好但不是全局提前算好、只提前一点点"的中间策略——例如 sliding window：值在第一次被需要时计算，但要跨越后续若干次循环迭代持续可见以便复用。把"何时算"和"存多久"拆成两个独立旋钮，才能表达这种中间地带。

**带来什么能力**：三个极端点 + 无穷多中间点：
- compute/store 都在最外层（root）→ breadth-first，各 stage 完整算完再交给下一 stage；
- compute/store 都在最内层（调用点）→ total fusion，随算随弹，可能重复计算共享值；
- store 在粗粒度、compute 在细粒度、中间隔着一层**串行**循环 → sliding window，值只算一次但跨迭代复用，代价是引入了循环间依赖（丧失该维度的并行性）。

**(b) 循环变换（domain order：split / reorder / tile / fuse / unroll / vectorize / parallel）**

**是什么**：对每个 Func 自己的定义域循环，决定遍历顺序和展开方式：`split(x, factor)` 把一维拆成 outer×factor+inner 两维；`reorder` 调整维度顺序（如 row-major ↔ column-major）；`tile` 是两次 `split` + `reorder` 的组合糖；`fuse` 把两个循环维合并成一个（常用于把二维 block 索引拍平成一维方便并行调度）；`vectorize`/`unroll` 把某个常量 extent 的维度替换成向量指令/展开代码；`parallel` 标记某维用线程池并行。

**为什么这样设计**：这些都是经典循环变换，但关键是 Halide **允许对每个 Func 独立选择**，且因为算法层保证了数据并行（无隐藏的顺序依赖），任意维度都可以自由地并行/向量化/重排，不需要像通用编译器那样先做复杂的依赖分析证明变换合法——合法性由函数式语义直接保证（除非涉及归约）。

**带来什么能力**：Split 之后递归地对 outer/inner 分别再 split，就能表达任意层次的 tiling；split + vectorize 内层就是 SIMD 化；split + parallel outer 就是多线程分块；这套原语组合出的空间覆盖了几乎所有实践中出现的手工循环变换模式。

**(c) 重计算 vs 存储复用（recompute / store-and-reuse / sliding window）**

**是什么**：这其实是 (a) 的具体后果，但值得单独强调——当 storage 粒度粗于 compute 粒度、且中间有串行循环时，编译器会自动做 **sliding window optimization**：用区间分析算出"当前迭代所需区间"减去"此前迭代已经算过的区间"，只计算新增的那一小片，历史结果留在（可能被 **storage folding** 缩小成环形缓冲区的）分配空间里继续复用。

**为什么这样设计**：如果没有这一步，"store 粗、compute 细"这种 schedule 组合会天然导致每次都要重新计算整个所需区间（因为不知道哪些已经算过），从而毫无意义。区间分析使编译器可以精确知道"上一次已经覆盖到哪"，从而只计算增量。

**带来什么能力**：在 blur 例子里，sliding window 让 `blur_x` 每个值恰好算一次（`work amplification = 1.0×`），同时把生产者-消费者的最大复用距离从"整幅图"降到"stencil 半径"（局部性达到该场景理论最优），代价是牺牲了这一维度上的并行性（循环必须串行执行）。

### 3.3 调度空间的四象限权衡

论文用一张表（对应原文 Fig. 3，以两级 blur 为例）定量刻画四个指标之间此消彼长的关系：

| 策略 | 并行跨度 span（可并行的迭代数） | 最大复用距离 reuse distance（衡量局部性，数值越小越好） | 冗余计算倍数 work amplification |
|---|---|---|---|
| Breadth-first（完全分离） | 全图规模 | 全图规模（很差） | 1.0× |
| Full fusion（完全融合，逐点重算） | 全图规模 | 3×3（极好） | 2.0×（每个共享值被重复算） |
| Sliding window（滑窗复用） | 仅一维/一条扫描线 | 3+3（好） | 1.0×（无冗余） |
| Tiled（重叠分块） | 全图规模 | tile 内尺度（好） | 1.0625×（仅 tile 边界有少量重算） |
| Sliding window within tiles（分块内滑窗） | tile 数量级 | 3+3 | 1.25× |

**结论**：任何单一极端策略都至少在一个指标上很差（要么局部性差、要么并行度差、要么冗余计算多，第四个隐含指标是内存占用——breadth-first 需要为每个中间 stage 分配全尺寸 buffer，而 sliding window/tiled 只需分配一个 stencil halo 大小的窄条）。**性能最好的调度几乎总是混合策略**：先按并行需求把域切成 tile（获得跨 tile 的并行性），再在 tile 内部用 sliding window 或完全融合（获得 tile 内的局部性），用"tile 边界处一点点冗余计算"换取"内部完全消除冗余、保留局部性"的双重收益。这正是 Halide schedule 语言要能同时、独立地表达"跨 tile 并行"和"tile 内部 compute_at/store_at 复用策略"的根本原因。

### 3.4 Reduction / RDom（更新定义）与它的调度

**是什么**：纯函数无法表达 histogram、卷积求和、扫描（scan）这类"依赖自身之前的值"的计算。Halide 用 **reduction** 表达：先给一个初始值函数，再给一个"递归更新规则"，更新规则在一个显式声明的 **RDom（reduction domain）** 上按字典序（lexicographic order）依次应用。例如直方图：

```cpp
RDom r(0, in.width(), 0, in.height());
Func histogram;
Var i;
histogram(i) = 0;
histogram(in(r.x, r.y)) += 1;
```

**为什么这样设计**：这是把"有副作用的迭代计算"限制在一个受控的、边界显式声明的子语言里，而不是让整个 Halide 退化为命令式语言。RDom 显式声明迭代边界，使得归约仍然可以纳入统一的区间分析和边界推断框架。

**带来什么能力**：RDom 维度**只有在更新运算满足结合律（associative）时才允许被重排或并行化**；否则必须保持声明顺序串行执行。这一条件检查把"归约的可调度性"变成了调度器可以静态判断的属性，而不是留给程序员猜。自由变量维度（非 RDom 的维度）仍可以像纯函数一样任意调度。

### 3.5 Bounds Inference 与区间分析

**是什么**：编译器从输出 Func 需要的区域开始，**沿调用图反向递归**，对每个 Func 的每个维度，用调用者中索引它的表达式做**区间分析（interval analysis）**，反推出该维度需要覆盖的 `[min, max]`，再把这些符号边界作为循环层的前导代码（preamble）注入回正向的 loop nest。

**为什么这样设计**：论文特意放弃了更强大的 polyhedral model（多面体分析只能处理仿射约束、轴对齐或凸多面体区域），换成表达能力更弱但**适用范围更广**的区间分析——区间分析可以穿透几乎任意表达式（算术、条件、超越函数、甚至内存读取的值经过 `clamp` 后），而 polyhedral 模型对非线性/数据依赖的下标（Halide 里很常见，比如 bilateral grid 的 data-dependent gather）束手无策。

**带来什么能力**：这一权衡让 Halide 可以对"任意 Halide 表达式"做边界推断，覆盖更广的算法类别（包括数据依赖访问、直方图归约），代价是只能表达轴对齐的矩形区域调度，不能表达一般的多面体（在图像处理场景里这个代价通常可接受）。同时，区间分析的边界推断结果直接驱动了 sliding window 优化和 storage folding（复用同一套机制）。

### 3.6 自动调度（autoscheduler）：论文当时的方案

**是什么**：论文估计 local Laplacian filters 这样的 pipeline 的合法 schedule 数量下界是 $10^{720}$，不可能穷举、也超出了 polyhedral 优化器能处理的规模。论文用**遗传算法（genetic algorithm）**做随机搜索：固定种群规模（128 个体/代），每代由精英保留（elitism）、两点交叉（crossover，交叉点选在不同 Func 之间）、变异（mutation）、随机个体四部分构成；非法 schedule（如把某维 compute/store 到调用者根本不存在的循环层）直接拒绝重采样。

**为什么这样设计**：调度空间维度极高且各维度间存在复杂的全局依赖（一个 Func 的调度选择会影响它所有调用者/被调用者的合法调度集合），传统基于梯度或精确解析模型的优化方法不适用；遗传算法对这种离散、高维、有大量局部极小值的空间有较好的鲁棒性（论文承认这是从 PetaBricks 的 autotuner 直接借用的思路）。

**带来什么能力**：论文额外加入了两类"领域知识变异规则"（如"loop fusion 规则"——把某 Func 调度成 tile + 向量化，并递归地把它的被调用者也 compute/store 到 tile 内层）以及"从模板库采样常见调度模式"，显著加速收敛。多数案例在几小时到两天内、几十到上百代收敛到最终性能的 85% 以内。论文也坦诚指出：这套暴力自动调优在**鲁棒性和易用性**上还远不成熟——调优是开发流程里额外的一整阶段，对噪声敏感，容易陷入局部极小需要重启，变异规则依赖领域知识、难以泛化到新的算法结构。这个"手写模板 + 搜索"的痛点，正是后来 TVM 的 AutoTVM/Ansor 要继续解决的问题。

### 3.7 GPU 调度映射（gpu_blocks / gpu_threads）

**是什么**：GPU 执行被建模成"把某些循环维度标注为 parallel，并打上 GPU block / thread 维度的标签"，本质上复用同一套调度原语，只是加了硬件约束：block 和 thread 这一串循环必须**连续**（中间不能插入其他普通循环，因为一次 kernel launch 对应一个多维、tile 化的并行循环嵌套），且 GPU kernel 循环之间**不能互相嵌套**（当时的 GPU 不支持嵌套数据并行），thread 维度的 extent 不能超过硬件限制。

**为什么这样设计**：不为 GPU 设计一套全新的调度语言，而是让同一套 `compute_at`/`store_at`/`split`/`tile` 语义直接映射到 GPU 的 block/thread 层级——这样 CPU 版和 GPU 版调度之间的差异仅仅体现在"给哪些循环打 block/thread 标签"，其余（tiling 策略、fusion 边界）完全共享同一套推理与代码生成基础设施。

**带来什么能力**：改一行 schedule 就能把同一套算法从"几十个 CPU 向量化循环"变成"几十个互相融合/独立的 CUDA kernel + host 端内存管理与同步代码"的完全不同图结构，且编译器自动插入设备内存分配与"仅在需要时才拷贝"的懒惰拷贝逻辑。论文用这一点在 bilateral grid 上找到了一个反直觉的 GPU 调度（牺牲部分并行度换取更少的同步开销），比专家手写的 370 行 CUDA 还快 2.3×。

## 4. 使用示例

### 4.1 3×3 box blur：同一算法，四种调度

**算法定义**（与调度完全无关，只写一次）：

```cpp
Func blur_x, blur_y;
Var x, y;

// 两个 3x1 通道分离的盒式滤波（分离实现 3x3 卷积）
blur_x(x, y) = (input(x-1, y) + input(x, y) + input(x+1, y)) / 3;
blur_y(x, y) = (blur_x(x, y-1) + blur_x(x, y) + blur_x(x, y+1)) / 3;
```

**调度 1：naive / breadth-first（compute_root）**

```cpp
blur_x.compute_root();
blur_y.compute_root();
```

等价伪循环：

```c
alloc blur_x[H][W];
for (y = 0; y < H; y++)
  for (x = 0; x < W; x++)
    blur_x[y][x] = (in[y][x-1] + in[y][x] + in[y][x+1]) / 3;

alloc blur_y[H-2][W];
for (y = 0; y < H - 2; y++)
  for (x = 0; x < W; x++)
    blur_y[y][x] = (blur_x[y][x] + blur_x[y+1][x] + blur_x[y+2][x]) / 3;
```

局部性最差：`blur_x` 整幅图必须先算完、写回内存，`blur_y` 再整幅重新读入；无冗余计算，并行度最高（两个循环各自完全数据并行）。

**调度 2：完全融合（compute_at 到最内层，无存储复用）**

```cpp
blur_x.compute_at(blur_y, x);
```

等价伪循环：

```c
alloc blur_y[H-2][W];
for (y = 0; y < H - 2; y++)
  for (x = 0; x < W; x++) {
    float bx[3];                       // 每个 (x,y) 都重新算这 3 个值
    for (int i = 0; i < 3; i++)
      bx[i] = (in[y+i][x-1] + in[y+i][x] + in[y+i][x+1]) / 3;
    blur_y[y][x] = (bx[0] + bx[1] + bx[2]) / 3;
  }
```

局部性最好（生产即消费，中间值只活在寄存器里），但相邻输出像素之间共享的 `blur_x` 值被反复重算，`work amplification ≈ 2.0×`。

**调度 3：sliding window（store 粗、compute 细，中间串行）**

```cpp
blur_x.store_root().compute_at(blur_y, y);
```

等价伪循环（`blur_x` 只分配 3 行的环形缓冲区，靠 storage folding 得到）：

```c
alloc blur_y[H-2][W];
alloc blur_x[3][W];                    // 折叠成 3 行环形缓冲
for (y = -1; y < H - 1; y++) {
  for (x = 0; x < W; x++)
    blur_x[y % 3][x] = (in[y][x-1] + in[y][x] + in[y][x+1]) / 3;
  if (y < 1) continue;
  for (x = 0; x < W; x++)
    blur_y[y-1][x] = (blur_x[(y-2)%3][x] + blur_x[(y-1)%3][x] + blur_x[y%3][x]) / 3;
}
```

每个 `blur_x` 值恰好算一次（`work amplification = 1.0×`），复用距离从"整幅图"降到"3 行"（接近理论最优局部性），但 `y` 这一维必须串行（丧失该维并行性）。

**调度 4：tiled + vectorized + parallel（生产实践中最常见的形态）**

```cpp
Var xo, yo, xi, yi;
blur_y.tile(x, y, xo, yo, xi, yi, 256, 32)
      .vectorize(xi, 8)
      .parallel(yo);
blur_x.compute_at(blur_y, xo)
      .vectorize(x, 8);
```

等价伪循环：

```c
parallel for (yo = 0; yo < H/32; yo++) {
  for (xo = 0; xo < W/256; xo++) {
    alloc blur_x[34][256];             // 32 行 + 2 行 halo，宽度一个 tile
    for (y = -1; y < 33; y++)
      vectorized(8) for (x = 0; x < 256; x++)
        blur_x[y][x] = (in[...][x-1] + in[...][x] + in[...][x+1]) / 3;
    for (yi = 0; yi < 32; yi++)
      vectorized(8) for (xi = 0; xi < 256; xi++)
        blur_y[yo*32+yi][xo*256+xi] =
            (blur_x[yi][xi] + blur_x[yi+1][xi] + blur_x[yi+2][xi]) / 3;
  }
}
```

跨 tile 并行（`parallel yo`）+ tile 内部向量化（`vectorize 8`）+ tile 边界处两行 halo 的少量冗余计算（`work amplification ≈ 1.06×`），是并行度、局部性、冗余计算三者的一个"甜点"折中——这正是 3.3 节表格里 Tiled 那一行对应的真实调度写法。

### 4.2 GPU schedule 示例

```cpp
Var bx, by, tx, ty;
blur_y.gpu_tile(x, y, bx, by, tx, ty, 16, 16);   // 16x16 的 block/thread 划分
blur_x.compute_at(blur_y, bx);                    // blur_x 在每个 GPU block 内重新计算
```

- `gpu_tile` 一步完成"先 `tile` 再把 outer 标为 block 维、inner 标为 thread 维"，等价于把 `blur_y` 的输出切成 16×16 的 tile，每个 tile 对应一个 CUDA block，block 内 256 个 thread 各自负责一个像素。
- `blur_x.compute_at(blur_y, bx)` 让 `blur_x` 在每个 block 开始时于（概念上的）shared memory 里重新计算一份该 block 所需的 halo 区域，避免每个 thread 各自去全局内存重复读取相邻像素——这与 CPU 版 `compute_at(blur_y, xo)` 是同一个 API、同一套 lowering 逻辑，只是外面套了 block/thread 的执行模型。

### 4.3 JIT realize 与 AOT Generator 用法

**JIT（适合调试/脚本场景，编译在 `realize()` 调用时触发）**：

```cpp
#include "Halide.h"
using namespace Halide;

int main() {
    Buffer<uint8_t> input = Tools::load_image("input.png");

    Var x("x"), y("y");
    Func blur_x, blur_y;
    blur_x(x, y) = (input(x - 1, y) + input(x, y) + input(x + 1, y)) / 3;
    blur_y(x, y) = (blur_x(x, y - 1) + blur_x(x, y) + blur_x(x, y + 1)) / 3;

    blur_y.tile(x, y, x, y, 256, 32).vectorize(x, 8).parallel(y);
    blur_x.compute_at(blur_y, x).vectorize(x, 8);

    // JIT 编译 + 立即在给定尺寸上执行，返回结果 Buffer
    Buffer<uint8_t> output = blur_y.realize({input.width(), input.height()});
    Tools::save_image(output, "output.png");
    return 0;
}
```

**AOT（面向部署，用 Generator 把算法+schedule 提前编译成 `.a`/`.o`，运行时零编译开销）**：

```cpp
// blur_generator.cpp
#include "Halide.h"
using namespace Halide;

class BlurGenerator : public Generator<BlurGenerator> {
public:
    Input<Buffer<uint8_t>>  input{"input", 2};
    Output<Buffer<uint8_t>> output{"output", 2};

    void generate() {
        Var x, y, xi, yi;
        Func blur_x;
        blur_x(x, y) = (input(x - 1, y) + input(x, y) + input(x + 1, y)) / 3;
        output(x, y) = (blur_x(x, y - 1) + blur_x(x, y) + blur_x(x, y + 1)) / 3;

        output.tile(x, y, xi, yi, 256, 32).vectorize(xi, 8).parallel(y);
        blur_x.compute_at(output, x).vectorize(x, 8);
    }
};

HALIDE_REGISTER_GENERATOR(BlurGenerator, blur)
```

```bash
# 编译出 generator 可执行文件（链接 Halide 库本身）
g++ blur_generator.cpp $(pkg-config --cflags --libs Halide) -o blur.generator

# 用 generator 针对目标架构 AOT 生成 blur.a / blur.h
./blur.generator -g blur -o . target=host

# 业务代码直接静态链接生成的目标文件，运行期无需 Halide 编译器/LLVM
g++ main.cpp blur.a -lpthread -ldl -o main
```

- JIT 路径适合快速迭代算法与 schedule、做算法验证；
- AOT 路径是生产部署的常规姿势：把"编译期"完全放在构建阶段，运行期产物是一个不依赖 Halide/LLVM 运行时的普通目标文件，这与 TVM 的 `relay.build` → 导出 `.so` 部署模型是同一思路。

## 5. 关键实验结论

论文在 quad-core Xeon W3520 + NVIDIA Tesla C2070 上，对 5 个真实图像处理应用（2~99 个 stage）做了"autotuner 找到的最优 Halide 调度"对"领域专家手写并调优过的 C/intrinsics/CUDA 实现"的对比：

- **性能全面超过专家手写实现**：CPU 上 1.2×~4.4×，GPU（CUDA）上 2.3×~9×；其中最复杂的 local Laplacian filters（99 stage）在 CPU 上快 1.7×，在 GPU 上快 7.5×~9×——说明当调度空间足够大、足够正确地建模了权衡关系时，**系统化搜索能找到人类专家凭直觉也难以想到的调度点**（论文明确提到 bilateral grid 的 GPU 调度对原作者是"反直觉"的）。
- **代码量减少一个数量级左右**：Halide 算法代码普遍是专家实现的 1/2 到 1/18（例如 blur 只用 2 行 vs. 35 行手写 SSE intrinsics；bilateral grid CUDA 版 34 行 vs. 370 行手写 CUDA）——说明算法/调度分离不仅带来性能收益，还极大降低了"写出可优化实现"的工程成本，把优化专业知识从"程序员脑子里"迁移到"编译器 + 调度语言 + 搜索算法"里。
- **同一份算法可以自动映射到 CPU 和 GPU 两种截然不同的调度**（甚至混合 CPU+GPU，如 local Laplacian filters 生成了 58 个不同的 CUDA kernel），且性能都达到或超过对应硬件上的专家实现——说明调度与算法解耦真正实现了**可移植性**：换硬件不需要重写算法，只需要重新搜索/编写 schedule。
- **调度空间极其庞大不可枚举**（估计 local Laplacian filters ≥ $10^{720}$），遗传算法搜索通常在几小时到两天内收敛到最终性能的 85% 以内；调优出的 schedule 对**中等程度**的分辨率/架构变化比较鲁棒（Fig. 8 的跨分辨率交叉测试显示从低分辨率调优、迁移到高分辨率通常只慢 1.0×~2×），但**极端**变化（如 GPU 调度直接搬到 CPU）会导致数量级的性能下降（7× 慢）——说明 schedule 不是与硬件/规模完全无关的抽象，它编码了针对特定硬件特征（cache 大小、SIMD 宽度、核数）和特定问题规模的具体决策，这也是后续 AutoTVM/Ansor 要用 cost model 取代"一次调优、到处复用"假设的动机之一。
- **论文也坦诚指出自动调优本身工程上不成熟**：搜索是开发流程里额外增加的一整个阶段，对测试环境噪声敏感，容易陷入局部极小值需要多次重启，变异规则携带领域知识、难以自动泛化——这些问题在后来的 AI 编译器（AutoTVM 依赖手写模板、Ansor 用进化搜索+学习的 cost model）里被持续改进，但至今仍是"调度自动化"这条技术路线尚未完全解决的痛点。

## 6. 与 TVM / MLIR 的衔接

**Halide → TVM**：TVM 的 Tensor Expression（TE）几乎是 Halide `Func`/`Expr` 的直接移植——`te.compute()` 定义"每个输出坐标的值是什么"，纯函数式、无副作用；TVM 早期代码库直接 fork 了 Halide 的 IR 基础设施，`tvm.te.Schedule` 上的 `split` / `reorder` / `tile` / `fuse` / `vectorize` / `parallel` / `unroll` / `compute_at` / `compute_root` / `compute_inline` / `cache_read`/`cache_write` 这些 schedule primitive 在命名和语义上都与本文的 domain order、call schedule 一一对应，`cache_read`/`cache_write` 相当于把 Halide 的"存储粒度"选择显式暴露为一个独立原语。TVM 把 lowering 之后的循环嵌套表示叫 **TIR（Tensor IR）**，对应本文里 lowering→bounds inference→flattening 产出的那一层命令式 IR。而"schedule 空间巨大、需要搜索"这一论文核心洞察，在 TVM 里演化成两代自动调度器：**AutoTVM** 需要人为给每个算子写调度模板（模板里留几个可调参数，机器学习 cost model 做贝叶斯/进化搜索选参数），**Ansor（Auto-Scheduler）** 则进一步取消了手写模板，直接从计算定义出发用规则自动生成大量候选调度骨架，再对细粒度参数做进化搜索——这几乎就是 Halide 遗传算法 autotuner 思路在算子级别、配合学习型 cost model 之后的规模化延伸。

**Halide → MLIR**：本文"loop nest 由 schedule 综合而来、边界由区间分析推断"的思想，在 MLIR 里主要体现在 **affine dialect**（`affine.for`/`affine.if` 携带的仿射约束，是比 Halide 区间分析更强的多面体式表达，配合 `-affine-loop-tile`、`affine loop fusion` pass 实现 tiling 与 fusion）和 **linalg dialect**（`linalg.generic` 用声明式方式描述"计算是什么"，天然对应 Halide 的 algorithm 层；`linalg` 上的 tiling/fusion transform 对应 schedule 层）。更进一步，MLIR 的 **transform dialect** 把"调度"本身也表示成一段可以被解析、组合、持久化、复用的 IR/脚本（而不是像 Halide/TVM 那样是宿主语言里调用的一组 API），这是对"算法与调度分离"这一范式的又一次升华：调度从"程序员在 C++/Python 里敲的一串方法调用"变成了"一等公民的、可被编译器本身分析和变换的数据"。

**Halide 范式的局限**（这也是它之后一系列工作要解决的问题）：
- 调度空间的搜索/设计仍然需要专家经验或昂贵的自动搜索（遗传算法/进化搜索/学习型 cost model），没有从根本上解决"谁来产生好调度"这个问题，只是把它从"人写循环"降级为"人（或机器）写调度决策"；
- 区间分析只能表达轴对齐矩形区域，对**动态 shape**（例如 LLM 推理里变长序列、动态 batch）和**数据依赖的控制流/稀疏结构**（如 MoE 的路由、稀疏 attention）支持天然偏弱——这些场景里循环边界本身依赖运行时数据，超出了静态区间推断能处理的范围，需要运行时符号化 shape、动态 tensor 或专门的稀疏编译技术来补充；
- Halide 的调度原语面向"稠密规则网格上的 stencil/element-wise/reduction"设计，对不规则的图结构、动态控制流（比如注意力机制里 KV cache 的分页管理、算力网场景下跨设备的通信调度）没有直接的原语可用，这些正是 TVM 之后、面向大模型的运行时（vLLM 的 PagedAttention、Ansor/MetaSchedule 之后的动态 shape 支持）要专门解决的问题。

## 7. 学习这篇论文时的最小必要集

面向"后续做 AI 编译器/多后端异构算力基础设施"的目标，必须吃透的 5-8 个点：

1. **`compute_at`/`store_at`（call schedule）的语义**：它们决定了某个中间结果的计算代码和分配语句被注入到调用者 loop nest 的哪一层，这是理解"融合"（fusion）在编译器层面到底是什么操作的关键，也是 TVM `compute_at` 的直接来源。
2. **`split`/`tile`/`reorder`/`fuse`/`vectorize`/`unroll`/`parallel`（domain order）对生成 loop nest 形状的实际影响**：要能在脑内把一段 schedule 直接翻译成等价的伪代码循环嵌套（本文档第 4.1 节的四个例子就是练习这个能力）。
3. **四象限权衡**（局部性 / 并行度 / 冗余计算 / 内存占用）及其量化指标（span、reuse distance、work amplification）——这是判断任何一个具体调度"好在哪、代价是什么"的分析框架，不限于 Halide，可直接搬到任何 tiling/fusion 决策上。
4. **sliding window + storage folding 背后的区间分析机制**：理解"存储粒度粗于计算粒度 + 串行中间循环"如何被编译器转译成"只计算增量区间"，这是局部性优化最核心的机制之一。
5. **Bounds inference 的反向递归过程**：从输出反推每一层的循环边界和分配大小，理解为什么这个过程保证了"生成的代码永远安全、永远终止"，以及为什么论文选区间分析而非 polyhedral model（表达力 vs 适用范围的权衡）。
6. **Reduction/RDom 的顺序语义与结合律条件**：理解归约为什么天生比纯函数难并行化，以及编译器如何静态判断"这个归约维度能不能重排/并行"。
7. **自动调度问题的本质**（调度空间巨大不可枚举 → 需要搜索/学习型方法），以及这个问题在 TVM AutoTVM/Ansor 里的后续演化路径，为学习这两个系统打好概念基础。
8. **GPU 调度映射的硬件约束**（block/thread 循环必须连续、不可嵌套 kernel launch）：理解调度原语如何在不改变 API 的前提下适配一个完全不同的执行模型，这对理解后续任何"一套 IR/调度语言打多个后端"的系统（TVM、MLIR）都是必要的心智模型。

可以先跳过的内容（不影响后续学 TVM/MLIR 的主线）：
- 各个案例算法本身的图像处理数学细节（demosaicking 具体算法、bilateral grid 的三线性插值公式、local Laplacian filters 的金字塔构造细节）——只需知道它们是"stage 数量多、结构复杂"的压力测试案例即可；
- 遗传算法的具体交叉/变异规则实现细节，以及与 PetaBricks autotuner 的具体渊源——知道"用启发式搜索解决高维离散调度空间"这个结论就够了，具体算法在 TVM 里已经被 cost-model-guided 搜索取代；
- GPU 代码生成里的具体 peephole 优化、SIMD intrinsic 映射规则等后端底层实现细节——这些是编译器工程实现问题，不影响理解调度范式本身。
