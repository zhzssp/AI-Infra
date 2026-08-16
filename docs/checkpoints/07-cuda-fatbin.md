# 检验体系 07｜CUDA fatbin

> **对应学习文档**：[`../learning-guides/cuda-fatbin-learning-guide.md`](../learning-guides/cuda-fatbin-learning-guide.md)  
> **对应动手项目**：[`tvm-fatbin-lab/`](../../tvm-fatbin-lab/) 的 CUDA 轨（`scripts/run_fatbin.sh`）  
> **分级与资源标签定义**：[`./README.md`](./README.md)

本册检验的是这些知识点能不能落成命令：**`-gencode` 三个字段各管什么、虚拟架构（`compute_XX`）与真实架构（`sm_XX`）的分工、fatbin 作为多镜像容器的结构、PTX 退路与运行时 JIT、以及它与 IREE executable variant 的同构关系**。

**入门线**：L0 两条 + L1 至少一条 + **L2-FATBIN-05 与 L2-FATBIN-07 必做**。

## 先看这一段：本册主体不需要 GPU 卡

`tvm-fatbin-lab` 的 CUDA 轨**只编译、不执行**——`cuda/add.cu` 里只有一个 `__global__` kernel，没有 host 代码，从头到尾没有一次 kernel launch。所以：

| 你需要的 | 你不需要的 |
|---------|-----------|
| CUDA Toolkit（提供 `nvcc` + `cuobjdump`） | 一块真的 NVIDIA GPU |

`scripts/env.sh` 的 `have_cuda()` 也只探测这两个二进制在不在 `PATH` 里，不检查设备。**八条里有六条纯属编译期操作，现在就能做**；只有 L2-FATBIN-06 的「运行验收」那一半和 L3-FATBIN-08 需要真卡。

```bash
# 开工前自查
nvcc --version
cuobjdump --version
nvcc --list-gpu-arch    # 本地 nvcc 支持哪些架构，后面改 SM_A/SM_B 时按这个挑
```

> **版本纪律**：新版 CUDA 会**移除**对老架构的支持（例如 CUDA 12 起不再支持 `sm_35`），老版本又不认识新架构。本册出现的 `sm_75` / `sm_80` 只是 lab 的默认值，**凡是架构号都以 `nvcc --list-gpu-arch` 的输出为准**，报 `Unsupported gpu architecture` 就换一个在列表里的。

---

## L0 复现

### L0-FATBIN-01｜跑通 CUDA 轨，数清 fatbin 里装了几个镜像

- **检验什么**：这条通过 = 你真的掌握了「一个 fatbin 是多个架构镜像的容器」，并且会用 `cuobjdump` 打开它看
- **前置**：无
- **资源**：本地+工具链
- **预计耗时**：0.5h

**任务**：跑 `scripts/run_fatbin.sh`，然后**不看 `READING.md`**，自己从 `dump_sass_only.txt` 里数出：`add.fatbin` 装了几个 ELF 镜像、分别对应哪个 `sm_XX`、`-lptx` 那一段为什么是空的。数完再打开 `out/cuda/READING.md` 对答案。

**验收命令**：

```bash
cd tvm-fatbin-lab
bash scripts/run_fatbin.sh
cuobjdump -lelf out/cuda/add.fatbin
cuobjdump -lptx out/cuda/add.fatbin
```

**通过标准**：

- `-lelf` 列出**两个**不同 `sm_*` 的 ELF 镜像（默认是 `sm_75` 与 `sm_80`；若脚本回退过则是 `sm_70` 与 `sm_75`，看脚本打印的那行 `[fatbin] 使用架构 ...`）
- `-lptx` 对 `add.fatbin` **没有输出**（或不含对应档），你能说清原因：两次 `-gencode` 的 `code=` 都只给了 `sm_XX`，没要求嵌 PTX
- 你能说出 `add.fatbin` 的体积大致等于两份 SASS 之和，而不是一份

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 脚本 `[SKIP] 未找到 nvcc / cuobjdump` | CUDA Toolkit 没装或不在 PATH——注意这不代表你需要显卡 |
| 出现 `[WARN] 默认架构失败，尝试 sm_70 + sm_75` | 本地 nvcc 太老不认识 `sm_80`，属正常回退；后面所有条目请用回退后的架构号 |
| 以为 `-lptx` 空是出错了 | 没建立「PTX 嵌不嵌由 `code=` 决定」的认识 |

---

### L0-FATBIN-02｜对比两份 dump，说清 PTX 与 SASS 的分工

- **检验什么**：这条通过 = 你真的掌握了「虚拟架构产 PTX、真实架构产 SASS」，以及「带 PTX 退路」在部署上换来了什么
- **前置**：L0-FATBIN-01
- **资源**：本地+工具链
- **预计耗时**：0.5h

**任务**：并排读 `dump_sass_only.txt` 与 `dump_with_ptx.txt`，找出**唯一的结构差异**，并对回 `run_fatbin.sh` 里那一行不同的 `-gencode`：

```bash
# 第一个 fatbin（纯 SASS）
-gencode arch=compute_80,code=sm_80
# 第二个 fatbin（同一档额外嵌 PTX）
-gencode arch=compute_80,code=[sm_80,compute_80]
```

**先预测再动手**：

1. `code=[sm_80,compute_80]` 这个方括号里放了两样东西，产物里因此多了什么？`-lelf` 的输出会变吗？
2. 两个 fatbin 谁更大？大出来的部分是什么？
3. 如果一台机器上装的是比 `sm_80` **更新**的卡（比如 `sm_90`），这两个 fatbin 分别会发生什么？

**验收命令**：

```bash
cd tvm-fatbin-lab
diff out/cuda/dump_sass_only.txt out/cuda/dump_with_ptx.txt
ls -l out/cuda/add.fatbin out/cuda/add_with_ptx.fatbin
```

**通过标准**：能指出差异只在 `-lptx` 段（`add_with_ptx.fatbin` 多出一个 `compute_80` 的 PTX 镜像），`-lelf` 段两者相同；并能用一句话说清 PTX 退路的作用——**它不是为当前的卡准备的，是为将来的卡准备的**。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 说「PTX 是给低端卡用的」 | 方向反了。PTX 是前向兼容手段，给的是**比编译时更新**的架构 |
| 说不清 `arch=` 与 `code=` 的区别 | `arch=compute_XX` 是编译**中间目标**（PTX 按哪个虚拟架构生成），`code=` 是**最终装进 fatbin 的东西** |

---

## L1 改一处

### L1-FATBIN-03｜换架构组合，再造一个只有 PTX 的 fatbin

- **检验什么**：这条通过 = 你彻底搞清了 `-gencode arch=compute_XX,code=sm_YY` 里每个字段的作用，以及三种 `code=` 写法的产物差异
- **前置**：L0-FATBIN-02
- **资源**：本地+工具链
- **预计耗时**：1h

**任务**：分两步。

1. 用环境变量换一组架构（先用 `nvcc --list-gpu-arch` 挑两个存在的，如 70 与 86），重跑脚本，验证 dump 里的镜像随之改变。
2. 手工编一个**只有 PTX、没有任何 SASS** 的 fatbin：把 `code=` 只写 `compute_XX`。

**先预测再动手**：

1. 只给 `code=compute_80` 时，`cuobjdump -lelf` 会列出几个镜像？`-lptx` 呢？
2. 这样的 fatbin 在任何一张卡上都能用吗？代价是什么？什么时候付这个代价？
3. `arch=compute_75,code=sm_80` 这种「虚拟低、真实高」的组合合法吗？反过来 `arch=compute_80,code=sm_75` 呢？（想清楚 PTX 是由谁生成、SASS 由谁生成）

**验收命令**：

```bash
cd tvm-fatbin-lab
nvcc --list-gpu-arch                      # 先确认可用架构号
SM_A=70 SM_B=86 bash scripts/run_fatbin.sh
cuobjdump -lelf out/cuda/add.fatbin       # 应变成 sm_70 / sm_86

# 只留 PTX，不留 SASS
nvcc -fatbin cuda/add.cu -o /tmp/add_ptx_only.fatbin \
     -gencode arch=compute_75,code=compute_75
cuobjdump -lelf /tmp/add_ptx_only.fatbin  # 预期：没有 ELF 镜像
cuobjdump -lptx /tmp/add_ptx_only.fatbin  # 预期：一个 compute_75 PTX
```

**通过标准**：架构组合能被替换并在 dump 中体现；纯 PTX 的 fatbin 满足 `-lelf` 无输出且 `-lptx` 有一条；你能解释第 3 问中哪个组合非法以及为什么。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| `Unsupported gpu architecture 'compute_86'` | 本地 nvcc 版本不支持；没养成先查 `--list-gpu-arch` 的习惯 |
| 认为 `arch=compute_80,code=sm_75` 能编 | SASS 是从 PTX 往下编的，不能拿更高的虚拟架构去生成更低的真实架构代码 |
| 纯 PTX fatbin 里还能 `-lelf` 出东西 | 命令写错（漏了 `code=` 的改动），或看的是上一次的产物 |

---

### L1-FATBIN-04｜用一个高架构特性，逼出架构不兼容的报错

- **检验什么**：这条通过 = 你理解「架构号不只是标签，它决定了哪些指令/内建可用」，编译期就会拦下来
- **前置**：L0-FATBIN-01
- **资源**：本地+工具链
- **预计耗时**：1h

**任务**：复制一份 kernel（如 `/tmp/bf16_add.cu`），把加法改成需要较高架构的类型运算，然后**故意用低架构编译**，读报错。

推荐用 bf16（需 `sm_80`+）：

```cuda
#include <cuda_bf16.h>
extern "C" __global__ void add_bf16(const __nv_bfloat16 *a,
                                    const __nv_bfloat16 *b,
                                    __nv_bfloat16 *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = __hadd(a[i], b[i]);   // bf16 算术要求 sm_80 及以上
}
```

**若本地 CUDA 版本不支持 bf16**，换 `__half` + `<cuda_fp16.h>`（半精度算术要求 `sm_53`+），用 `sm_50` 去编同样能触发不兼容。

**先预测再动手**：

1. 报错会发生在**哪个阶段**——预处理、PTX 生成（`arch=`），还是 SASS 生成（`code=`）？
2. 如果只写 `-gencode arch=compute_80,code=compute_80`（纯 PTX，不生成 SASS），低架构的报错还会出现吗？为什么？
3. 这个错误和「运行时把 fatbin 加载到不匹配的卡上」是同一类问题吗？

**验收命令**：

```bash
# 低架构：预期失败
nvcc -fatbin /tmp/bf16_add.cu -o /tmp/bf16_low.fatbin \
     -gencode arch=compute_70,code=sm_70
# 匹配架构：预期成功
nvcc -fatbin /tmp/bf16_add.cu -o /tmp/bf16_ok.fatbin \
     -gencode arch=compute_80,code=sm_80
cuobjdump -lelf /tmp/bf16_ok.fatbin
```

**通过标准**：低架构那条命令失败并给出可读的错误（形态因版本而异，通常指向该内建/类型在目标架构不可用）；高架构那条成功且 dump 出 `sm_80` 镜像。你能把报错归位到「虚拟架构决定了可用指令集」这一层。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 低架构居然编过了 | 你的代码没真的用到高架构特性（比如只做了类型转换没做算术）；换成 `__hadd` 这类算术内建 |
| 报的是「找不到 cuda_bf16.h」 | CUDA 版本太老，按上面说明改用 `__half` 方案 |

---

## L2 加组件（主判据）

### L2-FATBIN-05｜再加一个 kernel，验证 fatbin 的多逻辑镜像结构

- **检验什么**：这条通过 = 你掌握了 fatbin 的两个正交维度：**同一 kernel 的多架构镜像** vs **同一镜像里的多个 kernel 符号**
- **前置**：L0-FATBIN-01
- **资源**：本地+工具链
- **预计耗时**：1.5h

**任务**：

1. 新增 `cuda/saxpy.cu`，写一个 `extern "C" __global__ void saxpy(float a, const float *x, const float *y, float *out, int n)`。
2. 扩展 `scripts/run_fatbin.sh`（或先手工跑通命令再回填脚本），产出两种 fatbin：
   - **分开打包**：`add.fatbin` 与 `saxpy.fatbin` 各自独立
   - **合并打包**：把两个 `.cu` 一起编成一个 `both.fatbin`
3. 用 `cuobjdump` 验证合并版里同时存在两个 kernel 符号。

**先预测再动手**：

1. 合并打包后，`-lelf` 列出的**镜像数**会翻倍吗？还是镜像数不变、每个镜像里多一个符号？（这是本条的核心问题）
2. `add` 用了 `extern "C"`。如果去掉它，`cuobjdump -symbols` 里看到的符号名会变成什么样？为什么 CUDA 例子里普遍加 `extern "C"`？
3. 两个 kernel 各自的 SASS 会互相影响吗（比如寄存器用量）？

**验收命令**：

```bash
cd tvm-fatbin-lab
SM_A=${SM_A:-75}; SM_B=${SM_B:-80}
GC="-gencode arch=compute_${SM_A},code=sm_${SM_A} -gencode arch=compute_${SM_B},code=sm_${SM_B}"

nvcc -fatbin cuda/saxpy.cu -o out/cuda/saxpy.fatbin $GC
nvcc -fatbin cuda/add.cu cuda/saxpy.cu -o out/cuda/both.fatbin $GC 2>/dev/null \
  || echo "若 nvcc 不接受多输入文件产单一 fatbin，改用下面的分别编译 + 各自 dump 方案"

cuobjdump -lelf out/cuda/both.fatbin
cuobjdump -symbols out/cuda/both.fatbin | grep -E 'add|saxpy'
```

> `nvcc -fatbin` 对多输入文件的处理在不同版本上表现不同。若合并命令失败，退而求其次：分别编译两个 fatbin，用 `cuobjdump -lelf` 逐个验证，并把「一个编译单元一个 fatbin」这个事实本身写进结论——这同样回答了第 1 问。

**通过标准**：

- `saxpy.fatbin` 的 `-lelf` 与 `add.fatbin` 结构一致（同样两档架构）
- 合并成功时，`-symbols` 能同时 grep 到 `add` 与 `saxpy`，且 `-lelf` 的**镜像数仍是 2**（按架构分，不按 kernel 分）
- 合并失败时，你在结论里说清了 fatbin 的打包边界是编译单元

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 以为加一个 kernel 就多两个镜像 | 混淆了「架构维度」与「符号维度」——镜像按架构切分，kernel 是镜像内部的符号 |
| grep 不到 `saxpy` | 忘了 `extern "C"`，符号被 C++ name mangling 改名了 |

---

### L2-FATBIN-06｜写一个真正会 launch 的 host 程序（编译与运行分开验收）

- **检验什么**：这条通过 = 你走通了「device 代码怎么被嵌进 host 可执行文件」这条链路，并分清了**编译期依赖**（Toolkit）与**运行期依赖**（真卡 + driver）
- **前置**：L2-FATBIN-05
- **资源**：**编译验收 = 本地+工具链；运行验收 = 单卡GPU**
- **预计耗时**：2h

**任务**：新增 `cuda/main.cu`：包含 host 侧的 `cudaMalloc` / `cudaMemcpy` / kernel launch / 结果校验，编成可执行文件，再用 `cuobjdump` 检查**可执行文件里嵌着的 fatbin 段**。

```cuda
// cuda/main.cu
#include <cstdio>
#include "add.cu"          // 直接包含 kernel 定义，避开分离编译

int main() {
  const int n = 1024;
  size_t bytes = n * sizeof(float);
  float *ha = (float*)malloc(bytes), *hb = (float*)malloc(bytes), *hc = (float*)malloc(bytes);
  for (int i = 0; i < n; ++i) { ha[i] = i; hb[i] = 2.0f * i; }

  float *da, *db, *dc;
  cudaMalloc(&da, bytes); cudaMalloc(&db, bytes); cudaMalloc(&dc, bytes);
  cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice);

  add<<<(n + 255) / 256, 256>>>(da, db, dc, n);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { printf("[run] 失败: %s\n", cudaGetErrorString(err)); return 1; }

  cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost);
  printf("[run] c[10]=%.1f (期望 30.0)\n", hc[10]);
  return 0;
}
```

**先预测再动手**：

1. 为什么这里用 `#include "add.cu"` 而不是像 C 那样声明外部函数？（提示：跨编译单元调用 `__global__` 函数需要 `-rdc=true` 与 device link 这一步）
2. 可执行文件里嵌的是 fatbin 还是纯 SASS？`cuobjdump` 能直接打开一个 ELF 可执行文件吗？
3. 在**没有卡**的机器上，这个程序编译会成功吗？运行会停在哪一步、报什么错？

**验收命令**：

```bash
cd tvm-fatbin-lab
SM_A=${SM_A:-75}; SM_B=${SM_B:-80}
nvcc cuda/main.cu -o out/cuda/add_app \
  -gencode arch=compute_${SM_A},code=sm_${SM_A} \
  -gencode arch=compute_${SM_B},code=[sm_${SM_B},compute_${SM_B}]

# 编译验收（无卡也能做）
cuobjdump -lelf out/cuda/add_app
cuobjdump -lptx out/cuda/add_app
cuobjdump -sass out/cuda/add_app | head -40

# 运行验收（需要真卡）
./out/cuda/add_app
```

**通过标准**（**两套独立判定，无卡时只做第一套**）：

| 验收 | 条件 | 需要卡 |
|------|------|--------|
| 编译验收 | `add_app` 生成成功；`cuobjdump -lelf` 在**可执行文件**里列出两档 SASS 镜像；`-lptx` 列出 `compute_XX` 退路；`-sass` 能读到真实汇编指令 | 否 |
| 运行验收 | 程序打印 `c[10]=30.0` | 是 |

**无卡时的降级**：只做编译验收，并把 `./out/cuda/add_app` 的实际报错记下来（典型是 `no CUDA-capable device is detected`）。**这条报错本身就是最好的教材**——它证明了 device 代码早已静静躺在可执行文件里，缺的只是运行时的设备与 driver。  
**降级后拿不到的**：kernel 的数值正确性、以及下一条 L3 的 JIT 计时。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 链接期报 `undefined reference to add` | 用了外部声明却没开 `-rdc=true`；不了解 device 代码的分离编译需要额外的 device link 步骤 |
| `cuobjdump` 对可执行文件报错 | 版本差异，先跑 `cuobjdump --help` 看本地支持哪些 dump 选项，再用列出来的写法重试 |
| 以为编译失败是因为没有 GPU | 编译从不需要卡——这正是本条要打掉的错误直觉 |

---

### L2-FATBIN-07｜从 fatbin 侧反向验证与 IREE variant 的同构

- **检验什么**：这条通过 = 你能把「多目标打包」这件事从 CUDA 提升成一个**通用的编译器结构**，而不是当作 NVIDIA 的特有细节
- **前置**：L0-FATBIN-02；建议已完成 [`./03-iree.md`](./03-iree.md) 的 L2-IREE-08
- **资源**：本地+工具链
- **预计耗时**：1h

**任务**：[`./03-iree.md`](./03-iree.md) 的 L2-IREE-08 已经从 IREE 侧做过一次对照（一个 `.vmfb` 里多个 `hal.executable.variant`）。本条要求你**从 fatbin 侧反向填一遍这张表**，然后与那一册的结论逐行核对，看两边说法能不能对上。

先自己闭卷填，再打开 `out/cuda/READING.md` 末尾的同构表与 IREE 那册对答案：

| 维度 | CUDA fatbin | IREE executable |
|------|-------------|-----------------|
| 容器 | 一个 `.fatbin` / 嵌进可执行文件的段 | 一个 `.vmfb` 里的 `hal.executable` |
| 内部单元 | 每个 `sm_XX` 一份 SASS 镜像 | 每个 target 一个 `hal.executable.variant` |
| 谁决定装哪几份 | 编译期的多次 `-gencode` | 编译期的 `--iree-hal-target-*` 后端列表 |
| 运行时怎么选 | driver 按 compute capability 匹配 | HAL 按 device 能力匹配 variant |
| 选不中怎么办 | PTX 退路 + 运行时 JIT | 换更通用的 variant / 编译期就报缺 |
| 谁承担代价 | 首次启动的 JIT 延迟 | 产物体积与编译时间 |

**先预测再动手**：

1. PTX 退路对应 IREE 的什么机制？IREE 有没有等价的「运行时把中间表示再编一次」的路径？（想想 CPU 后端与 JIT）
2. 两边的「选择键」抽象层次一样吗——compute capability 是一个数字，IREE 的 target 是一组 feature，哪种更容易做到前向兼容？
3. 如果让你给 IREE 设计一个「PTX 退路」，你会把什么东西存进 variant？

**验收命令**：

```bash
cd tvm-fatbin-lab
cuobjdump -lelf out/cuda/add_with_ptx.fatbin   # 多镜像容器的直接证据
cat out/cuda/READING.md                        # 末尾有同构表，对答案用
```

**通过标准**：你填的表与 IREE 那册的结论**没有互相矛盾**；能用一句话概括共同结构——「编译期把 N 份目标代码装进一个容器，运行期按设备能力挑一份，挑不中就退到更通用的形式」。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 把 PTX 类比成 IREE 的 MLIR 中间层 | 层次不对：PTX 是**发布产物的一部分**，MLIR 中间层不进最终产物 |
| 认为两者只是「都打了多份」 | 没抓住关键共同点是**运行期的选择机制**，而不是打包动作本身 |

---

## L3 打通

### L3-FATBIN-08｜在真卡上观测 PTX JIT 的首次启动代价

- **检验什么**：这条通过 = 你亲眼见到了「前向兼容不是免费的」，并知道这笔代价什么时候付、怎么被缓存摊掉
- **前置**：L2-FATBIN-06 的运行验收
- **资源**：**单卡GPU**
- **预计耗时**：1.5h

> **本册唯一必须申请机器的条目。** 申请话术见 [`./README.md`](./README.md#62-申请机器时可以直接发的话术)。前七条全部只需 CUDA Toolkit。

**任务**：在一张真卡（设其架构为 `sm_N`）上，编两个版本的 `add_app` 并计时对比首次运行：

- **版本 A（命中 SASS）**：`-gencode arch=compute_N,code=sm_N`，卡上有现成机器码
- **版本 B（只能 JIT）**：`-gencode arch=compute_M,code=compute_M`，其中 `M < N` 且只嵌 PTX，driver 必须现场把 PTX 编成 SASS

用 `CUDA_CACHE_DISABLE=1` 关掉 JIT 磁盘缓存，测「冷启动」；再关掉该变量重复测，观察缓存生效后的差异。

**先预测再动手**：

1. 版本 B 的首次运行会比 A 慢多少量级——百分之几，还是几十上百毫秒的固定开销？这个开销与 kernel 本身的计算量有关吗？
2. 不设 `CUDA_CACHE_DISABLE` 时，第二次运行 B 还慢吗？缓存存在哪里？
3. 一个内含上百个 kernel 的真实框架，如果全靠 PTX JIT 上线，用户会有什么体感？

**验收命令**：

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv   # 先确认这张卡是 sm_N

# 版本 A：命中 SASS（把 N 换成实际值，如 86）
nvcc cuda/main.cu -o /tmp/app_sass -gencode arch=compute_86,code=sm_86
# 版本 B：只有更低虚拟架构的 PTX，必须 JIT
nvcc cuda/main.cu -o /tmp/app_ptx  -gencode arch=compute_70,code=compute_70

export CUDA_CACHE_DISABLE=1
time /tmp/app_sass
time /tmp/app_ptx      # 预期明显更慢

unset CUDA_CACHE_DISABLE
time /tmp/app_ptx      # 第一次填缓存
time /tmp/app_ptx      # 预期回落，接近 app_sass
```

**通过标准**：

- 两个版本都打印 `c[10]=30.0`（**数值必须一致**——JIT 不改变语义，只改变何时生成机器码）
- `CUDA_CACHE_DISABLE=1` 下，`app_ptx` 的耗时显著高于 `app_sass`，且差值表现为一个**与数据规模无关的固定开销**
- 取消该变量并预热一次后，`app_ptx` 的耗时明显回落

**降级方案（无卡）**：只做编译侧——用 L1-FATBIN-03 的方法造出纯 PTX fatbin，用 `cuobjdump -lelf` 证明里面确实没有 SASS，据此**论证**运行时必须 JIT。  
**降级后拿不到的**：JIT 的真实耗时量级、缓存前后的差值、以及「开销与计算量无关」这个只能靠计时看出来的结论。

**常见失败 → 说明你哪里没懂**：

| 现象 | 盲点 |
|------|------|
| 版本 B 直接报错跑不起来 | `arch=compute_M` 选得比卡还新，或写成了 `code=sm_M` 导致没有可用镜像也没有 PTX |
| 两个版本耗时看不出差别 | 没关缓存（此前已跑过一次，SASS 已被缓存），或计时被程序其余部分淹没 |
| 以为 JIT 会让结果变化 | JIT 只影响机器码何时生成，不影响语义 |

---

## 条目 × 资源需求速查

| 编号 | 任务 | 级别 | 资源 | 耗时 |
|------|------|------|------|------|
| L0-FATBIN-01 | 跑通 CUDA 轨，数清镜像数 | L0 | 本地+工具链 | 0.5h |
| L0-FATBIN-02 | 对比两份 dump，说清 PTX/SASS 分工 | L0 | 本地+工具链 | 0.5h |
| L1-FATBIN-03 | 换架构组合 + 造纯 PTX fatbin | L1 | 本地+工具链 | 1h |
| L1-FATBIN-04 | 用高架构特性逼出不兼容报错 | L1 | 本地+工具链 | 1h |
| **L2-FATBIN-05** | **加第二个 kernel，验证多镜像结构** | L2 | 本地+工具链 | 1.5h |
| L2-FATBIN-06 | 写 host 程序（编译/运行分开验收） | L2 | 编译：本地+工具链<br>运行：**单卡GPU** | 2h |
| **L2-FATBIN-07** | **与 IREE variant 反向对照** | L2 | 本地+工具链 | 1h |
| L3-FATBIN-08 | 真卡观测 PTX JIT 首启代价 | L3 | **单卡GPU** | 1.5h |

**合计约 9 小时。其中 7.5 小时不需要任何 GPU**——只有 L2-FATBIN-06 的运行验收（半小时）与 L3-FATBIN-08 需要真卡，且两者都给了降级方案，不做也不挡入门线。

**要申请机器的话**：这两条加起来两小时机时、单卡任意架构即可，可以和 [`./03-iree.md`](./03-iree.md) 的 L3、[`./04-tvm.md`](./04-tvm.md) 的 L3 拼成同一次申请一起做完。

**下一步**：本册与 [`./03-iree.md`](./03-iree.md) 讲的是同一件事的两种实现；[`./04-tvm.md`](./04-tvm.md) 用的是同一个 lab 的另一条轨（TVM 轨，纯 CPU），两轨互不依赖，可以任意顺序做。
