# 环境配置

面向一个具体处境：**Ubuntu 服务器、没有 root、装不了 apt、已经有 miniforge。**

结论先说：这个处境不构成任何障碍。六个 lab 需要的东西里，只有 NVIDIA 驱动必须管理员装，
而有卡的机器上驱动早就装好了。**其余全部——包括 clang、opt、llc、mlir-opt、cmake、ninja，
甚至 nvcc——都能从 conda-forge 装进你自己的目录。**

```bash
bash setup.sh          # 一键装齐
bash setup.sh --check  # 随时体检
```

---

## 1. 为什么不需要 root

很多人以为 `clang`、`llvm-config` 这类"系统工具"只能 `apt install`。不是的。
conda-forge 把整套 LLVM/MLIR 都打包了，装到 `~/miniforge3/envs/xxx/bin/` 下，
和 `/usr/bin/` 里的版本互不干扰。

`linux-apt-packages.txt` 里那份 apt 清单，逐项都有 conda 等价物：

| 你需要的东西 | apt（要 root） | conda-forge（不要 root） |
|---|---|---|
| C/C++ 编译器 | `build-essential` | `cxx-compiler` |
| 构建系统 | `cmake` `ninja-build` | `cmake` `ninja` |
| LLVM 前端 | `clang-17` | `clang=17.0.6` |
| LLVM 工具与库 | `llvm-17` `llvm-17-dev` | `llvm=17.0.6` `llvmdev=17.0.6` |
| MLIR | `libmlir-17-dev` `mlir-17-tools` | `mlir=17.0.6` |
| 测试驱动 | `python3-lit` | `lit` |
| CUDA 编译工具 | `nvidia-cuda-toolkit` | `cuda-nvcc` `cuda-cuobjdump` |
| **NVIDIA 驱动** | `nvidia-driver-xxx` | **无解，必须管理员装** |

最后一行是唯一的硬边界。但要分清两件事：

- **驱动**（`libcuda.so`、`nvidia-smi`）——内核模块，只能管理员装
- **编译工具**（`nvcc`、`cuobjdump`）——普通用户态程序，conda 装得了

所以 `tvm-fatbin-lab` 的 fatbin 轨（只编译、不执行 kernel）**没有卡也能跑完**，
你只要 conda 里那个 `nvcc`。

还有一层原因和权限无关：**这台机器的 GLIBC 偏旧**，LLVM 官方和 Ubuntu 的预编译包
链接的 GLIBC 版本更新，装上也起不来。conda-forge 的包基于旧 sysroot 构建，
反而是兼容性最好的那个。也就是说，即便你有 root，走 conda 仍然是更省事的选择。

---

## 2. 一键脚本干了什么

`setup.sh` 分七步，每步都会打印在做什么：

1. **定位 conda** —— 依次试 `$CONDA_HOME`、当前激活的 base、`PATH` 上的 `conda`、
   以及 `~/miniforge3` 等常见位置。找不到就打印在家目录装 miniforge 的命令。
2. **建/更新环境** —— 默认叫 `mlir-env`。已存在则只补缺失的包，不重建。
3. **装编译工具链** —— `llvm` `llvmdev` `mlir` `clang` `clangxx` `lld` `cmake` `ninja`
   `lit` 等，全部锁 17.0.6。顺带把 `FileCheck` 从 `libexec/llvm/` 软链到 `bin/`
   （conda 的布局和各 lab 脚本的预期不一致，不链的话测试会找不到它）。
4. **CUDA 工具**（`--with-cuda`）—— `cuda-nvcc` `cuda-cuobjdump` `cuda-nvdisasm`。
5. **Python 库** —— 四个 lab 的 pip 依赖，**逐组安装**。
   `apache-tvm` 在新版 Python 上偶尔没有 wheel，逐组装可以让它失败时不拖垮其余五个 lab。
6. **验收体检** —— 逐个检查二进制和 Python 模块，缺什么列什么。
7. **写出 `.ai-infra-env.sh`** —— 如果你的 conda 路径或环境名和 lab 的内置默认值不同，
   会告诉你该 export 什么。

### 常用参数

```bash
bash setup.sh --check                # 只体检，不装
bash setup.sh --with-cuda            # 加装 conda 版 nvcc / cuobjdump
bash setup.sh --torch cu128          # RTX 50 系（sm_120）用这个
bash setup.sh --torch cpu            # 没有卡时
bash setup.sh --minimal              # 只装编译工具链，跳过 Python
bash setup.sh --env-name aiinfra     # 换环境名
PY_VERSION=3.10 bash setup.sh        # 换 Python 版本（TVM 装不上时试这个）
LLVM_VERSION=17 bash setup.sh        # 放宽 LLVM 版本（精确版本解不出来时）
```

---

## 3. 环境名为什么是 `mlir-env`

仓库里六个 lab 的 `scripts/env.sh` 都会去激活一个 conda 环境，默认值统一是 `mlir-env`：

| lab | 环境变量 | 是否硬依赖 conda |
|---|---|---|
| `llvm-hello-compile` | `LLVM_ENV` | 否，找不到退回 `PATH` |
| `mlir-toy-dialect` | `TOY_ENV` | **是**，找不到直接退出 |
| `tvm-fatbin-lab` | `TVM_ENV` | 否 |
| `onnx-delegate-lab` | `LAB_ENV` | 否 |
| `iree-lab` | `IREE_ENV` | 否 |
| `dist-train-lab` | `DIST_ENV` | 否 |

沿用 `mlir-env` 这个名字，装完之后**所有 lab 零配置直接可跑**。名字是历史遗留
（最早只装 MLIR），现在里面装的远不止 MLIR，但改名的收益抵不上要多配六个环境变量的麻烦。

如果你非要换名字，或者 miniforge 不在默认位置，`setup.sh` 会生成 `.ai-infra-env.sh`：

```bash
source .ai-infra-env.sh    # 每次开新终端
```

或者写进 `~/.bashrc` 一劳永逸。脚本最后会把该写的行直接打出来。

### 关于"一个环境装所有东西"

`setup.sh` 默认把 LLVM 工具链和四个 lab 的 Python 库装进同一个环境。理由是简单，
而且实测冲突风险很低——`apache-tvm`、`iree-base-compiler`、`torch` 的 wheel
都各自静态链接或自带 LLVM，不会去用 conda 里那份 `libLLVM-17.so`。

真遇到符号冲突（表现为 `import tvm` 时报 undefined symbol），拆成两个环境：

```bash
bash setup.sh --minimal --env-name mlir-env      # 只放编译工具链
bash setup.sh --env-name pylabs --torch cu128    # 只放 Python 库
# 然后
export LLVM_ENV=mlir-env TOY_ENV=mlir-env
export TVM_ENV=pylabs LAB_ENV=pylabs IREE_ENV=pylabs DIST_ENV=pylabs
```

---

## 4. 各 lab 的依赖与降级行为

各 lab 的脚本都做了缺依赖检测，缺什么就打印 `[SKIP]` 并说明原因，**不会静默失败**。
所以你可以先跑起来一部分，缺的慢慢补。

| lab | 必需 | 缺了会怎样 |
|---|---|---|
| `llvm-hello-compile` | `clang` `opt` `llc` | 缺任一 → `env.sh` 退出并报名字 |
| | `llvm-config` `cmake` | 缺 → 跳过自定义 Pass 插件，其余步骤照跑 |
| | `FileCheck` | 缺 → 测试跳过，退出码 0 |
| `mlir-toy-dialect` | conda + `mlir` + `ninja` | 缺 conda → 直接退出，提示跑 `setup.sh` |
| `tvm-fatbin-lab` | `apache-tvm` | 缺 → TVM 轨 `[SKIP]` |
| | `nvcc` `cuobjdump` | 缺 → fatbin 轨 `[SKIP]`，TVM 轨不受影响 |
| `onnx-delegate-lab` | `onnx` `onnxruntime` | 缺 → `[SKIP]` |
| | `executorch` | 缺 → 走概念模拟，这是预期路径 |
| `iree-lab` | `iree-base-compiler/runtime` | 缺 → `[SKIP]`，提示 pip 命令 |
| `dist-train-lab` | `torch` | 缺 → 无法运行 |
| | NVIDIA 驱动 + 多卡 | 缺 → CPU gloo 后端，13 条判据里能过 6 条 |

---

## 5. GPU 相关：RTX 50 系的两个坑

如果你在 8×RTX 5090 那台机器上跑 `dist-train-lab`，有两件事必须先确认。

### 坑一：torch 的 CUDA 版本必须 ≥ 12.8

RTX 5090 是 Blackwell，compute capability `sm_120`。CUDA 12.8 之前的 toolkit
根本不认识这个架构，装了旧版 torch，**前向传播第一次 launch kernel 时**才会报：

```
no kernel image is available for execution on the device
```

这个报错的位置很有迷惑性——环境检查、`torch.cuda.is_available()`、
`torch.cuda.device_count()` 全都正常，看起来一切就绪，直到真正跑起来才炸。

```bash
bash setup.sh --torch cu128
# 或手动
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

`setup.sh --check` 会主动检测这个组合并报警。

### 坑二：NCCL 必须 ≥ 2.26

更早的 NCCL 在 Blackwell 消费卡上初始化就会失败：

```
ncclMaxSharedMem ... exceeds device maxSharedMem
```

torch 的 wheel 自带 NCCL，确认版本：

```bash
python -c "import torch; print(torch.cuda.nccl.version())"
```

### 顺带：P2P 关闭是正常的，不是配错了

消费级 GeForce 卡没有 NVLink，而且驱动里 P2P 是关掉的。所以
`torch.cuda.can_device_access_peer()` 会全返回 False，卡间通信要绕道主机内存，
带宽从卡内的 ~1790 GB/s 掉到 ~43 GB/s，**差约 40 倍**。

这不是需要修的问题——`dist-train-lab` 整个 lab 就是围绕这道"通信墙"设计的，
`comm_bench.py` 量的就是这个断崖。详见 `dist-train-lab/README.md`。

---

## 6. 排障

### `setup.sh` 找不到 conda

```bash
CONDA_HOME=/你的/miniforge3 bash setup.sh
```

完全没装的话，miniforge 装家目录不需要 root：

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p "$HOME/miniforge3"
"$HOME/miniforge3/bin/conda" init bash && exec bash
```

### conda 解依赖卡住很久

`llvm=17.0.6` 这种精确版本 pin 会让求解变慢。两个办法：

```bash
conda install -n base -c conda-forge mamba   # 装 mamba，setup.sh 会自动优先用它
LLVM_VERSION=17 bash setup.sh                # 放宽到 17.x
```

### `apache-tvm` 装不上

TVM 的 wheel 对 Python 版本挑剔，新版本经常还没有构建。开一个 3.10 的环境给它：

```bash
PY_VERSION=3.10 bash setup.sh --env-name aiinfra310
export TVM_ENV=aiinfra310
```

只影响 `tvm-fatbin-lab`，其余五个 lab 不受影响。

### `mlir-toy-dialect` 报找不到 conda

它是六个 lab 里唯一硬依赖 conda 的（`scripts/env.sh` 找不到就 `exit 1`）。
确认 `CONDA_HOME` 指对了，且 `mlir-env` 里确实有 `mlir-opt`：

```bash
bash setup.sh --check
ls "$CONDA_HOME/envs/mlir-env/bin/mlir-opt"
```

### 构建 `mlir-toy-dialect` 时找不到 `FileCheck`

conda 把 `FileCheck` 放在 `libexec/llvm/` 而不是 `bin/`。`setup.sh` 会自动软链，
手动补的话：

```bash
ln -sf "$CONDA_HOME/envs/mlir-env/libexec/llvm/FileCheck" \
       "$CONDA_HOME/envs/mlir-env/bin/FileCheck"
```

### 磁盘不够

家目录配额紧的话，整套环境大约 8–12 GB（torch 和 iree-base-compiler 是大头）。
把 conda 的包缓存挪到大盘：

```bash
conda config --add pkgs_dirs /大盘/路径/conda-pkgs
conda clean -a          # 清掉已有缓存
```

装环境本身到别处：

```bash
conda create -p /大盘/路径/mlir-env python=3.11
CONDA_HOME=... ENV_NAME=... bash setup.sh
```

---

## 7. 装完之后

```bash
bash setup.sh --check                        # 确认状态

bash llvm-hello-compile/scripts/run.sh       # 不需要卡
bash mlir-toy-dialect/scripts/all.sh         # 不需要卡，首次构建几分钟
bash iree-lab/scripts/run.sh                 # 不需要卡
bash onnx-delegate-lab/scripts/run.sh        # 不需要卡
bash tvm-fatbin-lab/scripts/run.sh           # TVM 轨不需要卡
bash dist-train-lab/scripts/run_cpu_smoke.sh # 不需要卡，但要 torch
bash dist-train-lab/scripts/run_8gpu_wall.sh # 需要 ≥2 张卡
```

六条里五条不需要 GPU。学习路线的入口见 [`docs/README.md`](README.md)，
端到端的串联见 [`docs/learning-guides/00-end-to-end-pipeline.md`](learning-guides/00-end-to-end-pipeline.md)。
