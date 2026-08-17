#!/usr/bin/env bash
# =============================================================================
# AI-Infra —— 一键环境配置（**不需要 root，不需要 apt**）
# -----------------------------------------------------------------------------
# 全仓库六个动手项目的依赖，除 NVIDIA 驱动外全部可以装在你自己的 conda 环境里：
#
#   系统级工具（clang / opt / llc / mlir-opt / cmake / ninja / lit）→ conda-forge
#   Python 库（tvm / onnx / iree / torch）                          → pip 装进同一环境
#   CUDA 编译工具（nvcc / cuobjdump）                                → conda-forge（可选）
#
# 唯一装不了的是 NVIDIA 驱动本身——那个必须由管理员装。但驱动通常已经装好了
# （能跑 nvidia-smi 就有），本脚本只补它上面的工具链。
#
# ---------- 用法 ----------
#   bash setup.sh                    # 默认：编译工具链 + 全部 Python lab
#   bash setup.sh --check            # 只体检，什么都不装
#   bash setup.sh --with-cuda        # 额外装 conda 版 nvcc/cuobjdump（fatbin 轨用）
#   bash setup.sh --torch cu128      # 装 Blackwell（RTX 50 系 / sm_120）用的 torch
#   bash setup.sh --torch cpu        # 装 CPU 版 torch（无卡时够用）
#   bash setup.sh --minimal          # 只装编译工具链，跳过 Python lab
#   bash setup.sh --env-name myenv   # 换环境名（默认 mlir-env，与各 lab 的默认值一致）
#
# ---------- 为什么默认叫 mlir-env ----------
# 仓库里六个 lab 的 scripts/env.sh 都默认去激活 `mlir-env`。沿用这个名字，
# 装完之后**所有 lab 零配置直接可跑**。名字是历史遗留，里面装的不只是 MLIR。
# =============================================================================
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- 参数 ----------
ENV_NAME="${ENV_NAME:-mlir-env}"
PY_VERSION="${PY_VERSION:-3.11}"
LLVM_VERSION="${LLVM_VERSION:-17.0.6}"
DO_CHECK_ONLY=0
DO_CUDA=0
DO_PYLABS=1
TORCH_FLAVOR="none"

need_value() {
  # $1=选项名 $2=传进来的值（可能为空）
  [[ -n "${2:-}" && "${2:-}" != -* ]] || {
    echo "参数 $1 需要一个取值，例如：$1 $3" >&2; exit 1; }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)      DO_CHECK_ONLY=1; shift ;;
    --with-cuda)  DO_CUDA=1; shift ;;
    --minimal)    DO_PYLABS=0; shift ;;
    --torch)      need_value --torch    "${2:-}" cu128
                  TORCH_FLAVOR="$2"; shift 2 ;;
    --env-name)   need_value --env-name "${2:-}" mlir-env
                  ENV_NAME="$2"; shift 2 ;;
    --python)     need_value --python   "${2:-}" 3.11
                  PY_VERSION="$2"; shift 2 ;;
    # 打印文件头注释块：从第 2 行起，遇到第一个非注释行即停（改动文件不会错位）
    -h|--help)    awk 'NR>1{ if (/^#/) { sub(/^# ?/,""); print } else exit }' \
                      "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "未知参数：$1（用 --help 看用法）"; exit 1 ;;
  esac
done

# ---------- 输出辅助 ----------
OK="  [ok]   "; BAD="  [MISS] "; WARN="  [warn] "; INFO="  [info] "
FAILED_ITEMS=()
note_fail() { FAILED_ITEMS+=("$1"); }

sec() {
  echo
  echo "═══════════════════════════════════════════════════════════════"
  echo "▶ $*"
  echo "═══════════════════════════════════════════════════════════════"
}

# =============================================================================
# 一、定位 conda
# =============================================================================
sec "① 定位 conda"

detect_conda_home() {
  # 顺序：显式 CONDA_HOME → 当前已激活的 base → PATH 上的 conda → 常见安装位置
  if [[ -n "${CONDA_HOME:-}" && -x "${CONDA_HOME}/bin/conda" ]]; then
    echo "${CONDA_HOME}"; return 0
  fi
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    local base; base="$(dirname "$(dirname "${CONDA_EXE}")")"
    [[ -x "${base}/bin/conda" ]] && { echo "${base}"; return 0; }
  fi
  local c
  c="$(command -v conda 2>/dev/null || true)"
  if [[ -n "${c}" ]]; then
    local base
    base="$("${c}" info --base 2>/dev/null || true)"
    [[ -n "${base}" && -x "${base}/bin/conda" ]] && { echo "${base}"; return 0; }
  fi
  local p
  for p in "${HOME}/miniforge3" "${HOME}/mambaforge" "${HOME}/miniconda3" \
           "${HOME}/anaconda3" "/data/mm64/hanishzheng/miniforge3" \
           "/opt/miniforge3" "/opt/conda"; do
    [[ -x "${p}/bin/conda" ]] && { echo "${p}"; return 0; }
  done
  return 1
}

CONDA_HOME="$(detect_conda_home || true)"
if [[ -z "${CONDA_HOME}" ]]; then
  cat <<'EOF'
  [MISS] 没找到 conda。

  你已经有 miniforge 的话，请显式指出它在哪，然后重跑：
      CONDA_HOME=/你的/miniforge3 bash setup.sh

  还没装的话（**不需要 root，装在家目录即可**）：
      curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
      bash Miniforge3-$(uname)-$(uname -m).sh -b -p "$HOME/miniforge3"
      "$HOME/miniforge3/bin/conda" init bash && exec bash
EOF
  exit 1
fi
export CONDA_HOME
echo "${OK}conda 根目录 = ${CONDA_HOME}"

# shellcheck disable=SC1091
eval "$("${CONDA_HOME}/bin/conda" shell.bash hook)"

# mamba 比 conda 的求解器快很多，有就用
SOLVER="conda"
if command -v mamba >/dev/null 2>&1; then
  SOLVER="mamba"
  echo "${INFO}检测到 mamba，将用它求解依赖（比 conda 快）"
fi

ENV_PREFIX="${CONDA_HOME}/envs/${ENV_NAME}"
env_exists() {
  conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"
}

# =============================================================================
# 体检函数（--check 与安装后验收共用）
# =============================================================================
run_doctor() {
  local prefix="$1"
  local bin="${prefix}/bin"
  local rc=0

  echo
  echo "── 编译工具链（llvm-hello-compile / mlir-toy-dialect）──"
  local t
  for t in clang clang++ opt llc llvm-config mlir-opt mlir-tblgen cmake ninja lit FileCheck; do
    if [[ -x "${bin}/${t}" ]]; then
      case "${t}" in
        clang|opt|llc)
          printf "%s%-14s %s\n" "${OK}" "${t}" "$("${bin}/${t}" --version 2>/dev/null | head -1)" ;;
        llvm-config)
          printf "%s%-14s LLVM %s\n" "${OK}" "${t}" "$("${bin}/${t}" --version 2>/dev/null)" ;;
        *)
          printf "%s%-14s %s\n" "${OK}" "${t}" "${bin}/${t}" ;;
      esac
    elif [[ "${t}" == "FileCheck" && -x "${prefix}/libexec/llvm/FileCheck" ]]; then
      # conda 把 FileCheck 放在 libexec/llvm 而非 bin，各 lab 脚本只看 bin。
      printf "%s%-14s 已安装但未软链到 bin（跑一次 bash setup.sh 自动修）\n" "${WARN}" "${t}"
    else
      printf "%s%-14s 未安装\n" "${BAD}" "${t}"
      note_fail "${t}"
      rc=1
    fi
  done

  echo
  echo "── Python 库 ──"
  local py="${bin}/python"
  if [[ ! -x "${py}" ]]; then
    echo "${BAD}python 未安装在该环境里"
    return 1
  fi
  printf "%s%-14s %s\n" "${OK}" "python" "$("${py}" --version 2>&1)"

  local mod desc
  while IFS='|' read -r mod desc; do
    [[ -z "${mod}" ]] && continue
    if "${py}" -c "import ${mod}" >/dev/null 2>&1; then
      local ver
      ver="$("${py}" -c "import ${mod} as m; print(getattr(m,'__version__',''))" 2>/dev/null)"
      printf "%s%-14s %s\n" "${OK}" "${mod}" "${ver}"
    else
      printf "%s%-14s 未安装  ← %s\n" "${BAD}" "${mod}" "${desc}"
      note_fail "python:${mod}"
      rc=1
    fi
  done <<'MODS'
numpy|全部 lab 共用
tvm|tvm-fatbin-lab TVM 轨
onnx|onnx-delegate-lab
onnxruntime|onnx-delegate-lab
iree.compiler|iree-lab（pip 包名 iree-base-compiler）
iree.runtime|iree-lab（pip 包名 iree-base-runtime）
torch|dist-train-lab
MODS

  echo
  echo "── 命令行入口 ──"
  for t in iree-compile iree-run-module torchrun; do
    if [[ -x "${bin}/${t}" ]]; then
      printf "%s%-16s %s\n" "${OK}" "${t}" "${bin}/${t}"
    else
      printf "%s%-16s 未安装\n" "${BAD}" "${t}"
    fi
  done

  echo
  echo "── CUDA（只有 tvm-fatbin-lab 的 fatbin 轨与多卡实验需要）──"
  local nvcc cuobjdump
  nvcc="$( [[ -x "${bin}/nvcc" ]] && echo "${bin}/nvcc" || command -v nvcc 2>/dev/null || true)"
  cuobjdump="$( [[ -x "${bin}/cuobjdump" ]] && echo "${bin}/cuobjdump" || command -v cuobjdump 2>/dev/null || true)"
  if [[ -n "${nvcc}" ]]; then
    printf "%s%-16s %s\n" "${OK}" "nvcc" "$("${nvcc}" --version 2>/dev/null | tail -2 | head -1)"
  else
    printf "%s%-16s 未安装（fatbin 轨会被跳过；补装：bash setup.sh --with-cuda）\n" "${WARN}" "nvcc"
  fi
  [[ -n "${cuobjdump}" ]] \
    && printf "%s%-16s %s\n" "${OK}" "cuobjdump" "${cuobjdump}" \
    || printf "%s%-16s 未安装\n" "${WARN}" "cuobjdump"

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
               --format=csv,noheader 2>/dev/null | sed 's/^/  [gpu]  /' || true
    if "${py}" -c "import torch" >/dev/null 2>&1; then
      "${py}" - <<'PY' 2>/dev/null || true
import torch
n = torch.cuda.device_count()
print(f"  [gpu]  torch 可见 {n} 张卡；torch={torch.__version__} cuda={torch.version.cuda}")
if n:
    p = torch.cuda.get_device_properties(0)
    print(f"  [gpu]  cuda:0 {p.name}  sm_{p.major}{p.minor}")
    if (p.major, p.minor) >= (12, 0) and torch.version.cuda:
        major, _, minor = torch.version.cuda.partition(".")
        if (int(major), int(minor or 0)) < (12, 8):
            print("  [MISS] 这张卡是 sm_120（Blackwell），但 torch 是 CUDA "
                  f"{torch.version.cuda} 构建 —— 跑 kernel 会报 no kernel image。")
            print("         修复：bash setup.sh --torch cu128")
if n >= 2:
    peer = any(torch.cuda.can_device_access_peer(i, j)
               for i in range(n) for j in range(n) if i != j)
    print(f"  [gpu]  P2P = {'可用' if peer else '全部关闭（消费级 GeForce 的典型情况）'}")
PY
    fi
  else
    echo "${INFO}没有 nvidia-smi，按无 GPU 处理（六个 lab 里五个不需要卡）"
  fi

  return ${rc}
}

# =============================================================================
# --check：只体检
# =============================================================================
if [[ ${DO_CHECK_ONLY} -eq 1 ]]; then
  sec "② 体检（--check，不安装任何东西）"
  if ! env_exists; then
    echo "${BAD}conda 环境 '${ENV_NAME}' 不存在。先跑一次不带 --check 的 setup.sh。"
    exit 1
  fi
  echo "${INFO}环境前缀 = ${ENV_PREFIX}"
  run_doctor "${ENV_PREFIX}"
  rc=$?
  echo
  if [[ ${rc} -eq 0 ]]; then
    echo "全部就绪。"
  else
    echo "有 ${#FAILED_ITEMS[@]} 项缺失：${FAILED_ITEMS[*]}"
    echo "补装：bash setup.sh"
  fi
  exit ${rc}
fi

# =============================================================================
# 二、创建 / 更新环境
# =============================================================================
sec "② 创建 / 更新 conda 环境 '${ENV_NAME}'"

if env_exists; then
  echo "${INFO}环境已存在，将补齐缺失的包（不会重建）"
else
  echo "${INFO}创建环境（python=${PY_VERSION}）..."
  "${SOLVER}" create -n "${ENV_NAME}" -y -c conda-forge "python=${PY_VERSION}" \
    || { echo "创建环境失败"; exit 1; }
fi

# =============================================================================
# 三、编译工具链（conda-forge）
# =============================================================================
sec "③ 安装编译工具链（LLVM/MLIR ${LLVM_VERSION} + 构建与测试工具）"

cat <<EOF
  这一步替代了 linux-apt-packages.txt 里的 clang-17 / llvm-17-* / libmlir-17-dev /
  cmake / ninja-build / build-essential / python3-lit 那一整段——全部装进
  ${ENV_PREFIX}，不碰系统目录，不需要 root。

EOF

# 版本必须整体锁在同一个 LLVM 上：llvm-hello-compile 的 Pass 插件由 clang 编译、
# 被 opt 加载，两边 ABI 不一致会在 dlopen 时静默失败或崩溃。
"${SOLVER}" install -n "${ENV_NAME}" -y -c conda-forge \
  "llvm=${LLVM_VERSION}" \
  "llvmdev=${LLVM_VERSION}" \
  "mlir=${LLVM_VERSION}" \
  "clang=${LLVM_VERSION}" \
  "clangxx=${LLVM_VERSION}" \
  "lld=${LLVM_VERSION}" \
  cmake ninja make pkg-config cxx-compiler lit \
  zlib zstd ncurses libxml2
TOOLCHAIN_RC=$?

if [[ ${TOOLCHAIN_RC} -ne 0 ]]; then
  echo
  echo "${WARN}工具链安装失败。最常见原因是 ${LLVM_VERSION} 这个精确版本在当前平台"
  echo "${WARN}解不出来。可以放宽版本重试："
  echo "         LLVM_VERSION=17 bash setup.sh"
  note_fail "toolchain"
fi

# FileCheck 在 conda 布局里位于 libexec/llvm，软链到 bin 便于各 lab 统一定位
if [[ -x "${ENV_PREFIX}/libexec/llvm/FileCheck" && ! -e "${ENV_PREFIX}/bin/FileCheck" ]]; then
  ln -sf "${ENV_PREFIX}/libexec/llvm/FileCheck" "${ENV_PREFIX}/bin/FileCheck"
  echo "${OK}已软链 FileCheck → ${ENV_PREFIX}/bin/FileCheck"
fi

# =============================================================================
# 四、CUDA 编译工具（可选）
# =============================================================================
if [[ ${DO_CUDA} -eq 1 ]]; then
  sec "④ 安装 CUDA 编译工具（nvcc / cuobjdump / nvdisasm）"
  cat <<'EOF'
  注意区分两件事：
    · **驱动**（libcuda.so、nvidia-smi）—— 必须管理员装，conda 装不了，通常已有
    · **编译工具**（nvcc、cuobjdump）—— 可以装在 conda 里，就是这一步

  只跑 fatbin 轨（只编译、不执行 kernel）的话，有编译工具就够，不需要卡。
EOF
  "${SOLVER}" install -n "${ENV_NAME}" -y -c conda-forge \
    cuda-nvcc cuda-cuobjdump cuda-nvdisasm cuda-cudart-dev \
    || {
      echo "${WARN}conda-forge 的 CUDA 包安装失败。"
      echo "${WARN}若系统上已有 CUDA Toolkit，直接把它加进 PATH 即可，例如："
      echo "         export PATH=/usr/local/cuda/bin:\$PATH"
      note_fail "cuda-toolkit"
    }
fi

# =============================================================================
# 五、Python 实验库
# =============================================================================
if [[ ${DO_PYLABS} -eq 1 ]]; then
  sec "⑤ 安装 Python 实验库（pip，装进同一个 conda 环境）"

  PIP=("${ENV_PREFIX}/bin/python" -m pip)
  "${PIP[@]}" install -q -U pip setuptools wheel >/dev/null 2>&1

  # 逐组安装而不是一把梭：某一组失败不该拖垮其余五个 lab。
  install_group() {
    local label="$1"; shift
    echo
    echo "  ── ${label} ──"
    if "${PIP[@]}" install "$@"; then
      echo "${OK}${label}"
    else
      echo "${BAD}${label} 安装失败（其余组不受影响，稍后看汇总）"
      note_fail "${label}"
    fi
  }

  # psutil / matplotlib 归到共用组：dist-train-lab 也要它们，
  # 挂在 tvm 组里的话，apache-tvm 一失败会连带丢掉。
  install_group "共用基础库"            numpy "psutil>=5.9" "matplotlib>=3.7"
  install_group "onnx-delegate-lab"     "onnx>=1.14" "onnxruntime>=1.16"
  install_group "iree-lab"              iree-base-compiler iree-base-runtime
  install_group "tvm-fatbin-lab"        apache-tvm decorator attrs scipy \
                                        tornado cloudpickle xgboost

  case "${TORCH_FLAVOR}" in
    cu128)
      install_group "dist-train-lab (torch cu128, 支持 sm_120)" \
        --index-url https://download.pytorch.org/whl/cu128 torch
      ;;
    cu124|cu121)
      install_group "dist-train-lab (torch ${TORCH_FLAVOR})" \
        --index-url "https://download.pytorch.org/whl/${TORCH_FLAVOR}" torch
      ;;
    cpu)
      install_group "dist-train-lab (torch CPU 版)" \
        --index-url https://download.pytorch.org/whl/cpu torch
      ;;
    none)
      echo
      echo "${INFO}未装 torch（dist-train-lab 需要它）。按你的卡选一个："
      echo "         RTX 50 系 / Blackwell sm_120 → bash setup.sh --torch cu128"
      echo "         RTX 40/30 系                 → bash setup.sh --torch cu124"
      echo "         没有 GPU                     → bash setup.sh --torch cpu"
      ;;
    *)
      echo "${WARN}未知的 --torch 取值 '${TORCH_FLAVOR}'，跳过"
      ;;
  esac
fi

# =============================================================================
# 六、验收
# =============================================================================
sec "⑥ 验收体检"
FAILED_ITEMS=()   # 只统计最终状态，安装过程中的告警不重复计入
run_doctor "${ENV_PREFIX}"
DOCTOR_RC=$?

# =============================================================================
# 七、写出环境变量脚本 + 下一步
# =============================================================================
sec "⑦ 收尾"

ENVFILE="${REPO_DIR}/.ai-infra-env.sh"
cat > "${ENVFILE}" <<EOF
# 由 setup.sh 生成。用法： source .ai-infra-env.sh
# 各 lab 的 scripts/env.sh 读 CONDA_HOME + 自己那个 *_ENV 来定位工具链。
export CONDA_HOME="${CONDA_HOME}"
export LLVM_ENV="${ENV_NAME}"   # llvm-hello-compile
export TOY_ENV="${ENV_NAME}"    # mlir-toy-dialect
export TVM_ENV="${ENV_NAME}"    # tvm-fatbin-lab
export LAB_ENV="${ENV_NAME}"    # onnx-delegate-lab
export IREE_ENV="${ENV_NAME}"   # iree-lab
export DIST_ENV="${ENV_NAME}"   # dist-train-lab
EOF
echo "${OK}已写出 ${ENVFILE}"

# 必须与各 lab scripts/env.sh 里 `: "${CONDA_HOME:=...}"` 的字面值保持一致，
# 否则这里会误报"无需配置"。改那边时记得同步改这里。
DEFAULT_CONDA_HOME="/data/mm64/hanishzheng/miniforge3"
NEED_EXPORT=0
[[ "${CONDA_HOME}" != "${DEFAULT_CONDA_HOME}" ]] && NEED_EXPORT=1
[[ "${ENV_NAME}" != "mlir-env" ]] && NEED_EXPORT=1

echo
if [[ ${NEED_EXPORT} -eq 1 ]]; then
  cat <<EOF
  ⚠ 你的 conda 路径或环境名与各 lab 的**内置默认值**不同，需要让它们知道。
    二选一：

    a) 每次开新终端先 source（改动最小）：
         cd ${REPO_DIR} && source .ai-infra-env.sh

    b) 写进 ~/.bashrc（一劳永逸）：
         echo 'export CONDA_HOME="${CONDA_HOME}"' >> ~/.bashrc
         echo 'export LLVM_ENV=${ENV_NAME} TOY_ENV=${ENV_NAME} TVM_ENV=${ENV_NAME}' >> ~/.bashrc
         echo 'export LAB_ENV=${ENV_NAME} IREE_ENV=${ENV_NAME} DIST_ENV=${ENV_NAME}' >> ~/.bashrc
EOF
else
  echo "${OK}路径与环境名都是各 lab 的默认值，无需额外配置环境变量。"
fi

cat <<EOF

  ── 验证六个 lab（每条独立，可挑着跑）──
    cd ${REPO_DIR}
    bash llvm-hello-compile/scripts/run.sh          # 不需要卡
    bash mlir-toy-dialect/scripts/all.sh            # 不需要卡（首次构建约几分钟）
    bash iree-lab/scripts/run.sh                    # 不需要卡
    bash onnx-delegate-lab/scripts/run.sh           # 不需要卡
    bash tvm-fatbin-lab/scripts/run.sh              # TVM 轨不需要卡，fatbin 轨要 nvcc
    bash dist-train-lab/scripts/run_cpu_smoke.sh    # 不需要卡，但要 torch
    bash dist-train-lab/scripts/run_8gpu_wall.sh    # 需要 ≥2 张卡

  ── 随时重新体检 ──
    bash setup.sh --check

EOF

if [[ ${DOCTOR_RC} -eq 0 ]]; then
  echo "全部就绪。"
  exit 0
fi

echo "以下 ${#FAILED_ITEMS[@]} 项仍缺失：${FAILED_ITEMS[*]}"
cat <<'EOF'

  逐项对症：
    clang / opt / llc / mlir-*  → 工具链没装上，试 LLVM_VERSION=17 bash setup.sh
    python:tvm                  → apache-tvm 常在新版 Python 上没有 wheel，
                                  试 PY_VERSION=3.10 bash setup.sh --env-name aiinfra310
    python:torch                → 按卡选版本：bash setup.sh --torch cu128|cu124|cpu
    nvcc / cuobjdump            → bash setup.sh --with-cuda（或把系统 CUDA 加进 PATH）

  缺哪一项就只影响对应的 lab，其余照常可跑——各 lab 的脚本都会打印 [SKIP] 并说明原因。
EOF
exit 1
