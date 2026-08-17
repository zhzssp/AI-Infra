#!/usr/bin/env bash
# =============================================================================
# env.sh —— 定位 python/torchrun，探测卡数与 P2P 状况，导出统一的实验档位
# -----------------------------------------------------------------------------
# 本 lab 的一个前提：**没有 GPU 也要能跑**。所以这里探测到什么就用什么，
# 探测不到 CUDA 就自动退到 cpu + gloo，而不是直接报错退出。
# =============================================================================
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${PROJECT_DIR}/out"
export PROJECT_DIR OUT_DIR
mkdir -p "${OUT_DIR}"

# 与其它 lab 相同的 conda 约定（可用环境变量覆盖）
: "${CONDA_HOME:=/data/mm64/hanishzheng/miniforge3}"
: "${DIST_ENV:=mlir-env}"
if [[ -x "${CONDA_HOME}/bin/conda" ]]; then
  # shellcheck disable=SC1091
  eval "$("${CONDA_HOME}/bin/conda" shell.bash hook)"
  conda activate "${DIST_ENV}" 2>/dev/null || true
fi

find_tool() {
  local p
  p="$(command -v "$1" 2>/dev/null || true)"
  [[ -n "$p" ]] && { echo "$p"; return 0; }
  return 1
}

PYTHON_BIN="$(find_tool python3 || find_tool python || true)"
TORCHRUN="$(find_tool torchrun || true)"
export PYTHON_BIN TORCHRUN

have_torch() {
  [[ -n "${PYTHON_BIN}" ]] && "${PYTHON_BIN}" -c "import torch" >/dev/null 2>&1
}

skip_no_torch() {
  cat <<'EOF'
[SKIP] 未找到可用的 PyTorch。

安装（CPU 版即可完成本 lab 的全部 L0/L1 条目）：
    pip install -r requirements.txt

如果要用 RTX 50 系（Blackwell, sm_120），必须装 cu128 及以上的构建：
    pip install torch --index-url https://download.pytorch.org/whl/cu128
  cu124 及更早的 wheel **不支持 sm_120**，会报 "no kernel image is available"。

对应教材：docs/checkpoints/08-distributed.md
EOF
}

# --- 探测 GPU 数量与 P2P ------------------------------------------------------
N_GPU=0
DEVICE="cpu"
BACKEND="gloo"
if have_torch; then
  N_GPU="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
  if [[ "${N_GPU}" -gt 0 ]]; then
    DEVICE="cuda"
    BACKEND="nccl"
  fi
fi
export N_GPU DEVICE BACKEND

# --- 实验档位（口径统一，各脚本都引用这里）-----------------------------------
# 主判据要让【模型状态占主导】，所以是大 hidden + 小 batch/seq。
# 反过来验激活相关结论时（重计算）再调大 batch/seq —— 见 run_8gpu_wall.sh。
: "${HIDDEN:=2048}"
: "${LAYERS:=12}"
: "${BATCH:=4}"
: "${SEQ:=256}"
: "${STEPS:=30}"
export HIDDEN LAYERS BATCH SEQ STEPS

# CPU 冒烟用的小档位，秒级跑完
: "${SMOKE_HIDDEN:=256}"
: "${SMOKE_LAYERS:=4}"
export SMOKE_HIDDEN SMOKE_LAYERS

sec() {
  echo
  echo "────────────────────────────────────────────────────────────"
  echo "▶ $*"
  echo "────────────────────────────────────────────────────────────"
}

if [[ "${DIST_ENV_QUIET:-0}" != "1" ]]; then
  echo "[env] PROJECT_DIR = ${PROJECT_DIR}"
  echo "[env] python      = ${PYTHON_BIN:-<missing>}"
  echo "[env] torchrun    = ${TORCHRUN:-<missing>}"
  if have_torch; then
    "${PYTHON_BIN}" - <<'PY'
import torch
print(f"[env] torch       = {torch.__version__}  cuda={torch.version.cuda}")
n = torch.cuda.device_count()
print(f"[env] gpu count   = {n}")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"[env]   cuda:{i} {p.name}  sm_{p.major}{p.minor}  "
          f"{p.total_memory/2**30:.1f} GiB  {p.multi_processor_count} SM")
if n >= 2:
    peer = any(torch.cuda.can_device_access_peer(i, j)
               for i in range(n) for j in range(n) if i != j)
    print(f"[env] P2P         = {'可用' if peer else '全部关闭（消费级 GeForce 的典型情况）'}")
    if not peer:
        print("[env]   → NCCL 会退回 host 中转的 shared-memory 传输，卡间带宽远低于显存带宽。")
        print("[env]   → 这不是配置错误，而是本 lab 最重要的背景常数。")
PY
  else
    echo "[env] torch       = <missing>  （将只能跑纸面推演脚本）"
  fi
  echo "[env] device/backend = ${DEVICE} / ${BACKEND}   档位 hidden=${HIDDEN} layers=${LAYERS}"
fi
