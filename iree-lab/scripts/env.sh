#!/usr/bin/env bash
# =============================================================================
# env.sh —— 定位 iree-compile / iree-run-module / python，并探测编译标志的写法
# -----------------------------------------------------------------------------
# IREE 的目标设备标志在 2024 年前后改过一次写法，两代都还在用：
#   新：--iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu
#   旧：--iree-hal-target-backends=llvm-cpu
# 本文件会实际试一次，把能用的那套导出成 IREE_TARGET_FLAGS，后续脚本直接用。
# =============================================================================
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${PROJECT_DIR}/out"
MODELS_DIR="${PROJECT_DIR}/models"
export PROJECT_DIR OUT_DIR MODELS_DIR
mkdir -p "${OUT_DIR}/phases" "${OUT_DIR}/execute" "${OUT_DIR}/variants"

# 与其它 lab 相同的 conda 约定（可用环境变量覆盖）
: "${CONDA_HOME:=/data/mm64/hanishzheng/miniforge3}"
: "${IREE_ENV:=mlir-env}"
if [[ -x "${CONDA_HOME}/bin/conda" ]]; then
  # shellcheck disable=SC1091
  eval "$("${CONDA_HOME}/bin/conda" shell.bash hook)"
  conda activate "${IREE_ENV}" 2>/dev/null || true
fi

find_tool() {
  local p
  p="$(command -v "$1" 2>/dev/null || true)"
  [[ -n "$p" ]] && { echo "$p"; return 0; }
  return 1
}

PYTHON_BIN="$(find_tool python3 || find_tool python || true)"
IREE_COMPILE="$(find_tool iree-compile || true)"
IREE_RUN="$(find_tool iree-run-module || true)"
export PYTHON_BIN IREE_COMPILE IREE_RUN

have_iree() { [[ -n "${IREE_COMPILE}" && -n "${IREE_RUN}" ]]; }

# --- 探测哪一套目标标志能用 -------------------------------------------------
IREE_TARGET_FLAGS=""
if have_iree; then
  _probe="${OUT_DIR}/.probe.mlir"
  printf 'func.func @p(%%a: tensor<f32>) -> tensor<f32> { return %%a : tensor<f32> }\n' > "${_probe}"

  _new=(--iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu)
  _old=(--iree-hal-target-backends=llvm-cpu)

  if "${IREE_COMPILE}" "${_new[@]}" "${_probe}" -o /dev/null 2>/dev/null; then
    IREE_TARGET_FLAGS="${_new[*]}"
  elif "${IREE_COMPILE}" "${_old[@]}" "${_probe}" -o /dev/null 2>/dev/null; then
    IREE_TARGET_FLAGS="${_old[*]}"
  else
    # 两套都没通过：仍给出新式写法，让后续脚本报出真实的编译错误而不是空数组
    IREE_TARGET_FLAGS="${_new[*]}"
    echo "[env] ⚠️ 目标标志探测失败，暂用新式写法；若报 unknown option，"
    echo "[env]    请 iree-compile --help | grep -i 'target-device\|target-backends' 核对。"
  fi
  rm -f "${_probe}"
fi
export IREE_TARGET_FLAGS

# 运行时设备：local-sync（同步，最容易读）优先，其次 local-task
: "${IREE_DEVICE:=local-sync}"
export IREE_DEVICE

echo "[env] PROJECT_DIR   = ${PROJECT_DIR}"
echo "[env] python        = ${PYTHON_BIN:-<missing>}"
echo "[env] iree-compile  = ${IREE_COMPILE:-<missing>}"
echo "[env] iree-run-module = ${IREE_RUN:-<missing>}"
if have_iree; then
  echo "[env] iree version  = $("${IREE_COMPILE}" --version 2>/dev/null | head -1)"
  echo "[env] target flags  = ${IREE_TARGET_FLAGS:-<probe failed>}"
  echo "[env] runtime device= ${IREE_DEVICE}"
fi

# 缺依赖时统一的提示（各步骤脚本调用它然后 exit 2）
skip_no_iree() {
  cat <<'EOF'
[SKIP] 未找到 iree-compile / iree-run-module。

安装（只需 CPU 后端，不用编译源码、不用 GPU）：
    pip install -r requirements.txt

对应教材：docs/learning-guides/iree-learning-guide.md 第 5 章。
EOF
}
