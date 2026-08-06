#!/usr/bin/env bash
# =============================================================================
# env.sh —— 定位 Python / TVM / NVCC / cuobjdump
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${PROJECT_DIR}/out"
export PROJECT_DIR OUT_DIR
mkdir -p "${OUT_DIR}/tvm" "${OUT_DIR}/cuda" "${OUT_DIR}/steps"

: "${CONDA_HOME:=/data/mm64/hanishzheng/miniforge3}"
: "${TVM_ENV:=mlir-env}"

# Optional conda activate (same convention as llvm-hello-compile).
if [[ -x "${CONDA_HOME}/bin/conda" ]]; then
  # shellcheck disable=SC1091
  eval "$("${CONDA_HOME}/bin/conda" shell.bash hook)"
  conda activate "${TVM_ENV}" 2>/dev/null || true
fi

find_tool() {
  local name="$1"
  local p
  p="$(command -v "${name}" 2>/dev/null || true)"
  [[ -n "$p" ]] && { echo "$p"; return 0; }
  return 1
}

PYTHON_BIN="$(find_tool python3 || find_tool python || true)"
NVCC_BIN="$(find_tool nvcc || true)"
CUOBJDUMP_BIN="$(find_tool cuobjdump || true)"

export PYTHON_BIN NVCC_BIN CUOBJDUMP_BIN

have_tvm() {
  [[ -n "${PYTHON_BIN}" ]] || return 1
  "${PYTHON_BIN}" -c "import tvm" 2>/dev/null
}

have_cuda() {
  [[ -n "${NVCC_BIN}" && -n "${CUOBJDUMP_BIN}" ]]
}

echo "[env] PROJECT_DIR=${PROJECT_DIR}"
echo "[env] PYTHON=${PYTHON_BIN:-<missing>}"
echo "[env] NVCC=${NVCC_BIN:-<missing>}"
echo "[env] CUOBJDUMP=${CUOBJDUMP_BIN:-<missing>}"
if have_tvm; then
  echo "[env] TVM=$("${PYTHON_BIN}" -c 'import tvm; print(tvm.__version__)')"
else
  echo "[env] TVM=<not importable>"
fi
