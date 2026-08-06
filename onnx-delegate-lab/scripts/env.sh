#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${PROJECT_DIR}/out"
export PROJECT_DIR OUT_DIR
mkdir -p "${OUT_DIR}/onnx" "${OUT_DIR}/executorch" "${OUT_DIR}/steps"

: "${CONDA_HOME:=/data/mm64/hanishzheng/miniforge3}"
: "${LAB_ENV:=mlir-env}"

if [[ -x "${CONDA_HOME}/bin/conda" ]]; then
  # shellcheck disable=SC1091
  eval "$("${CONDA_HOME}/bin/conda" shell.bash hook)"
  conda activate "${LAB_ENV}" 2>/dev/null || true
fi

PYTHON_BIN="$(command -v python3 || command -v python || true)"
export PYTHON_BIN

have_onnx() {
  [[ -n "${PYTHON_BIN}" ]] || return 1
  "${PYTHON_BIN}" -c "import onnx, onnxruntime" 2>/dev/null
}

have_executorch() {
  [[ -n "${PYTHON_BIN}" ]] || return 1
  "${PYTHON_BIN}" -c "import executorch" 2>/dev/null
}

echo "[env] PROJECT_DIR=${PROJECT_DIR}"
echo "[env] PYTHON=${PYTHON_BIN:-<missing>}"
if have_onnx; then
  echo "[env] onnx/onnxruntime=OK"
else
  echo "[env] onnx/onnxruntime=<missing>  → pip install -r requirements.txt"
fi
if have_executorch; then
  echo "[env] executorch=OK"
else
  echo "[env] executorch=<optional>  → 未安装时走概念模拟"
fi
