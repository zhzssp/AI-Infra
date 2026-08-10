#!/usr/bin/env bash
# =============================================================================
# env.sh —— 公共环境：定位 LLVM 工具链（clang / opt / llc / lli / llvm-as ...）
# -----------------------------------------------------------------------------
# 本项目不需要编译任何 C++ 代码，只用现成的 LLVM 命令行工具走一遍编译流程。
# 工具查找顺序：优先用 conda 环境（mlir-env，内含 LLVM 17.0.6），再退回系统 PATH。
#
# 可用环境变量覆盖默认值：
#   CONDA_HOME  conda/miniforge 安装根目录
#   LLVM_ENV    conda 环境名（默认 mlir-env）
# =============================================================================

: "${CONDA_HOME:=/data/mm64/hanishzheng/miniforge3}"
: "${LLVM_ENV:=mlir-env}"

# --- 项目根目录：本文件所在 scripts/ 的上一级 ---
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${PROJECT_DIR}/out"
export PROJECT_DIR OUT_DIR

# --- 尝试激活 conda 环境（里面有 opt/llc/lli/llvm-as/llvm-dis 等 17.0.6 工具）---
CONDA_BIN=""
if [[ -x "${CONDA_HOME}/bin/conda" ]]; then
  eval "$("${CONDA_HOME}/bin/conda" shell.bash hook)"
  if conda activate "${LLVM_ENV}" 2>/dev/null; then
    CONDA_BIN="${CONDA_PREFIX}/bin"
  fi
fi

# --- 工具查找函数：先看 conda env/bin，再看 PATH ---
find_tool() {
  local name="$1"
  if [[ -n "${CONDA_BIN}" && -x "${CONDA_BIN}/${name}" ]]; then
    echo "${CONDA_BIN}/${name}"; return 0
  fi
  local p; p="$(command -v "${name}" 2>/dev/null)"
  [[ -n "$p" ]] && { echo "$p"; return 0; }
  return 1
}

# --- 必备工具（缺一不可）---
CLANG="$(find_tool clang)" || { echo "[env] 找不到 clang（前端）" >&2; return 1 2>/dev/null || exit 1; }
OPT="$(find_tool opt)"     || { echo "[env] 找不到 opt（优化器）" >&2; return 1 2>/dev/null || exit 1; }
LLC="$(find_tool llc)"     || { echo "[env] 找不到 llc（后端）" >&2; return 1 2>/dev/null || exit 1; }

# --- 可选工具（没有也能跑主流程）---
LLI="$(find_tool lli || true)"
LLVM_AS="$(find_tool llvm-as || true)"
LLVM_DIS="$(find_tool llvm-dis || true)"
LLVM_MCA="$(find_tool llvm-mca || true)"
# tour.sh 额外用到的：TableGen（第 6 章）、MC 层（§5.7）、lit 测试
LLVM_TBLGEN="$(find_tool llvm-tblgen || true)"
LLVM_MC="$(find_tool llvm-mc || true)"
LLVM_OBJDUMP="$(find_tool llvm-objdump || true)"
FILECHECK="$(find_tool FileCheck || true)"

export CLANG OPT LLC LLI LLVM_AS LLVM_DIS LLVM_MCA
export LLVM_TBLGEN LLVM_MC LLVM_OBJDUMP FILECHECK

echo "[env] 项目根目录 : ${PROJECT_DIR}"
echo "[env] clang      : ${CLANG}"
echo "[env] opt        : ${OPT}"
echo "[env] llc        : ${LLC}"
echo "[env] 版本       : $("${CLANG}" --version 2>/dev/null | head -1)"
