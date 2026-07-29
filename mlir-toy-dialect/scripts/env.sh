#!/usr/bin/env bash
# =============================================================================
# env.sh —— 公共环境（被其他脚本 source 引用，一般不直接单独运行）
# -----------------------------------------------------------------------------
# 作用：
#   1. 定位并激活承载 LLVM/MLIR 的 conda 环境（默认 mlir-env）。
#   2. 导出后续 CMake 配置需要的关键路径（MLIR_DIR / LLVM_DIR / lit / bin）。
#
# 可用环境变量覆盖默认值：
#   CONDA_HOME  conda/miniforge 安装根目录（默认见下）
#   TOY_ENV     conda 环境名（默认 mlir-env）
# =============================================================================

# --- 可覆盖的默认值 ---
: "${CONDA_HOME:=/data/mm64/hanishzheng/miniforge3}"
: "${TOY_ENV:=mlir-env}"

# --- 项目根目录：本文件所在的 scripts/ 的上一级 ---
TOY_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- 校验 conda 是否存在 ---
if [[ ! -x "${CONDA_HOME}/bin/conda" ]]; then
  echo "[env] 找不到 conda：${CONDA_HOME}/bin/conda" >&2
  echo "[env] 请先安装 miniforge（https://github.com/conda-forge/miniforge），" >&2
  echo "[env] 或设置环境变量 CONDA_HOME 指向已有的 conda 安装根目录。" >&2
  return 1 2>/dev/null || exit 1
fi

# --- 激活 conda 环境 ---
eval "$("${CONDA_HOME}/bin/conda" shell.bash hook)"
if ! conda activate "${TOY_ENV}" 2>/dev/null; then
  echo "[env] conda 环境 '${TOY_ENV}' 不存在。请先运行：scripts/setup.sh" >&2
  return 1 2>/dev/null || exit 1
fi

# --- 导出关键路径 ---
TOY_ENV_PREFIX="${CONDA_PREFIX}"
export TOY_PROJECT_DIR TOY_ENV_PREFIX
export MLIR_DIR="${TOY_ENV_PREFIX}/lib/cmake/mlir"   # find_package(MLIR) 用
export LLVM_DIR="${TOY_ENV_PREFIX}/lib/cmake/llvm"   # find_package(LLVM) 用
export TOY_LIT="${TOY_ENV_PREFIX}/bin/lit"           # 自动化测试驱动
export TOY_BIN_DIR="${TOY_ENV_PREFIX}/bin"           # 内含 FileCheck / mlir-* 等
export TOY_BUILD_DIR="${TOY_PROJECT_DIR}/build"      # 构建输出目录

echo "[env] 项目根目录 : ${TOY_PROJECT_DIR}"
echo "[env] conda 环境 : ${TOY_ENV}  (${TOY_ENV_PREFIX})"
