#!/usr/bin/env bash
# =============================================================================
# build_passes.sh —— 编译自定义 Pass 插件 MyPasses.so
#   依赖：conda 环境 mlir-env 里的 llvm-config / cmake / c++ 编译器
#   产物：build/passes/MyPasses.so
#   幂等：已存在且比源码新则跳过（除非传入 --force 强制重建）
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh" >/dev/null

PASS_SRC_DIR="${PROJECT_DIR}/passes"
PASS_BUILD_DIR="${PROJECT_DIR}/build/passes"
PLUGIN="${PASS_BUILD_DIR}/MyPasses.so"

FORCE="${1:-}"

# 定位 llvm-config 与其 cmake 目录
LLVM_CONFIG="$(command -v llvm-config || true)"
if [[ -z "${LLVM_CONFIG}" && -n "${CLANG}" ]]; then
  LLVM_CONFIG="$(dirname "${CLANG}")/llvm-config"
fi
[[ -x "${LLVM_CONFIG}" ]] || { echo "[build_passes] 找不到 llvm-config" >&2; exit 1; }
LLVM_CMAKE_DIR="$("${LLVM_CONFIG}" --cmakedir)"

# 幂等判断：插件已存在且比所有源码新，则不用重建
if [[ -z "${FORCE}" && -f "${PLUGIN}" ]]; then
  newest_src="$(ls -t "${PASS_SRC_DIR}"/*.cpp "${PASS_SRC_DIR}"/*.h "${PASS_SRC_DIR}"/CMakeLists.txt 2>/dev/null | head -1)"
  if [[ -z "${newest_src}" || "${PLUGIN}" -nt "${newest_src}" ]]; then
    echo "[build_passes] 插件已是最新，跳过构建：${PLUGIN}"
    echo "${PLUGIN}"
    exit 0
  fi
fi

echo "[build_passes] LLVM_CONFIG   : ${LLVM_CONFIG}"
echo "[build_passes] LLVM cmake dir: ${LLVM_CMAKE_DIR}"
echo "[build_passes] 配置 + 编译中 ..."

cmake -S "${PASS_SRC_DIR}" -B "${PASS_BUILD_DIR}" \
      -DLLVM_DIR="${LLVM_CMAKE_DIR}" \
      -DCMAKE_BUILD_TYPE=Release >/dev/null

cmake --build "${PASS_BUILD_DIR}" --parallel >/dev/null

if [[ -f "${PLUGIN}" ]]; then
  echo "[build_passes] 构建成功：${PLUGIN}"
  echo "${PLUGIN}"
else
  echo "[build_passes] 构建失败：未生成 ${PLUGIN}" >&2
  exit 1
fi
