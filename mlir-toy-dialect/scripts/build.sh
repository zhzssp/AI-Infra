#!/usr/bin/env bash
# =============================================================================
# build.sh —— 配置并编译本项目，产出 build/bin/toy-opt
# -----------------------------------------------------------------------------
# 两个阶段：
#   1. cmake 配置：通过 find_package 找到 conda 里的 MLIR/LLVM，
#                  并指定 lit / FileCheck 路径（供 check-toy 测试目标用）。
#   2. cmake 构建：先 mlir-tblgen 生成 .inc，再编译 MLIRToy 库与 toy-opt。
#
# 可选参数：传入 "clean" 先清空 build 目录再全新配置。
#   scripts/build.sh clean
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

if [[ "${1:-}" == "clean" ]]; then
  echo "[build] 清空构建目录：${TOY_BUILD_DIR}"
  rm -rf "${TOY_BUILD_DIR}"
fi

echo "[build] 阶段 1/2：CMake 配置 ..."
cmake -G Ninja -S "${TOY_PROJECT_DIR}" -B "${TOY_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DMLIR_DIR="${MLIR_DIR}" \
  -DLLVM_DIR="${LLVM_DIR}" \
  -DLLVM_EXTERNAL_LIT="${TOY_LIT}" \
  -DLLVM_EXTERNAL_LIT_DIR="${TOY_BIN_DIR}"

echo "[build] 阶段 2/2：编译 ..."
cmake --build "${TOY_BUILD_DIR}"

echo "[build] 完成：${TOY_BUILD_DIR}/bin/toy-opt"
