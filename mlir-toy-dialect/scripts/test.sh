#!/usr/bin/env bash
# =============================================================================
# test.sh —— 运行 lit 自动化测试套件（check-toy）
# -----------------------------------------------------------------------------
# 它用 lit 扫描 test/*.mlir，执行每个文件顶部的 // RUN: 命令，
# 再用 FileCheck 对照 // CHECK: 断言，最后汇总 PASS/FAIL。
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

if [[ ! -d "${TOY_BUILD_DIR}" ]]; then
  echo "[test] 未找到 build 目录，请先运行：scripts/build.sh" >&2
  exit 1
fi

echo "[test] 运行 lit 测试目标 check-toy ..."
cmake --build "${TOY_BUILD_DIR}" --target check-toy
echo "[test] 全部测试通过。"
