#!/usr/bin/env bash
# =============================================================================
# clean.sh —— 清理所有中间产物（out/ 目录）
# =============================================================================
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rm -rf "${PROJECT_DIR}/out"
echo "[clean] 已删除 ${PROJECT_DIR}/out"
