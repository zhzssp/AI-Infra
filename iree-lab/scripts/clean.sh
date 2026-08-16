#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rm -rf "${PROJECT_DIR}/out"
echo "已清空 ${PROJECT_DIR}/out"
