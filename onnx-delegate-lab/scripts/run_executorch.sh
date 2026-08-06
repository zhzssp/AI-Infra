#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[SKIP] 未找到 python"; exit 2
fi

# ExecuTorch 可选：未安装时 01 脚本仍会产出模拟结果
for s in 01_partitioner_lab.py 02_compare_three_systems.py; do
  echo
  echo ">>>>>>>>>> ExecuTorch: ${s}"
  "${PYTHON_BIN}" "${PROJECT_DIR}/executorch_lab/${s}"
done
echo "[OK] ExecuTorch 轨完成 → out/executorch/"
