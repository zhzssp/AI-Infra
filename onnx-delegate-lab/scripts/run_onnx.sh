#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

if ! have_onnx; then
  cat <<EOF
[SKIP] 需要 onnx + onnxruntime：
  pip install -r requirements.txt
EOF
  exit 2
fi

for s in 01_build_and_infer.py 02_mutate_graph.py 03_ort_ep_partition.py; do
  echo
  echo ">>>>>>>>>> ONNX: ${s}"
  "${PYTHON_BIN}" "${PROJECT_DIR}/onnx_lab/${s}"
done
echo "[OK] ONNX 轨完成 → out/onnx/"
