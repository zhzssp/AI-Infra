#!/usr/bin/env bash
# Run all TVM lab steps in order.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

if ! have_tvm; then
  cat <<EOF
[SKIP] TVM 不可用。请先安装：
  pip install apache-tvm
或按 https://tvm.apache.org/docs/install/ 构建。
可用环境变量 CONDA_HOME / TVM_ENV 指向含 TVM 的 conda 环境。
EOF
  exit 2
fi

STEPS=(
  01_te_matmul_schedules.py
  02_fusion_relay.py
  03_layout_transform.py
  04_tensorize_demo.py
  05_autotvm_tune.py
  06_packedfunc_run.py
)

idx=0
for s in "${STEPS[@]}"; do
  idx=$((idx + 1))
  echo
  echo ">>>>>>>>>> TVM step ${idx}/6: ${s}"
  "${PYTHON_BIN}" "${PROJECT_DIR}/tvm_lab/${s}"
done

echo
echo "[OK] 全部 TVM 步骤完成。产物在 out/tvm/"
