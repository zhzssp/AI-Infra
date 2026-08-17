#!/usr/bin/env bash
# run_fsdp.sh —— FSDP 三档 sharding_strategy 一次跑完。用法：
#   bash scripts/run_fsdp.sh [nproc]
#   STRATEGIES="full" bash scripts/run_fsdp.sh 8
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/env.sh"
cd "${PROJECT_DIR}"
have_torch || { skip_no_torch; exit 2; }

NPROC="${NPROC:-2}"
[[ "${1:-}" =~ ^[0-9]+$ ]] && { NPROC="$1"; shift; }
: "${STRATEGIES:=none grad_op full}"

for s in ${STRATEGIES}; do
  sec "FSDP --strategy ${s}   world=${NPROC}"
  "${TORCHRUN}" --standalone --nproc_per_node="${NPROC}" train_fsdp.py \
    --device "${DEVICE}" --backend "${BACKEND}" --strategy "${s}" \
    --hidden "${HIDDEN}" --layers "${LAYERS}" --batch "${BATCH}" --seq "${SEQ}" \
    --steps "${STEPS}" --precision amp --tag "zero_${s}" "$@"
done

sec "手算对照"
"${PYTHON_BIN}" mem_ledger.py --hidden "${HIDDEN}" --layers "${LAYERS}" \
  --precision mixed --ladder --n "${NPROC}"

sec "汇总"
"${PYTHON_BIN}" report.py --glob "${OUT_DIR}/zero_*.json" --rank 0 \
  --markdown "${OUT_DIR}/table_zero.md"
