#!/usr/bin/env bash
# run_ddp.sh —— DDP 单组运行。用法：
#   bash scripts/run_ddp.sh [nproc] [额外参数...]
#   NPROC=4 bash scripts/run_ddp.sh -- --bucket-mb 1 --tag bk1
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/env.sh"
cd "${PROJECT_DIR}"
have_torch || { skip_no_torch; exit 2; }

NPROC="${NPROC:-2}"
[[ "${1:-}" =~ ^[0-9]+$ ]] && { NPROC="$1"; shift; }
[[ "${1:-}" == "--" ]] && shift

"${TORCHRUN}" --standalone --nproc_per_node="${NPROC}" train_ddp.py \
  --device "${DEVICE}" --backend "${BACKEND}" \
  --hidden "${HIDDEN}" --layers "${LAYERS}" --batch "${BATCH}" --seq "${SEQ}" \
  --steps "${STEPS}" "$@"
