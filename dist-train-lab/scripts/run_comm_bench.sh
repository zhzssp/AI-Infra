#!/usr/bin/env bash
# run_comm_bench.sh —— 集合通信扫描。用法：
#   bash scripts/run_comm_bench.sh [nproc]
#
# 想在【单机上模拟慢链路】看带宽域的影响，加环境变量重跑一次对比：
#   NCCL_P2P_DISABLE=1 bash scripts/run_comm_bench.sh 2
# 注意：在本身就没有 P2P 的消费卡上，这个开关不会有额外效果（本来就走 shm）。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/env.sh"
cd "${PROJECT_DIR}"
have_torch || { skip_no_torch; exit 2; }

NPROC="${NPROC:-2}"
[[ "${1:-}" =~ ^[0-9]+$ ]] && { NPROC="$1"; shift; }

SUFFIX=""
[[ "${NCCL_P2P_DISABLE:-0}" == "1" ]] && SUFFIX="_nop2p"

"${TORCHRUN}" --standalone --nproc_per_node="${NPROC}" comm_bench.py \
  --device "${DEVICE}" --backend "${BACKEND}" --ops all_reduce,all_gather \
  --out "${OUT_DIR}/comm_${BACKEND}_ws${NPROC}${SUFFIX}.csv" "$@"
