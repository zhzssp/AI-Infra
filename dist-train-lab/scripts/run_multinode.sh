#!/usr/bin/env bash
# =============================================================================
# run_multinode.sh —— 2 节点跨机（L3-DIST-12）
# -----------------------------------------------------------------------------
# 两个节点各执行一次，--node-rank 分别为 0 和 1：
#   bash scripts/run_multinode.sh --node-rank 0 --master 10.0.0.1 --nproc 2 --job ddp
#   bash scripts/run_multinode.sh --node-rank 1 --master 10.0.0.1 --nproc 2 --job ddp
#
# 只有一台机器时的降级：用 NCCL_P2P_DISABLE=1 / NCCL_SHM_DISABLE=1 制造慢链路，
# 见 run_comm_bench.sh。降级只能定性复现「带宽域差异」，拿不到真实跨机延迟。
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/env.sh"
cd "${PROJECT_DIR}"
have_torch || { skip_no_torch; exit 2; }

NODE_RANK=0; MASTER=""; NPROC=2; NNODES=2; JOB="ddp"; PORT=29500
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --node-rank) NODE_RANK="$2"; shift 2 ;;
    --master)    MASTER="$2";    shift 2 ;;
    --nproc)     NPROC="$2";     shift 2 ;;
    --nnodes)    NNODES="$2";    shift 2 ;;
    --job)       JOB="$2";       shift 2 ;;
    --port)      PORT="$2";      shift 2 ;;
    *)           EXTRA+=("$1");  shift   ;;
  esac
done

[[ -z "${MASTER}" ]] && { echo "缺少 --master <MASTER_ADDR>"; exit 1; }

case "${JOB}" in
  ddp)        SCRIPT="train_ddp.py" ;;
  fsdp)       SCRIPT="train_fsdp.py" ;;
  comm_bench) SCRIPT="comm_bench.py" ;;
  *) echo "未知 --job ${JOB}（可选 ddp / fsdp / comm_bench）"; exit 1 ;;
esac

echo "[multinode] node_rank=${NODE_RANK}/${NNODES}  nproc=${NPROC}  master=${MASTER}:${PORT}"
echo "[multinode] 建议同时开 NCCL_DEBUG=INFO，抓 NCCL 实际选的算法与拓扑："
echo "            关键字 Ring / Tree / NVLS / via NET|IB / P2P|direct pointer"

"${TORCHRUN}" \
  --nnodes="${NNODES}" --node_rank="${NODE_RANK}" --nproc_per_node="${NPROC}" \
  --rdzv_backend=c10d --rdzv_endpoint="${MASTER}:${PORT}" \
  "${SCRIPT}" --device "${DEVICE}" --backend "${BACKEND}" \
  --hidden "${HIDDEN}" --layers "${LAYERS}" --batch "${BATCH}" --seq "${SEQ}" \
  --steps "${STEPS}" "${EXTRA[@]}"
