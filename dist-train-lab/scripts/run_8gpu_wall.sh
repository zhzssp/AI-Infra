#!/usr/bin/env bash
# =============================================================================
# run_8gpu_wall.sh —— 通信墙：多卡机器上的主线实验
# -----------------------------------------------------------------------------
# 这个脚本要测出来的，是一个在教科书上被 NVLink 掩盖、但在消费级多卡机器上
# 被放大十倍的事实：**通信可以比计算还贵**。
#
# 四组对照，卡数逐级加倍，每组都给出「预测 → 实测」：
#   A 通信带宽随卡数怎么变      → 拿到这台机器的背景常数
#   B DDP 的扩展效率            → 数据并行的通信代价
#   C 张量并行会不会塌          → 为什么 TP 需要高带宽域
#   D FSDP/ZeRO 阶梯            → 拿显存换通信的兑换率
#
# 全程约 30~60 分钟。产物在 out/，最后用 report.py 汇总成表。
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/env.sh"
cd "${PROJECT_DIR}"

have_torch || { skip_no_torch; exit 2; }

if [[ "${N_GPU}" -lt 2 ]]; then
  cat <<'EOF'
[SKIP] 本脚本需要 ≥2 张 GPU。

只有单卡时能做的：
    python train_single.py --device cuda --hidden 2048 --layers 12 --batch 4 --seq 256
    python mem_ledger.py --hidden 2048 --layers 12 --ladder --n 8
无卡时：
    bash scripts/run_cpu_smoke.sh
EOF
  exit 2
fi

# 按实际卡数生成 2 的幂次序列（最多到 N_GPU）
WORLDS=()
w=2
while [[ "${w}" -le "${N_GPU}" ]]; do WORLDS+=("${w}"); w=$((w * 2)); done
echo "[plan] 检测到 ${N_GPU} 张卡，将在 world_size = ${WORLDS[*]} 上做对照"

# -----------------------------------------------------------------------------
sec "A｜集合通信带宽 vs 卡数（这台机器的背景常数）"
cat <<'EOF'
  先预测再看结果：
   1. 卡数从 2 加到 8，all_reduce 的【总线带宽】会上升、持平，还是下降？
   2. 拐点会往左移还是往右移？（参与方变多 → 每轮的固定延迟怎么变）
   3. 把测出的峰值带宽和单卡显存带宽相除 —— 这个比值决定了后面三组的一切。
EOF
for ws in "${WORLDS[@]}"; do
  echo
  echo "  ── world_size = ${ws} ──"
  "${TORCHRUN}" --standalone --nproc_per_node="${ws}" comm_bench.py \
    --device "${DEVICE}" --backend "${BACKEND}" --ops all_reduce,all_gather \
    --min-exp 12 --max-exp 26 --iters 20 \
    --out "${OUT_DIR}/comm_${BACKEND}_ws${ws}.csv"
done

# -----------------------------------------------------------------------------
sec "B｜DDP 扩展效率：固定 global batch，看吞吐能不能线性涨"
cat <<EOF
  口径：global batch 固定，per-rank batch = global / world。
  这样比的才是「同样的活分给更多人干」，而不是「活变多了」。

  先预测：8 卡能拿到 8x 吗？拿不到的话，缺口主要被什么吃掉了？
EOF
GLOBAL_BATCH=$((BATCH * ${WORLDS[${#WORLDS[@]}-1]}))
echo "  global batch = ${GLOBAL_BATCH}（hidden=${HIDDEN} layers=${LAYERS} seq=${SEQ}）"

"${PYTHON_BIN}" train_single.py --device "${DEVICE}" \
  --hidden "${HIDDEN}" --layers "${LAYERS}" --batch "${GLOBAL_BATCH}" --seq "${SEQ}" \
  --steps "${STEPS}" --tag B_single 2>&1 | tail -n 6

for ws in "${WORLDS[@]}"; do
  per_rank=$((GLOBAL_BATCH / ws))
  echo
  echo "  ── DDP world_size=${ws}  per-rank batch=${per_rank} ──"
  "${TORCHRUN}" --standalone --nproc_per_node="${ws}" train_ddp.py \
    --device "${DEVICE}" --backend "${BACKEND}" \
    --hidden "${HIDDEN}" --layers "${LAYERS}" --batch "${per_rank}" --seq "${SEQ}" \
    --steps "${STEPS}" --tag "B_ddp_ws${ws}" 2>&1 | tail -n 5
done

sec "B'｜显存对照：固定 per-rank batch，看显存降不降"
echo "  这一组是【判显存】的。DDP 的每卡仍持有完整 16Φ —— 省的是时间，不是显存。"
"${PYTHON_BIN}" train_single.py --device "${DEVICE}" \
  --hidden "${HIDDEN}" --layers "${LAYERS}" --batch "${BATCH}" --seq "${SEQ}" \
  --steps "${STEPS}" --tag A_single 2>&1 | tail -n 6
"${TORCHRUN}" --standalone --nproc_per_node=2 train_ddp.py \
  --device "${DEVICE}" --backend "${BACKEND}" \
  --hidden "${HIDDEN}" --layers "${LAYERS}" --batch "${BATCH}" --seq "${SEQ}" \
  --steps "${STEPS}" --check-consistency --tag A_ddp 2>&1 | tail -n 6

# -----------------------------------------------------------------------------
sec "C｜张量并行：切得越细，通信越频繁"
cat <<'EOF'
  先预测：TP 把一个 FFN 的两个矩阵切到 N 张卡上，计算量每卡降到 1/N，
  但每次前向都要一次 all_reduce。加速比会是多少？

  在【有 NVLink】的机器上，TP=2/4 通常还能拿到正收益。
  在【没有 NVLink / P2P】的机器上，预期是 **加速比 < 1，即负优化**。
  如果你测出来正是这样 —— 那不是配置错了，那是这条工程约束的来源。
EOF
for ws in "${WORLDS[@]}"; do
  echo
  echo "  ── TP size = ${ws} ──"
  "${TORCHRUN}" --standalone --nproc_per_node="${ws}" tp_min.py \
    --device "${DEVICE}" --backend "${BACKEND}" --hidden 4096 --batch 32 \
    --check col,row,ffn --count-collectives --bench --iters 30 2>&1 | tail -n 14
done

# -----------------------------------------------------------------------------
sec "D｜FSDP / ZeRO 阶梯：用通信换显存"
cat <<EOF
  同一个模型、同一个 per-rank batch，只换分片策略。先看手算账：
EOF
MAXW="${WORLDS[${#WORLDS[@]}-1]}"
"${PYTHON_BIN}" mem_ledger.py --hidden "${HIDDEN}" --layers "${LAYERS}" \
  --precision mixed --ladder --n "${MAXW}"

for s in none grad_op full; do
  echo
  echo "  ── FSDP --strategy ${s}  world=${MAXW} ──"
  "${TORCHRUN}" --standalone --nproc_per_node="${MAXW}" train_fsdp.py \
    --device "${DEVICE}" --backend "${BACKEND}" --strategy "${s}" \
    --hidden "${HIDDEN}" --layers "${LAYERS}" --batch "${BATCH}" --seq "${SEQ}" \
    --steps "${STEPS}" --precision amp --tag "D_fsdp_${s}" 2>&1 | tail -n 5
done

echo
echo "  ── DDP + ZeroRedundancyOptimizer（真正的 ZeRO-1）──"
"${TORCHRUN}" --standalone --nproc_per_node="${MAXW}" train_ddp.py \
  --device "${DEVICE}" --backend "${BACKEND}" --zero1 \
  --hidden "${HIDDEN}" --layers "${LAYERS}" --batch "${BATCH}" --seq "${SEQ}" \
  --steps "${STEPS}" --tag D_zero1 2>&1 | tail -n 5

# -----------------------------------------------------------------------------
sec "汇总"
"${PYTHON_BIN}" report.py --glob "${OUT_DIR}/B_*.json" --rank 0 \
  --baseline-tag B_single --markdown "${OUT_DIR}/table_scaling.md"
echo
"${PYTHON_BIN}" report.py --glob "${OUT_DIR}/A_*.json,${OUT_DIR}/D_*.json" --rank 0 \
  --baseline-tag A_single --markdown "${OUT_DIR}/table_zero.md"

cat <<EOF

════════════════════════════════════════════════════════════
产物：
  out/comm_${BACKEND}_ws*.csv   通信带宽扫描（A 组）
  out/table_scaling.md          DDP 扩展效率（B 组）
  out/table_zero.md             显存与 ZeRO 阶梯（B' 与 D 组）

四个要能自己讲清楚的问题：
  1. A 组测出的卡间带宽，是单卡显存带宽的百分之几？
  2. B 组 8 卡的扩展效率是多少？缺口和 A 组的带宽对得上吗？
  3. C 组 TP 的加速比 —— 如果 < 1，把 all_reduce 的字节数除以 A 组的带宽，
     能不能解释掉这个差值？
  4. D 组里 ZeRO-0→1 省的显存，和 1→2、2→3 相比大多少？为什么第一级最划算？
EOF
