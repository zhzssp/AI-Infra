#!/usr/bin/env bash
# =============================================================================
# run_cpu_smoke.sh —— 无卡冒烟：几十秒跑完，验证骨架与集合通信语义
# -----------------------------------------------------------------------------
# 这个脚本的存在意义：**分布式训练里真正难的三件事——谁在跟谁通信、通信的是
# 什么、通信之后不变量是什么——和有没有 GPU 毫无关系。** 先在笔记本上把它们
# 学明白，上机时只需把 --device cuda --backend nccl 换上去。
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/env.sh"
cd "${PROJECT_DIR}"

have_torch || { skip_no_torch; exit 2; }

sec "① 主角数值锚点：放大版的骨架和另外四个 lab 是同一张图"
"${PYTHON_BIN}" model.py

sec "② 16Φ 显存账（纯手算，不碰 GPU）"
"${PYTHON_BIN}" mem_ledger.py --hidden "${SMOKE_HIDDEN}" --layers "${SMOKE_LAYERS}" \
  --precision mixed --zero 0 --ladder --n 8
echo
echo "  ★ 注意 ZeRO-0→1 差 6Φ，而 2→3 只差 1Φ。第一级收益最大。"
echo "    这直接决定了「卡少时该选哪一级」。"

sec "③ 单进程 baseline（CPU）"
"${PYTHON_BIN}" train_single.py --device cpu \
  --hidden "${SMOKE_HIDDEN}" --layers "${SMOKE_LAYERS}" --batch 8 --seq 128 --steps 20 \
  --tag smoke_single

sec "④ 两进程 DDP（gloo），验证 all_reduce 的两条语义"
if [[ -z "${TORCHRUN}" ]]; then
  echo "[SKIP] 没找到 torchrun"
else
  "${TORCHRUN}" --standalone --nproc_per_node=2 train_ddp.py \
    --device cpu --backend gloo \
    --hidden "${SMOKE_HIDDEN}" --layers "${SMOKE_LAYERS}" --batch 8 --seq 128 \
    --steps 20 --check-consistency --tag smoke_ddp
  echo
  echo "  ★ param_diff 必须是 0：两个 rank 喂的是【不同】数据，但梯度 all_reduce"
  echo "    取平均后参数保持逐比特一致 —— 这是 DDP 的核心不变量。"
  echo "  ★ grad_diff < 1e-6 证明 all_reduce 做的是【平均】而不是求和。"
fi

sec "⑤ 集合通信拐点（gloo，两进程）"
if [[ -n "${TORCHRUN}" ]]; then
  "${TORCHRUN}" --standalone --nproc_per_node=2 comm_bench.py \
    --device cpu --backend gloo --ops all_reduce,all_gather \
    --min-exp 10 --max-exp 22 --iters 10 \
    --out "${OUT_DIR}/comm_gloo_ws2.csv"
  echo
  echo "  ★ 小消息段涨 16 倍字节耗时几乎不变（延迟主导），大消息段才线性（带宽主导）。"
  echo "    拐点位置就是 DDP 要做梯度分桶的理由。"
fi

sec "⑥ 最小张量并行的数值等价（本条主判据不需要卡）"
if [[ -n "${TORCHRUN}" ]]; then
  "${TORCHRUN}" --standalone --nproc_per_node=2 tp_min.py \
    --device cpu --backend gloo --hidden 256 --check col,row,ffn --count-collectives \
    || echo "  （返回非 0 表示某项超出 atol，去看上面哪一行是 ✘）"
  echo
  echo "  ★ 列切配 all_gather、行切配 all_reduce —— 分片状态决定了补哪个 collective。"
  echo "    Megatron FFN 整块只要 1 次通信，这就是「先列切再行切」的理由。"
fi

sec "⑦ 汇总"
"${PYTHON_BIN}" report.py --glob "${OUT_DIR}/smoke_*.json" || true

cat <<EOF

════════════════════════════════════════════════════════════
冒烟完成。这一轮验证了「逻辑正确性」，但**拿不到任何比值型结论**：
  · CPU 上 max_memory_allocated 无意义 → 显存判据全部缺失
  · gloo 没有独立通信 stream → 通信与计算的重叠观察不到
  · 同机多进程受 CPU 核数制约 → 加速比不具参考性

有卡之后跑 scripts/run_8gpu_wall.sh，那里才有可写进报告的数字。
EOF
