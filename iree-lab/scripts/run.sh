#!/usr/bin/env bash
# =============================================================================
# run.sh —— 一键跑完 iree-lab 三步
# 用法：bash scripts/run.sh [2>&1 | tee out/all.log]
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for s in run_phases.sh run_execute.sh run_variants.sh; do
  echo
  echo "════════════════════════════════════════════════════════════"
  echo "▶ ${s}"
  echo "════════════════════════════════════════════════════════════"
  bash "${HERE}/${s}"
  rc=$?
  [[ ${rc} -eq 2 ]] && { echo "（缺依赖，后续步骤同样会跳过）"; break; }
done

echo
echo "全部结束。阅读顺序建议："
echo "  1. out/PHASES.md        —— 相位速查表"
echo "  2. out/phases/*.mlir    —— 按 flow → stream → hal 顺序 diff"
echo "  3. out/variants/        —— 三组对照实验"
