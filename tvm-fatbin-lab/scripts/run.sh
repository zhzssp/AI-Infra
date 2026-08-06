#!/usr/bin/env bash
# =============================================================================
# run.sh —— TVM + CUDA fatbin 一键学习流水线
#   产物：
#     out/full_run.log   全量终端日志
#     out/tvm/*          TE/融合/layout/tensorize/AutoTVM/PackedFunc
#     out/cuda/*         fatbin + cuobjdump
#     out/ANALYSIS.md    汇总报告（建议最先读）
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

exec > >(tee "${OUT_DIR}/full_run.log") 2>&1

TVM_STATUS="skipped"
FATBIN_STATUS="skipped"

echo "================================================================"
echo " tvm-fatbin-lab —— 一键跑完 TVM 核心概念 + CUDA fatbin"
echo "================================================================"
echo
echo "  主线故事："
echo "    TVM     = 同一个算法，怎么通过 schedule / 融合 / 搜索 算得快"
echo "    fatbin  = 同一份产物，怎么携带多架构变体并在运行时选择"
echo "    二者同构于「配置组合爆炸」的两种解法：搜最优 + 打包多变体"
echo

set +e
bash "${SCRIPT_DIR}/run_tvm.sh"
tvm_rc=$?
set -e
if [[ $tvm_rc -eq 0 ]]; then
  TVM_STATUS="ok"
elif [[ $tvm_rc -eq 2 ]]; then
  TVM_STATUS="missing-deps"
else
  TVM_STATUS="failed(${tvm_rc})"
fi

set +e
bash "${SCRIPT_DIR}/run_fatbin.sh"
fb_rc=$?
set -e
if [[ $fb_rc -eq 0 ]]; then
  FATBIN_STATUS="ok"
elif [[ $fb_rc -eq 2 ]]; then
  FATBIN_STATUS="missing-deps"
else
  FATBIN_STATUS="failed(${fb_rc})"
fi

# ---- ANALYSIS.md ----
ANALYSIS="${OUT_DIR}/ANALYSIS.md"
cat > "${ANALYSIS}" <<EOF
# tvm-fatbin-lab 汇总报告

> 建议**先读本文件**，再按链接打开各步骤产物。

## 运行状态

| 轨道 | 状态 | 说明 |
|------|------|------|
| TVM（6 步） | \`${TVM_STATUS}\` | 需 \`pip install apache-tvm\`（或 conda 环境） |
| CUDA fatbin | \`${FATBIN_STATUS}\` | 需 CUDA Toolkit（\`nvcc\` + \`cuobjdump\`） |

若某轨为 \`missing-deps\`，装好依赖后单独重跑：

\`\`\`bash
bash scripts/run_tvm.sh
bash scripts/run_fatbin.sh
\`\`\`

## 概念覆盖（与教材对照）

| # | 核心概念 | 产物 | 教材 |
|---|----------|------|------|
| T1 | 算法/调度分离；tile / cache_write / compute_at / vectorize | \`out/tvm/01_*.lower.txt\` + \`01_READING.md\` | tvm-learning-guide §3 |
| T2 | 四类融合规则（injective/reduction/complex/opaque） | \`out/tvm/02_relay_*fuse*.txt\` | tvm-learning-guide §2.2 |
| T3 | layout 变换与跨后端转换税 | \`out/tvm/03_*\` | 研究问题② |
| T4 | tensorize（异构接入钩子） | \`out/tvm/04_*\` | tvm-learning-guide §4 |
| T5 | AutoTVM 搜索闭环 + **log 复用** | \`out/tvm/05_*\` | 研究问题⑥ |
| T6 | PackedFunc + graph executor | \`out/tvm/06_*\` | tvm-learning-guide §6 |
| F1 | fatbin 多镜像结构 | \`out/cuda/add.fatbin\` | cuda-fatbin-learning-guide §2 |
| F2 | compute_* vs sm_*；PTX JIT 退路 | \`out/cuda/dump_*.txt\` | cuda-fatbin-learning-guide §3 |
| F3 | fatbin ↔ IREE ExecutableVariant 同构 | \`out/cuda/READING.md\` | cuda-fatbin §6 · iree §4.7 |

## 推荐阅读顺序

1. \`out/tvm/01_READING.md\` → 对比两份 matmul lower（**硬门槛**）
2. \`out/tvm/02_READING.md\` → 指出融合切开点
3. \`out/tvm/05_READING.md\` → 确认 log 复用比重新搜索快
4. \`out/cuda/READING.md\` → 对照 \`-lelf\` / \`-lptx\`
5. 回看根 README §4.1 / §5.4 与六个研究问题①②⑥

## 过关自问

- [ ] 指着两份 matmul lower，讲清至少 4 个 schedule 原语
- [ ] 四类融合各举一例能融 / 不能融
- [ ] 说明 tuning log 复用如何对抗配置组合爆炸
- [ ] 口述 fatbin 与 IREE \`ExecutableVariant\` 的同构（问题 / 打包 / 选择 / 退路）
- [ ] 一句话：TVM 擅长搜 kernel；fatbin/variant 擅长多目标打包

## 两轨如何拼成一张图

\`\`\`
同一 matmul 算子
   │
   ├─ TVM：TE compute → schedule/搜索 → 一份（或一组）快实现
   │
   └─ fatbin / IREE variant：把「多硬件实现」打进同一逻辑产物，运行时按能力选择
\`\`\`

搜索解决「哪份实现快」；多变体打包解决「发给谁、在哪台机器上能跑」。
算力网上两者都要。
EOF

echo
echo "================================================================"
echo " 完成：TVM=${TVM_STATUS}  fatbin=${FATBIN_STATUS}"
echo " 请先打开：${ANALYSIS}"
echo "================================================================"

# Exit non-zero only if a track hard-failed (not missing deps).
if [[ "${TVM_STATUS}" == failed* || "${FATBIN_STATUS}" == failed* ]]; then
  exit 1
fi
# If both missing, still succeed with report (so CI/docs cloning isn't blocked).
exit 0
