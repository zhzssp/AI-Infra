#!/usr/bin/env bash
# =============================================================================
# run.sh —— ONNX IR/ORT EP + ExecuTorch Partitioner 一键学习流水线
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

exec > >(tee "${OUT_DIR}/full_run.log") 2>&1

ONNX_STATUS="skipped"
ET_STATUS="skipped"

echo "================================================================"
echo " onnx-delegate-lab —— 子图划分与多后端委托"
echo "================================================================"
echo
echo "  主线故事："
echo "    ONNX/ORT EP  = Session 构建期：GetCapability → Compile → 派发"
echo "    ExecuTorch   = 导出期：Partitioner 打 tag → blob → call_delegate"
echo "    对照 IREE    = 编译期：flow.dispatch（见 iree-learning-guide）"
echo "    正交对照     = tvm-fatbin-lab（划完怎么算得快 / 多变体打包）"
echo

set +e
bash "${SCRIPT_DIR}/run_onnx.sh"
onnx_rc=$?
set -e
if [[ $onnx_rc -eq 0 ]]; then ONNX_STATUS="ok"
elif [[ $onnx_rc -eq 2 ]]; then ONNX_STATUS="missing-deps"
else ONNX_STATUS="failed(${onnx_rc})"; fi

set +e
bash "${SCRIPT_DIR}/run_executorch.sh"
et_rc=$?
set -e
if [[ $et_rc -eq 0 ]]; then ET_STATUS="ok"
elif [[ $et_rc -eq 2 ]]; then ET_STATUS="missing-deps"
else ET_STATUS="failed(${et_rc})"; fi

ANALYSIS="${OUT_DIR}/ANALYSIS.md"
cat > "${ANALYSIS}" <<EOF
# onnx-delegate-lab 汇总报告

> 建议**先读本文件**，再打开各步骤 \`*_READING.md\`。

## 运行状态

| 轨道 | 状态 | 依赖 |
|------|------|------|
| ONNX + ORT EP | \`${ONNX_STATUS}\` | \`pip install onnx onnxruntime numpy\` |
| ExecuTorch Partitioner | \`${ET_STATUS}\` | 可选 \`executorch\`；未装则概念模拟仍产出报告 |

分轨重跑：

\`\`\`bash
bash scripts/run_onnx.sh
bash scripts/run_executorch.sh
\`\`\`

## 概念覆盖

| # | 核心概念 | 产物 | 教材 |
|---|----------|------|------|
| O1 | Model/Graph/Node；initializer vs input | \`out/onnx/01_*\` | onnx-learning-guide §2 |
| O2 | helper 改图；checker / shape infer | \`out/onnx/02_*\` | onnx-learning-guide §6 |
| O3 | ORT EP 分区观察；四类边界代价 | \`out/onnx/03_*\` | onnx §7；研究问题①② |
| E1 | Partitioner 契约；per-node vs connected | \`out/executorch/01_*\` | executorch-learning-guide §4–7 |
| E2 | 三系统对照表 | \`out/executorch/02_THREE_SYSTEMS.md\` | executorch §8；foundations §3.4 |

## 推荐阅读顺序

1. \`out/onnx/01_READING.md\` — 默画结构层次
2. \`out/onnx/03_READING.md\` — 指着分区说边界
3. \`out/executorch/01_READING.md\` — 对比两种 tag 策略的边界数
4. \`out/executorch/02_THREE_SYSTEMS.md\` — 钉进 ORT / ET / IREE 一张表
5. 回看根 README §4.2 / §6 问题①②；对照 \`tvm-fatbin-lab\` 的融合步骤

## 过关自问

- [ ] 默画 \`ModelProto → GraphProto → Node\`，并区分 initializer / runtime input
- [ ] 说出 ORT 三拍：GetCapability → Compile → 派发
- [ ] 指着一次分区（或概念表）回答：边界在哪、四类代价可能是哪几类
- [ ] 解释 Partitioner 为何 partition 期不能乱改图
- [ ] 用一张表对比 ExecuTorch / ORT / IREE 的**决策时刻**

## 与其它动手项目的分工

\`\`\`
llvm-hello-compile     单层 IR + Pass
mlir-toy-dialect       多层 dialect + 渐进 lowering
tvm-fatbin-lab         算得快 + 多变体打包
onnx-delegate-lab      ★ 谁来划分 + 边界代价（本项目）
\`\`\`
EOF

echo
echo "================================================================"
echo " 完成：ONNX=${ONNX_STATUS}  ExecuTorch=${ET_STATUS}"
echo " 请先打开：${ANALYSIS}"
echo "================================================================"

if [[ "${ONNX_STATUS}" == failed* || "${ET_STATUS}" == failed* ]]; then
  exit 1
fi
exit 0
