#!/usr/bin/env bash
# Build dual-arch fatbins and dump with cuobjdump.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

CUDA_OUT="${OUT_DIR}/cuda"
SRC="${PROJECT_DIR}/cuda/add.cu"
mkdir -p "${CUDA_OUT}"

if ! have_cuda; then
  cat <<EOF
[SKIP] 未找到 nvcc / cuobjdump。
请安装 CUDA Toolkit，并确保二者在 PATH 中。
本步骤对应 docs/learning-guides/cuda-fatbin-learning-guide.md 第 8 章。
EOF
  exit 2
fi

# Allow override; default to two common arches. Older toolkits: export SM_A=70 SM_B=75
: "${SM_A:=75}"
: "${SM_B:=80}"

echo "[fatbin] 使用架构 sm_${SM_A} + sm_${SM_B}（可用 SM_A/SM_B 覆盖）"

set +e
"${NVCC_BIN}" -fatbin "${SRC}" -o "${CUDA_OUT}/add.fatbin" \
  -gencode "arch=compute_${SM_A},code=sm_${SM_A}" \
  -gencode "arch=compute_${SM_B},code=sm_${SM_B}" 2>"${CUDA_OUT}/nvcc_sass_only.err"
rc1=$?
set -e

if [[ $rc1 -ne 0 ]]; then
  echo "[WARN] 默认架构失败，尝试 sm_70 + sm_75 …"
  SM_A=70; SM_B=75
  "${NVCC_BIN}" -fatbin "${SRC}" -o "${CUDA_OUT}/add.fatbin" \
    -gencode "arch=compute_${SM_A},code=sm_${SM_A}" \
    -gencode "arch=compute_${SM_B},code=sm_${SM_B}"
fi

# With PTX fallback on the second arch.
"${NVCC_BIN}" -fatbin "${SRC}" -o "${CUDA_OUT}/add_with_ptx.fatbin" \
  -gencode "arch=compute_${SM_A},code=sm_${SM_A}" \
  -gencode "arch=compute_${SM_B},code=[sm_${SM_B},compute_${SM_B}]"

{
  echo "# cuobjdump — SASS-only fatbin"
  echo "## -lelf"
  "${CUOBJDUMP_BIN}" -lelf "${CUDA_OUT}/add.fatbin" || true
  echo
  echo "## -lptx"
  "${CUOBJDUMP_BIN}" -lptx "${CUDA_OUT}/add.fatbin" || true
} | tee "${CUDA_OUT}/dump_sass_only.txt"

{
  echo "# cuobjdump — fatbin with PTX fallback"
  echo "## -lelf"
  "${CUOBJDUMP_BIN}" -lelf "${CUDA_OUT}/add_with_ptx.fatbin" || true
  echo
  echo "## -lptx"
  "${CUOBJDUMP_BIN}" -lptx "${CUDA_OUT}/add_with_ptx.fatbin" || true
} | tee "${CUDA_OUT}/dump_with_ptx.txt"

cat > "${CUDA_OUT}/READING.md" <<EOF
# CUDA fatbin 阅读指引

## 产物

| 文件 | 含义 |
|------|------|
| \`add.fatbin\` | 仅两档 SASS（\`sm_${SM_A}\` / \`sm_${SM_B}\`） |
| \`add_with_ptx.fatbin\` | 同上，且第二档附带 \`compute_${SM_B}\` PTX 退路 |
| \`dump_sass_only.txt\` / \`dump_with_ptx.txt\` | \`cuobjdump -lelf/-lptx\` 输出 |

## 过关检查

- [ ] \`-lelf\` 能看到 **两个**不同的 \`sm_*\`
- [ ] SASS-only 的 \`-lptx\` 为空或没有对应档；with_ptx 能看到 \`compute_*\`
- [ ] 能解释：只有 sm_75/80、无 PTX 时，在更新的 GPU 上可能失败；有 \`compute_80\` PTX 时可 JIT

## 与 IREE ExecutableVariant 同构

| 维度 | CUDA fatbin | IREE |
|------|-------------|------|
| 编译期 | 多次 \`-gencode\` | 每个 variant 独立 lowering |
| 产物 | 一个 fatbin 多 image | 一个 \`hal.executable\` 多 \`variant\` |
| 选择键 | compute capability | target + condition |
| 退路 | PTX JIT | fallback export / 更通用 variant |

详见 [\`docs/learning-guides/cuda-fatbin-learning-guide.md\`](../../docs/learning-guides/cuda-fatbin-learning-guide.md) §6。
EOF

echo
echo "[OK] fatbin 步骤完成。先读 out/cuda/READING.md，再看 dump_*.txt。"
