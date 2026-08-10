#!/usr/bin/env bash
# =============================================================================
# run_tests.sh —— 跑 tests/ 下的 FileCheck 测试
#   tests/*.ll 用的是标准 lit 写法（RUN: / CHECK:），和 LLVM、MLIR 官方测试完全一致。
#   本脚本做的事就是 lit 干的事的极简版：把 RUN 行里的 %s / %plugin 替换掉再执行。
#
#   跑：bash scripts/run_tests.sh
# =============================================================================
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh" >/dev/null

PLUGIN="${PROJECT_DIR}/build/passes/MyPasses.so"
TEST_DIR="${PROJECT_DIR}/tests"

[[ -f "${PLUGIN}" ]] || bash "${PROJECT_DIR}/scripts/build_passes.sh" >/dev/null
[[ -f "${PLUGIN}" ]] || { echo "[tests] 插件未构建，无法测试：${PLUGIN}" >&2; exit 1; }
[[ -n "${FILECHECK}" ]] || { echo "[tests] 未找到 FileCheck，跳过测试。" >&2; exit 0; }

pass=0; fail=0
for f in "${TEST_DIR}"/*.ll; do
  # 取出 RUN: 行，做 lit 的变量替换
  runline="$(grep -m1 '^; RUN:' "$f" | sed 's/^; RUN: *//')"
  [[ -n "${runline}" ]] || continue
  runline="${runline//%plugin/${PLUGIN}}"
  runline="${runline//%s/$f}"
  runline="${runline//FileCheck/${FILECHECK}}"
  runline="${runline//opt /${OPT} }"

  printf '  %-24s ' "$(basename "$f")"
  if bash -c "${runline}" >/dev/null 2>&1; then
    echo "PASS"; ((pass++))
  else
    echo "FAIL"
    echo "    命令: ${runline}"
    ((fail++))
  fi
done

echo
echo "[tests] 通过 ${pass}，失败 ${fail}"
[[ ${fail} -eq 0 ]]
