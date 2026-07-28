#!/usr/bin/env bash
# =============================================================================
# run.sh —— 手动运行三个演示，肉眼观察 MLIR 三大机制
# -----------------------------------------------------------------------------
#   演示 1  解析回显      验证 dialect 的 Parser/Printer（TableGen 自动生成）
#   演示 2  --canonicalize  验证 fold()：常量折叠（10+20 → 30）
#   演示 3  --toy-simplify   验证 RewritePattern：代数化简（x*1 → x, x+0 → x）
#   演示 4  --toy-to-low     验证 lowering：高层 toy.* → 低层 low.*
#   演示 5  分层优化对比     同一个 x*4：toy 层不动，降到 low 层后削减为 x<<2
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

TOY_OPT="${TOY_BUILD_DIR}/bin/toy-opt"
if [[ ! -x "${TOY_OPT}" ]]; then
  echo "[run] 未找到 toy-opt，请先运行：scripts/build.sh" >&2
  exit 1
fi

sep() { printf '\n========== %s ==========\n' "$1"; }

sep "演示 1：解析并回显（dialect Parser/Printer）"
"${TOY_OPT}" "${TOY_PROJECT_DIR}/test/ops.mlir"

sep "演示 2：常量折叠 --canonicalize（fold 机制：toy.add/mul 变常量）"
"${TOY_OPT}" "${TOY_PROJECT_DIR}/test/canonicalize.mlir" --canonicalize

sep "演示 3：代数化简 --toy-simplify（RewritePattern：x*1=x, x+0=x）"
"${TOY_OPT}" "${TOY_PROJECT_DIR}/test/simplify.mlir" --toy-simplify

sep "演示 4：降低 --toy-to-low（高层 toy.* 改写成低层 low.*）"
"${TOY_OPT}" "${TOY_PROJECT_DIR}/test/lowering.mlir" --toy-to-low

sep "演示 5-a：x*4 在【高层 toy 层】做代数化简 —— 纹丝不动（高层没有移位概念）"
"${TOY_OPT}" "${TOY_PROJECT_DIR}/test/strength.mlir" --toy-simplify

sep "演示 5-b：同一个 x*4 降到【低层 low 层】做强度削减 —— x*4 变成 x<<2"
"${TOY_OPT}" "${TOY_PROJECT_DIR}/test/strength.mlir" --toy-to-low --low-strength-reduce

printf '\n[run] 五个演示运行完毕。对比演示 5-a 与 5-b，即可看清不同层级的优化分工。\n'
