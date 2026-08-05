#!/usr/bin/env bash
# =============================================================================
# run.sh —— 手动运行三个演示，肉眼观察 MLIR 三大机制
# -----------------------------------------------------------------------------
#   演示 1  解析回显      验证 dialect 的 Parser/Printer（TableGen 自动生成）
#   演示 2  --canonicalize  验证 fold()：常量折叠（10+20 → 30）
#   演示 3  --toy-simplify   验证 RewritePattern：代数化简（x*1 → x, x+0 → x）
#   演示 4  --toy-to-low     验证 lowering：高层 toy.* → 低层 low.*
#   演示 5  分层优化对比     同一个 x*4：toy 层不动，降到 low 层后削减为 x<<2
#   演示 6  Region 嵌套      Pass 自动递归进入区域，化简 toy.repeat 体内的 x*1
#   演示 7  自定义类型        !toy.num 的装箱/拆箱被 fold 成恒等
#   演示 8  Dialect Conversion  同一个降低的"正规写法"：合法性 + 类型转换
#   演示 9  接口与 cost model  同一个通用 Pass 给 toy/low 两层统一算代价
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

TOY_OPT="${TOY_BUILD_DIR}/bin/toy-opt"
if [[ ! -x "${TOY_OPT}" ]]; then
  echo "[run] 未找到 toy-opt，请先运行：scripts/build.sh" >&2
  exit 1
fi

# 最终输出落盘目录（与 build/ 同级的 out/）
OUT_DIR="${TOY_PROJECT_DIR}/out"
mkdir -p "${OUT_DIR}"

# 全量日志：把本脚本终端所见的一切（横幅 + 最终 IR + 调试 trace）都镜像到
# out/full_run.log。exec 之后的所有 stdout/stderr 都会经过这个 tee。
exec > >(tee "${OUT_DIR}/full_run.log") 2>&1

echo "############################################################"
echo "# toy-opt Pipeline 演示"
echo "#   构建: scripts/build.sh           -> build/bin/toy-opt"
echo "#   输入: test/*.mlir  (MLIR IR 文本，统一 Operation 树)"
echo "#   处理: 按 --xxx Pass 逐层降低/优化（含 IR 快照与 rewrite 日志）"
echo "#   输出: 最终 IR -> stdout；调试 trace（Pass 快照/rewrite 日志）-> stderr"
echo "#   落盘: 每个演示保存两份 —— out/<demo>.mlir(最终IR) + out/<demo>.log(调试trace)"
echo "#         另存一份 out/full_run.log（终端所见的全部内容：IR + trace 合并）"
echo "############################################################"
echo "# 输出目录: ${OUT_DIR}"

sep() { printf '\n========== %s ==========\n' "$1"; }

# 运行一个演示，同时把两路输出都落盘（且终端实时可见）：
#   stdout（最终 IR）      -> tee 到 out/<demo>.mlir
#   stderr（调试 trace）   -> tee 到 out/<demo>.log
# 用 process substitution 让两路各自 tee，互不污染。
run_demo() {
  local out_file="$1"; shift
  local log_file="${out_file%.mlir}.log"
  "${TOY_OPT}" "$@" \
    > >(tee "${out_file}") \
    2> >(tee "${log_file}" >&2)
  wait   # 等待两个 tee 子进程把文件写完，避免顺序错乱
}

sep "演示 1：解析并回显（dialect Parser/Printer）"
run_demo "${OUT_DIR}/demo1_parse.mlir" "${TOY_PROJECT_DIR}/test/ops.mlir"

sep "演示 2：常量折叠 --canonicalize（fold 机制：toy.add/mul 变常量）"
run_demo "${OUT_DIR}/demo2_canonicalize.mlir" "${TOY_PROJECT_DIR}/test/canonicalize.mlir" --canonicalize

sep "演示 3：代数化简 --toy-simplify（RewritePattern：x*1=x, x+0=x）"
run_demo "${OUT_DIR}/demo3_simplify.mlir" "${TOY_PROJECT_DIR}/test/simplify.mlir" --toy-simplify

sep "演示 4：降低 --toy-to-low（高层 toy.* 改写成低层 low.*）"
run_demo "${OUT_DIR}/demo4_lowering.mlir" "${TOY_PROJECT_DIR}/test/lowering.mlir" --toy-to-low

sep "演示 5-a：x*4 在【高层 toy 层】做代数化简 —— 纹丝不动（高层没有移位概念）"
run_demo "${OUT_DIR}/demo5a_simplify.mlir" "${TOY_PROJECT_DIR}/test/strength.mlir" --toy-simplify

sep "演示 5-b：同一个 x*4 降到【低层 low 层】做强度削减 —— x*4 变成 x<<2"
run_demo "${OUT_DIR}/demo5b_strength.mlir" "${TOY_PROJECT_DIR}/test/strength.mlir" --toy-to-low --low-strength-reduce

sep "演示 6：Region 嵌套 —— Pass 自动递归进区域，化简 toy.repeat 体内的 x*1"
run_demo "${OUT_DIR}/demo6_region.mlir" "${TOY_PROJECT_DIR}/test/region.mlir" --toy-simplify

sep "演示 7：自定义类型 !toy.num —— unbox(box(x)) 被 fold 成 x"
run_demo "${OUT_DIR}/demo7_types.mlir" "${TOY_PROJECT_DIR}/test/types.mlir" --canonicalize

sep "演示 8：Dialect Conversion 版降低 —— ConversionTarget + TypeConverter"
run_demo "${OUT_DIR}/demo8_convert.mlir" "${TOY_PROJECT_DIR}/test/convert.mlir" --toy-to-low-convert

sep "演示 9-a：接口/代价模型 —— 强度削减【前】，low.mul 代价 5"
run_demo "${OUT_DIR}/demo9a_cost_before.mlir" "${TOY_PROJECT_DIR}/test/cost.mlir" --toy-to-low --toy-print-cost

sep "演示 9-b：接口/代价模型 —— 强度削减【后】，low.shl 代价 1"
run_demo "${OUT_DIR}/demo9b_cost_after.mlir" "${TOY_PROJECT_DIR}/test/cost.mlir" --toy-to-low --low-strength-reduce --toy-print-cost

printf '\n[run] 九个演示运行完毕。\n'
printf '[run]   对比演示 5-a 与 5-b：不同层级的优化分工（Multi-Level 的价值）。\n'
printf '[run]   对比演示 4 与 8    ：贪心改写 vs Dialect Conversion 两种 lowering 写法。\n'
printf '[run]   对比演示 9-a 与 9-b：同一个接口驱动的 Pass 量化出优化收益（5 -> 1）。\n'
printf '[run] 各演示的最终 IR 已保存到：%s\n' "${OUT_DIR}"
