#!/usr/bin/env bash
# =============================================================================
# run_variants.sh —— 拧编译开关，看 IR 与机器码怎么变
# -----------------------------------------------------------------------------
# 前两个脚本回答「编译器做了什么」，这个脚本回答「我能改什么」。
# 三组对照，各自对应 00-end-to-end-pipeline.md 里的一条断链：
#
#   A  静态 vs 动态 shape        ── 站 ①：形状信息丢了，后面全链路补不回来
#   B  默认 CPU vs 指定 ISA      ── 站 ⑥：向量宽度是被目标特性决定的
#   C  把 kernel 的机器码挖出来  ── 站 ⑥⑦：IREE 的叶子就是 LLVM 的入口
#
# 产物：out/variants/
# 教材：iree-learning-guide 第 3 章 / llvm-learning-guide 第 5 章
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if ! have_iree; then skip_no_iree; exit 2; fi

V="${OUT_DIR}/variants"
# shellcheck disable=SC2206
TARGET_FLAGS=(${IREE_TARGET_FLAGS})

hr()  { printf '%s\n' "────────────────────────────────────────────────────────────"; }
sec() { echo; hr; echo "$1"; hr; }
show(){ sed 's/^/    /'; }

# 探测某个标志本版本是否存在
flag_ok() {
  local f="$1" probe="${V}/.probe.mlir"
  printf 'func.func @p(%%a: tensor<f32>) -> tensor<f32> { return %%a : tensor<f32> }\n' > "${probe}"
  "${IREE_COMPILE}" "${TARGET_FLAGS[@]}" "${f}" "${probe}" -o /dev/null 2>/dev/null
  local rc=$?; rm -f "${probe}"; return ${rc}
}

# ===========================================================================
sec "A｜静态 shape vs 动态 shape：站 ① 那条『丢了补不回来』的实验证据"

for pair in "tiny_mlp:static" "tiny_mlp_dynamic:dynamic"; do
  stem="${pair%%:*}"; tag="${pair##*:}"
  for ph in flow stream; do
    "${IREE_COMPILE}" "${TARGET_FLAGS[@]}" --compile-to="${ph}" \
      "${MODELS_DIR}/${stem}.mlir" -o "${V}/${tag}.${ph}.mlir" 2>/dev/null
  done
done

if [[ -f "${V}/static.stream.mlir" && -f "${V}/dynamic.stream.mlir" ]]; then
  s_lines=$(wc -l < "${V}/static.stream.mlir")
  d_lines=$(wc -l < "${V}/dynamic.stream.mlir")
  s_idx=$(grep -cE 'arith\.(muli|index_cast|maxsi)|util\.align' "${V}/static.stream.mlir"  || true)
  d_idx=$(grep -cE 'arith\.(muli|index_cast|maxsi)|util\.align' "${V}/dynamic.stream.mlir" || true)
  echo
  printf '  %-28s %-12s %-12s\n' "指标（stream 相位）" "静态[2,3]" "动态[?,3]"
  printf '  %-28s %-12s %-12s\n' "----------------------------" "----------" "----------"
  printf '  %-28s %-12s %-12s\n' "IR 行数"                  "${s_lines}" "${d_lines}"
  printf '  %-28s %-12s %-12s\n' "运行期尺寸计算指令"        "${s_idx}"   "${d_idx}"
  echo
  echo "  动态版本多出来的那些 arith.muli / index_cast，就是在【运行期重算 buffer 大小】。"
  echo "  静态版本里这些数字在编译期就折成常量了 —— 省的不只是几条指令，"
  echo "  而是让后面的 tiling、向量化、内存复用都有了确定的数可用。"
  echo
  echo "  动态版本里出现的运行期尺寸计算（前 10 条）："
  grep -nE 'arith\.muli|arith\.index_cast|util\.align' "${V}/dynamic.stream.mlir" | head -10 | show
fi

# ===========================================================================
sec "B｜目标 ISA：向量宽度是编译期被『目标特性』钉死的"

# 两代标志名都试
CPU_FEAT_FLAG=""
for f in --iree-llvmcpu-target-cpu-features --iree-llvmaot-target-cpu-features; do
  if flag_ok "${f}=+avx2"; then CPU_FEAT_FLAG="${f}"; break; fi
done

DUMP_FLAG=""
for f in --iree-hal-dump-executable-files-to --iree-hal-dump-executable-intermediates-to; do
  if flag_ok "${f}=${V}/.dumpprobe"; then DUMP_FLAG="${f}"; break; fi
done
rm -rf "${V}/.dumpprobe"

if [[ -z "${CPU_FEAT_FLAG}" || -z "${DUMP_FLAG}" ]]; then
  echo
  echo "  [跳过] 本版本未找到可用的标志组合："
  echo "    cpu-features = ${CPU_FEAT_FLAG:-<none>}"
  echo "    dump         = ${DUMP_FLAG:-<none>}"
  echo "  用 iree-compile --help | grep -i 'cpu-features\\|dump-executable' 查一下当前写法。"
else
  echo
  echo "  使用标志：${CPU_FEAT_FLAG} / ${DUMP_FLAG}"
  for combo in "baseline:" "avx2:+avx2" "avx512:+avx512f,+avx512vl"; do
    tag="${combo%%:*}"; feat="${combo##*:}"
    d="${V}/dump-${tag}"; rm -rf "${d}"; mkdir -p "${d}"
    args=("${TARGET_FLAGS[@]}" "${DUMP_FLAG}=${d}")
    [[ -n "${feat}" ]] && args+=("${CPU_FEAT_FLAG}=${feat}")
    "${IREE_COMPILE}" "${args[@]}" "${MODELS_DIR}/tiny_mlp.mlir" \
        -o "${V}/tiny_mlp.${tag}.vmfb" 2>"${V}/${tag}.log"
  done

  echo
  printf '  %-10s %-14s %-14s %-14s %-12s\n' "配置" "<4 x float>" "<8 x float>" "<16 x float>" "vmfb字节"
  printf '  %-10s %-14s %-14s %-14s %-12s\n' "--------" "------------" "------------" "------------" "----------"
  for tag in baseline avx2 avx512; do
    d="${V}/dump-${tag}"
    c4=0; c8=0; c16=0
    if compgen -G "${d}/*.ll" > /dev/null; then
      c4=$(cat "${d}"/*.ll  | grep -o '<4 x float>'  | wc -l)
      c8=$(cat "${d}"/*.ll  | grep -o '<8 x float>'  | wc -l)
      c16=$(cat "${d}"/*.ll | grep -o '<16 x float>' | wc -l)
    fi
    sz="-"; [[ -f "${V}/tiny_mlp.${tag}.vmfb" ]] && sz=$(wc -c < "${V}/tiny_mlp.${tag}.vmfb")
    printf '  %-10s %-14s %-14s %-14s %-12s\n' "${tag}" "${c4}" "${c8}" "${c16}" "${sz}"
  done
  echo
  echo "  ★ 这张表就是 00-end-to-end-pipeline.md 里那句优化目标"
  echo "    『内层循环走 8 宽向量』的最终落点：<8 x float> 出现，说明目标达成。"
  echo "    注意它不是任何一个 pass 单独决定的 —— linalg 层保住了并行语义、"
  echo "    flow 层没把循环切碎、LLVM 后端才有机会按 AVX 宽度打包。链路缺一环就退回 <4 x float>。"
fi

# ===========================================================================
sec "C｜把 kernel 的 LLVM IR / 汇编挖出来：IREE 的叶子 = LLVM 的入口"
d="${V}/dump-avx2"; [[ -d "${d}" ]] || d="${V}/dump-baseline"
if [[ -d "${d}" ]] && compgen -G "${d}/*" > /dev/null; then
  echo
  echo "  dump 目录里有什么："
  ls -1 "${d}" | head -20 | show
  echo
  ll=$(ls -1 "${d}"/*.ll 2>/dev/null | head -1)
  if [[ -n "${ll}" ]]; then
    echo "  kernel 的 LLVM IR 里的向量运算（$(basename "${ll}")）："
    grep -nE 'fmul <|fadd <|call <|llvm\.fmuladd|fcmp <|select <' "${ll}" | head -12 | show
    echo
    echo "  读法：这里的 fmul/fadd 就是 models/tiny_mlp.mlir 里 linalg.generic 体内"
    echo "  那两行 arith.mulf / arith.addf —— 中间隔了 flow/stream/hal 三层，"
    echo "  但计算本身一路原样传了下来，只是被 tile 过、向量化过。"
    echo
    echo "  ★ 接上 llvm-learning-guide：到这里就是 LLVM 指南第 5 章的输入。"
    echo "    IREE 负责『切到什么粒度』，LLVM 负责『这一块怎么编』——"
    echo "    两份指南在这个文件上握手。"
  fi
  s=$(ls -1 "${d}"/*.s 2>/dev/null | head -1)
  if [[ -n "${s}" ]]; then
    echo
    echo "  汇编里的向量指令频次（$(basename "${s}")）："
    grep -oE '\b(vmovup[sd]|vfmadd[0-9]*p[sd]|vmulp[sd]|vaddp[sd]|vmaxp[sd]|vbroadcastss)\b' "${s}" \
      | sort | uniq -c | sort -rn | head -10 | show
    echo
    echo "  vmaxps 出现 = Relu 被编成了一条向量 max，没有分支 —— 和 LLVM 指南里"
    echo "  clamp0 的 phi→select 是同一个故事，只是这次发生在 8 个元素上。"
  fi
fi

echo
hr
echo "完成。产物在 ${V}/。"
hr
