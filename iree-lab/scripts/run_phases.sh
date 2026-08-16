#!/usr/bin/env bash
# =============================================================================
# run_phases.sh —— 站 ⑤→⑦：把主角模型停在每一个编译相位上，逐相位看新增了什么
# -----------------------------------------------------------------------------
# 一句话：--compile-to=<phase> 让编译器【停在半路】把 IR 吐出来。
# 于是 flow / stream / hal / vm 这几个抽象层，从文档里的名词变成了可 diff 的文本。
#
# 产物：out/phases/*.mlir + out/PHASES.md
# 教材：docs/learning-guides/iree-learning-guide.md 第 1–3 章
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if ! have_iree; then skip_no_iree; exit 2; fi

P="${OUT_DIR}/phases"
MD="${OUT_DIR}/PHASES.md"
# shellcheck disable=SC2206
TARGET_FLAGS=(${IREE_TARGET_FLAGS})

hr()  { printf '%s\n' "────────────────────────────────────────────────────────────"; }
sec() { echo; hr; echo "$1"; hr; }
show(){ sed 's/^/    /'; }

# 按顺序尝试；不同版本相位名有出入，失败的自动跳过并记录
PHASES=(input abi preprocessing global-optimization dispatch-creation
        flow stream executable-sources executable-targets hal vm)

declare -A OK

compile_phase() {   # $1=model_stem  $2=phase
  local stem="$1" phase="$2"
  local src="${MODELS_DIR}/${stem}.mlir"
  local dst="${P}/${stem}.${phase}.mlir"
  if "${IREE_COMPILE}" "${TARGET_FLAGS[@]}" --compile-to="${phase}" \
        "${src}" -o "${dst}" 2>"${dst}.err"; then
    rm -f "${dst}.err"; OK["${stem}:${phase}"]=1; return 0
  fi
  rm -f "${dst}"; OK["${stem}:${phase}"]=0; return 1
}

sec "站 ⑤–⑦｜相位总览：同一个模型停在不同高度"
echo
echo "  abs.mlir（1 个 op）用来精读单相位；tiny_mlp.mlir（3 个 op）用来看融合。"
echo
printf '  %-24s %-10s %-10s %-10s %-10s\n' "phase" "abs行数" "mlp行数" "abs状态" "mlp状态"
printf '  %-24s %-10s %-10s %-10s %-10s\n' "------------------------" "--------" "--------" "--------" "--------"
for ph in "${PHASES[@]}"; do
  compile_phase abs      "${ph}"
  compile_phase tiny_mlp "${ph}"
  a="-"; m="-"; as="skip"; ms="skip"
  [[ ${OK["abs:${ph}"]} == 1 ]]      && { a=$(wc -l < "${P}/abs.${ph}.mlir");      as="ok"; }
  [[ ${OK["tiny_mlp:${ph}"]} == 1 ]] && { m=$(wc -l < "${P}/tiny_mlp.${ph}.mlir"); ms="ok"; }
  printf '  %-24s %-10s %-10s %-10s %-10s\n' "${ph}" "${a}" "${m}" "${as}" "${ms}"
done
echo
echo "  行数本身就是结论：越往下走，同一份语义要写的字越多 ——"
echo "  多出来的字全是【原来隐含、现在必须说清】的东西（谁分配、谁同步、谁等谁）。"

# ---------------------------------------------------------------------------
sec "① input / abi｜边界被写死成显式签名"
if [[ ${OK["abs:abi"]:-0} == 1 ]]; then
  echo "  abs 在 abi 之后："
  cat "${P}/abs.abi.mlir" | show
  echo
  echo "  读法：func.func 还在，但外部可见的那层已经被 ABI 规范化。"
  echo "  这一步固化的是【宿主怎么调它】，之后所有相位都不许再改签名。"
fi

# ---------------------------------------------------------------------------
sec "② flow｜dispatch 划分：这里决定了 kernel 边界（站 ②③ 的落地）"
if [[ ${OK["tiny_mlp:flow"]:-0} == 1 ]]; then
  echo "  tiny_mlp 被切成了几个 dispatch？"
  echo
  grep -nE 'flow\.dispatch|flow\.executable|util\.func public' \
       "${P}/tiny_mlp.flow.mlir" | head -20 | show
  n_disp=$(grep -cE '= flow\.dispatch ' "${P}/tiny_mlp.flow.mlir" || true)
  n_exec=$(grep -cE 'flow\.executable ' "${P}/tiny_mlp.flow.mlir" || true)
  echo
  echo "  dispatch 调用点 = ${n_disp}，executable = ${n_exec}"
  echo
  echo "  ★ 源码里写了 4 个 linalg.generic（广播 b / Gemm / Relu / Add bias2），"
  echo "    这里如果少于 4 个 dispatch，就说明【融合发生了】——"
  echo "    Relu 和 Add 是逐元素的，被吸进 Gemm 的 dispatch 里，中间张量不落 DRAM。"
  echo "    这正是 00-end-to-end-pipeline.md 站 ③ 那条优化目标在 IREE 里的兑现方式。"
  echo
  echo "  每个 executable 内部的计算体（前 40 行）："
  awk '/flow\.executable/,0' "${P}/tiny_mlp.flow.mlir" | head -40 | show
fi

# ---------------------------------------------------------------------------
sec "③ stream｜资源与时序：张量变成有生命周期的 buffer"
if [[ ${OK["tiny_mlp:stream"]:-0} == 1 ]]; then
  echo "  出现了哪些 stream 概念："
  grep -oE 'stream\.[a-z_.]+' "${P}/tiny_mlp.stream.mlir" | sort | uniq -c | sort -rn | head -15 | show
  echo
  echo "  资源的生命周期标注（Lifetime 是 stream 层最核心的新增信息）："
  grep -oE '!stream\.resource<[a-z]+>' "${P}/tiny_mlp.stream.mlir" | sort | uniq -c | show
  echo
  echo "  读法："
  echo "    external  跨边界、宿主能看见 → 不能随便复用"
  echo "    transient 一次执行内的中间量 → 可以和别人共享同一块内存"
  echo "    variable  跨执行存活（权重）"
  echo "  flow 层只说了『算什么』，stream 层第一次说清【谁占内存、占多久、谁等谁】。"
fi

# ---------------------------------------------------------------------------
sec "④ hal｜设备无关 → 设备相关：executable 里出现真的机器码"
if [[ ${OK["tiny_mlp:hal"]:-0} == 1 ]]; then
  echo "  HAL 对象出现的频次："
  grep -oE 'hal\.[a-z_.]+' "${P}/tiny_mlp.hal.mlir" | sort | uniq -c | sort -rn | head -15 | show
  echo
  echo "  目标变体（ExecutableVariant）——『同一个 kernel 的多份实现』就挂在这："
  grep -nE 'hal\.executable\.variant|hal\.executable\.target|hal\.executable private' \
       "${P}/tiny_mlp.hal.mlir" | head -10 | show
  echo
  echo "  ★ 和 cuda-fatbin 指南对上了：fatbin 里按 sm_xx 存多份 cubin，"
  echo "    IREE 里按 target 存多份 variant —— 同一个『一次编译多处运行』的问题，"
  echo "    两套系统给的是同构的答案。"
fi

# ---------------------------------------------------------------------------
sec "⑤ vm｜宿主侧字节码：调度逻辑本身也被编译了"
if [[ ${OK["tiny_mlp:vm"]:-0} == 1 ]]; then
  echo "  vm 模块的骨架："
  grep -nE 'vm\.module|vm\.func|vm\.import|vm\.rodata' "${P}/tiny_mlp.vm.mlir" | head -20 | show
  echo
  echo "  读法：到这一层，『先分配 buffer、再 dispatch、再等 semaphore』这套宿主逻辑"
  echo "  不再是 C++ 里的一段代码，而是一串可序列化的 VM 指令 ——"
  echo "  所以 IREE 的部署产物是单个 .vmfb，不需要宿主再链接一个调度器。"
fi

# ---------------------------------------------------------------------------
sec "⑥ 相邻两相的 diff：新增的信息到底是什么"
if [[ ${OK["abs:flow"]:-0} == 1 && ${OK["abs:stream"]:-0} == 1 ]]; then
  echo "  abs：flow → stream（只看新增行，前 25 行）"
  diff <(sed 's/[[:space:]]\+/ /g' "${P}/abs.flow.mlir") \
       <(sed 's/[[:space:]]\+/ /g' "${P}/abs.stream.mlir") \
    | grep '^>' | head -25 | show
  echo
  echo "  一个 math.absf 而已，stream 层却多出这么多行 —— 多出来的全是内存与时序。"
  echo "  这就是『抽象层不是包装，是补信息』最直观的证据。"
fi

# ---------------------------------------------------------------------------
sec "生成阅读指南 out/PHASES.md"
{
  echo "# IREE 相位速查（由 scripts/run_phases.sh 生成）"
  echo
  echo "模型：\`models/abs.mlir\`（1 op，精读用）、\`models/tiny_mlp.mlir\`（3 op，看融合用）"
  echo
  echo "编译标志：\`${IREE_TARGET_FLAGS}\`"
  echo
  echo "| 相位 | 这一相【新增】的信息 | abs 行数 | tiny_mlp 行数 |"
  echo "|------|----------------------|---------:|--------------:|"
  desc_input="还是输入方言，只做合法性归一"
  desc_abi="外部调用签名被固化"
  desc_preprocessing="用户可插入的预处理钩子"
  desc_global_optimization="全图级别的常量折叠 / 形状传播"
  desc_dispatch_creation="切 kernel 边界的准备工作"
  desc_flow="**dispatch 划分与融合**：kernel 边界在此固化"
  desc_stream="**资源生命周期与时序**：谁占内存、谁等谁"
  desc_executable_sources="每个 kernel 的独立源码副本"
  desc_executable_targets="按目标特化后的 kernel"
  desc_hal="**设备对象**：buffer / command_buffer / semaphore / variant"
  desc_vm="**宿主调度字节码**：调度逻辑本身被编译"
  for ph in "${PHASES[@]}"; do
    key="desc_${ph//-/_}"
    a="—"; m="—"
    [[ ${OK["abs:${ph}"]:-0} == 1 ]]      && a=$(wc -l < "${P}/abs.${ph}.mlir")
    [[ ${OK["tiny_mlp:${ph}"]:-0} == 1 ]] && m=$(wc -l < "${P}/tiny_mlp.${ph}.mlir")
    echo "| \`${ph}\` | ${!key:-} | ${a} | ${m} |"
  done
  echo
  echo "## 三个必看的点"
  echo
  echo "1. **flow 相位数 dispatch 个数**。源码 4 个 \`linalg.generic\`，"
  echo "   若 dispatch 少于 4 个 → 融合发生了，中间张量没落 DRAM。"
  echo "2. **stream 相位数 \`!stream.resource<transient>\`**。transient 越多，"
  echo "   说明编译器识别出的『可复用中间内存』越多。"
  echo "3. **hal 相位看 variant**。一个 kernel 多份 variant ↔ fatbin 一个 kernel 多份 cubin。"
  echo
  echo "配套阅读：\`docs/learning-guides/iree-learning-guide.md\` 第 1–3 章、"
  echo "\`docs/learning-guides/00-end-to-end-pipeline.md\` 站 ②③⑦。"
} > "${MD}"
echo "  已写出：${MD}"

echo
hr
echo "完成。IR 全在 ${P}/，建议按 flow → stream → hal 的顺序 diff 着读。"
hr
