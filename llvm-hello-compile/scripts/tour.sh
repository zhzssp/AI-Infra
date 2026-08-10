#!/usr/bin/env bash
# =============================================================================
# tour.sh —— 核心要点巡礼：一条命令走完 docs/llvm-learning-guide.md 的所有必学点
# -----------------------------------------------------------------------------
#   run.sh  回答「一份源码怎么变成可执行文件」（链路）
#   tour.sh 回答「学习指南 §9.1 那 11 条，分别长什么样」（要点）
#
#   场景：src/kernel.c —— 一个迷你 AI 算子（axpy + relu 求和），外加
#         src/poison_demo.ll（手写 IR 看 poison）与 src/mini.td（看 TableGen）。
#
#   每一站的格式固定：
#     [要点] 学习指南的哪一节        [命令] 你可以自己重跑的那行
#     [观察] 应该在输出里看到什么     ← 看不到就说明环境或版本有差异
#
#   产物：out/tour/*（IR / MIR / 汇编 / 报告 TOUR.md），全量日志 out/tour/tour.log
# =============================================================================
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

SRC="${PROJECT_DIR}/src/kernel.c"
POISON_LL="${PROJECT_DIR}/src/poison_demo.ll"
MINI_TD="${PROJECT_DIR}/src/mini.td"
T="${OUT_DIR}/tour"
PLUGIN="${PROJECT_DIR}/build/passes/MyPasses.so"
mkdir -p "${T}"

exec > >(tee "${T}/tour.log") 2>&1

# --- 小工具 -----------------------------------------------------------------
stop()  { printf '\n\n═══ 第 %s 站 · %s ═══\n' "$1" "$2"; }
point() { printf '  [要点] %s\n' "$*"; }
cmd()   { printf '  [命令] %s\n' "$*"; }
look()  { printf '  [观察] %s\n' "$*"; }
# show：把 stdin 装进方框打印；输入为空时给出明确提示（而不是画一个空框）
show()  {
  local buf; buf="$(cat)"
  if [[ -z "${buf}" ]]; then printf '  （无输出：可能是本机 LLVM 版本/目标不同，跳过即可）\n'; return; fi
  printf '  ┌──────────────────────────────────────────────────\n'
  printf '%s\n' "${buf}" | sed 's/^/  │ /'
  printf '  └──────────────────────────────────────────────────\n'
}
# grepshow <文件> <正则> <最多几行>
grepshow() { grep -nE "$2" "$1" 2>/dev/null | head -"${3:-6}" | show; }

CLANG_IR=( -S -emit-llvm -fno-discard-value-names )
TRIPLE="x86_64-unknown-linux-gnu"

echo "################################################################"
echo "#  LLVM 核心要点巡礼   场景: src/kernel.c"
echo "#  产物目录: ${T}"
echo "################################################################"

# =============================================================================
stop 0 "准备：把场景编成三份 IR（O0 / O2 / O2+AVX2）"
point "§2.1 IR 三态：.ll 文本 / .bc 位码 / 内存对象，三者等价可无损互转"
cmd "clang -S -emit-llvm -O0 -Xclang -disable-O0-optnone kernel.c -o 00_O0.ll"
"${CLANG}" "${CLANG_IR[@]}" -O0 -Xclang -disable-O0-optnone "${SRC}" -o "${T}/00_O0.ll"
"${CLANG}" "${CLANG_IR[@]}" -O2 "${SRC}" -o "${T}/01_O2.ll"
"${CLANG}" "${CLANG_IR[@]}" -O2 -mavx2 "${SRC}" -o "${T}/02_O2_avx2.ll" 2>/dev/null \
  || cp "${T}/01_O2.ll" "${T}/02_O2_avx2.ll"
if [[ -n "${LLVM_AS}" && -n "${LLVM_DIS}" ]]; then
  "${LLVM_AS}" "${T}/00_O0.ll" -o "${T}/00_O0.bc" && "${LLVM_DIS}" "${T}/00_O0.bc" -o "${T}/00_roundtrip.ll"
  look "文本 $(wc -c < "${T}/00_O0.ll") 字节 ↔ 位码 $(wc -c < "${T}/00_O0.bc") 字节，来回转换语义不变"
fi

# =============================================================================
stop 1 "§2.2 类型系统：iN / float / ptr(opaque) / <N x T> / 结构体"
cmd "grep -E 'alloca|%struct|<[0-9]+ x ' 01_O2.ll"
point "现代 LLVM 的指针是 opaque 的 —— 只写 ptr，不再写 i32*；类型信息靠指令自己带"
grepshow "${T}/00_O0.ll" '%struct\.|alloca (float|i32|%struct)' 6
grepshow "${T}/01_O2.ll" '<[0-9]+ x float>' 4

# =============================================================================
stop 2 "§2.3 SSA / 基本块 / terminator / phi"
cmd "opt -S -passes=mem2reg 00_O0.ll -o 03_mem2reg.ll"
"${OPT}" -S -passes=mem2reg "${T}/00_O0.ll" -o "${T}/03_mem2reg.ll"
look "alloca: $(grep -c 'alloca ' "${T}/00_O0.ll") → $(grep -c 'alloca ' "${T}/03_mem2reg.ll")；\
phi: $(grep -c ' phi ' "${T}/00_O0.ll") → $(grep -c ' phi ' "${T}/03_mem2reg.ll")"
point "每个基本块必须且只能以一条 terminator 结尾（ret/br/switch/...），所以 CFG 是显式的"
grepshow "${T}/03_mem2reg.ll" '^\s+(br |ret |switch )' 6
point "汇合点用 phi 表达：[值, 前驱块] 成对出现，且每个前驱恰好一个入口"
grepshow "${T}/03_mem2reg.ll" ' phi ' 4

# =============================================================================
stop 3 "§2.5 getelementptr：只算地址不访存；第一个索引是「跳几个整对象」"
point "relu_sum 里 t->data / t->len / data[i] 会展开成多条 GEP"
cmd "grep getelementptr 00_O0.ll"
grepshow "${T}/00_O0.ll" 'getelementptr' 6
look "inbounds 是一个语义承诺：越界即 poison。去掉它，别名分析和向量化都会变弱"

# =============================================================================
stop 4 "§2.6 属性与标志：优化能力的来源"
point "restrict → noalias；-O2 还会推导出 align / dereferenceable / nocapture"
cmd "grep 'define.*@axpy' 01_O2.ll"
grepshow "${T}/01_O2.ll" 'define.*@axpy' 4
point "浮点 contract 标志 = 允许把 a*b+c 收缩成 FMA（见 docs/notes/llvm-fma-contract.md）"
"${CLANG}" "${CLANG_IR[@]}" -O1 -ffp-contract=off  "${SRC}" -o "${T}/04_contract_off.ll"
"${CLANG}" "${CLANG_IR[@]}" -O1 -ffp-contract=fast "${SRC}" -o "${T}/05_contract_fast.ll"
cmd "clang -ffp-contract=off/fast → 对比 fmul+fadd 与 fmuladd"
echo "  ── contract=off（老老实实两条指令，两次舍入）"
grepshow "${T}/04_contract_off.ll" 'fmul|fadd' 3
echo "  ── contract=fast（收缩成一次舍入的融合乘加）"
grepshow "${T}/05_contract_fast.ll" 'fmuladd|fmul contract|fadd contract' 3

# =============================================================================
stop 5 "§2.7 poison / undef / freeze（最容易踩的坑）"
point "违反 nsw/nuw/inbounds 先产生 poison；把 poison 用到禁位才是 UB"
cmd "opt -S -passes='default<O2>' src/poison_demo.ll -o 06_poison_O2.ll"
"${OPT}" -S -passes='default<O2>' "${POISON_LL}" -o "${T}/06_poison_O2.ll" 2>/dev/null
look "and %poison, 0 数值必然是 0，但 O2 之后仍会折叠成 poison —— 这条最反直觉"
grepshow "${T}/06_poison_O2.ll" 'define|ret|freeze' 20
look "select 是毒性屏障：没被选中的那一臂即使是 poison 也不传染（if-conversion 的前提）"

# =============================================================================
stop 6 "§2.8 Metadata：丢掉它语义不变，所以承载的都是优化提示"
cmd "grep -E '!tbaa|!llvm.loop|!range' 01_O2.ll"
grepshow "${T}/01_O2.ll" '!tbaa|!llvm\.loop|!range|!alias\.scope' 6
look "从 MLIR 降下来时，!alias.scope / !llvm.loop.parallel_accesses 就是把「无依赖」告诉 LLVM 的官方通道"

# =============================================================================
stop 7 "§2.9 Intrinsics：跨越抽象层的官方逃生舱"
cmd "grep '@llvm\\.' 01_O2.ll"
grepshow "${T}/01_O2.ll" 'call .*@llvm\.' 6
look "向量规约 llvm.vector.reduce.* 是 reduction 落地的标准形式；接新硬件时用 target intrinsic 而不是扩 IR"

# =============================================================================
stop 8 "§3 New Pass Manager：四层嵌套 + 自定义 pass + PreservedAnalyses"
point "Module →(CGSCC)→ Function → Loop，每层有自己的 PassManager 与 AnalysisManager"
cmd "opt -passes='default<O2>' -debug-pass-manager 00_O0.ll -o /dev/null"
"${OPT}" -passes='default<O2>' -debug-pass-manager "${T}/00_O0.ll" -o /dev/null 2>&1 \
  | grep -E 'Running pass|Running analysis' | head -14 | show || true
look "注意 Running pass 后面跟的是 on [module] / on [function] / on [loop]，层级一目了然"
echo
point "现在把自己写的 pass 挂进同一套管线（插件方式，无需重编 LLVM）"
if bash "${PROJECT_DIR}/scripts/build_passes.sh" >/dev/null 2>&1 && [[ -f "${PLUGIN}" ]]; then
  echo "  ── ① count-ir（Module 级 · 分析型 · 返回 all()）"
  cmd "opt -load-pass-plugin=MyPasses.so -passes=count-ir -disable-output 00_O0.ll"
  "${OPT}" -load-pass-plugin="${PLUGIN}" -passes=count-ir -disable-output "${T}/00_O0.ll" 2>&1 | head -12 | show || true

  echo "  ── ② cfg-info（Function 级 · 分析型 · 向 AnalysisManager 要 DominatorTree/LoopInfo）"
  cmd "opt -load-pass-plugin=MyPasses.so -passes=cfg-info -disable-output 03_mem2reg.ll"
  "${OPT}" -load-pass-plugin="${PLUGIN}" -passes=cfg-info -disable-output "${T}/03_mem2reg.ll" 2>&1 | head -14 | show || true

  echo "  ── ③ strength-reduce（Function 级 · 变换型 · 返回 preserveSet<CFGAnalyses>()）"
  cmd "opt -load-pass-plugin=MyPasses.so -passes=strength-reduce -S 03_mem2reg.ll -o 07_strength_reduced.ll"
  "${OPT}" -load-pass-plugin="${PLUGIN}" -passes=strength-reduce -S "${T}/03_mem2reg.ll" -o "${T}/07_strength_reduced.ll" 2>&1 | head -6 | show || true
  look "scale8 里的 mul x, 8 变成了 shl x, 3；只改指令不动 CFG，所以支配树不必重算"
  grepshow "${T}/07_strength_reduced.ll" 'shl i32' 3

  echo "  ── ④ inject-log（Module 级 · 变换型 · 返回 none()）"
  cmd "opt -load-pass-plugin=MyPasses.so -passes=inject-log -S 03_mem2reg.ll -o 08_logged.ll"
  "${OPT}" -load-pass-plugin="${PLUGIN}" -passes=inject-log -S "${T}/03_mem2reg.ll" -o "${T}/08_logged.ll" >/dev/null 2>&1
  grepshow "${T}/08_logged.ll" 'trace_fmt|@printf' 3
  look "四种 PreservedAnalyses 写法：all() / none() / preserve<X>() / preserveSet<CFGAnalyses>()"
else
  echo "  [跳过] 插件未构建成功（缺 llvm-config 或 cmake），本站略过，不影响其它站点。"
fi

# =============================================================================
stop 9 "§4.1–4.2 分析与别名分析：四种回答"
point "NoAlias / MayAlias / PartialAlias / MustAlias —— 注意 MustAlias 不等于指针相等"
cmd "opt -aa-pipeline=basic-aa -passes=aa-eval -disable-output 01_O2.ll"
"${OPT}" -aa-pipeline=basic-aa -passes=aa-eval -disable-output "${T}/01_O2.ll" 2>&1 \
  | grep -iE 'alias|mod/ref|queries' | head -12 | show || true
look "axpy 带 restrict → noalias → 编译期就能证明不重叠；axpy_may_alias 只能 MayAlias"

# =============================================================================
stop 10 "§4.3–4.4 默认 pipeline：只看「真正改了 IR」的那些 pass"
cmd "opt -passes='default<O2>' -print-changed 00_O0.ll -S -o /dev/null"
"${OPT}" -passes='default<O2>' -print-changed "${T}/00_O0.ll" -S -o /dev/null 2>&1 \
  | grep -E '^\*\*\* IR Dump After' | head -18 | show || true
look "顺序逻辑：先规范化 → 内联 → 再简化 → 向量化放最后（它会破坏规范形状）"

# =============================================================================
stop 11 "§4.5 向量化：两个正交的向量化器 + 为什么没向量化"
cmd "clang -O2 -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -c kernel.c"
"${CLANG}" -O2 -Rpass=loop-vectorize -Rpass-missed=loop-vectorize \
  -Rpass-analysis=loop-vectorize -c "${SRC}" -o /dev/null 2>&1 | head -12 | show || true
look "-Rpass-missed 是调 kernel 性能时的第一手工具：它会直接告诉你为什么没向量化"
echo
point "restrict 与否的差别：编译期证不了不重叠时，向量化器会插运行时指针检查 + 双版本循环"
echo "  ── axpy（有 restrict）的向量体："
awk '/define.*@axpy\(/,/^}/' "${T}/01_O2.ll" | grep -cE '<[0-9]+ x float>' \
  | sed 's/^/  向量指令条数: /' || true
echo "  ── axpy_may_alias（无 restrict）里是否出现运行时检查（memcheck / found.conflict）："
awk '/define.*@axpy_may_alias\(/,/^}/' "${T}/01_O2.ll" \
  | grep -nE 'memcheck|found\.conflict|icmp ult ptr' | head -4 | show

# =============================================================================
stop 12 "§4.6 TargetTransformInfo：中端唯一的目标信息入口"
if [[ -f "${PLUGIN}" ]]; then
  echo "  ── 默认目标（基线 x86-64，通常 128bit 向量）"
  cmd "opt -load-pass-plugin=MyPasses.so -passes=tti-info -disable-output 01_O2.ll"
  "${OPT}" -load-pass-plugin="${PLUGIN}" -passes=tti-info -disable-output "${T}/01_O2.ll" 2>&1 | head -12 | show || true
  echo "  ── 同一段代码，但 IR 上带了 AVX2 的 target-features"
  "${OPT}" -load-pass-plugin="${PLUGIN}" -passes=tti-info -disable-output "${T}/02_O2_avx2.ll" 2>&1 | head -12 | show || true
  look "向量寄存器位宽和 fmul 代价变了 → 向量化器的决策也会跟着变。接新硬件时 TTI 填得准不准，直接决定性能"
else
  echo "  [跳过] 需要先构建插件。"
fi

# =============================================================================
stop 13 "§5 后端 CodeGen：七个阶段 / MIR / 寄存器分配"
cmd "llc -O2 -mtriple=${TRIPLE} 01_O2.ll -o 20_kernel.s"
"${LLC}" -O2 -mtriple="${TRIPLE}" "${T}/01_O2.ll" -o "${T}/20_kernel.s" 2>/dev/null
point "后端仍在用 legacy PM，可以直接把它的 pass 结构打出来"
cmd "llc -O2 -mtriple=${TRIPLE} -debug-pass=Structure 01_O2.ll -o /dev/null"
"${LLC}" -O2 -mtriple="${TRIPLE}" -debug-pass=Structure "${T}/01_O2.ll" -o /dev/null 2>&1 \
  | grep -iE 'select|schedul|register|prolog|expand|emit' | head -14 | show \
  || echo "  （该版本 llc 不支持 -debug-pass，可用 -print-after-all 代替）"
echo
point "指令选择刚结束 vs 寄存器分配之后：对比两份 MIR 最能说明「虚拟寄存器 + PHI 是怎么消失的」"
cmd "llc -stop-after=finalize-isel  → 21_after_isel.mir"
"${LLC}" -O2 -mtriple="${TRIPLE}" -stop-after=finalize-isel "${T}/01_O2.ll" -o "${T}/21_after_isel.mir" 2>/dev/null
cmd "llc -stop-after=virtregrewriter → 22_after_regalloc.mir"
"${LLC}" -O2 -mtriple="${TRIPLE}" -stop-after=virtregrewriter "${T}/01_O2.ll" -o "${T}/22_after_regalloc.mir" 2>/dev/null \
  || "${LLC}" -O2 -mtriple="${TRIPLE}" -stop-after=greedy "${T}/01_O2.ll" -o "${T}/22_after_regalloc.mir" 2>/dev/null
if [[ -s "${T}/21_after_isel.mir" && -s "${T}/22_after_regalloc.mir" ]]; then
  printf '  %-22s %10s %10s\n' "指标" "ISel后" "分配后"
  printf '  %-22s %10s %10s\n' "PHI 指令" \
    "$(grep -c 'PHI' "${T}/21_after_isel.mir")" "$(grep -c 'PHI' "${T}/22_after_regalloc.mir")"
  printf '  %-22s %10s %10s\n' "虚拟寄存器 %0/%1..." \
    "$(grep -cE '%[0-9]+' "${T}/21_after_isel.mir")" "$(grep -cE '%[0-9]+' "${T}/22_after_regalloc.mir")"
  printf '  %-22s %10s %10s\n' "物理寄存器 \$xmm/\$e.." \
    "$(grep -cE '\$(xmm|e[a-z]{2}|r[a-z0-9]+)' "${T}/21_after_isel.mir")" \
    "$(grep -cE '\$(xmm|e[a-z]{2}|r[a-z0-9]+)' "${T}/22_after_regalloc.mir")"
  look "链条：LiveVariables → LiveIntervals → PHIElimination(SSA 解构) → two-address → Greedy 分配"
fi

# =============================================================================
stop 14 "§5.7 MC 层：汇编 → 机器码字节"
if [[ -n "${LLVM_MC}" ]]; then
  cmd "llvm-mc -triple=${TRIPLE} -show-encoding 20_kernel.s | head"
  "${LLVM_MC}" -triple="${TRIPLE}" -show-encoding "${T}/20_kernel.s" 2>/dev/null \
    | grep -E 'encoding' | head -6 | show || true
  look "同一条指令在 MC 层才第一次有了确定的字节编码；汇编器/反汇编器/JIT 共用这一层"
fi
"${CLANG}" -no-pie -c "${T}/20_kernel.s" -o "${T}/23_kernel.o" 2>/dev/null
if [[ -n "${LLVM_OBJDUMP}" && -f "${T}/23_kernel.o" ]]; then
  cmd "llvm-objdump -d 23_kernel.o | head"
  "${LLVM_OBJDUMP}" -d "${T}/23_kernel.o" 2>/dev/null | sed -n '1,14p' | show || true
fi

# =============================================================================
stop 15 "§6 TableGen：LLVM 后端描述与 MLIR ODS 的同一套语言"
if [[ -n "${LLVM_TBLGEN}" ]]; then
  cmd "llvm-tblgen --print-records src/mini.td"
  "${LLVM_TBLGEN}" --print-records "${MINI_TD}" 2>&1 | head -30 | show || true
  look "class/def/multiclass/let 展开成一堆 record；LLVM 用它生成后端表，MLIR 用它生成 op 定义（ODS）"
else
  echo "  [跳过] 未找到 llvm-tblgen。src/mini.td 仍可直接阅读。"
fi

# =============================================================================
stop 16 "§7 与 MLIR / AI 编译器的接缝"
point "AI 编译器只用 LLVM 的最后一段：MLIR 负责多层 dialect，llvm dialect 之后才交给 LLVM"
cat <<'TXT' | show
mlir-opt --convert-vector-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts x.mlir \
  | mlir-translate --mlir-to-llvmir \
  | opt  -passes='default<O2>' -S \
  | llc  -mtriple=x86_64-- -o -

三个杠杆（决定「生成的 LLVM IR 跑不跑得快」）：
  1. noalias / !alias.scope      ← 别名信息传不下来，向量化直接失效
  2. align / dereferenceable     ← 访存宽度与安全性的依据
  3. TTI 代价模型                 ← 接新硬件时中端做决策的唯一入口
TXT
if command -v mlir-opt >/dev/null 2>&1; then
  look "本机检测到 mlir-opt，可以直接去 ../mlir-toy-dialect 把这条链跑通"
else
  look "本机没有 mlir-opt，这一站先读命令即可；到 mlir-toy-dialect 阶段再实跑"
fi

# =============================================================================
# 生成 TOUR.md
# =============================================================================
{
cat <<'MD'
# LLVM 核心要点巡礼 —— 本次运行报告

由 `scripts/tour.sh` 生成。场景：`src/kernel.c`（axpy + relu 求和）、
`src/poison_demo.ll`（手写 poison）、`src/mini.td`（TableGen）。

## 站点 ↔ 学习指南 ↔ 产物

| 站 | 学习指南 | 要点 | 产物 |
|----|---------|------|------|
| 0 | §2.1 | IR 三态（.ll / .bc 等价互转） | `00_O0.ll` `00_O0.bc` `00_roundtrip.ll` |
| 1 | §2.2 | 类型系统：iN / float / opaque ptr / 向量 / 结构体 | `01_O2.ll` |
| 2 | §2.3 | SSA、基本块、terminator、phi | `03_mem2reg.ll` |
| 3 | §2.5 | getelementptr 与 inbounds 承诺 | `00_O0.ll` |
| 4 | §2.6 | noalias/align/dereferenceable、contract → FMA | `04_contract_off.ll` `05_contract_fast.ll` |
| 5 | §2.7 | poison / undef / freeze / select 屏障 | `06_poison_O2.ll` |
| 6 | §2.8 | Metadata：!tbaa / !llvm.loop | `01_O2.ll` |
| 7 | §2.9 | Intrinsics：fmuladd / vector.reduce | `01_O2.ll` |
| 8 | §3 | New PM 四层嵌套 + 自定义 pass + PreservedAnalyses 四写法 | `07_strength_reduced.ll` `08_logged.ll` |
| 9 | §4.1–4.2 | 六个 Analysis + 别名分析四种回答 | （aa-eval 输出见 tour.log） |
| 10 | §4.3–4.4 | 默认 pipeline 骨架（-print-changed） | （见 tour.log） |
| 11 | §4.5 | 两个向量化器 + 为什么没向量化 | （remark 见 tour.log） |
| 12 | §4.6 | TTI：向量宽度与指令代价 | `02_O2_avx2.ll` 对比 |
| 13 | §5 | 后端七阶段 / MIR / 寄存器分配 | `20_kernel.s` `21_after_isel.mir` `22_after_regalloc.mir` |
| 14 | §5.7 | MC 层：指令编码与反汇编 | `23_kernel.o` |
| 15 | §6 | TableGen = LLVM 后端描述 + MLIR ODS 的同一套语言 | `src/mini.td` |
| 16 | §7 | 与 MLIR 的接缝与三个杠杆 | —— |

## 建议的精读顺序

1. `00_O0.ll` → `03_mem2reg.ll`：看 alloca/load/store 如何变成 phi（SSA 的门票）。
2. `04_contract_off.ll` vs `05_contract_fast.ll`：一次舍入的 FMA 是怎么被"许可"出来的。
3. `06_poison_O2.ll`：`and poison, 0` 仍是 poison；select 不传染。
4. `01_O2.ll` 里的 `@axpy` vs `@axpy_may_alias`：restrict 带来的 noalias 值多少钱。
5. `21_after_isel.mir` vs `22_after_regalloc.mir`：虚拟寄存器与 PHI 是怎么消失的。

全量日志：`tour.log`。
MD
} > "${T}/TOUR.md"

stop 17 "巡礼结束"
echo "  产物目录 : ${T}"
echo "  全量日志 : ${T}/tour.log"
echo "  阅读入口 : ${T}/TOUR.md  ← 建议先看这份"
echo
echo "  一句话回顾：IR 三态 → SSA/phi → GEP/属性/poison → New PM 与自定义 pass"
echo "              → 分析与别名 → 向量化与 TTI → 后端 MIR/寄存器分配 → MC → TableGen → MLIR 接缝"
