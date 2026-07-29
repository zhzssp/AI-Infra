#!/usr/bin/env bash
# =============================================================================
# run.sh —— 一步步走完 LLVM 的完整编译流程
#   目标：让你【通过输出】就能看懂整条链路 + 每一步的局部结果（尤其是 LLVM IR）
#   所有输出都会落盘（3 种形式）：
#     1) out/full_run.log     ——本次运行的【全量】终端输出（一份完整流水账）
#     2) out/steps/NN_*.log   ——【分步】日志：每个步骤的解说 + 结果单独存一份
#     3) out/ANALYSIS.md      ——【汇总报告】：流程图 + 指令统计对比 + 阅读指引
#   外加：out/diff_*.txt       ——各阶段 IR 的差异对比（O0→mem2reg→O2）
# -----------------------------------------------------------------------------
#   源码 sum.c
#     │ ① clang 前端        词法/语法/语义分析  ->  Clang AST
#     ▼
#   ② clang -emit-llvm     生成 LLVM IR (-O0，未优化，充满 alloca/load/store)
#     │
#     ├ ③a opt -passes=mem2reg   把内存变量提升为 SSA 寄存器 -> 出现 phi 节点（精髓！）
#     └ ③b opt -O2               完整优化：内联 + 常量传播 + 循环优化
#     ▼
#   ④ llvm-as / llvm-dis   文本 IR(.ll) <-> 位码 bitcode(.bc) 互转
#     ▼
#   ⑤ llc                  后端 CodeGen：IR -> 目标汇编(.s)（指令选择/寄存器分配）
#     ▼
#   ⑥ clang(汇编器+链接器)  .s -> .o -> 可执行文件 -> 运行
#     ▼
#   ⑦ lli （附赠）          不落地机器码，直接 JIT 执行 IR
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

SRC="${PROJECT_DIR}/src/sum.c"
STEP_DIR="${OUT_DIR}/steps"
mkdir -p "${OUT_DIR}" "${STEP_DIR}"

# 产物文件（供各步骤和报告统一引用）
F_O0="${OUT_DIR}/02_sum_O0.ll"
F_M2R="${OUT_DIR}/03a_sum_mem2reg.ll"
F_O2="${OUT_DIR}/03b_sum_O2.ll"
F_LOGGED="${OUT_DIR}/03c_sum_logged.ll"      # inject-log 注入 printf 后的 IR
F_LOGGED_S="${OUT_DIR}/03c_sum_logged.s"     # 上者的汇编
BIN_LOGGED="${OUT_DIR}/03c_sum_logged"       # 上者编译成的可执行（运行可见 trace）
PASS_PLUGIN="${PROJECT_DIR}/build/passes/MyPasses.so"  # 自定义 pass 插件

# 把整个终端输出同时镜像到 out/full_run.log（全量日志）
exec > >(tee "${OUT_DIR}/full_run.log") 2>&1

# ---------------------------------------------------------------------------
# 打印工具：所有内容既进终端（→ full_run.log），又 append 到"当前分步日志"
# ---------------------------------------------------------------------------
STEP_LOG="/dev/null"   # 由 step_begin() 切换

# 开启一个新步骤：step_begin <编号串> <标题>
step_begin() {
  STEP_LOG="${STEP_DIR}/$1.log"
  : > "${STEP_LOG}"
  {
    printf '================================================================\n'
    printf ' 步骤 %s\n' "$2"
    printf '================================================================\n'
  } | _emit
}
# p <文本...>：普通解说行
p() { printf '%s\n' "$*" | _emit; }
# 内部：把 stdin 同时写到 stdout 和当前分步日志
_emit() { tee -a "${STEP_LOG}"; }
# link <输入> <工具> <输出>：用统一格式打印这一步的"数据流"
link() { printf '  [链路] 输入: %-22s → 工具: %-14s → 输出: %s\n' "$1" "$2" "$3" | _emit; }
cmd()  { printf '  [命令] %s\n' "$*" | _emit; }
# dump <文件>：整篇打印一个文本产物（并落盘到分步日志）
dump() { { printf -- '----- %s（完整内容）-----\n' "$(basename "$1")"; cat "$1"; \
           printf -- '----------------------------------------------------------\n'; } | _emit; }
# dump_fn <文件> <函数名>：只打印某个函数的 IR/汇编函数体
dump_fn() {
  { printf -- '----- %s 里的 @%s 函数体 -----\n' "$(basename "$1")" "$2"
    awk -v fn="$2" '$0 ~ ("define.*@" fn "\\("){p=1} p{print} /^}/{if(p)exit}' "$1"
    printf -- '----------------------------------------------------------\n'; } | _emit || true
}
# dump_asm_fn <汇编文件> <符号>：打印一个汇编函数（从 label 到 .cfi_endproc）
dump_asm_fn() {
  { printf -- '----- %s 里的 %s 汇编 -----\n' "$(basename "$1")" "$2"
    awk -v fn="$2" '$0 ~ ("^" fn ":"){p=1} p{print} /\.cfi_endproc/{if(p)exit}' "$1"
    printf -- '----------------------------------------------------------\n'; } | _emit || true
}
# cnt <文件> <正则>：数某种指令出现次数（无匹配返回 0 而不报错）
cnt() { grep -cE "$2" "$1" 2>/dev/null || true; }

echo "############################################################"
echo "# LLVM 编译全流程演示   源码: src/sum.c"
echo "# 全量日志 : ${OUT_DIR}/full_run.log"
echo "# 分步日志 : ${OUT_DIR}/steps/*.log"
echo "# 汇总报告 : ${OUT_DIR}/ANALYSIS.md"
echo "############################################################"

# =============================================================================
step_begin "00_overview" "总览：这条编译链路长什么样"
p ""
p "  一份 C 源码要变成能跑的程序，LLVM 把它拆成如下可观察的环节："
p ""
p "    src/sum.c"
p "        │ ①前端(clang)  词法/语法/语义分析"
p "        ▼"
p "      Clang AST ─────────────────────────────► 01_ast.txt"
p "        │ ②emit-llvm  (-O0，未优化)"
p "        ▼"
p "      LLVM IR (含 alloca/load/store) ─────────► 02_sum_O0.ll"
p "        │"
p "        ├─ ③a opt mem2reg  内存→SSA寄存器，出现 phi ► 03a_sum_mem2reg.ll"
p "        └─ ③b opt -O2      内联/常量传播/循环优化   ► 03b_sum_O2.ll"
p "        │ ④ llvm-as/llvm-dis  文本.ll ↔ 位码.bc"
p "        ▼"
p "      优化后 IR ─ ⑤llc后端 ─► x86-64 汇编 ────────► 05_sum_O2.s"
p "        │ ⑥ clang(as+ld) 汇编+链接"
p "        ▼"
p "      可执行文件 06_sum ── 运行 ──► 输出 55"
p "        · ⑦ lli：不落地机器码，直接 JIT 执行 IR 验证语义"
p ""
p "  下面每一步都会打印 [链路]/[命令]/[结果]，并把该步输出单独存进 steps/。"
p ""
p "  工具链版本（三者需一致，否则 IR 无法衔接）："
p "    clang : $("${CLANG}" --version 2>/dev/null | head -1)"
p "    opt   : $("${OPT}"  --version 2>/dev/null | grep -i 'version' | head -1 | sed 's/^ *//')"
p "    llc   : $("${LLC}"  --version 2>/dev/null | grep -i 'version' | head -1 | sed 's/^ *//')"
p ""
p "  这次要编译的源码："
dump "${SRC}"

# =============================================================================
step_begin "00_baseline" "0：基线——先直接编译运行，看到最终效果"
link "src/sum.c" "clang -O2" "00_baseline (可执行)"
cmd "clang -O2 src/sum.c -o out/00_baseline"
"${CLANG}" -O2 "${SRC}" -o "${OUT_DIR}/00_baseline"
p "  [结果] 运行输出如下（记住这个 55，后面每一步的产物都应算出同样结果）："
p "  ----------------------------------------"
"${OUT_DIR}/00_baseline" | _emit
p "  ----------------------------------------"

# =============================================================================
step_begin "01_frontend_ast" "① 前端：clang 做词法/语法/语义分析，产出 Clang AST"
link "src/sum.c" "clang -ast-dump" "01_ast.txt"
cmd "clang -Xclang -ast-dump -fsyntax-only src/sum.c"
p "  [目的] AST(抽象语法树) 是编译器对源码结构的理解：函数、循环、表达式都变成树节点。"
p "         这是一切编译的起点——之后才谈得上生成 IR。"
"${CLANG}" -Xclang -ast-dump -fsyntax-only "${SRC}" > "${OUT_DIR}/01_ast.txt" 2>&1 || true
p "  [结果] AST 很长(已全存 01_ast.txt)，这里看 sum_of_squares 那棵子树的开头，"
p "         注意 FunctionDecl / ForStmt / CallExpr 这些节点如何对应源码结构："
p "  ----------------------------------------"
grep -n -A 14 "sum_of_squares" "${OUT_DIR}/01_ast.txt" | head -22 | _emit || true
p "  ----------------------------------------"

# =============================================================================
step_begin "02_ir_O0" "② 生成 LLVM IR（-O0 未优化版）——IR 的原始形态"
link "src/sum.c" "clang -emit-llvm -O0" "02_sum_O0.ll"
cmd "clang -S -emit-llvm -O0 -Xclang -disable-O0-optnone -fno-discard-value-names src/sum.c"
p "  [目的] 把 AST 降低(lower)为 LLVM IR。-O0 下每个局部变量都是一块 alloca 栈内存，"
p "         读写靠 load/store —— 这【不是】SSA，是留给 mem2reg 优化的原料。"
p "  [陷阱] -Xclang -disable-O0-optnone 去掉 optnone 属性，否则后面 opt 拒绝优化它；"
p "         -fno-discard-value-names 保留 %acc/%i 这类可读名字（否则显示为 %1 %2）。"
"${CLANG}" -S -emit-llvm -O0 -Xclang -disable-O0-optnone -fno-discard-value-names \
    "${SRC}" -o "${F_O0}"
p ""
p "  [结果] 完整的未优化 IR（整份文件都在这里，方便你逐行对照）："
dump "${F_O0}"
p "  [读 IR 的钥匙] 上面这些指令的含义："
p "     alloca i32       —— 在栈上分配一块 i32 内存，返回它的指针(ptr)"
p "     store v, ptr p   —— 把值 v 写入指针 p 指向的内存"
p "     %x = load ..., p —— 从指针 p 读出一个值，命名为 %x"
p "     br label %L      —— 无条件跳转到基本块 %L"
p "     br i1 %c, %A,%B  —— 条件跳转：%c 为真去 %A，否则去 %B（if/循环靠它）"
p "     call @f(...)     —— 函数调用（这里能看到 @square 和 @printf 的调用）"
p "     ret i32 %x       —— 返回"
p ""
p "  [观察] 注意 sum_of_squares 里 acc/i 被反复 load→运算→store，很啰嗦——这正是优化的切入点。"

# =============================================================================
step_begin "03a_mem2reg" "③a 优化(单个Pass)：opt mem2reg —— 理解 SSA 的核心一步"
link "02_sum_O0.ll" "opt -passes=mem2reg" "03a_sum_mem2reg.ll"
cmd "opt -S -passes=mem2reg 02_sum_O0.ll"
p "  [目的] mem2reg 把 alloca 内存变量'提升'成寄存器 SSA 值。循环变量 acc/i 在"
p "         '循环入口' 和 '上一轮迭代' 两个前驱块都有定值，必须用 phi 节点汇合——"
p "         phi(φ) 就是 SSA(静态单赋值:每个名字只赋值一次) 的标志性特征！"
"${OPT}" -S -passes=mem2reg "${F_O0}" -o "${F_M2R}"
p ""
p "  [结果] mem2reg 后的 sum_of_squares：alloca/load/store 基本消失，冒出了 phi："
dump_fn "${F_M2R}" "sum_of_squares"
p "  [读 phi 的钥匙]"
p "     %acc.0 = phi i32 [ 0, %entry ], [ %add, %loop ]"
p "       └ 含义：如果从 %entry 块进来，acc 取 0；如果从 %loop 块(上一轮)进来，取 %add。"
p "         phi 让'同一个变量在不同路径有不同来源值'能用单赋值形式表达——这就是 SSA。"
p ""
p "  [对比] 生成 02→03a 的差异文件 diff_02_to_03a.txt（左 - 是消失的内存操作，右 + 是新增的 SSA）："
diff -u "${F_O0}" "${F_M2R}" > "${OUT_DIR}/diff_02_to_03a.txt" 2>&1 || true
p "  ----------------------------------------"
sed -n '1,40p' "${OUT_DIR}/diff_02_to_03a.txt" | _emit || true
p "  ----------------------------------------"
p ""
p "  [量化] alloca / phi 数量变化："
p "     -O0 原始 IR  : $(cnt "${F_O0}"  'alloca ') 个 alloca，$(cnt "${F_O0}"  'phi ') 个 phi"
p "     mem2reg 之后 : $(cnt "${F_M2R}" 'alloca ') 个 alloca，$(cnt "${F_M2R}" 'phi ') 个 phi"

# =============================================================================
step_begin "03b_O2" "③b 优化(完整 -O2)：内联 + 常量传播 + 循环优化"
link "02_sum_O0.ll" "opt -O2" "03b_sum_O2.ll"
cmd "opt -S -O2 02_sum_O0.ll"
p "  [目的] -O2 是一整条 Pass 流水线。最直观的变化：square() 被内联进循环，"
p "         很多计算被折叠，IR 变得紧凑。"
"${OPT}" -S -O2 "${F_O0}" -o "${F_O2}"
p ""
p "  [结果] -O2 后的 sum_of_squares（对比 03a，指令更少、更紧凑）："
dump_fn "${F_O2}" "sum_of_squares"
p "  [观察] square 是否还独立存在？"
if grep -q "define.*@square(" "${F_O2}"; then
  p "     仍存在独立的 @square 定义。"
else
  p "     @square 已被内联消除（符合预期）——它的代码被直接展开进了调用处。"
fi
p ""
p "  [对比] 生成 03a→03b 的差异文件 diff_03a_to_03b.txt（看优化到底改了哪些行）："
diff -u "${F_M2R}" "${F_O2}" > "${OUT_DIR}/diff_03a_to_03b.txt" 2>&1 || true
p "  ----------------------------------------"
sed -n '1,40p' "${OUT_DIR}/diff_03a_to_03b.txt" | _emit || true
p "  ----------------------------------------"

# =============================================================================
step_begin "03c_custom_pass" "③c 自定义 Pass：用 opt 加载插件跑我们手写的两个 Pass"
p "  [目的] 前面 mem2reg/-O2 都是 LLVM 内置 Pass。这一步演示如何把'自己写的 Pass'"
p "         挂进同一套管线：用 New PassManager 插件方式，opt 加载 .so 后按名字调用。"
p ""
p "  先编译插件（源码在 passes/，产物 build/passes/MyPasses.so）："
cmd "cmake -S passes -B build/passes -DLLVM_DIR=\$(llvm-config --cmakedir) && cmake --build build/passes"
if bash "${PROJECT_DIR}/scripts/build_passes.sh" >/dev/null 2>&1 && [[ -f "${PASS_PLUGIN}" ]]; then
  p "  [结果] 插件就绪：build/passes/MyPasses.so"
  p ""
  p "  ---- Pass 1：count-ir（分析型，只统计不改 IR）----"
  link "02_sum_O0.ll" "opt count-ir" "(统计信息，无产物)"
  cmd "opt -load-pass-plugin=MyPasses.so -passes=count-ir -disable-output 02_sum_O0.ll"
  p "  [说明] 它遍历每个函数，数基本块/指令/call，返回 PreservedAnalyses::all()（不动 IR）。"
  p "  [结果]"
  p "  ----------------------------------------"
  "${OPT}" -load-pass-plugin="${PASS_PLUGIN}" -passes=count-ir -disable-output "${F_O0}" 2>&1 | _emit || true
  p "  ----------------------------------------"
  p ""
  p "  ---- Pass 2：inject-log（变换型，真的改 IR）----"
  link "03a_sum_mem2reg.ll" "opt inject-log" "03c_sum_logged.ll"
  cmd "opt -load-pass-plugin=MyPasses.so -passes=inject-log -S 03a_sum_mem2reg.ll -o 03c_sum_logged.ll"
  p "  [说明] 它在每个函数入口插一句 printf(\"[trace] enter <fn>\")，返回 PreservedAnalyses::none()。"
  "${OPT}" -load-pass-plugin="${PASS_PLUGIN}" -passes=inject-log -S "${F_M2R}" -o "${F_LOGGED}" 2>&1 | _emit || true
  p "  [结果] 注入后新增的全局字符串与 printf 调用（截取）："
  p "  ----------------------------------------"
  grep -nE 'trace_fmt|call i32 .*@printf' "${F_LOGGED}" | head -8 | _emit || true
  p "  ----------------------------------------"
  p ""
  p "  [验证] 把注入后的 IR 编译成可执行并运行——每进一个函数都会打印一行 trace，"
  p "         最后仍算出 55（证明 pass 只加了日志，没破坏原逻辑）："
  cmd "llc -O2 03c_sum_logged.ll -o 03c_sum_logged.s ; clang -no-pie 03c_sum_logged.s -o 03c_sum_logged"
  "${LLC}" -O2 "${F_LOGGED}" -o "${F_LOGGED_S}" 2>/dev/null || true
  "${CLANG}" -no-pie "${F_LOGGED_S}" -o "${BIN_LOGGED}" 2>/dev/null || true
  p "  ----------------------------------------"
  if [[ -x "${BIN_LOGGED}" ]]; then
    "${BIN_LOGGED}" | _emit || true
  else
    p "  （可执行未生成，跳过运行）"
  fi
  p "  ----------------------------------------"
  p "  [观察] trace 行揭示了实际调用关系：main → sum_of_squares → square×5。"
else
  p "  [跳过] 插件构建失败（可能缺 llvm-config/cmake 开发环境），本步略过，不影响主流程。"
fi

# =============================================================================
step_begin "04_bitcode" "④ 文本 IR(.ll) 与位码 bitcode(.bc) 互转"
if [[ -n "${LLVM_AS}" && -n "${LLVM_DIS}" ]]; then
  link "02_sum_O0.ll" "llvm-as/llvm-dis" "04_sum_O0.bc / 04_roundtrip.ll"
  cmd "llvm-as 02_sum_O0.ll -o 04_sum_O0.bc ; llvm-dis 04_sum_O0.bc -o 04_roundtrip.ll"
  p "  [目的] .ll 是给人看的文本；.bc 是等价的紧凑二进制(编译器内部/磁盘缓存用)。二者可无损互转。"
  "${LLVM_AS}" "${F_O0}" -o "${OUT_DIR}/04_sum_O0.bc"
  "${LLVM_DIS}" "${OUT_DIR}/04_sum_O0.bc" -o "${OUT_DIR}/04_roundtrip.ll"
  p "  [结果] 体积对比："
  p "     文本 .ll : $(wc -c < "${F_O0}") 字节"
  p "     位码 .bc : $(wc -c < "${OUT_DIR}/04_sum_O0.bc") 字节（更紧凑的二进制）"
  if diff -q "${F_O0}" "${OUT_DIR}/04_roundtrip.ll" >/dev/null 2>&1; then
    p "     roundtrip 校验：.ll → .bc → .ll 内容一致 ✔"
  else
    p "     roundtrip 校验：转回文本后有格式差异(语义等价，属正常，可 diff 04_roundtrip.ll 02_sum_O0.ll 查看)。"
  fi
else
  p "  （未找到 llvm-as/llvm-dis，跳过此步）"
fi

# =============================================================================
step_begin "05_codegen_asm" "⑤ 后端 CodeGen：llc 把 LLVM IR 编译成目标汇编(.s)"
link "03b_sum_O2.ll" "llc -O2" "05_sum_O2.s"
cmd "llc -O2 03b_sum_O2.ll -o 05_sum_O2.s"
p "  [目的] 后端把与平台无关的 IR 翻译成具体机器的汇编，内部经历："
p "         指令选择(ISel) → 寄存器分配 → 指令调度 → 发射目标平台(x86-64)汇编。"
"${LLC}" -O2 "${F_O2}" -o "${OUT_DIR}/05_sum_O2.s"
p ""
p "  [结果] sum_of_squares 的真实 x86-64 汇编（IR 里的 phi/循环 已变成 imul/add/寄存器/跳转）："
dump_asm_fn "${OUT_DIR}/05_sum_O2.s" "sum_of_squares"
p "  [观察] 到这里，抽象的 SSA 值已经落到具体寄存器(%eax/%ecx...)和机器指令上了。"

# =============================================================================
step_begin "06_assemble_link" "⑥ 汇编器 + 链接器：.s → 可执行文件 → 运行"
link "05_sum_O2.s" "clang(as+ld)" "06_sum (可执行)"
cmd "clang -no-pie 05_sum_O2.s -o 06_sum   # clang 自动调用汇编器 as 与链接器 ld"
p "  [陷阱] llc 默认产出非位置无关(non-PIC)汇编，而本机链接器默认要生成 PIE，"
p "         二者冲突会报 'R_X86_64_32 ... can not be used when making a PIE object'。"
p "         加 -no-pie 让链接器生成传统可执行即可（教学场景足够）。"
"${CLANG}" -no-pie "${OUT_DIR}/05_sum_O2.s" -o "${OUT_DIR}/06_sum"
p "  [结果] 运行最终可执行文件（应与步骤0基线一致，输出 55）："
p "  ----------------------------------------"
"${OUT_DIR}/06_sum" | _emit
p "  ----------------------------------------"

# =============================================================================
step_begin "07_jit_lli" "⑦ 附赠：lli 直接 JIT 执行 LLVM IR（不生成机器码文件）"
if [[ -n "${LLI}" ]]; then
  link "03b_sum_O2.ll" "lli" "(直接执行，无产物)"
  cmd "lli 03b_sum_O2.ll"
  p "  [目的] lli 是 LLVM 自带的解释器/JIT，直接跑 IR，用来快速验证 IR 语义是否正确。"
  p "  [结果] JIT 执行输出（同样应为 55）："
  p "  ----------------------------------------"
  "${LLI}" "${F_O2}" | _emit || true
  p "  ----------------------------------------"
else
  p "  （未找到 lli，跳过此步）"
fi

# =============================================================================
step_begin "08_summary" "汇总：同一段逻辑在各阶段的指令统计对比"
p "  下表统计同一个程序在 3 个 IR 阶段的关键指令数量，一眼看清优化的走向："
p "  （alloca 减少=内存变寄存器；phi 出现=SSA 化；load/store 减少=优化生效）"
p ""
printf '  %-10s %8s %10s %8s\n' "指令" "O0" "mem2reg" "O2" | _emit
printf '  %-10s %8s %10s %8s\n' "------" "----" "-------" "----" | _emit
row() {
  printf '  %-10s %8s %10s %8s\n' "$1" \
    "$(cnt "${F_O0}" "$2")" "$(cnt "${F_M2R}" "$2")" "$(cnt "${F_O2}" "$2")" | _emit
}
row "alloca" 'alloca '
row "load"   'load '
row "store"  'store '
row "phi"    'phi '
row "br"     'br '
row "call"   'call '
row "mul"    ' mul '
row "add"    ' add '
row "ret"    'ret '
p ""
p "  典型走向：O0 里一堆 alloca/load/store（没有 phi）  →  mem2reg 后 alloca 归零、"
p "  出现 phi（进入 SSA）  →  -O2 进一步内联折叠，指令进一步精简。"

# =============================================================================
# 生成汇总报告 out/ANALYSIS.md（把关键信息固化成一份可回看的文档）
# =============================================================================
{
cat <<'MD'
# LLVM 编译全流程 —— 本次运行分析报告

> 本报告由 `scripts/run.sh` 自动生成。源码：`src/sum.c`（计算 1²+2²+…+5² = 55）。
> 建议配合 `out/steps/*.log`（分步详情）与 `out/full_run.log`（全量日志）一起看。

## 一、编译链路总览

```
src/sum.c
   │ ①前端(clang)  词法/语法/语义分析
   ▼
 Clang AST ─────────────────────────────► out/01_ast.txt
   │ ②emit-llvm (-O0)
   ▼
 LLVM IR (alloca/load/store) ────────────► out/02_sum_O0.ll
   │
   ├─ ③a opt mem2reg  内存→SSA，出现 phi ─► out/03a_sum_mem2reg.ll
   └─ ③b opt -O2      内联/常量传播/循环 ─► out/03b_sum_O2.ll
   │ ④ llvm-as/llvm-dis  .ll ↔ .bc
   ▼
 优化后 IR ─ ⑤llc ─► x86-64 汇编 ─────────► out/05_sum_O2.s
   │ ⑥ clang(as+ld)
   ▼
 可执行文件 ─────────────────────────────► out/06_sum   （运行输出 55）
   · ⑦ lli 直接 JIT 执行 IR 验证语义
```

## 二、各阶段关键指令统计
MD

printf '\n| 指令 | O0 | mem2reg | O2 |\n|------|----|---------|----|\n'
mdrow() { printf '| %s | %s | %s | %s |\n' "$1" \
  "$(cnt "${F_O0}" "$2")" "$(cnt "${F_M2R}" "$2")" "$(cnt "${F_O2}" "$2")"; }
mdrow "alloca" 'alloca '
mdrow "load"   'load '
mdrow "store"  'store '
mdrow "phi"    'phi '
mdrow "br"     'br '
mdrow "call"   'call '
mdrow "mul"    ' mul '
mdrow "add"    ' add '
mdrow "ret"    'ret '

cat <<'MD'

**读法**：`alloca/load/store` 一路减少、`phi` 从 0 出现，说明变量从「内存读写」被提升为「SSA 寄存器」；
`-O2` 下 `call` 减少（`square` 被内联）、算术指令折叠，IR 进一步精简。

## 三、产物清单（都在 out/）

| 文件 | 阶段 | 说明 |
|------|------|------|
| `00_baseline` | 0 | 直接 `-O2` 编译的可执行（基线，输出 55） |
| `01_ast.txt` | ① | 前端 Clang AST（源码的树形理解） |
| `02_sum_O0.ll` | ② | 未优化 IR（充满 alloca/load/store，非 SSA） |
| `03a_sum_mem2reg.ll` | ③a | mem2reg 后（alloca 消失、出现 phi，**SSA 精髓**） |
| `03b_sum_O2.ll` | ③b | `-O2` 完整优化后（square 被内联） |
| `03c_sum_logged.ll` / `03c_sum_logged` | ③c | **自定义 pass** inject-log 注入 printf 后的 IR 及其可执行（运行可见 trace） |
| `04_sum_O0.bc` / `04_roundtrip.ll` | ④ | 位码及其转回文本（.ll↔.bc 等价） |
| `05_sum_O2.s` | ⑤ | 目标 x86-64 汇编 |
| `06_sum` | ⑥ | 最终可执行文件（输出 55） |
| `diff_02_to_03a.txt` | ②→③a | mem2reg 带来的差异（内存操作→SSA） |
| `diff_03a_to_03b.txt` | ③a→③b | -O2 优化带来的差异 |
| `steps/*.log` | 全部 | 每一步的解说+结果（分步日志） |
| `full_run.log` | 全部 | 本次运行的完整终端输出 |

## 四、建议阅读顺序

1. `02_sum_O0.ll` → `03a_sum_mem2reg.ll`：看 `alloca/load/store` 如何变成 `phi`（**最关键**）。
2. `diff_02_to_03a.txt`：直接看 mem2reg 改了哪些行。
3. `03a_sum_mem2reg.ll` → `03b_sum_O2.ll`：看 `-O2` 的内联与折叠。
4. `03b_sum_O2.ll` → `05_sum_O2.s`：看 IR 如何落成真实机器指令。

你会亲眼看到同一段逻辑：内存读写 → SSA 寄存器 → 优化精简 → 机器指令。
MD
} > "${OUT_DIR}/ANALYSIS.md"

# =============================================================================
step_begin "09_done" "全流程完成"
p "  所有输出已保存到 3 个地方："
p "    1) 全量日志 : ${OUT_DIR}/full_run.log"
p "    2) 分步日志 : ${OUT_DIR}/steps/   （共 $(ls -1 "${STEP_DIR}" | wc -l | tr -d ' ') 个文件，每步一份）"
p "    3) 汇总报告 : ${OUT_DIR}/ANALYSIS.md  ← 建议先看这份"
p "    额外 IR 差异: ${OUT_DIR}/diff_02_to_03a.txt , ${OUT_DIR}/diff_03a_to_03b.txt"
p ""
p "  分步日志清单："
for f in "${STEP_DIR}"/*.log; do printf '    %s\n' "steps/$(basename "$f")" | _emit; done
p ""
p "  一句话回顾：C源码 →(前端)AST →(emit-llvm)IR →(mem2reg)SSA →(-O2)优化 →(llc)汇编 →(as+ld)可执行 = 55"
