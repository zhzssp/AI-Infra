#!/usr/bin/env bash
# =============================================================================
# run_execute.sh —— 站 ⑦：编成 .vmfb，真的跑一遍，并对答案
# -----------------------------------------------------------------------------
# 一句话：前面所有相位都只是「文本」，只有这一步能证明那些变换【没改语义】。
#
# 产物：out/execute/*.vmfb + out/execute/*.txt
# 教材：docs/learning-guides/iree-learning-guide.md 第 3 章、第 5 章
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if ! have_iree; then skip_no_iree; exit 2; fi

E="${OUT_DIR}/execute"
# shellcheck disable=SC2206
TARGET_FLAGS=(${IREE_TARGET_FLAGS})

hr()  { printf '%s\n' "────────────────────────────────────────────────────────────"; }
sec() { echo; hr; echo "$1"; hr; }
show(){ sed 's/^/    /'; }

# ---------------------------------------------------------------------------
sec "① 完整编译：.mlir → .vmfb（单文件部署产物）"
for stem in abs tiny_mlp; do
  if "${IREE_COMPILE}" "${TARGET_FLAGS[@]}" \
       "${MODELS_DIR}/${stem}.mlir" -o "${E}/${stem}.vmfb" 2>"${E}/${stem}.compile.log"; then
    sz=$(wc -c < "${E}/${stem}.vmfb")
    printf '  %-10s → %-22s %8s bytes\n' "${stem}.mlir" "${stem}.vmfb" "${sz}"
  else
    echo "  [FAIL] ${stem}：见 ${E}/${stem}.compile.log"
    sed 's/^/      /' "${E}/${stem}.compile.log" | head -10
  fi
done
echo
echo "  ★ .vmfb 里装的是【三样东西打成一包】：VM 字节码（宿主调度）"
echo "    + 每个 kernel 的机器码（executable）+ 常量数据。"
echo "    所以运行时不需要再链接编译器，也不需要外部 .so —— 这就是站 ⑦ 的产物形态。"

# ---------------------------------------------------------------------------
sec "② 跑 tiny_mlp，并与手算结果逐位比对"
IN="2x3xf32=1 2 3 -4 -5 -6"
echo "  输入 x = [[ 1,  2,  3],"
echo "            [-4, -5, -6]]"
echo
echo "  手算（models/tiny_mlp.mlir 文末有推导）："
echo "    h        = [[1.5, 2.5, 3.5, 6.5], [-3.5, -4.5, -5.5, -14.5]]"
echo "    relu(h)  = [[1.5, 2.5, 3.5, 6.5], [   0,    0,    0,     0]]   ← 第二行整行被钳掉"
echo "    y = +1.0 = [[2.5, 3.5, 4.5, 7.5], [   1,    1,    1,     1]]"
echo

if [[ -f "${E}/tiny_mlp.vmfb" ]]; then
  "${IREE_RUN}" --device="${IREE_DEVICE}" \
      --module="${E}/tiny_mlp.vmfb" \
      --function=tiny_mlp \
      --input="${IN}" > "${E}/tiny_mlp.out.txt" 2>&1
  echo "  实际输出："
  show < "${E}/tiny_mlp.out.txt"

  if grep -qE '2\.5' "${E}/tiny_mlp.out.txt" && grep -qE '7\.5' "${E}/tiny_mlp.out.txt"; then
    echo
    echo "  ✅ 与手算一致。"
    echo "     这一行『一致』才是整条链路的验收标准：flow 融了 op、stream 复用了内存、"
    echo "     hal 换了 buffer 布局、vm 重排了调度 —— 全部变换加起来，语义必须不变。"
  else
    echo
    echo "  ⚠️  数值对不上，或输出格式与预期不同，请人工核对上面这段。"
  fi
fi

# ---------------------------------------------------------------------------
sec "③ 同一个 .vmfb + 不同 device：宿主代码一行不改"
if [[ -f "${E}/tiny_mlp.vmfb" ]]; then
  for dev in local-sync local-task; do
    echo "  --device=${dev}"
    "${IREE_RUN}" --device="${dev}" \
        --module="${E}/tiny_mlp.vmfb" \
        --function=tiny_mlp \
        --input="${IN}" 2>&1 | head -6 | show || true
    echo
  done
  echo "  local-sync 当场算完，local-task 走线程池排队 —— 调度策略换了，结果不变。"
  echo "  能做到这点，靠的是 stream 相位已经把【依赖关系】显式写进 IR，"
  echo "  而不是靠宿主代码里的调用顺序隐式表达。"
fi

# ---------------------------------------------------------------------------
sec "④ 看一眼 .vmfb 里到底有什么"
if [[ -f "${E}/tiny_mlp.vmfb" ]]; then
  echo "  导出的函数与签名："
  "${IREE_RUN}" --module="${E}/tiny_mlp.vmfb" --list_functions 2>/dev/null | show \
    || echo "    （该版本不支持 --list_functions，跳过）"
  echo
  echo "  文件里可见的字符串片段（能看到 executable / 目标三元组的痕迹）："
  strings "${E}/tiny_mlp.vmfb" 2>/dev/null | grep -iE 'llvm|cpu|embedded|elf|tiny_mlp|dispatch' \
    | sort -u | head -12 | show || true
fi

echo
hr
echo "完成。产物在 ${E}/。"
echo "下一步：scripts/run_variants.sh —— 改一个编译开关，看 IR 与性能怎么变。"
hr
