#!/usr/bin/env bash
# =============================================================================
# run_upstream.sh —— 跑 examples/upstream/ 下的上游 dialect 样例
# -----------------------------------------------------------------------------
#   这些样例【不属于 toy dialect】，用 conda 环境里现成的 mlir-opt 跑，
#   不需要构建本项目的 toy-opt。
#
#   为什么单独放一个入口：toy 是标量 i32 dialect，讲不了
#   「张量语义 / 索引映射 / bufferization / memref 降到 LLVM」这几件事，
#   而它们是读懂任何真实 AI 编译器 IR 的前提。
#
#   产物：out/upstream/*.mlir（各阶段 IR）+ UPSTREAM.md（阅读指引）
# =============================================================================
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

EX="${TOY_PROJECT_DIR}/examples/upstream"
OUT="${TOY_PROJECT_DIR}/out/upstream"
MLIR_OPT="${TOY_BIN_DIR}/mlir-opt"
MLIR_TRANSLATE="${TOY_BIN_DIR}/mlir-translate"
mkdir -p "${OUT}"

if [[ ! -x "${MLIR_OPT}" ]]; then
  echo "[upstream] 找不到 mlir-opt：${MLIR_OPT}" >&2
  echo "[upstream] 请先运行 scripts/setup.sh 建好 conda 环境。" >&2
  exit 1
fi

exec > >(tee "${OUT}/upstream.log") 2>&1

sep()  { printf '\n========== %s ==========\n' "$1"; }
cmd()  { printf '  [命令] %s\n' "$*"; }
look() { printf '  [观察] %s\n' "$*"; }
show() {
  local buf; buf="$(cat)"
  if [[ -z "${buf}" ]]; then printf '  （无输出：本机 MLIR 版本可能不同，跳过即可）\n'; return; fi
  printf '%s\n' "${buf}" | sed 's/^/  │ /'
}

echo "############################################################"
echo "#  上游 dialect 样例（linalg / bufferize / memref→llvm）"
echo "#  产物目录: ${OUT}"
echo "############################################################"

# -----------------------------------------------------------------------------
sep "① linalg：indexing_maps + iterator_types + DPS"
cmd "mlir-opt 01-linalg-generic.mlir"
"${MLIR_OPT}" "${EX}/01-linalg-generic.mlir" -o "${OUT}/01_parsed.mlir" \
  && look "解析通过。四个函数：gemm(具名 op) / relu / bias_add(广播) / row_sum(归约)"

cmd "mlir-opt --linalg-generalize-named-ops  → 看 linalg.matmul 的真身"
"${MLIR_OPT}" --linalg-generalize-named-ops "${EX}/01-linalg-generic.mlir" \
  -o "${OUT}/01_generalized.mlir" 2>/dev/null \
  && grep -A4 'iterator_types' "${OUT}/01_generalized.mlir" | head -12 | show
look "matmul 展开后 iterator_types 是 [parallel, parallel, reduction] —— 第三维就是 K"

# -----------------------------------------------------------------------------
sep "② bufferize：哪几处必须真的 alloc + copy"
cmd "mlir-opt --one-shot-bufferize='bufferize-function-boundaries'"
"${MLIR_OPT}" --one-shot-bufferize="bufferize-function-boundaries" \
  "${EX}/02-bufferize.mlir" -o "${OUT}/02_bufferized.mlir" 2>/dev/null
if [[ -s "${OUT}/02_bufferized.mlir" ]]; then
  printf '  %-28s %s\n' "memref.alloc 次数:" "$(grep -c 'memref.alloc' "${OUT}/02_bufferized.mlir")"
  printf '  %-28s %s\n' "memref.copy  次数:" "$(grep -c 'memref.copy'  "${OUT}/02_bufferized.mlir")"
  look "relu_can_inplace 不需要 alloc；relu_must_copy 必须要 —— 差别只在「%h 之后还被不被用」"
else
  look "本机 MLIR 的 one-shot-bufferize 选项名可能不同，直接读 02-bufferize.mlir 的注释即可"
fi

# -----------------------------------------------------------------------------
sep "③ memref → LLVM：一个 memref 摊平成几个参数"
MEMREF_PASS="--finalize-memref-to-llvm"
if ! "${MLIR_OPT}" ${MEMREF_PASS} "${EX}/03-memref-to-llvm.mlir" -o /dev/null 2>/dev/null; then
  MEMREF_PASS="--convert-memref-to-llvm"     # 老版本的叫法
fi
cmd "mlir-opt --convert-arith-to-llvm ${MEMREF_PASS} --convert-func-to-llvm --reconcile-unrealized-casts"
"${MLIR_OPT}" --convert-arith-to-llvm ${MEMREF_PASS} \
  --convert-func-to-llvm --reconcile-unrealized-casts \
  "${EX}/03-memref-to-llvm.mlir" -o "${OUT}/03_llvm_dialect.mlir" 2>/dev/null
if [[ -s "${OUT}/03_llvm_dialect.mlir" ]]; then
  grep -E '^\s*llvm\.func @relu_first' "${OUT}/03_llvm_dialect.mlir" | head -6 | show
  look "rank=1 → 3+2*1 = 5 个参数；rank=2 → 3+2*2 = 7 个。allocated/aligned/offset/sizes/strides"
fi

if [[ -x "${MLIR_TRANSLATE}" && -s "${OUT}/03_llvm_dialect.mlir" ]]; then
  cmd "mlir-translate --mlir-to-llvmir  → 真正的 .ll"
  "${MLIR_TRANSLATE}" --mlir-to-llvmir "${OUT}/03_llvm_dialect.mlir" -o "${OUT}/03_out.ll" 2>/dev/null
  if [[ -s "${OUT}/03_out.ll" ]]; then
    grep -E '^define .*@relu_first' "${OUT}/03_out.ll" | head -4 | show
    look "注意这些 ptr 参数【没有 noalias / align】—— 站 ⑤ 不主动附上，站 ⑥ 就补不回来了"
  fi
fi

# -----------------------------------------------------------------------------
{
cat <<'MD'
# 上游 dialect 样例 —— 本次运行报告

由 `scripts/run_upstream.sh` 生成。这些样例**不属于 toy dialect**，
用 conda 环境自带的 `mlir-opt` 跑，补的是 toy 讲不了的三件事。

| 文件 | 补什么 | 对应教材 |
|------|--------|---------|
| `01-linalg-generic.mlir` | 张量语义、`indexing_maps`、`iterator_types`、DPS | mlir-learning-guide §8 |
| `02-bufferize.mlir` | tensor→memref、in-place vs copy | mlir-learning-guide §8 |
| `03-memref-to-llvm.mlir` | memref descriptor 摊平、与 LLVM 的接缝 | llvm-learning-guide 第 7 章 |

## 产物

| 文件 | 说明 |
|------|------|
| `01_parsed.mlir` | 解析回显 |
| `01_generalized.mlir` | `linalg.matmul` 展开成 `linalg.generic` 的真身 |
| `02_bufferized.mlir` | bufferize 之后，数 `memref.alloc` / `memref.copy` |
| `03_llvm_dialect.mlir` | 全部进 `llvm` dialect，看函数签名被摊平成几个参数 |
| `03_out.ll` | 真正的 LLVM IR |

## 三个必须能回答的问题

1. `linalg.generic` 的 `outs` 为什么不是"输出缓冲"而是"累加初始值"？
2. `relu_can_inplace` 与 `relu_must_copy` 的源码几乎一样，为什么后者必须 alloc + copy？
3. `03_out.ll` 里那两个 `ptr` 参数没有 `noalias`，下游会因此少做什么优化？

答不上第 3 问，就回 `docs/learning-guides/00-end-to-end-pipeline.md` 第 5 章断链表。
MD
} > "${OUT}/UPSTREAM.md"

sep "结束"
echo "  产物目录 : ${OUT}"
echo "  阅读入口 : ${OUT}/UPSTREAM.md  ← 建议先看这份"
