// =============================================================================
// 03-memref-to-llvm.mlir —— MLIR 与 LLVM 的接缝：一个 memref 被摊平成 5 个参数
// -----------------------------------------------------------------------------
// 这不是 toy dialect，是【上游 MLIR 自带的 dialect】，用现成的 mlir-opt / mlir-translate 跑。
//
// 这个文件是链路上站 ⑤（多层降低）交给站 ⑥（LLVM 指令生成）的那一刻，
// 也是 docs/learning-guides/llvm-learning-guide.md 第 7 章「接缝」那一节的实物。
//
// 内容仍是 tiny_mlp 的 Relu，但 shape 故意写成动态的 memref<?xf32>，
// 这样 descriptor 里的 sizes / strides 才有东西可看。
//
// 跑法（一步步看它怎么塌下去）：
//   mlir-opt examples/upstream/03-memref-to-llvm.mlir                       # ① 原始
//   mlir-opt examples/upstream/03-memref-to-llvm.mlir \
//     --convert-arith-to-llvm --finalize-memref-to-llvm \
//     --convert-func-to-llvm --reconcile-unrealized-casts                   # ② 全进 llvm dialect
//   ... | mlir-translate --mlir-to-llvmir                                   # ③ 变成真正的 .ll
//
//   老版本 MLIR 里 --finalize-memref-to-llvm 叫 --convert-memref-to-llvm。
//   run_upstream.sh 会自动试两个名字。
//
// 看完 ② 之后要能回答：函数签名里那 5 个参数分别是什么？（答案在文件末尾）
// =============================================================================

// -----------------------------------------------------------------------------
// ① 动态 shape：descriptor 必须带上 size 和 stride
// -----------------------------------------------------------------------------
func.func @relu_first(%buf: memref<?xf32>) {
  %c0   = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %v = memref.load %buf[%c0] : memref<?xf32>
  %p = arith.cmpf ogt, %v, %zero : f32
  %r = arith.select %p, %v, %zero : f32
  memref.store %r, %buf[%c0] : memref<?xf32>
  return
}

// -----------------------------------------------------------------------------
// ② 静态 shape：对照组。size/stride 是编译期常量，
//    descriptor 字段还在，但地址计算里的乘加会被常量折叠掉。
// -----------------------------------------------------------------------------
func.func @relu_first_static(%buf: memref<8xf32>) {
  %c0   = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %v = memref.load %buf[%c0] : memref<8xf32>
  %p = arith.cmpf ogt, %v, %zero : f32
  %r = arith.select %p, %v, %zero : f32
  memref.store %r, %buf[%c0] : memref<8xf32>
  return
}

// -----------------------------------------------------------------------------
// ③ 二维动态：sizes/strides 各两项，摊平后参数个数 = 3 + 2*rank
//    自己先数一遍再跑，看猜得对不对。
// -----------------------------------------------------------------------------
func.func @relu_first_2d(%buf: memref<?x?xf32>) {
  %c0   = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %v = memref.load %buf[%c0, %c0] : memref<?x?xf32>
  %p = arith.cmpf ogt, %v, %zero : f32
  %r = arith.select %p, %v, %zero : f32
  memref.store %r, %buf[%c0, %c0] : memref<?x?xf32>
  return
}

// =============================================================================
// 答案与要点
// -----------------------------------------------------------------------------
// @relu_first 降完之后签名是：
//   (!llvm.ptr, !llvm.ptr, i64, i64, i64)
//    ^allocated ^aligned   ^offset ^size[0] ^stride[0]
//
//   allocated ptr —— malloc 返回的原始指针，【只用于 free】
//   aligned ptr   —— 对齐后的数据基址，【所有访存都从它算起】
//   offset/size/stride —— 单位是【元素】不是字节
//
// 地址公式（读懂所有 memref lowering 结果的钥匙）：
//   addr = aligned_ptr + (offset + Σ_d index_d * stride_d) * sizeof(elem)
//
// 【接缝上最值得注意的一点】
//   降完之后那两个 ptr 参数【什么属性都没有】—— 没有 noalias、没有 align、
//   没有 dereferenceable。LLVM 只能给出 MayAlias，向量化随之退化。
//   这些属性必须由 lowering 主动附上，是链路上典型的「信息只能丢、不能补」。
//   对照 llvm-hello-compile 里 axpy（有 restrict）与 axpy_may_alias（无）的两份 IR。
// =============================================================================
