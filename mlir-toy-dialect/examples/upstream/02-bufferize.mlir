// =============================================================================
// 02-bufferize.mlir —— tensor 与 memref 的分界：谁能就地写，谁必须拷一份
// -----------------------------------------------------------------------------
// 这不是 toy dialect，是【上游 MLIR 自带的 dialect】，用现成的 mlir-opt 跑。
//
// 一句话主题：
//   tensor 是【值】—— 不可变，"改一个元素"在语义上是造一个新张量；
//   memref 是【缓冲】—— 有地址、可写、会别名。
//   bufferization 就是把前者落成后者，并决定【哪几处必须真的分配和拷贝】。
//
// 沿用 tiny_mlp 的 Relu 段。对应 00-end-to-end-pipeline.md 站 ⑤。
//
// 跑法（对比三份输出，这是本文件的全部价值）：
//   mlir-opt examples/upstream/02-bufferize.mlir                       # ① 原始 tensor 版
//   mlir-opt examples/upstream/02-bufferize.mlir \
//     --one-shot-bufferize="bufferize-function-boundaries"             # ② 落成 memref
//   mlir-opt examples/upstream/02-bufferize.mlir \
//     --one-shot-bufferize="bufferize-function-boundaries" \
//     --buffer-deallocation-pipeline                                   # ③ 补上释放
//
//   数一下 ② 里出现了几处 memref.alloc / memref.copy —— 那就是"没能就地做"的代价。
// =============================================================================

#id2 = affine_map<(d0, d1) -> (d0, d1)>

// -----------------------------------------------------------------------------
// ① 能就地做：%h 之后再没人用，outs 直接复用它
//    bufferize 之后【不需要】新的 alloc —— 输入缓冲被原地改写。
// -----------------------------------------------------------------------------
func.func @relu_can_inplace(%h: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %zero = arith.constant 0.0 : f32
  %out = linalg.generic {
      indexing_maps  = [#id2, #id2],
      iterator_types = ["parallel", "parallel"]
    } ins(%h : tensor<2x4xf32>) outs(%h : tensor<2x4xf32>) {   // ← outs 就是 ins
    ^bb0(%in: f32, %o: f32):
      %p = arith.cmpf ogt, %in, %zero : f32
      %r = arith.select %p, %in, %zero : f32
      linalg.yield %r : f32
  } -> tensor<2x4xf32>
  return %out : tensor<2x4xf32>
}

// -----------------------------------------------------------------------------
// ② 必须拷贝：%h 在 relu 之后【还要再被用一次】
//    如果就地改写 %h，第二个使用者读到的就是被 relu 过的值 —— 语义错了。
//    所以 bufferization 必须插入一次 alloc + copy。这就是「读后写冲突」。
// -----------------------------------------------------------------------------
func.func @relu_must_copy(%h: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>) {
  %zero = arith.constant 0.0 : f32
  %relu = linalg.generic {
      indexing_maps  = [#id2, #id2],
      iterator_types = ["parallel", "parallel"]
    } ins(%h : tensor<2x4xf32>) outs(%h : tensor<2x4xf32>) {
    ^bb0(%in: f32, %o: f32):
      %p = arith.cmpf ogt, %in, %zero : f32
      %r = arith.select %p, %in, %zero : f32
      linalg.yield %r : f32
  } -> tensor<2x4xf32>

  // %h 在这里【第二次】被读 —— 它必须还是原值。
  %twice = linalg.generic {
      indexing_maps  = [#id2, #id2],
      iterator_types = ["parallel", "parallel"]
    } ins(%h : tensor<2x4xf32>) outs(%h : tensor<2x4xf32>) {
    ^bb0(%in: f32, %o: f32):
      %d = arith.addf %in, %in : f32
      linalg.yield %d : f32
  } -> tensor<2x4xf32>

  return %relu, %twice : tensor<2x4xf32>, tensor<2x4xf32>
}

// -----------------------------------------------------------------------------
// ③ 已经在 memref 世界里：别名是显式的，编译器不再替你判断
//    同一段 relu 写成 memref 版 —— 没有 tensor 那层「值语义」保护，
//    %buf 被就地改写是你自己写下的事实，写错就是错，没人拦。
//    这正是 bufferization 之后【信息变少】的地方：站 ⑤ 之后 noalias 之类的保证
//    必须由降低过程主动附上（见 03-memref-to-llvm.mlir）。
// -----------------------------------------------------------------------------
func.func @relu_memref(%buf: memref<2x4xf32>) {
  %zero = arith.constant 0.0 : f32
  linalg.generic {
      indexing_maps  = [#id2, #id2],
      iterator_types = ["parallel", "parallel"]
    } ins(%buf : memref<2x4xf32>) outs(%buf : memref<2x4xf32>) {
    ^bb0(%in: f32, %o: f32):
      %p = arith.cmpf ogt, %in, %zero : f32
      %r = arith.select %p, %in, %zero : f32
      linalg.yield %r : f32
  }
  return
}
