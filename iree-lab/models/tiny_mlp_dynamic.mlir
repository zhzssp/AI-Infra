// =============================================================================
// tiny_mlp_dynamic.mlir —— 与 tiny_mlp.mlir 逐字相同，只把 batch 维改成 ?
// -----------------------------------------------------------------------------
// 存在的唯一目的：做【对照组】。
//
//   tiny_mlp.mlir          batch = 2   编译期什么都知道
//   tiny_mlp_dynamic.mlir  batch = ?   编译期只知道「有这么一维」
//
// 两份编出来的 IR 一 diff，就能看见「静态形状」到底买到了什么：
// 循环边界能不能常量化、要不要在运行期算 workload、buffer 大小能不能提前定。
// 这是 00-end-to-end-pipeline.md 站 ① 那句「形状一旦丢了就补不回来」的实验证据。
//
// 跑法：scripts/run_variants.sh
// =============================================================================

#mapX = affine_map<(m, n, k) -> (m, k)>
#mapW = affine_map<(m, n, k) -> (n, k)>
#mapH = affine_map<(m, n, k) -> (m, n)>

#id2   = affine_map<(d0, d1) -> (d0, d1)>
#bcast = affine_map<(d0, d1) -> (d1)>

func.func @tiny_mlp_dynamic(%x: tensor<?x3xf32>) -> tensor<?x4xf32> {
  %c0   = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32

  // 静态版本里没有这一行 —— batch 只能到运行期才问得出来
  %m = tensor.dim %x, %c0 : tensor<?x3xf32>

  %W = arith.constant dense<[[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [1.0, 1.0, 1.0]]> : tensor<4x3xf32>
  %b     = arith.constant dense<0.5> : tensor<4xf32>
  %bias2 = arith.constant dense<1.0> : tensor<4xf32>

  %init = tensor.empty(%m) : tensor<?x4xf32>
  %acc = linalg.generic {
      indexing_maps  = [#bcast, #id2],
      iterator_types = ["parallel", "parallel"]
    } ins(%b : tensor<4xf32>) outs(%init : tensor<?x4xf32>) {
    ^bb0(%bv: f32, %o: f32):
      linalg.yield %bv : f32
  } -> tensor<?x4xf32>

  %h = linalg.generic {
      indexing_maps  = [#mapX, #mapW, #mapH],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%x, %W : tensor<?x3xf32>, tensor<4x3xf32>)
      outs(%acc : tensor<?x4xf32>) {
    ^bb0(%xv: f32, %wv: f32, %a: f32):
      %mm = arith.mulf %xv, %wv : f32
      %s  = arith.addf %a, %mm : f32
      linalg.yield %s : f32
  } -> tensor<?x4xf32>

  %init2 = tensor.empty(%m) : tensor<?x4xf32>
  %r = linalg.generic {
      indexing_maps  = [#id2, #id2],
      iterator_types = ["parallel", "parallel"]
    } ins(%h : tensor<?x4xf32>) outs(%init2 : tensor<?x4xf32>) {
    ^bb0(%in: f32, %o: f32):
      %p = arith.cmpf ogt, %in, %zero : f32
      %v = arith.select %p, %in, %zero : f32
      linalg.yield %v : f32
  } -> tensor<?x4xf32>

  %init3 = tensor.empty(%m) : tensor<?x4xf32>
  %y = linalg.generic {
      indexing_maps  = [#id2, #bcast, #id2],
      iterator_types = ["parallel", "parallel"]
    } ins(%r, %bias2 : tensor<?x4xf32>, tensor<4xf32>)
      outs(%init3 : tensor<?x4xf32>) {
    ^bb0(%in: f32, %bv: f32, %o: f32):
      %v = arith.addf %in, %bv : f32
      linalg.yield %v : f32
  } -> tensor<?x4xf32>

  return %y : tensor<?x4xf32>
}
