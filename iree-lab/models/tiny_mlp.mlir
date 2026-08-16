// =============================================================================
// tiny_mlp.mlir —— 全仓库图级主角在 linalg 层的样子
// -----------------------------------------------------------------------------
//   x:[2,3] ──Gemm(W:[4,3], b:[4], transB=1)──▶ h:[2,4] ──Relu──▶ ──Add(bias2)──▶ y:[2,4]
//
// 与 onnx-delegate-lab/onnx_lab/01_build_and_infer.py 建的是【同一个模型】，
// batch 取 2、权重取好认的整数，方便手算校验。
//
// 为什么 iree-lab 要有两个模型：
//   models/abs.mlir       只有一个 op → 用来【逐相位精读】，每一相都短到能整篇看完
//   models/tiny_mlp.mlir  三个 op    → 用来看【融合与 dispatch 划分】，abs 演示不了
//
// 链路位置：站 ⑤（多层降低）→ 站 ⑦（打包与运行时）。
// 全图见 docs/learning-guides/00-end-to-end-pipeline.md。
// =============================================================================

// ---- 索引映射 ---------------------------------------------------------------
// 【本文件最值得看的一行】：ONNX 里 transB=1 这个 attribute，
// 到了 linalg 层就是 #mapW 把 W 的下标写成 (n, k) 而不是 (k, n)。
// 不需要真的转置数据，也不需要一个 transpose op —— 换个下标顺序而已。
#mapX = affine_map<(m, n, k) -> (m, k)>     // x[m][k]
#mapW = affine_map<(m, n, k) -> (n, k)>     // W[n][k]  ← transB 就体现在这
#mapH = affine_map<(m, n, k) -> (m, n)>     // h[m][n]，不含 k → k 是归约维

#id2   = affine_map<(d0, d1) -> (d0, d1)>
#bcast = affine_map<(d0, d1) -> (d1)>       // 广播：out[i][j] 读 bias[j]

func.func @tiny_mlp(%x: tensor<2x3xf32>) -> tensor<2x4xf32> {
  %zero = arith.constant 0.0 : f32

  // W 用 [out, in] 布局，与 nn.Linear / relay.nn.dense / ONNX Gemm(transB=1) 一致
  %W = arith.constant dense<[[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [1.0, 1.0, 1.0]]> : tensor<4x3xf32>
  %b     = arith.constant dense<0.5> : tensor<4xf32>
  %bias2 = arith.constant dense<1.0> : tensor<4xf32>

  // ---------------------------------------------------------------------------
  // ① Gemm 的 C（也就是 b）就是累加的【初始值】—— DPS 的直接体现。
  //    先把 b 广播成 [2,4] 当 outs，而不是 fill 0 之后再加一遍。
  // ---------------------------------------------------------------------------
  %init = tensor.empty() : tensor<2x4xf32>
  %acc = linalg.generic {
      indexing_maps  = [#bcast, #id2],
      iterator_types = ["parallel", "parallel"]
    } ins(%b : tensor<4xf32>) outs(%init : tensor<2x4xf32>) {
    ^bb0(%bv: f32, %o: f32):
      linalg.yield %bv : f32
  } -> tensor<2x4xf32>

  // ② Gemm 本体：iterator_types 里那个 reduction 就是 K 维
  %h = linalg.generic {
      indexing_maps  = [#mapX, #mapW, #mapH],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%x, %W : tensor<2x3xf32>, tensor<4x3xf32>)
      outs(%acc : tensor<2x4xf32>) {
    ^bb0(%xv: f32, %wv: f32, %a: f32):
      %m = arith.mulf %xv, %wv : f32
      %s = arith.addf %a, %m : f32
      linalg.yield %s : f32
  } -> tensor<2x4xf32>

  // ③ Relu：纯逐元素，两维都 parallel → 能被吸进上游 Gemm 的 dispatch
  %init2 = tensor.empty() : tensor<2x4xf32>
  %r = linalg.generic {
      indexing_maps  = [#id2, #id2],
      iterator_types = ["parallel", "parallel"]
    } ins(%h : tensor<2x4xf32>) outs(%init2 : tensor<2x4xf32>) {
    ^bb0(%in: f32, %o: f32):
      %p = arith.cmpf ogt, %in, %zero : f32
      %v = arith.select %p, %in, %zero : f32
      linalg.yield %v : f32
  } -> tensor<2x4xf32>

  // ④ Add bias2：同样逐元素，只是第二个输入走广播映射
  %init3 = tensor.empty() : tensor<2x4xf32>
  %y = linalg.generic {
      indexing_maps  = [#id2, #bcast, #id2],
      iterator_types = ["parallel", "parallel"]
    } ins(%r, %bias2 : tensor<2x4xf32>, tensor<4xf32>)
      outs(%init3 : tensor<2x4xf32>) {
    ^bb0(%in: f32, %bv: f32, %o: f32):
      %v = arith.addf %in, %bv : f32
      linalg.yield %v : f32
  } -> tensor<2x4xf32>

  return %y : tensor<2x4xf32>
}

// =============================================================================
// 手算校验（scripts/run_execute.sh 会自动比对）
// -----------------------------------------------------------------------------
//   x = [[ 1,  2,  3],
//        [-4, -5, -6]]
//
//   h = x @ W^T + b
//     row0 = [1, 2, 3, 1+2+3]      + 0.5 = [ 1.5,  2.5,  3.5,   6.5]
//     row1 = [-4, -5, -6, -4-5-6]  + 0.5 = [-3.5, -4.5, -5.5, -14.5]
//
//   relu(h)
//     row0 = [1.5, 2.5, 3.5, 6.5]
//     row1 = [0,   0,   0,   0  ]        ← 整行被钳掉，Relu 不是摆设
//
//   y = relu(h) + 1.0
//     row0 = [2.5, 3.5, 4.5, 7.5]
//     row1 = [1,   1,   1,   1  ]
// =============================================================================
