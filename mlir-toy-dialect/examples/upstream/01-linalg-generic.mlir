// =============================================================================
// 01-linalg-generic.mlir —— 结构化 op：Gemm → Relu 在 linalg 层长什么样
// -----------------------------------------------------------------------------
// 这不是 toy dialect，是【上游 MLIR 自带的 dialect】，用现成的 mlir-opt 跑。
// 放在这里的理由：toy 是标量 i32 dialect，讲不了「张量 / 索引映射 / DPS」，
// 而这几个概念是读懂任何真实 AI 编译器 IR 的前提。
//
// 主角沿用全仓库的 tiny_mlp（Gemm → Relu → Add），取前两个算子，batch 取 2：
//     x:[2,3]  @  W:[3,4]  + b:[4]  →  h:[2,4]  →  relu  →  h_act:[2,4]
// 对应 docs/learning-guides/00-end-to-end-pipeline.md 的站 ⑤（多层降低）。
//
// 跑法：
//   bash scripts/run_upstream.sh          # 一次跑完三个文件
//   mlir-opt examples/upstream/01-linalg-generic.mlir            # 只解析回显
//   mlir-opt examples/upstream/01-linalg-generic.mlir \
//     --linalg-generalize-named-ops                              # matmul → generic，看它的真身
// =============================================================================

// 索引映射：把「循环变量」映射到「张量下标」。这是 linalg 的核心声明。
#id2  = affine_map<(d0, d1) -> (d0, d1)>          // 二维恒等：out[i,j] 用 in[i,j]
#bcast = affine_map<(d0, d1) -> (d1)>             // 广播：out[i,j] 用 bias[j]，i 被丢弃

// -----------------------------------------------------------------------------
// ① Gemm：用具名 op linalg.matmul
//    注意 outs 不是「输出缓冲」而是【累加的初始值】—— 所以要先 fill 成 0。
//    这就是 DPS（Destination-Passing Style）：结果与 outs 一一对应。
// -----------------------------------------------------------------------------
func.func @gemm(%x: tensor<2x3xf32>, %w: tensor<3x4xf32>) -> tensor<2x4xf32> {
  %zero = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<2x4xf32>
  %acc  = linalg.fill ins(%zero : f32) outs(%init : tensor<2x4xf32>) -> tensor<2x4xf32>
  %h    = linalg.matmul ins(%x, %w : tensor<2x3xf32>, tensor<3x4xf32>)
                        outs(%acc : tensor<2x4xf32>) -> tensor<2x4xf32>
  return %h : tensor<2x4xf32>
}

// -----------------------------------------------------------------------------
// ② Relu：用通用 op linalg.generic 手写一遍
//    三件事各归各位：
//      indexing_maps   —— 谁读谁的哪个下标（访问模式，编译器据此做依赖分析）
//      iterator_types  —— 每个循环维是 parallel 还是 reduction（可否并行/重排）
//      region 里的负载 —— 每个元素上到底算什么（payload）
// -----------------------------------------------------------------------------
func.func @relu(%h: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %zero = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<2x4xf32>
  %out = linalg.generic {
      indexing_maps  = [#id2, #id2],
      iterator_types = ["parallel", "parallel"]     // 两维都可并行 → 可任意 tile / 向量化
    } ins(%h : tensor<2x4xf32>) outs(%init : tensor<2x4xf32>) {
    ^bb0(%in: f32, %unused: f32):
      // 用 cmpf + select 而不是 maxf：跨 MLIR 版本都能跑，
      // 而且和 llvm-hello-compile 里 relu_sum 的 `v > 0 ? v : 0` 降下来完全同形。
      %p = arith.cmpf ogt, %in, %zero : f32
      %r = arith.select %p, %in, %zero : f32
      linalg.yield %r : f32
  } -> tensor<2x4xf32>
  return %out : tensor<2x4xf32>
}

// -----------------------------------------------------------------------------
// ③ bias_add：广播的访问模式只体现在 indexing_maps 上，region 里毫无察觉
//    对照 ② 只差第二条 map：#id2 换成 #bcast。
// -----------------------------------------------------------------------------
func.func @bias_add(%h: tensor<2x4xf32>, %b: tensor<4xf32>) -> tensor<2x4xf32> {
  %init = tensor.empty() : tensor<2x4xf32>
  %out = linalg.generic {
      indexing_maps  = [#id2, #bcast, #id2],
      iterator_types = ["parallel", "parallel"]
    } ins(%h, %b : tensor<2x4xf32>, tensor<4xf32>) outs(%init : tensor<2x4xf32>) {
    ^bb0(%in: f32, %bias: f32, %unused: f32):
      %r = arith.addf %in, %bias : f32
      linalg.yield %r : f32
  } -> tensor<2x4xf32>
  return %out : tensor<2x4xf32>
}

// -----------------------------------------------------------------------------
// ④ 归约：把 iterator_types 里出现 "reduction" 的样子摆出来，与 ②③ 对照
//    输出 map 里【不出现】的那个循环维，就是被归约掉的维。
//    这也是判断「两个 op 能不能融合」的第一眼依据。
// -----------------------------------------------------------------------------
#rowmap = affine_map<(d0, d1) -> (d0, d1)>
#redmap = affine_map<(d0, d1) -> (d0)>            // d1 消失了 → d1 是归约维

func.func @row_sum(%h: tensor<2x4xf32>) -> tensor<2xf32> {
  %zero = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<2xf32>
  %acc  = linalg.fill ins(%zero : f32) outs(%init : tensor<2xf32>) -> tensor<2xf32>
  %out = linalg.generic {
      indexing_maps  = [#rowmap, #redmap],
      iterator_types = ["parallel", "reduction"]   // d1 归约 → 不能随意并行/重排
    } ins(%h : tensor<2x4xf32>) outs(%acc : tensor<2xf32>) {
    ^bb0(%in: f32, %a: f32):
      %s = arith.addf %a, %in : f32                // 注意负载里要显式带上累加值 %a
      linalg.yield %s : f32
  } -> tensor<2xf32>
  return %out : tensor<2xf32>
}
