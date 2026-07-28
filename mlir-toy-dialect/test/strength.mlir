// 强度削减测试：展示"层级分工"最精彩的对比。
//
// RUN: %toy-opt %s --toy-to-low --low-strength-reduce | %FileCheck %s
//
// 完整链路（两个 Pass 串联）：
//   toy.mul %x, 4   --toy-to-low-->   low.mul %x, 4
//                   --low-strength-reduce-->   low.shl %x, 2
//
// 关键对比：
//   * "x * 4" 是 2 的幂，能被强度削减成左移 2 位；
//   * "x * 3" 不是 2 的幂，保持 low.mul 不变；
//   * 注意：这两个乘法在【高层 toy 层的 --toy-simplify 里都纹丝不动】，
//     因为高层只懂 x*1=x，且高层根本没有"移位"这个概念。
//     只有降低到 low 层之后，x*4 的优化机会才"看得见"。
//
// 运行方式：
//   手动：  toy-opt test/strength.mlir --toy-to-low --low-strength-reduce
//   自动：  cmake --build build --target check-toy

// x * 4  ->  x << 2
// CHECK-LABEL: func.func @mul_pow2
// CHECK-NOT: low.mul
// CHECK: low.shl %arg0, 2 : i32
func.func @mul_pow2(%arg0: i32) -> i32 {
  %c4 = toy.constant 4 : i32
  %r  = toy.mul %arg0, %c4 : i32
  return %r : i32
}

// x * 3  ->  不是 2 的幂，保持乘法
// CHECK-LABEL: func.func @mul_non_pow2
// CHECK-NOT: low.shl
// CHECK: low.mul
func.func @mul_non_pow2(%arg0: i32) -> i32 {
  %c3 = toy.constant 3 : i32
  %r  = toy.mul %arg0, %c3 : i32
  return %r : i32
}
