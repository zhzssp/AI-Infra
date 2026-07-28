// 代数化简测试：验证自定义 Pass --toy-simplify（RewritePattern 演示）。
//
// RUN: %toy-opt %s --toy-simplify | %FileCheck %s
//
// 经过 --toy-simplify（调用我们写的 RewritePattern）后：
//   - x * 1 应被化简为 x     → toy.mul 消失
//   - x + 0 应被化简为 x     → toy.add 消失
//   - 变成死代码的 toy.constant 会被贪心驱动器顺带清理
//
// 运行方式：
//   手动：  toy-opt test/simplify.mlir --toy-simplify
//   自动：  cmake --build build --target check-toy

// CHECK-LABEL: func.func @mul_by_one
// CHECK-NOT: toy.mul
// CHECK: return %arg0
func.func @mul_by_one(%arg0: i32) -> i32 {
  %one = toy.constant 1 : i32
  %r = toy.mul %arg0, %one : i32
  return %r : i32
}

// CHECK-LABEL: func.func @add_zero
// CHECK-NOT: toy.add
// CHECK: return %arg0
func.func @add_zero(%arg0: i32) -> i32 {
  %zero = toy.constant 0 : i32
  %r = toy.add %zero, %arg0 : i32
  return %r : i32
}
