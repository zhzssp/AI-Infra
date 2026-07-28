// 降低测试：验证 --toy-to-low 把高层 toy.* 改写成低层 low.*。
//
// RUN: %toy-opt %s --toy-to-low | %FileCheck %s
//
// 经过 --toy-to-low 后：
//   - 所有 toy.* 操作都应被替换成对应的 low.* 操作；
//   - 输出里不应再出现任何 "toy." 前缀。
//
// 运行方式：
//   手动：  toy-opt test/lowering.mlir --toy-to-low
//   自动：  cmake --build build --target check-toy

// CHECK-LABEL: func.func @lower_all
// CHECK-NOT: toy.
// CHECK: low.constant
// CHECK: low.mul
// CHECK: low.add
func.func @lower_all(%arg0: i32) -> i32 {
  %c3 = toy.constant 3 : i32
  %c5 = toy.constant 5 : i32
  %m  = toy.mul %arg0, %c3 : i32
  %r  = toy.add %m, %c5 : i32
  return %r : i32
}
