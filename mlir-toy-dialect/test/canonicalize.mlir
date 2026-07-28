// 常量折叠测试：本项目的重点，展示 MLIR 的优化机制。
//
// RUN: %toy-opt %s --canonicalize | %FileCheck %s
//
// 经过 --canonicalize（调用我们实现的 fold()）后：
//   - 10 + 20 应折叠为 toy.constant 30 : i32
//   - (10 + 20) * 2 应折叠为 toy.constant 60 : i32
//   - 中间的 toy.add / toy.mul 应被消除
//
// 运行方式：
//   手动：  toy-opt test/canonicalize.mlir --canonicalize
//   自动：  cmake --build build --target check-toy

// CHECK-NOT: toy.add
// CHECK: toy.constant 30 : i32
func.func @fold_add() -> i32 {
  %0 = toy.constant 10 : i32
  %1 = toy.constant 20 : i32
  %2 = toy.add %0, %1 : i32
  return %2 : i32
}

// CHECK-NOT: toy.mul
// CHECK: toy.constant 60 : i32
func.func @fold_add_mul() -> i32 {
  %0 = toy.constant 10 : i32
  %1 = toy.constant 20 : i32
  %2 = toy.constant 2 : i32
  %3 = toy.add %0, %1 : i32
  %4 = toy.mul %3, %2 : i32
  return %4 : i32
}
