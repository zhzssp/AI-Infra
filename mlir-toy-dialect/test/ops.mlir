// 基本语法测试：验证 toy dialect 的解析与打印。
//
// RUN 行会被 lit 解析执行；CHECK 行用于 FileCheck 断言输出。
//   RUN: toy-opt %s | FileCheck %s
//
// 运行方式：
//   手动：  toy-opt test/ops.mlir
//   自动：  cmake --build build --target check-toy

// CHECK: toy.constant 10 : i32
%0 = toy.constant 10 : i32
// CHECK: toy.constant 20 : i32
%1 = toy.constant 20 : i32

// CHECK: toy.add %{{.*}}, %{{.*}} : i32
%2 = toy.add %0, %1 : i32

// CHECK: toy.mul %{{.*}}, %{{.*}} : i32
%3 = toy.mul %2, %0 : i32
