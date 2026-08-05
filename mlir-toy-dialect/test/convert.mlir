// Dialect Conversion 测试 —— 真实编译器做 lowering 的标准姿势。
//
// RUN: %toy-opt %s --toy-to-low-convert | %FileCheck %s
//
// 与 test/lowering.mlir（贪心版 --toy-to-low）对照着看，两者结果一样，
// 但机制完全不同：
//   贪心版：  "反复套用改写规则，直到没规则能命中" —— 漏写规则会静默留下 toy.*
//   转换版：  "先声明什么样的 IR 才合法(ConversionTarget)，
//             再让框架保证走到那个终点" —— 漏写规则会直接报
//             error: failed to legalize operation 'toy.mul'
//
// 转换版还多了一样贪心版给不了的能力：【类型转换】。
// TypeConverter 把 !toy.num 映射成 i32，装箱/拆箱随之退化成恒等操作而消失。
// 真实世界的对应物：bufferization（tensor -> memref）、
// ConvertMemRefToLLVM（memref -> !llvm.struct）、ConvertGPUToNVVM ……
//
// 运行方式：
//   手动：  toy-opt test/convert.mlir --toy-to-low-convert
//   自动：  cmake --build build --target check-toy

// 基本算术：toy.* 全部换成 low.*
// CHECK-LABEL: func.func @arith
// CHECK-NOT: toy.
// CHECK: low.constant 7 : i32
// CHECK: low.add
func.func @arith(%arg0: i32) -> i32 {
  %c = toy.constant 7 : i32
  %s = toy.add %arg0, %c : i32
  return %s : i32
}

// 类型转换：!toy.num 被 TypeConverter 换成 i32，box/unbox 一起消失。
// CHECK-LABEL: func.func @type_lowering
// CHECK-NOT: !toy.num
// CHECK-NOT: toy.box
// CHECK-NOT: toy.unbox
// CHECK: low.mul
func.func @type_lowering(%arg0: i32) -> i32 {
  %n = toy.box %arg0 : i32 -> !toy.num
  %x = toy.unbox %n : !toy.num -> i32
  %c = toy.constant 4 : i32
  %m = toy.mul %x, %c : i32
  return %m : i32
}

// toy.repeat / toy.yield 被 target.addLegalOp<> 显式放行，所以原样保留；
// 但它区域【内部】的 toy.add 仍然会被降低 —— 转换同样会递归进入 Region。
// CHECK-LABEL: func.func @inside_region
// CHECK: toy.repeat 2
// CHECK: low.add
// CHECK: toy.yield
func.func @inside_region(%arg0: i32, %arg1: i32) -> i32 {
  %r = toy.repeat 2 {
    %s = toy.add %arg0, %arg1 : i32
    toy.yield %s : i32
  } : i32
  return %r : i32
}
