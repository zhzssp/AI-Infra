// Region（嵌套 IR）测试 —— MLIR 与 LLVM IR 最本质的区别。
//
// RUN: %toy-opt %s --toy-simplify | %FileCheck %s
//
// 看点一：IR 是一棵【树】，不是一条平坦的指令流。
//   module -> func.func -> (region) -> block -> toy.repeat -> (region)
//          -> block -> toy.mul / toy.yield
//   LLVM IR 里不可能出现"指令里再嵌一个函数体"，而这在 MLIR 里是常态：
//   scf.for / scf.if / gpu.launch / linalg.generic / func.func 全是带 Region 的 op。
//
// 看点二：区域【不是】IsolatedFromAbove，所以区域内可以直接引用外层的 %arg0。
//   （func.func 和 GPU kernel 则相反，它们是隔离的，必须显式传参。）
//
// 看点三：Pass 自动递归进入 Region —— 这才是最该记住的一点。
//   我们写 --toy-simplify 时压根没考虑过嵌套，但贪心驱动器会遍历整棵 IR 树，
//   所以区域【内部】的 x*1 照样被化简成 x，一行额外代码都不用写。
//
// 运行方式：
//   手动：  toy-opt test/region.mlir --toy-simplify
//   自动：  cmake --build build --target check-toy

// CHECK-LABEL: func.func @nested_simplify
// CHECK: toy.repeat 3
// 区域内部的 toy.mul 被化简掉了，连带死掉的 toy.constant 也被清理：
// CHECK-NOT: toy.mul
// CHECK-NOT: toy.constant
// CHECK: toy.yield %arg0 : i32
func.func @nested_simplify(%arg0: i32) -> i32 {
  %r = toy.repeat 3 {
    %one = toy.constant 1 : i32
    %m = toy.mul %arg0, %one : i32
    toy.yield %m : i32
  } : i32
  return %r : i32
}

// 区域里放一个不可化简的表达式，验证解析/打印的往返是正确的。
// CHECK-LABEL: func.func @nested_keep
// CHECK: toy.repeat 2
// CHECK: toy.add
// CHECK: toy.yield
func.func @nested_keep(%arg0: i32, %arg1: i32) -> i32 {
  %r = toy.repeat 2 {
    %s = toy.add %arg0, %arg1 : i32
    toy.yield %s : i32
  } : i32
  return %r : i32
}
