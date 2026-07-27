//===- ToyOps.td - Toy dialect 的操作定义 ------------------*- tablegen -*-===//
//
// 这个文件定义 toy dialect 里的三个操作：constant / add / mul。
// 这是整个项目最值得反复阅读的文件 —— 它展示了如何用声明式的方式
// 定义一个 Operation 的：输入、输出、属性、特征(trait)、装配格式和验证。
//
//===----------------------------------------------------------------------===//

#ifndef TOY_OPS
#define TOY_OPS

include "Toy/ToyDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// ConstantOp: toy.constant
//===----------------------------------------------------------------------===//
//
// 产生一个常量整数。例如：
//   %0 = toy.constant 42 : i32
//
// 学习要点：
//   - Pure trait：表示该 op 没有副作用，可被 CSE / DCE / 常量折叠等优化处理。
//   - 有一个 I32Attr 属性作为常量的值，一个 I32 类型的结果。
//   - hasFolder = 1：告诉 MLIR 我们会实现 fold() 方法（在 ToyOps.cpp 中）。
//
def Toy_ConstantOp : Toy_Op<"constant", [Pure, ConstantLike]> {
  let summary = "定义一个 i32 常量";

  let arguments = (ins I32Attr:$value);
  let results = (outs I32:$result);

  // 自定义装配格式：让文本形式写成 `toy.constant 42 : i32`。
  let assemblyFormat = "$value attr-dict `:` type($result)";

  // 我们会实现 fold()，以便常量能参与折叠。
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AddOp: toy.add
//===----------------------------------------------------------------------===//
//
// 两个 i32 相加。例如：
//   %2 = toy.add %0, %1 : i32
//
// 学习要点：
//   - Commutative trait：加法可交换，MLIR 优化时可利用这一点。
//   - hasFolder = 1：我们实现常量折叠 —— 当两个操作数都是常量时，
//     直接算出结果，把 toy.add 折叠成 toy.constant。
//
def Toy_AddOp : Toy_Op<"add", [Pure, Commutative]> {
  let summary = "两个 i32 相加";

  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// MulOp: toy.mul
//===----------------------------------------------------------------------===//
//
// 两个 i32 相乘。例如：
//   %3 = toy.mul %0, %1 : i32
//
// 练习建议：仿照 AddOp，给 MulOp 也实现 fold()，
// 并尝试加入 "x * 1 = x"、"x * 0 = 0" 这样的代数化简规则。
//
def Toy_MulOp : Toy_Op<"mul", [Pure, Commutative]> {
  let summary = "两个 i32 相乘";

  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";

  let hasFolder = 1;
}

#endif // TOY_OPS
