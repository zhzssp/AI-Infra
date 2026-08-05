//===- ToyOps.td - Toy dialect 的操作定义 ------------------*- tablegen -*-===//
//
// 这个文件定义 toy dialect 里的操作。
// 这是整个项目最值得反复阅读的文件 —— 它展示了如何用声明式的方式
// 定义一个 Operation 的：输入、输出、属性、特征(trait)、接口(interface)、
// 装配格式和验证。
//
// 文件分三部分：
//   第一部分  最小算术操作：constant / add / mul
//             —— 演示 Operand / Result / Attribute / Trait / Folder
//   第二部分  自定义类型的搬运工：box / unbox
//             —— 演示 Type 可扩展性（!toy.num）与"fold 返回已有 Value"
//   第三部分  带 Region 的操作：repeat / yield
//             —— 演示 MLIR 相对 LLVM 最本质的区别：嵌套 IR
//
//===----------------------------------------------------------------------===//

#ifndef TOY_OPS
#define TOY_OPS

include "Toy/ToyDialect.td"
include "Toy/ToyTypes.td"
include "Toy/ToyInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// 第一部分：最小算术操作
//===----------------------------------------------------------------------===//

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
//   - 刻意【不】实现 ToyCostOpInterface：常量不产生运行时代价，
//     借此演示"通用 Pass 只处理实现了接口的 op，其余自动跳过"。
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
//   - ToyCostOpInterface：直接挂上接口名，【不加】DeclareOpInterfaceMethods，
//     表示"我用接口的默认实现"（代价 = 1），一行 C++ 都不用写。
//
def Toy_AddOp : Toy_Op<"add", [Pure, Commutative, ToyCostOpInterface]> {
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
// 学习要点：
//   - DeclareOpInterfaceMethods<ToyCostOpInterface>：与 add 相反，
//     这里声明"我要【覆盖】接口的默认实现"，于是 tblgen 会在 MulOp 类里
//     生成 `unsigned getCost();` 的声明，必须在 ToyOps.cpp 里给出定义
//     （我们返回 5：乘法比加法贵）。
//
// 练习建议：给 MulOp 的 fold() 加入 "x * 1 = x"、"x * 0 = 0" 这样的化简规则。
//
def Toy_MulOp : Toy_Op<"mul",
    [Pure, Commutative, DeclareOpInterfaceMethods<ToyCostOpInterface>]> {
  let summary = "两个 i32 相乘";

  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// 第二部分：自定义类型 !toy.num 的装箱 / 拆箱
//===----------------------------------------------------------------------===//
//
// 这两个 op 唯一的存在意义，就是让【自定义类型】真正在 IR 里流动起来，
// 从而能演示两件事：
//   1) 降低时如何用 TypeConverter 把 !toy.num 换成 i32（ConvertToyToLow.cpp）；
//   2) fold() 除了"算出常量"，还可以"直接返回一个已经存在的 Value"。

//===----------------------------------------------------------------------===//
// BoxOp: toy.box —— 把内置 i32 装进 toy 层的 !toy.num
//===----------------------------------------------------------------------===//
//   %n = toy.box %x : i32 -> !toy.num
def Toy_BoxOp : Toy_Op<"box", [Pure]> {
  let summary = "把 i32 装箱成 !toy.num";

  let arguments = (ins I32:$input);
  let results = (outs Toy_NumType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// UnboxOp: toy.unbox —— 把 !toy.num 拆回内置 i32
//===----------------------------------------------------------------------===//
//   %y = toy.unbox %n : !toy.num -> i32
//
// hasFolder = 1：实现 unbox(box(x)) = x。
// 注意这个 fold 返回的是【一个已经存在的 Value】（box 的输入），
// 而不是像 add/mul 那样返回一个新算出来的 Attribute。
// OpFoldResult 就是 "Attribute | Value" 的联合体，这两种都合法。
def Toy_UnboxOp : Toy_Op<"unbox", [Pure]> {
  let summary = "把 !toy.num 拆箱回 i32";

  let arguments = (ins Toy_NumType:$input);
  let results = (outs I32:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// 第三部分：带 Region 的操作 —— MLIR 与 LLVM 最本质的区别
//===----------------------------------------------------------------------===//
//
// LLVM IR 是"扁平"的：一个函数 = 一串基本块，指令里不可能再嵌一个函数体。
// MLIR 的 Operation 里可以挂【Region（区域）】，Region 里装 Block，
// Block 里又装 Operation —— IR 是一棵可以无限嵌套的树。
//
// 这为什么重要？因为 for 循环、if 分支、GPU kernel、函数体、
// linalg.generic 的计算体、scf.forall 的并行域……全都是"带 Region 的 op"。
// 有了嵌套，高层结构（比如"这是一个可以并行的循环"）在降低到底层
// 跳转指令之前，可以【一直保留在 IR 里】，优化器因此看得见它。
// 这正是 "Multi-Level" 里 "Level" 得以存在的结构基础。

//===----------------------------------------------------------------------===//
// YieldOp: toy.yield —— Region 的终结符（Terminator）
//===----------------------------------------------------------------------===//
//
// 每个 Block 必须以一个 Terminator 结尾（就像 LLVM 基本块必须以 br/ret 结尾）。
// toy.yield 把 Region 内算出的值"交还"给外层的 toy.repeat 作为结果。
//
// 这里刻意用 AnyType 而不是 I32：终结符本身不限定类型，
// "交还的类型必须和 toy.repeat 的结果类型一致"这条约束交给
// RepeatOp::verify() 去检查——这正是"结构约束用 trait/ODS，
// 语义约束用自定义 verifier"的分工示范。
def Toy_YieldOp : Toy_Op<"yield", [Pure, Terminator]> {
  let summary = "toy.repeat 区域的终结符，把区域内的结果交还给外层";

  let arguments = (ins AnyType:$value);

  let assemblyFormat = "$value attr-dict `:` type($value)";
}

//===----------------------------------------------------------------------===//
// RepeatOp: toy.repeat —— 一个带 Region 的"重复执行 N 次"操作
//===----------------------------------------------------------------------===//
//
//   %r = toy.repeat 3 {
//     %one = toy.constant 1 : i32
//     %m   = toy.mul %arg0, %one : i32
//     toy.yield %m : i32
//   } : i32
//
// 学习要点：
//   - regions = (region SizedRegion<1>:$body)：声明一个恰好含 1 个 Block 的区域。
//   - SingleBlock trait：框架会替你验证"这个区域只有一个块且以终结符收尾"。
//   - 区域【不是】IsolatedFromAbove，所以区域内可以直接引用外层的 %arg0
//     （若加上 IsolatedFromAbove，就像函数一样必须显式传参，
//       GPU kernel、func.func 就是这么做的）。
//   - hasVerifier = 1：我们手写一个 verify()（在 ToyOps.cpp），
//     检查 count > 0，以及"区域交还的类型必须和本 op 的结果类型一致"。
//   - 所有 Pass 都会【自动递归进入 Region】。test/region.mlir 里，
//     --toy-simplify 会把区域内部的 x*1 也化简掉，你不用为此写任何额外代码。
//
def Toy_RepeatOp : Toy_Op<"repeat", [SingleBlock]> {
  let summary = "把区域体重复执行 count 次，产出区域 yield 的值";

  let arguments = (ins I32Attr:$count);
  let results = (outs I32:$result);
  let regions = (region SizedRegion<1>:$body);

  // attr-dict-with-keyword 会打印成 `attributes {...}`，必须放在 region 之前，
  // 否则解析器分不清后面那个 `{` 是属性字典还是区域的开始。
  let assemblyFormat = "$count attr-dict-with-keyword $body `:` type($result)";

  let hasVerifier = 1;
}

#endif // TOY_OPS
