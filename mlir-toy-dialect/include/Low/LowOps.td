//===- LowOps.td - Low dialect 的操作定义 ------------------*- tablegen -*-===//
//
// 定义 low dialect 里的四个操作：constant / add / mul / shl。
// 其中 shl（左移）是高层 toy dialect 没有的【硬件级】概念，
// 它是"层级差异"最直观的体现。
//
//===----------------------------------------------------------------------===//

#ifndef LOW_OPS
#define LOW_OPS

include "Low/LowDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// 复用【toy 层定义的】代价接口。
// 这一行是本文件最值得注意的地方：接口不属于任何一个 dialect，
// low 层的 op 照样可以实现它，于是 --toy-print-cost 这个 Pass
// 不用改一行代码就能同时给 toy 层和 low 层算代价。
include "Toy/ToyInterfaces.td"

//===----------------------------------------------------------------------===//
// low.constant —— 低层常量，语义等价于 toy.constant
//===----------------------------------------------------------------------===//
def Low_ConstantOp : Low_Op<"constant", [Pure, ConstantLike]> {
  let summary = "定义一个 i32 常量（低层）";
  let arguments = (ins I32Attr:$value);
  let results = (outs I32:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// low.add —— 低层加法
//===----------------------------------------------------------------------===//
// 用接口的默认实现：代价 = 1。
def Low_AddOp : Low_Op<"add", [Pure, Commutative, ToyCostOpInterface]> {
  let summary = "两个 i32 相加（低层）";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// low.mul —— 低层乘法（强度削减的改写目标）
//===----------------------------------------------------------------------===//
// 覆盖接口默认实现：代价 = 5（见 LowOps.cpp 的 MulOp::getCost）。
// extraClassDeclaration 把声明写进生成的类；不要只靠
// DeclareOpInterfaceMethods 的第二个参数（增量构建漏跑 tblgen 时会缺声明）。
// 强度削减把它换成 low.shl（代价 1）之后，--toy-print-cost 算出来的
// 总代价会从 5 掉到 1 —— 这就是"优化到底赚了多少"的量化证据。
def Low_MulOp : Low_Op<"mul", [Pure, Commutative, ToyCostOpInterface]> {
  let summary = "两个 i32 相乘（低层）";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let extraClassDeclaration = [{
    unsigned getCost();
  }];
}

//===----------------------------------------------------------------------===//
// low.shl —— 逻辑左移：把 $operand 左移 $amount 位（= 乘以 2^amount）
//===----------------------------------------------------------------------===//
//
// 这是 toy 层【没有】的操作。移位量 $amount 用一个属性(I32Attr)携带，
// 文本形如： low.shl %x, 2 : i32
//
// 强度削减 Pass 会把 "low.mul %x, 4" 改写成 "low.shl %x, 2"。
//
def Low_ShlOp : Low_Op<"shl", [Pure, ToyCostOpInterface]> {
  let summary = "逻辑左移 amount 位（等价于乘以 2^amount）";
  let arguments = (ins I32:$operand, I32Attr:$amount);
  let results = (outs I32:$result);
  let assemblyFormat = "$operand `,` $amount attr-dict `:` type($result)";
}

#endif // LOW_OPS
