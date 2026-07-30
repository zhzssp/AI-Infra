//===- ToyOps.cpp - Toy dialect 操作的实现 --------------------------------===//
//
// 实现 toy 操作中那些"非自动生成"的部分，主要是 fold() 方法。
// fold() 是理解 MLIR 优化机制最好的入口：
//   - 当一个 op 的输入是编译期已知的常量时，fold() 可以在编译期把它算出来，
//     用一个常量替换掉这个 op。这就是 "constant folding"。
//
//===----------------------------------------------------------------------===//

#include "Toy/ToyOps.h"
#include "Toy/ToyDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::toy;

// 引入 TableGen 生成的操作定义（ToyOps.cpp.inc），
// 包含各操作的解析、打印、验证、构造等样板实现。
#define GET_OP_CLASSES
#include "Toy/ToyOps.cpp.inc"

//===----------------------------------------------------------------------===//
// ConstantOp::fold
//===----------------------------------------------------------------------===//
//
// 一个常量 op 折叠成"它自己的值"。这让常量可以被其他 op 的 fold 识别到。
// 对于 ConstantLike 的 op，返回它的 value 属性即可。
//
OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  llvm::errs() << "[trace] dialect: ConstantOp::fold 进入，直接返回常量值 "
               << getValueAttr() << "\n";
  // 直接返回该常量携带的属性值。
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// AddOp::fold
//===----------------------------------------------------------------------===//
//
// 当 lhs 和 rhs 都是常量时，直接算出 lhs + rhs 并返回一个新的整数属性，
// MLIR 框架会自动用一个 toy.constant 替换掉这个 toy.add。
//
// adaptor 里保存了各操作数"如果是常量的话"对应的 Attribute；
// 若某个操作数不是常量，则对应项为 null。
//
OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto lhs = llvm::dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  auto rhs = llvm::dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());

  llvm::errs() << "[trace] dialect: AddOp::fold 进入，lhs=";
  if (lhs) llvm::errs() << lhs.getValue().getZExtValue();
  else llvm::errs() << "(非常量)";
  llvm::errs() << " rhs=";
  if (rhs) llvm::errs() << rhs.getValue().getZExtValue();
  else llvm::errs() << "(非常量)";
  llvm::errs() << "\n";

  // 两个操作数都必须是常量，否则无法折叠。
  if (!lhs || !rhs) {
    llvm::errs() << "[trace] dialect: AddOp::fold 操作数非全常量，跳过\n";
    return {};
  }

  // 计算 lhs + rhs，构造一个新的 IntegerAttr 作为折叠结果。
  APInt result = lhs.getValue() + rhs.getValue();
  llvm::errs() << "[trace] dialect: AddOp::fold 常量折叠结果="
               << result.getZExtValue() << "\n";
  return IntegerAttr::get(getType(), result);
}

//===----------------------------------------------------------------------===//
// MulOp::fold
//===----------------------------------------------------------------------===//
//
// 常量 * 常量 -> 常量。
// 练习：在这里加入 "x * 1 = x"（返回 getLhs()/getRhs()）
//       和 "x * 0 = 0"（返回一个 0 常量属性）等代数化简规则。
//
OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  auto lhs = llvm::dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  auto rhs = llvm::dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());

  llvm::errs() << "[trace] dialect: MulOp::fold 进入，lhs=";
  if (lhs) llvm::errs() << lhs.getValue().getZExtValue();
  else llvm::errs() << "(非常量)";
  llvm::errs() << " rhs=";
  if (rhs) llvm::errs() << rhs.getValue().getZExtValue();
  else llvm::errs() << "(非常量)";
  llvm::errs() << "\n";

  if (!lhs || !rhs) {
    llvm::errs() << "[trace] dialect: MulOp::fold 操作数非全常量，跳过\n";
    return {};
  }

  APInt result = lhs.getValue() * rhs.getValue();
  llvm::errs() << "[trace] dialect: MulOp::fold 常量折叠结果="
               << result.getZExtValue() << "\n";
  return IntegerAttr::get(getType(), result);
}
