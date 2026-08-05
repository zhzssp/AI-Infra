//===- ToyDialect.cpp - Toy dialect 的实现 --------------------------------===//
//
// 实现 Toy dialect 的初始化：把所有操作和自定义类型注册进 dialect。
//
//===----------------------------------------------------------------------===//

#include "Toy/ToyDialect.h"
#include "Toy/ToyOps.h"
#include "Toy/ToyTypes.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::toy;

// 引入 TableGen 生成的【自定义类型】定义（ToyOpsTypes.cpp.inc，
// 由 ToyOps.td 里 include 的 ToyTypes.td 生成）。
// 它会给出 NumType 的存储、解析、打印实现，以及两个静态函数
// generatedTypeParser / generatedTypePrinter。
// 必须放在下面的 ToyOpsDialect.cpp.inc 之前 —— 因为 dialect 的
// parseType()/printType() 就是去调用这两个静态函数的。
#define GET_TYPEDEF_CLASSES
#include "Toy/ToyOpsTypes.cpp.inc"

// 引入 TableGen 生成的 Dialect 定义（ToyOpsDialect.cpp.inc），
// 里面包含 ToyDialect 类的构造/析构、parseType/printType 等样板代码。
#include "Toy/ToyOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Toy dialect 初始化
//===----------------------------------------------------------------------===//

// initialize() 由 MLIR 在 dialect 加载时调用。
// 我们在这里把所有操作和类型注册进来 —— 这样解析器才认识
// toy.constant 这样的操作，以及 !toy.num 这样的类型。
void ToyDialect::initialize() {
  llvm::errs() << "[trace] dialect: ToyDialect::initialize 开始注册 toy 操作与类型\n";

  addOperations<
#define GET_OP_LIST
#include "Toy/ToyOps.cpp.inc"
      >();

  // 注册自定义类型。没有这一步，写 !toy.num 会报 "unknown type"。
  addTypes<
#define GET_TYPEDEF_LIST
#include "Toy/ToyOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// 常量物化 (materializeConstant)
//===----------------------------------------------------------------------===//
//
// 当某个 op 的 fold() 返回一个属性值(Attribute)时，框架需要把这个纯粹的
// "值"重新变成 IR 中的一个真实操作，才能替换掉被折叠的 op。
// 对 toy dialect 而言，就是造出一个 toy.constant。
//
// 参数：
//   - value: fold() 算出的属性（这里应是一个 IntegerAttr）
//   - type : 结果类型（i32）
// 返回：一个新建的 toy.constant 操作；若无法物化则返回 nullptr。
//
mlir::Operation *ToyDialect::materializeConstant(mlir::OpBuilder &builder,
                                                 mlir::Attribute value,
                                                 mlir::Type type,
                                                 mlir::Location loc) {
  auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(value);
  if (!intAttr) {
    llvm::errs() << "[trace] dialect: ToyDialect::materializeConstant 收到非整数属性，无法物化\n";
    return nullptr;
  }

  llvm::errs() << "[trace] dialect: ToyDialect::materializeConstant 物化 toy.constant 值="
               << intAttr.getValue() << "\n";

  // 用 ODS 自动生成的 builder 造一个 toy.constant。
  return builder.create<ConstantOp>(loc, type, intAttr);
}
