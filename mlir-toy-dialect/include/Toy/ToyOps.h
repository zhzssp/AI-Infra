//===- ToyOps.h - Toy dialect 操作的 C++ 声明 -----------------------------===//
//
// 声明 toy dialect 的所有操作。类定义由 mlir-tblgen 从 ToyOps.td 生成。
//
// 注意包含顺序：生成的 ToyOps.h.inc 会引用自定义类型 NumType 和接口
// ToyCostOpInterface，所以必须先包含它们的声明。
//
//===----------------------------------------------------------------------===//

#ifndef TOY_TOYOPS_H
#define TOY_TOYOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "Toy/ToyDialect.h"
#include "Toy/ToyInterfaces.h"
#include "Toy/ToyTypes.h"

// 引入 TableGen 生成的操作声明（ToyOps.h.inc）。
// GET_OP_CLASSES 宏控制生成的是"声明"部分。
#define GET_OP_CLASSES
#include "Toy/ToyOps.h.inc"

#endif // TOY_TOYOPS_H
