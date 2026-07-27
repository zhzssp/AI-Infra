//===- ToyOps.h - Toy dialect 操作的 C++ 声明 -----------------------------===//
//
// 声明 toy dialect 的所有操作。类定义由 mlir-tblgen 从 ToyOps.td 生成。
//
//===----------------------------------------------------------------------===//

#ifndef TOY_TOYOPS_H
#define TOY_TOYOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "Toy/ToyDialect.h"

// 引入 TableGen 生成的操作声明（ToyOps.h.inc）。
// GET_OP_CLASSES 宏控制生成的是"声明"部分。
#define GET_OP_CLASSES
#include "Toy/ToyOps.h.inc"

#endif // TOY_TOYOPS_H
