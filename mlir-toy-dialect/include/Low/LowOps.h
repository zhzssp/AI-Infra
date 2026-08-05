//===- LowOps.h - Low dialect 操作的 C++ 声明 -----------------------------===//
//
// 声明 low dialect 的所有操作。类定义由 mlir-tblgen 从 LowOps.td 生成。
//
//===----------------------------------------------------------------------===//

#ifndef LOW_LOWOPS_H
#define LOW_LOWOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "Low/LowDialect.h"
// low 层的 op 也实现了 toy 层定义的代价接口，所以要包含它的声明。
#include "Toy/ToyInterfaces.h"

#define GET_OP_CLASSES
#include "Low/LowOps.h.inc"

#endif // LOW_LOWOPS_H
