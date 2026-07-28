//===- LowDialect.h - Low dialect 的 C++ 声明 -----------------------------===//
//
// 声明 Low dialect。真正的类定义由 mlir-tblgen 从 LowDialect.td 生成，
// 通过下面的 .h.inc 引入。
//
//===----------------------------------------------------------------------===//

#ifndef LOW_LOWDIALECT_H
#define LOW_LOWDIALECT_H

#include "mlir/IR/Dialect.h"

// add_mlir_dialect(LowOps low) 生成的 dialect 声明文件名是 LowOpsDialect.h.inc。
#include "Low/LowOpsDialect.h.inc"

#endif // LOW_LOWDIALECT_H
