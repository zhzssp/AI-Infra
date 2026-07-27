//===- ToyDialect.h - Toy dialect 的 C++ 声明 -----------------------------===//
//
// 声明 Toy dialect。真正的类定义由 mlir-tblgen 从 ToyDialect.td 生成，
// 生成结果通过下面的 .h.inc 引入。
//
//===----------------------------------------------------------------------===//

#ifndef TOY_TOYDIALECT_H
#define TOY_TOYDIALECT_H

#include "mlir/IR/Dialect.h"

// 引入 TableGen 生成的 Dialect 声明。
// 注意：add_mlir_dialect(ToyOps toy) 生成的文件名是 ToyOpsDialect.h.inc
//（在 op 定义文件名 ToyOps 后追加 "Dialect"）。构建时生成在 build 目录下。
#include "Toy/ToyOpsDialect.h.inc"

#endif // TOY_TOYDIALECT_H
