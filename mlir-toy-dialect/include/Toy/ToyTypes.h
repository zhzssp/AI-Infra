//===- ToyTypes.h - Toy dialect 自定义类型的 C++ 声明 ---------------------===//
//
// 引入 mlir-tblgen 从 ToyTypes.td 生成的类型类声明（mlir::toy::NumType）。
//
// 用法与内置类型完全一致：
//   Type t = toy::NumType::get(context);   // 构造
//   if (llvm::isa<toy::NumType>(t)) ...    // 判别
//
//===----------------------------------------------------------------------===//

#ifndef TOY_TOYTYPES_H
#define TOY_TOYTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"

// 生成的文件名以【输入的 td 文件名】ToyOps.td 为前缀，
// 而 ToyOps.td 里 include 了 ToyTypes.td，所以 NumType 的声明落在这里。
#define GET_TYPEDEF_CLASSES
#include "Toy/ToyOpsTypes.h.inc"

#endif // TOY_TOYTYPES_H
