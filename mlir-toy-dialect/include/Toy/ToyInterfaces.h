//===- ToyInterfaces.h - Toy Op 接口的 C++ 声明 ---------------------------===//
//
// 引入 mlir-tblgen 从 ToyInterfaces.td 生成的接口类声明。
// 生成结果是一个 C++ 类 mlir::toy::ToyCostOpInterface，用法与普通
// Op 类一样：dyn_cast<ToyCostOpInterface>(op) 成功即说明该 op 实现了接口。
//
//===----------------------------------------------------------------------===//

#ifndef TOY_TOYINTERFACES_H
#define TOY_TOYINTERFACES_H

#include "mlir/IR/OpDefinition.h"

#include "Toy/ToyInterfaces.h.inc"

#endif // TOY_TOYINTERFACES_H
