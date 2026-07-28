//===- LowDialect.cpp - Low dialect 的实现 --------------------------------===//
//
// 实现 Low dialect 的初始化：把所有 low.* 操作注册进 dialect，
// 并实现 low.constant 的 fold()。
//
//===----------------------------------------------------------------------===//

#include "Low/LowDialect.h"
#include "Low/LowOps.h"

using namespace mlir;
using namespace mlir::low;

// 引入 TableGen 生成的 Dialect 定义（构造/析构等样板）。
#include "Low/LowOpsDialect.cpp.inc"

// initialize() 在 dialect 加载时被调用，注册所有操作，
// 这样解析器才认识 low.constant / low.add / low.mul / low.shl。
void LowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Low/LowOps.cpp.inc"
      >();
}
