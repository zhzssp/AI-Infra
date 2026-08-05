//===- LowOps.cpp - Low dialect 操作的实现 --------------------------------===//
//
// 引入 TableGen 生成的操作定义，并实现 low.constant 的 fold()。
//
// 注意：low.add / low.mul 这里【故意不做常量折叠】。
// 常量折叠是"高层数学语义"的活儿，本项目安排在 toy 层完成；
// low 层专注于"硬件指令代价"的优化（强度削减，见 LowPasses.cpp）。
// 这种刻意的分工，正是为了让你看清"不同层级各司其职"。
//
//===----------------------------------------------------------------------===//

#include "Low/LowOps.h"
#include "Low/LowDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::low;

#define GET_OP_CLASSES
#include "Low/LowOps.cpp.inc"

// 常量折叠成"它自己的值"，让常量能被其它改写识别到。
OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  llvm::errs() << "[trace] dialect: low::ConstantOp::fold 进入，返回常量值 "
               << getValueAttr() << "\n";
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// MulOp::getCost —— 实现 toy 层定义的 ToyCostOpInterface
//===----------------------------------------------------------------------===//
//
// low.add / low.shl 沿用接口默认实现（代价 1），只有 low.mul 覆盖成 5。
// 于是 "x*4 -> x<<2" 这个强度削减，在 --toy-print-cost 眼里就是
// 总代价从 5 降到 1。真实编译器的 cost model 就是这个套路的放大版。
//
unsigned MulOp::getCost() {
  return 5;
}
