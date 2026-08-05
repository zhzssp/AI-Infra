//===- ToyCostPass.cpp - 一个"不认识任何 dialect"的通用 Pass --------------===//
//
// 本文件实现 --toy-print-cost。它是全项目最重要的一个"观念演示"：
//
//   这个 Pass 里【没有出现任何一个具体的 op 类型】。
//   没有 toy::AddOp，没有 low::MulOp，没有一长串 isa<> 判断。
//   它只问一句话："你实现 ToyCostOpInterface 了吗？实现了就报个价。"
//
// 于是同一个 Pass 同时适用于 toy 层和 low 层；将来你再加第三个、第四个
// dialect，只要新 op 实现了这个接口，这个 Pass 一行都不用改。
//
// 这就是 MLIR 生态能容纳上百个 dialect 却仍然共用一套优化 Pass 的原因：
// 通用 Pass 依赖【接口】，而不是依赖【具体 dialect】。
// 真实世界里的对应物：LoopLikeOpInterface（所有"像循环"的 op）、
// MemoryEffectOpInterface（所有"会读写内存"的 op）、
// DestinationStyleOpInterface（linalg 全家）……
//
// 这个 Pass 做两件事：
//   1) 给每个实现了接口的 op 打上一个 `toy.cost = N : i32` 属性，
//      这样代价会直接出现在输出的 IR 文本里（顺便演示"属性可以随手挂"）；
//   2) 在 stderr 上打印总代价。
//
// 典型用法（观察优化前后的代价差）：
//   toy-opt test/strength.mlir --toy-to-low --toy-print-cost
//   toy-opt test/strength.mlir --toy-to-low --low-strength-reduce --toy-print-cost
//
//===----------------------------------------------------------------------===//

#include "Toy/ToyInterfaces.h"
#include "Toy/ToyPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct ToyPrintCostPass
    : public PassWrapper<ToyPrintCostPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyPrintCostPass)

  StringRef getArgument() const final { return "toy-print-cost"; }
  StringRef getDescription() const final {
    return "按 ToyCostOpInterface 统计并标注每个操作的执行代价（跨 dialect 通用）";
  }

  void runOnOperation() override {
    llvm::errs() << "\n========== [Pass] --toy-print-cost (基于接口的代价统计) ==========\n";

    unsigned total = 0;
    OpBuilder builder(&getContext());

    // walk() 会【递归遍历】整棵 IR 树：模块 -> 函数 -> 区域 -> 块 -> 操作，
    // 连 toy.repeat 区域里嵌套的操作也一个不落。
    getOperation()->walk([&](Operation *op) {
      // 关键的一行：把 Operation* 转成接口。
      // 转换成功 == "这个 op 实现了 ToyCostOpInterface"，
      // 至于它是 toy.add 还是 low.shl，本 Pass 一点都不关心。
      auto costOp = llvm::dyn_cast<toy::ToyCostOpInterface>(op);
      if (!costOp) {
        llvm::errs() << "[trace] cost: " << op->getName()
                     << " 未实现 ToyCostOpInterface，跳过\n";
        return;
      }

      unsigned cost = costOp.getCost();
      total += cost;

      // 把代价作为一个"可丢弃属性(discardable attribute)"挂到 op 上。
      // 它会直接出现在打印的 IR 里：toy.add %a, %b {toy.cost = 1 : i32} : i32
      // 这类属性不参与 op 的语义，任何 Pass 都可以随手加/删。
      op->setAttr("toy.cost", builder.getI32IntegerAttr(cost));

      llvm::errs() << "[trace] cost: " << op->getName() << " => " << cost << "\n";
    });

    llvm::errs() << "[cost] 本模块总代价 = " << total << "\n";
  }
};

} // namespace

std::unique_ptr<Pass> mlir::toy::createToyPrintCostPass() {
  return std::make_unique<ToyPrintCostPass>();
}

void mlir::toy::registerToyPrintCostPass() {
  PassRegistration<ToyPrintCostPass>();
}
