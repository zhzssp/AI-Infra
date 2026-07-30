//===----------------------------------------------------------------------===//
// CountIR.cpp —— 分析型 Pass 实现
//   作用：遍历模块内每个"有函数体"的函数，统计
//         基本块数 / 指令数 / call 指令数，并打印到 stderr。
//   特点：只读不改，run() 结束返回 PreservedAnalyses::all()，
//         告诉 PassManager "我没动 IR，后续分析结果都还有效"。
//===----------------------------------------------------------------------===//
#include "Passes.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

PreservedAnalyses CountIRPass::run(Module &M, ModuleAnalysisManager & /*AM*/) {
  errs() << "[trace] LLVM Pass: CountIRPass::run 进入，开始遍历模块 "
         << M.getName() << "\n";
  errs() << "==================== [CountIR 分析型 Pass] ====================\n";
  errs() << "  模块: " << M.getName() << "\n";

  size_t totalBB = 0, totalInst = 0, totalCall = 0;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue; // 只声明没函数体（如 printf）的跳过

    size_t nBB = 0, nInst = 0, nCall = 0;
    for (BasicBlock &BB : F) {
      ++nBB;
      for (Instruction &I : BB) {
        ++nInst;
        if (isa<CallInst>(I))
          ++nCall;
      }
    }

    errs() << "  函数 @" << F.getName() << "  参数=" << F.arg_size()
           << "  基本块=" << nBB << "  指令=" << nInst << "  调用=" << nCall
           << "\n";

    totalBB += nBB;
    totalInst += nInst;
    totalCall += nCall;
  }

  errs() << "  ---- 合计: 基本块=" << totalBB << "  指令=" << totalInst
         << "  调用=" << totalCall << " ----\n";
  errs() << "===============================================================\n";

  // 分析型 Pass：没改 IR，保留所有已有分析结果。
  errs() << "[trace] LLVM Pass: CountIRPass::run 退出\n";
  return PreservedAnalyses::all();
}
