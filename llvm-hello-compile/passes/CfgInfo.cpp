//===----------------------------------------------------------------------===//
// CfgInfo.cpp —— Function 级分析型 Pass
//   目的：演示 §3.4「pass 如何向 AnalysisManager 索取分析结果」。
//   打印：基本块 / terminator 种类 / 支配树根 / 循环嵌套深度。
//   返回：PreservedAnalyses::all() —— 只读，不让任何分析失效。
//
//   跑：opt -load-pass-plugin=MyPasses.so -passes=cfg-info -disable-output x.ll
//===----------------------------------------------------------------------===//
#include "Passes.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>

using namespace llvm;

PreservedAnalyses CfgInfoPass::run(Function &F, FunctionAnalysisManager &AM) {
  if (F.isDeclaration())
    return PreservedAnalyses::all();

  // ★ 关键一行：分析不是自己算的，是向 AnalysisManager 要的。
  //   命中缓存就直接返回，失效了框架才会重新 run()。
  DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
  LoopInfo &LI = AM.getResult<LoopAnalysis>(F);

  errs() << "==================== [cfg-info] @" << F.getName()
         << " ====================\n";

  unsigned nBB = 0;
  for (BasicBlock &BB : F) {
    ++nBB;
    // 每个基本块【必须且只能】以一条 terminator 结尾（§2.3）
    const Instruction *T = BB.getTerminator();
    errs() << "  block %" << BB.getName() << "  指令=" << BB.size()
           << "  前驱=" << pred_size(&BB)
           << "  terminator=" << (T ? T->getOpcodeName() : "<none>")
           << "  循环深度=" << LI.getLoopDepth(&BB) << "\n";
  }

  errs() << "  基本块总数=" << nBB
         << "  支配树根=%" << DT.getRootNode()->getBlock()->getName()
         << "  顶层自然循环数=" << std::distance(LI.begin(), LI.end()) << "\n";

  // 只读 pass：告诉框架「所有已有分析继续有效」。
  return PreservedAnalyses::all();
}
