//===----------------------------------------------------------------------===//
// StrengthReduce.cpp —— Function 级变换型 Pass（窥孔优化 / 强度削减）
//   规则：把 `mul %x, 2^k` 改写成 `shl %x, k`。
//   看点一：这就是 instcombine 那类"规范化 + 窥孔"pass 的最小可运行版本。
//   看点二：只改指令、不动控制流，所以返回 preserveSet<CFGAnalyses>()，
//           支配树 / LoopInfo 这些只依赖 CFG 的分析【不必重算】——
//           这是 §3.4 里最常用也最容易被忘的一种 PreservedAnalyses 写法。
//   看点三：原来的 mul 可能带 nsw/nuw，我们【故意不复制】这些标志。
//           保守 = 安全；随手复制标志正是最常见的 poison/UB 来源（§2.7）。
//
//   跑：opt -load-pass-plugin=MyPasses.so -passes=strength-reduce -S x.ll
//===----------------------------------------------------------------------===//
#include "Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

PreservedAnalyses StrengthReducePass::run(Function &F,
                                          FunctionAnalysisManager & /*AM*/) {
  unsigned Changed = 0;

  for (BasicBlock &BB : F) {
    // 会边遍历边删指令，所以要用 early_inc_range 保证迭代器安全
    for (Instruction &I : make_early_inc_range(BB)) {
      auto *BO = dyn_cast<BinaryOperator>(&I);
      if (!BO || BO->getOpcode() != Instruction::Mul)
        continue;

      auto *C = dyn_cast<ConstantInt>(BO->getOperand(1));
      if (!C || !C->getValue().isPowerOf2())
        continue;

      unsigned ShiftAmt = C->getValue().logBase2();
      IRBuilder<> B(BO);
      // 注意：这里没有把 BO 的 nsw/nuw 带过来（见文件头"看点三"）
      Value *Shl = B.CreateShl(BO->getOperand(0), ShiftAmt, BO->getName() + ".shl");

      errs() << "[strength-reduce] @" << F.getName() << ": mul x, "
             << C->getZExtValue() << "  →  shl x, " << ShiftAmt << "\n";

      BO->replaceAllUsesWith(Shl);
      BO->eraseFromParent();
      ++Changed;
    }
  }

  if (Changed == 0)
    return PreservedAnalyses::all();

  // 改了指令，但一条分支都没动 → 只依赖 CFG 的分析仍然有效。
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
