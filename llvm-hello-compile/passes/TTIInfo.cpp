//===----------------------------------------------------------------------===//
// TTIInfo.cpp —— Function 级分析型 Pass，演示 §4.6 的 TargetTransformInfo
//   TTI 是【中端唯一被允许知道的目标信息接口】：向量宽度多少、某条指令多贵。
//   向量化 / 展开 / 内联全靠它做决策；接新硬件时它就是"告诉 LLVM 硬件长什么样"的入口。
//
//   跑：opt -load-pass-plugin=MyPasses.so -passes=tti-info -disable-output x.ll
//   提示：IR 里的 target triple / "target-cpu" / "target-features" 属性会直接影响下面的答案，
//         所以用 clang -mavx2 编出来的 IR 和默认 IR 打印出的数字不一样 —— 这正是重点。
//===----------------------------------------------------------------------===//
#include "Passes.h"

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

PreservedAnalyses TTIInfoPass::run(Function &F, FunctionAnalysisManager &AM) {
  if (F.isDeclaration())
    return PreservedAnalyses::all();

  const TargetTransformInfo &TTI = AM.getResult<TargetIRAnalysis>(F);
  LLVMContext &Ctx = F.getContext();
  Type *F32 = Type::getFloatTy(Ctx);

  errs() << "==================== [tti-info] @" << F.getName()
         << " ====================\n";
  errs() << "  target triple : " << F.getParent()->getTargetTriple() << "\n";
  errs() << "  标量寄存器位宽 : "
         << TTI.getRegisterBitWidth(TargetTransformInfo::RGK_Scalar)
                .getFixedValue()
         << " bit\n";
  errs() << "  向量寄存器位宽 : "
         << TTI.getRegisterBitWidth(TargetTransformInfo::RGK_FixedWidthVector)
                .getFixedValue()
         << " bit   ← 向量化器据此决定一次处理几个元素\n";

  // 同一条 fmul，标量 / 4 宽 / 8 宽 的代价对比：这就是"代价模型"最直观的样子。
  for (unsigned N : {1u, 4u, 8u}) {
    Type *Ty = (N == 1) ? F32 : (Type *)FixedVectorType::get(F32, N);
    InstructionCost C = TTI.getArithmeticInstrCost(Instruction::FMul, Ty);
    errs() << "  fmul  ";
    if (N == 1)
      errs() << "float        ";
    else
      errs() << "<" << N << " x float> ";
    errs() << " 代价 = " << C << "\n";
  }

  errs() << "  [读法] 若 <8 x float> 的代价 ≈ <4 x float>，说明目标有 256bit 向量单元；\n";
  errs() << "         若代价成倍增长，说明得靠拆分模拟 —— 向量化器就会选更窄的宽度。\n";

  return PreservedAnalyses::all();
}
