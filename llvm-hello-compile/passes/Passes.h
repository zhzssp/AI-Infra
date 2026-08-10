//===----------------------------------------------------------------------===//
// Passes.h —— 自定义 Pass 声明（LLVM 17 New PassManager 风格）
//
// 五个 pass 刚好把 docs/llvm-learning-guide.md 第 3 章的要点占满：
//
//   pass              层级      类型    PreservedAnalyses 返回值
//   ---------------------------------------------------------------------
//   count-ir          Module    分析型  all()                （没动 IR）
//   cfg-info          Function  分析型  all()                （消费 DT/LoopInfo）
//   tti-info          Function  分析型  all()                （消费 TTI 代价模型）
//   strength-reduce   Function  变换型  preserveSet<CFGAnalyses>()（只改指令不改 CFG）
//   inject-log        Module    变换型  none()               （保守全失效）
//
// 「Module / Function 两个层级」+「四种 PreservedAnalyses 写法」在这里全都能看到。
//===----------------------------------------------------------------------===//
#pragma once

#include "llvm/IR/PassManager.h"

// 分析型（Module 级）：遍历模块里的函数，统计规模并打印，不修改 IR。
struct CountIRPass : llvm::PassInfoMixin<CountIRPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};

// 分析型（Function 级）：向 FunctionAnalysisManager 索取 DominatorTree / LoopInfo，
// 打印基本块、terminator、支配关系与循环结构 —— 演示「pass 如何消费 analysis」。
struct CfgInfoPass : llvm::PassInfoMixin<CfgInfoPass> {
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);
};

// 分析型（Function 级）：向 TargetTransformInfo 询问目标的向量宽度与指令代价，
// 演示「中端唯一的目标信息入口」。
struct TTIInfoPass : llvm::PassInfoMixin<TTIInfoPass> {
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);
};

// 变换型（Function 级）：把 `mul x, 2^k` 改写成 `shl x, k`（经典窥孔/强度削减）。
// 只改指令、不动控制流，所以返回 preserveSet<CFGAnalyses>()。
struct StrengthReducePass : llvm::PassInfoMixin<StrengthReducePass> {
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);
};

// 变换型（Module 级）：在每个"有函数体"的函数入口插入 printf("[trace] enter <fn>\n")。
struct InjectLoggingPass : llvm::PassInfoMixin<InjectLoggingPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};
