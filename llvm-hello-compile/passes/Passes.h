//===----------------------------------------------------------------------===//
// Passes.h —— 两个自定义 Pass 的声明（LLVM 17 New PassManager 风格）
//   1) CountIRPass     : 分析型，只统计不改 IR（打印每个函数的基本块/指令/调用数）
//   2) InjectLoggingPass: 变换型，在每个函数入口插一句 printf，最终可执行文件能看到效果
//===----------------------------------------------------------------------===//
#pragma once

#include "llvm/IR/PassManager.h"

// 分析型 Pass：遍历模块里的函数，统计规模并打印，不修改 IR。
struct CountIRPass : llvm::PassInfoMixin<CountIRPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};

// 变换型 Pass：在每个"有函数体"的函数入口处插入 printf("[trace] enter <fn>\n")。
struct InjectLoggingPass : llvm::PassInfoMixin<InjectLoggingPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};
