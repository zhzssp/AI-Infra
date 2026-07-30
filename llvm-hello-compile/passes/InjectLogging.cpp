//===----------------------------------------------------------------------===//
// InjectLogging.cpp —— 变换型 Pass 实现
//   作用：在每个"有函数体"的函数入口最前面，插入一句
//         printf("[trace] enter <函数名>\n");
//   效果：把这个 pass 处理过的 IR 编译成可执行文件后，
//         每进入一个函数就会打印一行 trace —— 肉眼可见 pass 真的改了程序。
//   要点：
//     - 用 getOrInsertFunction 拿到（或声明）printf 的原型 i32(ptr, ...)
//     - 用 IRBuilder 在 entry 块的"第一个可插入点"构造字符串常量并发起 call
//     - 改了 IR，返回 PreservedAnalyses::none()（保守起见让分析全部失效重算）
//===----------------------------------------------------------------------===//
#include "Passes.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

PreservedAnalyses InjectLoggingPass::run(Module &M,
                                         ModuleAnalysisManager & /*AM*/) {
  errs() << "[trace] LLVM Pass: InjectLoggingPass::run 进入，准备为各函数插桩\n";
  LLVMContext &Ctx = M.getContext();

  // printf 原型：i32 printf(ptr, ...)  —— LLVM 17 用不透明指针 ptr
  FunctionType *PrintfTy =
      FunctionType::get(Type::getInt32Ty(Ctx),
                        {PointerType::getUnqual(Ctx)}, /*isVarArg=*/true);
  FunctionCallee Printf = M.getOrInsertFunction("printf", PrintfTy);

  int injected = 0;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;              // 没函数体的跳过
    if (F.getName() == "printf")
      continue;              // 别给 printf 自己插桩

    // 在入口基本块的第一个"可插入点"处准备构造指令
    errs() << "[trace] LLVM Pass: 处理函数 @" << F.getName() << "\n";
    BasicBlock &Entry = F.getEntryBlock();
    IRBuilder<> Builder(&*Entry.getFirstInsertionPt());

    // 造一个全局字符串常量 "[trace] enter <fn>\n"，返回指向它的 ptr
    std::string Msg = "[trace] enter " + F.getName().str() + "\n";
    Value *Fmt = Builder.CreateGlobalStringPtr(Msg, "trace_fmt");

    // 发起 printf 调用
    Builder.CreateCall(Printf, {Fmt});
    ++injected;
  }

  errs() << "==================== [InjectLogging 变换型 Pass] ==============\n";
  errs() << "  已在 " << injected << " 个函数入口注入 printf 追踪语句。\n";
  errs() << "===============================================================\n";

  // 改动了 IR：保守地让所有分析失效（教学场景足够）
  errs() << "[trace] LLVM Pass: InjectLoggingPass::run 退出，共注入 " << injected
         << " 处\n";
  return injected ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
