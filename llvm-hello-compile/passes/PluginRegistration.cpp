//===----------------------------------------------------------------------===//
// PluginRegistration.cpp —— 把自定义 Pass 注册进 opt 的插件入口
//   opt 加载 .so 后会调用 llvmGetPassPluginInfo()，我们在这里把管线名映射到 Pass。
//
//   注意有【两个】注册回调：
//     - ModulePassManager  版：count-ir / inject-log
//     - FunctionPassManager版：cfg-info / tti-info / strength-reduce
//   这正对应 §3.2 的层级：Module → CGSCC → Function → Loop。
//   注册到 Function 层后，opt 会自动帮你套上 module→function 的 adaptor，
//   所以 `-passes=cfg-info` 和 `-passes='function(cfg-info)'` 都能跑。
//
//   用法：opt -load-pass-plugin=MyPasses.so -passes=count-ir input.ll
//===----------------------------------------------------------------------===//
#include "Passes.h"

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

static PassPluginLibraryInfo getMyPassesPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "MyPasses", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            // ---- Module 级 ----
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "count-ir") {
                    MPM.addPass(CountIRPass());
                    return true;
                  }
                  if (Name == "inject-log") {
                    MPM.addPass(InjectLoggingPass());
                    return true;
                  }
                  return false;
                });

            // ---- Function 级 ----
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "cfg-info") {
                    FPM.addPass(CfgInfoPass());
                    return true;
                  }
                  if (Name == "tti-info") {
                    FPM.addPass(TTIInfoPass());
                    return true;
                  }
                  if (Name == "strength-reduce") {
                    FPM.addPass(StrengthReducePass());
                    return true;
                  }
                  return false;
                });
          }};
}

// opt / clang 通过这个 C 符号发现插件里的 Pass。
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getMyPassesPluginInfo();
}
