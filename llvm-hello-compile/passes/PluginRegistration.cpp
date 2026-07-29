//===----------------------------------------------------------------------===//
// PluginRegistration.cpp —— 把两个 Pass 注册进 opt 的插件入口
//   opt 加载 .so 后会调用 llvmGetPassPluginInfo()，我们在这里把
//   管线名字 "count-ir" / "inject-log" 映射到对应的 Pass 对象。
//   用法：opt -load-pass-plugin=MyPasses.so -passes=count-ir  input.ll
//===----------------------------------------------------------------------===//
#include "Passes.h"

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

// 构造插件信息：声明支持哪些管线名，以及如何构造对应的 Pass。
static PassPluginLibraryInfo getMyPassesPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "MyPasses", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
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
          }};
}

// opt / clang 通过这个 C 符号发现插件里的 Pass。
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getMyPassesPluginInfo();
}
