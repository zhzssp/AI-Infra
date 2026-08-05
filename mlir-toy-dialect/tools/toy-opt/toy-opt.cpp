//===- toy-opt.cpp - Toy dialect 的命令行驱动 -----------------------------===//
//
// 这是一个 mlir-opt 风格的工具。它能：
//   1. 读入一个 .mlir 文件；
//   2. 解析其中的 toy 操作（以及内置 dialect 的操作）；
//   3. 按命令行指定的 Pass（如 --canonicalize）做变换；
//   4. 把结果打印出来。
//
// 用法示例：
//   toy-opt input.mlir                 # 仅解析并回显
//   toy-opt input.mlir --canonicalize  # 跑规范化（含常量折叠）
//
//===----------------------------------------------------------------------===//

#include "Toy/ToyDialect.h"
#include "Toy/ToyPasses.h"
#include "Low/LowDialect.h"
#include "Low/LowPasses.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  // 注册所有内置的 Pass（包括 --canonicalize、--cse 等）。
  mlir::registerAllPasses();

  // 注册本项目自定义的 Pass。
  //   toy 层：--toy-simplify
  //   low 层：--toy-to-low（降低） / --low-strength-reduce（强度削减）
  mlir::toy::registerToyPasses();
  mlir::low::registerLowPasses();

  // 建立一个 dialect 注册表：注册 MLIR 内置 dialect，再加上我们自己的
  // 两个 dialect —— 高层 toy 与低层 low。
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::toy::ToyDialect>();
  registry.insert<mlir::low::LowDialect>();

  // 复用 MLIR 提供的 opt 主流程。它会自动处理命令行参数、文件读写、Pass 调度。

  // === 演示用：打印 pipeline 启动信息（仅用于观察，不影响编译功能）===
  llvm::errs() << "############################################################\n"
               << "# [Pipeline] toy-opt 启动\n"
               << "#   阶段 A 构建: scripts/build.sh   ->  产出 build/bin/toy-opt\n"
               << "#   阶段 B 输入: 读取 .mlir，解析为统一 MLIR IR（Operation 树）\n"
               << "#   阶段 C 处理: toy / low 两个 dialect 在同一体系下逐层降低+优化\n"
               << "#   阶段 D 输出: 最终 IR 打印到 stdout（各 Pass 前后另有 IR 快照）\n"
               << "#   已注册 dialect: toy(高层，含自定义类型 !toy.num) / low(低层)\n"
               << "#   已注册 Pass: --toy-simplify        代数化简（RewritePattern）\n"
               << "#                --toy-to-low         降低（贪心驱动器版）\n"
               << "#                --toy-to-low-convert 降低（Dialect Conversion 版）\n"
               << "#                --low-strength-reduce 强度削减 x*2^k -> x<<k\n"
               << "#                --toy-print-cost     基于接口的代价统计（跨 dialect）\n"
               << "############################################################\n";

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Toy optimizer driver\n", registry));
}
