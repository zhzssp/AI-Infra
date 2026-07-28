//===- ToyPasses.h - Toy dialect 的 Pass 声明 -----------------------------===//
//
// 声明 toy dialect 自带的 Pass。
//
// 学习要点：
//   本项目原本只演示了 fold()（配合内置的 --canonicalize）。但 MLIR 里
//   改写 IR 有【两条主线】：
//     1) fold()          —— 就地把一个 op "算成"一个常量值（窄而快）。
//     2) RewritePattern   —— 结构化地"匹配一个 IR 子图 → 重写成另一个子图"
//                            （通用的 DAG-to-DAG 改写，是 MLIR 优化的骨架）。
//   下面这个 Pass 就是用来演示第 2 条主线：如何【自己写一个 Pass】，
//   并在 Pass 里用 RewritePattern + 贪心驱动器改写 IR。
//
//===----------------------------------------------------------------------===//

#ifndef TOY_TOYPASSES_H
#define TOY_TOYPASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace toy {

// 创建"代数化简" Pass。
// 它用 RewritePattern 实现两条恒等式化简：
//   x * 1 = x     （乘以 1 直接去掉乘法）
//   x + 0 = x     （加 0 直接去掉加法）
// 注意：这些化简 fold() 也能做，但这里刻意用 RewritePattern 来演示
//       "匹配-重写"这条通用机制，方便对照理解两者差异。
std::unique_ptr<mlir::Pass> createToySimplifyPass();

// 把本 dialect 的所有 Pass 注册到全局 Pass 列表，
// 这样 toy-opt 命令行才能识别 --toy-simplify。
void registerToyPasses();

} // namespace toy
} // namespace mlir

#endif // TOY_TOYPASSES_H
