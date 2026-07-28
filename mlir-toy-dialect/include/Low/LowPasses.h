//===- LowPasses.h - Low 层相关 Pass 的声明 -------------------------------===//
//
// 这里声明两个 Pass，用来演示【层级分工】：
//
//   1) --toy-to-low          （降低 / lowering）
//      把高层 toy.* 操作逐个改写成语义等价的低层 low.* 操作。
//      这一步【不做优化】，只负责"换层"。
//
//   2) --low-strength-reduce （低层优化 / 强度削减）
//      在 low 层做一个高层做不到的优化：把 "low.mul %x, 2^k"
//      改写成 "low.shl %x, k"（移位比乘法便宜）。
//      注意：这个优化需要 shl 这个【只有 low 层才有】的概念。
//
//===----------------------------------------------------------------------===//

#ifndef LOW_LOWPASSES_H
#define LOW_LOWPASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace low {

// 降低：toy.* -> low.*
std::unique_ptr<mlir::Pass> createToyToLowPass();

// 低层优化：low.mul %x, 2^k -> low.shl %x, k
std::unique_ptr<mlir::Pass> createLowStrengthReducePass();

// 注册以上 Pass，使 toy-opt 能识别 --toy-to-low / --low-strength-reduce。
void registerLowPasses();

} // namespace low
} // namespace mlir

#endif // LOW_LOWPASSES_H
