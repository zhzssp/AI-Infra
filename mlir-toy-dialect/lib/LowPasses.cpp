//===- LowPasses.cpp - Low 层相关 Pass 的实现 -----------------------------===//
//
// 本文件实现两个 Pass，合起来演示 MLIR 的核心思想【逐层降低 + 分层优化】：
//
//   ┌──────────── toy 层（高层，数学语义）────────────┐
//   │  toy.constant / toy.add / toy.mul               │
//   │  优化：x*1=x、x+0=x、常量折叠（见 ToyPasses.cpp）│
//   └──────────────────────┬──────────────────────────┘
//                          │  --toy-to-low （降低 / lowering）
//                          ▼
//   ┌──────────── low 层（低层，贴近硬件）────────────┐
//   │  low.constant / low.add / low.mul / low.shl     │
//   │  优化：强度削减 low.mul %x,2^k => low.shl %x,k    │
//   └─────────────────────────────────────────────────┘
//
// 关键对比：
//   "x * 4" 这个乘法，在 toy 层 --toy-simplify 里【纹丝不动】
//   （因为高层只懂 x*1=x 这种代数恒等式，且高层没有"移位"概念）；
//   一旦降低到 low 层，--low-strength-reduce 就能把它变成 "x << 2"。
//   这就是"不同层级看到不同信息、能做不同优化"的最直观例子。
//
//===----------------------------------------------------------------------===//

#include "Low/LowPasses.h"
#include "Low/LowDialect.h"
#include "Low/LowOps.h"
#include "Toy/ToyDialect.h"
#include "Toy/ToyOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// 第一部分：降低 Pass —— toy.* 改写为 low.*
//===----------------------------------------------------------------------===//
//
// 每条规则匹配一个 toy 操作，用 replaceOpWithNewOp 造出等价的 low 操作。
// 这一步不追求"更优"，只负责"换层"。降低是 MLIR 里最常见的动作 ——
// 一个真实编译器就是靠一层层降低，把高层抽象逐步翻译到底层硬件。

// toy.constant -> low.constant
struct LowerConstant : public OpRewritePattern<toy::ConstantOp> {
  using OpRewritePattern<toy::ConstantOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(toy::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<low::ConstantOp>(op, op.getType(),
                                                 op.getValueAttr());
    return success();
  }
};

// toy.add -> low.add
struct LowerAdd : public OpRewritePattern<toy::AddOp> {
  using OpRewritePattern<toy::AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(toy::AddOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<low::AddOp>(op, op.getType(), op.getLhs(),
                                            op.getRhs());
    return success();
  }
};

// toy.mul -> low.mul
struct LowerMul : public OpRewritePattern<toy::MulOp> {
  using OpRewritePattern<toy::MulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(toy::MulOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<low::MulOp>(op, op.getType(), op.getLhs(),
                                            op.getRhs());
    return success();
  }
};

struct ToyToLowPass
    : public PassWrapper<ToyToLowPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToLowPass)

  StringRef getArgument() const final { return "toy-to-low"; }
  StringRef getDescription() const final {
    return "降低：把高层 toy.* 操作改写成等价的低层 low.* 操作";
  }

  // 降低会产生 low.* 操作，必须声明"我会用到 low dialect"，
  // 否则框架不允许在改写中创建它的操作。
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<low::LowDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerConstant, LowerAdd, LowerMul>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// 第二部分：强度削减 Pass —— low.mul %x, 2^k => low.shl %x, k
//===----------------------------------------------------------------------===//
//
// 这是一个【只有低层才做得了】的优化：它把"乘以 2 的幂"换成"左移"，
// 因为在真实硬件上，移位指令比乘法指令便宜得多。
// 高层 toy 里没有 shl 这个概念，所以这个优化在高层根本无从谈起。

// 若 v 是一个值为 2 的幂的 low.constant，返回 true，并把 log2(值) 写入 shift。
static bool isPowerOfTwoConst(Value v, unsigned &shift) {
  if (auto c = v.getDefiningOp<low::ConstantOp>()) {
    uint32_t val = c.getValue();
    if (val != 0 && (val & (val - 1)) == 0) { // 判断 2 的幂：只有一个比特为 1
      shift = llvm::Log2_32(val);
      return true;
    }
  }
  return false;
}

struct MulToShift : public OpRewritePattern<low::MulOp> {
  using OpRewritePattern<low::MulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(low::MulOp op,
                                PatternRewriter &rewriter) const override {
    unsigned shift = 0;
    // 因为 mul 是 Commutative，2 的幂可能在左也可能在右，两边都试。
    if (isPowerOfTwoConst(op.getRhs(), shift)) {
      rewriter.replaceOpWithNewOp<low::ShlOp>(
          op, op.getType(), op.getLhs(), rewriter.getI32IntegerAttr(shift));
      return success();
    }
    if (isPowerOfTwoConst(op.getLhs(), shift)) {
      rewriter.replaceOpWithNewOp<low::ShlOp>(
          op, op.getType(), op.getRhs(), rewriter.getI32IntegerAttr(shift));
      return success();
    }
    return failure(); // 非 2 的幂（如 x*3）：保持 low.mul 不动。
  }
};

struct LowStrengthReducePass
    : public PassWrapper<LowStrengthReducePass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowStrengthReducePass)

  StringRef getArgument() const final { return "low-strength-reduce"; }
  StringRef getDescription() const final {
    return "低层强度削减：low.mul %x, 2^k 改写为 low.shl %x, k";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MulToShift>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// 对外接口
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::low::createToyToLowPass() {
  return std::make_unique<ToyToLowPass>();
}

std::unique_ptr<Pass> mlir::low::createLowStrengthReducePass() {
  return std::make_unique<LowStrengthReducePass>();
}

void mlir::low::registerLowPasses() {
  PassRegistration<ToyToLowPass>();
  PassRegistration<LowStrengthReducePass>();
}
