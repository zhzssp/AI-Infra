//===- ToyPasses.cpp - Toy dialect 的 Pass 实现 ---------------------------===//
//
// 本文件手写（不经 TableGen）实现一个最小的 Pass，用来完整演示 MLIR 的
// 【Pass + RewritePattern + 贪心改写驱动器】这条核心链路：
//
//   ┌───────────────┐   add 到   ┌──────────────────┐  交给   ┌─────────────────────────┐
//   │ RewritePattern │ ────────▶ │ RewritePatternSet │ ──────▶ │ applyPatternsAndFold-   │
//   │ (匹配-重写规则) │            │ (规则集合)         │         │ Greedily (贪心驱动器)    │
//   └───────────────┘            └──────────────────┘         └─────────────────────────┘
//                                                                        │ 反复遍历 IR，
//                                                                        ▼ 命中就重写，直到不动点
//                                                                   改写后的 IR
//
// 与 fold() 的对照：
//   - fold()          在 ToyOps.cpp，做「常量+常量→常量」这类"算值"化简。
//   - RewritePattern  在这里，做「x*1→x」「x+0→x」这类"换结构"化简。
//   两者都是把旧 op 换掉，但 RewritePattern 是通用的子图重写框架，
//   是写 Pass、做 lowering / conversion 时最常用的武器。
//
//===----------------------------------------------------------------------===//

#include "Toy/ToyPasses.h"
#include "Toy/ToyDialect.h"
#include "Toy/ToyOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::toy;

namespace {

//===----------------------------------------------------------------------===//
// 小工具：判断某个 SSA 值是否是"值等于 target 的 toy.constant"
//===----------------------------------------------------------------------===//
//
// v.getDefiningOp<ConstantOp>() 会向上追溯：如果 v 是由一个 toy.constant
// 产生的，就拿到那个 ConstantOp；否则返回空。
// ConstantOp::getValue() 由 ODS 生成，返回 uint32_t（因为属性是 I32Attr）。
//
static bool isConstantWithValue(Value v, uint32_t target) {
  bool hit = false;
  if (auto c = v.getDefiningOp<ConstantOp>())
    hit = (c.getValue() == target);
  llvm::errs() << "[trace] match: isConstantWithValue 目标=" << target
               << " 命中=" << (hit ? "yes" : "no") << "\n";
  return hit;
}

//===----------------------------------------------------------------------===//
// 规则 1：x * 1 = x
//===----------------------------------------------------------------------===//
//
// OpRewritePattern<MulOp> 表示"我只匹配 toy.mul 这种 op"。
// matchAndRewrite 返回：
//   - success()：我匹配上并且改写了 IR；
//   - failure()：这次不匹配，交给别的规则。
//
struct SimplifyMulByOne : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    llvm::errs() << "[trace] match: SimplifyMulByOne 进入，检查 " << op << "\n";
    // 因为 mul 带 Commutative，1 可能在左也可能在右，两边都查。
    if (isConstantWithValue(op.getRhs(), 1)) {
      // 用 op 的另一个操作数，整体替换掉这个 mul（连同它的所有使用者）。
      llvm::errs() << "  [rewrite] toy-simplify: toy.mul x*1 -> x (rhs=1)\n";
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    if (isConstantWithValue(op.getLhs(), 1)) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }
    llvm::errs() << "[trace] match: SimplifyMulByOne 无匹配，返回 failure\n";
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// 规则 2：x + 0 = x
//===----------------------------------------------------------------------===//
struct SimplifyAddZero : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    llvm::errs() << "[trace] match: SimplifyAddZero 进入，检查 " << op << "\n";
    if (isConstantWithValue(op.getRhs(), 0)) {
      llvm::errs() << "  [rewrite] toy-simplify: toy.add x+0 -> x (rhs=0)\n";
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    if (isConstantWithValue(op.getLhs(), 0)) {
      llvm::errs() << "  [rewrite] toy-simplify: toy.add x+0 -> x (lhs=0)\n";
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }
    llvm::errs() << "[trace] match: SimplifyAddZero 无匹配，返回 failure\n";
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pass 本体
//===----------------------------------------------------------------------===//
//
// PassWrapper 是"手写 Pass"的便捷基类：
//   模板参数 <本类, OperationPass<>> 表示这是一个作用在【任意顶层 op】
//   （这里就是整个 ModuleOp）上的 Pass。
//
struct ToySimplifyPass
    : public PassWrapper<ToySimplifyPass, OperationPass<>> {
  // 为本地定义的 Pass 生成 TypeID（PassWrapper 要求）。
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToySimplifyPass)

  // 命令行开关名：toy-opt --toy-simplify
  StringRef getArgument() const final { return "toy-simplify"; }
  StringRef getDescription() const final {
    return "Toy 代数化简：x*1=x, x+0=x（RewritePattern 演示）";
  }

  // runOnOperation() 是 Pass 的入口，框架对每个目标 op 调用一次。
  void runOnOperation() override {
    llvm::errs() << "\n========== [Pass] --toy-simplify (Toy 层代数化简) ==========\n";
    llvm::errs() << "--- 进入 Pass 前的 IR（输入快照）---\n";
    getOperation()->print(llvm::errs());

    // 1) 把规则装进规则集合。
    RewritePatternSet patterns(&getContext());
    patterns.add<SimplifyMulByOne, SimplifyAddZero>(&getContext());

    // 2) 交给贪心驱动器：它会反复遍历 IR、命中就重写，直到不再变化
    //    （到达不动点）。同时它会顺带做 fold + 死代码消除，
    //    所以被替换掉的 toy.constant 会被自动清理。
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();

    llvm::errs() << "--- 退出 Pass 后的 IR（输出快照）---\n";
    getOperation()->print(llvm::errs());
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// 对外接口
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::toy::createToySimplifyPass() {
  return std::make_unique<ToySimplifyPass>();
}

void mlir::toy::registerToyPasses() {
  // PassRegistration 把 Pass 登记到全局，使 MlirOptMain 能通过
  // getArgument() 的名字（toy-simplify）在命令行找到并运行它。
  PassRegistration<ToySimplifyPass>();

  // 基于接口的代价统计 Pass（--toy-print-cost，定义在 ToyCostPass.cpp）。
  registerToyPrintCostPass();
}
