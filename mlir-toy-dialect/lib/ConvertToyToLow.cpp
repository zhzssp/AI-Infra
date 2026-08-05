//===- ConvertToyToLow.cpp - 用 DialectConversion 做降低 ------------------===//
//
// 本文件实现 --toy-to-low-convert：和 LowPasses.cpp 里的 --toy-to-low
// 完成【同一件事】（toy.* 变成 low.*），但用的是完全不同的框架。
// 两个 Pass 并排放在项目里，就是为了让你亲手对比这两条路线。
//
//   ┌───────────────────────┬──────────────────────────┬────────────────────────┐
//   │                       │ 贪心改写驱动器            │ Dialect Conversion      │
//   │                       │ (--toy-to-low)           │ (--toy-to-low-convert)  │
//   ├───────────────────────┼──────────────────────────┼────────────────────────┤
//   │ 入口函数               │ applyPatternsAndFold-    │ applyPartialConversion  │
//   │                       │ Greedily                 │ / applyFullConversion   │
//   │ "什么算做完了"          │ 没有规则能再命中           │ 所有【非法】op 都消失了   │
//   │ 漏写一条规则会怎样      │ 悄悄留下一个 toy.mul      │ 直接报错，指出哪个 op    │
//   │                       │ ——你可能压根没发现         │ 没能合法化               │
//   │ 能不能换类型            │ 不能（类型必须自己对齐）    │ 能，TypeConverter 专门做 │
//   │ 是否原子                │ 边改边生效                │ 全部成功才提交，失败回滚  │
//   └───────────────────────┴──────────────────────────┴────────────────────────┘
//
// 结论：【真实编译器的 lowering 一律用 Dialect Conversion】。
// 原因就是上表第 2、3 行——它把"我到底降完了没有"变成一个由框架强制检查的
// 契约（ConversionTarget 描述"什么是合法 IR"），而不是靠人肉自觉。
// MLIR 里 ConvertVectorToLLVM、ConvertGPUToNVVM、ConvertLinalgToLoops……
// 无一例外都是这个结构。
//
//===----------------------------------------------------------------------===//

#include "Low/LowDialect.h"
#include "Low/LowOps.h"
#include "Low/LowPasses.h"
#include "Toy/ToyDialect.h"
#include "Toy/ToyOps.h"
#include "Toy/ToyTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// 第一部分：TypeConverter —— 声明"高层类型如何翻译成低层类型"
//===----------------------------------------------------------------------===//
//
// 这是贪心驱动器【做不到】的事。降低往往伴随类型的改变：
//   本项目：  !toy.num          ->  i32
//   真实世界： tensor<4xf32>     ->  memref<4xf32>   （bufferization）
//             memref<...>       ->  !llvm.struct<...>（ConvertMemRefToLLVM）
//             index             ->  i64
// 类型一变，函数签名、块参数、所有用到它的地方都要跟着改。
// TypeConverter + ConversionPatternRewriter 会自动完成这套连锁替换，
// 你只需要在这里声明"映射规则"。
//
static void populateToyTypeConversions(TypeConverter &converter) {
  // 规则 1（兜底）：其它类型原样保留。顺序上写在前面的是"最后兜底"的，
  // 因为 TypeConverter 是【倒序】尝试各条规则的。
  converter.addConversion([](Type type) { return type; });

  // 规则 2：!toy.num -> i32
  converter.addConversion([](toy::NumType type) -> Type {
    llvm::errs() << "[trace] convert: TypeConverter !toy.num -> i32\n";
    return IntegerType::get(type.getContext(), 32);
  });

  // 万一某个值的类型被转换了、但它的使用者还没被改写（跨越"已转换/未转换"
  // 的边界），框架需要临时插一个"转接头"把两边接上。
  // builtin 的 unrealized_conversion_cast 就是这个官方转接头：
  //   %1 = builtin.unrealized_conversion_cast %0 : i32 to !toy.num
  // 如果转换最终全部成功，这些转接头会被自动消掉；如果还剩下，
  // 说明你的 lowering 没做干净——这是一条非常有用的错误信号。
  auto materialize = [](OpBuilder &builder, Type resultType, ValueRange inputs,
                        Location loc) -> std::optional<Value> {
    if (inputs.size() != 1)
      return std::nullopt;
    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  };
  converter.addSourceMaterialization(materialize);
  converter.addTargetMaterialization(materialize);
}

//===----------------------------------------------------------------------===//
// 第二部分：ConversionPattern —— 与 OpRewritePattern 的关键差别
//===----------------------------------------------------------------------===//
//
// OpConversionPattern 的 matchAndRewrite 比 OpRewritePattern 多一个参数：
//
//   matchAndRewrite(Op op, OpAdaptor adaptor, ConversionPatternRewriter &r)
//                   ~~~~~  ~~~~~~~~~~~~~~~~
//                    ↑          ↑
//         【旧】的 op，        【新】的操作数——已经被前面的模式改写过、
//         类型还是老的          类型已经转换过的那一批 Value
//
// 记住一句话就够了：**读属性/类型用 op，读操作数一律用 adaptor**。
// 用 op.getLhs() 会拿到还没转换的旧值，是初学者最常见的 bug。

// toy.constant -> low.constant
struct ConstantOpLowering : public OpConversionPattern<toy::ConstantOp> {
  using OpConversionPattern<toy::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "  [convert] toy.constant -> low.constant\n";
    // 常量没有操作数，只有属性，所以这里直接从 op 上取属性。
    rewriter.replaceOpWithNewOp<low::ConstantOp>(op, op.getType(),
                                                 op.getValueAttr());
    return success();
  }
};

// toy.add -> low.add
struct AddOpLowering : public OpConversionPattern<toy::AddOp> {
  using OpConversionPattern<toy::AddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "  [convert] toy.add -> low.add\n";
    rewriter.replaceOpWithNewOp<low::AddOp>(op, op.getType(), adaptor.getLhs(),
                                            adaptor.getRhs());
    return success();
  }
};

// toy.mul -> low.mul
struct MulOpLowering : public OpConversionPattern<toy::MulOp> {
  using OpConversionPattern<toy::MulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "  [convert] toy.mul -> low.mul\n";
    rewriter.replaceOpWithNewOp<low::MulOp>(op, op.getType(), adaptor.getLhs(),
                                            adaptor.getRhs());
    return success();
  }
};

// toy.box / toy.unbox 直接消失。
//
// 这两条规则是"类型降低"的典型形态：装箱/拆箱只在高层有意义
// （!toy.num 承载高层语义），一旦 !toy.num 被 TypeConverter 换成 i32，
// 装箱拆箱就退化成恒等操作，直接用转换后的操作数替换掉自己即可。
// 对照真实世界：bufferization 里 to_memref / to_tensor 的消解就是这个套路。
struct BoxOpLowering : public OpConversionPattern<toy::BoxOp> {
  using OpConversionPattern<toy::BoxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::BoxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "  [convert] toy.box 消解（!toy.num 已被换成 i32）\n";
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

struct UnboxOpLowering : public OpConversionPattern<toy::UnboxOp> {
  using OpConversionPattern<toy::UnboxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::UnboxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "  [convert] toy.unbox 消解（!toy.num 已被换成 i32）\n";
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// 第三部分：Pass 本体 —— ConversionTarget 是这里的主角
//===----------------------------------------------------------------------===//

struct ConvertToyToLowPass
    : public PassWrapper<ConvertToyToLowPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertToyToLowPass)

  StringRef getArgument() const final { return "toy-to-low-convert"; }
  StringRef getDescription() const final {
    return "降低（Dialect Conversion 版）：ConversionTarget + TypeConverter";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<low::LowDialect>();
  }

  void runOnOperation() override {
    llvm::errs() << "\n========== [Pass] --toy-to-low-convert (Dialect Conversion 版降低) ==========\n";
    llvm::errs() << "--- 进入 Pass 前的 IR（高层快照）---\n";
    getOperation()->print(llvm::errs());

    //-- ① 描述"转换结束后，什么样的 IR 才算合法" --------------------------
    //
    // 这是 Dialect Conversion 与贪心改写最大的观念差别：
    // 你不是在描述"做哪些改写"，而是在描述"终点长什么样"，
    // 由框架去检查你有没有真的走到终点。
    ConversionTarget target(getContext());

    // low.* 是我们的目标形态，合法。
    target.addLegalDialect<low::LowDialect>();
    // toy.* 一律非法：转换结束后一个都不许剩。
    target.addIllegalDialect<toy::ToyDialect>();
    // 但 toy.repeat / toy.yield 这次不在降低范围内，单独放行。
    // 注意：op 级别的设置会覆盖 dialect 级别的设置，顺序无关。
    target.addLegalOp<toy::RepeatOp, toy::YieldOp>();
    //
    // 没有被显式标记的 op（比如 func.func / func.return）在
    // applyPartialConversion 下【视为允许保留】。
    // 如果换成 applyFullConversion，那就要求全 IR 都显式合法——
    // 那时你必须把 func 的签名转换等一并做掉，代价高得多。

    //-- ② 类型如何翻译 ----------------------------------------------------
    TypeConverter typeConverter;
    populateToyTypeConversions(typeConverter);

    //-- ③ 装配改写规则 ----------------------------------------------------
    // 注意 add 的第一个参数是 typeConverter：ConversionPattern 需要它
    // 才知道该把操作数/结果的类型换成什么。
    RewritePatternSet patterns(&getContext());
    patterns.add<ConstantOpLowering, AddOpLowering, MulOpLowering,
                 BoxOpLowering, UnboxOpLowering>(typeConverter, &getContext());

    //-- ④ 执行 ------------------------------------------------------------
    // 试试看：把上面 patterns.add 里的 MulOpLowering 删掉再跑，
    // 你会看到一条明确的报错
    //   error: failed to legalize operation 'toy.mul'
    // 而贪心版的 --toy-to-low 在同样情况下会一声不吭地留下 toy.mul。
    // 这条差别，就是真实编译器坚持用 Dialect Conversion 的全部理由。
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      llvm::errs() << "[convert] 合法化失败：还有 toy.* 操作没能被降低\n";
      signalPassFailure();
      return;
    }

    llvm::errs() << "--- 退出 Pass 后的 IR（低层快照）---\n";
    getOperation()->print(llvm::errs());
  }
};

} // namespace

std::unique_ptr<Pass> mlir::low::createConvertToyToLowPass() {
  return std::make_unique<ConvertToyToLowPass>();
}

void mlir::low::registerConvertToyToLowPass() {
  PassRegistration<ConvertToyToLowPass>();
}
