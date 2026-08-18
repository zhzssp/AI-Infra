//===- ToyOps.cpp - Toy dialect 操作的实现 --------------------------------===//
//
// 实现 toy 操作中那些"非自动生成"的部分，主要是 fold() 方法。
// fold() 是理解 MLIR 优化机制最好的入口：
//   - 当一个 op 的输入是编译期已知的常量时，fold() 可以在编译期把它算出来，
//     用一个常量替换掉这个 op。这就是 "constant folding"。
//
//===----------------------------------------------------------------------===//

#include "Toy/ToyOps.h"
#include "Toy/ToyDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::toy;

// 引入 TableGen 生成的操作定义（ToyOps.cpp.inc），
// 包含各操作的解析、打印、验证、构造等样板实现。
#define GET_OP_CLASSES
#include "Toy/ToyOps.cpp.inc"

//===----------------------------------------------------------------------===//
// ConstantOp::fold
//===----------------------------------------------------------------------===//
//
// 一个常量 op 折叠成"它自己的值"。这让常量可以被其他 op 的 fold 识别到。
// 对于 ConstantLike 的 op，返回它的 value 属性即可。
//
OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  llvm::errs() << "[trace] dialect: ConstantOp::fold 进入，直接返回常量值 "
               << getValueAttr() << "\n";
  // 直接返回该常量携带的属性值。
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// AddOp::fold
//===----------------------------------------------------------------------===//
//
// 当 lhs 和 rhs 都是常量时，直接算出 lhs + rhs 并返回一个新的整数属性，
// MLIR 框架会自动用一个 toy.constant 替换掉这个 toy.add。
//
// adaptor 里保存了各操作数"如果是常量的话"对应的 Attribute；
// 若某个操作数不是常量，则对应项为 null。
//
OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto lhs = llvm::dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  auto rhs = llvm::dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());

  llvm::errs() << "[trace] dialect: AddOp::fold 进入，lhs=";
  if (lhs) llvm::errs() << lhs.getValue().getZExtValue();
  else llvm::errs() << "(非常量)";
  llvm::errs() << " rhs=";
  if (rhs) llvm::errs() << rhs.getValue().getZExtValue();
  else llvm::errs() << "(非常量)";
  llvm::errs() << "\n";

  // 两个操作数都必须是常量，否则无法折叠。
  if (!lhs || !rhs) {
    llvm::errs() << "[trace] dialect: AddOp::fold 操作数非全常量，跳过\n";
    return {};
  }

  // 计算 lhs + rhs，构造一个新的 IntegerAttr 作为折叠结果。
  APInt result = lhs.getValue() + rhs.getValue();
  llvm::errs() << "[trace] dialect: AddOp::fold 常量折叠结果="
               << result.getZExtValue() << "\n";
  return IntegerAttr::get(getType(), result);
}

//===----------------------------------------------------------------------===//
// MulOp::fold
//===----------------------------------------------------------------------===//
//
// 常量 * 常量 -> 常量。
// 练习：在这里加入 "x * 1 = x"（返回 getLhs()/getRhs()）
//       和 "x * 0 = 0"（返回一个 0 常量属性）等代数化简规则。
//
OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  auto lhs = llvm::dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  auto rhs = llvm::dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());

  llvm::errs() << "[trace] dialect: MulOp::fold 进入，lhs=";
  if (lhs) llvm::errs() << lhs.getValue().getZExtValue();
  else llvm::errs() << "(非常量)";
  llvm::errs() << " rhs=";
  if (rhs) llvm::errs() << rhs.getValue().getZExtValue();
  else llvm::errs() << "(非常量)";
  llvm::errs() << "\n";

  if (!lhs || !rhs) {
    llvm::errs() << "[trace] dialect: MulOp::fold 操作数非全常量，跳过\n";
    return {};
  }

  APInt result = lhs.getValue() * rhs.getValue();
  llvm::errs() << "[trace] dialect: MulOp::fold 常量折叠结果="
               << result.getZExtValue() << "\n";
  return IntegerAttr::get(getType(), result);
}

//===----------------------------------------------------------------------===//
// MulOp::getCost —— 实现 ToyCostOpInterface
//===----------------------------------------------------------------------===//
//
// toy.mul 在 .td 的 extraClassDeclaration 里声明了 getCost()，
// 所以必须在这里给出定义（覆盖接口默认的代价 1）。
// 对比：toy.add 直接挂接口名、没有 Declare...，于是沿用默认实现（返回 1），
//       一行 C++ 都不用写。
//
unsigned MulOp::getCost() {
  return 5; // 乘法比加法贵得多
}

//===----------------------------------------------------------------------===//
// UnboxOp::fold —— unbox(box(x)) = x
//===----------------------------------------------------------------------===//
//
// 这是 fold 的【第二种】用法，非常值得和上面的 add/mul 对照：
//   - add/mul 的 fold 返回一个 Attribute（"我算出了一个新的常量值"），
//     框架随后调用 dialect 的 materializeConstant() 把它变成 toy.constant；
//   - 这里的 fold 返回一个 Value（"别用我了，直接用这个已有的值"），
//     框架会把 toy.unbox 的所有使用者直接接到 box 的输入上。
// OpFoldResult 正是 "Attribute 或 Value" 的联合体，两种都合法。
//
// 效果：一对多余的装箱/拆箱在 --canonicalize 后完全消失，
//       连同中间的 !toy.num 值一起被清理掉。
//
OpFoldResult UnboxOp::fold(FoldAdaptor adaptor) {
  if (auto box = getInput().getDefiningOp<BoxOp>()) {
    llvm::errs() << "[trace] dialect: UnboxOp::fold 命中 unbox(box(x))，直接返回 x\n";
    return box.getInput();
  }
  llvm::errs() << "[trace] dialect: UnboxOp::fold 输入不是 toy.box，跳过\n";
  return {};
}

//===----------------------------------------------------------------------===//
// RepeatOp::verify —— 自定义验证器
//===----------------------------------------------------------------------===//
//
// hasVerifier = 1 之后，MLIR 会在【每次解析、每个 Pass 跑完之后】自动调用
// 这个函数。验证器是 MLIR 的"安全网"：它保证任何一个 Pass 都不可能悄悄
// 产出一段非法 IR —— 出问题会当场报错，而不是留到底层 CodeGen 才崩。
//
// emitOpError() 会带上位置信息(Location)打印诊断，形如：
//   test/verify.mlir:6:8: error: 'toy.repeat' op repeat count must be > 0
//
// 注意：结构性的检查（"区域只能有一个块"、"必须以终结符结尾"）已经由
// SingleBlock trait 自动完成了，这里只写 trait 表达不了的语义检查。
//
LogicalResult RepeatOp::verify() {
  if (getCount() == 0)
    return emitOpError("repeat count must be > 0（重复次数必须大于 0）");

  // 取区域 -> 唯一的块 -> 块末尾的终结符。
  // 这三层正是 MLIR 的嵌套结构：Operation -> Region -> Block -> Operation。
  Block &bodyBlock = getOperation()->getRegion(0).front();
  auto yieldOp = llvm::dyn_cast<YieldOp>(bodyBlock.getTerminator());
  if (!yieldOp)
    return emitOpError("region must end with toy.yield"
                       "（区域必须以 toy.yield 结尾）");

  if (yieldOp.getValue().getType() != getResult().getType())
    return emitOpError("yielded type must match result type"
                       "（区域交还的类型必须与本 op 的结果类型一致）");

  return success();
}
