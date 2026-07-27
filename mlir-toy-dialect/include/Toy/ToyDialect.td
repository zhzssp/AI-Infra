//===- ToyDialect.td - Toy dialect 定义 --------------------*- tablegen -*-===//
//
// 这个文件用 TableGen (ODS, Operation Definition Specification) 声明式地
// 定义一个名为 "toy" 的 Dialect。
//
// 学习要点：
//   - Dialect 是 MLIR 中扩展 IR 的基本单位，拥有独立的命名空间（namespace）。
//   - 下面的定义会被 mlir-tblgen 处理，生成 ToyDialect.h.inc / ToyDialect.cpp.inc。
//
//===----------------------------------------------------------------------===//

#ifndef TOY_DIALECT
#define TOY_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Toy dialect 定义
//===----------------------------------------------------------------------===//

def Toy_Dialect : Dialect {
  // 在 .mlir 文本中，该 dialect 的操作会以 "toy." 作为前缀，例如 toy.add。
  let name = "toy";

  let summary = "一个用于学习 MLIR 的极小 dialect。";

  let description = [{
    Toy dialect 演示了如何定义一个最小可用的 MLIR dialect，
    包含常量、加法、乘法三个操作，并展示常量折叠优化。
  }];

  // 生成的 C++ 类会放在 mlir::toy 命名空间下。
  let cppNamespace = "::mlir::toy";

  // 声明本 dialect 提供 materializeConstant() —— 常量物化。
  // fold() 只是算出一个"属性值(Attribute)"，框架需要把这个属性
  // 重新变成一个真正的常量 Op(toy.constant) 才能替换原来的 op。
  // 这个方法就是负责"从属性值造出一个 toy.constant"。
  let hasConstantMaterializer = 1;
}

// 一个便捷基类：本 dialect 内所有 Op 都继承它，
// 这样就不用每个 Op 都重复写 Toy_Dialect 和 mnemonic 拼接逻辑。
class Toy_Op<string mnemonic, list<Trait> traits = []>
    : Op<Toy_Dialect, mnemonic, traits>;

#endif // TOY_DIALECT
