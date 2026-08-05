//===- ToyTypes.td - Toy dialect 的自定义类型 --------------*- tablegen -*-===//
//
// MLIR 的可扩展性有【三根支柱】：Operation、Type、Attribute。
// 本项目原本只演示了 Operation，类型一律用内置的 i32。
// 这个文件补上第二根支柱：定义一个属于 toy dialect 的自定义类型 `!toy.num`。
//
// 为什么类型必须可扩展？
//   因为高层语义往往【没有】对应的内置类型可用：
//     - tensor<1x3x224x224xf32> 是 builtin 的，但 !tf.resource、
//       !quant.uniform<i8:f32, 0.02>、!gpu.async.token 都不是；
//     - 你要做算力网调度，很可能需要 !mesh.device、!net.channel 这类
//       "本层才有意义"的类型。
//   MLIR 允许每个 dialect 定义自己的类型，然后在【降低】时用
//   TypeConverter 把它换成下层认识的类型（见 lib/ConvertToyToLow.cpp：
//   !toy.num 被换成内置的 i32）。
//
//===----------------------------------------------------------------------===//

#ifndef TOY_TYPES
#define TOY_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Toy/ToyDialect.td"

//===----------------------------------------------------------------------===//
// !toy.num —— toy 层的"数值"类型
//===----------------------------------------------------------------------===//
//
// TypeDef<方言, "C++类名">：生成 C++ 类 mlir::toy::NumType。
//   - mnemonic       决定文本形式的名字：!toy.num
//   - assemblyFormat 为空串表示"没有参数，光写名字即可"
//     （若要写成 !toy.num<8> 这种带参数的类型，就在 parameters 里加字段，
//       并把 assemblyFormat 写成 "`<` $width `>`"）
//
def Toy_NumType : TypeDef<Toy_Dialect, "Num"> {
  let mnemonic = "num";
  let summary = "toy 层的标量数值类型，降低到 low 层后变成内置 i32";
  let description = [{
    一个刻意"没有任何内容"的类型，用来演示 MLIR 的类型可扩展性：
    它在 toy 层携带高层语义，在降低时由 TypeConverter 映射为 i32。
  }];
  let assemblyFormat = "";
}

#endif // TOY_TYPES
