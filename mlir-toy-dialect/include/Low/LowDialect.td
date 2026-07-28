//===- LowDialect.td - Low dialect 定义 --------------------*- tablegen -*-===//
//
// 这个文件定义一个名为 "low" 的【低层】 Dialect，用来和高层的 "toy"
// Dialect 形成对比，直观展示 MLIR 里【不同层级 dialect 分工协作】的思想：
//
//   - toy  层（高层）：关注"数学/代数语义"，做 x*1=x、常量折叠这类优化。
//   - low  层（低层）：关注"硬件指令代价"，做强度削减 x*4 => x<<2 这类优化。
//
// 关键点：low 层多出一个高层【根本不存在】的概念 —— 移位(shl)。
//         正因为高层看不到"移位指令"，"x*4 换成左移 2 位"这个优化
//         只能在降低(lowering)到 low 层之后才做得了。
//
//===----------------------------------------------------------------------===//

#ifndef LOW_DIALECT
#define LOW_DIALECT

include "mlir/IR/OpBase.td"

def Low_Dialect : Dialect {
  // 在 .mlir 文本中，本 dialect 的操作以 "low." 为前缀，例如 low.shl。
  let name = "low";

  let summary = "一个贴近硬件指令的低层 dialect，用于对比展示层级分工。";

  let description = [{
    Low dialect 演示"降低(lowering)后的低层 IR"长什么样：
    它含有 constant/add/mul，还多了一个高层没有的 shl(左移)。
    低层优化（强度削减）利用 shl 把"乘 2 的幂"换成移位。
  }];

  // 生成的 C++ 类放在 mlir::low 命名空间下。
  let cppNamespace = "::mlir::low";
}

// 便捷基类：本 dialect 内所有 Op 都继承它。
class Low_Op<string mnemonic, list<Trait> traits = []>
    : Op<Low_Dialect, mnemonic, traits>;

#endif // LOW_DIALECT
