// =============================================================================
// mini.td —— 30 行体会 TableGen（对应 docs/llvm-learning-guide.md 第 6 章）
//
// 要点：LLVM 后端描述（寄存器/指令/Pattern）和 MLIR 的 ODS 用的是【同一套语言】，
//       只是交给不同的 backend 生成不同的代码。
//
// 跑一下看看它到底"展开"成了什么：
//   llvm-tblgen --print-records src/mini.td
// =============================================================================

// class = 模板：定义字段，本身不产生记录
class Reg<string name, int num> {
  string AsmName = name;
  int    Encoding = num;
}

// def = 具体记录：TableGen 的输出就是一堆这样的 record
def R0 : Reg<"r0", 0>;
def R1 : Reg<"r1", 1>;
def R2 : Reg<"r2", 2>;

class Instr<string mnemonic, int opcode> {
  string Mnemonic  = mnemonic;
  int    Opcode    = opcode;
  bit    IsCommutable = 0;
}

// multiclass = 一次生成一组记录，正是 LLVM 后端避免样板爆炸的手段
multiclass ArithPair<string base, int op> {
  def _rr : Instr<base # "_rr", op>;         // 寄存器-寄存器 形式
  def _ri : Instr<base # "_ri", !add(op, 1)>; // 寄存器-立即数 形式
}

defm ADD : ArithPair<"add", 0x10>;
defm SUB : ArithPair<"sub", 0x20>;

// let = 批量覆盖字段，等价于 LLVM 后端里给一组指令统一打标记
let IsCommutable = 1 in {
  def MUL_rr : Instr<"mul_rr", 0x30>;
}
