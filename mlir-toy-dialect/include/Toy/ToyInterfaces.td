//===- ToyInterfaces.td - Toy 的 Op 接口定义 ---------------*- tablegen -*-===//
//
// 定义一个【操作接口 (OpInterface)】：ToyCostOpInterface。
//
// 为什么需要接口？
//   Trait 只能表达"是/否"（这个 op 是不是 Pure？是不是 Commutative？），
//   而接口可以表达"能力"——"你能不能回答我一个问题？"。
//
//   有了接口，一个 Pass 就可以【完全不认识任何具体 dialect】，
//   只要 op 实现了接口，就能对它做事。这正是 MLIR 生态可组合性的根基：
//   linalg / vector / memref 等上百个 dialect 之所以能被同一批通用 Pass
//   处理，靠的就是 LoopLikeOpInterface、MemoryEffectOpInterface 这类接口，
//   而不是在 Pass 里写一长串 `if (isa<AddOp>) ... else if (isa<MulOp>) ...`。
//
// 本文件定义的 ToyCostOpInterface 让 op 自报"我的执行代价是多少"，
// 然后 lib/ToyCostPass.cpp 里的 --toy-print-cost 这个 Pass 就能给
// toy 层和 low 层【两个 dialect】统一算代价——它一行 dialect 相关代码都没有。
//
//===----------------------------------------------------------------------===//

#ifndef TOY_INTERFACES
#define TOY_INTERFACES

include "mlir/IR/OpBase.td"

def ToyCostOpInterface : OpInterface<"ToyCostOpInterface"> {
  let description = [{
    表示"这个操作可以报告自己的执行代价"。
    代价是一个无量纲的相对值：移位/加法约 1，乘法约 5。
    这是真实编译器里 cost model 的最小雏形——多后端选择 kernel、
    自动调度选择 schedule，本质上都是在比这样一个数。
  }];

  // 生成的 C++ 类会放在 mlir::toy 命名空间下：mlir::toy::ToyCostOpInterface。
  // 注意：low dialect 的 op 也可以实现它——接口【不属于】某一个 dialect，
  // 跨 dialect 复用正是它存在的意义。
  let cppNamespace = "::mlir::toy";

  let methods = [
    InterfaceMethod<
      /*desc=*/        "返回该操作的估算执行代价（越小越便宜）",
      /*retTy=*/       "unsigned",
      /*methodName=*/  "getCost",
      /*args=*/        (ins),
      /*methodBody=*/  "",
      // 默认实现：不写 getCost() 的 op 一律算 1。
      // op 若想给出自己的代价，在 .td 里用 DeclareOpInterfaceMethods<...>
      // 声明，然后在 .cpp 里写一个真正的 getCost()（见 toy.mul / low.mul）。
      /*defaultImplementation=*/[{
        return 1;
      }]
    >
  ];
}

#endif // TOY_INTERFACES
