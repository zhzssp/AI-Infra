// 自定义类型测试 —— MLIR 可扩展性的第二根支柱（Operation / Type / Attribute）。
//
// RUN: %toy-opt %s --canonicalize | %FileCheck %s
//
// 看点一：!toy.num 是我们自己定义的类型（见 include/Toy/ToyTypes.td）。
//   解析器之所以认识它，靠的是两件事：
//     1) dialect 里 useDefaultTypePrinterParser = 1（自动生成解析/打印）；
//     2) ToyDialect::initialize() 里的 addTypes<NumType>()。
//   忘了第 2 步，写 !toy.num 就会报 unknown type。
//
// 看点二：UnboxOp::fold 演示了 fold 的【第二种】返回值。
//   add/mul 的 fold 返回 Attribute（"我算出了一个新常量"）；
//   这里的 fold 返回 Value（"别用我，直接用这个已有的值"）。
//   OpFoldResult 就是这两者的联合体。
//   于是 unbox(box(x)) 整对消失，%arg0 被直接接到 return 上。
//
// 运行方式：
//   手动：  toy-opt test/types.mlir --canonicalize
//   自动：  cmake --build build --target check-toy

// 装箱后立刻拆箱 —— 一对多余的包装，应该被完全折叠掉。
// CHECK-LABEL: func.func @roundtrip
// CHECK-NOT: toy.box
// CHECK-NOT: toy.unbox
// CHECK: return %arg0 : i32
func.func @roundtrip(%arg0: i32) -> i32 {
  %n = toy.box %arg0 : i32 -> !toy.num
  %y = toy.unbox %n : !toy.num -> i32
  return %y : i32
}

// !toy.num 真的会流出函数边界 —— 验证类型的解析/打印往返正确。
// CHECK-LABEL: func.func @keep_num
// CHECK: toy.box %arg0 : i32 -> !toy.num
// CHECK: return %{{.*}} : !toy.num
func.func @keep_num(%arg0: i32) -> !toy.num {
  %n = toy.box %arg0 : i32 -> !toy.num
  return %n : !toy.num
}
