// OpInterface 测试 —— MLIR 生态可组合性的根基。
//
// RUN: %toy-opt %s --toy-to-low --toy-print-cost | %FileCheck %s --check-prefix=BEFORE
// RUN: %toy-opt %s --toy-to-low --low-strength-reduce --toy-print-cost | %FileCheck %s --check-prefix=AFTER
//
// --toy-print-cost 这个 Pass（lib/ToyCostPass.cpp）里【没有出现任何一个
// 具体 op 类型】。它只做一次 dyn_cast<ToyCostOpInterface>，问 op
// "你能报个价吗"。因此：
//   * 同一个 Pass 同时适用于 toy 层和 low 层；
//   * 将来新增 dialect，只要 op 实现这个接口，Pass 一行都不用改。
// 这正是 MLIR 能容纳上百个 dialect 却共用一套通用 Pass 的原因：
// 通用 Pass 依赖【接口】，而不是依赖【具体 dialect】。
// 真实对应物：LoopLikeOpInterface、MemoryEffectOpInterface、
//            DestinationStyleOpInterface（linalg 全家）……
//
// 接口的两种实现方式在本项目里都能看到：
//   low.add / low.shl  —— .td 里只写接口名，沿用默认实现（代价 1），零 C++ 代码；
//   low.mul            —— .td 里 extraClassDeclaration 声明 getCost()，
//                          在 LowOps.cpp 里覆盖成 5。
//   low.constant       —— 根本不实现接口，Pass 自动跳过（不会被打上 toy.cost）。
//
// 这个测试同时量化了"优化到底赚了多少"：
//   强度削减前： low.mul  代价 5
//   强度削减后： low.shl  代价 1
// 这就是 cost model 的最小雏形——多后端选 kernel、自动调度选 schedule，
// 本质上都是在比这样一个数。
//
// 运行方式：
//   手动：  toy-opt test/cost.mlir --toy-to-low --toy-print-cost
//          toy-opt test/cost.mlir --toy-to-low --low-strength-reduce --toy-print-cost
//   自动：  cmake --build build --target check-toy

// 提示：输出 IR 里 low.constant 上【不会】出现 toy.cost 属性，
// 因为它没有实现接口——这一点肉眼看输出即可确认。
// BEFORE-LABEL: func.func @cost_demo
// BEFORE: low.mul
// BEFORE-SAME: toy.cost = 5 : i32

// AFTER-LABEL: func.func @cost_demo
// AFTER-NOT: low.mul
// AFTER: low.shl
// AFTER-SAME: toy.cost = 1 : i32
func.func @cost_demo(%arg0: i32) -> i32 {
  %c = toy.constant 4 : i32
  %m = toy.mul %arg0, %c : i32
  return %m : i32
}
