// Verifier（验证器）测试 —— MLIR 的"安全网"。
//
// RUN: %toy-opt %s -split-input-file -verify-diagnostics
//
// 两个新参数值得记住：
//   -split-input-file    用 `// -----` 把本文件切成多个独立模块分别处理，
//                        这样一个非法用例不会阻断后面的用例。
//   -verify-diagnostics  不把错误当成失败，而是核对"报出来的错"是否
//                        与 `expected-error` 标注的一致；一致才算 PASS。
//                        换句话说：这个测试断言的是"该报错的地方必须报错"。
//
// 验证器什么时候被调用？解析完成后、以及【每个 Pass 跑完之后】自动调用。
// 因此任何 Pass 都不可能悄悄产出一段非法 IR —— 出问题当场就被拦住，
// 而不是拖到最底层 CodeGen 才莫名其妙地崩掉。
//
// 分工提醒：
//   - "区域只能有一个块""必须以终结符结尾"由 SingleBlock trait 自动验证；
//   - "count 必须大于 0""交还类型必须匹配"是语义约束，
//     trait 表达不了，所以写在 RepeatOp::verify() 里（见 lib/ToyOps.cpp）。
//
// 运行方式：
//   手动：  toy-opt test/verify.mlir -split-input-file -verify-diagnostics
//   自动：  cmake --build build --target check-toy

// 用例 1：重复次数为 0 —— 语义上没有意义，必须报错。
func.func @bad_count(%arg0: i32) -> i32 {
  // expected-error @+1 {{repeat count must be > 0}}
  %r = toy.repeat 0 {
    toy.yield %arg0 : i32
  } : i32
  return %r : i32
}

// -----

// 用例 2：区域交还的类型与 op 结果类型不一致。
func.func @type_mismatch(%arg0: i32, %arg1: i64) -> i32 {
  // expected-error @+1 {{yielded type must match result type}}
  %r = toy.repeat 2 {
    toy.yield %arg1 : i64
  } : i32
  return %r : i32
}

// -----

// 用例 3：合法的写法，不应有任何诊断。
func.func @good(%arg0: i32) -> i32 {
  %r = toy.repeat 1 {
    toy.yield %arg0 : i32
  } : i32
  return %r : i32
}
