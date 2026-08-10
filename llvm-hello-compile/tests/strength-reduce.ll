; =============================================================================
; strength-reduce.ll —— 自定义 pass 的 lit/FileCheck 测试
;   这就是 LLVM（以及 MLIR）测试的标准写法：RUN: 跑命令，CHECK: 断言输出。
;   跑：bash scripts/run_tests.sh
; =============================================================================

; RUN: opt -load-pass-plugin=%plugin -passes=strength-reduce -S %s | FileCheck %s

define i32 @scale8(i32 %x) {
; 乘 2 的幂应该被改写成移位
; CHECK-LABEL: define i32 @scale8
; CHECK: shl i32 %x, 3
; CHECK-NOT: mul i32 %x, 8
  %m = mul i32 %x, 8
  ret i32 %m
}

define i32 @scale3(i32 %x) {
; 不是 2 的幂，必须原样保留
; CHECK-LABEL: define i32 @scale3
; CHECK: mul i32 %x, 3
  %m = mul i32 %x, 3
  ret i32 %m
}
