; =============================================================================
; cfg-info.ll —— 分析型 pass 的测试：它只打印，不改 IR
;   跑：bash scripts/run_tests.sh
; =============================================================================

; RUN: opt -load-pass-plugin=%plugin -passes=cfg-info -disable-output %s 2>&1 | FileCheck %s

; CHECK: [cfg-info] @loop_fn
; CHECK: terminator=br
; CHECK: 顶层自然循环数=1
define i32 @loop_fn(i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %loop ]
  %acc.next = add i32 %acc, %i
  %i.next = add i32 %i, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %acc.next
}
