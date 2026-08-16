; =============================================================================
; poison_demo.ll —— 手写 4 个小函数，专门观察 poison / undef / freeze
;   对应 docs/learning-guides/llvm-learning-guide.md §2.7 与 docs/notes/llvm-poison-ub.md
;
; 怎么玩：
;   opt -S -passes='instcombine' src/poison_demo.ll     ← 看优化器怎么利用 poison
;   opt -S -passes='default<O2>' src/poison_demo.ll
;
; 关键观察点写在每个函数上面。整份文件不到 40 行，读完就懂。
; =============================================================================

target triple = "x86_64-unknown-linux-gnu"

; ① poison 的产生：nuw 承诺「无符号减不回绕」，0-1 违反了承诺
;    注意：违反标志本身【不是】UB，只是结果变成 poison（延期标记）
define i32 @make_poison() {
  %p = sub nuw i32 0, 1
  ret i32 %p
}

; ② 最反直觉的一条：数值上必然是 0，但整个表达式 DAG 仍被污染成 poison
;    所以 O2 之后你会看到它被直接折叠成 `ret i32 poison`
define i32 @and_poison_zero() {
  %p = sub nuw i32 0, 1
  %z = and i32 %p, 0
  ret i32 %z
}

; ③ select 是「毒性屏障」：只有条件 poison、或【被选中的那一臂】poison，结果才 poison
;    条件为真时，另一臂即使是 poison 也不会传染 —— 这正是 if-conversion 能做的前提
define i32 @select_barrier(i1 %c, i32 %a) {
  %p = sub nuw i32 0, 1
  %r = select i1 %c, i32 %a, i32 %p
  ret i32 %r
}

; ④ freeze 终结 poison：把「可以是任意值」收敛成「某个任意但固定的值」
define i32 @freeze_it() {
  %p = sub nuw i32 0, 1
  %f = freeze i32 %p
  ret i32 %f
}

; ⑤ 真正的 UB：把 poison 当指针去 store（这是「禁位」之一）
;    只有走到这一步，程序才从「有 poison」升级成「未定义行为」
define void @poison_becomes_ub(ptr %base) {
  %bad = sub nuw i32 0, 1
  %p   = getelementptr inbounds i32, ptr %base, i32 %bad
  store i32 42, ptr %p
  ret void
}
