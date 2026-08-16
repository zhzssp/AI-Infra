// =============================================================================
// kernel.c —— 「一个迷你 AI 算子」场景，用来一次性观察 LLVM 的全部核心要点
// -----------------------------------------------------------------------------
// 为什么是这几个函数？每一个都对应 docs/learning-guides/llvm-learning-guide.md 里的一个必学点：
//
//   axpy()           restrict → IR 上的 noalias / align / dereferenceable  (§2.6)
//                    a*x+y    → llvm.fmuladd / contract / FMA              (§2.6, §2.9)
//                    循环      → 可向量化 <4 x float>                       (§4.5)
//   axpy_may_alias() 去掉 restrict → 别名未知 → 运行时指针检查 + 双版本循环  (§4.2)
//   relu_sum()       struct + 数组下标 → getelementptr inbounds            (§2.5)
//                    三元表达式        → select（poison 屏障那条规则的主角） (§2.7)
//                    累加              → phi + 向量规约 intrinsic           (§2.3, §2.9)
//   scale8()         乘 2 的幂 → 给自定义 strength-reduce pass 练手          (§3.3)
//   clamp0()         if/else 汇合 → alloca → phi → select 三份 IR 的演进      (§2.3, §2.7)
//   main()           调用 printf → 能真正链接成可执行文件跑起来
//
// 代码部分刻意保持在 60 行以内：读一遍就能记住，跑一遍就能对上 IR。
//
// 为什么是「axpy + ReLU」这个组合？因为它就是全仓库图级主角 tiny_mlp
// （Gemm → Relu → Add）里那个 Gemm 降到 C 层之后的样子：
//   Gemm 的最内层 acc += x[i][k]*W[j][k]  ≡  axpy 的 y[i] = a*x[i] + y[i]
//   紧跟其后的 Relu                        ≡  relu_sum 里的 v > 0 ? v : 0
// 链路全图见 docs/learning-guides/00-end-to-end-pipeline.md。
// =============================================================================

#include <stdio.h>

// 一个最小的「张量」：数据指针 + 长度。用来产生多级 getelementptr。
typedef struct {
    float *data;
    int    len;
} Tensor;

// ① AXPY：y = a*x + y
//    restrict 告诉编译器 x/y 不重叠 → clang 会在 IR 参数上打 noalias
void axpy(int n, float a, const float *restrict x, float *restrict y) {
    for (int i = 0; i < n; ++i)
        y[i] = a * x[i] + y[i];
}

// ② 同一个循环，但没有 restrict：编译器只能假设 x/y 可能重叠（MayAlias）
void axpy_may_alias(int n, float a, const float *x, float *y) {
    for (int i = 0; i < n; ++i)
        y[i] = a * x[i] + y[i];
}

// ③ ReLU 求和：t->data[i] 会展开成 getelementptr；三元表达式会变成 select
float relu_sum(const Tensor *t) {
    float s = 0.0f;
    for (int i = 0; i < t->len; ++i) {
        float v = t->data[i];
        s += v > 0.0f ? v : 0.0f;
    }
    return s;
}

// ④ 乘 2 的幂：自定义 pass 会把 mul x, 8 改写成 shl x, 3（窥孔 / 强度削减）
int scale8(int x) { return x * 8; }

// ⑤ 钳位：r 在两条路径上各被赋值一次 —— SSA 不允许，汇合点必须选值。
//    同一个函数在三份 IR 里长三个样，是观察 SSA 的最小样本：
//      -O0            r 是一块 alloca，两条路各 store 一次
//      +mem2reg       alloca 消失，汇合块出现 phi [%add, %if.then], [0, %if.else]
//      -O2            phi 被 if-conversion 换成 select（§2.7 的毒性屏障主角）
int clamp0(int a, int b) {
    int y = a + b;
    int r;
    if (y > 0) r = y;
    else       r = 0;
    return r;
}

int main(void) {
    float x[8] = {1, -2, 3, -4, 5, -6, 7, -8};
    float y[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    axpy(8, 2.0f, x, y);                 // y = 2*x
    Tensor t = {.data = y, .len = 8};

    printf("relu_sum = %.1f\n", relu_sum(&t));   // 2*(1+3+5+7) = 32.0
    printf("scale8(5) = %d\n", scale8(5));       // 40
    printf("clamp0(3,-5) = %d\n", clamp0(3, -5));// -2 → 钳到 0
    return 0;
}
