# New PM：层级、默认 pipeline、以及「PM 也是 pass」

> 来源：[`llvm-learning-guide.md`](../llvm-learning-guide.md) 第 3 章 · §4.4

## 两套引擎（别和层级搞混）

| | New PM | Legacy PM |
|--|--------|-----------|
| 用在哪 | **中端**优化（`opt`、`-O2`） | **后端** CodeGen |
| 现状 | 中端正统 | codegen 迁移中 |

## 四层 PassManager（同一套 New PM 里）

```text
Module → CGSCC → Function → Loop
```

| 层 | 一次看什么 | 典型内置 pass | 为何单独一层 |
|----|------------|---------------|--------------|
| Module | 整个翻译单元 | `globalopt`、`globaldce` | 跨函数 / 全局符号 |
| CGSCC | 调用图强连通分量 | **`inline`** | 按调用图后序：先优化被调者 |
| Function | 单个函数 | `instcombine`、`gvn`、`sroa` | 分析按函数缓存 |
| Loop | 单个循环 | `loop-rotate`、`licm` | 循环规范形与专用分析 |

分层为了：**作用域匹配、分析缓存粒度、调度局部性**（对函数 A 跑完一串再处理 B，优于全体先跑 Pass1 再全体 Pass2）。

## 内置 vs 自定义

各层 PM **首先是调度框架**，不是空架子专门留给自定义 pass。

- **默认会跑**：`clang -O2` / `opt -passes='default<O2>'` → `PassBuilder` 已排好大量内置 pass  
- **自定义**：可选插件，挂进某层，或 `opt -passes=helloworld` 单独测  

查看：`opt -passes='default<O2>' -debug-pass-manager`；`opt --print-passes`。

## 为什么「PassManager 本身就是 pass」

New PM 里 **pass = 可调度的一单位工作**（对该层 IR 跑一遍，返回 `PreservedAnalyses`），不要求必须是「优化器」。

`FunctionPassManager` 对外也是 Function pass：

```text
run(Function &F, AM):
  for p in 我装着的 pass:
    p.run(F, AM)
```

于是 Module 可用 adaptor 把整个 FPM 当成「一个」Function 级 pass 来调度 → **嵌套 = 把内层 PM 塞进外层 PM**。

| 直觉 | 实际 |
|------|------|
| PM = 外面的框架，pass = 工人 | PM **也是**工人，只是会再指挥一小队工人 |

```text
ModulePassManager
  └─ adaptor → FunctionPassManager   ← 对外是一个 pass
                 ├─ InstCombine
                 └─ adaptor → LoopPassManager
                                └─ LICM
```
