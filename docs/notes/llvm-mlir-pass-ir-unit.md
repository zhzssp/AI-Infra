# Pass 传入的 IR 单元：LLVM 四层 vs MLIR Op 层级

> 来源：[`llvm-learning-guide.md`](../llvm-learning-guide.md) 第 3 章 · [`mlir-learning-guide.md`](../mlir-learning-guide.md) §5.4

## 是什么

两边都是：**解析后的内存 IR 对象**按「调度单元」交给 Pass；框架管何时、对哪块根调用，你在回调里用 API 读/改。  
「层级」= PassManager 的 for-each 粒度，不等于优化只能做得很粗。

## 程序员视角

| 框架做 | 你做 |
|--------|------|
| 解析 pipeline、按层拆 IR、调 `run` / `runOnOperation`、分析缓存失效 | 实现回调；遍历/匹配/改写；向 AnalysisManager 要结果 |

不是「只写配置」；抽象的是**调度**，改 IR 仍是普通 C++。

## 如何知道「哪些 IR 会进来」

契约在声明时钉死，不是运行时猜：

| | LLVM | MLIR |
|--|------|------|
| 声明 | Pass 层级 → `run(Module&)` / `run(Function&)` / … | `OperationPass<Op>` → `getOperation()` |
| 多实例 | adaptor：`function(...)` / `loop(...)` 对每个单元回调 | pipeline：`builtin.module(func.func(...))` |
| 根以下 | 自己 `for (BB/Inst)` | 自己 `walk` / Pattern root |

**Function Pass ≠ 黑盒函数**：传入完整 `Function`（BB/Inst + use-def），可用 `AM.getResult<DominatorTreeAnalysis>(F)` 等。  
「对每个函数调度」≠「看不见内部、无差别瞎优化」。

## 层级：固定四档 vs Op 嵌套

- **LLVM**：Module → CGSCC → Function → Loop，贴合偏扁的 LLVM IR。细匹配在 Pass 内（逐指令 / LoopInfo），不靠第 5 层 PM。
- **MLIR**：层 = IR 树里选定的 Op（可自定义），服务多抽象层渐进降低。  
  同构直觉：都是分层跑 Pass；差别是层由固定 IR 粒度还是开放 Op 嵌套定义。

四层不会「复杂场景天然更差」：调度粒度够用；细活与启发式在 Pass/分析里。多层领域 IR 通常 **MLIR 在上、降到 LLVM**。

## 内存形态（共同）

文本 `.ll` / `.mlir` → parse → **包含树 + use-def 图** → PM 传入某层根引用。

## 示例：`clamp0`（对照核心）

同一语义：`y = a + b`；若 `y > 0` 返回 `y`，否则返回 `0`。

```c
int clamp0(int a, int b) {
  int y = a + b;
  if (y > 0) return y;
  return 0;
}
```

### LLVM：文本 → Function Pass 看到的对象

```llvm
define i32 @clamp0(i32 %a, i32 %b) {
entry:
  %y = add i32 %a, %b
  %cmp = icmp sgt i32 %y, 0
  br i1 %cmp, label %then, label %else
then:
  ret i32 %y
else:
  ret i32 0
}
```

pipeline 如 `function(你的pass)` 时，每次根是 **`Function &F`**（`@clamp0`）：

```text
Module
└─ Function "clamp0"          ← run(F) 的 F
   ├─ arg: %a, %b
   ├─ BasicBlock "entry"
   │    ├─ AddInst / ICmpInst / BranchInst
   ├─ BasicBlock "then"  → ReturnInst ret %y
   └─ BasicBlock "else"  → ReturnInst ret 0
```

```cpp
// 伪代码：体内可读完整结构 + 分析
PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
  for (BasicBlock &BB : F)
    for (Instruction &I : BB) { /* ... */ }
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
}
```

### MLIR：文本 → `func.func` Pass 看到的对象

```mlir
func.func @clamp0(%a: i32, %b: i32) -> i32 {
  %y = arith.addi %a, %b : i32
  %c0 = arith.constant 0 : i32
  %cmp = arith.cmpi sgt, %y, %c0 : i32
  cf.cond_br %cmp, ^then, ^else
^then:
  return %y : i32
^else:
  return %c0 : i32
}
```

pipeline 如 `builtin.module(func.func(你的pass))` 时，每次 **`getOperation()`** 是该 **`func.func`**：

```text
builtin.module
└─ func.func @clamp0           ← getOperation()
   └─ Region
      ├─ Block ^entry (%a, %b)
      │    ├─ arith.addi / constant / cmpi / cf.cond_br
      ├─ Block ^then → func.return %y
      └─ Block ^else → func.return %c0
```

```cpp
void runOnOperation() override {
  auto f = cast<func::FuncOp>(getOperation());
  f.walk([](Operation *op) { /* addi / cmpi / ... */ });
}
```

### 并排

| | LLVM Function Pass | MLIR `func.func` Pass |
|--|--------------------|------------------------|
| 传入根 | `Function &` | `Operation*` ≈ `FuncOp` |
| 下一层 | `BasicBlock` | `Block`（在 `Region` 里） |
| 叶子 | `Instruction` | 子 `Operation`（如 `arith.addi`） |

## 对照一句

**调度单元**：LLVM 看 `run` 参数类型；MLIR 看 `OperationPass<T>` + pipeline 路径。  
**能看见多少**：根由 PM 给定；子结构全靠 API 自己挖。
