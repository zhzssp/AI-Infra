# TableGen：LLVM 目标描述与 MLIR ODS 的共同语言

> 来源：[`llvm-learning-guide.md`](../llvm-learning-guide.md) §6

## 是什么

**TableGen 不是 LLVM 里某个“内置函数”，而是一套用于声明式描述信息并生成代码的机制。** 它先解析 `.td` 文件里的 `class`、`def`、`multiclass`、`let`、`dag` 等结构，再交给不同的 backend 生成目标代码或元数据。

在 LLVM 里，TableGen 主要用来描述目标机器：指令、寄存器、调度模型、指令选择匹配模式等。这样可以把大量重复的 C++ 样板和目标相关规则抽象成数据。

## LLVM 里的 `.td` 做什么

1. 描述指令、寄存器、调用约定、调度模型等目标信息
2. 为指令选择器生成 `Pattern` 匹配表
3. 生成 `TargetInstrInfo`、`TargetRegisterInfo` 等代码
4. 让一个架构的大量共性通过继承与记录复用

## MLIR 和 LLVM 的关系

**MLIR 的 ODS（Operation Definition Specification）也是 TableGen 的一个 backend。**

也就是说：

- LLVM 的 `.td`：主要面向目标后端描述
- MLIR 的 `.td`：主要面向 dialect / operation 定义

两者使用的是**同一套 TableGen 语法和解析器**，只是交给不同的 backend 生成不同类型的代码。

## 关键理解

1. **TableGen 本身是“描述 + 代码生成”的基础设施，不是单一 backend。**
2. **LLVM 里的 `.td` 侧重目标描述，MLIR 里的 `.td` 侧重 IR 语义建模。**
3. **同样的语言解决的是不同层级的样板代码爆炸问题。**
4. **读懂 LLVM 的 `.td`，会直接帮助理解 MLIR ODS。**

## 记忆方式

把 TableGen 记成：

> **“用声明式记录把重复的目标/IR 定义变成可生成的代码。”**
