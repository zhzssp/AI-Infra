# Conversion：LLVM 指令类 vs MLIR Dialect Conversion

> 来源：[`llvm-learning-guide.md`](../llvm-learning-guide.md) §2.4 · [`mlir-learning-guide.md`](../mlir-learning-guide.md) 第 6 章

英文都叫 Conversion，层级完全不同。

## LLVM IR：一类改值类型的指令

仍在同一份 LLVM IR 里，把值从类型 A 变成类型 B（cast）：

| 指令 | 作用 |
|------|------|
| `trunc` / `zext` / `sext` | 整数变窄 / 零扩展 / 符号扩展 |
| `fptrunc` / `fpext` | 浮点精度升降 |
| `fptoui` / `sitofp` … | 整↔浮 |
| `ptrtoint` / `inttoptr` | 指针↔整数 |
| `bitcast` | 位模式不变、换类型解读 |
| `addrspacecast` | 换地址空间 |

## MLIR：跨 dialect 的 lowering 框架

**Dialect Conversion** 是 Pass 基础设施，把非法 op / 非法类型合法化到另一层 dialect（如 `vector`/`func` → `llvm`）。

三件套：

1. **ConversionTarget** — 哪些 op 合法 / 非法
2. **ConversionPattern** — 非法 op 怎么改写
3. **TypeConverter**（可选）— 类型怎么映射（如 `!toy.num` → `i32`）

## 对照

| | LLVM Conversion | MLIR Dialect Conversion |
|--|-----------------|-------------------------|
| 是什么 | 一类 **IR 指令** | 一套 **Pass / 重写框架** |
| 改什么 | 同一个值的 **类型** | **op + 类型**，常跨 dialect |
| 典型例子 | `%b = zext i32 %a to i64` | `--convert-vector-to-llvm` |
| 类比 | C 里的 `(int)x` | 编译器里「这一层 → 下一层」 |

`--convert-*-to-llvm` 用的是 **MLIR Conversion**；进到真正的 `.ll` 后，才偶尔再看到 LLVM 的 `zext`/`bitcast` 等 Conversion **指令**。
