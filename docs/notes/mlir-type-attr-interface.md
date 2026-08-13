# Type、Attribute、Interface

> 来源：[`mlir-learning-guide.md`](../mlir-learning-guide.md) §2.2 · 第 4 章

同一行里三个概念：

```mlir
%c = toy.complex_add %a, %b {fastmath = true} : !toy.complex<f32>
     ^^^^^^^^^^^^^^^^       ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^
     Op                      Attribute          结果 Type
```

## Type：值是什么

运行时 SSA 值的静态类型，挂在 Value / 结果上。

| 例子 | 在描述 |
|------|--------|
| `i32`、`!toy.complex<f32>` | 标量 / 领域抽象（不必是「内存对象」） |
| `tensor<4x8xf32>` | 值语义形状 + 元素类型 |
| `memref<…, strided<…>>` | 缓冲视图（这里才明显带布局） |

**不是**「必须用内存布局描述的对象」；布局只是部分类型才有。与 Attribute 分工：Type = 值的类别；Attribute = 编译期静态信息。

## Attribute：编译期已知的静态数据

不可变、可 uniqued。常见：整数/浮点/字符串/数组/AffineMap/符号引用等。

```mlir
arith.constant {value = 42 : i32} : i32
//              ^^^^^ Attribute      ^^^ Type
```

- `{ … }` 字典是**常见挂载形式**，不是 Attribute 的唯一定义。
- **值的种类**多是 MLIR 内置的（`IntegerAttr` 等）。
- **名字（key）**一般由该 Op/Dialect 在 ODS 里声明（如 `value`、`fastmath`），**不是**全局内置变量名；换 op 就换一套约定。

## Interface：跨 dialect 的能力契约

**不是**常量折叠专用 hooks。折叠更靠近 Op 自己的 `fold` / canonicalize。

Interface = 一组可 `dyn_cast` 的方法，让通用 Pass 不写死 op 名：

| Interface | Pass 问什么 |
|-----------|-------------|
| `ToyCostOpInterface` | `getCost()` |
| `LoopLikeOpInterface` | 归纳变量、上下界、循环体 |
| `BufferizableOpInterface` | 如何变成 memref |
| `MemoryEffectsOpInterface` | 读/写/分配效应 |

| | Trait | Interface | fold |
|--|-------|-----------|------|
| 角色 | 标签（有没有某性质） | 方法契约（怎么做） | 单 Op 化简 |

## 自定义类型为什么写成 `!mydialect.point<2>`

`!` 不是“随便加的装饰”，而是把类型放进某个 dialect 的命名空间里：

```mlir
i32
f32
tensor<4xf32>
!toy.rect
!mydialect.point<2>
!quant.uniform<i8:f32, 0.125>
```

- `i32`、`tensor<...>` 是 MLIR 内置类型，不带 `!`
- `!toy.rect` / `!mydialect.point<2>` 是**自定义类型**，属于某个方言（dialect）
- `!` 的作用是区分“内置类型”和“方言类型”，避免名字冲突

### 为什么要用模板参数

因为自定义类型往往不是只有一个名字，而是“同一类类型的多个具体实例”——例如：

```mlir
!mydialect.point<1>
!mydialect.point<2>
!mydialect.point<3>
```

这里三者都叫 `point`，但参数决定了它们分别代表 1D / 2D / 3D 的点语义。参数本身就是类型的一部分：

- 维度
- 元素类型
- 量化缩放/精度
- 布局或语义配置

也就是说，模板参数会把“类型家族”和“具体实例”一起编码进去，编译器才能把它们视为不同的静态类型，并在 op 的签名中精确校验。

```mlir
func.func @demo(%p: !mydialect.point<2>) -> !mydialect.point<2> {
  return %p : !mydialect.point<2>
}
```

这里 `%p` 的类型不仅是 `point`，而是“2D point”，因此 `point<1>` 和 `point<2>` 不是同一个类型，不能互传。如果不带参数，类型信息就太粗糙，很多高层语义无法表达。

## 口诀

**Type 描述值，Attribute 描述编译期事实，Interface 描述这类 op 能配合什么通用变换。**

Trait 与 Interface 的能力边界、候选窗口、正确性：见 [mlir-trait-vs-interface.md](./mlir-trait-vs-interface.md)。
