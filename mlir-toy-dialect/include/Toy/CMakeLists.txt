# ===========================================================================
# 调用 mlir-tblgen，从 .td 文件生成 C++ 代码（.inc 文件）
# ===========================================================================
#
# add_mlir_dialect(<td文件名不带扩展> <dialect名>) 是 MLIR 提供的便捷宏，
# 它会为 ToyOps.td 生成：
#   - ToyOps.h.inc / ToyOps.cpp.inc         （操作的声明与定义）
#   - ToyDialect.h.inc / ToyDialect.cpp.inc （dialect 的声明与定义）
# 并创建一个名为 MLIRToyOpsIncGen 的目标，供库目标声明依赖。
#
# 注意：这里第一个参数是承载操作定义的 td 文件名（ToyOps），
#       第二个参数是 dialect 的名字（toy）。
# ===========================================================================

add_mlir_dialect(ToyOps toy)

# 顺带说明：add_mlir_dialect 展开后其实是 6 次 mlir_tablegen 调用，
# 其中两次是给【自定义类型】用的：
#   -gen-typedef-decls -> ToyOpsTypes.h.inc     （NumType 的类声明）
#   -gen-typedef-defs  -> ToyOpsTypes.cpp.inc   （NumType 的解析/打印/存储实现）
# 文件名之所以是 ToyOpsTypes 而不是 ToyTypes，是因为它以【输入的 td 文件名】
# ToyOps.td 为前缀 —— 而 ToyOps.td 用 include 引入了 ToyTypes.td，
# 所以 !toy.num 的定义也在这一趟里被一起处理了。

# ===========================================================================
# 操作接口（ToyCostOpInterface）—— 需要手动写 tablegen 调用
# ===========================================================================
#
# 接口不在 add_mlir_dialect 的覆盖范围内，正好借这个机会看清
# 「.td -> .inc」这条流水线到底是怎么驱动的：
#   set(LLVM_TARGET_DEFINITIONS <输入.td>)   指定输入
#   mlir_tablegen(<输出.inc> <生成器开关>)     每调一次产出一个文件
#   add_public_tablegen_target(<目标名>)      打包成一个可被 DEPENDS 的目标
#
#   -gen-op-interface-decls -> ToyInterfaces.h.inc   （接口类声明）
#   -gen-op-interface-defs  -> ToyInterfaces.cpp.inc （接口的默认实现、分发表）
# ===========================================================================

set(LLVM_TARGET_DEFINITIONS ToyInterfaces.td)
mlir_tablegen(ToyInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ToyInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRToyInterfacesIncGen)

# 可选：生成 dialect 的 markdown 文档（cmake --build . --target mlir-doc）
add_mlir_doc(ToyOps ToyDialect Toy/ -gen-dialect-doc)
