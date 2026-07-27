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

# 可选：生成 dialect 的 markdown 文档（cmake --build . --target mlir-doc）
add_mlir_doc(ToyOps ToyDialect Toy/ -gen-dialect-doc)
