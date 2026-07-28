# ===========================================================================
# 调用 mlir-tblgen，从 LowOps.td 生成 low dialect 的 C++ 代码（.inc 文件）
# ===========================================================================
#
# 与 include/Toy/CMakeLists.txt 完全同构：
#   add_mlir_dialect(<td文件名不带扩展> <dialect名>)
# 会生成 LowOps.{h,cpp}.inc / LowOpsDialect.{h,cpp}.inc，
# 并创建 TableGen 目标 MLIRLowOpsIncGen 供库目标声明依赖。
# ===========================================================================

add_mlir_dialect(LowOps low)
