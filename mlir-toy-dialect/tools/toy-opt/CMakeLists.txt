# ===========================================================================
# 构建 toy-opt 命令行工具
# ===========================================================================
#
# 它复用 MLIR 的 MlirOptMain，因此需要链接：
#   - 我们自己的 MLIRToy 库；
#   - MLIR 所有 dialect 库（用于 registerAllDialects）；
#   - MLIR 所有 conversion 库；
#   - MlirOptLib（提供 MlirOptMain 主流程）。
# ===========================================================================

# 收集 MLIR 中所有已注册的 dialect / conversion / extension 库。
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIRToy
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRIR
  )

add_llvm_executable(toy-opt toy-opt.cpp)

llvm_update_compile_flags(toy-opt)
target_link_libraries(toy-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(toy-opt)
