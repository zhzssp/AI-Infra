# ===========================================================================
# 构建 Toy dialect 库
# ===========================================================================
#
# add_mlir_dialect_library 是 MLIR 提供的宏，用于声明一个 dialect 库。
#   - DEPENDS: 声明对 TableGen 生成目标(MLIRToyOpsIncGen)的依赖，
#              确保 .inc 文件先于本库编译生成。
#   - LINK_LIBS PUBLIC: 链接所需的 MLIR 库。
# ===========================================================================

add_mlir_dialect_library(MLIRToy
  ToyDialect.cpp
  ToyOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/Toy

  DEPENDS
  MLIRToyOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  )
