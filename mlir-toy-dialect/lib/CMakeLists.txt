# ===========================================================================
# 构建 Toy dialect 库
# ===========================================================================
#
# add_mlir_dialect_library 是 MLIR 提供的宏，用于声明一个 dialect 库。
#   - DEPENDS: 声明对 TableGen 生成目标(MLIRToyOpsIncGen 等)的依赖，
#              确保 .inc 文件先于本库编译生成。
#   - LINK_LIBS PUBLIC: 链接所需的 MLIR 库。
# ===========================================================================

add_mlir_dialect_library(MLIRToy
  # --- toy（高层）dialect 本体 ---
  ToyDialect.cpp          # dialect 注册 + 自定义类型注册 + 常量物化
  ToyOps.cpp              # 操作实现：fold / verify / 接口方法
  ToyInterfaces.cpp       # 操作接口 ToyCostOpInterface 的生成代码
  ToyPasses.cpp           # RewritePattern + 贪心驱动器（--toy-simplify）
  ToyCostPass.cpp         # 基于接口的通用 Pass（--toy-print-cost）

  # --- low（低层）dialect 及降低相关 ---
  LowDialect.cpp          # low dialect 注册
  LowOps.cpp              # low 操作实现
  LowPasses.cpp           # 贪心版降低 + 强度削减
  ConvertToyToLow.cpp     # Dialect Conversion 版降低（--toy-to-low-convert）

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/Toy
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/Low

  DEPENDS
  MLIRToyOpsIncGen          # toy 的 op / dialect / 自定义类型 !toy.num
  MLIRToyInterfacesIncGen   # ToyCostOpInterface
  MLIRLowOpsIncGen          # low 的 op / dialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  # ToyPasses.cpp 需要：Pass 基础设施 + 贪心改写驱动器。
  MLIRPass
  MLIRTransforms
  # ConvertToyToLow.cpp 需要：DialectConversion（在 TransformUtils 里）。
  MLIRTransformUtils
  )
