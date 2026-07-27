#===- test/CMakeLists.txt - 注册 lit 测试目标 -----------------------------===#
#
# 这里创建一个名为 check-toy 的测试目标：
#   cmake --build build --target check-toy
# 它会用 LLVM 的 lit 运行本目录下所有 .mlir 测试。
#===========================================================================#

# 计算工具的真实路径，注入到 lit.site.cfg.py
set(TOY_OPT "${CMAKE_BINARY_DIR}/bin/toy-opt")
set(FILECHECK "${LLVM_EXTERNAL_LIT_DIR}/FileCheck")

# 生成 lit.site.cfg.py（把 @VAR@ 替换成真实值）
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  @ONLY
)

# 使用 LLVM 提供的 lit 测试套件宏
add_lit_testsuite(check-toy
  "Running the toy dialect tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS toy-opt
)

# 顺手建一个 check 别名
add_custom_target(check DEPENDS check-toy)
