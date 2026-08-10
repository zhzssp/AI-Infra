# =============================================================================
# passes/CMakeLists.txt —— 把自定义 Pass 编译成可被 opt 加载的插件 MyPasses.so
#   用法（见 scripts/build_passes.sh）：
#     cmake -S passes -B build/passes -DLLVM_DIR=$(llvm-config --cmakedir)
#     cmake --build build/passes
#   产物：build/passes/MyPasses.so
# =============================================================================
cmake_minimum_required(VERSION 3.13)
project(MyPasses LANGUAGES CXX)

# 找到 LLVM 开发包（依赖 LLVMConfig.cmake，由 -DLLVM_DIR 指定其所在目录）
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# 引入 LLVM 的头文件路径与预处理宏
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# C++ 标准与 RTTI：必须与 LLVM 编译时保持一致，否则链接/运行会出问题
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT LLVM_ENABLE_RTTI)
  add_compile_options(-fno-rtti)
endif()

# MODULE 库 = 运行时用 dlopen 加载的插件（.so）。
# 插件里用到的 LLVM 符号在 opt 加载时解析，故无需在此链接 LLVM 静态库。
add_library(MyPasses MODULE
  CountIR.cpp          # Module 级 · 分析型
  CfgInfo.cpp          # Function 级 · 分析型（消费 DominatorTree / LoopInfo）
  TTIInfo.cpp          # Function 级 · 分析型（消费 TargetTransformInfo）
  StrengthReduce.cpp   # Function 级 · 变换型（preserveSet<CFGAnalyses>）
  InjectLogging.cpp    # Module 级 · 变换型
  PluginRegistration.cpp
)

# 统一产物名为 MyPasses.so（去掉 lib 前缀，跨平台后缀强制 .so）
set_target_properties(MyPasses PROPERTIES
  PREFIX ""
  SUFFIX ".so"
)
