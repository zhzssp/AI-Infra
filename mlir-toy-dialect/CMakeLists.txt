# ===========================================================================
# 顶层 CMake 配置：一个树外(out-of-tree) MLIR dialect 项目
# ===========================================================================
#
# 这个文件负责：
#   1. 通过 find_package 找到已经构建好的 LLVM 和 MLIR。
#   2. 引入 MLIR/LLVM 提供的 CMake 宏（add_mlir_dialect 等）。
#   3. 添加各个子目录。
#
# 配置命令示例（LLVM_BUILD 指向你的 llvm-project/build）：
#   cmake -G Ninja -S . -B build \
#     -DMLIR_DIR=$LLVM_BUILD/lib/cmake/mlir \
#     -DLLVM_DIR=$LLVM_BUILD/lib/cmake/llvm
# ===========================================================================

cmake_minimum_required(VERSION 3.20.0)

project(toy-dialect LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

# --- 找到 MLIR 和 LLVM ---
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# --- 引入 LLVM/MLIR 的 CMake 模块和宏 ---
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# --- 头文件搜索路径 ---
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
# 本项目自己的 include（源码目录）和生成目录（build 目录）都要加入。
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# --- 子目录 ---
add_subdirectory(include/Toy)
add_subdirectory(include/Low)
add_subdirectory(lib)
add_subdirectory(tools/toy-opt)

# --- 测试（lit / check-toy） ---
# 推导 LLVM build 目录，用于定位 FileCheck（build/bin/FileCheck）。
get_filename_component(LLVM_BUILD_DIR "${LLVM_DIR}/../.." ABSOLUTE)
set(LLVM_EXTERNAL_LIT_DIR "${LLVM_BUILD_DIR}/bin" CACHE PATH "LLVM bin dir")

# add_lit_testsuite 需要 LLVM_EXTERNAL_LIT 指向 llvm-lit。
# 如果配置时未通过 -DLLVM_EXTERNAL_LIT=... 传入，则尝试在 build/bin 下查找。
if(NOT DEFINED LLVM_EXTERNAL_LIT OR NOT LLVM_EXTERNAL_LIT)
  if(EXISTS "${LLVM_EXTERNAL_LIT_DIR}/llvm-lit.py")
    set(LLVM_EXTERNAL_LIT "${LLVM_EXTERNAL_LIT_DIR}/llvm-lit.py")
  elseif(EXISTS "${LLVM_EXTERNAL_LIT_DIR}/llvm-lit")
    set(LLVM_EXTERNAL_LIT "${LLVM_EXTERNAL_LIT_DIR}/llvm-lit")
  endif()
endif()
message(STATUS "LLVM_EXTERNAL_LIT = ${LLVM_EXTERNAL_LIT}")

add_subdirectory(test)
