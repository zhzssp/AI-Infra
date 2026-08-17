#!/usr/bin/env bash
# =============================================================================
# setup.sh —— 一次性安装 LLVM/MLIR 运行与构建依赖（幂等，可重复运行）
# -----------------------------------------------------------------------------
# 它会用 conda-forge 创建/补齐一个自包含环境，内含：
#   - mlir / llvm / llvmdev 17.0.6  MLIR 头文件、静态库、CMake 配置、mlir-tblgen、FileCheck
#   - clang / clangxx 17.0.6        llvm-hello-compile 的前端（本 lab 不用，同环境顺带装上）
#   - cmake / ninja                 构建系统
#   - cxx-compiler (gcc)            与 conda-forge 二进制 ABI 匹配的编译器
#   - lit                           LLVM 的自动化测试驱动
# 并把 FileCheck 软链到 <env>/bin，方便 CMake 与 PATH 定位。
#
# 为什么用 conda：本机系统 GLIBC 偏旧，官方 Ubuntu 预编译包无法运行；
#                conda-forge 的包基于旧 sysroot 构建，兼容性最好。
#                另外它不需要 root——没有 sudo 权限时这是唯一可行路径。
#
# 想一次配好**全部六个 lab**（含 TVM / ONNX / IREE / torch），用仓库根目录的：
#     bash setup.sh
# 本脚本只管 MLIR 这一条轨，保留是为了单独跑本 lab 时不必装其余依赖。
# =============================================================================
set -euo pipefail

: "${CONDA_HOME:=/data/mm64/hanishzheng/miniforge3}"
: "${TOY_ENV:=mlir-env}"

if [[ ! -x "${CONDA_HOME}/bin/conda" ]]; then
  echo "[setup] 找不到 conda：${CONDA_HOME}/bin/conda"
  echo "[setup] 请先安装 miniforge：https://github.com/conda-forge/miniforge"
  echo "[setup]   或设置 CONDA_HOME 指向已有 conda 安装根目录后重试。"
  exit 1
fi

eval "$("${CONDA_HOME}/bin/conda" shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${TOY_ENV}"; then
  echo "[setup] conda 环境 '${TOY_ENV}' 已存在，检查并补齐依赖 ..."
else
  echo "[setup] 创建 conda 环境 '${TOY_ENV}' ..."
  conda create -n "${TOY_ENV}" -y -c conda-forge llvm=17.0.6 mlir=17.0.6
fi

echo "[setup] 安装/确认 MLIR + 构建 + 测试工具 ..."
# clang/clangxx 本 lab 用不上，但同一个 mlir-env 也服务 llvm-hello-compile，
# 而那边硬依赖 clang 做前端。无 root 时系统里拿不到，只能从这里一并装上。
conda install -n "${TOY_ENV}" -y -c conda-forge \
  mlir=17.0.6 llvm=17.0.6 llvmdev=17.0.6 clang=17.0.6 clangxx=17.0.6 \
  cmake ninja cxx-compiler lit

PREFIX="${CONDA_HOME}/envs/${TOY_ENV}"

# FileCheck 在 conda 布局里位于 libexec/llvm，软链到 bin 便于统一定位。
if [[ -x "${PREFIX}/libexec/llvm/FileCheck" && ! -e "${PREFIX}/bin/FileCheck" ]]; then
  ln -sf "${PREFIX}/libexec/llvm/FileCheck" "${PREFIX}/bin/FileCheck"
  echo "[setup] 已软链 FileCheck -> ${PREFIX}/bin/FileCheck"
fi

echo "[setup] 依赖就绪。环境前缀：${PREFIX}"
echo "[setup] 下一步：scripts/build.sh"
