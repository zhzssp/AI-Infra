#===- lit.cfg.py - Toy dialect 的 lit 测试配置 ----------------------------===#
#
# lit 是 LLVM 自带的测试驱动。它会：
#   1. 扫描本目录下所有 .mlir 文件；
#   2. 解析顶部的 // RUN: 行，执行其中的命令（如 toy-opt ... | FileCheck）；
#   3. 用 // CHECK: 行校验命令输出；
#   4. 汇总 PASS / FAIL。
#
# 本文件由 lit 自动加载（通过 lit.site.cfg.py）。
#===========================================================================#

import os
import lit.formats

# 测试名称
config.name = 'toy-dialect'
config.test_format = lit.formats.ShTest(True)

# 只处理 .mlir 文件
config.suffixes = ['.mlir']

# 测试源码根目录与输出根目录
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.toy_obj_root, 'test')

# 用 toy-opt 和 FileCheck 的完整路径替换 RUN 行里的 %toy-opt / %FileCheck
config.substitutions.append(('%toy-opt', config.toy_opt))
config.substitutions.append(('%FileCheck', config.filecheck))
