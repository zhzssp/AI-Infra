#!/usr/bin/env python3
"""Step ET2 — Write the ExecuTorch / ORT EP / IREE comparison cheat-sheet into out/."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_common import banner, out_dir, write_text


def main() -> None:
    banner("ET2 三系统对照表落盘")
    text = """# 三系统对照：ExecuTorch / ORT EP / IREE flow.dispatch

> 本表与 [`docs/learning-guides/executorch-learning-guide.md`](../../docs/learning-guides/executorch-learning-guide.md) §8、
> [`docs/learning-guides/ai-compiler-foundations-learning-guide.md`](../../docs/learning-guides/ai-compiler-foundations-learning-guide.md) §3.3–3.4 对齐。

| 维度 | ExecuTorch Partitioner | ORT EP `GetCapability` | IREE `flow.dispatch` |
|------|------------------------|------------------------|----------------------|
| 输入 IR | Edge Dialect | ONNX Graph | Linalg 等 → IREE 流水线 |
| 谁声明能力 | 你写的 `Partitioner` | 各 EP 回调 | 编译器 pass（非厂商回调） |
| 划分结果 | `delegation_tag` | capability 子图 | `flow.dispatch` region |
| 编译子图 | `preprocess` → blob → `.pte` | `Compile` → EP 内部 | Codegen → `.vmfb` |
| 运行时派发 | `call_delegate` | Session 按分区调 EP | HAL `command_buffer.dispatch` |
| 未覆盖算子 | portable kernel | 回退其他 EP（常 CPU） | 同一程序内换 variant/device |
| 决策时刻 | **导出 / to_backend 期** | **Session 构建期** | **编译相位** |
| 更像什么 | 声明式委托 API | 插件式运行时分区 | 把划分编进程序 |

## 同一张图，四个数字

主角 `tiny_mlp`（`Gemm→Relu→Add`）在四个 lab 里都跑过，各自数一个量：

| 系统 | 数什么 | 在哪跑 |
|------|--------|--------|
| ExecuTorch | delegate 子图数（per_node=6 / connected=2） | `executorch_lab/01_partitioner_lab.py` |
| ORT | optimized 图里剩几个节点（融了几个） | `onnx_lab/03_ort_ep_partition.py` |
| TVM | `FuseOps` 前后的函数个数 | `../tvm-fatbin-lab/tvm_lab/02_fusion_relay.py` |
| IREE | `flow.dispatch` 调用点个数 | `../iree-lab/scripts/run_phases.sh` |

**源码里写了 4 个逐元素/归约算子，四个系统最后各留下几个 kernel？**
差值就是各自融合能力的量化结果。

## 一句话

- **ORT**：运行时图 + 多 EP 竞价式能力声明——本 lab ONNX 轨步骤 03。
- **ExecuTorch**：导出期打 tag，blob 进 `.pte`——本 lab ExecuTorch 轨步骤 01。
- **IREE**：划分是编译期一等公民——动手见 `iree-lab/`，教材见 `docs/learning-guides/iree-learning-guide.md`。

## 一条不能补救的因果

ExecuTorch 的 `per_node` 粒度下 `Relu` 自成一个 delegate，中间张量被强制物化。
**TVM 的 `FuseOps` 与 IREE 的 `flow.dispatch` 都只在自己的子图内部做融合**——
跨 delegate 边界它们连图都看不见，所以这次丢掉的融合机会谁都补不回来。
见 `docs/learning-guides/00-end-to-end-pipeline.md` 第 5 章断链表第 ② 行。

## 与 tvm-fatbin-lab 的关系（正交，不互相替代）

| 项目 | 回答的问题 |
|------|-----------|
| `onnx-delegate-lab`（本项目） | **谁来划分、边界付什么代价**（研究问题①②） |
| `tvm-fatbin-lab` | **划完怎么算得快** + **多变体怎么打包**（研究问题⑥为主） |
| `iree-lab` | **一条工业流水线怎么把上面这些串成相位**（编译期一等公民） |

融合（TVM）≈ 单后端上的划分；EP/Partitioner ≈ 多后端上的划分。同构，分层不同。
"""
    write_text(out_dir() / "executorch" / "02_THREE_SYSTEMS.md", text)
    print(text)
    print("[OK] 对照表已写出。")


if __name__ == "__main__":
    main()
