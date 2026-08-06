"""Shared helpers for the TVM lab."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def out_dir() -> Path:
    d = Path(os.environ.get("OUT_DIR", project_root() / "out"))
    d.mkdir(parents=True, exist_ok=True)
    (d / "tvm").mkdir(parents=True, exist_ok=True)
    return d


def require_tvm():
    try:
        import tvm  # noqa: F401
    except ImportError as e:
        print(
            "[ERROR] 未找到 Apache TVM。\n"
            "  安装建议：\n"
            "    pip install apache-tvm\n"
            "  或按 https://tvm.apache.org/docs/install/ 从源码构建（带 LLVM）。\n"
            f"  原始错误: {e}",
            file=sys.stderr,
        )
        sys.exit(2)
    import tvm

    return tvm


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"  [写出] {path}")


def banner(title: str) -> None:
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)
