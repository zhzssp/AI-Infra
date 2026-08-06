"""Shared helpers for onnx-delegate-lab."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent


def out_dir() -> Path:
    d = Path(os.environ.get("OUT_DIR", project_root() / "out"))
    (d / "onnx").mkdir(parents=True, exist_ok=True)
    (d / "executorch").mkdir(parents=True, exist_ok=True)
    return d


def banner(title: str) -> None:
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"  [写出] {path}")


def require_onnx():
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError as e:
        print(
            "[ERROR] 需要 onnx + onnxruntime。\n"
            "  pip install onnx onnxruntime numpy\n"
            f"  原始错误: {e}",
            file=sys.stderr,
        )
        sys.exit(2)
    import onnx
    import onnxruntime as ort

    return onnx, ort
