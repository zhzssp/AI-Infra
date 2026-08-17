"""读 out/*.json → 生成 markdown 对比表。

各实验的 JSON 字段是统一的（见 README「产物格式」），所以策略之间可以横着比。
默认会自动挑一行做基线，把显存与吞吐都换算成相对比值——**判据看的是比值，
不是绝对值**，因为绝对值随卡型变化，比值才是知识点。
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import pathlib

GIB = 2**30

DEFAULT_COLS = [
    "strategy",
    "world_size",
    "rank",
    "params",
    "peak_alloc_bytes",
    "peak_reserved_bytes",
    "ledger_16phi_bytes",
    "step_time_s",
    "tokens_per_s_global",
]

PRETTY = {
    "strategy": "策略",
    "world_size": "卡数",
    "rank": "rank",
    "params": "Φ",
    "peak_alloc_bytes": "峰值 alloc",
    "peak_reserved_bytes": "峰值 resvd",
    "ledger_16phi_bytes": "手算模型状态",
    "step_time_s": "每步",
    "tokens_per_s_global": "tokens/s(全局)",
    "comm_time_s": "通信耗时",
    "comm_calls": "通信次数",
    "comm_ratio": "通信占比",
}


def fmt(col: str, v) -> str:
    if v is None:
        return "—"
    if col.endswith("_bytes"):
        return f"{v / GIB:.3f} GiB"
    if col == "params":
        return f"{v / 1e6:.1f} M"
    if col == "step_time_s":
        return f"{v * 1e3:.2f} ms"
    if col in ("tokens_per_s_global", "samples_per_s_global"):
        return f"{v:,.0f}"
    if col == "comm_time_s":
        return f"{v * 1e3:.2f} ms"
    if col == "comm_ratio":
        return f"{v:.1%}"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def load(patterns: list[str]) -> list[dict]:
    seen: dict[str, dict] = {}
    for pat in patterns:
        for p in globlib.glob(pat):
            path = pathlib.Path(p)
            if path.suffix != ".json":
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    rec = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            rec["_file"] = path.name
            seen[str(path)] = rec
    rows = list(seen.values())
    rows.sort(key=lambda r: (r.get("world_size", 0), r.get("tag", ""), r.get("rank", 0)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="汇总 out/*.json 成 markdown 表")
    ap.add_argument("--glob", default="out/*.json", help="逗号分隔的 glob 模式")
    ap.add_argument("--cols", default=None, help="逗号分隔的列名")
    ap.add_argument("--markdown", default=None, help="额外写出到这个文件")
    ap.add_argument("--baseline-tag", default=None, help="用哪个 tag 做相对比值的基线")
    ap.add_argument("--rank", type=int, default=None, help="只看某个 rank（默认全列）")
    args = ap.parse_args()

    rows = load([p.strip() for p in args.glob.split(",") if p.strip()])
    if args.rank is not None:
        rows = [r for r in rows if r.get("rank") == args.rank]
    if not rows:
        print(f"[warn] {args.glob} 没匹配到任何 JSON。先跑一次实验再来。")
        return

    cols = [c.strip() for c in args.cols.split(",")] if args.cols else list(DEFAULT_COLS)
    cols = [c for c in cols if any(c in r for r in rows)]

    header = ["tag"] + cols
    lines = [
        "| " + " | ".join(PRETTY.get(c, c) for c in header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for r in rows:
        lines.append(
            "| " + " | ".join([str(r.get("tag", "?"))] + [fmt(c, r.get(c)) for c in cols]) + " |"
        )

    # 相对比值：显存与吞吐各一列
    base = None
    if args.baseline_tag:
        base = next((r for r in rows if r.get("tag") == args.baseline_tag), None)
    if base is None:
        base = next((r for r in rows if r.get("world_size") == 1), rows[0])

    rel = []
    if base.get("peak_alloc_bytes") or base.get("tokens_per_s_global"):
        rel.append("")
        rel.append(f"**相对基线 `{base.get('tag')}`**（world_size={base.get('world_size')}）")
        rel.append("")
        rel.append("| tag | 相对显存 | 相对吞吐 |")
        rel.append("|---|---|---|")
        for r in rows:
            m = (
                r["peak_alloc_bytes"] / base["peak_alloc_bytes"]
                if r.get("peak_alloc_bytes") and base.get("peak_alloc_bytes")
                else None
            )
            t = (
                r["tokens_per_s_global"] / base["tokens_per_s_global"]
                if r.get("tokens_per_s_global") and base.get("tokens_per_s_global")
                else None
            )
            rel.append(
                f"| {r.get('tag')} | "
                f"{f'{m:.3f}x' if m else '—'} | "
                f"{f'{t:.3f}x' if t else '—'} |"
            )

    text = "\n".join(lines + rel)
    print(text)

    if args.markdown:
        p = pathlib.Path(args.markdown)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n", encoding="utf-8")
        print(f"\n[写出] {p}")


if __name__ == "__main__":
    main()
