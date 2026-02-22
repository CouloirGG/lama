"""Generate GitHub Actions Job Summary for calibration harvest runs.

Reads JSONL calibration data and outputs GitHub-flavored markdown to stdout.
Pure stdlib — no external dependencies.

Usage:
    python scripts/harvest_summary.py ~/.poe2-price-overlay/cache/
    python scripts/harvest_summary.py --dir ~/.poe2-price-overlay/cache/ --shard path/to/shard.json.gz
"""

import argparse
import gzip
import io
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Ensure stdout handles Unicode (Windows cp1252 can't print block chars)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

def load_jsonl_files(directory: Path) -> list[dict]:
    """Load all calibration/elite JSONL files from *directory*."""
    records = []
    for pattern in ("calibration_shard_*.jsonl", "elite_shard_*.jsonl"):
        for path in sorted(directory.glob(pattern)):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
    return records


def load_shard(path: Path) -> dict | None:
    """Load a compressed shard and return its metadata."""
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_state_files(directory: Path) -> list[dict]:
    """Load all harvester state JSON files."""
    states = []
    for pattern in ("harvester_state_p*.json", "elite_harvester_state_p*.json"):
        for path in sorted(directory.glob(pattern)):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                    data["_filename"] = path.name
                    states.append(data)
            except (json.JSONDecodeError, OSError):
                pass
    return states


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def bar(value: int, max_value: int, width: int = 20) -> str:
    """Render a simple text bar using block chars."""
    if max_value <= 0:
        return ""
    filled = round(value / max_value * width)
    filled = min(filled, width)
    return "\u2593" * filled + "\u2591" * (width - filled)


def pct(count: int, total: int) -> str:
    """Format a percentage string."""
    if total == 0:
        return "0.0%"
    return f"{count / total * 100:.1f}%"


def fmt_size(nbytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} GB"


# ---------------------------------------------------------------------------
# Summary sections
# ---------------------------------------------------------------------------

def section_header(records: list[dict]) -> str:
    league = "Unknown"
    if records:
        league = records[0].get("league", "Unknown")
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        f"# Calibration Harvest Summary",
        "",
        f"**Date:** {date}  ",
        f"**League:** {league}  ",
        f"**Total samples:** {len(records):,}",
        "",
    ]
    return "\n".join(lines)


def section_harvester_breakdown(directory: Path) -> str:
    std_count = 0
    elite_count = 0
    for p in directory.glob("calibration_shard_*.jsonl"):
        with open(p) as f:
            std_count += sum(1 for line in f if line.strip())
    for p in directory.glob("elite_shard_*.jsonl"):
        with open(p) as f:
            elite_count += sum(1 for line in f if line.strip())

    total = std_count + elite_count
    lines = [
        "## Harvester Breakdown",
        "",
        "| Harvester | Samples | Share |",
        "|-----------|--------:|------:|",
        f"| Standard | {std_count:,} | {pct(std_count, total)} |",
        f"| Elite | {elite_count:,} | {pct(elite_count, total)} |",
        f"| **Total** | **{total:,}** | **100%** |",
        "",
    ]
    return "\n".join(lines)


def section_category(records: list[dict]) -> str:
    counts = Counter(r.get("item_class", "Unknown") for r in records)
    if not counts:
        return ""
    sorted_cats = counts.most_common()
    max_count = sorted_cats[0][1] if sorted_cats else 1

    lines = [
        "## Samples per Category",
        "",
        "| Category | Count | Distribution |",
        "|----------|------:|:-------------|",
    ]
    for cat, count in sorted_cats:
        lines.append(f"| {cat} | {count:,} | `{bar(count, max_count)}` |")
    lines.append("")
    return "\n".join(lines)


def section_grade(records: list[dict]) -> str:
    grade_order = ["S", "A", "B", "C", "JUNK"]
    counts = Counter(r.get("grade", "?") for r in records)
    total = len(records)

    lines = [
        "## Grade Distribution",
        "",
        "| Grade | Count | Pct | |",
        "|-------|------:|----:|:--|",
    ]
    max_count = max(counts.values()) if counts else 1
    for g in grade_order:
        c = counts.get(g, 0)
        lines.append(f"| {g} | {c:,} | {pct(c, total)} | `{bar(c, max_count, 15)}` |")
    lines.append("")
    return "\n".join(lines)


def section_price_brackets(records: list[dict]) -> str:
    brackets = [
        ("<1d", 0, 1),
        ("1-10d", 1, 10),
        ("10-50d", 10, 50),
        ("50-100d", 50, 100),
        ("100-300d", 100, 300),
        ("300-1000d", 300, 1000),
        ("1000-1500d", 1000, 1500),
    ]
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        price = r.get("min_divine", 0)
        for label, lo, hi in brackets:
            if lo <= price < hi:
                counts[label] += 1
                break
        else:
            if price >= 1500:
                counts["1000-1500d"] += 1

    total = len(records)
    max_count = max(counts.values()) if counts else 1

    lines = [
        "## Price Bracket Coverage",
        "",
        "| Bracket | Count | Pct | |",
        "|---------|------:|----:|:--|",
    ]
    for label, _, _ in brackets:
        c = counts.get(label, 0)
        lines.append(f"| {label} | {c:,} | {pct(c, total)} | `{bar(c, max_count, 15)}` |")
    lines.append("")
    return "\n".join(lines)


def section_shard(shard_path: Path) -> str:
    shard = load_shard(shard_path)
    if shard is None:
        return ""
    sample_count = shard.get("sample_count", len(shard.get("samples", [])))
    generated = shard.get("generated_at", "unknown")
    size = shard_path.stat().st_size

    lines = [
        "## Shard Stats",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| Samples (post-filter) | {sample_count:,} |",
        f"| Generated at | {generated} |",
        f"| File size | {fmt_size(size)} |",
        "",
    ]
    return "\n".join(lines)


def section_state(directory: Path) -> str:
    states = load_state_files(directory)
    if not states:
        return ""

    lines = [
        "## Skip/Filter Stats",
        "",
        "| State File | Completed | Dead Combos | Samples |",
        "|------------|----------:|------------:|--------:|",
    ]
    for s in states:
        name = s.get("_filename", "?")
        completed = len(s.get("completed_queries", []))
        dead = len(s.get("dead_combos", []))
        samples = s.get("total_samples", 0)
        lines.append(f"| {name} | {completed} | {dead} | {samples:,} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate harvest summary markdown")
    parser.add_argument("dir", nargs="?", default=None, help="Cache directory with JSONL files")
    parser.add_argument("--dir", dest="dir_flag", default=None, help="Cache directory (flag form)")
    parser.add_argument("--shard", default=None, help="Path to compressed shard file")
    args = parser.parse_args()

    cache_dir = args.dir_flag or args.dir
    if not cache_dir:
        print("Error: cache directory required", file=sys.stderr)
        sys.exit(1)

    cache_path = Path(cache_dir).expanduser()
    if not cache_path.is_dir():
        print(f"Error: {cache_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    records = load_jsonl_files(cache_path)

    parts = [
        section_header(records),
        section_harvester_breakdown(cache_path),
        section_category(records),
        section_grade(records),
        section_price_brackets(records),
    ]

    if args.shard:
        shard_path = Path(args.shard).expanduser()
        parts.append(section_shard(shard_path))

    parts.append(section_state(cache_path))

    print("\n".join(parts))


if __name__ == "__main__":
    main()
