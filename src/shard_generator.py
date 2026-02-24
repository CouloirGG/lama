"""
LAMA - Calibration Shard Generator

Pipeline: raw harvester JSONL -> quality filters -> dedup -> compact gzipped JSON shard.

Shard format (gzipped JSON):
{
  "version": 4,
  "league": "Fate of the Vaal",
  "generated_at": "2026-02-18T12:00:00Z",
  "sample_count": 5432,
  "mod_index": [["IncreasedLife", 2.0], ["FireResist", 0.3], ...],
  "base_index": ["Astral Plate", "Coral Ring", ...],
  "samples": [
    {"s": 0.584, "g": 2, "p": 4.5, "c": "Rings", "d": 1.0, "f": 1.0, "v": 1.02, "t": 2, "n": 5, "m": [0, 1], "b": 3}
  ]
}

Fields: s=score, g=grade_num, p=divine_price, c=item_class, d=dps_factor, f=defense_factor,
        v=somv_factor, t=top_tier_count, n=mod_count, m=mod_group_indices (into mod_index),
        b=base_type_index (into base_index)

Usage:
    python shard_generator.py --input harvester_output.jsonl --output shard.json.gz
    python shard_generator.py --input "cache/calibration_shard_*.jsonl" --output shard.json.gz
    python shard_generator.py --validate --input shard.json.gz
"""

import argparse
import glob
import gzip
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Numeric grade mapping (matches calibration.py)
_GRADE_NUM = {"S": 4, "A": 3, "B": 2, "C": 1, "JUNK": 0}
_GRADE_FROM_NUM = {v: k for k, v in _GRADE_NUM.items()}

# Quality filter thresholds
MAX_PRICE_DIVINE = 1500.0
MIN_PRICE_DIVINE = 0.01
OUTLIER_IQR_MULTIPLIER = 3.0  # IQR fence multiplier in log-price space
OUTLIER_MAX_DROP_RATE = 0.20  # sanity cap — fail if outlier removal exceeds 20%


def load_raw_records(input_paths: List[str]) -> List[dict]:
    """Load all JSONL records from one or more input files (supports globs)."""
    records = []
    expanded = []
    for p in input_paths:
        expanded.extend(glob.glob(p))

    if not expanded:
        print(f"ERROR: No input files found matching: {input_paths}")
        sys.exit(1)

    for path in expanded:
        print(f"  Reading: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        records.append(rec)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"  Warning: could not read {path}: {e}")

    return records


def quality_filter(records: List[dict]) -> Tuple[List[dict], dict]:
    """Apply quality filters, return (filtered_records, stats)."""
    stats = {
        "input": len(records),
        "no_score": 0,
        "no_price": 0,
        "price_too_high": 0,
        "price_too_low": 0,
        "estimate": 0,
        "passed": 0,
    }
    filtered = []

    for rec in records:
        score = rec.get("score")
        divine = rec.get("min_divine")

        if score is None:
            stats["no_score"] += 1
            continue
        if divine is None:
            stats["no_price"] += 1
            continue
        if divine > MAX_PRICE_DIVINE:
            stats["price_too_high"] += 1
            continue
        if divine <= MIN_PRICE_DIVINE:
            stats["price_too_low"] += 1
            continue
        if rec.get("estimate", False):
            stats["estimate"] += 1
            continue

        filtered.append(rec)
        stats["passed"] += 1

    return filtered, stats


def remove_outliers(records: List[dict]) -> Tuple[List[dict], int]:
    """Remove price outliers within each (grade, item_class) group.

    Uses IQR fences in log-price space — statistically sound for
    log-normally distributed price data and adapts to each group's
    actual spread.  Groups with < 5 records are kept as-is (IQR needs
    a reasonable sample size).
    """
    from collections import defaultdict

    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for rec in records:
        grade = rec.get("grade", "C")
        item_class = rec.get("item_class", "")
        groups[(grade, item_class)].append(rec)

    kept = []
    removed = 0
    for key, group in groups.items():
        if len(group) < 5:
            kept.extend(group)
            continue

        log_prices = sorted(math.log(r["min_divine"]) for r in group
                            if r["min_divine"] > 0)
        if len(log_prices) < 5:
            kept.extend(group)
            continue

        n = len(log_prices)
        q1 = log_prices[n // 4]
        q3 = log_prices[(3 * n) // 4]
        iqr = q3 - q1
        lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
        upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr

        for rec in group:
            price = rec["min_divine"]
            if price <= 0:
                removed += 1
                continue
            lp = math.log(price)
            if lp < lower or lp > upper:
                removed += 1
            else:
                kept.append(rec)

    return kept, removed


def dedup_records(records: List[dict]) -> Tuple[List[dict], int]:
    """Remove duplicate (score, price, item_class, grade) entries."""
    seen = set()
    deduped = []
    dup_count = 0

    for rec in records:
        key = (
            round(rec["score"], 3),
            round(rec["min_divine"], 2),
            rec.get("item_class", ""),
            rec.get("grade", ""),
        )
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)
        deduped.append(rec)

    return deduped, dup_count


def compact_record(rec: dict, mod_to_idx: dict = None, base_to_idx: dict = None) -> dict:
    """Convert a full record to compact shard format."""
    compact = {
        "s": round(rec["score"], 3),
        "g": _GRADE_NUM.get(rec.get("grade", "C"), 1),
        "p": round(rec["min_divine"], 4),
        "c": rec.get("item_class", ""),
        "d": round(rec.get("dps_factor", 1.0), 3),
        "f": round(rec.get("defense_factor", 1.0), 3),
        "v": round(rec.get("somv_factor", 1.0), 3),
        "t": rec.get("top_tier_count", 0),
        "n": rec.get("mod_count", 4),
    }
    if mod_to_idx:
        groups = rec.get("mod_groups", [])
        if groups:
            compact["m"] = sorted(set(
                mod_to_idx[g] for g in groups if g in mod_to_idx
            ))
    if base_to_idx:
        bt = rec.get("base_type", "")
        if bt and bt in base_to_idx:
            compact["b"] = base_to_idx[bt]
    return compact


def generate_shard(records: List[dict], league: str, output_path: str):
    """Generate a compact gzipped JSON shard from filtered records."""
    print(f"\nGenerating shard: {output_path}")

    # Quality filter
    filtered, qstats = quality_filter(records)
    print(f"  Quality filter: {qstats['input']} -> {qstats['passed']} "
          f"(dropped: {qstats['no_score']} no_score, {qstats['no_price']} no_price, "
          f"{qstats['price_too_high']} price_cap, {qstats['price_too_low']} too_low, "
          f"{qstats['estimate']} estimates)")

    # Outlier removal
    cleaned, outlier_count = remove_outliers(filtered)
    drop_rate = outlier_count / len(filtered) if filtered else 0
    print(f"  Outlier removal (IQR, log-price): {len(filtered)} -> {len(cleaned)} "
          f"({outlier_count} outliers removed, {drop_rate:.1%}, fence={OUTLIER_IQR_MULTIPLIER}x IQR)")
    if drop_rate > OUTLIER_MAX_DROP_RATE:
        print(f"  ERROR: Outlier removal dropped {drop_rate:.1%} of records "
              f"(cap is {OUTLIER_MAX_DROP_RATE:.0%}). This suggests a bug in the "
              f"outlier algorithm, not bad data. Aborting.")
        sys.exit(1)

    # Dedup
    deduped, dup_count = dedup_records(cleaned)
    print(f"  Dedup: {len(cleaned)} -> {len(deduped)} ({dup_count} duplicates removed)")

    # Build mod group index from all records
    all_mod_groups = set()
    for rec in deduped:
        for g in rec.get("mod_groups", []):
            if g:
                all_mod_groups.add(g)

    sorted_groups = sorted(all_mod_groups)
    mod_to_idx = {g: i for i, g in enumerate(sorted_groups)}

    # Build mod_index with weights: [[group_name, weight], ...]
    mod_index = []
    if sorted_groups:
        try:
            from mod_database import _get_weight_for_group
            mod_index = [[g, _get_weight_for_group(g) or 0.3]
                         for g in sorted_groups]
        except ImportError:
            mod_index = [[g, 0.3] for g in sorted_groups]

    # Build base type index from all records
    all_base_types = set()
    for rec in deduped:
        bt = rec.get("base_type", "")
        if bt:
            all_base_types.add(bt)

    sorted_bases = sorted(all_base_types)
    base_to_idx = {bt: i for i, bt in enumerate(sorted_bases)}

    # Compact format
    samples = [compact_record(r, mod_to_idx, base_to_idx) for r in deduped]

    # Train learned weights from the deduped records
    learned_weights_dict = None
    try:
        from weight_learner import train_weights
        training_records = []
        for rec in deduped:
            training_records.append({
                "c": rec.get("item_class", ""),
                "g": _GRADE_NUM.get(rec.get("grade", "C"), 1),
                "p": rec.get("min_divine", 0),
                "t": rec.get("top_tier_count", 0),
                "n": rec.get("mod_count", 4),
                "d": rec.get("dps_factor", 1.0),
                "f": rec.get("defense_factor", 1.0),
                "v": rec.get("somv_factor", 1.0),
                "mod_groups_resolved": [g for g in rec.get("mod_groups", []) if g],
                "base_type_resolved": rec.get("base_type", ""),
            })
        lw = train_weights(training_records)
        if lw._models:
            learned_weights_dict = lw.to_dict()
            print(f"\n  {lw.summary()}")
    except ImportError:
        print("  Note: numpy not available, skipping regression training")
    except Exception as e:
        print(f"  Warning: regression training failed: {e}")

    # Detect league from records if not specified
    if not league:
        leagues = set(r.get("league", "") for r in records if r.get("league"))
        league = leagues.pop() if len(leagues) == 1 else "unknown"

    shard = {
        "version": 5,
        "league": league,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(samples),
        "samples": samples,
    }
    if mod_index:
        shard["mod_index"] = mod_index
    if sorted_bases:
        shard["base_index"] = sorted_bases
    if learned_weights_dict:
        shard["learned_weights"] = learned_weights_dict

    # Write gzipped JSON
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(shard, f, separators=(",", ":"))

    size_kb = out.stat().st_size / 1024
    print(f"  Written: {out} ({size_kb:.1f} KB, {len(samples)} samples)")

    # Class breakdown
    by_class: Dict[str, int] = {}
    for s in samples:
        c = s["c"] or "unknown"
        by_class[c] = by_class.get(c, 0) + 1
    print(f"\n  Per-class sample counts:")
    for cls in sorted(by_class, key=by_class.get, reverse=True):
        print(f"    {cls:20s}: {by_class[cls]:5d}")

    # Grade breakdown
    by_grade: Dict[str, int] = {}
    for s in samples:
        g = _GRADE_FROM_NUM.get(s["g"], "?")
        by_grade[g] = by_grade.get(g, 0) + 1
    print(f"\n  Per-grade sample counts:")
    for g in ["S", "A", "B", "C", "JUNK"]:
        print(f"    {g:5s}: {by_grade.get(g, 0):5d}")

    return shard


# ─── Validation ──────────────────────────────────────

def validate_shard(shard_path: str, seed: int = 42):
    """Hold-out validation: split 80/20, measure k-NN accuracy.

    Target: >=70% of estimates within 2x of actual price.
    """
    print(f"\nValidating shard: {shard_path}")

    # Load shard
    path = Path(shard_path)
    if path.suffix == ".gz" or str(path).endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            shard = json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            shard = json.load(f)

    samples = shard.get("samples", [])
    if not samples:
        print("  ERROR: No samples in shard")
        return

    print(f"  Shard: {shard.get('league', '?')}, {len(samples)} samples")

    # Shuffle and split 80/20
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    split = int(len(indices) * 0.8)
    train_idx = indices[:split]
    test_idx = indices[split:]

    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Build k-NN from training set
    from calibration import CalibrationEngine
    engine = CalibrationEngine()

    # Load mod index (v3+ shards)
    mod_index = shard.get("mod_index", [])
    idx_to_group = {}
    weights = {}
    for i, entry in enumerate(mod_index):
        if isinstance(entry, list) and len(entry) >= 2:
            idx_to_group[i] = entry[0]
            weights[entry[0]] = entry[1]
    if weights:
        engine.set_mod_weights(weights)

    # Load base type index (v4+ shards)
    base_index = shard.get("base_index", [])

    def _sample_mod_groups(s):
        return [idx_to_group[idx] for idx in s.get("m", []) if idx in idx_to_group]

    def _sample_base_type(s):
        idx = s.get("b")
        return base_index[idx] if (idx is not None and idx < len(base_index)) else ""

    for i in train_idx:
        s = samples[i]
        engine._insert(
            score=s["s"],
            divine=s["p"],
            item_class=s.get("c", ""),
            grade_num=s.get("g", 1),
            dps_factor=s.get("d", 1.0),
            defense_factor=s.get("f", 1.0),
            top_tier_count=s.get("t", 0),
            mod_count=s.get("n", 4),
            mod_groups=_sample_mod_groups(s),
            base_type=_sample_base_type(s),
        )

    # Test on holdout
    within_2x = 0
    within_3x = 0
    total_tested = 0
    errors_by_class: Dict[str, List[float]] = {}
    errors_by_grade: Dict[str, List[float]] = {}

    for i in test_idx:
        s = samples[i]
        actual = s["p"]
        item_class = s.get("c", "")
        grade_num = s.get("g", 1)
        grade = _GRADE_FROM_NUM.get(grade_num, "C")

        est = engine.estimate(s["s"], item_class, grade=grade,
                              dps_factor=s.get("d", 1.0),
                              defense_factor=s.get("f", 1.0),
                              top_tier_count=s.get("t", 0),
                              mod_count=s.get("n", 4),
                              mod_groups=_sample_mod_groups(s),
                              base_type=_sample_base_type(s))
        if est is None:
            continue

        total_tested += 1
        ratio = max(est / actual, actual / est) if actual > 0 else float("inf")

        if ratio <= 2.0:
            within_2x += 1
        if ratio <= 3.0:
            within_3x += 1

        if item_class not in errors_by_class:
            errors_by_class[item_class] = []
        errors_by_class[item_class].append(ratio)

        if grade not in errors_by_grade:
            errors_by_grade[grade] = []
        errors_by_grade[grade].append(ratio)

    if total_tested == 0:
        print("  ERROR: No estimates produced (insufficient samples per class?)")
        return

    pct_2x = within_2x / total_tested * 100
    pct_3x = within_3x / total_tested * 100
    target_met = pct_2x >= 70

    print(f"\n  Results ({total_tested} items estimated):")
    print(f"    Within 2x: {within_2x}/{total_tested} ({pct_2x:.1f}%) "
          f"{'PASS' if target_met else 'FAIL'} (target: >=70%)")
    print(f"    Within 3x: {within_3x}/{total_tested} ({pct_3x:.1f}%)")

    # Per-class breakdown
    print(f"\n  Per-class accuracy (within 2x):")
    class_results = []
    for cls, ratios in sorted(errors_by_class.items()):
        n = len(ratios)
        n_2x = sum(1 for r in ratios if r <= 2.0)
        pct = n_2x / n * 100 if n > 0 else 0
        median_ratio = sorted(ratios)[len(ratios) // 2]
        class_results.append((cls, n, pct, median_ratio))

    for cls, n, pct, median in sorted(class_results, key=lambda x: -x[2]):
        status = "OK" if pct >= 70 else "!!"
        print(f"    {status} {cls:20s}: {pct:5.1f}% ({n:4d} samples, "
              f"median error: {median:.2f}x)")

    # Per-grade breakdown
    print(f"\n  Per-grade accuracy (within 2x):")
    for g in ["S", "A", "B", "C", "JUNK"]:
        ratios = errors_by_grade.get(g, [])
        if not ratios:
            print(f"    -- {g:5s}: no samples")
            continue
        n = len(ratios)
        n_2x = sum(1 for r in ratios if r <= 2.0)
        pct = n_2x / n * 100
        median_ratio = sorted(ratios)[len(ratios) // 2]
        status = "OK" if pct >= 50 else "!!"
        print(f"    {status} {g:5s}: {pct:5.1f}% ({n:4d} samples, "
              f"median error: {median_ratio:.2f}x)")

    # ── Regression validation ─────────────────────────────
    lw_data = shard.get("learned_weights")
    if lw_data:
        try:
            from weight_learner import LearnedWeights
            lw = LearnedWeights.from_dict(lw_data)
            print(f"\n  Regression validation ({len(lw._models)} models):")

            reg_within_2x = 0
            reg_within_3x = 0
            reg_total = 0
            reg_by_class: Dict[str, List[float]] = {}

            for i in test_idx:
                s = samples[i]
                actual = s["p"]
                item_class = s.get("c", "")
                grade_num = s.get("g", 1)
                mod_groups = _sample_mod_groups(s)
                base_type = _sample_base_type(s)

                est = lw.predict(
                    item_class, mod_groups, base_type,
                    grade_num=grade_num,
                    top_tier_count=s.get("t", 0),
                    mod_count=s.get("n", 4),
                    dps_factor=s.get("d", 1.0),
                    defense_factor=s.get("f", 1.0),
                    somv_factor=s.get("v", 1.0),
                )
                if est is None:
                    continue

                reg_total += 1
                ratio = max(est / actual, actual / est) if actual > 0 else float("inf")
                if ratio <= 2.0:
                    reg_within_2x += 1
                if ratio <= 3.0:
                    reg_within_3x += 1

                if item_class not in reg_by_class:
                    reg_by_class[item_class] = []
                reg_by_class[item_class].append(ratio)

            if reg_total > 0:
                reg_pct_2x = reg_within_2x / reg_total * 100
                reg_pct_3x = reg_within_3x / reg_total * 100
                print(f"    Within 2x: {reg_within_2x}/{reg_total} ({reg_pct_2x:.1f}%)")
                print(f"    Within 3x: {reg_within_3x}/{reg_total} ({reg_pct_3x:.1f}%)")

                print(f"\n  Per-class regression accuracy (within 2x):")
                for cls, ratios in sorted(reg_by_class.items()):
                    n = len(ratios)
                    n_2x = sum(1 for r in ratios if r <= 2.0)
                    pct = n_2x / n * 100 if n > 0 else 0
                    median_ratio = sorted(ratios)[len(ratios) // 2]
                    print(f"    {cls:20s}: {pct:5.1f}% ({n:4d} samples, "
                          f"median error: {median_ratio:.2f}x)")
            else:
                print("    No regression estimates produced")
        except Exception as e:
            print(f"    Regression validation failed: {e}")


# ─── CLI Entry Point ─────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate calibration shards from harvester output")
    parser.add_argument("--input", "-i", nargs="+", required=True,
                        help="Input JSONL file(s) from harvester (supports globs)")
    parser.add_argument("--output", "-o",
                        help="Output shard file path (.json.gz)")
    parser.add_argument("--league",
                        help="League name (auto-detected from records if omitted)")
    parser.add_argument("--validate", action="store_true",
                        help="Run hold-out validation on a shard file")

    args = parser.parse_args()

    if args.validate:
        # Validate mode: input is a shard file
        for inp in args.input:
            validate_shard(inp)
        return

    # Generate mode: input is JSONL, output is shard
    if not args.output:
        print("ERROR: --output is required for shard generation")
        sys.exit(1)

    print("Loading raw records...")
    records = load_raw_records(args.input)
    print(f"  Loaded {len(records)} records")

    generate_shard(records, args.league or "", args.output)


if __name__ == "__main__":
    main()
