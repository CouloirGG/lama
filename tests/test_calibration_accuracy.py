"""Comprehensive calibration accuracy diagnostic.

80/20 holdout validation using ALL available shard features.
This is the benchmark for iterating toward 75% within-2x accuracy.
"""

import gzip
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import pytest
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from calibration import CalibrationEngine

_GRADE_FROM_NUM = {4: "S", 3: "A", 2: "B", 1: "C", 0: "JUNK"}
SHARD_PATH = PROJECT_ROOT / "resources" / "calibration_shard.json.gz"


def _load_shard_raw():
    """Load raw shard data with mod_index and base_index for reconstruction."""
    if not SHARD_PATH.exists():
        pytest.skip("Bundled shard not found")
    with gzip.open(SHARD_PATH, "rt", encoding="utf-8") as f:
        shard = json.load(f)
    samples = shard.get("samples", [])
    if len(samples) < 100:
        pytest.skip(f"Shard too small ({len(samples)} samples)")

    # Build index lookups
    mod_index = shard.get("mod_index", [])
    idx_to_group = {}
    mod_weights = {}
    for i, entry in enumerate(mod_index):
        if isinstance(entry, list) and len(entry) >= 2:
            idx_to_group[i] = entry[0]
            mod_weights[entry[0]] = entry[1]

    base_index = shard.get("base_index", [])

    return shard, samples, idx_to_group, mod_weights, base_index


def _reconstruct_sample(s, idx_to_group, base_index):
    """Reconstruct full features from a compact shard sample."""
    m_indices = s.get("m", [])
    mod_groups = [idx_to_group[idx] for idx in m_indices if idx in idx_to_group]

    bt_idx = s.get("b", s.get("bt"))
    base_type = ""
    if bt_idx is not None and bt_idx < len(base_index):
        base_type = base_index[bt_idx]

    # Reconstruct mod_tiers
    mt_arr = s.get("mt", [])
    mod_tiers = {}
    if mt_arr and len(mt_arr) == len(m_indices):
        for j, idx in enumerate(m_indices):
            if idx in idx_to_group and mt_arr[j] > 0:
                mod_tiers[idx_to_group[idx]] = mt_arr[j]

    # Reconstruct mod_rolls
    mr_arr = s.get("mr", [])
    mod_rolls = {}
    if mr_arr and len(mr_arr) == len(m_indices):
        for j, idx in enumerate(m_indices):
            if idx in idx_to_group and mr_arr[j] >= 0:
                mod_rolls[idx_to_group[idx]] = mr_arr[j]

    return {
        "score": s["s"],
        "divine": s["p"],
        "item_class": s.get("c", ""),
        "grade_num": s.get("g", 1),
        "dps_factor": s.get("d", 1.0),
        "defense_factor": s.get("f", 1.0),
        "top_tier_count": s.get("t", 0),
        "mod_count": s.get("n", 4),
        "somv_factor": s.get("v", 1.0),
        "mod_groups": mod_groups,
        "base_type": base_type,
        "mod_tiers": mod_tiers,
        "mod_rolls": mod_rolls,
        "pdps": s.get("pd", 0.0),
        "edps": s.get("ed", 0.0),
        "sale_confidence": s.get("sc", 1.0),
    }


def _build_class_features(items, all_mods, all_bases):
    """Build feature matrix and names for a set of items in one class."""
    import numpy as np

    core_features = [
        "grade_num", "score", "top_tier_count", "mod_count",
        "dps_factor", "defense_factor", "somv_factor",
        "tier_score", "best_tier", "avg_tier",
        "pdps", "edps", "total_dps",
    ]
    mod_features = sorted(f"mod:{g}" for g in all_mods)
    base_features = sorted(f"base:{b}" for b in all_bases)
    feature_names = core_features + mod_features + base_features

    X = []
    y = []
    for r in items:
        mt = r.get("mod_tiers", {})
        mr = r.get("mod_rolls", {})
        mg_set = set(r["mod_groups"])
        tiers = [t for t in mt.values() if t > 0]
        tier_score = sum(1.0 / t for t in tiers) if tiers else 0.0
        best_tier = min(tiers) if tiers else 0
        avg_tier = sum(tiers) / len(tiers) if tiers else 0.0
        pdps = r.get("pdps", 0.0)
        edps = r.get("edps", 0.0)
        row = [
            r["grade_num"], r["score"], r["top_tier_count"], r["mod_count"],
            r["dps_factor"], r["defense_factor"], r["somv_factor"],
            tier_score, best_tier, avg_tier,
            pdps, edps, pdps + edps,
        ]
        for fn in mod_features:
            g = fn[4:]
            if g in mg_set:
                tier = mt.get(g, 0)
                rq = mr.get(g, -1)
                if rq >= 0 and tier > 0:
                    row.append(rq * (1.0 / tier))
                elif tier > 0:
                    row.append(0.5 * (1.0 / tier))
                else:
                    row.append(0.25)
            else:
                row.append(0.0)
        for fn in base_features:
            b = fn[5:]
            row.append(1.0 if r["base_type"] == b else 0.0)

        X.append(row)
        y.append(math.log(r["divine"]))

    return feature_names, np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _train_gbm_models(train_samples, idx_to_group, mod_weights, base_index,
                      min_samples=50):
    """Train per-class GBM models using sklearn, return serialized dict."""
    from sklearn.ensemble import GradientBoostingRegressor
    import numpy as np

    by_class = defaultdict(list)
    for s in train_samples:
        r = _reconstruct_sample(s, idx_to_group, base_index)
        by_class[r["item_class"]].append(r)

    gbm_models = {}
    for item_class, items in by_class.items():
        if len(items) < min_samples:
            continue

        all_mods = set()
        all_bases = set()
        for r in items:
            all_mods.update(r["mod_groups"])
            if r["base_type"]:
                all_bases.add(r["base_type"])

        feature_names, X, y = _build_class_features(items, all_mods, all_bases)

        gbr = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.8, min_samples_leaf=5,
        )
        gbr.fit(X, y)

        trees_data = []
        for tree_est in gbr.estimators_:
            tree = tree_est[0].tree_
            trees_data.append({
                "feature": tree.feature.tolist(),
                "threshold": tree.threshold.tolist(),
                "left": tree.children_left.tolist(),
                "right": tree.children_right.tolist(),
                "value": [float(v[0][0]) for v in tree.value],
            })

        gbm_models[item_class] = {
            "feature_names": feature_names,
            "trees": trees_data,
            "base_prediction": float(gbr.init_.constant_[0][0]),
            "learning_rate": gbr.learning_rate,
        }

    return gbm_models


def _run_holdout(use_all_features=True, seed=42, train_ratio=0.8, verbose=False,
                 use_gbm="shard", retrain_gbm=False):
    """Run 80/20 holdout and return accuracy metrics.

    use_gbm: "shard" = load pre-built models, "none" = skip GBM, "retrain" = train fresh
    Returns dict with overall and per-class accuracy stats.
    """
    shard, samples, idx_to_group, mod_weights, base_index = _load_shard_raw()

    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    split = int(len(indices) * train_ratio)
    train_idx = indices[:split]
    test_idx = indices[split:]

    engine = CalibrationEngine()
    if use_all_features:
        engine.set_mod_weights(mod_weights)

    # GBM model loading strategy
    if use_gbm == "shard":
        gbm_data = shard.get("gbm_models")
        if gbm_data:
            engine._gbm_models = dict(gbm_data)
    # use_gbm == "none": skip GBM entirely (k-NN only)
    # use_gbm == "retrain": handled after training data insertion

    # Load price tables from shard
    pt_data = shard.get("price_tables")
    if pt_data:
        engine._price_tables = dict(pt_data)

    # Insert training data
    for i in train_idx:
        s = samples[i]
        r = _reconstruct_sample(s, idx_to_group, base_index)
        if use_all_features:
            engine._insert(
                score=r["score"], divine=r["divine"],
                item_class=r["item_class"], grade_num=r["grade_num"],
                dps_factor=r["dps_factor"], defense_factor=r["defense_factor"],
                top_tier_count=r["top_tier_count"], mod_count=r["mod_count"],
                mod_groups=r["mod_groups"], base_type=r["base_type"],
                mod_tiers=r["mod_tiers"], somv_factor=r["somv_factor"],
                mod_rolls=r["mod_rolls"],
                pdps=r["pdps"], edps=r["edps"],
                sale_confidence=r["sale_confidence"],
            )
        else:
            engine._insert(
                score=r["score"], divine=r["divine"],
                item_class=r["item_class"], grade_num=r["grade_num"],
                dps_factor=r["dps_factor"], defense_factor=r["defense_factor"],
                top_tier_count=r["top_tier_count"], mod_count=r["mod_count"],
            )

    # Retrain GBM on training split if requested
    if use_gbm == "retrain" and use_all_features:
        engine._gbm_models = _train_gbm_models(
            [samples[i] for i in train_idx], idx_to_group, mod_weights, base_index)
        if verbose:
            print(f"  Retrained GBM for {len(engine._gbm_models)} classes")

    # Test
    within_2x = 0
    within_1_5x = 0
    total_tested = 0
    log_errors = []
    per_class = defaultdict(lambda: {"within_2x": 0, "total": 0, "errors": []})
    per_grade = defaultdict(lambda: {"within_2x": 0, "total": 0})
    per_price_bucket = defaultdict(lambda: {"within_2x": 0, "total": 0})
    worst_misses = []

    for i in test_idx:
        s = samples[i]
        r = _reconstruct_sample(s, idx_to_group, base_index)
        actual = r["divine"]
        grade = _GRADE_FROM_NUM.get(r["grade_num"], "C")

        if use_all_features:
            est = engine.estimate(
                r["score"], r["item_class"],
                grade=grade,
                dps_factor=r["dps_factor"],
                defense_factor=r["defense_factor"],
                top_tier_count=r["top_tier_count"],
                mod_count=r["mod_count"],
                mod_groups=r["mod_groups"],
                base_type=r["base_type"],
                somv_factor=r["somv_factor"],
                mod_tiers=r["mod_tiers"],
                mod_rolls=r["mod_rolls"],
                pdps=r["pdps"],
                edps=r["edps"],
            )
        else:
            est = engine.estimate(
                r["score"], r["item_class"],
                grade=grade,
                dps_factor=r["dps_factor"],
                defense_factor=r["defense_factor"],
                top_tier_count=r["top_tier_count"],
                mod_count=r["mod_count"],
            )

        if est is None:
            continue

        total_tested += 1
        ratio = max(est / actual, actual / est) if actual > 0 else float("inf")
        log_err = abs(math.log(est) - math.log(actual)) if actual > 0 and est > 0 else 10.0
        log_errors.append(log_err)

        hit_2x = ratio <= 2.0
        hit_1_5x = ratio <= 1.5
        if hit_2x:
            within_2x += 1
        if hit_1_5x:
            within_1_5x += 1

        cls = r["item_class"]
        per_class[cls]["total"] += 1
        if hit_2x:
            per_class[cls]["within_2x"] += 1
        per_class[cls]["errors"].append(log_err)

        gn = r["grade_num"]
        per_grade[gn]["total"] += 1
        if hit_2x:
            per_grade[gn]["within_2x"] += 1

        # Price buckets
        if actual < 1:
            bucket = "<1d"
        elif actual < 5:
            bucket = "1-5d"
        elif actual < 20:
            bucket = "5-20d"
        elif actual < 50:
            bucket = "20-50d"
        else:
            bucket = "50+d"
        per_price_bucket[bucket]["total"] += 1
        if hit_2x:
            per_price_bucket[bucket]["within_2x"] += 1

        if not hit_2x:
            worst_misses.append((ratio, actual, est, cls, grade,
                                 r.get("mod_groups", [])[:3]))

    pct_2x = within_2x / total_tested * 100 if total_tested else 0
    pct_1_5x = within_1_5x / total_tested * 100 if total_tested else 0
    median_log_err = statistics.median(log_errors) if log_errors else 0
    mean_log_err = statistics.mean(log_errors) if log_errors else 0

    result = {
        "total_tested": total_tested,
        "within_2x": within_2x,
        "within_1_5x": within_1_5x,
        "pct_2x": pct_2x,
        "pct_1_5x": pct_1_5x,
        "median_log_error": median_log_err,
        "mean_log_error": mean_log_err,
        "per_class": dict(per_class),
        "per_grade": dict(per_grade),
        "per_price_bucket": dict(per_price_bucket),
        "worst_misses": sorted(worst_misses, reverse=True)[:20],
    }

    if verbose:
        _print_report(result, use_all_features)

    return result


def _print_report(r, use_all_features, label=None):
    """Print a detailed accuracy report."""
    mode = label or ("ALL FEATURES" if use_all_features else "BASIC ONLY")
    print(f"\n{'='*60}")
    print(f"  CALIBRATION ACCURACY REPORT ({mode})")
    print(f"{'='*60}")
    print(f"  Total tested: {r['total_tested']}")
    print(f"  Within 2x:    {r['within_2x']}/{r['total_tested']} "
          f"= {r['pct_2x']:.1f}%")
    print(f"  Within 1.5x:  {r['within_1_5x']}/{r['total_tested']} "
          f"= {r['pct_1_5x']:.1f}%")
    print(f"  Median log error: {r['median_log_error']:.3f}")
    print(f"  Mean log error:   {r['mean_log_error']:.3f}")

    print(f"\n  Per-class accuracy (within 2x):")
    for cls in sorted(r["per_class"], key=lambda c: r["per_class"][c]["total"],
                      reverse=True):
        d = r["per_class"][cls]
        pct = d["within_2x"] / d["total"] * 100 if d["total"] else 0
        med_err = statistics.median(d["errors"]) if d["errors"] else 0
        print(f"    {cls:20s}: {pct:5.1f}% ({d['within_2x']:4d}/{d['total']:4d}) "
              f"  med_log_err={med_err:.3f}")

    print(f"\n  Per-grade accuracy (within 2x):")
    grade_names = {0: "JUNK", 1: "C", 2: "B", 3: "A", 4: "S"}
    for gn in sorted(r["per_grade"]):
        d = r["per_grade"][gn]
        pct = d["within_2x"] / d["total"] * 100 if d["total"] else 0
        print(f"    {grade_names.get(gn, '?'):5s}: {pct:5.1f}% "
              f"({d['within_2x']:4d}/{d['total']:4d})")

    print(f"\n  Per-price-bucket accuracy (within 2x):")
    for bucket in ["<1d", "1-5d", "5-20d", "20-50d", "50+d"]:
        d = r["per_price_bucket"].get(bucket, {"within_2x": 0, "total": 0})
        pct = d["within_2x"] / d["total"] * 100 if d["total"] else 0
        print(f"    {bucket:8s}: {pct:5.1f}% ({d['within_2x']:4d}/{d['total']:4d})")

    print(f"\n  Worst 10 misses (ratio, actual, est, class, grade, mods):")
    for ratio, actual, est, cls, grade, mods in r["worst_misses"][:10]:
        mods_str = ",".join(mods) if mods else "-"
        print(f"    {ratio:6.1f}x | actual={actual:7.1f} est={est:7.1f} | "
              f"{cls} ({grade}) | {mods_str}")
    print(f"{'='*60}\n")


# ── Tests ─────────────────────────────────────────────────


def test_accuracy_with_all_features():
    """80/20 holdout with ALL features. Target: 75% within 2x."""
    r = _run_holdout(use_all_features=True, verbose=True, use_gbm="none")
    assert r["total_tested"] > 0, "No estimates produced"
    # Current baseline - will be increased as we improve
    assert r["pct_2x"] >= 20.0, (
        f"Accuracy {r['pct_2x']:.1f}% below 20% floor "
        f"({r['within_2x']}/{r['total_tested']})"
    )


def test_accuracy_retrained_gbm():
    """80/20 holdout with retrained GBM for all classes."""
    r = _run_holdout(use_all_features=True, verbose=True, use_gbm="retrain")
    assert r["total_tested"] > 0, "No estimates produced"
    assert r["pct_2x"] >= 20.0, (
        f"Accuracy {r['pct_2x']:.1f}% below 20% floor "
        f"({r['within_2x']}/{r['total_tested']})"
    )


def test_accuracy_baseline_comparison():
    """Compare k-NN-only vs retrained-GBM to quantify GBM value."""
    r_knn = _run_holdout(use_all_features=True, verbose=True, use_gbm="none")
    r_gbm = _run_holdout(use_all_features=True, verbose=True, use_gbm="retrain")

    print(f"\n  GBM IMPACT:")
    print(f"    k-NN only:     {r_knn['pct_2x']:.1f}%")
    print(f"    Retrained GBM: {r_gbm['pct_2x']:.1f}%")
    print(f"    Improvement:   +{r_gbm['pct_2x'] - r_knn['pct_2x']:.1f}%")


if __name__ == "__main__":
    # Run as script for quick diagnostics
    print("Running accuracy diagnostic...")
    _run_holdout(use_all_features=True, verbose=True, use_gbm="none")
    _run_holdout(use_all_features=True, verbose=True, use_gbm="retrain")
