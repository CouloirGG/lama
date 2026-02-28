"""
LAMA - Accuracy Lab

Standalone experiment harness for testing calibration accuracy improvements.
Loads JSONL data directly, splits 80/20, trains models, measures accuracy.

Usage:
    python accuracy_lab.py                         # run all experiments
    python accuracy_lab.py --experiment baseline    # run specific experiment
    python accuracy_lab.py --input "path/*.jsonl"   # custom data path
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shard_generator import (
    load_raw_records, quality_filter, remove_outliers, dedup_records,
    _enrich_record, _compute_tier_aggregates, _GRADE_NUM, _GRADE_FROM_NUM,
)
from weight_learner import compute_archetype_scores

# Default data location
DEFAULT_JSONL_GLOB = os.path.expanduser(
    "~/.poe2-price-overlay/cache/calibration_shard_*.jsonl"
)

SEED = 42


# ── Data loading ──────────────────────────────────────────

def load_and_prepare(input_globs: List[str]) -> List[dict]:
    """Load JSONL, apply quality filters, dedup, enrich."""
    records = load_raw_records(input_globs)
    print(f"  Raw records: {len(records)}")

    filtered, qstats = quality_filter(records)
    print(f"  After quality filter: {len(filtered)} "
          f"(dropped: {qstats['no_score']} no_score, {qstats['no_price']} no_price, "
          f"{qstats['price_too_high']} price_cap, {qstats['price_too_low']} too_low, "
          f"{qstats['estimate']} estimates)")

    cleaned, outlier_count = remove_outliers(filtered)
    print(f"  After outlier removal: {len(cleaned)} ({outlier_count} outliers)")

    deduped, dup_count = dedup_records(cleaned)
    print(f"  After dedup: {len(deduped)} ({dup_count} duplicates)")

    enriched = 0
    for rec in deduped:
        had = bool(rec.get("mod_groups"))
        _enrich_record(rec)
        if not had and rec.get("mod_groups"):
            enriched += 1
    if enriched:
        print(f"  Enriched: {enriched} records gained mod_groups from top_mods")

    return deduped


def split_data(records: List[dict], seed: int = SEED,
               train_frac: float = 0.8) -> Tuple[List[dict], List[dict]]:
    """Deterministic 80/20 train/test split."""
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    split = int(len(indices) * train_frac)
    train = [records[i] for i in indices[:split]]
    test = [records[i] for i in indices[split:]]
    return train, test


# ── Metrics ────────────────────────────────────────────────

def compute_metrics(predictions: List[Tuple[float, float]],
                    records: List[dict] = None) -> dict:
    """Compute accuracy metrics from (estimated, actual) pairs.

    Returns dict with within_2x, within_3x, median_error, per_class breakdown.
    """
    if not predictions:
        return {"within_2x": 0, "within_3x": 0, "total": 0,
                "pct_2x": 0.0, "pct_3x": 0.0, "median_error": 0.0}

    ratios = []
    by_class = defaultdict(list)
    by_grade = defaultdict(list)

    for i, (est, actual) in enumerate(predictions):
        if est is None or actual <= 0:
            continue
        ratio = max(est / actual, actual / est)
        ratios.append(ratio)
        if records and i < len(records):
            cls = records[i].get("item_class", "?")
            grade = records[i].get("grade", "?")
            by_class[cls].append(ratio)
            by_grade[grade].append(ratio)

    if not ratios:
        return {"within_2x": 0, "within_3x": 0, "total": 0,
                "pct_2x": 0.0, "pct_3x": 0.0, "median_error": 0.0}

    total = len(ratios)
    w2x = sum(1 for r in ratios if r <= 2.0)
    w3x = sum(1 for r in ratios if r <= 3.0)
    sorted_ratios = sorted(ratios)
    median = sorted_ratios[len(sorted_ratios) // 2]

    # Per-class breakdown
    class_breakdown = {}
    for cls, cls_ratios in sorted(by_class.items()):
        n = len(cls_ratios)
        n_2x = sum(1 for r in cls_ratios if r <= 2.0)
        cls_sorted = sorted(cls_ratios)
        class_breakdown[cls] = {
            "n": n,
            "pct_2x": n_2x / n * 100,
            "median": cls_sorted[n // 2],
        }

    return {
        "within_2x": w2x,
        "within_3x": w3x,
        "total": total,
        "pct_2x": w2x / total * 100,
        "pct_3x": w3x / total * 100,
        "median_error": median,
        "per_class": class_breakdown,
    }


def print_metrics(name: str, metrics: dict):
    """Pretty-print experiment results."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Total tested: {metrics['total']}")
    print(f"  Within 2x: {metrics['within_2x']}/{metrics['total']} "
          f"({metrics['pct_2x']:.1f}%)")
    print(f"  Within 3x: {metrics['within_3x']}/{metrics['total']} "
          f"({metrics['pct_3x']:.1f}%)")
    print(f"  Median error: {metrics['median_error']:.2f}x")

    if "per_class" in metrics:
        print(f"\n  Per-class (within 2x):")
        for cls, info in sorted(metrics["per_class"].items(),
                                key=lambda x: -x[1]["pct_2x"]):
            print(f"    {cls:20s}: {info['pct_2x']:5.1f}% "
                  f"({info['n']:4d} samples, median {info['median']:.2f}x)")


# ── GBM training helper ──────────────────────────────────

def prepare_gbm_records(records: List[dict],
                        include_new_features: bool = False,
                        include_listing_age: bool = False) -> List[dict]:
    """Convert raw JSONL records to GBM training format.

    Args:
        include_new_features: Add item_level, raw defenses, total_dps/defense
        include_listing_age: Compute listing age in days for sample weighting
    """
    gbm_records = []
    now = time.time()

    for rec in records:
        price = rec.get("min_divine", 0)
        if price <= 0:
            continue

        mod_groups = [g for g in rec.get("mod_groups", []) if g]
        mod_tiers = rec.get("mod_tiers", {})
        ts, bt, at = _compute_tier_aggregates(rec)

        arch = compute_archetype_scores(mod_groups) if mod_groups else {}

        out = {
            "item_class": rec.get("item_class", ""),
            "grade_num": _GRADE_NUM.get(rec.get("grade", "C"), 1),
            "score": rec.get("score", 0),
            "min_divine": price,
            "top_tier_count": rec.get("top_tier_count", 0),
            "mod_count": rec.get("mod_count", 4),
            "dps_factor": rec.get("dps_factor", 1.0),
            "defense_factor": rec.get("defense_factor", 1.0),
            "somv_factor": rec.get("somv_factor", 1.0),
            "tier_score": ts,
            "best_tier": bt,
            "avg_tier": at,
            "coc_score": arch.get("coc_spell", 0.0),
            "es_score": arch.get("ci_es", 0.0),
            "mana_score": arch.get("mom_mana", 0.0),
            "mod_groups": mod_groups,
            "base_type": rec.get("base_type", ""),
            "mod_tiers": mod_tiers,
            "mod_rolls": rec.get("mod_rolls", {}),
            "pdps": rec.get("pdps", 0.0),
            "edps": rec.get("edps", 0.0),
            "sale_confidence": rec.get("sale_confidence", 1.0),
        }

        if include_new_features:
            out["item_level"] = rec.get("item_level", 0)
            out["armour"] = rec.get("armour", 0)
            out["evasion"] = rec.get("evasion", 0)
            out["energy_shield"] = rec.get("energy_shield", 0)
            out["total_dps"] = rec.get("total_dps", 0.0) or (
                rec.get("pdps", 0.0) + rec.get("edps", 0.0))
            out["total_defense"] = rec.get("total_defense", 0) or (
                rec.get("armour", 0) + rec.get("evasion", 0)
                + rec.get("energy_shield", 0))
            out["quality"] = rec.get("quality", 0)
            out["sockets"] = rec.get("sockets", 0)
            out["corrupted"] = 1 if rec.get("corrupted", False) else 0
            out["open_prefixes"] = rec.get("open_prefixes", 0)
            out["open_suffixes"] = rec.get("open_suffixes", 0)

        if include_listing_age:
            listing_ts = rec.get("listing_ts", "")
            if listing_ts:
                try:
                    lt = datetime.fromisoformat(
                        listing_ts.replace("Z", "+00:00"))
                    age_days = (datetime.now(timezone.utc) - lt).total_seconds() / 86400
                    out["listing_age_days"] = max(0, age_days)
                except (ValueError, TypeError):
                    out["listing_age_days"] = 14.0  # default: 2 weeks
            else:
                out["listing_age_days"] = 14.0

        gbm_records.append(out)

    return gbm_records


def gbm_predict(model: dict, rec: dict,
                use_new_features: bool = False) -> Optional[float]:
    """Pure-Python GBM inference on a single record."""
    feature_names = model.get("feature_names", [])
    if not feature_names:
        return None

    features = {
        "grade_num": rec.get("grade_num", 2),
        "score": rec.get("score", 0),
        "top_tier_count": rec.get("top_tier_count", 0),
        "mod_count": rec.get("mod_count", 4),
        "dps_factor": rec.get("dps_factor", 1.0),
        "defense_factor": rec.get("defense_factor", 1.0),
        "somv_factor": rec.get("somv_factor", 1.0),
        "tier_score": rec.get("tier_score", 0.0),
        "best_tier": rec.get("best_tier", 0),
        "avg_tier": rec.get("avg_tier", 0.0),
        "arch_coc_spell": rec.get("coc_score", 0.0),
        "arch_ci_es": rec.get("es_score", 0.0),
        "arch_mom_mana": rec.get("mana_score", 0.0),
        "pdps": rec.get("pdps", 0.0),
        "edps": rec.get("edps", 0.0),
        "demand_score": 0.0,
    }

    if use_new_features:
        features["item_level"] = rec.get("item_level", 0)
        features["armour"] = rec.get("armour", 0)
        features["evasion"] = rec.get("evasion", 0)
        features["energy_shield"] = rec.get("energy_shield", 0)
        features["total_dps"] = rec.get("total_dps", 0.0)
        features["total_defense"] = rec.get("total_defense", 0)
        features["quality"] = rec.get("quality", 0)
        features["sockets"] = rec.get("sockets", 0)
        features["corrupted"] = rec.get("corrupted", 0)
        features["open_prefixes"] = rec.get("open_prefixes", 0)
        features["open_suffixes"] = rec.get("open_suffixes", 0)

    # Mod features
    mt = rec.get("mod_tiers", {})
    mr = rec.get("mod_rolls", {})
    for g in rec.get("mod_groups", []):
        tier = mt.get(g, 0)
        rq = mr.get(g, -1)
        if rq >= 0 and tier > 0:
            features[f"mod:{g}"] = rq * (1.0 / tier)
        elif tier > 0:
            features[f"mod:{g}"] = 0.5 * (1.0 / tier)
        else:
            features[f"mod:{g}"] = 0.25

    # Base type
    bt = rec.get("base_type", "")
    if bt:
        features[f"base:{bt}"] = 1.0

    # Item class (for unified model)
    item_class = rec.get("item_class", "")
    if item_class:
        features[f"class:{item_class}"] = 1.0

    feat_array = [features.get(fn, 0.0) for fn in feature_names]

    pred = model["base_prediction"]
    lr = model["learning_rate"]
    for tree in model["trees"]:
        t_feat = tree["feature"]
        t_thresh = tree["threshold"]
        t_left = tree["left"]
        t_right = tree["right"]
        t_val = tree["value"]
        node = 0
        while t_feat[node] >= 0:
            if feat_array[t_feat[node]] <= t_thresh[node]:
                node = t_left[node]
            else:
                node = t_right[node]
        pred += lr * t_val[node]

    return math.exp(pred)


# ── Experiments ──────────────────────────────────────────

def experiment_baseline(train: List[dict], test: List[dict]) -> dict:
    """Experiment A: Current GBM features only (baseline)."""
    from gbm_trainer import train_gbm_models

    train_recs = prepare_gbm_records(train)
    test_recs = prepare_gbm_records(test)
    models = train_gbm_models(train_recs)

    predictions = []
    for rec in test_recs:
        item_class = rec["item_class"]
        model = models.get(item_class)
        if model and rec.get("mod_groups"):
            est = gbm_predict(model, rec)
        else:
            est = None
        predictions.append((est, rec["min_divine"]))

    metrics = compute_metrics(predictions, test_recs)
    print_metrics("Experiment A: Baseline (current GBM)", metrics)
    return metrics


def experiment_new_features(train: List[dict], test: List[dict]) -> dict:
    """Experiment B: Add item_level, raw defenses, total_dps/defense."""
    from gbm_trainer import train_gbm_models, _train_class_gbm
    import numpy as np

    train_recs = prepare_gbm_records(train, include_new_features=True)
    test_recs = prepare_gbm_records(test, include_new_features=True)

    # Train with extended feature list
    models = _train_gbm_extended(train_recs)

    predictions = []
    for rec in test_recs:
        item_class = rec["item_class"]
        model = models.get(item_class)
        if model and rec.get("mod_groups"):
            est = gbm_predict(model, rec, use_new_features=True)
        else:
            est = None
        predictions.append((est, rec["min_divine"]))

    metrics = compute_metrics(predictions, test_recs)
    print_metrics("Experiment B: + item_level, raw defenses, total_dps/defense",
                  metrics)
    return metrics


def experiment_listing_age(train: List[dict], test: List[dict]) -> dict:
    """Experiment C: Listing age decay sample weights."""
    train_recs = prepare_gbm_records(train, include_new_features=True,
                                      include_listing_age=True)
    test_recs = prepare_gbm_records(test, include_new_features=True)

    models = _train_gbm_extended(train_recs, use_age_weights=True)

    predictions = []
    for rec in test_recs:
        item_class = rec["item_class"]
        model = models.get(item_class)
        if model and rec.get("mod_groups"):
            est = gbm_predict(model, rec, use_new_features=True)
        else:
            est = None
        predictions.append((est, rec["min_divine"]))

    metrics = compute_metrics(predictions, test_recs)
    print_metrics("Experiment C: + listing age decay weights", metrics)
    return metrics


def experiment_unified(train: List[dict], test: List[dict]) -> dict:
    """Experiment D: Single unified cross-class GBM model."""
    train_recs = prepare_gbm_records(train, include_new_features=True,
                                      include_listing_age=True)
    test_recs = prepare_gbm_records(test, include_new_features=True)

    model = _train_unified_gbm(train_recs)
    if model is None:
        print("  Unified model failed to train")
        return {"pct_2x": 0, "total": 0}

    predictions = []
    for rec in test_recs:
        if rec.get("mod_groups"):
            est = gbm_predict(model, rec, use_new_features=True)
        else:
            est = None
        predictions.append((est, rec["min_divine"]))

    metrics = compute_metrics(predictions, test_recs)
    print_metrics("Experiment D: Unified cross-class GBM", metrics)
    return metrics


def experiment_hyperparams(train: List[dict], test: List[dict]) -> dict:
    """Experiment G: Tuned hyperparameters (more trees, deeper, lower lr)."""
    train_recs = prepare_gbm_records(train, include_new_features=True,
                                      include_listing_age=True)
    test_recs = prepare_gbm_records(test, include_new_features=True)

    models = _train_gbm_extended(
        train_recs, use_age_weights=True,
        n_estimators=300, max_depth=6, learning_rate=0.05,
        min_r2=-1.0,  # no quality gate
    )

    predictions = []
    for rec in test_recs:
        item_class = rec["item_class"]
        model = models.get(item_class)
        if model and rec.get("mod_groups"):
            est = gbm_predict(model, rec, use_new_features=True)
        else:
            est = None
        predictions.append((est, rec["min_divine"]))

    metrics = compute_metrics(predictions, test_recs)
    print_metrics("Experiment G: Tuned hyperparams (300 trees, depth 6, lr 0.05)",
                  metrics)
    return metrics


def experiment_interactions(train: List[dict], test: List[dict]) -> dict:
    """Experiment H: Feature interactions."""
    train_recs = prepare_gbm_records(train, include_new_features=True,
                                      include_listing_age=True)
    test_recs = prepare_gbm_records(test, include_new_features=True)

    # Add interaction features
    for recs in [train_recs, test_recs]:
        for rec in recs:
            il = rec.get("item_level", 0)
            ttc = rec.get("top_tier_count", 0)
            mc = rec.get("mod_count", 4)
            at = rec.get("avg_tier", 0.0)
            ar = rec.get("armour", 0)
            ev = rec.get("evasion", 0)
            es = rec.get("energy_shield", 0)

            rec["ilvl_x_ttc"] = il * ttc
            rec["mc_x_avg_tier"] = mc * (1.0 / at if at > 0 else 0)

            # Defense type category
            total = ar + ev + es + 0.01
            rec["ar_ratio"] = ar / total
            rec["ev_ratio"] = ev / total
            rec["es_ratio"] = es / total

    models = _train_gbm_extended(
        train_recs, use_age_weights=True,
        n_estimators=300, max_depth=6, learning_rate=0.05,
        min_r2=-1.0,
        extra_numeric=["ilvl_x_ttc", "mc_x_avg_tier",
                       "ar_ratio", "ev_ratio", "es_ratio"],
    )

    predictions = []
    for rec in test_recs:
        item_class = rec["item_class"]
        model = models.get(item_class)
        if model and rec.get("mod_groups"):
            est = gbm_predict(model, rec, use_new_features=True)
        else:
            est = None
        predictions.append((est, rec["min_divine"]))

    metrics = compute_metrics(predictions, test_recs)
    print_metrics("Experiment H: + feature interactions", metrics)
    return metrics


def experiment_enriched(train: List[dict], test: List[dict]) -> dict:
    """Experiment: All new enrichment features (quality, sockets, corruption, open affixes)."""
    from gbm_trainer import train_gbm_models

    train_recs = prepare_gbm_records(train, include_new_features=True)
    test_recs = prepare_gbm_records(test, include_new_features=True)

    models = train_gbm_models(train_recs)

    predictions = []
    for rec in test_recs:
        item_class = rec["item_class"]
        model = models.get(item_class)
        if model and rec.get("mod_groups"):
            est = gbm_predict(model, rec, use_new_features=True)
        else:
            est = None
        predictions.append((est, rec["min_divine"]))

    metrics = compute_metrics(predictions, test_recs)
    print_metrics("Experiment: Enriched (all new features via standard GBM)", metrics)
    return metrics


def experiment_open_affixes(train: List[dict], test: List[dict]) -> dict:
    """Experiment: Test open affix impact in isolation."""
    train_recs = prepare_gbm_records(train, include_new_features=True)
    test_recs = prepare_gbm_records(test, include_new_features=True)

    # Zero out all new features except open_prefixes/open_suffixes
    for recs in [train_recs, test_recs]:
        for rec in recs:
            rec["quality"] = 0
            rec["sockets"] = 0
            rec["corrupted"] = 0

    models = _train_gbm_extended(train_recs)

    predictions = []
    for rec in test_recs:
        item_class = rec["item_class"]
        model = models.get(item_class)
        if model and rec.get("mod_groups"):
            est = gbm_predict(model, rec, use_new_features=True)
        else:
            est = None
        predictions.append((est, rec["min_divine"]))

    metrics = compute_metrics(predictions, test_recs)
    print_metrics("Experiment: Open affixes only (quality/sockets/corrupt zeroed)",
                  metrics)
    return metrics


def experiment_best_combo(train: List[dict], test: List[dict]) -> dict:
    """Best combination: all improvements together."""
    train_recs = prepare_gbm_records(train, include_new_features=True,
                                      include_listing_age=True)
    test_recs = prepare_gbm_records(test, include_new_features=True)

    # Add interaction features
    for recs in [train_recs, test_recs]:
        for rec in recs:
            il = rec.get("item_level", 0)
            ttc = rec.get("top_tier_count", 0)
            mc = rec.get("mod_count", 4)
            at = rec.get("avg_tier", 0.0)
            ar = rec.get("armour", 0)
            ev = rec.get("evasion", 0)
            es = rec.get("energy_shield", 0)

            rec["ilvl_x_ttc"] = il * ttc
            rec["mc_x_avg_tier"] = mc * (1.0 / at if at > 0 else 0)
            total = ar + ev + es + 0.01
            rec["ar_ratio"] = ar / total
            rec["ev_ratio"] = ev / total
            rec["es_ratio"] = es / total

    # Try per-class with tuned hyperparams
    per_class = _train_gbm_extended(
        train_recs, use_age_weights=True,
        n_estimators=300, max_depth=6, learning_rate=0.05,
        min_r2=-1.0,
        extra_numeric=["ilvl_x_ttc", "mc_x_avg_tier",
                       "ar_ratio", "ev_ratio", "es_ratio"],
    )

    # Also train unified as fallback
    unified = _train_unified_gbm(
        train_recs, use_age_weights=True,
        n_estimators=300, max_depth=6, learning_rate=0.05,
        extra_numeric=["ilvl_x_ttc", "mc_x_avg_tier",
                       "ar_ratio", "ev_ratio", "es_ratio"],
    )

    predictions = []
    for rec in test_recs:
        item_class = rec["item_class"]
        model = per_class.get(item_class)
        if model and rec.get("mod_groups"):
            est = gbm_predict(model, rec, use_new_features=True)
        elif unified and rec.get("mod_groups"):
            est = gbm_predict(unified, rec, use_new_features=True)
        else:
            est = None
        predictions.append((est, rec["min_divine"]))

    metrics = compute_metrics(predictions, test_recs)
    print_metrics("BEST COMBO: per-class + unified fallback, all features",
                  metrics)
    return metrics


# ── Extended GBM training ──────────────────────────────────

def _train_gbm_extended(records: List[dict],
                        use_age_weights: bool = False,
                        n_estimators: int = None,
                        max_depth: int = None,
                        learning_rate: float = None,
                        min_r2: float = 0.02,
                        min_class_samples: int = 50,
                        extra_numeric: List[str] = None,
                        ) -> Dict[str, dict]:
    """Train per-class GBM with extended features."""
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    from gbm_trainer import (
        MIN_MOD_FREQUENCY, MAX_BASE_FEATURES, _serialize_trees,
        GBM_LEARNING_RATE, GBM_LOSS, GBM_SUBSAMPLE,
    )

    by_class = defaultdict(list)
    for rec in records:
        ic = rec.get("item_class", "")
        if ic and rec.get("min_divine", 0) > 0:
            by_class[ic].append(rec)

    models = {}
    for item_class, class_records in by_class.items():
        n = len(class_records)
        if n < min_class_samples:
            continue

        # Adaptive hyperparameters
        if n_estimators is None:
            n_est = 150 if n >= 1500 else (100 if n >= 500 else 60)
        else:
            n_est = n_estimators
        if max_depth is None:
            depth = 5 if n >= 1500 else (4 if n >= 500 else 3)
        else:
            depth = max_depth
        lr = learning_rate or GBM_LEARNING_RATE
        leaf = 20 if n >= 1500 else (15 if n >= 500 else 10)

        # Feature discovery
        mod_freq = defaultdict(int)
        for rec in class_records:
            for g in rec.get("mod_groups", []):
                if g:
                    mod_freq[g] += 1
        valid_mods = sorted(g for g, c in mod_freq.items()
                            if c >= MIN_MOD_FREQUENCY)

        base_freq = defaultdict(int)
        for rec in class_records:
            bt = rec.get("base_type", "")
            if bt:
                base_freq[bt] += 1
        valid_bases = sorted(
            (bt for bt, c in base_freq.items() if c >= 3),
            key=lambda bt: -base_freq[bt]
        )[:MAX_BASE_FEATURES]

        # Build feature names
        numeric_names = [
            "grade_num", "score", "top_tier_count", "mod_count",
            "dps_factor", "defense_factor", "somv_factor",
            "tier_score", "best_tier", "avg_tier",
            "arch_coc_spell", "arch_ci_es", "arch_mom_mana",
            "pdps", "edps", "demand_score",
            # New features
            "item_level", "armour", "evasion", "energy_shield",
            "total_dps", "total_defense",
            "quality", "sockets", "corrupted",
            "open_prefixes", "open_suffixes",
        ]
        if extra_numeric:
            numeric_names.extend(extra_numeric)

        mod_feature_names = [f"mod:{g}" for g in valid_mods]
        base_feature_names = [f"base:{bt}" for bt in valid_bases]
        feature_names = numeric_names + mod_feature_names + base_feature_names
        n_features = len(feature_names)

        mod_idx = {g: len(numeric_names) + i for i, g in enumerate(valid_mods)}
        base_idx = {bt: len(numeric_names) + len(valid_mods) + i
                    for i, bt in enumerate(valid_bases)}

        X = np.zeros((n, n_features), dtype=np.float64)
        y = np.zeros(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64)

        for row, rec in enumerate(class_records):
            y[row] = math.log(max(rec["min_divine"], 0.01))

            # Numeric features
            for fi, fname in enumerate(numeric_names):
                # Map archetype feature names
                key = fname
                if fname == "arch_coc_spell":
                    key = "coc_score"
                elif fname == "arch_ci_es":
                    key = "es_score"
                elif fname == "arch_mom_mana":
                    key = "mana_score"
                X[row, fi] = rec.get(key, 0.0)

            # Mod features
            mt = rec.get("mod_tiers", {})
            mr = rec.get("mod_rolls", {})
            for g in rec.get("mod_groups", []):
                if g in mod_idx:
                    tier = mt.get(g, 0)
                    rq = mr.get(g, -1)
                    if rq >= 0 and tier > 0:
                        X[row, mod_idx[g]] = rq * (1.0 / tier)
                    elif tier > 0:
                        X[row, mod_idx[g]] = 0.5 * (1.0 / tier)
                    else:
                        X[row, mod_idx[g]] = 0.25

            # Base type features
            bt = rec.get("base_type", "")
            if bt in base_idx:
                X[row, base_idx[bt]] = 1.0

            # Sample weights
            sc = rec.get("sale_confidence", 1.0)
            if sc and sc != 1.0:
                weights[row] = sc

            if use_age_weights:
                age = rec.get("listing_age_days", 14.0)
                weights[row] *= 1.0 / (1.0 + age / 7.0)

        gbm = GradientBoostingRegressor(
            n_estimators=n_est, max_depth=depth,
            learning_rate=lr, subsample=GBM_SUBSAMPLE,
            min_samples_leaf=leaf, loss=GBM_LOSS, random_state=42,
        )
        gbm.fit(X, y, sample_weight=weights)

        # Cross-validation R2
        try:
            cv_scores = cross_val_score(
                GradientBoostingRegressor(
                    n_estimators=n_est, max_depth=depth,
                    learning_rate=lr, subsample=GBM_SUBSAMPLE,
                    min_samples_leaf=leaf, loss=GBM_LOSS, random_state=42,
                ),
                X, y, cv=3, scoring="r2",
            )
            r2_cv = max(0.0, float(np.mean(cv_scores)))
        except Exception:
            r2_cv = 0.0

        if r2_cv < min_r2:
            continue

        trees = _serialize_trees(gbm, feature_names)
        init_pred = gbm._raw_predict_init(X[:1])
        base_prediction = float(init_pred.flat[0])

        models[item_class] = {
            "learning_rate": lr,
            "base_prediction": round(base_prediction, 6),
            "trees": trees,
            "feature_names": feature_names,
            "mod_features": valid_mods,
            "base_features": valid_bases,
            "n_train": n,
            "r2_cv": round(r2_cv, 4),
        }

    return models


def _train_unified_gbm(records: List[dict],
                       use_age_weights: bool = False,
                       n_estimators: int = 200,
                       max_depth: int = 6,
                       learning_rate: float = 0.05,
                       extra_numeric: List[str] = None,
                       ) -> Optional[dict]:
    """Train a single unified GBM across all classes."""
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from gbm_trainer import (
        MIN_MOD_FREQUENCY, MAX_BASE_FEATURES, _serialize_trees,
        GBM_LOSS, GBM_SUBSAMPLE,
    )

    valid = [r for r in records if r.get("min_divine", 0) > 0
             and r.get("item_class")]
    if len(valid) < 100:
        return None

    n = len(valid)

    # Feature discovery
    mod_freq = defaultdict(int)
    for rec in valid:
        for g in rec.get("mod_groups", []):
            if g:
                mod_freq[g] += 1
    valid_mods = sorted(g for g, c in mod_freq.items()
                        if c >= MIN_MOD_FREQUENCY)

    base_freq = defaultdict(int)
    for rec in valid:
        bt = rec.get("base_type", "")
        if bt:
            base_freq[bt] += 1
    valid_bases = sorted(
        (bt for bt, c in base_freq.items() if c >= 3),
        key=lambda bt: -base_freq[bt]
    )[:MAX_BASE_FEATURES * 3]  # more bases for unified

    # Item class discovery
    class_freq = defaultdict(int)
    for rec in valid:
        class_freq[rec["item_class"]] += 1
    valid_classes = sorted(
        (c for c, cnt in class_freq.items() if cnt >= 20),
        key=lambda c: -class_freq[c]
    )

    numeric_names = [
        "grade_num", "score", "top_tier_count", "mod_count",
        "dps_factor", "defense_factor", "somv_factor",
        "tier_score", "best_tier", "avg_tier",
        "arch_coc_spell", "arch_ci_es", "arch_mom_mana",
        "pdps", "edps", "demand_score",
        "item_level", "armour", "evasion", "energy_shield",
        "total_dps", "total_defense",
        "quality", "sockets", "corrupted",
        "open_prefixes", "open_suffixes",
    ]
    if extra_numeric:
        numeric_names.extend(extra_numeric)

    mod_feature_names = [f"mod:{g}" for g in valid_mods]
    base_feature_names = [f"base:{bt}" for bt in valid_bases]
    class_feature_names = [f"class:{c}" for c in valid_classes]
    feature_names = (numeric_names + mod_feature_names
                     + base_feature_names + class_feature_names)
    n_features = len(feature_names)

    mod_idx = {g: len(numeric_names) + i for i, g in enumerate(valid_mods)}
    base_idx = {bt: len(numeric_names) + len(valid_mods) + i
                for i, bt in enumerate(valid_bases)}
    class_idx = {c: len(numeric_names) + len(valid_mods) + len(valid_bases) + i
                 for i, c in enumerate(valid_classes)}

    X = np.zeros((n, n_features), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    weights = np.ones(n, dtype=np.float64)

    for row, rec in enumerate(valid):
        y[row] = math.log(max(rec["min_divine"], 0.01))

        for fi, fname in enumerate(numeric_names):
            key = fname
            if fname == "arch_coc_spell":
                key = "coc_score"
            elif fname == "arch_ci_es":
                key = "es_score"
            elif fname == "arch_mom_mana":
                key = "mana_score"
            X[row, fi] = rec.get(key, 0.0)

        mt = rec.get("mod_tiers", {})
        mr = rec.get("mod_rolls", {})
        for g in rec.get("mod_groups", []):
            if g in mod_idx:
                tier = mt.get(g, 0)
                rq = mr.get(g, -1)
                if rq >= 0 and tier > 0:
                    X[row, mod_idx[g]] = rq * (1.0 / tier)
                elif tier > 0:
                    X[row, mod_idx[g]] = 0.5 * (1.0 / tier)
                else:
                    X[row, mod_idx[g]] = 0.25

        bt = rec.get("base_type", "")
        if bt in base_idx:
            X[row, base_idx[bt]] = 1.0

        ic = rec.get("item_class", "")
        if ic in class_idx:
            X[row, class_idx[ic]] = 1.0

        sc = rec.get("sale_confidence", 1.0)
        if sc and sc != 1.0:
            weights[row] = sc
        if use_age_weights:
            age = rec.get("listing_age_days", 14.0)
            weights[row] *= 1.0 / (1.0 + age / 7.0)

    gbm = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=GBM_SUBSAMPLE,
        min_samples_leaf=20, loss=GBM_LOSS, random_state=42,
    )
    gbm.fit(X, y, sample_weight=weights)

    trees = _serialize_trees(gbm, feature_names)
    init_pred = gbm._raw_predict_init(X[:1])
    base_prediction = float(init_pred.flat[0])

    return {
        "learning_rate": learning_rate,
        "base_prediction": round(base_prediction, 6),
        "trees": trees,
        "feature_names": feature_names,
        "mod_features": valid_mods,
        "base_features": valid_bases,
        "n_train": n,
        "r2_cv": 0.0,  # skip CV for speed
    }


# ── Main ─────────────────────────────────────────────────

EXPERIMENTS = {
    "baseline": experiment_baseline,
    "new_features": experiment_new_features,
    "enriched": experiment_enriched,
    "open_affixes": experiment_open_affixes,
    "listing_age": experiment_listing_age,
    "unified": experiment_unified,
    "hyperparams": experiment_hyperparams,
    "interactions": experiment_interactions,
    "best": experiment_best_combo,
}


def main():
    parser = argparse.ArgumentParser(
        description="LAMA Accuracy Lab - experiment harness")
    parser.add_argument("--input", "-i", nargs="+",
                        default=[DEFAULT_JSONL_GLOB],
                        help="Input JSONL file(s) (supports globs)")
    parser.add_argument("--experiment", "-e", nargs="*",
                        help=f"Experiments to run: {', '.join(EXPERIMENTS.keys())} "
                             f"(default: all)")
    args = parser.parse_args()

    print("=" * 60)
    print("  LAMA Accuracy Lab")
    print("=" * 60)

    print(f"\nLoading data from: {args.input}")
    records = load_and_prepare(args.input)
    print(f"\nTotal prepared records: {len(records)}")

    train, test = split_data(records)
    print(f"Train: {len(train)}, Test: {len(test)}")

    # Class distribution
    class_counts = defaultdict(int)
    for r in records:
        class_counts[r.get("item_class", "?")] += 1
    print(f"\nItem classes ({len(class_counts)}):")
    for cls in sorted(class_counts, key=class_counts.get, reverse=True)[:15]:
        print(f"  {cls:20s}: {class_counts[cls]:5d}")

    # Run experiments
    exps = args.experiment if args.experiment else list(EXPERIMENTS.keys())
    results = {}

    for name in exps:
        if name not in EXPERIMENTS:
            print(f"\nUnknown experiment: {name}")
            continue
        t0 = time.time()
        try:
            metrics = EXPERIMENTS[name](train, test)
            results[name] = metrics
        except Exception as e:
            print(f"\n  Experiment {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
        elapsed = time.time() - t0
        print(f"  (took {elapsed:.1f}s)")

    # Summary
    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print("  SUMMARY")
        print(f"{'=' * 60}")
        for name in exps:
            if name in results:
                m = results[name]
                print(f"  {name:20s}: {m['pct_2x']:5.1f}% within 2x "
                      f"({m['total']} tested, median {m.get('median_error', 0):.2f}x)")


if __name__ == "__main__":
    main()
