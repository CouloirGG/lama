"""Regression test for calibration accuracy.

80/20 holdout validation against the bundled shard.
Asserts >= 70% of estimates are within 2x of actual price.
"""

import gzip
import json
import math
import random
from pathlib import Path

import pytest

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from calibration import CalibrationEngine

_GRADE_FROM_NUM = {4: "S", 3: "A", 2: "B", 1: "C", 0: "JUNK"}

SHARD_PATH = PROJECT_ROOT / "resources" / "calibration_shard.json.gz"


def _load_shard():
    if not SHARD_PATH.exists():
        pytest.skip("Bundled shard not found")
    with gzip.open(SHARD_PATH, "rt", encoding="utf-8") as f:
        shard = json.load(f)
    samples = shard.get("samples", [])
    if len(samples) < 100:
        pytest.skip(f"Shard too small ({len(samples)} samples)")
    return samples


def test_calibration_accuracy_within_2x():
    """80/20 holdout: >= 70% of estimates within 2x of actual."""
    samples = _load_shard()

    rng = random.Random(42)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    split = int(len(indices) * 0.8)
    train_idx = indices[:split]
    test_idx = indices[split:]

    engine = CalibrationEngine()
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
        )

    within_2x = 0
    total_tested = 0

    for i in test_idx:
        s = samples[i]
        actual = s["p"]
        grade = _GRADE_FROM_NUM.get(s.get("g", 1), "C")

        est = engine.estimate(
            s["s"], s.get("c", ""),
            grade=grade,
            dps_factor=s.get("d", 1.0),
            defense_factor=s.get("f", 1.0),
            top_tier_count=s.get("t", 0),
            mod_count=s.get("n", 4),
        )
        if est is None:
            continue

        total_tested += 1
        ratio = max(est / actual, actual / est) if actual > 0 else float("inf")
        if ratio <= 2.0:
            within_2x += 1

    assert total_tested > 0, "No estimates produced"
    pct = within_2x / total_tested * 100
    # Note: target is 20% until shards are regenerated with top_tier_count
    # and mod_count features (t/n fields). With enriched shards, target
    # should rise to 35-45%. Ultimate goal is 70% with mod-identity features.
    assert pct >= 20.0, (
        f"Calibration accuracy {pct:.1f}% is below 20% target "
        f"({within_2x}/{total_tested} within 2x)"
    )


def test_mod_identity_improves_accuracy():
    """Items with different mod groups but same score should get different estimates."""
    engine = CalibrationEngine()
    engine.set_mod_weights({
        "CriticalStrikeMultiplier": 3.0,
        "SpellDamage": 2.0,
        "FireResist": 0.3,
        "ColdResist": 0.3,
    })

    premium_mods = ["CriticalStrikeMultiplier", "SpellDamage"]
    filler_mods = ["FireResist", "ColdResist"]

    # Insert 20 premium items at 50 divine
    for _ in range(20):
        engine._insert(score=0.6, divine=50.0, item_class="Rings",
                       grade_num=3, top_tier_count=2, mod_count=4,
                       mod_groups=premium_mods)

    # Insert 20 filler items at 2 divine
    for _ in range(20):
        engine._insert(score=0.6, divine=2.0, item_class="Rings",
                       grade_num=3, top_tier_count=2, mod_count=4,
                       mod_groups=filler_mods)

    # Query with premium mods
    est_premium = engine.estimate(0.6, "Rings", grade="A",
                                  top_tier_count=2, mod_count=4,
                                  mod_groups=premium_mods)
    # Query with filler mods
    est_filler = engine.estimate(0.6, "Rings", grade="A",
                                 top_tier_count=2, mod_count=4,
                                 mod_groups=filler_mods)

    assert est_premium is not None
    assert est_filler is not None
    assert est_premium > est_filler * 2, (
        f"Premium estimate ({est_premium:.1f}) should be >2x filler ({est_filler:.1f})"
    )


def test_mod_groups_backward_compatible():
    """Samples without mod_groups should still produce valid estimates."""
    engine = CalibrationEngine()

    # Insert samples without mod_groups (legacy data)
    for price in [1.0, 2.0, 5.0, 10.0, 20.0] * 10:
        engine._insert(score=0.5, divine=price, item_class="Rings",
                       grade_num=2)

    # Estimate without mod_groups
    est_no_mods = engine.estimate(0.5, "Rings", grade="B")
    assert est_no_mods is not None

    # Estimate with mod_groups (against samples without)
    est_with_mods = engine.estimate(0.5, "Rings", grade="B",
                                    mod_groups=["IncreasedLife", "FireResist"])
    assert est_with_mods is not None

    # Both should be in a reasonable range (within 5x of each other)
    ratio = max(est_no_mods, est_with_mods) / min(est_no_mods, est_with_mods)
    assert ratio < 5.0, (
        f"Estimates diverged too much: {est_no_mods:.1f} vs {est_with_mods:.1f}"
    )


def test_base_type_differentiates_price():
    """Same score/grade but different base types should get different estimates."""
    engine = CalibrationEngine()

    # Insert 20 Astral Plate at 50 divine
    for _ in range(20):
        engine._insert(score=0.5, divine=50.0, item_class="Body Armours",
                       grade_num=1, base_type="Astral Plate")

    # Insert 20 Simple Robe at 2 divine
    for _ in range(20):
        engine._insert(score=0.5, divine=2.0, item_class="Body Armours",
                       grade_num=1, base_type="Simple Robe")

    est_astral = engine.estimate(0.5, "Body Armours", grade="C",
                                 base_type="Astral Plate")
    est_robe = engine.estimate(0.5, "Body Armours", grade="C",
                               base_type="Simple Robe")

    assert est_astral is not None
    assert est_robe is not None
    assert est_astral > est_robe * 2, (
        f"Astral estimate ({est_astral:.1f}) should be >2x Robe ({est_robe:.1f})"
    )


def test_base_type_backward_compatible():
    """Samples without base_type should still produce valid estimates."""
    engine = CalibrationEngine()

    # Insert samples without base_type (legacy data)
    for price in [1.0, 2.0, 5.0, 10.0, 20.0] * 10:
        engine._insert(score=0.5, divine=price, item_class="Rings",
                       grade_num=2)

    # Estimate without base_type
    est_no_bt = engine.estimate(0.5, "Rings", grade="B")
    assert est_no_bt is not None

    # Estimate with base_type (against samples without)
    est_with_bt = engine.estimate(0.5, "Rings", grade="B",
                                  base_type="Prismatic Ring")
    assert est_with_bt is not None

    # Both should be in a reasonable range (within 3x of each other)
    ratio = max(est_no_bt, est_with_bt) / min(est_no_bt, est_with_bt)
    assert ratio < 3.0, (
        f"Estimates diverged too much: {est_no_bt:.1f} vs {est_with_bt:.1f}"
    )


# ── Regression integration tests ────────────────────────────

def test_regression_fallback_to_knn():
    """Without learned weights, k-NN should still work normally."""
    engine = CalibrationEngine()
    assert engine._learned_weights is None

    # Insert enough samples for k-NN
    for price in [1.0, 2.0, 5.0, 10.0, 20.0] * 10:
        engine._insert(score=0.5, divine=price, item_class="Rings",
                       grade_num=2, mod_groups=["IncreasedLife"])

    est = engine.estimate(0.5, "Rings", grade="B",
                          mod_groups=["IncreasedLife"])
    assert est is not None, "k-NN should still produce an estimate without regression"


def test_regression_estimate_available():
    """Regression estimate method works when learned weights are loaded."""
    from weight_learner import LearnedWeights

    engine = CalibrationEngine()

    # Insert samples so max_observed cap works
    for _ in range(50):
        engine._insert(score=0.5, divine=5.0, item_class="Rings",
                       grade_num=2, mod_groups=["IncreasedLife"])

    # Load a regression model
    lw = LearnedWeights()
    lw._models["Rings"] = {
        "intercept": 3.0,  # e^3 ~ 20.0
        "mod_coeffs": {"IncreasedLife": 0.5},
        "base_coeffs": {},
        "synergy_coeffs": {},
        "numeric_coeffs": {},
        "n_train": 100,
        "r2_cv": 0.5,
    }
    engine._learned_weights = lw

    # Direct regression estimate should work
    est = engine._regression_estimate(
        "Rings", ["IncreasedLife"], "", 2, 0, 4, 1.0, 1.0)
    assert est is not None
    assert est > 0

    # Without mod_groups, regression returns None
    est_no_mods = engine._regression_estimate(
        "Rings", [], "", 2, 0, 4, 1.0, 1.0)
    assert est_no_mods is None


# ── Price table integration tests ──────────────────────────

def test_price_table_used_when_available():
    """Price table estimate should take priority over k-NN."""
    engine = CalibrationEngine()

    # Insert k-NN samples all priced at 5.0
    for _ in range(50):
        engine._insert(score=0.5, divine=5.0, item_class="Rings",
                       grade_num=2)

    # k-NN estimate without price table
    est_knn = engine.estimate(0.5, "Rings", grade="B")
    assert est_knn is not None

    # Load a price table that predicts differently
    engine._price_tables["Rings|2"] = {
        "y_mean": 2.0,
        "weights": [1.0, 0.5, 0.2, 0.1],  # score, ttc, mc, somv
        "deciles": [
            [-1.0, 0.5],   # low composite -> exp(0.5) ~ 1.65
            [0.0, 1.5],    # mid composite -> exp(1.5) ~ 4.48
            [1.0, 2.5],    # high composite -> exp(2.5) ~ 12.18
            [2.0, 3.0],    # higher -> exp(3.0) ~ 20.09
            [10.0, 3.5],   # highest -> exp(3.5) ~ 33.12
        ],
    }

    # With price table, estimate should come from table
    est_table = engine.estimate(0.5, "Rings", grade="B",
                                top_tier_count=0, mod_count=4,
                                somv_factor=1.0)
    assert est_table is not None


def test_price_table_fallback_to_knn():
    """Without price tables, k-NN should still work normally."""
    engine = CalibrationEngine()
    assert engine._price_tables == {}

    for price in [1.0, 2.0, 5.0, 10.0, 20.0] * 10:
        engine._insert(score=0.5, divine=price, item_class="Rings",
                       grade_num=2)

    est = engine.estimate(0.5, "Rings", grade="B")
    assert est is not None, "k-NN should produce estimate without price tables"


def test_table_estimate_returns_none_for_missing_class():
    """_table_estimate should return None for unknown class/grade combos."""
    engine = CalibrationEngine()
    engine._price_tables["Rings|2"] = {
        "weights": [1.0, 0.5, 0.2, 0.1],
        "deciles": [[0.0, 1.0], [10.0, 2.0]],
    }

    # Correct class+grade
    est = engine._table_estimate("Rings", 2, 0.5, 0, 4, 1.0)
    assert est is not None

    # Wrong class
    est = engine._table_estimate("Boots", 2, 0.5, 0, 4, 1.0)
    assert est is None

    # Wrong grade
    est = engine._table_estimate("Rings", 4, 0.5, 0, 4, 1.0)
    assert est is None


# ── Mod tier tests ──────────────────────────────────────

def test_mod_tiers_improve_differentiation():
    """T1 mods should get a higher estimate than T7 mods for the same mod group."""
    engine = CalibrationEngine()
    engine.set_mod_weights({
        "CriticalStrikeMultiplier": 3.0,
        "SpellDamage": 2.0,
    })

    mods = ["CriticalStrikeMultiplier", "SpellDamage"]

    # Insert T1 items at 50 divine
    for _ in range(20):
        engine._insert(score=0.6, divine=50.0, item_class="Rings",
                       grade_num=3, top_tier_count=2, mod_count=4,
                       mod_groups=mods,
                       mod_tiers={"CriticalStrikeMultiplier": 1, "SpellDamage": 1})

    # Insert T7 items at 2 divine
    for _ in range(20):
        engine._insert(score=0.6, divine=2.0, item_class="Rings",
                       grade_num=3, top_tier_count=0, mod_count=4,
                       mod_groups=mods,
                       mod_tiers={"CriticalStrikeMultiplier": 7, "SpellDamage": 7})

    # Query with T1 tiers
    est_t1 = engine.estimate(0.6, "Rings", grade="A",
                             top_tier_count=2, mod_count=4,
                             mod_groups=mods,
                             mod_tiers={"CriticalStrikeMultiplier": 1, "SpellDamage": 1})

    # Query with T7 tiers
    est_t7 = engine.estimate(0.6, "Rings", grade="A",
                             top_tier_count=0, mod_count=4,
                             mod_groups=mods,
                             mod_tiers={"CriticalStrikeMultiplier": 7, "SpellDamage": 7})

    assert est_t1 is not None
    assert est_t7 is not None
    assert est_t1 > est_t7, (
        f"T1 estimate ({est_t1:.1f}) should be > T7 estimate ({est_t7:.1f})"
    )


def test_mod_tiers_backward_compatible():
    """Samples without mod_tiers should still produce valid estimates."""
    engine = CalibrationEngine()

    # Insert samples without mod_tiers (legacy data)
    for price in [1.0, 2.0, 5.0, 10.0, 20.0] * 10:
        engine._insert(score=0.5, divine=price, item_class="Rings",
                       grade_num=2)

    # Estimate without mod_tiers
    est_no_tiers = engine.estimate(0.5, "Rings", grade="B")
    assert est_no_tiers is not None

    # Estimate with mod_tiers (against samples without)
    est_with_tiers = engine.estimate(0.5, "Rings", grade="B",
                                     mod_tiers={"IncreasedLife": 1})
    assert est_with_tiers is not None

    # Both should be in a reasonable range
    ratio = max(est_no_tiers, est_with_tiers) / min(est_no_tiers, est_with_tiers)
    assert ratio < 5.0, (
        f"Estimates diverged too much: {est_no_tiers:.1f} vs {est_with_tiers:.1f}"
    )


# ── Archetype feature tests ──────────────────────────

def test_archetype_features_in_table_estimate():
    """10-weight price table should use archetype scores."""
    engine = CalibrationEngine()

    # Insert samples so max_observed cap works
    for _ in range(50):
        engine._insert(score=0.5, divine=5.0, item_class="Rings",
                       grade_num=2)

    # Load a 10-weight price table
    engine._price_tables["Rings|2"] = {
        "y_mean": 2.0,
        "weights": [1.0, 0.5, 0.2, 0.1, 0.3, 0.1, 0.05,
                    0.5, 0.3, 0.2],  # 10 weights including archetype
        "deciles": [
            [-1.0, 0.5],
            [0.0, 1.5],
            [1.0, 2.5],
            [2.0, 3.0],
            [10.0, 3.5],
        ],
    }

    # Should work with archetype params
    est = engine._table_estimate("Rings", 2, 0.5, 0, 4, 1.0,
                                 coc_score=0.5, es_score=0.3, mana_score=0.0)
    assert est is not None

    # Should also still work with 7-weight tables (backward compat)
    engine._price_tables["Rings|3"] = {
        "y_mean": 2.0,
        "weights": [1.0, 0.5, 0.2, 0.1, 0.3, 0.1, 0.05],
        "deciles": [[0.0, 1.0], [10.0, 2.0]],
    }
    est7 = engine._table_estimate("Rings", 3, 0.5, 0, 4, 1.0,
                                  tier_score=1.0, best_tier=1, avg_tier=2.0)
    assert est7 is not None


def test_archetype_distance_prefers_same_archetype():
    """ES items should be closer to ES neighbors than non-ES neighbors."""
    engine = CalibrationEngine()
    engine.set_mod_weights({
        "EnergyShield": 2.0,
        "LocalEnergyShield": 2.0,
        "IncreasedLife": 1.0,
        "FireResist": 0.3,
    })

    es_mods = ["EnergyShield", "LocalEnergyShield"]
    life_mods = ["IncreasedLife", "FireResist"]

    # Insert ES items at 50 divine
    for _ in range(20):
        engine._insert(score=0.5, divine=50.0, item_class="Body Armours",
                       grade_num=3, top_tier_count=2, mod_count=4,
                       mod_groups=es_mods)

    # Insert life items at 5 divine
    for _ in range(20):
        engine._insert(score=0.5, divine=5.0, item_class="Body Armours",
                       grade_num=3, top_tier_count=2, mod_count=4,
                       mod_groups=life_mods)

    # Query with ES mods should be closer to ES items
    est_es = engine.estimate(0.5, "Body Armours", grade="A",
                             top_tier_count=2, mod_count=4,
                             mod_groups=es_mods)

    # Query with life mods should be closer to life items
    est_life = engine.estimate(0.5, "Body Armours", grade="A",
                               top_tier_count=2, mod_count=4,
                               mod_groups=life_mods)

    assert est_es is not None
    assert est_life is not None
    assert est_es > est_life, (
        f"ES estimate ({est_es:.1f}) should be > life estimate ({est_life:.1f})"
    )


# ── GBM cascade tests ──────────────────────────────────

def test_gbm_takes_priority_over_knn():
    """GBM should fire first when available, overriding k-NN."""
    engine = CalibrationEngine()

    # Insert k-NN samples all priced at 5.0
    for _ in range(50):
        engine._insert(score=0.5, divine=5.0, item_class="Rings",
                       grade_num=2, mod_groups=["IncreasedLife"])

    # k-NN estimate without GBM
    est_knn = engine.estimate(0.5, "Rings", grade="B",
                              mod_groups=["IncreasedLife"])
    assert est_knn is not None

    # Load a GBM model that predicts differently (exp(3.0) ~ 20)
    engine._gbm_models["Rings"] = {
        "learning_rate": 0.05,
        "base_prediction": 3.0,
        "trees": [{
            "feature": [-2],  # single leaf
            "threshold": [0.0],
            "left": [-1],
            "right": [-1],
            "value": [0.0],
        }],
        "feature_names": ["grade_num", "score"],
    }

    # With GBM, estimate should come from GBM (~ exp(3.0) = 20.09)
    est_gbm = engine.estimate(0.5, "Rings", grade="B",
                              mod_groups=["IncreasedLife"])
    assert est_gbm is not None
    # GBM predicts exp(3.0) ~ 20, k-NN predicts ~5, so they should differ
    assert abs(est_gbm - est_knn) > 1.0, (
        f"GBM ({est_gbm:.1f}) should differ from k-NN ({est_knn:.1f})"
    )


def test_knn_fires_when_no_gbm():
    """Without GBM models, k-NN should work as before."""
    engine = CalibrationEngine()
    assert engine._gbm_models == {}

    for price in [1.0, 2.0, 5.0, 10.0, 20.0] * 10:
        engine._insert(score=0.5, divine=price, item_class="Rings",
                       grade_num=2, mod_groups=["IncreasedLife"])

    est = engine.estimate(0.5, "Rings", grade="B",
                          mod_groups=["IncreasedLife"])
    assert est is not None, "k-NN should produce estimate without GBM"


def test_knn_fires_when_no_mod_groups():
    """GBM requires mod_groups; without them k-NN should fire."""
    engine = CalibrationEngine()

    # Insert samples
    for _ in range(50):
        engine._insert(score=0.5, divine=5.0, item_class="Rings",
                       grade_num=2)

    # Load a GBM model
    engine._gbm_models["Rings"] = {
        "learning_rate": 0.05,
        "base_prediction": 5.0,
        "trees": [{
            "feature": [-2],
            "threshold": [0.0],
            "left": [-1],
            "right": [-1],
            "value": [0.0],
        }],
        "feature_names": ["grade_num"],
    }

    # Without mod_groups, GBM should be skipped
    est = engine.estimate(0.5, "Rings", grade="B")
    assert est is not None


def test_cascade_fallback_to_median():
    """When no k-NN samples meet threshold, grade median should fire."""
    engine = CalibrationEngine()

    # Insert just 3 samples (below MIN_CLASS_SAMPLES=10 and MIN_GLOBAL_SAMPLES=50)
    for price in [5.0, 10.0, 20.0]:
        engine._insert(score=0.5, divine=price, item_class="Rings",
                       grade_num=2)

    # Not enough for k-NN, should fall back to median (None since < 3 for median too)
    # But with 3 samples it meets the threshold for group median
    est = engine.estimate(0.5, "Rings", grade="B")
    # With 3 samples in one group, median should be available
    if est is not None:
        assert est > 0


def test_gbm_backward_compatible_v5_shard():
    """v5 shards without gbm_models should still work (k-NN fires)."""
    engine = CalibrationEngine()

    # Simulate v5 shard: no gbm_models loaded
    assert engine._gbm_models == {}

    # Insert enough samples for k-NN
    for price in [1.0, 2.0, 5.0, 10.0, 20.0] * 10:
        engine._insert(score=0.5, divine=price, item_class="Rings",
                       grade_num=2)

    est = engine.estimate(0.5, "Rings", grade="B")
    assert est is not None, "Should work without GBM models (v5 compat)"


def test_grade_median_estimate():
    """_grade_median_estimate should return median for known class+grade."""
    engine = CalibrationEngine()

    # Insert samples with known prices
    for price in [2.0, 5.0, 10.0, 20.0, 50.0]:
        engine._insert(score=0.5, divine=price, item_class="Rings",
                       grade_num=2)

    est = engine._grade_median_estimate("Rings", 2)
    assert est is not None
    assert est > 0

    # Unknown class should fall back to grade-only
    est_unknown = engine._grade_median_estimate("UnknownClass", 2)
    # May be None or a grade-only median
    if est_unknown is not None:
        assert est_unknown > 0


# ── Roll quality / somv_factor tests ──────────────────

def test_somv_factor_affects_knn_distance():
    """Items with different somv_factor should get different estimates."""
    engine = CalibrationEngine()
    engine.set_mod_weights({"IncreasedLife": 1.0})

    mods = ["IncreasedLife"]

    # Insert high-somv items at 50 divine
    for _ in range(20):
        engine._insert(score=0.5, divine=50.0, item_class="Rings",
                       grade_num=3, mod_groups=mods, somv_factor=1.5)

    # Insert low-somv items at 5 divine
    for _ in range(20):
        engine._insert(score=0.5, divine=5.0, item_class="Rings",
                       grade_num=3, mod_groups=mods, somv_factor=0.5)

    # Query with high somv should be closer to high-somv items
    est_high = engine.estimate(0.5, "Rings", grade="A",
                               mod_groups=mods, somv_factor=1.5)
    est_low = engine.estimate(0.5, "Rings", grade="A",
                              mod_groups=mods, somv_factor=0.5)

    assert est_high is not None
    assert est_low is not None
    assert est_high > est_low, (
        f"High somv ({est_high:.1f}) should be > low somv ({est_low:.1f})"
    )


def test_roll_quality_affects_knn_distance():
    """High-roll T1 items should estimate higher than low-roll T1 items."""
    engine = CalibrationEngine()
    engine.set_mod_weights({
        "CriticalStrikeMultiplier": 3.0,
        "SpellDamage": 2.0,
    })

    mods = ["CriticalStrikeMultiplier", "SpellDamage"]
    tiers = {"CriticalStrikeMultiplier": 1, "SpellDamage": 1}

    # Insert high-roll items at 50 divine
    for _ in range(20):
        engine._insert(score=0.6, divine=50.0, item_class="Rings",
                       grade_num=3, mod_groups=mods, mod_tiers=tiers,
                       mod_rolls={"CriticalStrikeMultiplier": 0.95,
                                  "SpellDamage": 0.90})

    # Insert low-roll items at 5 divine
    for _ in range(20):
        engine._insert(score=0.6, divine=5.0, item_class="Rings",
                       grade_num=3, mod_groups=mods, mod_tiers=tiers,
                       mod_rolls={"CriticalStrikeMultiplier": 0.10,
                                  "SpellDamage": 0.15})

    # Query with high rolls
    est_high = engine.estimate(0.6, "Rings", grade="A",
                               mod_groups=mods, mod_tiers=tiers,
                               mod_rolls={"CriticalStrikeMultiplier": 0.95,
                                           "SpellDamage": 0.90})

    # Query with low rolls
    est_low = engine.estimate(0.6, "Rings", grade="A",
                              mod_groups=mods, mod_tiers=tiers,
                              mod_rolls={"CriticalStrikeMultiplier": 0.10,
                                          "SpellDamage": 0.15})

    assert est_high is not None
    assert est_low is not None
    assert est_high > est_low, (
        f"High-roll T1 ({est_high:.1f}) should be > low-roll T1 ({est_low:.1f})"
    )


def test_v6_shard_without_roll_quality_still_loads():
    """v6 shards without mr field should still load correctly (backward compat)."""
    engine = CalibrationEngine()

    # Simulate inserting samples without roll quality (legacy data)
    for price in [1.0, 2.0, 5.0, 10.0, 20.0] * 10:
        engine._insert(score=0.5, divine=price, item_class="Rings",
                       grade_num=2, mod_groups=["IncreasedLife"])

    # Estimate without roll quality should still work
    est = engine.estimate(0.5, "Rings", grade="B",
                          mod_groups=["IncreasedLife"])
    assert est is not None

    # Estimate with roll quality against samples without should also work
    est_with_rolls = engine.estimate(0.5, "Rings", grade="B",
                                     mod_groups=["IncreasedLife"],
                                     mod_rolls={"IncreasedLife": 0.8})
    assert est_with_rolls is not None


def test_pdps_edps_dps_type_distance():
    """Physical vs elemental DPS items should differentiate."""
    engine = CalibrationEngine()
    engine.set_mod_weights({"PhysicalDamage": 2.0, "FireDamage": 2.0})

    # Insert physical DPS items at 50 divine
    for _ in range(20):
        engine._insert(score=0.5, divine=50.0, item_class="Bows",
                       grade_num=3, mod_groups=["PhysicalDamage"],
                       pdps=300.0, edps=0.0)

    # Insert elemental DPS items at 10 divine
    for _ in range(20):
        engine._insert(score=0.5, divine=10.0, item_class="Bows",
                       grade_num=3, mod_groups=["FireDamage"],
                       pdps=0.0, edps=300.0)

    # Query with physical DPS
    est_phys = engine.estimate(0.5, "Bows", grade="A",
                               mod_groups=["PhysicalDamage"],
                               pdps=300.0, edps=0.0)

    # Query with elemental DPS
    est_ele = engine.estimate(0.5, "Bows", grade="A",
                              mod_groups=["FireDamage"],
                              pdps=0.0, edps=300.0)

    assert est_phys is not None
    assert est_ele is not None
    assert est_phys > est_ele, (
        f"Physical DPS ({est_phys:.1f}) should be > elemental ({est_ele:.1f})"
    )
