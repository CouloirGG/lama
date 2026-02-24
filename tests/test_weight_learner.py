"""Tests for weight_learner — Ridge regression training and prediction."""

import math
import random

import pytest

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from weight_learner import LearnedWeights, train_weights, SYNERGY_PAIRS

numpy = pytest.importorskip("numpy", reason="numpy required for training tests")


# ── Helpers ──────────────────────────────────────────────────

def _make_synthetic_records(n=500, seed=42):
    """Generate records with known price structure for Rings.

    True mod effects (in log-divine space):
        CriticalStrikeMultiplier: +1.5
        SpellDamage: +1.0
        IncreasedLife: +0.5
        FireResist: -0.1
        Accuracy: -0.3

    Base price (intercept): log(2.0) ~ 0.693
    """
    rng = random.Random(seed)

    MOD_POOL = [
        "CriticalStrikeMultiplier",
        "SpellDamage",
        "IncreasedLife",
        "FireResist",
        "ColdResist",
        "Accuracy",
        "Thorns",
        "ItemRarity",
    ]
    TRUE_EFFECTS = {
        "CriticalStrikeMultiplier": 1.5,
        "SpellDamage": 1.0,
        "IncreasedLife": 0.5,
        "FireResist": -0.1,
        "ColdResist": 0.0,
        "Accuracy": -0.3,
        "Thorns": -0.2,
        "ItemRarity": -0.15,
    }

    BASE_TYPES = ["Prismatic Ring", "Coral Ring", "Sapphire Ring"]
    BASE_EFFECTS = {"Prismatic Ring": 0.5, "Coral Ring": 0.0, "Sapphire Ring": -0.2}

    records = []
    for _ in range(n):
        # Random 3-5 mods
        n_mods = rng.randint(3, 5)
        mods = rng.sample(MOD_POOL, min(n_mods, len(MOD_POOL)))
        base = rng.choice(BASE_TYPES)
        grade = rng.choice([1, 2, 3, 4])

        # Compute true log-price
        log_price = math.log(2.0)  # intercept
        for m in mods:
            log_price += TRUE_EFFECTS.get(m, 0)
        log_price += BASE_EFFECTS.get(base, 0)
        log_price += 0.2 * (grade - 2)  # grade effect
        log_price += rng.gauss(0, 0.3)  # noise

        price = max(0.01, math.exp(log_price))

        records.append({
            "c": "Rings",
            "g": grade,
            "p": price,
            "t": rng.randint(0, 3),
            "n": len(mods),
            "d": 1.0,
            "f": 1.0,
            "v": 1.0,
            "mod_groups_resolved": mods,
            "base_type_resolved": base,
        })

    return records


# ── TestLearnedWeights (unit tests) ─────────────────────────

class TestLearnedWeights:

    def test_round_trip_serialization(self):
        """to_dict -> from_dict produces identical predictions."""
        records = _make_synthetic_records(200)
        lw = train_weights(records, min_class_samples=20, min_mod_frequency=5)
        assert lw.has_model("Rings")

        # Serialize and deserialize
        data = lw.to_dict()
        lw2 = LearnedWeights.from_dict(data)

        # Compare predictions
        for rec in records[:20]:
            p1 = lw.predict("Rings", rec["mod_groups_resolved"],
                            rec["base_type_resolved"], rec["g"],
                            rec["t"], rec["n"], rec["d"], rec["f"], rec["v"])
            p2 = lw2.predict("Rings", rec["mod_groups_resolved"],
                             rec["base_type_resolved"], rec["g"],
                             rec["t"], rec["n"], rec["d"], rec["f"], rec["v"])
            assert p1 == p2, f"Predictions differ: {p1} vs {p2}"

    def test_predict_unknown_class_returns_none(self):
        """Unknown item class should return None."""
        records = _make_synthetic_records(200)
        lw = train_weights(records, min_class_samples=20, min_mod_frequency=5)
        result = lw.predict("Wands", ["SpellDamage"], "Wand Base",
                            3, 1, 4, 1.0, 1.0)
        assert result is None

    def test_predict_unknown_mod_ignored(self):
        """Unknown mod groups should be ignored (coeff=0, no crash)."""
        records = _make_synthetic_records(200)
        lw = train_weights(records, min_class_samples=20, min_mod_frequency=5)

        # Predict with known mods
        p_known = lw.predict("Rings", ["SpellDamage"], "Prismatic Ring",
                             3, 1, 4, 1.0, 1.0)
        # Predict with known + unknown mods
        p_unknown = lw.predict("Rings", ["SpellDamage", "TotallyFakeMod"],
                               "Prismatic Ring", 3, 1, 4, 1.0, 1.0)

        assert p_known is not None
        assert p_unknown is not None
        # Unknown mod has zero effect — predictions should be identical
        assert abs(p_known - p_unknown) < 0.01

    def test_mod_coefficients_affect_price(self):
        """Positive coefficients should increase price, negative decrease."""
        records = _make_synthetic_records(500)
        lw = train_weights(records, min_class_samples=20, min_mod_frequency=5)

        # CritMulti is strongly positive
        p_crit = lw.predict("Rings", ["CriticalStrikeMultiplier"],
                            "Coral Ring", 2, 1, 4, 1.0, 1.0)
        # Accuracy is negative
        p_acc = lw.predict("Rings", ["Accuracy"],
                           "Coral Ring", 2, 1, 4, 1.0, 1.0)
        # Baseline (no valuable mods)
        p_base = lw.predict("Rings", ["ColdResist"],
                            "Coral Ring", 2, 1, 4, 1.0, 1.0)

        assert p_crit is not None
        assert p_acc is not None
        assert p_base is not None
        assert p_crit > p_base, "CritMulti should increase price"
        assert p_acc < p_base or abs(p_acc - p_base) / p_base < 0.5, \
            "Accuracy should not significantly increase price"

    def test_price_clamped_to_valid_range(self):
        """Extreme intercepts should be clamped to [0.01, 1500]."""
        lw = LearnedWeights()
        # Model with extreme positive intercept
        lw._models["TestClass"] = {
            "intercept": 20.0,  # e^20 ~ 485 million
            "mod_coeffs": {},
            "base_coeffs": {},
            "synergy_coeffs": {},
            "numeric_coeffs": {},
            "n_train": 100,
            "r2_cv": 0.5,
        }
        p = lw.predict("TestClass", ["SomeMod"], "", 2, 1, 4, 1.0, 1.0)
        assert p is not None
        assert p <= 1500.0

        # Model with extreme negative intercept
        lw._models["TestClass2"] = {
            "intercept": -20.0,  # e^-20 ~ 0.000000002
            "mod_coeffs": {},
            "base_coeffs": {},
            "synergy_coeffs": {},
            "numeric_coeffs": {},
            "n_train": 100,
            "r2_cv": 0.5,
        }
        p2 = lw.predict("TestClass2", ["SomeMod"], "", 2, 1, 4, 1.0, 1.0)
        assert p2 is not None
        assert p2 >= 0.01


# ── TestTrainWeights (integration with synthetic data) ──────

class TestTrainWeights:

    def test_train_produces_models(self):
        """Training should produce a model for Rings."""
        records = _make_synthetic_records(500)
        lw = train_weights(records, min_class_samples=20, min_mod_frequency=5)
        assert lw.has_model("Rings")
        model = lw._models["Rings"]
        assert model["n_train"] == 500
        assert "intercept" in model
        assert len(model["mod_coeffs"]) > 0

    def test_learned_coefficients_match_direction(self):
        """CritMulti should be positive, Accuracy should be negative."""
        records = _make_synthetic_records(500)
        lw = train_weights(records, min_class_samples=20, min_mod_frequency=5)
        model = lw._models["Rings"]
        mod_coeffs = model["mod_coeffs"]

        # CritMulti should have a positive coefficient
        assert "CriticalStrikeMultiplier" in mod_coeffs
        assert mod_coeffs["CriticalStrikeMultiplier"] > 0, \
            f"CritMulti coeff should be positive, got {mod_coeffs['CriticalStrikeMultiplier']}"

        # Accuracy should have a negative coefficient
        if "Accuracy" in mod_coeffs:
            assert mod_coeffs["Accuracy"] < 0, \
                f"Accuracy coeff should be negative, got {mod_coeffs['Accuracy']}"

    def test_prediction_accuracy_on_synthetic(self):
        """>=60% within 2x on synthetic 80/20 split."""
        records = _make_synthetic_records(500)
        split = int(len(records) * 0.8)
        train_recs = records[:split]
        test_recs = records[split:]

        lw = train_weights(train_recs, min_class_samples=20, min_mod_frequency=5)
        assert lw.has_model("Rings")

        within_2x = 0
        total = 0
        for rec in test_recs:
            est = lw.predict("Rings", rec["mod_groups_resolved"],
                             rec["base_type_resolved"], rec["g"],
                             rec["t"], rec["n"], rec["d"], rec["f"], rec["v"])
            if est is None:
                continue
            actual = rec["p"]
            total += 1
            ratio = max(est / actual, actual / est)
            if ratio <= 2.0:
                within_2x += 1

        assert total > 0
        pct = within_2x / total * 100
        assert pct >= 60.0, (
            f"Synthetic accuracy {pct:.1f}% is below 60% target "
            f"({within_2x}/{total} within 2x)"
        )

    def test_skip_small_classes(self):
        """Classes with < min_class_samples should produce no model."""
        records = _make_synthetic_records(30)  # only 30 samples
        lw = train_weights(records, min_class_samples=50)
        assert not lw.has_model("Rings")

    def test_empty_records(self):
        """Empty input should not crash."""
        lw = train_weights([])
        assert len(lw._models) == 0

    def test_summary_output(self):
        """summary() should return a non-empty string."""
        records = _make_synthetic_records(200)
        lw = train_weights(records, min_class_samples=20, min_mod_frequency=5)
        s = lw.summary()
        assert "Learned weights" in s
        assert "Rings" in s
