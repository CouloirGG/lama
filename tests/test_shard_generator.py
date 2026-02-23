"""Tests for shard_generator.remove_outliers — IQR in log-price space."""

import math
import random

import pytest

from shard_generator import remove_outliers, OUTLIER_IQR_MULTIPLIER


def _make_rec(price, grade="C", item_class="Rings"):
    return {"min_divine": price, "grade": grade, "item_class": item_class,
            "score": 0.5, "top_tier_count": 0, "mod_count": 4}


# ── Basic behaviour ──────────────────────────────────────

class TestRemoveOutliersBasic:
    def test_small_group_kept_entirely(self):
        """Groups with < 5 records should be kept as-is."""
        recs = [_make_rec(p) for p in [0.1, 1.0, 100.0, 9999.0]]
        kept, removed = remove_outliers(recs)
        assert removed == 0
        assert len(kept) == 4

    def test_uniform_prices_no_removal(self):
        """Identical prices → IQR=0, everything within fences."""
        recs = [_make_rec(5.0) for _ in range(20)]
        kept, removed = remove_outliers(recs)
        assert removed == 0
        assert len(kept) == 20

    def test_tight_spread_no_removal(self):
        """Prices within a narrow band should all survive."""
        recs = [_make_rec(p) for p in [4.0, 4.5, 5.0, 5.5, 6.0, 5.2, 4.8]]
        kept, removed = remove_outliers(recs)
        assert removed == 0

    def test_extreme_outlier_removed(self):
        """A price many orders of magnitude away should be caught."""
        recs = [_make_rec(p) for p in [1.0]*20 + [1_000_000.0]]
        kept, removed = remove_outliers(recs)
        assert removed >= 1
        prices = [r["min_divine"] for r in kept]
        assert 1_000_000.0 not in prices

    def test_zero_price_removed(self):
        """Records with price <= 0 should be dropped."""
        recs = [_make_rec(p) for p in [1.0]*10 + [0.0, -1.0]]
        kept, removed = remove_outliers(recs)
        assert removed == 2
        assert all(r["min_divine"] > 0 for r in kept)


# ── Wide but legitimate spreads ──────────────────────────

class TestWideSpread:
    def test_log_normal_spread_preserved(self):
        """A realistic log-normal price spread should lose very few records.

        Simulates a (grade, class) group with prices from 0.5 to 50 divine
        — a 100x range that's normal for C-grade rares.
        """
        rng = random.Random(42)
        prices = [math.exp(rng.gauss(1.0, 1.2)) for _ in range(200)]
        # Clamp to realistic range
        prices = [max(0.01, min(p, 1500)) for p in prices]
        recs = [_make_rec(p) for p in prices]

        kept, removed = remove_outliers(recs)
        drop_rate = removed / len(recs)
        assert drop_rate < 0.10, (
            f"Log-normal spread lost {drop_rate:.1%} — too aggressive")

    def test_bimodal_distribution_preserved(self):
        """Two price clusters (cheap + expensive) within same group.

        This is common for item classes where some bases are worth more.
        """
        rng = random.Random(99)
        cheap = [math.exp(rng.gauss(-0.5, 0.5)) for _ in range(50)]
        expensive = [math.exp(rng.gauss(2.5, 0.5)) for _ in range(50)]
        recs = [_make_rec(p) for p in cheap + expensive]

        kept, removed = remove_outliers(recs)
        drop_rate = removed / len(recs)
        assert drop_rate < 0.10, (
            f"Bimodal distribution lost {drop_rate:.1%} — too aggressive")


# ── Grouping ─────────────────────────────────────────────

class TestGrouping:
    def test_groups_are_independent(self):
        """Outlier in one group shouldn't affect another."""
        normal = [_make_rec(5.0, grade="C", item_class="Rings") for _ in range(10)]
        outlier_group = ([_make_rec(5.0, grade="A", item_class="Boots")] * 10
                         + [_make_rec(999999.0, grade="A", item_class="Boots")])
        recs = normal + outlier_group

        kept, removed = remove_outliers(recs)
        ring_kept = [r for r in kept if r["item_class"] == "Rings"]
        assert len(ring_kept) == 10, "Normal group should be untouched"

    def test_multiple_grades_same_class(self):
        """Different grades within the same item class are separate groups."""
        c_recs = [_make_rec(1.0, grade="C", item_class="Helmets") for _ in range(10)]
        a_recs = [_make_rec(50.0, grade="A", item_class="Helmets") for _ in range(10)]
        recs = c_recs + a_recs

        kept, removed = remove_outliers(recs)
        assert removed == 0, "Separate grade groups should not interfere"


# ── Regression: overall drop rate ────────────────────────

class TestDropRateGuard:
    def test_overall_drop_rate_under_20pct(self):
        """Simulate a full harvester dataset and verify < 20% removal.

        Uses 5 grade/class groups with realistic log-normal spreads.
        """
        rng = random.Random(123)
        classes = ["Rings", "Boots", "Helmets", "Wands", "Body Armours"]
        grades = ["C", "C", "JUNK", "B", "A"]
        mus = [0.5, 0.8, -0.3, 1.5, 2.5]
        sigmas = [1.0, 1.2, 0.8, 1.0, 0.9]

        all_recs = []
        for cls, grade, mu, sigma in zip(classes, grades, mus, sigmas):
            prices = [math.exp(rng.gauss(mu, sigma)) for _ in range(500)]
            prices = [max(0.01, min(p, 1500)) for p in prices]
            all_recs.extend(_make_rec(p, grade=grade, item_class=cls) for p in prices)

        kept, removed = remove_outliers(all_recs)
        drop_rate = removed / len(all_recs)
        assert drop_rate < 0.20, (
            f"Overall drop rate {drop_rate:.1%} exceeds 20% cap — "
            f"outlier algorithm is too aggressive")
