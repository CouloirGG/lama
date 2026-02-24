"""Tests for shard_generator — IQR outlier removal and mod groups in shards."""

import math
import random

import pytest

from shard_generator import (remove_outliers, compact_record, OUTLIER_IQR_MULTIPLIER,
                             _compute_tier_aggregates, _enrich_record,
                             _SHORT_TO_GROUP)


def _make_rec(price, grade="C", item_class="Rings", mod_groups=None,
              base_type=None, mod_tiers=None, top_mods=None):
    rec = {"min_divine": price, "grade": grade, "item_class": item_class,
           "score": 0.5, "top_tier_count": 0, "mod_count": 4}
    if mod_groups is not None:
        rec["mod_groups"] = mod_groups
    if base_type is not None:
        rec["base_type"] = base_type
    if mod_tiers is not None:
        rec["mod_tiers"] = mod_tiers
    if top_mods is not None:
        rec["top_mods"] = top_mods
    return rec


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


# ── Mod groups in shards ────────────────────────────────

class TestModGroupsInShard:
    def test_compact_record_with_mod_groups(self):
        """compact_record should include 'm' field with integer indices."""
        mod_to_idx = {"IncreasedLife": 0, "FireResist": 1, "ColdResist": 2}
        rec = _make_rec(5.0, mod_groups=["IncreasedLife", "ColdResist"])
        compact = compact_record(rec, mod_to_idx)

        assert "m" in compact
        assert compact["m"] == [0, 2]  # sorted indices

    def test_compact_record_without_mod_groups(self):
        """compact_record should omit 'm' when no mod_groups present."""
        mod_to_idx = {"IncreasedLife": 0}
        rec = _make_rec(5.0)
        compact = compact_record(rec, mod_to_idx)

        assert "m" not in compact

    def test_compact_record_no_mod_index(self):
        """compact_record with no mod_to_idx should not include 'm'."""
        rec = _make_rec(5.0, mod_groups=["IncreasedLife"])
        compact = compact_record(rec)

        assert "m" not in compact

    def test_compact_record_deduplicates_mod_groups(self):
        """Duplicate mod groups should be collapsed."""
        mod_to_idx = {"IncreasedLife": 0, "FireResist": 1}
        rec = _make_rec(5.0, mod_groups=["IncreasedLife", "IncreasedLife", "FireResist"])
        compact = compact_record(rec, mod_to_idx)

        assert compact["m"] == [0, 1]


class TestBaseTypeInShard:
    def test_compact_record_with_base_type(self):
        """compact_record should include 'b' field with base type index."""
        base_to_idx = {"Astral Plate": 0, "Simple Robe": 1}
        rec = _make_rec(5.0, base_type="Astral Plate")
        compact = compact_record(rec, base_to_idx=base_to_idx)

        assert "b" in compact
        assert compact["b"] == 0

    def test_compact_record_without_base_type(self):
        """compact_record should omit 'b' when no base_type present."""
        base_to_idx = {"Astral Plate": 0}
        rec = _make_rec(5.0)
        compact = compact_record(rec, base_to_idx=base_to_idx)

        assert "b" not in compact

    def test_compact_record_no_base_index(self):
        """compact_record with no base_to_idx should not include 'b'."""
        rec = _make_rec(5.0, base_type="Astral Plate")
        compact = compact_record(rec)

        assert "b" not in compact

    def test_compact_record_unknown_base_type(self):
        """compact_record should omit 'b' for base types not in index."""
        base_to_idx = {"Astral Plate": 0}
        rec = _make_rec(5.0, base_type="Unknown Base")
        compact = compact_record(rec, base_to_idx=base_to_idx)

        assert "b" not in compact


# ── Mod tiers in shards ───────────────────────────────

class TestModTiersInShard:
    def test_compact_record_with_mod_tiers(self):
        """compact_record should include 'mt' parallel to 'm' with correct tier numbers."""
        mod_to_idx = {"IncreasedLife": 0, "FireResist": 1, "ColdResist": 2}
        rec = _make_rec(5.0,
                        mod_groups=["IncreasedLife", "ColdResist"],
                        mod_tiers={"IncreasedLife": 1, "ColdResist": 4})
        compact = compact_record(rec, mod_to_idx)

        assert "mt" in compact
        assert compact["m"] == [0, 2]  # sorted indices
        assert compact["mt"] == [1, 4]  # parallel tiers
        assert len(compact["mt"]) == len(compact["m"])

    def test_compact_record_tier_aggregates(self):
        """compact_record should include 'ts', 'bt', 'at' computed from mod_tiers."""
        mod_to_idx = {"IncreasedLife": 0, "FireResist": 1}
        rec = _make_rec(5.0,
                        mod_groups=["IncreasedLife", "FireResist"],
                        mod_tiers={"IncreasedLife": 1, "FireResist": 4})
        compact = compact_record(rec, mod_to_idx)

        assert "ts" in compact
        assert "bt" in compact
        assert "at" in compact
        # tier_score = 1/1 + 1/4 = 1.25
        assert compact["ts"] == 1.25
        # best_tier = min(1, 4) = 1
        assert compact["bt"] == 1
        # avg_tier = (1 + 4) / 2 = 2.5
        assert compact["at"] == 2.5

    def test_compact_record_tier_from_top_mods(self):
        """Tier aggregates should fall back to parsing top_mods string."""
        rec = _make_rec(5.0, top_mods="T1 CastSpd, T4 Mana")
        compact = compact_record(rec)

        assert "ts" in compact
        # tier_score = 1/1 + 1/4 = 1.25
        assert compact["ts"] == 1.25
        assert compact["bt"] == 1
        assert compact["at"] == 2.5

    def test_compact_record_no_tiers(self):
        """compact_record without tier data should omit mt/ts/bt/at."""
        mod_to_idx = {"IncreasedLife": 0}
        rec = _make_rec(5.0, mod_groups=["IncreasedLife"])
        compact = compact_record(rec, mod_to_idx)

        assert "mt" not in compact
        assert "ts" not in compact


class TestComputeTierAggregates:
    def test_with_mod_tiers(self):
        rec = {"mod_tiers": {"A": 1, "B": 2, "C": 3}}
        ts, bt, at = _compute_tier_aggregates(rec)
        assert ts == round(1.0 + 0.5 + 1/3, 3)
        assert bt == 1
        assert at == 2.0

    def test_with_top_mods_fallback(self):
        rec = {"top_mods": "T2 CritMulti, T5 Life"}
        ts, bt, at = _compute_tier_aggregates(rec)
        assert ts == round(0.5 + 0.2, 3)
        assert bt == 2
        assert at == 3.5

    def test_empty_returns_zeros(self):
        ts, bt, at = _compute_tier_aggregates({})
        assert ts == 0.0
        assert bt == 0
        assert at == 0.0


# ── Feature enrichment tests ────────────────────────────

class TestEnrichRecord:

    def test_enrich_populates_mod_groups(self):
        """top_mods string should be parsed into mod_groups."""
        rec = _make_rec(5.0, top_mods="T1 CastSpd, T4 Mana")
        _enrich_record(rec)
        assert "mod_groups" in rec
        assert "CastSpeed" in rec["mod_groups"]
        assert "MaximumMana" in rec["mod_groups"]

    def test_enrich_populates_mod_tiers(self):
        """top_mods should populate mod_tiers with correct tier numbers."""
        rec = _make_rec(5.0, top_mods="T1 CritMulti, T3 SpellDmg")
        _enrich_record(rec)
        assert rec.get("mod_tiers", {}).get("CriticalStrikeMultiplier") == 1
        assert rec.get("mod_tiers", {}).get("SpellDamage") == 3

    def test_enrich_preserves_existing_mod_groups(self):
        """Existing mod_groups should not be overwritten."""
        rec = _make_rec(5.0,
                        mod_groups=["IncreasedLife", "FireResist"],
                        top_mods="T1 CritMulti, T2 SpellDmg")
        _enrich_record(rec)
        # Should keep original groups
        assert rec["mod_groups"] == ["IncreasedLife", "FireResist"]

    def test_enrich_adds_tiers_to_existing_groups(self):
        """When mod_groups exist but mod_tiers don't, should add tiers from top_mods."""
        rec = _make_rec(5.0,
                        mod_groups=["CastSpeed", "MaximumMana"],
                        top_mods="T1 CastSpd, T4 Mana")
        _enrich_record(rec)
        # Should add tiers without overwriting groups
        assert rec["mod_groups"] == ["CastSpeed", "MaximumMana"]
        assert rec.get("mod_tiers", {}).get("CastSpeed") == 1
        assert rec.get("mod_tiers", {}).get("MaximumMana") == 4

    def test_enrich_preserves_existing_mod_tiers(self):
        """Existing mod_tiers should not be overwritten."""
        rec = _make_rec(5.0,
                        mod_groups=["IncreasedLife"],
                        mod_tiers={"IncreasedLife": 2},
                        top_mods="T1 Life")
        _enrich_record(rec)
        # Should keep original tiers
        assert rec["mod_tiers"]["IncreasedLife"] == 2

    def test_enrich_no_top_mods_no_change(self):
        """Without top_mods, record should be unchanged."""
        rec = _make_rec(5.0)
        _enrich_record(rec)
        assert "mod_groups" not in rec or rec.get("mod_groups") is None

    def test_enrich_unknown_short_names_skipped(self):
        """Short names not in _SHORT_TO_GROUP should be skipped."""
        rec = _make_rec(5.0, top_mods="T1 UnknownMod, T2 CastSpd")
        _enrich_record(rec)
        groups = rec.get("mod_groups", [])
        # UnknownMod shouldn't appear, CastSpeed should
        assert "CastSpeed" in groups
        assert len(groups) == 1


# ── Mod rolls in shards ─────────────────────────────

class TestModRollsInShard:
    def test_compact_record_with_mod_rolls(self):
        """compact_record should include 'mr' parallel to 'm' with roll quality values."""
        mod_to_idx = {"IncreasedLife": 0, "FireResist": 1, "ColdResist": 2}
        rec = _make_rec(5.0,
                        mod_groups=["IncreasedLife", "ColdResist"],
                        mod_tiers={"IncreasedLife": 1, "ColdResist": 4})
        rec["mod_rolls"] = {"IncreasedLife": 0.85, "ColdResist": 0.42}
        compact = compact_record(rec, mod_to_idx)

        assert "mr" in compact
        assert compact["m"] == [0, 2]  # sorted indices
        assert compact["mr"] == [0.85, 0.42]  # parallel roll quality
        assert len(compact["mr"]) == len(compact["m"])

    def test_compact_record_without_mod_rolls(self):
        """compact_record without mod_rolls should omit 'mr'."""
        mod_to_idx = {"IncreasedLife": 0, "FireResist": 1}
        rec = _make_rec(5.0, mod_groups=["IncreasedLife"])
        compact = compact_record(rec, mod_to_idx)

        assert "mr" not in compact

    def test_compact_record_partial_mod_rolls(self):
        """mod_rolls with only some mods should produce -1.0 for missing."""
        mod_to_idx = {"IncreasedLife": 0, "FireResist": 1}
        rec = _make_rec(5.0,
                        mod_groups=["IncreasedLife", "FireResist"])
        rec["mod_rolls"] = {"IncreasedLife": 0.7}  # FireResist missing
        compact = compact_record(rec, mod_to_idx)

        assert "mr" in compact
        assert compact["mr"][0] == 0.7
        assert compact["mr"][1] == -1.0  # missing roll = -1


# ── Combat stats in shards ──────────────────────────

class TestCombatStatsInShard:
    def test_compact_record_with_pdps(self):
        """compact_record should include 'pd' when pdps > 0."""
        rec = _make_rec(5.0)
        rec["pdps"] = 250.5
        compact = compact_record(rec)
        assert compact["pd"] == 250.5

    def test_compact_record_without_pdps(self):
        """compact_record should omit 'pd' when pdps = 0."""
        rec = _make_rec(5.0)
        compact = compact_record(rec)
        assert "pd" not in compact

    def test_compact_record_with_defense_stats(self):
        """compact_record should include ar/ev/es when > 0."""
        rec = _make_rec(5.0)
        rec["armour"] = 500
        rec["evasion"] = 300
        rec["energy_shield"] = 200
        compact = compact_record(rec)
        assert compact["ar"] == 500
        assert compact["ev"] == 300
        assert compact["es"] == 200

    def test_compact_record_with_item_level(self):
        """compact_record should include 'il' when > 0."""
        rec = _make_rec(5.0)
        rec["item_level"] = 82
        compact = compact_record(rec)
        assert compact["il"] == 82

    def test_compact_record_zero_stats_omitted(self):
        """Zero combat stats should be omitted from compact."""
        rec = _make_rec(5.0)
        rec["armour"] = 0
        rec["evasion"] = 0
        rec["energy_shield"] = 0
        rec["item_level"] = 0
        rec["pdps"] = 0.0
        rec["edps"] = 0.0
        compact = compact_record(rec)
        assert "ar" not in compact
        assert "ev" not in compact
        assert "es" not in compact
        assert "il" not in compact
        assert "pd" not in compact
        assert "ed" not in compact


class TestShortToGroupMap:

    def test_all_common_labels_mapped(self):
        """Common display labels should all have mappings."""
        common = ["CritMulti", "CritChance", "SpellDmg", "Life", "ES",
                  "AtkSpd", "CastSpd", "MoveSpd", "FireRes", "ColdRes"]
        for label in common:
            assert label in _SHORT_TO_GROUP, f"Missing mapping for {label}"

    def test_mapped_values_are_strings(self):
        """All mapped values should be non-empty strings."""
        for short, group in _SHORT_TO_GROUP.items():
            assert isinstance(group, str) and group, (
                f"Bad mapping: {short} -> {group!r}"
            )
