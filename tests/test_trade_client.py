"""Tests for trade_client.py — stat filter building, common mod classification."""

import pytest

from mod_parser import ParsedMod
from trade_client import TradeClient, RarePriceResult


# ── Fixtures ─────────────────────────────────────────────

@pytest.fixture
def client():
    """Create a TradeClient (no actual API calls in these tests)."""
    return TradeClient(
        league="Fate of the Vaal",
        divine_to_chaos_fn=lambda: 68.0,
        divine_to_exalted_fn=lambda: 300.0,
    )


# ── Common mod pattern matching ──────────────────────────

def test_common_mod_patterns_filler():
    """Known filler mods match common patterns."""
    filler_texts = [
        "+30% to Fire Resistance",
        "+25 to Strength",
        "+50 to maximum Mana",
        "+100 to Accuracy Rating",
        "5 to 10 Physical Thorns damage",
        "16% increased Rarity of Items found",
    ]
    for text in filler_texts:
        text_lower = text.lower()
        matched = any(pat in text_lower for pat in TradeClient._COMMON_MOD_PATTERNS)
        assert matched, f"Filler mod should match common patterns: {text!r}"


def test_common_mod_patterns_key():
    """Known key mods do NOT match common patterns."""
    key_texts = [
        "35% increased Movement Speed",
        "40% increased Critical Damage Bonus",
        "90% increased Spell Damage",
        "24% increased Attack Speed",
        "+3 to Level of all Spell Skills",
    ]
    for text in key_texts:
        text_lower = text.lower()
        matched = any(pat in text_lower for pat in TradeClient._COMMON_MOD_PATTERNS)
        assert not matched, f"Key mod should NOT match common patterns: {text!r}"


# ── Stat filter building ────────────────────────────────

def test_build_stat_filters(client):
    """ParsedMod list → correct trade API filter structure."""
    mods = [
        ParsedMod(raw_text="+42 to maximum Life",
                  stat_id="explicit.stat_3299347043",
                  value=42.0, mod_type="explicit"),
        ParsedMod(raw_text="+30% to Fire Resistance",
                  stat_id="explicit.stat_fire_res",
                  value=30.0, mod_type="explicit"),
    ]
    filters = client._build_stat_filters(mods)
    assert len(filters) == 2
    for f in filters:
        assert "id" in f
        assert "value" in f
        assert "min" in f["value"]
        assert f["value"]["min"] > 0


def test_stat_filter_min_value(client):
    """Min value uses value-dependent tightness."""
    # Low value (1-10): 95%
    assert client._compute_min_value(10.0) == pytest.approx(9.5)
    # Medium value (11-50): 90%
    assert client._compute_min_value(50.0) == pytest.approx(45.0)
    # High value (51+): 80% (TRADE_MOD_MIN_MULTIPLIER)
    assert client._compute_min_value(100.0) == pytest.approx(80.0)


def test_stat_filter_negative_value(client):
    """Negative values preserve sign with correct multiplier."""
    result = client._compute_min_value(-20.0)
    # -20 * 0.90 = -18
    assert result == pytest.approx(-18.0)


def test_fractured_mod_filter(client):
    """Fractured mods are included in priceable filters."""
    mods = [
        ParsedMod(raw_text="88% increased Spell Damage",
                  stat_id="fractured.stat_spell_dmg",
                  value=88.0, mod_type="fractured"),
    ]
    filters = client._build_stat_filters(mods)
    assert len(filters) == 1
    assert filters[0]["id"] == "fractured.stat_spell_dmg"


def test_build_stat_filters_custom():
    """Custom multiplier applies correctly."""
    mods = [
        ParsedMod(raw_text="+100 to maximum Life",
                  stat_id="explicit.stat_life",
                  value=100.0, mod_type="explicit"),
    ]
    filters = TradeClient._build_stat_filters_custom(mods, 0.85)
    assert len(filters) == 1
    assert filters[0]["value"]["min"] == 85


# ── RarePriceResult formatting ───────────────────────────

def test_rare_price_result_display():
    """RarePriceResult fields are correct."""
    result = RarePriceResult(
        min_price=5.0,
        max_price=8.0,
        num_results=15,
        display="~5-8 Divine",
        tier="good",
    )
    assert result.min_price == 5.0
    assert result.max_price == 8.0
    assert result.num_results == 15
    assert result.tier == "good"
    assert "Divine" in result.display
    assert result.estimate is False


def test_rare_price_result_estimate():
    """Estimate flag set correctly."""
    result = RarePriceResult(
        min_price=10.0,
        max_price=10.0,
        num_results=5,
        display="10+ Divine (est.)",
        tier="high",
        estimate=True,
    )
    assert result.estimate is True


# ── Price tier classification ────────────────────────────

def test_determine_tier(client):
    """Price tiers classified correctly."""
    assert client._determine_tier(10.0) == "high"
    assert client._determine_tier(5.0) == "high"
    assert client._determine_tier(2.0) == "good"
    assert client._determine_tier(0.5) == "decent"
    assert client._determine_tier(0.01) == "low"


# ── Classify filters ─────────────────────────────────────

def test_classify_filters(client):
    """Key mods separated from common mods."""
    mods = [
        ParsedMod(raw_text="40% increased Critical Damage Bonus",
                  stat_id="explicit.stat_crit",
                  value=40.0, mod_type="explicit"),
        ParsedMod(raw_text="+30% to Fire Resistance",
                  stat_id="explicit.stat_fire",
                  value=30.0, mod_type="explicit"),
        ParsedMod(raw_text="+25 to Strength",
                  stat_id="explicit.stat_str",
                  value=25.0, mod_type="explicit"),
    ]
    filters = client._build_stat_filters(mods)
    key_m, common_m, key_f, common_f = client._classify_filters(mods, filters)

    # Crit multi is a key mod
    assert len(key_m) >= 1
    assert any("Critical Damage" in m.raw_text for m in key_m)

    # Fire res and strength are common
    assert len(common_m) >= 1
    assert any("Fire Resistance" in m.raw_text for m in common_m)


def test_classify_implicit_as_common(client):
    """Implicit mods always classified as common."""
    mods = [
        ParsedMod(raw_text="40% increased Critical Damage Bonus",
                  stat_id="implicit.stat_crit",
                  value=40.0, mod_type="implicit"),
    ]
    filters = client._build_stat_filters(mods)
    key_m, common_m, key_f, common_f = client._classify_filters(mods, filters)
    assert len(common_m) == 1
    assert len(key_m) == 0


# ── Weight-table-backed classification ──────────────────

class _MockModDatabase:
    """Lightweight mock of ModDatabase for classify_mod tests."""

    def __init__(self, classifications: dict, loaded: bool = True):
        """classifications: stat_id -> bool (True=key, False=common)"""
        self._classifications = classifications
        self.loaded = loaded

    def classify_mod(self, stat_id, raw_text, mod_type):
        return self._classifications.get(stat_id, False)


def test_classify_with_mod_database_leech_as_key():
    """Life leech (weight 2.0 in weight table) should be key, not common."""
    mock_db = _MockModDatabase({
        "explicit.stat_lifeleech": True,   # weight table: Key (2.0)
        "explicit.stat_fire_res": False,   # weight table: Common (0.3)
    })
    tc = TradeClient(league="Fate of the Vaal", mod_database=mock_db)
    mods = [
        ParsedMod(raw_text="0.4% of Physical Damage leeches Life",
                  stat_id="explicit.stat_lifeleech",
                  value=0.4, mod_type="explicit"),
        ParsedMod(raw_text="+30% to Fire Resistance",
                  stat_id="explicit.stat_fire_res",
                  value=30.0, mod_type="explicit"),
    ]
    filters = tc._build_stat_filters(mods)
    key_m, common_m, key_f, common_f = tc._classify_filters(mods, filters)
    assert len(key_m) == 1
    assert "leech" in key_m[0].raw_text.lower()
    assert len(common_m) == 1
    assert "Fire Resistance" in common_m[0].raw_text


def test_classify_with_mod_database_added_fire_as_key():
    """Added Fire Damage (weight 2.0) should be key, not caught by pattern."""
    mock_db = _MockModDatabase({
        "explicit.stat_added_fire": True,
    })
    tc = TradeClient(league="Fate of the Vaal", mod_database=mock_db)
    mods = [
        ParsedMod(raw_text="Adds 10 to 20 Fire Damage to Attacks",
                  stat_id="explicit.stat_added_fire",
                  value=15.0, mod_type="explicit"),
    ]
    filters = tc._build_stat_filters(mods)
    key_m, common_m, key_f, common_f = tc._classify_filters(mods, filters)
    assert len(key_m) == 1
    assert "Fire Damage" in key_m[0].raw_text
    assert len(common_m) == 0


def test_classify_fallback_without_mod_database(client):
    """Without mod_database, fallback pattern matching still works."""
    # client fixture has no mod_database — uses pattern fallback
    mods = [
        ParsedMod(raw_text="40% increased Critical Damage Bonus",
                  stat_id="explicit.stat_crit",
                  value=40.0, mod_type="explicit"),
        ParsedMod(raw_text="+30% to Fire Resistance",
                  stat_id="explicit.stat_fire_res",
                  value=30.0, mod_type="explicit"),
    ]
    filters = client._build_stat_filters(mods)
    key_m, common_m, key_f, common_f = client._classify_filters(mods, filters)
    assert len(key_m) == 1
    assert "Critical Damage" in key_m[0].raw_text
    assert len(common_m) == 1
    assert "Fire Resistance" in common_m[0].raw_text


def test_classify_implicit_common_with_mod_database():
    """Implicit mods remain common even when mod_database is loaded."""
    mock_db = _MockModDatabase({
        "implicit.stat_crit": False,  # classify_mod returns False for implicits
    })
    tc = TradeClient(league="Fate of the Vaal", mod_database=mock_db)
    mods = [
        ParsedMod(raw_text="40% increased Critical Damage Bonus",
                  stat_id="implicit.stat_crit",
                  value=40.0, mod_type="implicit"),
    ]
    filters = tc._build_stat_filters(mods)
    key_m, common_m, key_f, common_f = tc._classify_filters(mods, filters)
    assert len(common_m) == 1
    assert len(key_m) == 0


def test_classify_unloaded_mod_database_uses_fallback():
    """When mod_database exists but isn't loaded, use pattern fallback."""
    mock_db = _MockModDatabase(
        {"explicit.stat_lifeleech": True},
        loaded=False,
    )
    tc = TradeClient(league="Fate of the Vaal", mod_database=mock_db)
    mods = [
        ParsedMod(raw_text="0.4% of Physical Damage leeches Life",
                  stat_id="explicit.stat_lifeleech",
                  value=0.4, mod_type="explicit"),
    ]
    filters = tc._build_stat_filters(mods)
    key_m, common_m, key_f, common_f = tc._classify_filters(mods, filters)
    # Fallback: "leech" pattern catches it as common
    assert len(common_m) == 1
    assert len(key_m) == 0
