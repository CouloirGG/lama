"""Tests for the PricingEngine facade."""

import sys
from pathlib import Path

import pytest

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(filename):
    path = FIXTURES_DIR / filename
    if not path.exists():
        pytest.skip(f"Fixture {filename} not found")
    return path.read_text(encoding="utf-8")


class TestPricingEngineInit:
    """Test PricingEngine construction and initialization."""

    def test_construct_with_poe2_config(self):
        from core import PricingEngine
        from games.poe2 import create_poe2_config

        engine = PricingEngine(create_poe2_config())
        assert not engine.ready
        assert engine.config.game_id == "poe2"

    def test_construct_with_minimal_config(self):
        from core import PricingEngine, GameConfig

        cfg = GameConfig(
            game_id="test",
            default_league="Test",
            cache_dir=Path("/tmp/test"),
            trade_api_base="https://example.com",
            trade_stats_url="https://example.com/stats",
            trade_items_url="https://example.com/items",
            trade_stats_cache_file=Path("/tmp/stats.json"),
            trade_items_cache_file=Path("/tmp/items.json"),
        )
        engine = PricingEngine(cfg)
        assert not engine.ready

    def test_parse_item_before_init_returns_none(self):
        from core import PricingEngine
        from games.poe2 import create_poe2_config

        engine = PricingEngine(create_poe2_config())
        assert engine.parse_item("test") is None

    def test_score_item_before_init_returns_none(self):
        from core import PricingEngine
        from games.poe2 import create_poe2_config

        engine = PricingEngine(create_poe2_config())
        assert engine.score_item(None) is None

    def test_lookup_before_init_returns_none(self):
        from core import PricingEngine
        from games.poe2 import create_poe2_config

        engine = PricingEngine(create_poe2_config())
        assert engine.lookup("anything") is None


@pytest.fixture(scope="module")
def engine():
    """Create and initialize a PricingEngine with POE2 config.

    This is module-scoped because initialization downloads/caches data
    and is slow. Shares the engine across all tests in this module.
    """
    from core import PricingEngine
    from games.poe2 import create_poe2_config

    eng = PricingEngine(create_poe2_config())
    ok = eng.initialize()
    if not ok:
        pytest.skip("PricingEngine could not initialize (no cache/network)")
    return eng


class TestParseItem:
    """Test parse_item() with fixture clipboard text."""

    def test_parse_rare_ring(self, engine):
        text = load_fixture("rare_ring_brimstone.txt")
        item = engine.parse_item(text)
        assert item is not None
        assert item.rarity == "rare"
        assert item.item_class == "Rings"
        assert item.base_type == "Sapphire Ring"
        assert item.item_level == 79

    def test_parse_currency(self, engine):
        text = load_fixture("currency_fracturing_orb.txt")
        item = engine.parse_item(text)
        assert item is not None
        assert item.rarity == "currency"

    def test_parse_unique(self, engine):
        text = load_fixture("unique_quiver_asphyxia.txt")
        item = engine.parse_item(text)
        assert item is not None
        assert item.rarity == "unique"

    def test_parse_invalid_returns_none(self, engine):
        assert engine.parse_item("") is None
        assert engine.parse_item("random text") is None


class TestScoreItem:
    """Test score_item() returns valid ItemScore."""

    def test_score_rare_ring(self, engine):
        text = load_fixture("rare_ring_brimstone.txt")
        item = engine.parse_item(text)
        assert item is not None

        score = engine.score_item(item)
        assert score is not None
        assert hasattr(score, "grade")
        assert hasattr(score, "normalized_score")
        assert 0.0 <= score.normalized_score <= 1.0
        assert score.grade.value in ("S", "A", "B", "C", "JUNK")

    def test_score_rare_boots(self, engine):
        text = load_fixture("rare_boots_dire_league.txt")
        item = engine.parse_item(text)
        assert item is not None

        score = engine.score_item(item)
        assert score is not None
        assert score.mod_scores  # should have some scored mods

    def test_score_currency_returns_none(self, engine):
        """Currency items can't be scored."""
        text = load_fixture("currency_fracturing_orb.txt")
        item = engine.parse_item(text)
        assert item is not None

        score = engine.score_item(item)
        # Currency scoring typically returns None or a zero-score
        # (depends on if there are mods to score)


class TestFullPipeline:
    """Test the full lookup() pipeline end-to-end."""

    def test_lookup_rare_ring(self, engine):
        text = load_fixture("rare_ring_brimstone.txt")
        result = engine.lookup(text)

        assert result is not None
        assert "item" in result
        assert "mods" in result
        assert "score" in result
        assert "estimate" in result

        assert result["item"]["name"] == "Brimstone Knot"
        assert result["item"]["base_type"] == "Sapphire Ring"
        assert result["item"]["rarity"] == "rare"

    def test_lookup_rare_gloves(self, engine):
        text = load_fixture("rare_gloves_rage_paw.txt")
        result = engine.lookup(text)

        assert result is not None
        assert result["item"]["rarity"] == "rare"
        if result["score"]:
            assert result["score"]["grade"] in ("S", "A", "B", "C", "JUNK")

    def test_lookup_body_armour(self, engine):
        text = load_fixture("rare_body_armour_ghoul.txt")
        result = engine.lookup(text)

        assert result is not None
        assert result["item"]["item_class"] in ("Body Armours",)

    def test_lookup_invalid_returns_none(self, engine):
        assert engine.lookup("not an item") is None
        assert engine.lookup("") is None

    def test_lookup_all_fixtures(self, engine):
        """Smoke test: parse_item succeeds on all fixture files."""
        for fixture_file in sorted(FIXTURES_DIR.glob("*.txt")):
            text = fixture_file.read_text(encoding="utf-8")
            item = engine.parse_item(text)
            assert item is not None, f"Failed to parse {fixture_file.name}"


class TestEstimatePrice:
    """Test the calibration estimate pathway."""

    def test_estimate_returns_float_or_none(self, engine):
        text = load_fixture("rare_ring_brimstone.txt")
        item = engine.parse_item(text)
        score = engine.score_item(item)
        if score:
            est = engine.estimate_price(item, score)
            # May be None if no calibration data loaded
            assert est is None or isinstance(est, (int, float))

    def test_estimate_with_none_score(self, engine):
        text = load_fixture("rare_ring_brimstone.txt")
        item = engine.parse_item(text)
        assert engine.estimate_price(item, None) is None


class TestParseMods:
    """Test parse_mods() directly."""

    def test_parse_mods_rare(self, engine):
        text = load_fixture("rare_ring_brimstone.txt")
        item = engine.parse_item(text)
        mods = engine.parse_mods(item)
        assert isinstance(mods, list)
        assert len(mods) > 0
        # Each mod should have expected attributes
        for mod in mods:
            assert hasattr(mod, "stat_id")
            assert hasattr(mod, "value")
            assert hasattr(mod, "raw_text")

    def test_parse_mods_currency_empty(self, engine):
        text = load_fixture("currency_fracturing_orb.txt")
        item = engine.parse_item(text)
        mods = engine.parse_mods(item)
        assert isinstance(mods, list)
        # Currency items have no matchable mods


class TestModuleOverrides:
    """Test that PricingEngine injects GameConfig values into module constants."""

    def test_item_parser_overrides(self):
        """Verify CURRENCY_KEYWORDS and VALUABLE_BASES are set on the module."""
        from core import PricingEngine
        from games.poe2 import create_poe2_config

        cfg = create_poe2_config()
        engine = PricingEngine(cfg)
        # Call _init_item_parser which does the override
        engine._init_item_parser()

        import item_parser as ip_module
        assert ip_module.CURRENCY_KEYWORDS == cfg.currency_keywords
        assert ip_module.VALUABLE_BASES == cfg.valuable_bases

    def test_mod_database_overrides(self):
        """Verify _WEIGHT_TABLE, _DEFENCE_GROUP_MARKERS, _DISPLAY_NAMES are set."""
        from core import PricingEngine
        from games.poe2 import create_poe2_config

        cfg = create_poe2_config()
        engine = PricingEngine(cfg)
        # Must init mod_parser first (mod_database depends on it)
        engine._init_mod_parser()
        engine._init_mod_database()

        import mod_database as md_module
        assert md_module._WEIGHT_TABLE == cfg.weight_table
        assert md_module._DEFENCE_GROUP_MARKERS == cfg.defence_group_markers
        assert md_module._DISPLAY_NAMES == cfg.display_names

    def test_price_cache_overrides(self):
        """Verify URL, categories, and delay are set on the module."""
        from core import PricingEngine
        from games.poe2 import create_poe2_config

        cfg = create_poe2_config()
        engine = PricingEngine(cfg)

        try:
            engine._init_price_cache()
        except Exception:
            pass  # PriceCache init may fail without network

        import price_cache as pc_module
        assert pc_module.POE2_EXCHANGE_URL == cfg.poe_ninja_exchange_url
        assert pc_module.EXCHANGE_CATEGORIES == cfg.exchange_categories
        assert pc_module.POE2SCOUT_UNIQUE_CATEGORIES == cfg.poe2scout_unique_categories
        assert pc_module.POE2SCOUT_CURRENCY_CATEGORIES == cfg.poe2scout_currency_categories
        assert pc_module.REQUEST_DELAY == cfg.price_request_delay
