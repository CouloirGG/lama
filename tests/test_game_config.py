"""Tests for GameConfig and POE2 game config factory."""

import sys
from pathlib import Path

import pytest

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestGameConfig:
    """Test the GameConfig dataclass itself."""

    def test_create_minimal(self):
        from core.game_config import GameConfig
        cfg = GameConfig(
            game_id="test",
            default_league="Test League",
            cache_dir=Path("/tmp/test-cache"),
            trade_api_base="https://example.com/api",
            trade_stats_url="https://example.com/api/stats",
            trade_items_url="https://example.com/api/items",
            trade_stats_cache_file=Path("/tmp/stats.json"),
            trade_items_cache_file=Path("/tmp/items.json"),
        )
        assert cfg.game_id == "test"
        assert cfg.default_league == "Test League"
        assert cfg.cache_dir == Path("/tmp/test-cache")

    def test_defaults(self):
        from core.game_config import GameConfig
        cfg = GameConfig(
            game_id="test",
            default_league="Test",
            cache_dir=Path("/tmp"),
            trade_api_base="",
            trade_stats_url="",
            trade_items_url="",
            trade_stats_cache_file=Path("/tmp/s.json"),
            trade_items_cache_file=Path("/tmp/i.json"),
        )
        assert cfg.trade_max_requests_per_second == 1
        assert cfg.trade_result_count == 8
        assert cfg.trade_cache_ttl == 300
        assert cfg.trade_mod_min_multiplier == 0.8
        assert cfg.repoe_cache_ttl == 7 * 86400
        assert cfg.price_refresh_interval == 900
        assert cfg.calibration_max_price_divine == 1500.0
        assert cfg.shard_refresh_interval == 86400
        assert cfg.dps_item_classes == frozenset()
        assert cfg.grade_tier_map == {}


class TestPoe2Config:
    """Test the POE2 config factory."""

    def test_create_default(self):
        from games.poe2 import create_poe2_config
        cfg = create_poe2_config()

        assert cfg.game_id == "poe2"
        assert cfg.default_league  # non-empty
        assert cfg.cache_dir.parts  # is a real path

    def test_league_override(self):
        from games.poe2 import create_poe2_config
        cfg = create_poe2_config(league="Custom League")
        assert cfg.default_league == "Custom League"

    def test_cache_dir_override(self):
        from games.poe2 import create_poe2_config
        custom_dir = Path("/tmp/custom-cache")
        cfg = create_poe2_config(cache_dir=custom_dir)
        assert cfg.cache_dir == custom_dir

    def test_trade_api_fields(self):
        from games.poe2 import create_poe2_config
        cfg = create_poe2_config()

        assert "pathofexile.com" in cfg.trade_api_base
        assert cfg.trade_stats_url.startswith(cfg.trade_api_base)
        assert cfg.trade_items_url.startswith(cfg.trade_api_base)
        assert cfg.trade_stats_cache_file.suffix == ".json"
        assert cfg.trade_items_cache_file.suffix == ".json"

    def test_repoe_fields(self):
        from games.poe2 import create_poe2_config
        cfg = create_poe2_config()

        assert "repoe" in cfg.repoe_base_url.lower() or "github" in cfg.repoe_base_url
        assert cfg.repoe_cache_dir is not None
        assert cfg.repoe_cache_ttl > 0

    def test_item_classes(self):
        from games.poe2 import create_poe2_config
        cfg = create_poe2_config()

        assert "Bows" in cfg.dps_item_classes
        assert "Daggers" in cfg.dps_item_classes
        assert "Bows" in cfg.two_hand_classes
        assert "Body Armours" in cfg.defense_item_classes
        assert "Shields" in cfg.defense_item_classes

    def test_dps_brackets(self):
        from games.poe2 import create_poe2_config
        cfg = create_poe2_config()

        assert 68 in cfg.dps_brackets_2h
        assert 0 in cfg.dps_brackets_2h
        assert 68 in cfg.dps_brackets_1h
        assert len(cfg.dps_brackets_2h[68]) == 4  # (terrible, low, decent, good)

    def test_defense_thresholds(self):
        from games.poe2 import create_poe2_config
        cfg = create_poe2_config()

        assert "Body Armours" in cfg.defense_thresholds
        assert "Helmets" in cfg.defense_thresholds
        assert len(cfg.defense_thresholds["Body Armours"]) == 3

    def test_price_source(self):
        from games.poe2 import create_poe2_config
        cfg = create_poe2_config()

        assert "poe2scout" in cfg.price_source_url

    def test_calibration_fields(self):
        from games.poe2 import create_poe2_config
        cfg = create_poe2_config()

        assert cfg.calibration_log_file is not None
        assert cfg.shard_dir is not None
        assert cfg.shard_github_repo  # non-empty
        assert cfg.calibration_max_price_divine == 1500.0

    def test_grade_tier_map(self):
        from games.poe2 import create_poe2_config
        cfg = create_poe2_config()

        assert "S" in cfg.grade_tier_map
        assert "A" in cfg.grade_tier_map
        assert cfg.grade_tier_map["S"] == "high"
        assert cfg.grade_tier_map["A"] == "good"

    def test_weight_table(self):
        from games.poe2 import create_poe2_config
        cfg = create_poe2_config()

        assert len(cfg.weight_table) == 6  # 6 tiers
        weights = [w for w, _ in cfg.weight_table]
        assert weights[0] == 3.0   # premium
        assert weights[-1] == 0.1  # near-zero
        # Spot-check: "movespeed" in first tier
        assert "movespeed" in cfg.weight_table[0][1]

    def test_defence_group_markers(self):
        from games.poe2 import create_poe2_config
        cfg = create_poe2_config()

        assert len(cfg.defence_group_markers) == 3
        assert "reductionrating" in cfg.defence_group_markers
        assert "evasionrating" in cfg.defence_group_markers
        assert "energyshield" in cfg.defence_group_markers

    def test_display_names(self):
        from games.poe2 import create_poe2_config
        cfg = create_poe2_config()

        assert len(cfg.display_names) >= 50
        # Spot-check a few mappings
        dn_dict = dict(cfg.display_names)
        assert dn_dict["movespeed"] == "MoveSpd"
        assert dn_dict["criticalstrikemultiplier"] == "CritMulti"
        assert dn_dict["maximumlife"] == "Life"
        assert dn_dict["fireresist"] == "FireRes"

    def test_currency_keywords(self):
        from games.poe2 import create_poe2_config
        cfg = create_poe2_config()

        assert isinstance(cfg.currency_keywords, frozenset)
        assert len(cfg.currency_keywords) >= 23
        assert "divine orb" in cfg.currency_keywords
        assert "chaos orb" in cfg.currency_keywords
        assert "mirror of kalandra" in cfg.currency_keywords

    def test_valuable_bases(self):
        from games.poe2 import create_poe2_config
        cfg = create_poe2_config()

        assert isinstance(cfg.valuable_bases, frozenset)
        assert len(cfg.valuable_bases) >= 19
        assert "stellar amulet" in cfg.valuable_bases
        assert "astral plate" in cfg.valuable_bases
        assert "opal ring" in cfg.valuable_bases

    def test_exchange_categories(self):
        from games.poe2 import create_poe2_config
        cfg = create_poe2_config()

        assert len(cfg.exchange_categories) == 14
        assert "Currency" in cfg.exchange_categories
        assert "Fragments" in cfg.exchange_categories

    def test_poe2scout_categories(self):
        from games.poe2 import create_poe2_config
        cfg = create_poe2_config()

        assert len(cfg.poe2scout_unique_categories) == 7
        assert "armour" in cfg.poe2scout_unique_categories
        assert "weapon" in cfg.poe2scout_unique_categories
        assert len(cfg.poe2scout_currency_categories) >= 14
        assert "currency" in cfg.poe2scout_currency_categories

    def test_poe_ninja_exchange_url(self):
        from games.poe2 import create_poe2_config
        cfg = create_poe2_config()

        assert "poe.ninja" in cfg.poe_ninja_exchange_url
        assert "exchange" in cfg.poe_ninja_exchange_url

    def test_price_request_delay(self):
        from games.poe2 import create_poe2_config
        cfg = create_poe2_config()

        assert cfg.price_request_delay == 0.3

    def test_all_required_fields_populated(self):
        """Verify no required field is left empty/None."""
        from games.poe2 import create_poe2_config
        cfg = create_poe2_config()

        # These must all be non-empty
        assert cfg.game_id
        assert cfg.default_league
        assert cfg.trade_api_base
        assert cfg.trade_stats_url
        assert cfg.trade_items_url
        assert cfg.repoe_base_url
        assert cfg.price_source_url
        assert cfg.shard_github_repo
        assert len(cfg.dps_item_classes) > 0
        assert len(cfg.two_hand_classes) > 0
        assert len(cfg.defense_item_classes) > 0
        assert len(cfg.dps_brackets_2h) > 0
        assert len(cfg.dps_brackets_1h) > 0
        assert len(cfg.defense_thresholds) > 0
        assert len(cfg.grade_tier_map) > 0
