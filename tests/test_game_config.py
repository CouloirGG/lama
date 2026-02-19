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
        assert cfg.calibration_max_price_divine == 300.0
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
        assert len(cfg.defense_thresholds["Body Armours"]) == 4

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
        assert cfg.calibration_max_price_divine == 300.0

    def test_grade_tier_map(self):
        from games.poe2 import create_poe2_config
        cfg = create_poe2_config()

        assert "S" in cfg.grade_tier_map
        assert "A" in cfg.grade_tier_map
        assert cfg.grade_tier_map["S"] == "high"
        assert cfg.grade_tier_map["A"] == "good"

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
