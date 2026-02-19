"""
GameConfig — game-agnostic configuration dataclass.

Every game-specific value that core modules need is a field here.
Consumers create a GameConfig (via a game factory like create_poe2_config)
and pass it to PricingEngine, which injects values into existing modules.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple


@dataclass
class GameConfig:
    """Complete configuration for a game's pricing engine."""

    # ── Identity ────────────────────────────────────────────
    game_id: str                          # e.g. "poe2", "last_epoch"
    default_league: str                   # e.g. "Fate of the Vaal"
    cache_dir: Path                       # base cache directory

    # ── Trade API ───────────────────────────────────────────
    trade_api_base: str                   # e.g. "https://www.pathofexile.com/api/trade2"
    trade_stats_url: str                  # e.g. "{trade_api_base}/data/stats"
    trade_items_url: str                  # e.g. "{trade_api_base}/data/items"
    trade_stats_cache_file: Path          # disk cache for stat definitions
    trade_items_cache_file: Path          # disk cache for item definitions
    trade_max_requests_per_second: int = 1
    trade_result_count: int = 8
    trade_cache_ttl: int = 300            # seconds
    trade_mod_min_multiplier: float = 0.8
    trade_dps_filter_mult: float = 0.75
    trade_defense_filter_mult: float = 0.70

    # ── Mod Database (RePoE) ────────────────────────────────
    repoe_base_url: str = ""              # e.g. "https://repoe-fork.github.io/poe2"
    repoe_cache_dir: Optional[Path] = None
    repoe_cache_ttl: int = 7 * 86400     # 7 days

    # Item classification sets
    dps_item_classes: FrozenSet[str] = field(default_factory=frozenset)
    two_hand_classes: FrozenSet[str] = field(default_factory=frozenset)
    defense_item_classes: FrozenSet[str] = field(default_factory=frozenset)

    # DPS brackets: {min_ilvl: (terrible, low, decent, good)}
    dps_brackets_2h: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    dps_brackets_1h: Dict[int, Tuple[int, ...]] = field(default_factory=dict)

    # Defense thresholds per slot: {slot: (terrible, low, decent, good)}
    defense_thresholds: Dict[str, Tuple[int, ...]] = field(default_factory=dict)

    # ── Price Sources ───────────────────────────────────────
    price_source_url: str = ""            # e.g. "https://poe2scout.com/api"
    price_refresh_interval: int = 900     # seconds
    rate_history_file: Optional[Path] = None
    rate_history_backup: Optional[Path] = None

    # ── Calibration ─────────────────────────────────────────
    calibration_log_file: Optional[Path] = None
    calibration_max_price_divine: float = 300.0
    calibration_min_results: int = 3
    shard_dir: Optional[Path] = None
    shard_refresh_interval: int = 86400   # seconds
    shard_github_repo: str = ""           # e.g. "CouloirGG/lama"

    # ── Grade display mapping ───────────────────────────────
    grade_tier_map: Dict[str, str] = field(default_factory=dict)
