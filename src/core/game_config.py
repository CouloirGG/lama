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
    """Complete configuration for a game's analysis engine."""

    # ── Identity ────────────────────────────────────────────
    game_id: str                          # e.g. "poe2", "last_epoch"
    default_league: str                   # e.g. "Fate of the Vaal"
    cache_dir: Path                       # base cache directory

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

    # ── Mod Database scoring ─────────────────────────────────
    weight_table: List[Tuple[float, List[str]]] = field(default_factory=list)
    defence_group_markers: Tuple[str, ...] = ()
    display_names: List[Tuple[str, str]] = field(default_factory=list)

    # ── Item Parser classification ────────────────────────────
    currency_keywords: FrozenSet[str] = field(default_factory=frozenset)
    valuable_bases: FrozenSet[str] = field(default_factory=frozenset)

    # ── Price Cache endpoints & categories ────────────────────
    poe_ninja_exchange_url: str = ""
    exchange_categories: List[str] = field(default_factory=list)
    poe2scout_unique_categories: List[str] = field(default_factory=list)
    poe2scout_currency_categories: List[str] = field(default_factory=list)
    price_request_delay: float = 0.0

    # ── Grade display mapping ───────────────────────────────
    grade_tier_map: Dict[str, str] = field(default_factory=dict)
