"""
PricingEngine — game-agnostic facade for the LAMA scoring pipeline.

Single entry point wrapping ItemParser, ModParser, ModDatabase,
and PriceCache. Consumers pass a GameConfig to configure all modules
without them importing from config.py directly.

Usage:
    from core import PricingEngine
    from games.poe2 import create_poe2_config

    engine = PricingEngine(create_poe2_config())
    engine.initialize()
    result = engine.parse_item(clipboard_text)
    score  = engine.score_item(result)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.game_config import GameConfig

logger = logging.getLogger(__name__)


class PricingEngine:
    """Game-agnostic scoring engine facade.

    Wraps existing modules and injects GameConfig values where they
    would otherwise import from config.py.
    """

    def __init__(self, config: GameConfig):
        self.config = config
        self._item_parser = None
        self._mod_parser = None
        self._mod_database = None
        self._price_cache = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def initialize(self) -> bool:
        """Load all components. Returns True if core modules are ready.

        Configures each module by overriding module-level constants
        from GameConfig before calling any methods. This is the
        lowest-friction approach — no changes to existing module files.
        """
        try:
            self._init_item_parser()
            self._init_mod_parser()
            self._init_mod_database()
            self._init_price_cache()

            self._ready = (
                self._mod_parser is not None
                and getattr(self._mod_parser, 'loaded', False)
                and self._mod_database is not None
                and getattr(self._mod_database, 'loaded', False)
            )
            logger.info(f"PricingEngine initialized (ready={self._ready}, "
                        f"game={self.config.game_id})")
            return self._ready
        except Exception as e:
            logger.error(f"PricingEngine init failed: {e}", exc_info=True)
            return False

    # ── Public API ──────────────────────────────────────────

    def parse_item(self, clipboard_text: str):
        """Parse clipboard text into a ParsedItem.

        Returns:
            ParsedItem or None if parsing fails.
        """
        if self._item_parser is None:
            return None
        return self._item_parser.parse_clipboard(clipboard_text)

    def parse_mods(self, item) -> list:
        """Match item mods to trade API stat IDs.

        Args:
            item: ParsedItem from parse_item().

        Returns:
            List of ParsedMod objects.
        """
        if self._mod_parser is None or not self._mod_parser.loaded:
            return []
        return self._mod_parser.parse_mods(item)

    def score_item(self, item, parsed_mods=None):
        """Score an item locally (instant, no API calls).

        Args:
            item: ParsedItem from parse_item().
            parsed_mods: Optional list of ParsedMod. If None, will parse mods first.

        Returns:
            ItemScore or None if scoring isn't available.
        """
        if self._mod_database is None or not self._mod_database.loaded:
            return None
        if parsed_mods is None:
            parsed_mods = self.parse_mods(item)
        return self._mod_database.score_item(item, parsed_mods)

    def lookup_price(self, item) -> Optional[Dict[str, Any]]:
        """Look up static price for uniques/currencies (from PriceCache).

        Args:
            item: ParsedItem from parse_item().

        Returns:
            Price dict with divine_value, chaos_value, etc. or None.
        """
        if self._price_cache is None:
            return None
        name = getattr(item, "name", "") or ""
        base_type = getattr(item, "base_type", "") or ""
        item_level = getattr(item, "item_level", 0) or 0
        return self._price_cache.lookup(name, base_type, item_level)

    def lookup(self, text: str) -> Optional[Dict[str, Any]]:
        """Full pipeline: parse, score, estimate — like ItemLookup.lookup().

        Returns dict with keys: item, mods, score, estimate
        or None if parsing fails.
        """
        item = self.parse_item(text)
        if not item:
            return None

        parsed_mods = self.parse_mods(item)
        score = self.score_item(item, parsed_mods)

        return {
            "item": {
                "name": getattr(item, "name", None),
                "base_type": getattr(item, "base_type", None),
                "rarity": getattr(item, "rarity", None),
                "item_level": getattr(item, "item_level", None),
                "item_class": getattr(item, "item_class", None),
            },
            "mods": [
                {
                    "text": ms.raw_text,
                    "tier_label": ms.tier_label,
                    "weight": round(ms.weight, 2),
                    "stat_id": ms.stat_id,
                }
                for ms in (score.mod_scores if score else [])
            ],
            "score": {
                "grade": score.grade.value if score else None,
                "normalized_score": (round(score.normalized_score, 4)
                                     if score else None),
                "top_mods": score.top_mods_summary if score else None,
                "top_tier_count": score.top_tier_count if score else 0,
            } if score else None,
            "estimate": None,
        }

    def start_price_updates(self):
        """Start background price refresh thread."""
        if self._price_cache:
            self._price_cache.start()

    def stop_price_updates(self):
        """Stop background price refresh thread."""
        if self._price_cache:
            self._price_cache.stop()

    # ── Internal initialization ─────────────────────────────

    def _init_item_parser(self):
        """Configure item_parser module constants, then instantiate."""
        import item_parser as ip_module

        if self.config.currency_keywords:
            ip_module.CURRENCY_KEYWORDS = self.config.currency_keywords
        if self.config.valuable_bases:
            ip_module.VALUABLE_BASES = self.config.valuable_bases

        from item_parser import ItemParser
        self._item_parser = ItemParser()

    def _init_mod_parser(self):
        """Load mod parser with bundled stat definitions."""
        from mod_parser import ModParser
        self._mod_parser = ModParser()
        self._mod_parser.load_stats()

    def _init_mod_database(self):
        """Configure mod_database module constants, then load."""
        if self._mod_parser is None or not self._mod_parser.loaded:
            return

        import mod_database as md_module

        # Inject GameConfig values into module-level constants
        md_module.REPOE_BASE_URL = self.config.repoe_base_url
        md_module.REPOE_CACHE_DIR = self.config.repoe_cache_dir
        md_module.REPOE_CACHE_TTL = self.config.repoe_cache_ttl
        md_module.DPS_ITEM_CLASSES = self.config.dps_item_classes
        md_module.TWO_HAND_CLASSES = self.config.two_hand_classes
        md_module.DEFENSE_ITEM_CLASSES = self.config.defense_item_classes
        md_module.DPS_BRACKETS_2H = self.config.dps_brackets_2h
        md_module.DPS_BRACKETS_1H = self.config.dps_brackets_1h
        md_module.DEFENSE_THRESHOLDS = self.config.defense_thresholds
        if self.config.weight_table:
            md_module._WEIGHT_TABLE = self.config.weight_table
        if self.config.defence_group_markers:
            md_module._DEFENCE_GROUP_MARKERS = self.config.defence_group_markers
        if self.config.display_names:
            md_module._DISPLAY_NAMES = self.config.display_names

        from mod_database import ModDatabase
        self._mod_database = ModDatabase()
        self._mod_database.load(self._mod_parser)

    def _init_price_cache(self):
        """Configure price_cache module constants, then create instance."""
        try:
            import price_cache as pc_module

            # Inject GameConfig values
            pc_module.PRICE_REFRESH_INTERVAL = self.config.price_refresh_interval
            pc_module.CACHE_DIR = self.config.cache_dir
            pc_module.DEFAULT_LEAGUE = self.config.default_league
            pc_module.POE2SCOUT_BASE_URL = self.config.price_source_url
            if self.config.poe_ninja_exchange_url:
                pc_module.POE2_EXCHANGE_URL = self.config.poe_ninja_exchange_url
            if self.config.exchange_categories:
                pc_module.EXCHANGE_CATEGORIES = self.config.exchange_categories
            if self.config.poe2scout_unique_categories:
                pc_module.POE2SCOUT_UNIQUE_CATEGORIES = self.config.poe2scout_unique_categories
            if self.config.poe2scout_currency_categories:
                pc_module.POE2SCOUT_CURRENCY_CATEGORIES = self.config.poe2scout_currency_categories
            if self.config.price_request_delay > 0:
                pc_module.REQUEST_DELAY = self.config.price_request_delay

            from price_cache import PriceCache
            self._price_cache = PriceCache(league=self.config.default_league)
        except Exception as e:
            logger.debug(f"PriceCache init skipped: {e}")

