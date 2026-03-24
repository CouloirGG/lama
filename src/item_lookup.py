"""
item_lookup.py — Facade for parsing and scoring items from clipboard text.

Used by the dashboard's Item Lookup card to analyze pasted item text
without requiring the overlay scanner to be running.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ItemLookup:
    """Wraps ItemParser + ModParser + ModDatabase for item scoring."""

    def __init__(self):
        self._item_parser = None
        self._mod_parser = None
        self._mod_database = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def initialize(self) -> bool:
        """Load all components. Returns True if ready."""
        try:
            from item_parser import ItemParser
            from mod_parser import ModParser
            from mod_database import ModDatabase

            self._item_parser = ItemParser()

            self._mod_parser = ModParser()
            self._mod_parser.load_stats()

            self._mod_database = ModDatabase()
            if self._mod_parser.loaded:
                self._mod_database.load(self._mod_parser)

            self._ready = (
                self._mod_parser.loaded
                and self._mod_database.loaded
            )
            logger.info(f"ItemLookup initialized (ready={self._ready})")
            return self._ready
        except Exception as e:
            logger.error(f"ItemLookup init failed: {e}")
            return False

    def lookup(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse and score item text.

        Returns dict with keys: item, mods, score
        or None if parsing fails.
        """
        if not self._ready:
            return None

        # Parse item
        item = self._item_parser.parse_clipboard(text)
        if not item:
            return None

        # Parse mods
        parsed_mods = self._mod_parser.parse_mods(item)

        # Score item
        score = self._mod_database.score_item(item, parsed_mods)

        # Build response
        result = {
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
                "normalized_score": round(score.normalized_score, 4) if score else None,
                "top_mods": score.top_mods_summary if score else None,
                "top_tier_count": score.top_tier_count if score else 0,
            } if score else None,
            "estimate": None,
        }
        return result
