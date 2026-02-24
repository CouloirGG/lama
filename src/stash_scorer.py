"""
stash_scorer.py — Batch-scores stash items using LAMA's scoring pipeline.

Wraps ItemLookup components to score ParsedItem objects from the stash API,
aggregate wealth per tab and overall, and maintain a wealth history file.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

SETTINGS_DIR = Path.home() / ".poe2-price-overlay"
WEALTH_HISTORY_FILE = SETTINGS_DIR / "wealth_history.json"
MAX_HISTORY_DAYS = 30

# Item classes that can be meaningfully scored (rare/unique equipment)
SCORABLE_RARITIES = {"rare", "unique"}

# Skip these — not scorable
SKIP_RARITIES = {"currency", "gem"}


@dataclass
class ScoredItem:
    """A stash item with scoring results."""
    name: str = ""
    base_type: str = ""
    item_class: str = ""
    rarity: str = ""
    item_level: int = 0
    grade: Optional[str] = None       # "S", "A", "B", "C", "JUNK"
    score: float = 0.0                # normalized_score (0-1)
    estimate_divine: float = 0.0      # calibration estimate
    estimate_chaos: float = 0.0       # estimate * d2c
    icon_url: str = ""
    stack_size: int = 1
    listed_price: float = 0.0         # from stash note
    tab_name: str = ""
    tab_id: str = ""
    mods: list = field(default_factory=list)   # [{text, tier_label, weight}, ...]
    top_mods: str = ""                # summary string
    total_dps: float = 0.0
    total_defense: int = 0


@dataclass
class TabSummary:
    """Aggregated value for a stash tab."""
    id: str = ""
    name: str = ""
    type: str = ""
    colour: Optional[dict] = None
    item_count: int = 0
    scored_count: int = 0
    total_divine: float = 0.0
    items: List[ScoredItem] = field(default_factory=list)


class StashScorer:
    """Batch-scores stash items using the existing LAMA scoring pipeline."""

    def __init__(self):
        self._mod_parser = None
        self._mod_database = None
        self._calibration = None
        self._divine_to_chaos = 0.0
        self._ready = False

    def initialize(self) -> bool:
        """Load scoring components. Returns True if ready."""
        try:
            from mod_parser import ModParser
            from mod_database import ModDatabase
            from calibration import CalibrationEngine
            from config import CALIBRATION_LOG_FILE, SHARD_DIR

            self._mod_parser = ModParser()
            self._mod_parser.load_stats()

            self._mod_database = ModDatabase()
            if self._mod_parser.loaded:
                self._mod_database.load(self._mod_parser)

            self._calibration = CalibrationEngine()
            self._calibration.load(CALIBRATION_LOG_FILE)
            try:
                self._calibration.load_shards(SHARD_DIR)
            except Exception:
                pass

            self._ready = self._mod_parser.loaded and self._mod_database.loaded
            logger.info(f"StashScorer initialized (ready={self._ready})")
            return self._ready
        except Exception as e:
            logger.error(f"StashScorer init failed: {e}")
            return False

    @property
    def ready(self) -> bool:
        return self._ready

    def set_divine_to_chaos(self, rate: float):
        """Update exchange rate for chaos value calculation."""
        self._divine_to_chaos = rate

    def score_item(self, parsed_item) -> Optional[ScoredItem]:
        """Score a single ParsedItem. Returns ScoredItem or None."""
        if not self._ready:
            return None

        rarity = getattr(parsed_item, "rarity", "")

        # Currency/gems: return with stack value if listed
        if rarity in SKIP_RARITIES:
            return None

        # Only score rare/unique equipment
        if rarity not in SCORABLE_RARITIES:
            return None

        try:
            parsed_mods = self._mod_parser.parse_mods(parsed_item)
            score = self._mod_database.score_item(parsed_item, parsed_mods)

            if not score:
                return None

            # Estimate price
            estimate_divine = 0.0
            if self._calibration and score:
                try:
                    mod_scores = getattr(score, "mod_scores", [])
                    mod_tiers = {ms.mod_group: int(ms.tier_label[1:])
                                 for ms in mod_scores
                                 if ms.mod_group and ms.tier_label and ms.tier_label[1:].isdigit()}
                    mod_rolls = {ms.mod_group: round(ms.roll_quality, 3)
                                 for ms in mod_scores
                                 if ms.mod_group and hasattr(ms, 'roll_quality')
                                 and ms.roll_quality is not None}
                    est = self._calibration.estimate(
                        score.normalized_score,
                        getattr(parsed_item, "item_class", "") or "",
                        grade=score.grade.value,
                        top_tier_count=getattr(score, "top_tier_count", 0),
                        mod_count=len(mod_scores) or 4,
                        mod_groups=[ms.mod_group for ms in mod_scores if ms.mod_group],
                        base_type=getattr(parsed_item, "base_type", ""),
                        mod_tiers=mod_tiers,
                        mod_rolls=mod_rolls,
                        somv_factor=getattr(score, "somv_factor", 1.0),
                        pdps=getattr(parsed_item, "physical_dps", 0.0),
                        edps=getattr(parsed_item, "elemental_dps", 0.0),
                    )
                    if est is not None:
                        estimate_divine = round(est, 2)
                except Exception:
                    pass

            return ScoredItem(
                name=parsed_item.name,
                base_type=parsed_item.base_type,
                item_class=getattr(parsed_item, "item_class", ""),
                rarity=rarity,
                item_level=parsed_item.item_level,
                grade=score.grade.value if score else None,
                score=round(score.normalized_score, 4) if score else 0,
                estimate_divine=estimate_divine,
                estimate_chaos=round(estimate_divine * self._divine_to_chaos, 0) if self._divine_to_chaos else 0,
                stack_size=parsed_item.stack_size,
                mods=[
                    {
                        "text": ms.raw_text,
                        "tier_label": ms.tier_label,
                        "weight": round(ms.weight, 2),
                    }
                    for ms in (score.mod_scores if score else [])
                ],
                top_mods=score.top_mods_summary if score else "",
                total_dps=getattr(parsed_item, "total_dps", 0) or 0,
                total_defense=getattr(parsed_item, "total_defense", 0) or 0,
            )
        except Exception as e:
            logger.debug(f"Scoring failed for {parsed_item.name}: {e}")
            return None

    def score_tab(self, tab, stash_items) -> TabSummary:
        """Score all items in a tab and return a TabSummary."""
        summary = TabSummary(
            id=tab.id,
            name=tab.name,
            type=tab.type,
            colour=tab.colour,
            item_count=len(stash_items),
        )

        for si in stash_items:
            scored = self.score_item(si.parsed)
            if scored:
                scored.icon_url = si.icon_url
                scored.tab_name = si.tab_name
                scored.tab_id = si.tab_id
                scored.listed_price = si.listed_price
                scored.stack_size = si.stack_size
                summary.items.append(scored)
                summary.scored_count += 1
                # Use listed price if available, otherwise estimate
                value = si.listed_price if si.listed_price > 0 else scored.estimate_divine
                summary.total_divine += value * scored.stack_size

        # Sort items by value descending
        summary.items.sort(key=lambda x: x.estimate_divine, reverse=True)
        summary.total_divine = round(summary.total_divine, 2)
        return summary

    @staticmethod
    def save_wealth_snapshot(tab_summaries: List[TabSummary]):
        """Append a wealth snapshot to the history file."""
        total = sum(t.total_divine for t in tab_summaries)
        breakdown = {
            t.name: round(t.total_divine, 2)
            for t in tab_summaries
            if t.total_divine > 0
        }

        snapshot = {
            "timestamp": int(time.time()),
            "total_divine": round(total, 2),
            "tab_count": len(tab_summaries),
            "scored_items": sum(t.scored_count for t in tab_summaries),
            "breakdown": breakdown,
        }

        try:
            SETTINGS_DIR.mkdir(parents=True, exist_ok=True)

            # Load existing history
            history = []
            if WEALTH_HISTORY_FILE.exists():
                try:
                    with open(WEALTH_HISTORY_FILE) as f:
                        history = json.load(f)
                except Exception:
                    history = []

            # Prune old entries (>30 days)
            cutoff = time.time() - (MAX_HISTORY_DAYS * 86400)
            history = [h for h in history if h.get("timestamp", 0) > cutoff]

            history.append(snapshot)

            with open(WEALTH_HISTORY_FILE, "w") as f:
                json.dump(history, f)

            logger.info(f"Wealth snapshot saved: {total:.1f} divine across {len(tab_summaries)} tabs")
        except Exception as e:
            logger.warning(f"Failed to save wealth snapshot: {e}")

    @staticmethod
    def load_wealth_history() -> List[dict]:
        """Load wealth history for sparkline display."""
        if not WEALTH_HISTORY_FILE.exists():
            return []
        try:
            with open(WEALTH_HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            return []
