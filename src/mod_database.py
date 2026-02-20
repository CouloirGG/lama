"""
LAMA - Mod Database & Scoring Engine

Evaluates rare items locally by identifying mod tiers and computing a
weighted score.  Zero API calls — uses static RePoE data cached on disk.

Data source: repoe-fork.github.io/poe2/ (mods.json, mods_by_base.json,
base_items.json).  Cached in ~/.poe2-price-overlay/cache/repoe/, 7-day TTL,
stale-cache fallback.

Class API:
    db = ModDatabase()
    db.load(mod_parser)          # download/cache RePoE, build bridge + ladders
    score = db.score_item(item, mods)   # returns ItemScore with grade
    tier  = db.get_tier_info(stat_id, value, item_class)
"""

import re
import json
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from pathlib import Path

import requests

from config import (
    REPOE_BASE_URL, REPOE_CACHE_DIR, REPOE_CACHE_TTL,
    DPS_ITEM_CLASSES, TWO_HAND_CLASSES, DEFENSE_ITEM_CLASSES,
    DPS_BRACKETS_2H, DPS_BRACKETS_1H, DEFENSE_THRESHOLDS,
)

logger = logging.getLogger(__name__)


# ─── Data Structures ─────────────────────────────────

class Grade(Enum):
    S = "S"
    A = "A"
    B = "B"
    C = "C"
    JUNK = "JUNK"


@dataclass(frozen=True)
class TierInfo:
    tier_num: int        # 1 = best (T1), N = worst
    mod_key: str         # RePoE key, e.g. "IncreasedLife9"
    required_level: int  # ilvl needed to roll this tier
    stat_min: float      # value range lower bound
    stat_max: float      # value range upper bound
    name: str            # e.g. "Prime"


@dataclass
class TierLadder:
    group: str           # "IncreasedLife"
    generation_type: str # "prefix" or "suffix"
    item_class: str      # "Gloves"
    tiers: List[TierInfo]  # ordered T1-first (highest ilvl = T1)

    def identify_tier(self, value: float, item_level: int = 0) -> Optional[TierInfo]:
        """Which tier does this value fall into?

        When item_level > 0, skips tiers the item can't roll (required_level > item_level).
        """
        if not self.tiers:
            return None
        abs_val = abs(value)
        # Walk from T1 (best) down — first tier whose range contains the value
        for tier in self.tiers:
            # Skip tiers the item can't roll
            if item_level and tier.required_level > item_level:
                continue
            t_min = min(abs(tier.stat_min), abs(tier.stat_max))
            if abs_val >= t_min:
                return tier
        # Below worst tier's minimum — return worst rollable tier
        if item_level:
            rollable = [t for t in self.tiers if t.required_level <= item_level]
            if rollable:
                return rollable[-1]
        return self.tiers[-1]

    def max_tier_for_ilvl(self, item_level: int) -> Optional[TierInfo]:
        """Best tier this ilvl can roll (T1 = best)."""
        for tier in self.tiers:  # T1-first order
            if tier.required_level <= item_level:
                return tier
        return None

    def global_min_for_ilvl(self, item_level: int = 0) -> float:
        """Lowest possible value across rollable tiers."""
        tiers = self._rollable_tiers(item_level)
        if not tiers:
            return self.global_min
        vals = []
        for t in tiers:
            vals.extend([abs(t.stat_min), abs(t.stat_max)])
        return min(vals)

    def global_max_for_ilvl(self, item_level: int = 0) -> float:
        """Highest possible value across rollable tiers."""
        tiers = self._rollable_tiers(item_level)
        if not tiers:
            return self.global_max
        vals = []
        for t in tiers:
            vals.extend([abs(t.stat_min), abs(t.stat_max)])
        return max(vals)

    def _rollable_tiers(self, item_level: int) -> List[TierInfo]:
        """Return tiers the item can roll (required_level <= item_level)."""
        if not item_level:
            return self.tiers
        return [t for t in self.tiers if t.required_level <= item_level]

    @property
    def global_min(self) -> float:
        """Lowest possible value across all tiers."""
        if not self.tiers:
            return 0.0
        vals = []
        for t in self.tiers:
            vals.extend([abs(t.stat_min), abs(t.stat_max)])
        return min(vals)

    @property
    def global_max(self) -> float:
        """Highest possible value across all tiers."""
        if not self.tiers:
            return 1.0
        vals = []
        for t in self.tiers:
            vals.extend([abs(t.stat_min), abs(t.stat_max)])
        return max(vals)


@dataclass
class ModScore:
    raw_text: str
    stat_id: str
    value: float
    mod_group: str
    generation_type: str
    tier: Optional[TierInfo]
    tier_label: str        # "T1", "T2", etc.
    percentile: float      # 0.0-1.0
    weight: float          # mod importance
    weighted_score: float  # percentile * weight
    is_key_mod: bool       # True if not a common/filler mod
    roll_quality: float = 0.5  # 0.0-1.0 within tier (0=bottom, 1=perfect)


@dataclass
class ItemScore:
    normalized_score: float  # 0.0-1.0
    grade: Grade
    prefix_count: int
    suffix_count: int
    mod_scores: List[ModScore]
    top_mods_summary: str    # "T1 CritMulti, T2 MovementSpeed"
    top_tier_count: int = 0  # count of valuable T1/T2 mods (weight >= 1.0)
    dps_factor: float = 1.0
    defense_factor: float = 1.0
    somv_factor: float = 1.0   # roll quality multiplier (0.90-1.10)
    total_dps: float = 0.0
    total_defense: int = 0
    quality: int = 0
    sockets: int = 0

    def format_overlay_text(self, price_estimate: float = None,
                            divine_to_chaos: float = 0,
                            show_grade: bool = True,
                            show_price: bool = True,
                            show_stars: bool = True,
                            show_mods: bool = True,
                            show_dps: bool = True) -> str:
        """Format for overlay display.

        JUNK:  '✗'
        C:     'C'
        B/A/S without price:  'A 67% ★3: T1 SpellCrit, T1 CritChance, T1 ES'
        B/A/S with price:     'A ~130d ★3: T1 SpellCrit, T1 CritChance, T1 ES'

        Display flags control which parts are included.
        """
        if self.grade in (Grade.JUNK, Grade.C):
            if self.quality > 0 or self.sockets > 0:
                return "SCRAP"
            return "\u2717"

        parts = []

        # Grade letter
        if show_grade:
            parts.append(self.grade.value)

        # Price or score tag
        if show_price:
            if price_estimate is not None and price_estimate > 0:
                if price_estimate >= 10:
                    parts.append(f"~{price_estimate:.0f}d")
                elif price_estimate >= 1.0:
                    parts.append(f"~{price_estimate:.1f}d")
                elif divine_to_chaos > 0:
                    chaos = price_estimate * divine_to_chaos
                    parts.append(f"~{chaos:.0f}c")
                else:
                    parts.append(f"~{price_estimate:.2f}d")
            else:
                pct = int(self.normalized_score * 100)
                parts.append(f"{pct}%")

        # Combat stat tag (DPS/defense when factor penalizes)
        combat_tag = ""
        if show_dps:
            if self.dps_factor < 1.0 and self.total_dps > 0:
                combat_tag = f"{int(self.total_dps)}dps"
            elif self.defense_factor < 1.0 and self.total_defense > 0:
                combat_tag = f"{self.total_defense}def"

        # Star count: valuable T1/T2 mods (repeated ★ like a rating)
        star = ""
        if show_stars and self.top_tier_count > 0:
            star = "\u2605" * self.top_tier_count

        if combat_tag:
            parts.append(combat_tag)
        elif star:
            parts.append(star)

        prefix = " ".join(parts)

        if show_mods and self.top_mods_summary:
            if prefix:
                return f"{prefix}: {self.top_mods_summary}"
            return self.top_mods_summary
        # Fallback: if everything is toggled off, show a single star
        # so there's still a visible colored indicator
        return prefix or "\u2605"


# ─── Weight Table ─────────────────────────────────────

# Maps mod group patterns → weight.  Checked by substring match against
# the RePoE mod group name (case-insensitive).
_WEIGHT_TABLE: List[Tuple[float, List[str]]] = [
    (3.0, [
        "movementvelocity", "movespeed",
        "addedskilllevels", "skilllevels", "gemlevels",
        "critmulti", "criticalmulti", "criticalstrikemultiplier",
        "critchance", "criticalstrikechance", "localcriticalstrikechance",
        "spelldamage", "percentagespelldamage",
        "physicaldamage", "localphysicaldamagepercent", "localphysicaldamage",
        "localaddedphysicaldamage",
    ]),
    (2.0, [
        "attackspeed", "localattackspeed",
        "castspeed",
        "addedfiredamage", "addedcolddamage", "addedlightningdamage",
        "addedchaosdamage", "addedelementaldamage",
        "manareservation", "manareservationefficiency",
        "liferecoup", "lifeonhit", "lifeleech",
        "projectilespeed",
        "areaofdamage", "areadamage",
    ]),
    (1.0, [
        "increasedlife", "maximumlife",
        "energyshield", "localenergyshield", "increasedenergy",
        "spirit",
    ]),
    (0.5, [
        "armour", "evasion",
        "localphysicaldamagereductionrating", "localevasionrating",
        "defencespercent", "alldefences",
        "chaosresist",  # chaos res is rarer, lower cap, bypasses ES
    ]),
    (0.3, [
        "resistance", "fireresist", "coldresist", "lightningresist",
        "allresist", "elementalresist",
        "strength", "dexterity", "intelligence", "allattributes",
        "maximummana", "increasedmana",
        "accuracy", "accuracyrating",
        "regen", "liferegeneration", "manaregeneration",
        "energyshieldrecharge",
        "flask", "flaskcharge", "flaskeffect",
        "charmduration", "charmeffect",
        "stun", "blockandstun", "stunrecovery",
        "reducedattributerequirements",
    ]),
    (0.1, [
        "thorns", "thornsdamage",
        "damagetakenonblock", "reflectdamage",
        "lightradius", "itemrarity",
    ]),
]


# Groups that must NOT match the premium "physicaldamage" patterns —
# these are armour/evasion/ES defence groups, not damage groups.
_DEFENCE_GROUP_MARKERS = ("reductionrating", "evasionrating", "energyshield")


def _get_weight_for_group(group: str) -> Optional[float]:
    """Look up weight by RePoE mod group name.  Returns None if no match."""
    group_lower = group.lower()
    # Quick check: if group is a defence group, skip premium damage patterns
    is_defence = any(m in group_lower for m in _DEFENCE_GROUP_MARKERS)
    for weight, patterns in _WEIGHT_TABLE:
        for pat in patterns:
            if pat in group_lower:
                # Prevent defence groups from matching damage patterns
                if is_defence and weight >= 2.0 and "damage" in pat:
                    continue
                return weight
    return None


# Patterns for common/filler mods — reuses the same list from trade_client.py
# Imported at class level to avoid circular dependency.
_COMMON_MOD_PATTERNS: Optional[tuple] = None


def _get_common_patterns() -> tuple:
    """Lazy-import _COMMON_MOD_PATTERNS from TradeClient."""
    global _COMMON_MOD_PATTERNS
    if _COMMON_MOD_PATTERNS is None:
        try:
            from trade_client import TradeClient
            _COMMON_MOD_PATTERNS = TradeClient._COMMON_MOD_PATTERNS
        except ImportError:
            _COMMON_MOD_PATTERNS = ()
    return _COMMON_MOD_PATTERNS


def _is_common_mod(raw_text: str) -> bool:
    """Check if a mod's raw text matches common/filler patterns."""
    patterns = _get_common_patterns()
    text_lower = raw_text.lower()
    return any(pat in text_lower for pat in patterns)


# ─── DPS & Defense Factors ────────────────────────────

def _select_bracket(brackets: dict, item_level: int) -> tuple:
    """Select the DPS bracket for a given item level.

    Brackets are keyed by minimum ilvl. Returns the highest bracket
    whose min_ilvl <= item_level.
    """
    best_ilvl = 0
    best = brackets[0]
    for min_ilvl, thresholds in brackets.items():
        if min_ilvl <= item_level and min_ilvl >= best_ilvl:
            best_ilvl = min_ilvl
            best = thresholds
    return best


def _interpolate(value: float, thresholds: tuple,
                 factors: tuple) -> float:
    """Linear interpolation between threshold/factor pairs.

    thresholds: (terrible, low, decent, good)
    factors:    (below_terrible, at_terrible, at_low, at_decent, at_good, above_good)
    """
    terrible, low, decent, good = thresholds
    f_below, f_terrible, f_low, f_decent, f_good, f_above = factors

    if value < terrible:
        return f_below
    if value < low:
        t = (value - terrible) / (low - terrible)
        return f_terrible + t * (f_low - f_terrible)
    if value < decent:
        t = (value - low) / (decent - low)
        return f_low + t * (f_decent - f_low)
    if value < good:
        t = (value - decent) / (good - decent)
        return f_decent + t * (f_good - f_decent)
    # Above good — cap at f_above
    return min(f_above, f_good + (value - good) / good * (f_above - f_good))


def _dps_factor(total_dps: float, item_class: str, item_level: int) -> float:
    """Multiplicative DPS factor for attack weapon scoring.

    Returns 1.0 for non-attack-weapons or when DPS=0 (unparsed).
    For attack weapons, penalizes low DPS and slightly rewards high DPS.
    """
    if not total_dps or total_dps <= 0:
        return 1.0
    if item_class not in DPS_ITEM_CLASSES:
        return 1.0

    brackets = DPS_BRACKETS_2H if item_class in TWO_HAND_CLASSES else DPS_BRACKETS_1H
    thresholds = _select_bracket(brackets, item_level)

    #                    below   terrible  low   decent  good   above
    return _interpolate(total_dps, thresholds,
                        (0.15,  0.15,     0.5,  0.85,   1.0,   1.15))


def _defense_factor(total_defense: int, item_class: str, item_level: int = 0) -> float:
    """Multiplicative defense factor for armor piece scoring.

    Returns 1.0 for non-armor items or when total_defense=0.
    Narrower range than DPS — defense is a softer value signal.
    """
    if not total_defense or total_defense <= 0:
        return 1.0
    if item_class not in DEFENSE_ITEM_CLASSES:
        return 1.0

    brackets = DEFENSE_THRESHOLDS.get(item_class)
    if not brackets:
        return 1.0

    thresholds = _select_bracket(brackets, item_level)

    #                    below  terrible  low   decent  good   above
    return _interpolate(float(total_defense), thresholds,
                        (0.6,  0.6,      0.75, 0.9,    1.0,   1.05))


# ─── Display Names ───────────────────────────────────

# Maps RePoE group name substrings (lowercase) → short overlay label.
# Checked in order; first match wins.
_DISPLAY_NAMES: List[Tuple[str, str]] = [
    # Premium (3.0)
    ("movementvelocity", "MoveSpd"),
    ("movespeed", "MoveSpd"),
    ("socketedgemlevel", "GemLvl"),
    ("skilllevels", "SkillLvl"),
    ("gemlevels", "GemLvl"),
    ("criticalstrikemultiplier", "CritMulti"),
    ("critmulti", "CritMulti"),
    ("criticalmulti", "CritMulti"),
    ("spellcriticalstrikechance", "SpellCrit"),
    ("criticalstrikechance", "CritChance"),
    ("critchance", "CritChance"),
    ("spelldamage", "SpellDmg"),
    ("physicaldamagereduction", "Armour"),
    ("physicaldamage", "PhysDmg"),
    # Key (2.0)
    ("attackspeed", "AtkSpd"),
    ("castspeed", "CastSpd"),
    ("firedamage", "FireDmg"),
    ("colddamage", "ColdDmg"),
    ("lightningdamage", "LightDmg"),
    ("chaosdamage", "ChaosDmg"),
    ("elementaldamage", "EleDmg"),
    ("damagetophysical", "AddPhys"),
    ("damagetofire", "AddFire"),
    ("damagetocold", "AddCold"),
    ("damagetolightning", "AddLight"),
    ("damagetochaos", "AddChaos"),
    ("manareservation", "ManaRes"),
    ("liferecoup", "Recoup"),
    ("lifeonhit", "LifeOnHit"),
    ("lifeleech", "Leech"),
    ("projectilespeed", "ProjSpd"),
    ("areadamage", "AreaDmg"),
    ("areaofdamage", "AreaDmg"),
    # Standard (1.0)
    ("maximumlife", "Life"),
    ("increasedlife", "Life"),
    ("energyshieldregeneration", "ESRegen"),
    ("energyshield", "ES"),
    ("maximummana", "Mana"),
    ("increasedmana", "Mana"),
    ("spirit", "Spirit"),
    ("armour", "Armour"),
    ("evasion", "Evasion"),
    ("defencespercent", "Def%"),
    ("alldefences", "AllDef"),
    # Filler (0.3)
    ("allresist", "AllRes"),
    ("elementalresist", "AllRes"),
    ("fireresist", "FireRes"),
    ("coldresist", "ColdRes"),
    ("lightningresist", "LightRes"),
    ("chaosresist", "ChaosRes"),
    ("resistance", "Res"),
    ("allattributes", "AllAttr"),
    ("strength", "Str"),
    ("dexterity", "Dex"),
    ("intelligence", "Int"),
    ("accuracy", "Acc"),
    ("liferegeneration", "LifeRegen"),
    ("manaregeneration", "ManaRegen"),
    ("energyshieldrecharge", "ESRecharge"),
    ("regen", "Regen"),
    ("flask", "Flask"),
    ("stun", "Stun"),
    ("block", "Block"),
    # Near-zero (0.1)
    ("thorns", "Thorns"),
    ("lightradius", "Light"),
    ("itemrarity", "Rarity"),
    ("itemfoundrarity", "Rarity"),
]


def _display_name(group: str) -> str:
    """Convert a RePoE mod group name to a short overlay-friendly label."""
    group_lower = group.lower()
    for pattern, label in _DISPLAY_NAMES:
        if pattern in group_lower:
            return label
    # Fallback: strip common prefixes and add spaces before capitals
    for prefix in ("Increased", "Local", "Added", "Base", "Maximum", "Percentage"):
        if group.startswith(prefix):
            group = group[len(prefix):]
            break
    # "CriticalStrikeChance" → "Critical Strike Chance" → trim to first 2 words
    spaced = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', group)
    words = spaced.split()
    if len(words) > 2:
        return ''.join(w[0] for w in words)  # Acronym: "CSC"
    return spaced


# ─── Stat ID Bridge ──────────────────────────────────

# Normalize a RePoE text template to a comparable form:
#   "(10-19)" → "#",  "[tag|display]" → "display",  collapse whitespace
_RANGE_RE = re.compile(r"\([\d.+-]+\)")
_TAG_DISPLAY_RE = re.compile(r"\[([^|]*?\|)?([^\]]+)\]")
_WHITESPACE_RE = re.compile(r"\s+")
# Trailing parenthetical tags the trade API appends: (Local), (Minion), etc.
_TRAILING_TAG_RE = re.compile(r"\s*\([a-z]+\)\s*$")


def _normalize_for_bridge(text: str) -> str:
    """Final normalization step common to both RePoE and trade texts.

    Strips leading +/- signs before '#', removes trailing (Local)/(Minion)/etc.,
    lowercases, and collapses whitespace.
    """
    out = text.lower().strip()
    # Strip trailing parenthetical tags like (local), (minion)
    out = _TRAILING_TAG_RE.sub("", out)
    # Strip leading +/- before the first # placeholder
    out = re.sub(r'^[+\-]\s*(?=#)', '', out)
    # Collapse whitespace
    out = _WHITESPACE_RE.sub(" ", out).strip()
    return out


def _normalize_repoe_text(text: str) -> str:
    """Normalize RePoE mod text to match trade API template format.

    Examples:
        "+(10-19) to maximum Life"          → "# to maximum life"
        "+(200-214) to [Life|maximum Life]"  → "# to maximum life"
        "Adds (5-10) to (15-20) Fire Damage" → "adds # to # fire damage"
    """
    # Replace (min-max) ranges with #
    out = _RANGE_RE.sub("#", text)
    # Strip [tag|display] markup → keep display part
    out = _TAG_DISPLAY_RE.sub(r"\2", out)
    return _normalize_for_bridge(out)


def _normalize_trade_text(text: str) -> str:
    """Normalize trade API stat template to comparable form.

    Trade templates already use '#' for values.  We strip leading +/-,
    trailing (Local)/(Minion), lowercase, and collapse whitespace.
    """
    return _normalize_for_bridge(text)


# ─── ModDatabase ──────────────────────────────────────

class ModDatabase:
    """Local mod tier database and scoring engine.

    Usage:
        db = ModDatabase()
        ok = db.load(mod_parser)
        if ok:
            score = db.score_item(item, mods)
    """

    def __init__(self):
        self._loaded = False
        # Raw RePoE data
        self._mods_data: Dict = {}
        self._mods_by_base_data: Dict = {}
        self._base_items_data: Dict = {}
        # Processed structures
        # Bridge: trade stat_id → (group, generation_type)
        self._bridge: Dict[str, Tuple[str, str]] = {}
        # Tier ladders: (group, item_class) → TierLadder
        self._ladders: Dict[Tuple[str, str], TierLadder] = {}
        # item_class aliases: maps clipboard item_class → RePoE item_class
        self._class_aliases: Dict[str, str] = {}

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self, mod_parser) -> bool:
        """Download/cache RePoE files, build bridge + tier ladders.

        Args:
            mod_parser: ModParser instance (must have load_stats() called).
                        Used to build the stat ID bridge from trade stat defs.

        Returns:
            True if database loaded successfully.
        """
        t0 = time.time()

        # Step 1: Download/cache RePoE JSON files
        if not self._load_repoe_data():
            logger.warning("ModDatabase: failed to load RePoE data")
            return False

        # Step 2: Build stat ID bridge (trade stat_id → RePoE group)
        self._build_bridge(mod_parser)

        # Step 3: Build tier ladders from mods_by_base
        self._build_ladders()

        elapsed = (time.time() - t0) * 1000
        logger.info(
            f"ModDatabase: ready in {elapsed:.0f}ms — "
            f"bridge={len(self._bridge)} mappings, "
            f"ladders={len(self._ladders)}"
        )
        self._loaded = True
        return True

    def score_item(self, item, mods) -> ItemScore:
        """Score a complete item.

        Args:
            item: ParsedItem with .item_class, .base_type, .item_level
            mods: List[ParsedMod] from ModParser

        Returns:
            ItemScore with grade + breakdown
        """
        item_class = self._resolve_item_class(item)
        mod_scores = []
        prefix_count = 0
        suffix_count = 0

        item_level = getattr(item, 'item_level', 0) or 0

        for mod in mods:
            ms = self._score_mod(mod, item_class, item_level)
            mod_scores.append(ms)
            if ms.generation_type == "prefix":
                prefix_count += 1
            elif ms.generation_type == "suffix":
                suffix_count += 1

        # Compute normalized score
        total_weighted = sum(ms.weighted_score for ms in mod_scores)
        max_possible = sum(ms.weight for ms in mod_scores)
        normalized = total_weighted / max_possible if max_possible > 0 else 0.0

        # Apply SOMV (Sum of Mod Values) factor — rewards items where mods
        # rolled well within their tiers.  A perfect-roll tri-res ring gets
        # a ~10% boost; an all-bottom-roll item gets a ~10% penalty.
        # Only mods with tier data contribute; neutral (0.5) for unknown mods.
        somv_mods = [ms for ms in mod_scores if ms.tier is not None]
        if somv_mods:
            avg_roll = sum(ms.roll_quality for ms in somv_mods) / len(somv_mods)
            # Map avg_roll (0.0-1.0) to factor (0.90-1.10)
            somv_factor = 0.90 + 0.20 * avg_roll
        else:
            somv_factor = 1.0

        normalized = normalized * somv_factor

        # Apply DPS/defense combat factors
        item_class_raw = getattr(item, 'item_class', '') or ''
        total_dps = getattr(item, 'total_dps', 0.0) or 0.0
        total_defense = getattr(item, 'total_defense', 0) or 0

        d_factor = _dps_factor(total_dps, item_class_raw, item_level)
        a_factor = _defense_factor(total_defense, item_class_raw, item_level)
        combat_factor = min(d_factor, a_factor)  # only one applies per item

        normalized = max(0.0, min(1.0, normalized * combat_factor))

        # Count key mods and high-tier key mods
        key_mods = [ms for ms in mod_scores if ms.is_key_mod]
        high_tier_key = [
            ms for ms in key_mods
            if ms.tier and ms.tier.tier_num <= 2
        ]

        # Count valuable T1/T2 mods (weight >= 1.0 excludes filler like thorns/rarity)
        valuable_top_tier = [
            ms for ms in mod_scores
            if ms.tier and ms.tier.tier_num <= 2 and ms.weight >= 1.0
        ]

        # Count special affixes (fractured/desecrated — permanent and rare)
        special_affix_count = sum(
            1 for mod in mods if mod.mod_type in ("fractured", "desecrated"))

        # Grade assignment
        grade = self._assign_grade(normalized, key_mods, high_tier_key,
                                   total_mods=len(mod_scores),
                                   special_affix_count=special_affix_count)

        # Build summary of top mods
        top = sorted(mod_scores, key=lambda m: m.weighted_score, reverse=True)
        summary_parts = []
        for ms in top[:3]:
            if ms.tier_label and ms.mod_group:
                short = _display_name(ms.mod_group)
                summary_parts.append(f"{ms.tier_label} {short}")
        top_mods_summary = ", ".join(summary_parts) if summary_parts else ""

        return ItemScore(
            normalized_score=round(normalized, 3),
            grade=grade,
            prefix_count=prefix_count,
            suffix_count=suffix_count,
            mod_scores=mod_scores,
            top_mods_summary=top_mods_summary,
            top_tier_count=len(valuable_top_tier),
            dps_factor=round(d_factor, 3),
            defense_factor=round(a_factor, 3),
            somv_factor=round(somv_factor, 3),
            total_dps=round(total_dps, 1),
            total_defense=total_defense,
            quality=getattr(item, 'quality', 0) or 0,
            sockets=getattr(item, 'sockets', 0) or 0,
        )

    def get_tier_info(self, stat_id: str, value: float,
                      item_class: str) -> Tuple[str, Optional[TierInfo]]:
        """Single mod tier lookup (for overlay tier labels).

        Returns:
            (tier_label, TierInfo) — e.g. ("T1", TierInfo(...))
        """
        bridge_entry = self._bridge.get(stat_id)
        if not bridge_entry:
            return ("", None)

        group, gen_type = bridge_entry
        ladder = self._ladders.get((group, item_class))
        if not ladder:
            # Try without item_class (universal ladder)
            ladder = self._ladders.get((group, "*"))
        if not ladder:
            return ("", None)

        tier = ladder.identify_tier(value)
        if tier:
            return (f"T{tier.tier_num}", tier)
        return ("", None)

    def get_stats(self) -> dict:
        """Diagnostics: bridge size, ladder count, etc."""
        return {
            "loaded": self._loaded,
            "bridge_size": len(self._bridge),
            "ladder_count": len(self._ladders),
            "repoe_mods": len(self._mods_data),
            "repoe_base_items": len(self._base_items_data),
            "repoe_item_classes": len(self._mods_by_base_data),
        }

    # ─── Internal: Data Loading ───────────────────

    _REPOE_FILES = {
        "mods": "mods.min.json",
        "mods_by_base": "mods_by_base.min.json",
        "base_items": "base_items.min.json",
    }

    def _load_repoe_data(self) -> bool:
        """Download or load cached RePoE JSON files."""
        REPOE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        all_ok = True
        for key, filename in self._REPOE_FILES.items():
            cache_path = REPOE_CACHE_DIR / filename
            data = self._load_cached_or_download(cache_path, filename)
            if data is None:
                all_ok = False
                continue

            if key == "mods":
                self._mods_data = data
            elif key == "mods_by_base":
                self._mods_by_base_data = data
            elif key == "base_items":
                self._base_items_data = data

        if not all_ok:
            # Check if we have at least mods + mods_by_base (minimum viable)
            if self._mods_data and self._mods_by_base_data:
                logger.warning("ModDatabase: base_items missing, proceeding with partial data")
                return True
            return False

        return True

    def _load_cached_or_download(self, cache_path: Path,
                                 filename: str) -> Optional[dict]:
        """Load from disk cache if fresh, otherwise download."""
        # Try cache
        if cache_path.exists():
            age = time.time() - cache_path.stat().st_mtime
            if age < REPOE_CACHE_TTL:
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    logger.debug(f"ModDatabase: loaded {filename} from cache")
                    return data
                except Exception as e:
                    logger.warning(f"ModDatabase: cache read failed for {filename}: {e}")

        # Download
        url = f"{REPOE_BASE_URL}/{filename}"
        try:
            logger.info(f"ModDatabase: downloading {filename}...")
            resp = requests.get(url, timeout=30,
                                headers={"User-Agent": "LAMA/1.0"})
            if resp.status_code != 200:
                logger.warning(f"ModDatabase: HTTP {resp.status_code} for {url}")
                # Fall back to stale cache
                return self._load_stale_cache(cache_path, filename)

            data = resp.json()

            # Save to cache
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, separators=(",", ":"))
                logger.info(f"ModDatabase: cached {filename} ({cache_path.stat().st_size / 1024:.0f} KB)")
            except Exception as e:
                logger.warning(f"ModDatabase: cache write failed for {filename}: {e}")

            return data

        except Exception as e:
            logger.warning(f"ModDatabase: download failed for {filename}: {e}")
            return self._load_stale_cache(cache_path, filename)

    @staticmethod
    def _load_stale_cache(cache_path: Path, filename: str) -> Optional[dict]:
        """Load stale cache as fallback (better stale than nothing)."""
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.warning(f"ModDatabase: using stale cache for {filename}")
                return data
            except Exception:
                pass
        return None

    # ─── Internal: Stat ID Bridge ─────────────────

    def _build_bridge(self, mod_parser):
        """Build mapping from trade stat IDs to RePoE mod groups.

        Matches by normalizing both trade API text templates and RePoE mod
        text, then comparing.  Only considers craftable item mods (domain=item).
        """
        if not hasattr(mod_parser, '_stats') or not mod_parser._stats:
            logger.warning("ModDatabase: mod_parser has no stats, bridge empty")
            return

        # Build lookup: normalized_text → (group, generation_type, mod_key)
        # from RePoE mods.  Item + misc domains (misc includes jewel mods).
        repoe_lookup: Dict[str, Tuple[str, str, str]] = {}
        for mod_key, mod_data in self._mods_data.items():
            if mod_data.get("domain") not in ("item", "misc"):
                continue
            gen_type = mod_data.get("generation_type", "")
            if gen_type not in ("prefix", "suffix"):
                continue
            groups = mod_data.get("groups", [])
            if not groups:
                continue
            text = mod_data.get("text", "")
            if not text:
                continue

            normalized = _normalize_repoe_text(text)
            if normalized and "#" in normalized:
                group = groups[0]
                # Store first match per normalized text (higher tiers
                # have the same text template, just different ranges)
                if normalized not in repoe_lookup:
                    repoe_lookup[normalized] = (group, gen_type, mod_key)

        # Match trade stats against RePoE texts
        matched = 0
        for stat_def in mod_parser._stats:
            if not stat_def.text or "#" not in stat_def.text:
                continue
            trade_normalized = _normalize_trade_text(stat_def.text)
            if not trade_normalized:
                continue

            entry = repoe_lookup.get(trade_normalized)
            if entry:
                group, gen_type, _ = entry
                self._bridge[stat_def.id] = (group, gen_type)
                matched += 1

        logger.info(f"ModDatabase: bridge has {matched} mappings "
                     f"(from {len(mod_parser._stats)} trade stats × "
                     f"{len(repoe_lookup)} RePoE texts)")

    # ─── Internal: Tier Ladders ───────────────────

    def _build_ladders(self):
        """Build tier ladders from mods_by_base data.

        For each item class, iterates all tag combinations, collects
        mod tiers per group, and builds TierLadder objects.

        Multiple tag combinations per item class are merged — the scoring
        uses the superset of available tiers.

        IMPORTANT: mods_by_base uses its own group key names which often
        differ from the groups[0] field in mods.json (e.g. "SpellDamage"
        in mods_by_base vs "WeaponCasterDamagePrefix" in mods.json).
        The bridge uses groups[0] from mods.json, so we register ladders
        under BOTH names to ensure lookups work.
        """
        if not self._mods_by_base_data:
            logger.warning("ModDatabase: no mods_by_base data, no ladders")
            return

        for item_class, tag_groups in self._mods_by_base_data.items():
            for _tag_key, tag_data in tag_groups.items():
                mods_block = tag_data.get("mods", {})
                for gen_type in ("prefix", "suffix"):
                    groups = mods_block.get(gen_type, {})
                    for group_name, mod_tiers_dict in groups.items():
                        # Build tier list from mod_tiers_dict
                        tiers = self._build_tier_list(mod_tiers_dict, gen_type)
                        if not tiers:
                            continue

                        # Find canonical group name from mods.json groups[0]
                        # (which the bridge uses for lookups)
                        canonical_names = {group_name}
                        for mod_key in mod_tiers_dict:
                            mod_data = self._mods_data.get(mod_key, {})
                            mj_groups = mod_data.get("groups", [])
                            if mj_groups:
                                canonical_names.add(mj_groups[0])

                        # Register ladder under all names
                        for name in canonical_names:
                            ladder_key = (name, item_class)
                            self._merge_into_ladder(
                                ladder_key, name, gen_type,
                                item_class, tiers)

        logger.info(f"ModDatabase: built {len(self._ladders)} tier ladders")

    def _merge_into_ladder(self, ladder_key, group_name, gen_type,
                           item_class, tiers):
        """Merge tiers into an existing ladder or create a new one."""
        existing = self._ladders.get(ladder_key)
        if existing:
            existing_keys = {t.mod_key for t in existing.tiers}
            for t in tiers:
                if t.mod_key not in existing_keys:
                    existing.tiers.append(t)
                    existing_keys.add(t.mod_key)
            # Re-sort by required_level descending and renumber
            existing.tiers.sort(
                key=lambda t: t.required_level, reverse=True)
            for i, tier in enumerate(existing.tiers):
                existing.tiers[i] = TierInfo(
                    tier_num=i + 1,
                    mod_key=tier.mod_key,
                    required_level=tier.required_level,
                    stat_min=tier.stat_min,
                    stat_max=tier.stat_max,
                    name=tier.name,
                )
        else:
            self._ladders[ladder_key] = TierLadder(
                group=group_name,
                generation_type=gen_type,
                item_class=item_class,
                tiers=tiers,
            )

    def _build_tier_list(self, mod_tiers_dict: dict,
                         gen_type: str) -> List[TierInfo]:
        """Build a sorted list of TierInfo from a mod group's tier dict.

        Args:
            mod_tiers_dict: {"IncreasedLife1": 1, "IncreasedLife9": 60, ...}
            gen_type: "prefix" or "suffix"
        """
        tiers = []
        for mod_key, req_level in mod_tiers_dict.items():
            mod_data = self._mods_data.get(mod_key)
            if not mod_data:
                continue
            stats = mod_data.get("stats", [])
            if not stats:
                continue
            # Use first stat entry for min/max
            stat = stats[0]
            stat_min = stat.get("min", 0)
            stat_max = stat.get("max", 0)
            name = mod_data.get("name", "")

            tiers.append(TierInfo(
                tier_num=0,  # assigned after sorting
                mod_key=mod_key,
                required_level=req_level,
                stat_min=float(stat_min),
                stat_max=float(stat_max),
                name=name,
            ))

        # Sort by required_level descending (highest ilvl = T1 = best)
        tiers.sort(key=lambda t: t.required_level, reverse=True)

        # Assign tier numbers
        numbered = []
        for i, t in enumerate(tiers):
            numbered.append(TierInfo(
                tier_num=i + 1,
                mod_key=t.mod_key,
                required_level=t.required_level,
                stat_min=t.stat_min,
                stat_max=t.stat_max,
                name=t.name,
            ))
        return numbered

    # ─── Internal: Scoring ────────────────────────

    def _resolve_item_class(self, item) -> str:
        """Get the item_class string matching mods_by_base keys."""
        raw = getattr(item, "item_class", "") or ""
        # Direct match
        if raw in self._mods_by_base_data:
            return raw
        # Try common aliases (clipboard uses plurals, RePoE may not)
        aliases = {
            "Amulets": "Amulet",
            "Rings": "Ring",
            "Belts": "Belt",
            "Gloves": "Gloves",
            "Boots": "Boots",
            "Helmets": "Helmet",
            "Body Armours": "Body Armour",
            "Shields": "Shield",
            "Bows": "Bow",
            "Wands": "Wand",
            "Staves": "Staff",
            "Sceptres": "Sceptre",
            "Daggers": "Dagger",
            "Claws": "Claw",
            "One Hand Swords": "One Hand Sword",
            "Two Hand Swords": "Two Hand Sword",
            "One Hand Axes": "One Hand Axe",
            "Two Hand Axes": "Two Hand Axe",
            "One Hand Maces": "One Hand Mace",
            "Two Hand Maces": "Two Hand Mace",
            "Crossbows": "Crossbow",
            "Flails": "Flail",
            "Spears": "Spear",
            "Quivers": "Quiver",
            "Foci": "Focus",
            "Bucklers": "Buckler",
            "Warstaves": "Warstaff",
        }
        mapped = aliases.get(raw, raw)
        if mapped in self._mods_by_base_data:
            return mapped
        # Try case-insensitive match
        for key in self._mods_by_base_data:
            if key.lower() == raw.lower() or key.lower() == mapped.lower():
                return key
        return raw

    def _score_mod(self, mod, item_class: str, item_level: int = 0) -> ModScore:
        """Score a single parsed mod."""
        bridge_entry = self._bridge.get(mod.stat_id)
        group = ""
        gen_type = mod.mod_type if mod.mod_type in ("prefix", "suffix") else ""
        tier = None
        tier_label = ""
        percentile = 0.5  # default neutral
        roll_quality = 0.5  # default neutral (no tier data)

        if bridge_entry:
            group, gen_type = bridge_entry

            # Look up tier ladder
            ladder = self._ladders.get((group, item_class))
            if not ladder:
                # Try all item classes for this group
                for (g, ic), lad in self._ladders.items():
                    if g == group:
                        ladder = lad
                        break

            if ladder:
                tier = ladder.identify_tier(mod.value, item_level=item_level)
                if tier and item_level:
                    # Log when ilvl filtering changed the tier assignment
                    uncapped = ladder.identify_tier(mod.value, item_level=0)
                    if uncapped and uncapped.tier_num < tier.tier_num:
                        best = ladder.max_tier_for_ilvl(item_level)
                        best_label = f"T{best.tier_num}" if best else "none"
                        logger.debug(
                            f"ilvl {item_level} caps \"{tier.name}\" at T{tier.tier_num} "
                            f"(T{uncapped.tier_num} needs ilvl {uncapped.required_level}) "
                            f"— best rollable={best_label}"
                        )
                if tier:
                    tier_label = f"T{tier.tier_num}"
                    # Compute roll quality within tier
                    t_min = min(abs(tier.stat_min), abs(tier.stat_max))
                    t_max = max(abs(tier.stat_min), abs(tier.stat_max))
                    if t_max > t_min:
                        roll_quality = (abs(mod.value) - t_min) / (t_max - t_min)
                        roll_quality = max(0.0, min(1.0, roll_quality))
                    else:
                        roll_quality = 1.0  # single-value tier = perfect
                # Compute percentile against rollable tiers only
                g_min = ladder.global_min_for_ilvl(item_level)
                g_max = ladder.global_max_for_ilvl(item_level)
                if g_max > g_min:
                    percentile = (abs(mod.value) - g_min) / (g_max - g_min)
                    percentile = max(0.0, min(1.0, percentile))
                else:
                    percentile = 0.5

        # Determine weight
        weight = 1.0  # default standard
        if group:
            w = _get_weight_for_group(group)
            if w is not None:
                weight = w
            else:
                # Unknown group — classify by common pattern
                weight = 0.3 if _is_common_mod(mod.raw_text) else 1.0
        else:
            # No bridge entry — use common/key classification
            weight = 0.3 if _is_common_mod(mod.raw_text) else 1.0

        # Fractured/desecrated mods are inherently premium: permanent (can't
        # reroll) and rare.  Boost their weight to reflect this.
        if mod.mod_type in ("fractured", "desecrated"):
            weight = max(weight, 2.0)  # at least Key-tier weight
            percentile = max(percentile, 0.85)  # treat as high-roll
            roll_quality = max(roll_quality, 0.85)

        # Percentile floor for key/premium mods: even a low-tier roll of a
        # premium mod type is valuable because the mod's PRESENCE matters.
        # E.g. T8 spell damage on a staff is still spell damage.
        # But skip the floor for very deep tiers (T10+) — those are
        # bottom-barrel rolls in oversized ladders (e.g. T40 WDTP).
        tier_num = tier.tier_num if tier else 999
        if weight >= 2.0 and tier_num <= 10:
            percentile = max(percentile, 0.30)

        # Key mod: weight >= 1.0.  When group is known, trust the weight
        # table (Life is weight 1.0 despite matching common patterns).
        # When group is unknown, defer to the common pattern check —
        # BUT always respect boosted weight (desecrated/fractured mods
        # get weight >= 2.0 and should always be key).
        if weight >= 1.0:
            is_key = True
        elif group:
            is_key = False
        else:
            is_key = not _is_common_mod(mod.raw_text)
        weighted_score = percentile * weight

        return ModScore(
            raw_text=mod.raw_text,
            stat_id=mod.stat_id,
            value=mod.value,
            mod_group=group,
            generation_type=gen_type,
            tier=tier,
            tier_label=tier_label,
            percentile=round(percentile, 3),
            weight=weight,
            weighted_score=round(weighted_score, 3),
            is_key_mod=is_key,
            roll_quality=round(roll_quality, 3),
        )

    @staticmethod
    def _assign_grade(normalized: float, key_mods: list,
                      high_tier_key: list,
                      total_mods: int = 0,
                      special_affix_count: int = 0) -> Grade:
        """Assign grade from normalized score + mod composition.

        | Grade | Criteria |
        |-------|----------|
        | S | 2+ T1/T2 key mods AND normalized >= 0.75 AND 3+ total mods |
        | A | 1+ T1/T2 key mods AND normalized >= 0.60 AND 2+ total mods |
        | B | normalized >= 0.45 AND 2+ key mods |
        | C | normalized >= 0.30 OR 1+ key mods |
        | C | no key mods BUT normalized >= 0.65 AND 3+ total mods |
        | JUNK | everything else |

        Special affixes (fractured/desecrated) lower thresholds by 0.10
        per affix — these mods are permanent and rare, making the item
        inherently more valuable than its raw percentile suggests.

        Many key mods (4+) lower thresholds because having multiple
        premium mod types on one item is inherently rare and valuable,
        even if individual rolls are mediocre.

        No key mods at all -> usually JUNK, but pure-defense items with
        very high scores (T1 rolls on all filler mods) can reach C.
        Items with <2 mods can't be S-tier, <2 can't be A-tier (prevents
        single-mod items from inflating grades).
        """
        if not key_mods:
            # Pure-defense/filler items with excellent rolls still deserve C
            # (e.g., T1 Evasion + T1 Evasion + T1 Ailment Threshold = 0.83)
            if normalized >= 0.65 and total_mods >= 3:
                return Grade.C
            return Grade.JUNK

        n_high = len(high_tier_key)
        n_key = len(key_mods)

        # Special affixes lower score thresholds (0.10 per affix, max 0.20)
        bonus = min(special_affix_count * 0.10, 0.20)
        # Many key mods (3+) further lower thresholds — having multiple
        # premium mod types on one item is inherently rare and valuable.
        if n_key >= 3:
            bonus += min((n_key - 2) * 0.05, 0.15)

        if (n_high >= 2 and n_key >= 3
                and normalized >= (0.75 - bonus) and total_mods >= 4):
            return Grade.S
        if n_high >= 1 and normalized >= (0.60 - bonus) and total_mods >= 3:
            return Grade.A
        if normalized >= (0.45 - bonus) and n_key >= 2:
            return Grade.B
        if normalized >= (0.30 - bonus) or n_key >= 1:
            return Grade.C
        return Grade.JUNK


# ─── Test Harness ─────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from mod_parser import ModParser, ParsedMod
    from item_parser import ParsedItem

    print("=" * 70)
    print("  ModDatabase Comprehensive Test Suite")
    print("=" * 70)

    # ── Setup ─────────────────────────────────────────
    print("\n[Setup] Loading ModParser + ModDatabase...")
    mp = ModParser()
    mp.load_stats()
    if not mp.loaded:
        print("FAIL: ModParser could not load stats")
        sys.exit(1)

    db = ModDatabase()
    ok = db.load(mp)
    if not ok:
        print("FAIL: ModDatabase could not load")
        sys.exit(1)

    stats = db.get_stats()
    print(f"  bridge={stats['bridge_size']}, ladders={stats['ladder_count']}, "
          f"mods={stats['repoe_mods']}")

    # ── Stat ID resolver ──────────────────────────────
    # Build a lookup dict: short name -> stat_id
    _stat_patterns = {
        "life":         "# to maximum life",
        "%life":        "#% increased maximum life",
        "mana":         "# to maximum mana",
        "es":           "# to maximum energy shield",
        "%es":          "#% increased maximum energy shield",
        "armour":       "# to armour",
        "%armour":      "#% increased armour",
        "evasion":      "# to evasion rating",
        "%evasion":     "#% increased evasion rating",
        "spirit":       "# to spirit",
        "fire_res":     "fire resistance",
        "cold_res":     "cold resistance",
        "lightning_res": "lightning resistance",
        "chaos_res":    "chaos resistance",
        "all_res":      "all elemental resistances",
        "str":          "# to strength",
        "dex":          "# to dexterity",
        "int":          "# to intelligence",
        "all_attr":     "# to all attributes",
        "atk_spd":      "#% increased attack speed",
        "cast_spd":     "#% increased cast speed",
        "crit_chance":  "#% increased critical hit chance",
        "spell_crit":   "critical hit chance for spells",
        "crit_multi":   "#% increased critical damage bonus",
        "spell_dmg":    "#% increased spell damage",
        "phys_dmg":     "adds # to # physical damage",
        "%phys":        "#% increased physical damage",
        "fire_dmg":     "adds # to # fire damage",
        "cold_dmg":     "adds # to # cold damage",
        "lightning_dmg": "adds # to # lightning damage",
        "ele_dmg_atk":  "#% increased elemental damage with attacks",
        "atk_dmg":      "#% increased attack damage",
        "accuracy":     "# to accuracy rating",
        "proj_speed":   "#% increased projectile speed",
        "life_leech":   "leech #% of physical attack damage as life",
        "life_regen":   "# life regeneration per second",
        "mana_regen":   "#% increased mana regeneration rate",
        "item_rarity":  "#% increased rarity of items found",
        "thorns":       "# to # physical thorns damage",
        "stun_thresh":  "# to stun threshold",
        "life_recoup":  "#% of damage taken recouped as life",
        "block":        "#% increased block chance",
        "mana_cost":    "#% increased mana cost efficiency",
    }

    stat_ids = {}
    missing_stats = []
    for name, pattern in _stat_patterns.items():
        found = None
        for s in mp._stats:
            if s.text and pattern in s.text.lower() and s.type == "explicit":
                found = s
                break
        # Fallback: try enchant type too
        if not found:
            for s in mp._stats:
                if s.text and pattern in s.text.lower():
                    found = s
                    break
        if found:
            stat_ids[name] = found.id
        else:
            missing_stats.append(name)

    print(f"  Resolved {len(stat_ids)}/{len(_stat_patterns)} stat IDs")
    if missing_stats:
        print(f"  Missing: {', '.join(missing_stats)}")

    def mod(name: str, value: float, raw: str = "") -> ParsedMod:
        """Shorthand to create a ParsedMod from a stat name."""
        sid = stat_ids.get(name, f"unknown.{name}")
        return ParsedMod(raw_text=raw or name, stat_id=sid,
                         value=value, mod_type="explicit")

    def item(name, base, cls, ilvl=80, total_dps=0.0, total_defense=0):
        """Shorthand to create a ParsedItem."""
        it = ParsedItem()
        it.name = name
        it.base_type = base
        it.item_class = cls
        it.rarity = "rare"
        it.item_level = ilvl
        it.total_dps = total_dps
        it.total_defense = total_defense
        return it

    # ── Test Cases ────────────────────────────────────
    # Each: (description, expected_grade_range, ParsedItem, [ParsedMod, ...])
    # expected_grade_range is a set of acceptable grades (allows flexibility
    # for borderline cases while catching obvious misclassifications)

    test_cases = [
        # ── S-tier: God-rolled items ──────────────────
        ("S1: God STR Gloves (T1 life + T1 atk spd + T1 crit multi + fire res)",
         {"S"},
         item("Apocalypse Grip", "Plated Gauntlets", "Gloves"),
         [mod("life", 145, "+145 to maximum Life"),
          mod("atk_spd", 24, "24% increased Attack Speed"),
          mod("crit_multi", 40, "40% increased Critical Damage Bonus"),
          mod("fire_res", 25, "+25% to Fire Resistance")]),

        ("S2: God Caster Amulet (T1 spell dmg + T1 cast spd + T1 crit chance + spirit)",
         {"S", "A"},
         item("Damnation Pendant", "Gold Amulet", "Amulets"),
         [mod("spell_dmg", 90, "90% increased Spell Damage"),
          mod("cast_spd", 24, "24% increased Cast Speed"),
          mod("crit_chance", 35, "35% increased Critical Hit Chance"),
          mod("spirit", 30, "+30 to Spirit")]),

        ("S3: God Phys Bow (T1 %phys + T1 flat phys + T1 crit + T1 atk spd)",
         {"S", "A"},
         item("Armageddon Thirst", "Recurve Bow", "Bows"),
         [mod("%phys", 170, "170% increased Physical Damage"),
          mod("phys_dmg", 50, "Adds 30 to 50 Physical Damage"),
          mod("crit_chance", 35, "35% increased Critical Hit Chance"),
          mod("atk_spd", 24, "24% increased Attack Speed")]),

        # ── A-tier: Strong items ──────────────────────
        ("A1: Good Boots (T1 life + T2 all res + cold res)",
         {"A", "B"},
         item("Storm Trail", "Wrapped Boots", "Boots"),
         [mod("life", 130, "+130 to maximum Life"),
          mod("all_res", 12, "+12% to all Elemental Resistances"),
          mod("cold_res", 35, "+35% to Cold Resistance")]),

        ("A2: Good ES Helmet (T1 flat ES + T1 %ES + int)",
         {"S", "A"},
         item("Entropy Crown", "Arcane Crown", "Helmets"),
         [mod("es", 100, "+100 to maximum Energy Shield"),
          mod("%es", 80, "80% increased maximum Energy Shield"),
          mod("int", 40, "+40 to Intelligence")]),

        ("A3: Good Ring (T1 crit multi + T2 life + lightning res)",
         {"S", "A"},
         item("Horror Turn", "Ruby Ring", "Rings"),
         [mod("crit_multi", 38, "38% increased Critical Damage Bonus"),
          mod("life", 70, "+70 to maximum Life"),
          mod("lightning_res", 30, "+30% to Lightning Resistance")]),

        ("A4: Good Two-Hand Sword (T1 %phys + T2 flat phys + T3 atk spd)",
         {"S", "A"},
         item("Dread Edge", "Broad Sword", "Two Hand Swords"),
         [mod("%phys", 160, "160% increased Physical Damage"),
          mod("phys_dmg", 40, "Adds 25 to 40 Physical Damage"),
          mod("atk_spd", 14, "14% increased Attack Speed")]),

        ("A5: Good Sceptre (T1 spell dmg + T1 ele dmg + cast speed)",
         {"S", "A"},
         item("Vortex Sceptre", "Blood Sceptre", "Sceptres"),
         [mod("spell_dmg", 85, "85% increased Spell Damage"),
          mod("ele_dmg_atk", 35, "35% increased Elemental Damage with Attacks"),
          mod("cast_spd", 18, "18% increased Cast Speed")]),

        # ── B-tier: Decent items ──────────────────────
        ("B1: Decent Body Armour (T3 life + T4 armour + fire res)",
         {"B", "C"},
         item("Havoc Shell", "Full Plate", "Body Armours"),
         [mod("life", 70, "+70 to maximum Life"),
          mod("armour", 200, "+200 to Armour"),
          mod("fire_res", 28, "+28% to Fire Resistance")]),

        ("B2: Decent Belt (T2 life + T4 fire res + T5 cold res)",
         {"B", "C"},
         item("Storm Cord", "Leather Belt", "Belts"),
         [mod("life", 80, "+80 to maximum Life"),
          mod("fire_res", 22, "+22% to Fire Resistance"),
          mod("cold_res", 18, "+18% to Cold Resistance")]),

        ("B3: Decent Shield (T2 %armour + T3 life + lightning res)",
         {"B", "C"},
         item("Rampart Tower", "Tower Shield", "Shields"),
         [mod("%armour", 80, "80% increased Armour"),
          mod("life", 65, "+65 to maximum Life"),
          mod("lightning_res", 25, "+25% to Lightning Resistance")]),

        ("B4: Decent Wand (T3 spell dmg + T3 cast spd + mana)",
         {"B", "C", "A"},
         item("Ghoul Song", "Bone Wand", "Wands"),
         [mod("spell_dmg", 55, "55% increased Spell Damage"),
          mod("cast_spd", 14, "14% increased Cast Speed"),
          mod("mana", 50, "+50 to maximum Mana")]),

        ("B5: Decent Crossbow (T2 %phys + T3 crit + cold dmg)",
         {"B", "C", "A"},
         item("Storm Bane", "Gemini Crossbow", "Crossbows"),
         [mod("%phys", 120, "120% increased Physical Damage"),
          mod("crit_chance", 20, "20% increased Critical Hit Chance"),
          mod("cold_dmg", 30, "Adds 15 to 30 Cold Damage")]),

        # ── C-tier: Mediocre items ────────────────────
        ("C1: Mediocre Gloves (T5 life + low atk spd + T5 res)",
         {"C"},
         item("Kraken Mitts", "Ringmail Gauntlets", "Gloves"),
         [mod("life", 35, "+35 to maximum Life"),
          mod("atk_spd", 7, "7% increased Attack Speed"),
          mod("fire_res", 15, "+15% to Fire Resistance")]),

        ("C2: Mediocre Amulet (T4 int + T5 mana + low res)",
         {"C", "JUNK"},
         item("Skull Choker", "Jade Amulet", "Amulets"),
         [mod("int", 18, "+18 to Intelligence"),
          mod("mana", 30, "+30 to maximum Mana"),
          mod("cold_res", 12, "+12% to Cold Resistance")]),

        ("C3: Mediocre Helmet (low life + low armour)",
         {"C", "JUNK"},
         item("Doom Cage", "Iron Hat", "Helmets"),
         [mod("life", 25, "+25 to maximum Life"),
          mod("armour", 50, "+50 to Armour")]),

        ("C4: Mediocre Ring (low str + low fire res)",
         {"C", "JUNK"},
         item("Grim Band", "Iron Ring", "Rings"),
         [mod("str", 12, "+12 to Strength"),
          mod("fire_res", 14, "+14% to Fire Resistance")]),

        # ── JUNK: Trash items ─────────────────────────
        ("J1: All filler resistances (no key mods)",
         {"JUNK"},
         item("Ash Wrap", "Chain Gloves", "Gloves"),
         [mod("fire_res", 18, "+18% to Fire Resistance"),
          mod("cold_res", 15, "+15% to Cold Resistance"),
          mod("lightning_res", 12, "+12% to Lightning Resistance")]),

        ("J2: Pure attributes only",
         {"JUNK"},
         item("Drake Ring", "Iron Ring", "Rings"),
         [mod("str", 10, "+10 to Strength"),
          mod("dex", 12, "+12 to Dexterity"),
          mod("int", 8, "+8 to Intelligence")]),

        ("J3: Near-zero weight mods (thorns + stun)",
         {"JUNK", "C"},
         item("Pain Carapace", "Full Plate", "Body Armours"),
         [mod("thorns", 15, "10 to 15 Physical Thorns damage"),
          mod("stun_thresh", 50, "+50 to Stun Threshold")]),

        ("J4: Single low-tier filler mod",
         {"JUNK"},
         item("Gale Coif", "Iron Hat", "Helmets"),
         [mod("cold_res", 10, "+10% to Cold Resistance")]),

        # ── SOMV: Roll Quality Tests ──────────────────
        # These pairs compare high-roll vs low-roll versions of the same mods.
        # High rolls should score higher due to SOMV factor.
        ("V1: Tri-res ring — PERFECT rolls (45/43/40)",
         {"C", "B"},
         item("Godly Band", "Ruby Ring", "Rings"),
         [mod("fire_res", 45, "+45% to Fire Resistance"),
          mod("cold_res", 43, "+43% to Cold Resistance"),
          mod("lightning_res", 40, "+40% to Lightning Resistance")]),

        ("V2: Tri-res ring — BOTTOM rolls (12/10/11)",
         {"JUNK"},
         item("Trash Band", "Ruby Ring", "Rings"),
         [mod("fire_res", 12, "+12% to Fire Resistance"),
          mod("cold_res", 10, "+10% to Cold Resistance"),
          mod("lightning_res", 11, "+11% to Lightning Resistance")]),

        ("V3: Life + resist ring — HIGH rolls (T3 life 78, T1 fire 45, T1 cold 44)",
         {"C"},
         item("Inferno Loop", "Ruby Ring", "Rings"),
         [mod("life", 78, "+78 to maximum Life"),
          mod("fire_res", 45, "+45% to Fire Resistance"),
          mod("cold_res", 44, "+44% to Cold Resistance")]),

        ("V4: Life + resist ring — LOW rolls (T5 life 25, T6 fire 14, T6 cold 12)",
         {"C", "JUNK"},
         item("Dim Loop", "Ruby Ring", "Rings"),
         [mod("life", 25, "+25 to maximum Life"),
          mod("fire_res", 14, "+14% to Fire Resistance"),
          mod("cold_res", 12, "+12% to Cold Resistance")]),

        ("V5: Gloves T1 mods perfect rolls (life 145, atk spd 24, crit multi 42)",
         {"S"},
         item("Divine Grip", "Plated Gauntlets", "Gloves"),
         [mod("life", 145, "+145 to maximum Life"),
          mod("atk_spd", 24, "24% increased Attack Speed"),
          mod("crit_multi", 42, "42% increased Critical Damage Bonus"),
          mod("fire_res", 45, "+45% to Fire Resistance")]),

        ("V6: Gloves same tiers bottom rolls (life 100, atk spd 13, crit multi 25)",
         {"B", "A", "C"},
         item("Worn Grip", "Plated Gauntlets", "Gloves"),
         [mod("life", 100, "+100 to maximum Life"),
          mod("atk_spd", 13, "13% increased Attack Speed"),
          mod("crit_multi", 25, "25% increased Critical Damage Bonus"),
          mod("fire_res", 20, "+20% to Fire Resistance")]),

        # ── Edge Cases ────────────────────────────────
        ("E1: Unknown item class (falls back to any ladder match)",
         {"S", "A", "B", "C"},
         item("Mystery Box", "Unknown Base", "FooBarBaz"),
         [mod("life", 100, "+100 to maximum Life"),
          mod("crit_multi", 30, "30% increased Critical Damage Bonus")]),

        ("E2: Mod not in bridge (fake stat ID + life)",
         {"B", "C"},
         item("Enigma Grip", "Chain Gloves", "Gloves"),
         [ParsedMod(raw_text="50% increased Foo Power", stat_id="explicit.stat_fake_123",
                    value=50.0, mod_type="explicit"),
          mod("life", 60, "+60 to maximum Life")]),

        ("E3: Empty mod list",
         {"JUNK"},
         item("Bare Plate", "Full Plate", "Body Armours"),
         []),

        ("E4: Single god mod only (T1 crit multi — capped at C by mod count)",
         {"C"},
         item("Havoc Loop", "Ruby Ring", "Rings"),
         [mod("crit_multi", 42, "42% increased Critical Damage Bonus")]),

        ("E5: Value exceeding T1 max (sanctified/corrupted body armour)",
         {"S", "A", "B"},
         item("Transcendent Mail", "Full Plate", "Body Armours"),
         [mod("life", 250, "+250 to maximum Life"),
          mod("%life", 15, "15% increased maximum Life"),
          mod("fire_res", 45, "+45% to Fire Resistance"),
          mod("cold_res", 42, "+42% to Cold Resistance")]),

        ("E6: Body armour mid-tier life (+145 is T5 on body, not T1)",
         {"C", "B"},
         item("Fortress Plate", "Full Plate", "Body Armours"),
         [mod("life", 145, "+145 to maximum Life"),
          mod("all_res", 14, "+14% to all Elemental Resistances"),
          mod("str", 40, "+40 to Strength")]),

        ("E7: Dagger with crit + spell + attack hybrid",
         {"A", "S", "B"},
         item("Soul Fang", "Stiletto", "Daggers"),
         [mod("spell_dmg", 70, "70% increased Spell Damage"),
          mod("crit_chance", 30, "30% increased Critical Hit Chance"),
          mod("atk_spd", 20, "20% increased Attack Speed")]),

        ("E8: Focus with ES + spirit + cast speed",
         {"A", "S", "B", "C"},
         item("Omen Lens", "Bone Focus", "Foci"),
         [mod("es", 80, "+80 to maximum Energy Shield"),
          mod("spirit", 25, "+25 to Spirit"),
          mod("cast_spd", 20, "20% increased Cast Speed")]),

        # ── DPS/Defense Factor Tests ─────────────────
        ("D1: Low DPS bow (100 dps, ilvl 80) — crushed to JUNK",
         {"JUNK", "C"},
         item("Trash Bow", "Recurve Bow", "Bows", ilvl=80, total_dps=100),
         [mod("%phys", 170, "170% increased Physical Damage"),
          mod("phys_dmg", 50, "Adds 30 to 50 Physical Damage"),
          mod("crit_chance", 35, "35% increased Critical Hit Chance"),
          mod("atk_spd", 24, "24% increased Attack Speed")]),

        ("D2: Good DPS bow (400 dps, ilvl 80) — grade unchanged",
         {"S", "A"},
         item("Storm Thirst", "Recurve Bow", "Bows", ilvl=80, total_dps=400),
         [mod("%phys", 170, "170% increased Physical Damage"),
          mod("phys_dmg", 50, "Adds 30 to 50 Physical Damage"),
          mod("crit_chance", 35, "35% increased Critical Hit Chance"),
          mod("atk_spd", 24, "24% increased Attack Speed")]),

        ("D3: Low defense body armour (150 total) — penalized",
         {"C", "JUNK"},
         item("Weak Plate", "Full Plate", "Body Armours", total_defense=150),
         [mod("life", 70, "+70 to maximum Life"),
          mod("armour", 200, "+200 to Armour"),
          mod("fire_res", 28, "+28% to Fire Resistance")]),

        ("D4: Wand — DPS factor = 1.0 (excluded from DPS scoring)",
         {"B", "C", "A"},
         item("Caster Wand", "Bone Wand", "Wands", total_dps=50),
         [mod("spell_dmg", 55, "55% increased Spell Damage"),
          mod("cast_spd", 14, "14% increased Cast Speed"),
          mod("mana", 50, "+50 to maximum Mana")]),

        ("D5: Ring — both factors = 1.0 (excluded from both)",
         {"S", "A"},
         item("Plain Ring", "Ruby Ring", "Rings", total_dps=0, total_defense=0),
         [mod("crit_multi", 38, "38% increased Critical Damage Bonus"),
          mod("life", 70, "+70 to maximum Life"),
          mod("lightning_res", 30, "+30% to Lightning Resistance")]),
    ]

    # ── Run Tests ─────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Running {len(test_cases)} test cases")
    print(f"{'='*70}")

    passed = 0
    failed = 0
    warnings = 0

    for desc, expected_grades, test_item, test_mods in test_cases:
        score = db.score_item(test_item, test_mods)
        grade_str = score.grade.value
        ok = grade_str in expected_grades

        # Status indicator
        if ok:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1

        # Compact output
        mods_summary = score.top_mods_summary or "(no mods)"
        factors = ""
        if score.dps_factor != 1.0:
            factors += f"  dps_f={score.dps_factor:.2f}"
        if score.defense_factor != 1.0:
            factors += f"  def_f={score.defense_factor:.2f}"
        if score.somv_factor != 1.0:
            factors += f"  somv={score.somv_factor:.3f}"
        print(f"\n  [{status}] {desc}")
        print(f"         Grade={grade_str} (expected {'/'.join(sorted(expected_grades))})"
              f"  score={score.normalized_score:.3f}"
              f"  P={score.prefix_count} S={score.suffix_count}{factors}")
        print(f"         {mods_summary}")

        if not ok:
            # Show detailed breakdown for failures
            for ms in score.mod_scores:
                print(f"           [{ms.tier_label or '?':>3}] {ms.raw_text}"
                      f"  w={ms.weight} pct={ms.percentile:.2f}"
                      f"  ws={ms.weighted_score:.2f}"
                      f"  key={ms.is_key_mod}")

    # ── Tier Comparison: Body Armour vs Gloves ────────
    print(f"\n{'='*70}")
    print(f"  Tier Comparison: Life on Body Armours vs Gloves")
    print(f"{'='*70}")
    if "life" in stat_ids:
        for cls, vals in [("Body Armours", [145, 100, 60, 30]),
                          ("Gloves", [145, 100, 60, 30])]:
            print(f"\n  {cls}:")
            for v in vals:
                label, info = db.get_tier_info(stat_ids["life"], float(v), cls)
                if info:
                    print(f"    +{v} Life -> {label} ({info.name}), "
                          f"range {info.stat_min:.0f}-{info.stat_max:.0f}")
                else:
                    print(f"    +{v} Life -> {label or '?'} (no tier info)")

    # ── DPS/Defense Factor Diagnostics ───────────────
    print(f"\n{'='*70}")
    print(f"  DPS/Defense Factor Diagnostics")
    print(f"{'='*70}")

    print("\n  DPS factors (Bows, 2H, ilvl 80):")
    for dps_val in [50, 100, 150, 250, 400, 600, 800]:
        f = _dps_factor(dps_val, "Bows", 80)
        print(f"    {dps_val:>4} DPS -> factor {f:.3f}")

    print("\n  DPS factors (One Hand Swords, 1H, ilvl 80):")
    for dps_val in [30, 80, 150, 250, 400, 500]:
        f = _dps_factor(dps_val, "One Hand Swords", 80)
        print(f"    {dps_val:>4} DPS -> factor {f:.3f}")

    print("\n  DPS factors (Wands — excluded):")
    for dps_val in [50, 100, 200]:
        f = _dps_factor(dps_val, "Wands", 80)
        print(f"    {dps_val:>4} DPS -> factor {f:.3f}")

    print("\n  Defense factors (Body Armours, ilvl 80):")
    for def_val in [100, 200, 400, 700, 1000, 1200]:
        f = _defense_factor(def_val, "Body Armours", 80)
        print(f"    {def_val:>4} def -> factor {f:.3f}")

    print("\n  Defense factors (Gloves, ilvl 80):")
    for def_val in [50, 80, 160, 280, 400, 500]:
        f = _defense_factor(def_val, "Gloves", 80)
        print(f"    {def_val:>4} def -> factor {f:.3f}")

    print("\n  Defense factors (Rings — excluded):")
    for def_val in [50, 100]:
        f = _defense_factor(def_val, "Rings", 80)
        print(f"    {def_val:>4} def -> factor {f:.3f}")

    # ── SOMV Diagnostics ─────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SOMV (Roll Quality) Diagnostics")
    print(f"{'='*70}")

    # Compare V1 vs V2 (tri-res perfect vs bottom)
    print("\n  Tri-res ring — roll quality comparison:")
    for label, res_vals in [("Perfect rolls", [45, 43, 40]), ("Bottom rolls", [12, 10, 11])]:
        test_mods = [
            mod("fire_res", res_vals[0], f"+{res_vals[0]}% to Fire Resistance"),
            mod("cold_res", res_vals[1], f"+{res_vals[1]}% to Cold Resistance"),
            mod("lightning_res", res_vals[2], f"+{res_vals[2]}% to Lightning Resistance"),
        ]
        test_item = item("Test Ring", "Ruby Ring", "Rings")
        score = db.score_item(test_item, test_mods)
        print(f"    {label} ({res_vals}):")
        print(f"      grade={score.grade.value}  score={score.normalized_score:.3f}  somv={score.somv_factor:.3f}")
        for ms in score.mod_scores:
            dn = _display_name(ms.mod_group) if ms.mod_group else "?"
            print(f"        {dn}: val={ms.value:.0f}  {ms.tier_label}  pct={ms.percentile:.3f}  rq={ms.roll_quality:.3f}")

    # Show roll quality across a range for fire res
    print("\n  Fire resistance roll quality curve (Rings):")
    if "fire_res" in stat_ids:
        for val in [10, 15, 20, 25, 30, 35, 40, 45, 46]:
            test_mods = [mod("fire_res", val, f"+{val}% to Fire Resistance")]
            test_item = item("Test Ring", "Ruby Ring", "Rings")
            score = db.score_item(test_item, test_mods)
            ms = score.mod_scores[0]
            print(f"    +{val:>2}% fire -> {ms.tier_label:>3}  pct={ms.percentile:.3f}  rq={ms.roll_quality:.3f}  somv={score.somv_factor:.3f}")

    # ── Summary ───────────────────────────────────────
    print(f"\n{'='*70}")
    total = passed + failed
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print(f"  All tests passed!")
    else:
        print(f"  {failed} test(s) need investigation")
    print(f"{'='*70}")
