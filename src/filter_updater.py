"""
POE2 Price Overlay - Loot Filter Updater

Parses a NeverSink-style .filter file, re-tiers economy blocks based on
live price data from PriceCache, and writes the updated filter to the
POE2 game directory.

Economy sections updated:
  - currency (s/a/b/c/d/e)
  - currency->emotions (s/a/b/c/d/e)
  - currency->catalysts (s/a/b/c/d/e)
  - currency->essence (s/a/b/c/d/e)
  - currency->omen (s/a/b/c/d/e)
  - sockets->general (s/a/b/c/d/e)
  - uniques (t1/t2/t3/hideable)
  - fragments->generic (a/b/c)
"""

import logging
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import (
    FILTER_OUTPUT_DIR,
    FILTER_UPDATE_INTERVAL,
    FILTER_CURRENCY_CHAOS_THRESHOLDS,
    FILTER_UNIQUE_CHAOS_THRESHOLDS,
    FILTER_FRAGMENT_CHAOS_THRESHOLDS,
    FILTER_LAST_UPDATE_FILE,
    CACHE_DIR,
    STRICTNESS_PRESETS,
)

logger = logging.getLogger(__name__)


# ─── Section definitions ─────────────────────────────────────────

# Maps filter $type to the PriceCache categories used for price lookups.
# "thresholds" key selects which threshold table to use.

ECONOMY_SECTIONS = {
    "currency": {
        "categories": ["currency"],
        "thresholds": "currency",
        "tiers": ["s", "a", "b", "c", "d", "e"],
    },
    "currency->emotions": {
        "categories": ["delirium"],
        "thresholds": "currency",
        "tiers": ["s", "a", "b", "c", "d", "e"],
    },
    "currency->catalysts": {
        "categories": ["breach"],
        "thresholds": "currency",
        "tiers": ["s", "a", "b", "c", "d", "e"],
    },
    "currency->essence": {
        "categories": ["essences"],
        "thresholds": "currency",
        "tiers": ["s", "a", "b", "c", "d", "e"],
    },
    "currency->omen": {
        "categories": ["ritual"],
        "thresholds": "currency",
        "tiers": ["s", "a", "b", "c", "d", "e"],
    },
    "sockets->general": {
        "categories": ["runes", "ultimatum", "idol", "abyss"],
        "thresholds": "currency",
        "tiers": ["s", "a", "b", "c", "d", "e"],
    },
    "uniques": {
        "categories": [
            "unique/accessory", "unique/armour", "unique/flask",
            "unique/jewel", "unique/map", "unique/weapon", "unique/sanctum",
        ],
        "thresholds": "unique",
        "tiers": ["t1", "t2", "t3", "hideable"],
    },
    "fragments->generic": {
        "categories": ["fragments", "vaultkeys"],
        "thresholds": "fragment",
        "tiers": ["a", "b", "c"],
    },
}

# Tiers that are NEVER modified (special mechanics, not economy-driven)
SKIP_TIERS = {
    "restex", "artifactlike", "supplymagic", "supplieslow", "supplylow",
    "socketleveling", "socketleveling1", "socketleveling2",
    "sekhemaring", "twicecorrupteduniques", "vaalmodunique",
    "multispecialhigh", "multispecial", "overqualityuniques",
    "oversocketuniques1", "oversocketuniques2", "earlyleague",
    "corrupteduniques", "vaaltypeuniques", "t3boss",
    "exraretablets",
}

CHAOS_THRESHOLD_TABLES = {
    "currency": FILTER_CURRENCY_CHAOS_THRESHOLDS,
    "unique": FILTER_UNIQUE_CHAOS_THRESHOLDS,
    "fragment": FILTER_FRAGMENT_CHAOS_THRESHOLDS,
}


def build_divine_thresholds(divine_to_chaos: float) -> dict:
    """Convert chaos-based thresholds to divine values using the current rate."""
    result = {}
    for key, chaos_table in CHAOS_THRESHOLD_TABLES.items():
        result[key] = {}
        for tier, chaos_val in chaos_table.items():
            if chaos_val == 0 or divine_to_chaos <= 0:
                result[key][tier] = 0.0
            else:
                result[key][tier] = chaos_val / divine_to_chaos
    return result


# ─── Filter block data model ─────────────────────────────────────

@dataclass
class FilterBlock:
    """One Show/Hide rule from the filter file."""
    header_line: str          # "Show # $type->currency $tier->s ..." or "#Show ..."
    body_lines: list = field(default_factory=list)  # indented lines after header
    section_type: str = ""    # e.g. "currency", "currency->emotions"
    tier: str = ""            # e.g. "s", "a", "t1", "exhide"
    is_commented: bool = False  # #Show / #Hide block
    is_show: bool = True      # Show vs Hide
    base_types: list = field(default_factory=list)  # parsed BaseType items
    basetype_line_idx: int = -1  # index in body_lines of the BaseType line


def parse_annotation(header: str) -> tuple:
    """Extract $type and $tier from a filter block header comment."""
    type_match = re.search(r'\$type->(\S+)', header)
    tier_match = re.search(r'\$tier->(\S+)', header)
    section_type = type_match.group(1) if type_match else ""
    tier = tier_match.group(1) if tier_match else ""
    return section_type, tier


def parse_basetype_line(line: str) -> list:
    """Extract quoted BaseType names from a line like: BaseType == "Foo" "Bar"."""
    return re.findall(r'"([^"]+)"', line)


def build_basetype_line(prefix: str, items: list) -> str:
    """Build a BaseType line from prefix and sorted item list."""
    quoted = " ".join(f'"{item}"' for item in sorted(items))
    return f"{prefix}{quoted}"


# ─── Parser ───────────────────────────────────────────────────────

def _is_filter_keyword(line: str) -> bool:
    """Check if a line starts with a known filter keyword (condition or styling)."""
    stripped = line.lstrip('#').strip()
    if not stripped:
        return False
    first_word = stripped.split()[0] if stripped.split() else ""
    return first_word in _ALL_FILTER_KEYWORDS


# All keywords that can appear as body lines in a filter block
_ALL_FILTER_KEYWORDS = frozenset({
    # Conditions
    "Class", "BaseType", "Rarity", "AreaLevel", "ItemLevel",
    "StackSize", "Sockets", "Quality", "TwiceCorrupted",
    "HasVaalUniqueMod", "IsVaalUnique", "AnyEnchantment", "Corrupted",
    "Width", "Height", "DropLevel", "UnidentifiedItemTier",
    "Mirrored", "WaystoneTier",
    # Styling
    "SetFontSize", "SetTextColor", "SetBorderColor", "SetBackgroundColor",
    "PlayAlertSound", "PlayEffect", "MinimapIcon",
    # Control
    "Continue",
})


def parse_filter(lines: list) -> tuple:
    """
    Parse filter lines into blocks and passthrough lines.

    Returns a list of (type, data) tuples:
      ("passthrough", line_string)  — comments, blanks, section headers
      ("block", FilterBlock)        — a Show/Hide rule

    Handles both tab-indented blocks and non-indented blocks
    (the latter terminated by blank lines or next block headers).
    """
    result = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Detect block start: Show/Hide (or #Show/#Hide for commented blocks)
        if re.match(r'^#?(Show|Hide)\b', stripped):
            block = FilterBlock(header_line=line)
            block.is_commented = stripped.startswith('#')
            block.is_show = 'Show' in stripped.split('#')[0] if not block.is_commented else 'Show' in stripped

            block.section_type, block.tier = parse_annotation(stripped)
            i += 1

            # Peek at next line to determine indentation style
            if i < n and (lines[i].startswith('\t') or lines[i].startswith('#\t')):
                # Tab-indented block (most of the filter)
                while i < n:
                    bline = lines[i]
                    if bline.startswith('\t') or bline.startswith('#\t'):
                        block.body_lines.append(bline)
                        clean = bline.lstrip('#').strip()
                        if clean.startswith('BaseType'):
                            block.base_types = parse_basetype_line(clean)
                            block.basetype_line_idx = len(block.body_lines) - 1
                        i += 1
                    else:
                        break
            else:
                # Non-indented block (ut->rare, ut->magic, etc.)
                # Body continues until blank line or next Show/Hide header
                while i < n:
                    bline = lines[i]
                    bstripped = bline.strip()
                    # Stop at blank line
                    if not bstripped:
                        break
                    # Stop at next block header
                    if re.match(r'^#?(Show|Hide)\b', bstripped):
                        break
                    # Stop at section comment headers (=====)
                    if bstripped.startswith('#===') or bstripped.startswith('#---'):
                        break
                    # It's a body line — normalize to tab-indented
                    clean = bline.lstrip('#').strip()
                    if block.is_commented:
                        block.body_lines.append('#\t' + clean)
                    else:
                        block.body_lines.append('\t' + clean)
                    if clean.startswith('BaseType'):
                        block.base_types = parse_basetype_line(clean)
                        block.basetype_line_idx = len(block.body_lines) - 1
                    i += 1

            result.append(("block", block))
        else:
            result.append(("passthrough", line))
            i += 1

    return result


def serialize_filter(parsed: list) -> list:
    """Convert parsed structure back to lines."""
    out = []
    for kind, data in parsed:
        if kind == "passthrough":
            out.append(data)
        else:
            block = data
            out.append(block.header_line)
            out.extend(block.body_lines)
    return out


# ─── Price lookup helpers ─────────────────────────────────────────

def get_item_divine_value(item_name: str, prices: dict) -> Optional[float]:
    """Look up an item's divine value from the price cache dict."""
    key = item_name.strip().lower()
    entry = prices.get(key)
    if entry:
        return entry.get("divine_value", 0)
    return None


def get_unique_base_max_value(base_type: str, prices: dict) -> Optional[float]:
    """
    For uniques, find the highest divine value among all uniques sharing
    a base type. Returns None if no uniques found for this base.
    """
    bt_lower = base_type.strip().lower()
    max_val = None
    for data in prices.values():
        cat = data.get("category", "")
        if not cat.startswith("unique/"):
            continue
        if data.get("base_type", "").lower() == bt_lower:
            dv = data.get("divine_value", 0)
            if max_val is None or dv > max_val:
                max_val = dv
    return max_val


def assign_tier(divine_value: float, threshold_type: str, thresholds: dict) -> str:
    """Assign an item to the correct tier based on its divine value."""
    table = thresholds[threshold_type]
    # Sort tiers by threshold descending so we match highest first
    for tier_name, threshold in sorted(table.items(), key=lambda x: -x[1]):
        if divine_value >= threshold:
            return tier_name
    # Fallback to lowest tier
    return list(table.keys())[-1]


# ─── Core re-tiering logic ───────────────────────────────────────

def retier_filter(parsed: list, prices: dict, divine_to_chaos: float,
                   dry_run: bool = False) -> dict:
    """
    Re-assign items to tiers based on price data.

    Returns a changes dict: {section_type: [(item_name, old_tier, new_tier), ...]}
    Modifies parsed blocks in-place (unless dry_run, but we still compute changes).
    """
    thresholds = build_divine_thresholds(divine_to_chaos)
    changes = {}

    # Step 1: Collect all blocks by section, grouped by type
    section_blocks = {}  # type -> {tier -> FilterBlock}
    for kind, data in parsed:
        if kind != "block":
            continue
        block = data
        if not block.section_type or not block.tier:
            continue
        if block.tier in SKIP_TIERS:
            continue
        if block.section_type not in ECONOMY_SECTIONS:
            continue

        section_cfg = ECONOMY_SECTIONS[block.section_type]
        if block.tier not in section_cfg["tiers"] and block.tier != "exhide":
            continue

        key = block.section_type
        if key not in section_blocks:
            section_blocks[key] = {}
        section_blocks[key][block.tier] = block

    # Step 2: For each economy section, re-tier items
    for section_type, cfg in ECONOMY_SECTIONS.items():
        blocks = section_blocks.get(section_type, {})
        if not blocks:
            continue

        threshold_type = cfg["thresholds"]
        tier_order = cfg["tiers"]
        is_unique = threshold_type == "unique"

        # Collect all items currently in tiered blocks (not exhide)
        current_assignments = {}  # item_name -> current_tier
        for tier_name, block in blocks.items():
            if tier_name == "exhide":
                continue
            for item in block.base_types:
                current_assignments[item] = tier_name

        # Compute new tier for each item
        new_assignments = {}  # item_name -> new_tier
        for item_name, old_tier in current_assignments.items():
            if is_unique:
                dv = get_unique_base_max_value(item_name, prices)
            else:
                dv = get_item_divine_value(item_name, prices)

            if dv is not None:
                new_tier = assign_tier(dv, threshold_type, thresholds)
                new_assignments[item_name] = new_tier
            else:
                # No price data — keep current tier
                new_assignments[item_name] = old_tier

        # Check for items in price data not currently in any tier block
        # (new items that appeared in the economy)
        if not is_unique:
            for cache_key, data in prices.items():
                cat = data.get("category", "")
                if cat not in cfg["categories"]:
                    continue
                item_name = data.get("name", "")
                if not item_name or item_name in new_assignments:
                    continue
                # Check if this item is in a skip-tier block (don't steal it)
                already_placed = False
                for kind2, data2 in parsed:
                    if kind2 != "block":
                        continue
                    if data2.section_type == section_type and data2.tier in SKIP_TIERS:
                        if item_name in data2.base_types:
                            already_placed = True
                            break
                if already_placed:
                    continue
                dv = data.get("divine_value", 0)
                new_tier = assign_tier(dv, threshold_type, thresholds)
                new_assignments[item_name] = new_tier
                current_assignments[item_name] = "(new)"

        # Record changes
        section_changes = []
        for item_name, new_tier in new_assignments.items():
            old_tier = current_assignments.get(item_name, "(new)")
            if old_tier != new_tier:
                section_changes.append((item_name, old_tier, new_tier))
        if section_changes:
            changes[section_type] = section_changes

        if dry_run:
            continue

        # Step 3: Rebuild BaseType lines for each tier block
        tier_items = {t: [] for t in tier_order}
        for item_name, new_tier in new_assignments.items():
            if new_tier in tier_items:
                tier_items[new_tier].append(item_name)

        for tier_name, block in blocks.items():
            if tier_name == "exhide":
                continue
            if tier_name not in tier_items:
                continue
            items = tier_items[tier_name]
            _update_block_basetypes(block, items)

        # Step 4: Rebuild exhide block (contains items from b-tier and below)
        if "exhide" in blocks:
            exhide_block = blocks["exhide"]
            # exhide gets all items NOT in the top 2 tiers (s,a for currency; t1,t2 for uniques; a for fragments)
            top_tiers = set(tier_order[:2])
            exhide_items = []
            for item_name, new_tier in new_assignments.items():
                if new_tier not in top_tiers:
                    exhide_items.append(item_name)
            # Also include items from skip-tier blocks that are supply/low level
            for kind2, data2 in parsed:
                if kind2 != "block":
                    continue
                if data2.section_type == section_type:
                    if data2.tier in ("supplymagic", "supplieslow", "supplylow",
                                      "socketleveling1", "socketleveling2",
                                      "artifactlike"):
                        for item in data2.base_types:
                            if item not in exhide_items:
                                exhide_items.append(item)
            _update_block_basetypes(exhide_block, exhide_items)

    return changes


def _update_block_basetypes(block: FilterBlock, items: list):
    """Update a block's BaseType line with new items. Handles commented blocks."""
    if not items:
        # Empty tier — comment out the block if it isn't already
        if not block.is_commented:
            block.header_line = "#" + block.header_line
            block.body_lines = ["#" + bl for bl in block.body_lines]
            block.is_commented = True
        return

    # Uncomment the block if it was commented and now has items
    if block.is_commented:
        if block.header_line.startswith('#'):
            block.header_line = block.header_line[1:]
        block.body_lines = [bl[1:] if bl.startswith('#') else bl for bl in block.body_lines]
        block.is_commented = False

    if block.basetype_line_idx >= 0:
        old_line = block.body_lines[block.basetype_line_idx].lstrip('#')
        # Preserve the prefix (e.g. "\tBaseType == " or "\tBaseType ")
        bt_match = re.match(r'(\s*BaseType\s*==?\s*)', old_line)
        if bt_match:
            prefix = bt_match.group(1)
        else:
            prefix = "\tBaseType == "
        block.body_lines[block.basetype_line_idx] = build_basetype_line(prefix, items)
    else:
        # Block had no BaseType line (was empty/commented) — insert one after Class line
        insert_idx = 0
        for i, line in enumerate(block.body_lines):
            if line.lstrip('#').strip().startswith("Class"):
                insert_idx = i + 1
                break
        new_line = build_basetype_line("\tBaseType == ", items)
        block.body_lines.insert(insert_idx, new_line)
        block.basetype_line_idx = insert_idx
    block.base_types = sorted(items)


# ─── Value-based styling ──────────────────────────────────────────
#
# Every economy block gets styling that reflects the ACTUAL value of
# the items it contains.  Condition lines (Class, BaseType, Rarity,
# etc.) are preserved; all visual lines are replaced.

# Keywords that are filter conditions or control flow (kept as-is)
_CONDITION_PREFIXES = frozenset({
    "Class", "BaseType", "Rarity", "AreaLevel", "ItemLevel",
    "StackSize", "Sockets", "Quality", "TwiceCorrupted",
    "HasVaalUniqueMod", "IsVaalUnique", "AnyEnchantment", "Corrupted",
    "Width", "Height", "DropLevel", "UnidentifiedItemTier",
    "Mirrored", "WaystoneTier",
    "Continue",  # control flow — preserved for decorator blocks
})

# ── Styling profiles ──

# S tier (>= 25c) — top value, full alert
STYLE_S = [
    "\tSetFontSize 45",
    "\tSetTextColor 255 0 0 255",
    "\tSetBorderColor 255 0 0 255",
    "\tSetBackgroundColor 255 255 255 255",
    "\tPlayAlertSound 6 300",
    "\tPlayEffect Red",
    "\tMinimapIcon 0 Red Star",
]

# A tier (>= 5c) — good value, prominent
STYLE_A = [
    "\tSetFontSize 45",
    "\tSetTextColor 255 255 255 255",
    "\tSetBorderColor 255 255 255 255",
    "\tSetBackgroundColor 245 105 90 255",
    "\tPlayAlertSound 1 300",
    "\tPlayEffect Red",
    "\tMinimapIcon 0 Red Circle",
]

# B tier (>= 2c) — decent, visible
STYLE_B = [
    "\tSetFontSize 42",
    "\tSetTextColor 0 0 0 255",
    "\tSetBorderColor 0 0 0 255",
    "\tSetBackgroundColor 245 105 90 255",
    "\tPlayAlertSound 2 300",
    "\tPlayEffect Yellow",
    "\tMinimapIcon 1 Yellow Circle",
]

# C tier (>= 1c) — marginal, moderate visibility
STYLE_C = [
    "\tSetFontSize 38",
    "\tSetTextColor 0 0 0 255",
    "\tSetBorderColor 0 0 0 255",
    "\tSetBackgroundColor 210 160 80 200",
]

# D tier (same threshold as C, mostly empty)
STYLE_D = STYLE_C

# E tier (< 1c) — white text, no fill, no effects
STYLE_E = [
    "\tSetFontSize 32",
    "\tSetTextColor 200 200 200 255",
]

# Hideable unique (< 0.5c) — white text + tiny orange star as unique marker
STYLE_UNIQUE_HIDEABLE = [
    "\tSetFontSize 32",
    "\tSetTextColor 200 200 200 255",
    "\tMinimapIcon 2 Orange Star",
]

# Unique T3 (>= 0.5c) — muted but visible unique styling
STYLE_UNIQUE_T3 = [
    "\tSetFontSize 38",
    "\tSetTextColor 188 96 37 255",
    "\tSetBorderColor 188 96 37 180",
    "\tPlayEffect Brown Temp",
    "\tMinimapIcon 2 Brown Star",
]

# Exhide test mode — tiny grey text so you can spot-check
STYLE_EXHIDE_TEST = [
    "\tSetFontSize 18",
    "\tSetTextColor 150 150 150 180",
    "\tSetBorderColor 0 0 0 0",
    "\tSetBackgroundColor 0 0 0 150",
]

# Per-rarity gear defaults
STYLE_GEAR_RARE = [
    "\tSetFontSize 28",
    "\tSetTextColor 220 200 100 255",       # yellow
]
STYLE_GEAR_MAGIC = [
    "\tSetFontSize 28",
    "\tSetTextColor 136 136 255 255",       # blue
]
STYLE_GEAR_NORMAL = [
    "\tSetFontSize 28",
    "\tSetTextColor 200 200 200 255",       # grey
]

# Decorator blocks that use Continue — strip all styling, keep just Continue
STYLE_DECORATOR_EMPTY = []

# Section types for rare/magic gear, their decorators, and hide layers
GEAR_SECTION_TYPES = {"ut->rare", "ut->magic"}
GEAR_DECORATOR_TYPES = {"decorators->rareeg"}
# Hide layers that catch remaining rare/magic/normal gear — convert to Show + small text
GEAR_HIDELAYER_TIERS = {"normalmagicendgame", "raresendgame"}

# Fragment tiers
STYLE_FRAG_A = STYLE_A
STYLE_FRAG_B = STYLE_B
STYLE_FRAG_C = STYLE_C

# Map tier names to styling profiles
_CURRENCY_STYLES = {
    "s": STYLE_S, "a": STYLE_A, "b": STYLE_B,
    "c": STYLE_C, "d": STYLE_D, "e": STYLE_E,
}
_UNIQUE_STYLES = {
    "t1": STYLE_S, "t2": STYLE_A,
    "t3": STYLE_UNIQUE_T3, "hideable": STYLE_UNIQUE_HIDEABLE,
}
_FRAGMENT_STYLES = {
    "a": STYLE_FRAG_A, "b": STYLE_FRAG_B, "c": STYLE_FRAG_C,
}

STYLE_MAP = {
    "currency": _CURRENCY_STYLES,
    "unique": _CURRENCY_STYLES,  # sockets, emotions, etc. use currency tiers
    "fragment": _FRAGMENT_STYLES,
}


def _build_style_lines(overrides: dict, base_style: list) -> list:
    """
    Merge user JSON style overrides with base filter style lines.

    overrides keys: font_size, text_color, border_color, bg_color,
                    sound_enabled, sound_id, beam_enabled, beam_color,
                    minimap_enabled
    Colors are hex strings like "#ff0000".
    """
    if not overrides:
        return list(base_style)

    # Parse base style into a dict of keyword -> line
    base_map = {}
    for line in base_style:
        stripped = line.strip()
        if stripped:
            keyword = stripped.split()[0]
            base_map[keyword] = line

    def hex_to_rgba(hex_color, alpha=255):
        """Convert #rrggbb to 'R G B A' string."""
        h = hex_color.lstrip("#")
        if len(h) == 6:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"{r} {g} {b} {alpha}"
        return f"200 200 200 {alpha}"

    result = dict(base_map)

    if "font_size" in overrides:
        result["SetFontSize"] = f"\tSetFontSize {overrides['font_size']}"
    if "text_color" in overrides:
        result["SetTextColor"] = f"\tSetTextColor {hex_to_rgba(overrides['text_color'])}"
    if "border_color" in overrides:
        result["SetBorderColor"] = f"\tSetBorderColor {hex_to_rgba(overrides['border_color'])}"
    if "bg_color" in overrides:
        result["SetBackgroundColor"] = f"\tSetBackgroundColor {hex_to_rgba(overrides['bg_color'])}"

    # Sound: if explicitly disabled, remove; if enabled, set sound ID
    if overrides.get("sound_enabled") is False:
        result.pop("PlayAlertSound", None)
    elif overrides.get("sound_enabled") and "sound_id" in overrides:
        result["PlayAlertSound"] = f"\tPlayAlertSound {overrides['sound_id']} 300"

    # Beam: if explicitly disabled, remove; if enabled, set color
    if overrides.get("beam_enabled") is False:
        result.pop("PlayEffect", None)
    elif overrides.get("beam_enabled") and "beam_color" in overrides:
        result["PlayEffect"] = f"\tPlayEffect {overrides['beam_color']}"

    # Minimap: if explicitly disabled, remove
    if overrides.get("minimap_enabled") is False:
        result.pop("MinimapIcon", None)

    return list(result.values())


def apply_styling_overrides(parsed: list, test_mode: bool = False,
                            user_styles: dict = None):
    """
    Replace styling on every economy block to match actual item value.
    Also downstyles rare/magic gear to small yellow text.
    Keeps all conditions (Class, BaseType, Rarity, etc.) intact.
    """
    for kind, block in parsed:
        if kind != "block":
            continue
        if not block.section_type:
            continue

        # ── Rare/magic gear: rarity-colored text (with user overrides) ──
        if block.section_type in GEAR_SECTION_TYPES:
            if block.section_type == "ut->rare":
                gear_key, style = "gear_rare", STYLE_GEAR_RARE
            else:
                gear_key, style = "gear_magic", STYLE_GEAR_MAGIC
            if user_styles and gear_key in user_styles:
                style = _build_style_lines(user_styles[gear_key], style)
            _replace_styling(block, style)
            continue

        # ── Rare decorators (Continue blocks): strip styling ──
        if block.section_type in GEAR_DECORATOR_TYPES:
            _replace_styling(block, STYLE_DECORATOR_EMPTY)
            continue

        # ── Hide layers for remaining gear: convert to Show + grey text ──
        if block.section_type == "hidelayer" and block.tier in GEAR_HIDELAYER_TIERS:
            style = STYLE_GEAR_NORMAL
            if user_styles and "gear_normal" in user_styles:
                style = _build_style_lines(user_styles["gear_normal"], style)
            _replace_styling(block, style, make_show=True)
            continue

        # ── Economy sections: value-based styling ──
        if block.section_type not in ECONOMY_SECTIONS:
            continue
        if block.tier in SKIP_TIERS:
            continue

        cfg = ECONOMY_SECTIONS[block.section_type]
        threshold_type = cfg["thresholds"]

        if block.tier == "exhide":
            if test_mode:
                _replace_styling(block, STYLE_EXHIDE_TEST, make_show=True)
            continue

        # Look up the styling profile for this tier
        style_table = STYLE_MAP.get(threshold_type, _CURRENCY_STYLES)

        # Special handling for uniques
        if threshold_type == "unique":
            style_table = _UNIQUE_STYLES

        style = style_table.get(block.tier)
        if style:
            # Apply user overrides if present for this tier
            if user_styles and block.tier in user_styles:
                style = _build_style_lines(user_styles[block.tier], style)
            _replace_styling(block, style)


def _replace_styling(block: FilterBlock, style_lines: list,
                     make_show: bool = False):
    """
    Replace all visual styling in a block while preserving conditions.
    Optionally convert Hide to Show.
    """
    if block.is_commented:
        return

    if make_show:
        header = block.header_line
        if header.startswith("Hide"):
            block.header_line = "Show" + header[4:]
            block.is_show = True

    # Split body into conditions and styling
    conditions = []
    for i, line in enumerate(block.body_lines):
        stripped = line.strip()
        first_word = stripped.split()[0] if stripped.split() else ""
        if first_word in _CONDITION_PREFIXES:
            conditions.append(line)

    # Rebuild: conditions + new styling
    block.body_lines = conditions + style_lines

    # Re-find basetype index
    block.basetype_line_idx = -1
    for i, line in enumerate(block.body_lines):
        if line.strip().startswith("BaseType"):
            block.basetype_line_idx = i
            break


# ─── File I/O ─────────────────────────────────────────────────────

def find_template_filter(project_dir: Path) -> Optional[Path]:
    """Find a .filter file in the project directory to use as template."""
    filters = list(project_dir.glob("*.filter"))
    if not filters:
        return None
    # Prefer the first one (usually only one)
    return filters[0]


def read_filter(path: Path) -> list:
    """Read filter file into lines (preserving line endings)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        return [line.rstrip("\n\r") for line in f]


def write_filter(path: Path, lines: list):
    """Atomic write: write to temp file, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Backup existing file
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)

    # Write to temp file in the same directory, then rename
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".filter.tmp", prefix="."
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        # On Windows, need to remove target first
        if path.exists():
            path.unlink()
        Path(tmp_path).rename(path)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ─── Main updater class ──────────────────────────────────────────

class FilterUpdater:
    """
    Background service that periodically re-tiers a loot filter
    based on live price data.
    """

    def __init__(self, price_cache, template_path: Optional[Path] = None,
                 test_mode: bool = False):
        self.price_cache = price_cache
        self.template_path = template_path
        self.test_mode = test_mode
        self._running = False
        self._thread = None

    def start(self):
        """Start the background update thread."""
        if not self.template_path:
            logger.warning("Filter updater: no template .filter found — disabled")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._update_loop, daemon=True, name="FilterUpdater"
        )
        self._thread.start()

    def stop(self):
        self._running = False

    def _update_loop(self):
        """Wait for prices, then update filter periodically."""
        # Wait for price data to be available
        for _ in range(60):
            if not self._running:
                return
            stats = self.price_cache.get_stats()
            if stats["total_items"] > 0:
                break
            time.sleep(1)
        else:
            logger.warning("Filter updater: timed out waiting for price data")
            return

        while self._running:
            # Check if update is needed
            if self._is_stale():
                try:
                    self.update_now()
                except Exception as e:
                    logger.error(f"Filter update failed: {e}", exc_info=True)

            # Sleep until next check (check every minute for shutdown)
            for _ in range(min(FILTER_UPDATE_INTERVAL, 3600)):
                if not self._running:
                    return
                time.sleep(1)

    def _is_stale(self) -> bool:
        """Check if the filter needs updating."""
        try:
            if FILTER_LAST_UPDATE_FILE.exists():
                ts = float(FILTER_LAST_UPDATE_FILE.read_text().strip())
                return (time.time() - ts) > FILTER_UPDATE_INTERVAL
        except (ValueError, OSError):
            pass
        return True

    def _mark_updated(self):
        """Record the current time as last update."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        FILTER_LAST_UPDATE_FILE.write_text(str(time.time()))

    def update_now(self, dry_run: bool = False, user_styles: dict = None,
                   section_visibility: dict = None,
                   strictness: str = "normal",
                   gear_classes: dict = None) -> dict:
        """
        Run a filter update immediately.

        Args:
            dry_run: Compute changes but don't write.
            user_styles: Per-tier style overrides from dashboard UI.
            section_visibility: {section_id: bool} — False hides a section.
            strictness: Preset name from STRICTNESS_PRESETS.
            gear_classes: {gear_key: {class_name: bool}} — per-class visibility.

        Returns the changes dict.
        """
        # Read filter preferences from env var if not provided directly
        # (set by server.py when spawning via subprocess)
        if user_styles is None and section_visibility is None:
            import json as _json
            prefs_json = os.environ.get("POE2_FILTER_PREFS")
            if prefs_json:
                try:
                    prefs = _json.loads(prefs_json)
                    user_styles = prefs.get("filter_tier_styles") or None
                    section_visibility = prefs.get("filter_section_visibility") or None
                    strictness = prefs.get("filter_strictness", strictness)
                    gear_classes = prefs.get("filter_gear_classes") or None
                except (ValueError, TypeError):
                    pass

        template = self.template_path
        if not template or not template.exists():
            logger.error(f"Filter template not found: {template}")
            return {}

        # Read and parse
        lines = read_filter(template)
        parsed = parse_filter(lines)

        # Get prices snapshot and exchange rate
        with self.price_cache._lock:
            prices = dict(self.price_cache.prices)
            divine_to_chaos = self.price_cache.divine_to_chaos

        # Apply strictness multiplier to divine_to_chaos
        # Higher multiplier = higher effective divine_to_chaos = lower divine
        # thresholds = fewer items shown
        multiplier = STRICTNESS_PRESETS.get(strictness, 1.0)
        effective_divine_to_chaos = divine_to_chaos / multiplier if multiplier else divine_to_chaos

        logger.info(
            f"Filter update: 1 divine = {divine_to_chaos:.0f} chaos, "
            f"strictness={strictness} ({multiplier}x)"
        )

        # Re-tier with adjusted thresholds
        changes = retier_filter(parsed, prices, effective_divine_to_chaos,
                                dry_run=dry_run)

        # Apply styling: strip alert sounds from cheap tiers, test mode for exhide
        if not dry_run:
            apply_styling_overrides(parsed, test_mode=self.test_mode,
                                    user_styles=user_styles)

        # Apply section visibility — comment out hidden sections
        if not dry_run and section_visibility:
            _apply_section_visibility(parsed, section_visibility)

        # Apply per-class gear visibility — remove hidden classes from blocks
        if not dry_run and gear_classes:
            _apply_gear_class_filters(parsed, gear_classes)

        if dry_run:
            _log_changes(changes, dry_run=True)
            return changes

        # Serialize
        new_lines = serialize_filter(parsed)

        total_moved = sum(len(v) for v in changes.values())

        # Write output (always write — styling overrides change the template)
        output_path = FILTER_OUTPUT_DIR / template.name
        write_filter(output_path, new_lines)
        self._mark_updated()

        if total_moved:
            _log_changes(changes)
        else:
            logger.info("Filter update: 0 items re-tiered, styling applied")
        logger.info(f"Filter output: {output_path}")

        return changes


def _apply_section_visibility(parsed: list, section_visibility: dict):
    """Comment out blocks belonging to hidden sections."""
    # Map gear UI keys to actual section_type values
    gear_map = {
        "gear_rare": "ut->rare",
        "gear_magic": "ut->magic",
    }
    # Resolve gear keys into their section_type equivalents
    resolved = dict(section_visibility)
    for ui_key, section_type in gear_map.items():
        if ui_key in resolved:
            resolved[section_type] = resolved.pop(ui_key)

    for kind, block in parsed:
        if kind != "block":
            continue
        if not block.section_type:
            continue

        # Check if this section is explicitly hidden
        visible = resolved.get(block.section_type, True)

        # Gear hide layers: check "gear_normal" key
        if block.section_type == "hidelayer" and block.tier in GEAR_HIDELAYER_TIERS:
            visible = resolved.get("gear_normal", True)

        if not visible and not block.is_commented:
            block.header_line = "#" + block.header_line
            block.body_lines = ["#" + bl for bl in block.body_lines]
            block.is_commented = True


def _apply_gear_class_filters(parsed: list, gear_classes: dict):
    """
    Remove hidden item classes from gear blocks' Class conditions.

    gear_classes: { "gear_rare": { "Bows": false, ... }, "gear_magic": {...}, ... }
    If ALL classes in a block are hidden, the entire block is commented out.
    """
    for kind, block in parsed:
        if kind != "block":
            continue
        if block.is_commented:
            continue

        # Map block to gear key
        gear_key = None
        if block.section_type == "ut->rare":
            gear_key = "gear_rare"
        elif block.section_type == "ut->magic":
            gear_key = "gear_magic"
        elif block.section_type == "hidelayer" and block.tier in GEAR_HIDELAYER_TIERS:
            gear_key = "gear_normal"

        if not gear_key or gear_key not in gear_classes:
            continue

        class_prefs = gear_classes[gear_key]
        if not class_prefs:
            continue

        # Find the Class line in body
        for i, line in enumerate(block.body_lines):
            stripped = line.lstrip('#').strip()
            if not stripped.startswith('Class'):
                continue

            current_classes = re.findall(r'"([^"]+)"', stripped)
            if not current_classes:
                break

            # Filter out explicitly hidden classes (default is visible)
            visible_classes = [c for c in current_classes
                               if class_prefs.get(c, True) is not False]

            if not visible_classes:
                # All classes hidden — comment out entire block
                block.header_line = "#" + block.header_line
                block.body_lines = ["#" + bl for bl in block.body_lines]
                block.is_commented = True
            elif len(visible_classes) < len(current_classes):
                # Some hidden — rebuild Class line with only visible ones
                prefix_match = re.match(r'(\s*Class\s*==?\s*)', stripped)
                prefix = prefix_match.group(1) if prefix_match else "\tClass == "
                quoted = " ".join(f'"{c}"' for c in visible_classes)
                block.body_lines[i] = f"{prefix}{quoted}"
            break


def _log_changes(changes: dict, dry_run: bool = False):
    """Log a summary of tier changes."""
    total = sum(len(v) for v in changes.values())
    prefix = "[DRY RUN] " if dry_run else ""
    logger.info(f"{prefix}Filter updated: {total} items re-tiered")

    for section, items in changes.items():
        if len(items) <= 5:
            detail = ", ".join(f"{name} {old}->{new}" for name, old, new in items)
        else:
            detail = f"{len(items)} items moved"
        logger.info(f"  {section}: {detail}")
