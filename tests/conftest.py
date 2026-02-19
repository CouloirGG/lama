"""Shared fixtures for LAMA test suite."""

import sys
import os
import logging
from pathlib import Path

import pytest

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from item_parser import ItemParser, ParsedItem
from mod_parser import ModParser, ParsedMod
from mod_database import ModDatabase, Grade

logger = logging.getLogger(__name__)

# ── Fixtures directory ───────────────────────────────────

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ── Session-scoped heavy fixtures ────────────────────────

@pytest.fixture(scope="session")
def mod_parser():
    """Load ModParser once per session (uses disk cache)."""
    mp = ModParser()
    mp.load_stats()
    if not mp.loaded:
        pytest.skip("ModParser could not load stats (no cache/network)")
    return mp


@pytest.fixture(scope="session")
def mod_database(mod_parser):
    """Load ModDatabase once per session via mod_parser."""
    db = ModDatabase()
    ok = db.load(mod_parser)
    if not ok:
        pytest.skip("ModDatabase could not load (no cache/network)")
    return db


@pytest.fixture(scope="session")
def item_parser():
    """Create an ItemParser instance."""
    return ItemParser()


@pytest.fixture(scope="session")
def stat_ids(mod_parser):
    """Resolve short stat names to trade stat IDs.

    Returns a dict like {"life": "explicit.stat_3299347043", ...}.
    """
    _stat_patterns = {
        "life":          "# to maximum life",
        "%life":         "#% increased maximum life",
        "mana":          "# to maximum mana",
        "es":            "# to maximum energy shield",
        "%es":           "#% increased maximum energy shield",
        "armour":        "# to armour",
        "%armour":       "#% increased armour",
        "evasion":       "# to evasion rating",
        "%evasion":      "#% increased evasion rating",
        "spirit":        "# to spirit",
        "fire_res":      "fire resistance",
        "cold_res":      "cold resistance",
        "lightning_res": "lightning resistance",
        "chaos_res":     "chaos resistance",
        "all_res":       "all elemental resistances",
        "str":           "# to strength",
        "dex":           "# to dexterity",
        "int":           "# to intelligence",
        "all_attr":      "# to all attributes",
        "atk_spd":       "#% increased attack speed",
        "cast_spd":      "#% increased cast speed",
        "crit_chance":   "#% increased critical hit chance",
        "spell_crit":    "critical hit chance for spells",
        "crit_multi":    "#% increased critical damage bonus",
        "spell_dmg":     "#% increased spell damage",
        "phys_dmg":      "adds # to # physical damage",
        "%phys":         "#% increased physical damage",
        "fire_dmg":      "adds # to # fire damage",
        "cold_dmg":      "adds # to # cold damage",
        "lightning_dmg": "adds # to # lightning damage",
        "ele_dmg_atk":   "#% increased elemental damage with attacks",
        "atk_dmg":       "#% increased attack damage",
        "accuracy":      "# to accuracy rating",
        "proj_speed":    "#% increased projectile speed",
        "life_leech":    "leech #% of physical attack damage as life",
        "life_regen":    "# life regeneration per second",
        "mana_regen":    "#% increased mana regeneration rate",
        "item_rarity":   "#% increased rarity of items found",
        "thorns":        "# to # physical thorns damage",
        "stun_thresh":   "# to stun threshold",
        "life_recoup":   "#% of damage taken recouped as life",
        "block":         "#% increased block chance",
        "mana_cost":     "#% increased mana cost efficiency",
    }

    resolved = {}
    for name, pattern in _stat_patterns.items():
        found = None
        for s in mod_parser._stats:
            if s.text and pattern in s.text.lower() and s.type == "explicit":
                found = s
                break
        if not found:
            for s in mod_parser._stats:
                if s.text and pattern in s.text.lower():
                    found = s
                    break
        if found:
            resolved[name] = found.id
    return resolved


# ── Helper factories ─────────────────────────────────────

def make_item(name="Test Item", base="Test Base", cls="Rings",
              ilvl=80, rarity="rare", total_dps=0.0, total_defense=0,
              **kwargs):
    """Shorthand to create a ParsedItem for testing."""
    it = ParsedItem()
    it.name = name
    it.base_type = base
    it.item_class = cls
    it.rarity = rarity
    it.item_level = ilvl
    it.total_dps = total_dps
    it.total_defense = total_defense
    for k, v in kwargs.items():
        setattr(it, k, v)
    return it


def make_mod(stat_ids_dict, name, value, raw_text="", mod_type="explicit"):
    """Shorthand to create a ParsedMod from a stat short name."""
    sid = stat_ids_dict.get(name, f"unknown.{name}")
    return ParsedMod(
        raw_text=raw_text or name,
        stat_id=sid,
        value=value,
        mod_type=mod_type,
    )


# ── Clipboard fixture loader ────────────────────────────

@pytest.fixture
def clipboard_files():
    """Yield list of all .txt fixture files."""
    if not FIXTURES_DIR.exists():
        pytest.skip("No fixtures directory")
    files = sorted(FIXTURES_DIR.glob("*.txt"))
    if not files:
        pytest.skip("No fixture files found")
    return files


def load_fixture(filename):
    """Load a single fixture file by name."""
    path = FIXTURES_DIR / filename
    if not path.exists():
        pytest.skip(f"Fixture {filename} not found")
    return path.read_text(encoding="utf-8")
