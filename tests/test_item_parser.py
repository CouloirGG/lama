"""Tests for item_parser.py — clipboard parsing, combat stats, regex patterns."""

import pytest
from pathlib import Path

from item_parser import ItemParser, ParsedItem
from tests.conftest import load_fixture, FIXTURES_DIR


# ── Fixtures ─────────────────────────────────────────────

@pytest.fixture
def parser():
    return ItemParser()


# ── Parametrized clipboard parsing ───────────────────────

RARE_FIXTURES = [
    ("rare_ring_brimstone.txt",    "Brimstone Knot", "Sapphire Ring",       "Rings",        "rare",  79),
    ("rare_boots_dire_league.txt", "Dire League",    "Sandsworn Sandals",   "Boots",        "rare",  82),
    ("rare_gloves_rage_paw.txt",   "Rage Paw",       "Opulent Gloves",      "Gloves",       "rare",  79),
    ("rare_body_armour_ghoul.txt", "Ghoul Carapace", "Flowing Raiment",     "Body Armours", "rare",  82),
    ("rare_amulet_ghoul_noose.txt","Ghoul Noose",    "Lunar Amulet",        "Amulets",      "rare",  82),
    ("rare_focus_pain_emblem.txt", "Pain Emblem",    "Tasalian Focus",      "Foci",         "rare",  81),
    ("rare_staff_horror_weaver.txt","Horror Weaver",  "Sanctified Staff",    "Staves",       "rare",  82),
]


@pytest.mark.parametrize("filename,exp_name,exp_base,exp_class,exp_rarity,exp_ilvl", RARE_FIXTURES)
def test_parse_clipboard_rare(parser, filename, exp_name, exp_base, exp_class, exp_rarity, exp_ilvl):
    """Parametrized: parse rare item clipboard → correct name/base/class/rarity/ilvl."""
    text = load_fixture(filename)
    item = parser.parse_clipboard(text)
    assert item is not None, f"parse_clipboard returned None for {filename}"
    assert item.name == exp_name
    assert item.base_type == exp_base
    assert item.item_class == exp_class
    assert item.rarity == exp_rarity
    assert item.item_level == exp_ilvl


def test_parse_clipboard_extracts_mods(parser):
    """Rare item has correct mod count and mod types."""
    text = load_fixture("rare_ring_brimstone.txt")
    item = parser.parse_clipboard(text)
    assert item is not None
    assert len(item.mods) >= 5
    # Should have both explicit and special mod types
    mod_types = {mt for mt, _ in item.mods}
    assert "explicit" in mod_types
    # Brimstone Knot has an implicit and a desecrated mod
    assert "implicit" in mod_types
    assert "desecrated" in mod_types


def test_parse_clipboard_combat_stats(parser):
    """Defense stats extracted from body armour fixture."""
    text = load_fixture("rare_body_armour_ghoul.txt")
    item = parser.parse_clipboard(text)
    assert item is not None
    assert item.energy_shield == 882
    assert item.total_defense == 882  # ES-only body armour


def test_parse_clipboard_implicit_separation(parser):
    """Implicit mods separated from explicits."""
    text = load_fixture("rare_body_armour_ghoul.txt")
    item = parser.parse_clipboard(text)
    assert item is not None
    implicit_mods = [(mt, t) for mt, t in item.mods if mt == "implicit"]
    explicit_mods = [(mt, t) for mt, t in item.mods if mt == "explicit"]
    assert len(implicit_mods) >= 1
    assert len(explicit_mods) >= 3


def test_parse_clipboard_fractured_mods(parser):
    """Fractured and desecrated mods correctly identified."""
    text = load_fixture("rare_gloves_rage_paw.txt")
    item = parser.parse_clipboard(text)
    assert item is not None
    fractured = [(mt, t) for mt, t in item.mods if mt == "fractured"]
    desecrated = [(mt, t) for mt, t in item.mods if mt == "desecrated"]
    assert len(fractured) >= 1
    assert len(desecrated) >= 1
    assert item.corrupted is True


def test_parse_clipboard_sockets(parser):
    """'S S S' format → count 3."""
    text = load_fixture("rare_gloves_rage_paw.txt")
    item = parser.parse_clipboard(text)
    assert item is not None
    assert item.sockets == 3


def test_parse_clipboard_quality(parser):
    """Quality extracted correctly."""
    text = load_fixture("rare_boots_dire_league.txt")
    item = parser.parse_clipboard(text)
    assert item is not None
    assert item.quality == 20


def test_strip_quality_prefix():
    """'Superior Full Plate' → 'Full Plate'."""
    assert ItemParser._strip_quality_prefix("Superior Full Plate") == "Full Plate"
    assert ItemParser._strip_quality_prefix("Exceptional Gemini Crossbow") == "Gemini Crossbow"
    assert ItemParser._strip_quality_prefix("Masterful Bone Wand") == "Bone Wand"
    assert ItemParser._strip_quality_prefix("Ruby Ring") == "Ruby Ring"  # no prefix


def test_extract_item_level(parser):
    """Regex on various ilvl formats."""
    assert parser._extract_item_level("Item Level: 84") == 84
    assert parser._extract_item_level("Item Level:79") == 79
    assert parser._extract_item_level("item level 42") == 42
    assert parser._extract_item_level("no level here") == 0


def test_extract_sockets(parser):
    """Sockets extracted from various formats."""
    # Test via parse_clipboard with a multi-socket item
    text = load_fixture("rare_focus_pain_emblem.txt")
    item = parser.parse_clipboard(text)
    assert item is not None
    assert item.sockets == 2


def test_empty_and_garbage_input(parser):
    """None/empty string returns None."""
    assert parser.parse_clipboard(None) is None
    assert parser.parse_clipboard("") is None
    assert parser.parse_clipboard("x") is None
    assert parser.parse_clipboard("short") is None


def test_currency_detection(parser):
    """Currency items detected correctly."""
    text = load_fixture("currency_fracturing_orb.txt")
    item = parser.parse_clipboard(text)
    assert item is not None
    assert item.rarity == "currency"
    assert item.name == "Fracturing Orb"


def test_gem_detection(parser):
    """Gem items detected correctly."""
    text = load_fixture("gem_archmage.txt")
    item = parser.parse_clipboard(text)
    assert item is not None
    assert item.rarity == "gem"
    assert item.name == "Archmage"


def test_unique_detection(parser):
    """Unique items detected correctly."""
    text = load_fixture("unique_quiver_asphyxia.txt")
    item = parser.parse_clipboard(text)
    assert item is not None
    assert item.rarity == "unique"
    assert item.name == "Asphyxia's Wrath"
    assert item.base_type == "Broadhead Quiver"


def test_unique_corrupted_with_enchant(parser):
    """Unique with enchant, rune, mutated mods, and corrupted flag."""
    text = load_fixture("unique_helmet_vertex.txt")
    item = parser.parse_clipboard(text)
    assert item is not None
    assert item.rarity == "unique"
    assert item.name == "The Vertex"
    assert item.corrupted is True
    assert item.energy_shield == 85
    assert item.evasion == 200
    # Corrupted uniques now extract mods for roll-aware pricing
    assert len(item.mods) > 0
    mod_types = {t for t, _ in item.mods}
    assert "mutated" in mod_types
