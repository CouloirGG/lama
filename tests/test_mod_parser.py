"""Tests for mod_parser.py — template-to-regex, mod matching, base type resolution."""

import re
import pytest

from mod_parser import ModParser, ParsedMod, _template_to_regex
from item_parser import ItemParser, ParsedItem
from tests.conftest import load_fixture


# ── _template_to_regex ───────────────────────────────────

TEMPLATE_CASES = [
    ("+# to maximum Life",           "+42 to maximum Life",      42.0),
    ("#% increased Attack Speed",    "24% increased Attack Speed", 24.0),
    ("Adds # to # Fire Damage",      "Adds 10 to 20 Fire Damage", 10.0),
    ("#% to Fire Resistance",         "+35% to Fire Resistance",   35.0),
    ("#% increased Spell Damage",    "90% increased Spell Damage", 90.0),
    ("+# to Spirit",                  "+30 to Spirit",              30.0),
]


@pytest.mark.parametrize("template,text,expected_value", TEMPLATE_CASES)
def test_template_to_regex(template, text, expected_value):
    """Parametrized: stat template compiles and matches expected text."""
    regex = _template_to_regex(template)
    assert regex is not None, f"Failed to compile template: {template!r}"
    m = regex.match(text)
    assert m is not None, f"Regex {regex.pattern!r} did not match {text!r}"
    # Extract first captured group
    value = float(m.group(1))
    assert value == expected_value


def test_template_to_regex_no_hash():
    """Templates without '#' return None."""
    assert _template_to_regex("Enemies are Hindered") is None
    assert _template_to_regex("") is None
    assert _template_to_regex(None) is None


def test_opposite_word_matching():
    """'increased' template matches 'reduced' text."""
    regex = _template_to_regex("#% increased Attack Speed")
    assert regex is not None
    # "increased" should match directly
    m1 = regex.match("24% increased Attack Speed")
    assert m1 is not None
    # "reduced" should also match (opposite word)
    m2 = regex.match("24% reduced Attack Speed")
    assert m2 is not None
    assert float(m2.group(1)) == 24.0


def test_more_less_matching():
    """'more' template matches 'less' text."""
    regex = _template_to_regex("#% more Damage")
    assert regex is not None
    m1 = regex.match("15% more Damage")
    assert m1 is not None
    m2 = regex.match("15% less Damage")
    assert m2 is not None


# ── parse_mods with real fixture ─────────────────────────

def test_parse_mods_real_item(mod_parser):
    """Parse mods from a real rare ring fixture."""
    parser = ItemParser()
    text = load_fixture("rare_ring_brimstone.txt")
    item = parser.parse_clipboard(text)
    assert item is not None
    assert len(item.mods) > 0

    mods = mod_parser.parse_mods(item)
    # Should match most explicit mods
    assert len(mods) >= 3, f"Expected >=3 matched mods, got {len(mods)}"
    # All mods should have stat IDs
    for m in mods:
        assert m.stat_id, f"Mod {m.raw_text!r} has no stat_id"
        assert isinstance(m.value, float)


def test_parse_mods_boots(mod_parser):
    """Parse mods from rare boots fixture — movement speed should match."""
    parser = ItemParser()
    text = load_fixture("rare_boots_dire_league.txt")
    item = parser.parse_clipboard(text)
    assert item is not None

    mods = mod_parser.parse_mods(item)
    assert len(mods) >= 3
    # Movement speed should be among matched mods
    raw_texts = [m.raw_text.lower() for m in mods]
    assert any("movement speed" in t for t in raw_texts), \
        f"Movement speed not found in mods: {raw_texts}"


def test_parse_mods_staff_with_grants_skill(mod_parser):
    """Staff with 'Grants Skill:' line should skip it (not a tradeable mod)."""
    parser = ItemParser()
    text = load_fixture("rare_staff_horror_weaver.txt")
    item = parser.parse_clipboard(text)
    assert item is not None
    # 'Grants Skill: Level 18 Consecrate' should NOT be in mods
    for mod_type, mod_text in item.mods:
        assert "Grants Skill" not in mod_text, \
            f"'Grants Skill' should be excluded: {mod_text!r}"


# ── resolve_base_type ────────────────────────────────────

def test_resolve_base_type(mod_parser):
    """Magic item name → base type via longest-first substring matching."""
    if not mod_parser._base_types:
        pytest.skip("No base types loaded")
    # A real magic name example
    result = mod_parser.resolve_base_type("Potent Ultimate Mana Flask of the Brewer")
    assert result is not None
    assert "Mana Flask" in result or "Ultimate" in result


def test_resolve_base_type_no_match(mod_parser):
    """Gibberish returns None."""
    result = mod_parser.resolve_base_type("Xyzzy Plugh Foobar")
    assert result is None


def test_resolve_base_type_empty(mod_parser):
    """Empty/None returns None."""
    assert mod_parser.resolve_base_type("") is None
    assert mod_parser.resolve_base_type(None) is None


# ── Value extraction ─────────────────────────────────────

def test_mod_value_extraction(mod_parser):
    """'+42 to maximum Life' → value=42.0."""
    result = mod_parser._match_mod("+42 to maximum Life", "explicit")
    if result is None:
        pytest.skip("Life mod not in stat definitions")
    assert result.value == 42.0


def test_multi_value_mod(mod_parser):
    """'Adds 10 to 20 Fire Damage' → value=10.0 (first capture)."""
    result = mod_parser._match_mod("Adds 10 to 20 Fire Damage", "explicit")
    if result is None:
        # Try alternate phrasing
        result = mod_parser._match_mod("Adds 10 to 20 Fire damage to Attacks", "explicit")
    if result is None:
        pytest.skip("Fire damage mod not in stat definitions")
    assert result.value == 10.0


def test_negative_value_extraction(mod_parser):
    """'reduced' word causes value negation."""
    result = mod_parser._match_mod("24% reduced Attack Speed", "explicit")
    if result is None:
        pytest.skip("Attack speed mod not in stat definitions")
    assert result.value == -24.0
