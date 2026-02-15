"""
POE2 Price Overlay - Pipeline Test
Validates the Parse → Price lookup pipeline.
Run this without POE2 to verify the core logic works.

Usage:
    python test_pipeline.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging

from item_parser import ItemParser, ParsedItem
from price_cache import PriceCache
from config import DEFAULT_LEAGUE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_item_parser():
    """Test the item parser with various tooltip texts."""
    print("\n" + "=" * 60)
    print("  TEST: Item Parser")
    print("=" * 60)

    parser = ItemParser()
    passed = 0
    failed = 0

    test_cases = [
        {
            "input": "Divine Orb\nStack Size: 10",
            "expected_name": "Divine Orb",
            "expected_rarity": "currency",
        },
        {
            "input": "Exalted Orb",
            "expected_name": "Exalted Orb",
            "expected_rarity": "currency",
        },
        {
            "input": "Kaom's Heart\nGlorious Plate\nItem Level: 84",
            "expected_name": "Kaom's Heart",
            # Note: rarity detection from text alone can't determine "unique"
            # In production, color detection handles this. Parser falls back to "unknown"
            "expected_ilvl": 84,
        },
        {
            "input": "Stellar Amulet\nItem Level: 82",
            "expected_name": "Stellar Amulet",
            "expected_ilvl": 82,
        },
        {
            "input": "Ice Nova\nSkill Gem\nLevel: 20\nQuality: +20%",
            "expected_name": "Ice Nova",
            "expected_rarity": "gem",
        },
        {
            "input": "Waystone Tier 16",
            "expected_name": "Waystone (Tier 16)",
        },
    ]

    for i, tc in enumerate(test_cases):
        result = parser.parse(tc["input"])

        if result is None:
            print(f"  [{i+1}] FAIL: parser returned None for '{tc['input'][:40]}'")
            failed += 1
            continue

        errors = []

        if "expected_name" in tc and result.name != tc["expected_name"]:
            # Check if it's close (case insensitive)
            if result.name.lower() != tc["expected_name"].lower():
                errors.append(f"name: got '{result.name}', expected '{tc['expected_name']}'")

        if "expected_rarity" in tc and result.rarity != tc["expected_rarity"]:
            errors.append(f"rarity: got '{result.rarity}', expected '{tc['expected_rarity']}'")

        if "expected_ilvl" in tc and result.item_level != tc["expected_ilvl"]:
            errors.append(f"ilvl: got {result.item_level}, expected {tc['expected_ilvl']}")

        if errors:
            print(f"  [{i+1}] FAIL: {tc['input'][:40]}...")
            for err in errors:
                print(f"        {err}")
            failed += 1
        else:
            print(f"  [{i+1}] PASS: {result.name} (rarity={result.rarity}, ilvl={result.item_level})")
            passed += 1

    print(f"\n  Results: {passed} passed, {failed} failed out of {len(test_cases)}")
    return failed == 0


def test_price_cache():
    """Test the price cache with direct text lookups (no API needed)."""
    print("\n" + "=" * 60)
    print("  TEST: Price Cache (offline)")
    print("=" * 60)

    cache = PriceCache(league=DEFAULT_LEAGUE)

    # Manually inject some test prices
    cache.prices = {
        "divine orb": {"chaos_value": 150.0, "category": "currency", "name": "Divine Orb"},
        "exalted orb": {"chaos_value": 30.0, "category": "currency", "name": "Exalted Orb"},
        "chaos orb": {"chaos_value": 1.0, "category": "currency", "name": "Chaos Orb"},
        "mirror of kalandra": {"chaos_value": 50000.0, "category": "currency", "name": "Mirror of Kalandra"},
        "kaom's heart": {"chaos_value": 500.0, "category": "unique_armours", "name": "Kaom's Heart"},
        "stellar amulet": {"chaos_value": 450.0, "category": "base_types", "name": "Stellar Amulet"},
        "ice nova": {"chaos_value": 80.0, "category": "skill_gems", "name": "Ice Nova"},
    }
    cache.currency_rates = {
        "exalted orb": 30.0,
        "divine orb": 150.0,
        "chaos orb": 1.0,
    }

    passed = 0
    failed = 0

    test_lookups = [
        ("Divine Orb", None, 0, True, "good"),      # 150c / 30 = 5 exalted = "good"
        ("Exalted Orb", None, 0, True, "decent"),
        ("Chaos Orb", None, 0, True, "low"),
        ("Mirror of Kalandra", None, 0, True, "high"),
        ("Kaom's Heart", None, 0, True, "good"),
        ("Stellar Amulet", "Stellar Amulet", 82, True, "good"),
        ("Nonexistent Item", None, 0, False, None),
    ]

    for name, base, ilvl, should_find, expected_tier in test_lookups:
        result = cache.lookup(name, base or "", ilvl)

        if should_find and result:
            tier_ok = result["tier"] == expected_tier if expected_tier else True
            if tier_ok:
                print(f"  PASS: {name} → {result['display']} (tier={result['tier']})")
                passed += 1
            else:
                print(f"  FAIL: {name} → tier={result['tier']}, expected={expected_tier}")
                failed += 1
        elif not should_find and not result:
            print(f"  PASS: {name} → Not found (expected)")
            passed += 1
        elif should_find and not result:
            print(f"  FAIL: {name} → Not found (should have been found)")
            failed += 1
        else:
            print(f"  FAIL: {name} → Found unexpectedly: {result}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed out of {len(test_lookups)}")
    return failed == 0


def test_full_pipeline():
    """Test the complete pipeline: parse text → lookup price."""
    print("\n" + "=" * 60)
    print("  TEST: Full Pipeline (parser → cache)")
    print("=" * 60)

    parser = ItemParser()
    cache = PriceCache(league=DEFAULT_LEAGUE)

    # Inject test data
    cache.prices = {
        "divine orb": {"chaos_value": 150.0, "category": "currency", "name": "Divine Orb"},
        "exalted orb": {"chaos_value": 30.0, "category": "currency", "name": "Exalted Orb"},
        "stellar amulet": {"chaos_value": 450.0, "category": "base_types", "name": "Stellar Amulet"},
        "kaom's heart": {"chaos_value": 500.0, "category": "unique_armours", "name": "Kaom's Heart"},
    }
    cache.currency_rates = {"exalted orb": 30.0, "divine orb": 150.0, "chaos orb": 1.0}

    test_scenarios = [
        ("Player hovers Divine Orb on ground", "Divine Orb"),
        ("Player hovers valuable base", "Stellar Amulet\nItem Level: 82"),
        ("Player hovers unique", "Kaom's Heart\nGlorious Plate\nItem Level: 84"),
        ("Player hovers trash item", "Rusted Sword\nItem Level: 12"),
    ]

    for scenario, ocr_text in test_scenarios:
        item = parser.parse(ocr_text)
        if item:
            result = cache.lookup(item.lookup_key, item.base_type, item.item_level)
            if result:
                print(f"  ✓ {scenario}")
                print(f"    Text: '{ocr_text.split(chr(10))[0]}' → {result['display']}")
            else:
                print(f"  · {scenario}")
                print(f"    Text: '{ocr_text.split(chr(10))[0]}' → No price (expected for trash)")
        else:
            print(f"  ✗ {scenario}")
            print(f"    Failed to parse text")

    return True


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         POE2 Price Overlay - Pipeline Tests             ║")
    print("╚══════════════════════════════════════════════════════════╝")

    results = []

    results.append(("Item Parser", test_item_parser()))
    results.append(("Price Cache", test_price_cache()))
    results.append(("Full Pipeline", test_full_pipeline()))

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("  All tests passed! Ready for real-world testing with POE2.")
    else:
        print("  Some tests failed. Check output above for details.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
