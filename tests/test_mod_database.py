"""Tests for mod_database.py — migrated from __main__ + new tests.

All 40 existing test cases migrated to pytest parametrize,
plus new SOMV, chaos res weight, and grade boundary tests.
"""

import pytest

from item_parser import ParsedItem
from mod_parser import ParsedMod
from mod_database import ModDatabase, Grade, ModScore, _dps_factor, _defense_factor, _skill_level_factor
from tests.conftest import make_item, make_mod


# ── Parametrized grading tests ───────────────────────────
# Each tuple: (test_id, description, expected_grades, item_kwargs, mods_spec)
# mods_spec is a list of (stat_name, value, raw_text)

GRADE_CASES = [
    # ── S-tier: God-rolled items ──────────────────
    ("S1", "God STR Gloves (T1 life + T1 atk spd + T1 crit multi + fire res)",
     {"S"},
     dict(name="Apocalypse Grip", base="Plated Gauntlets", cls="Gloves"),
     [("life", 145, "+145 to maximum Life"),
      ("atk_spd", 24, "24% increased Attack Speed"),
      ("crit_multi", 40, "40% increased Critical Damage Bonus"),
      ("fire_res", 25, "+25% to Fire Resistance")]),

    ("S2", "God Caster Amulet (T1 spell dmg + T1 cast spd + T1 crit chance + spirit)",
     {"S", "A"},
     dict(name="Damnation Pendant", base="Gold Amulet", cls="Amulets"),
     [("spell_dmg", 90, "90% increased Spell Damage"),
      ("cast_spd", 24, "24% increased Cast Speed"),
      ("crit_chance", 35, "35% increased Critical Hit Chance"),
      ("spirit", 30, "+30 to Spirit")]),

    ("S3", "God Phys Bow no +skill levels (penalized by skill_level_factor)",
     {"B", "A"},
     dict(name="Armageddon Thirst", base="Recurve Bow", cls="Bows"),
     [("%phys", 170, "170% increased Physical Damage"),
      ("phys_dmg", 50, "Adds 30 to 50 Physical Damage"),
      ("crit_chance", 35, "35% increased Critical Hit Chance"),
      ("atk_spd", 24, "24% increased Attack Speed")]),

    # ── A-tier: Strong items ──────────────────────
    ("A1", "Good Boots (T1 life + T2 all res + cold res)",
     {"A", "B"},
     dict(name="Storm Trail", base="Wrapped Boots", cls="Boots"),
     [("life", 130, "+130 to maximum Life"),
      ("all_res", 12, "+12% to all Elemental Resistances"),
      ("cold_res", 35, "+35% to Cold Resistance")]),

    ("A2", "Good ES Helmet (T1 flat ES + T1 %ES + int)",
     {"S", "A"},
     dict(name="Entropy Crown", base="Arcane Crown", cls="Helmets"),
     [("es", 100, "+100 to maximum Energy Shield"),
      ("%es", 80, "80% increased maximum Energy Shield"),
      ("int", 40, "+40 to Intelligence")]),

    ("A3", "Good Ring (T1 crit multi + T2 life + lightning res)",
     {"S", "A"},
     dict(name="Horror Turn", base="Ruby Ring", cls="Rings"),
     [("crit_multi", 38, "38% increased Critical Damage Bonus"),
      ("life", 70, "+70 to maximum Life"),
      ("lightning_res", 30, "+30% to Lightning Resistance")]),

    ("A4", "Good Two-Hand Sword no +skill levels (penalized by skill_level_factor)",
     {"C", "B"},
     dict(name="Dread Edge", base="Broad Sword", cls="Two Hand Swords"),
     [("%phys", 160, "160% increased Physical Damage"),
      ("phys_dmg", 40, "Adds 25 to 40 Physical Damage"),
      ("atk_spd", 14, "14% increased Attack Speed")]),

    ("A5", "Good Sceptre no +skill levels (penalized by skill_level_factor)",
     {"C", "B"},
     dict(name="Vortex Sceptre", base="Blood Sceptre", cls="Sceptres"),
     [("spell_dmg", 85, "85% increased Spell Damage"),
      ("ele_dmg_atk", 35, "35% increased Elemental Damage with Attacks"),
      ("cast_spd", 18, "18% increased Cast Speed")]),

    # ── B-tier: Decent items ──────────────────────
    ("B1", "Decent Body Armour (T3 life + T4 armour + fire res)",
     {"B", "C"},
     dict(name="Havoc Shell", base="Full Plate", cls="Body Armours"),
     [("life", 70, "+70 to maximum Life"),
      ("armour", 200, "+200 to Armour"),
      ("fire_res", 28, "+28% to Fire Resistance")]),

    ("B2", "Decent Belt (T2 life + T4 fire res + T5 cold res)",
     {"B", "C"},
     dict(name="Storm Cord", base="Leather Belt", cls="Belts"),
     [("life", 80, "+80 to maximum Life"),
      ("fire_res", 22, "+22% to Fire Resistance"),
      ("cold_res", 18, "+18% to Cold Resistance")]),

    ("B3", "Decent Shield (T2 %armour + T3 life + lightning res)",
     {"B", "C"},
     dict(name="Rampart Tower", base="Tower Shield", cls="Shields"),
     [("%armour", 80, "80% increased Armour"),
      ("life", 65, "+65 to maximum Life"),
      ("lightning_res", 25, "+25% to Lightning Resistance")]),

    ("B4", "Decent Wand (T3 spell dmg + T3 cast spd + mana)",
     {"B", "C", "A"},
     dict(name="Ghoul Song", base="Bone Wand", cls="Wands"),
     [("spell_dmg", 55, "55% increased Spell Damage"),
      ("cast_spd", 14, "14% increased Cast Speed"),
      ("mana", 50, "+50 to maximum Mana")]),

    ("B5", "Decent Crossbow (T2 %phys + T3 crit + cold dmg)",
     {"B", "C", "A"},
     dict(name="Storm Bane", base="Gemini Crossbow", cls="Crossbows"),
     [("%phys", 120, "120% increased Physical Damage"),
      ("crit_chance", 20, "20% increased Critical Hit Chance"),
      ("cold_dmg", 30, "Adds 15 to 30 Cold Damage")]),

    # ── C-tier: Mediocre items ────────────────────
    ("C1", "Mediocre Gloves (T5 life + low atk spd + T5 res)",
     {"C"},
     dict(name="Kraken Mitts", base="Ringmail Gauntlets", cls="Gloves"),
     [("life", 35, "+35 to maximum Life"),
      ("atk_spd", 7, "7% increased Attack Speed"),
      ("fire_res", 15, "+15% to Fire Resistance")]),

    ("C2", "Mediocre Amulet (T4 int + T5 mana + low res)",
     {"C", "JUNK"},
     dict(name="Skull Choker", base="Jade Amulet", cls="Amulets"),
     [("int", 18, "+18 to Intelligence"),
      ("mana", 30, "+30 to maximum Mana"),
      ("cold_res", 12, "+12% to Cold Resistance")]),

    ("C3", "Mediocre Helmet (low life + low armour)",
     {"C", "JUNK"},
     dict(name="Doom Cage", base="Iron Hat", cls="Helmets"),
     [("life", 25, "+25 to maximum Life"),
      ("armour", 50, "+50 to Armour")]),

    ("C4", "Mediocre Ring (low str + low fire res)",
     {"C", "JUNK"},
     dict(name="Grim Band", base="Iron Ring", cls="Rings"),
     [("str", 12, "+12 to Strength"),
      ("fire_res", 14, "+14% to Fire Resistance")]),

    # ── JUNK: Trash items ─────────────────────────
    ("J1", "All filler resistances (no key mods)",
     {"JUNK"},
     dict(name="Ash Wrap", base="Chain Gloves", cls="Gloves"),
     [("fire_res", 18, "+18% to Fire Resistance"),
      ("cold_res", 15, "+15% to Cold Resistance"),
      ("lightning_res", 12, "+12% to Lightning Resistance")]),

    ("J2", "Pure attributes only",
     {"JUNK"},
     dict(name="Drake Ring", base="Iron Ring", cls="Rings"),
     [("str", 10, "+10 to Strength"),
      ("dex", 12, "+12 to Dexterity"),
      ("int", 8, "+8 to Intelligence")]),

    ("J3", "Near-zero weight mods (thorns + stun)",
     {"JUNK", "C"},
     dict(name="Pain Carapace", base="Full Plate", cls="Body Armours"),
     [("thorns", 15, "10 to 15 Physical Thorns damage"),
      ("stun_thresh", 50, "+50 to Stun Threshold")]),

    ("J4", "Single low-tier filler mod",
     {"JUNK"},
     dict(name="Gale Coif", base="Iron Hat", cls="Helmets"),
     [("cold_res", 10, "+10% to Cold Resistance")]),

    # ── SOMV: Roll Quality Tests ──────────────────
    ("V1", "Tri-res ring — PERFECT rolls (45/43/40)",
     {"C", "B"},
     dict(name="Godly Band", base="Ruby Ring", cls="Rings"),
     [("fire_res", 45, "+45% to Fire Resistance"),
      ("cold_res", 43, "+43% to Cold Resistance"),
      ("lightning_res", 40, "+40% to Lightning Resistance")]),

    ("V2", "Tri-res ring — BOTTOM rolls (12/10/11)",
     {"JUNK"},
     dict(name="Trash Band", base="Ruby Ring", cls="Rings"),
     [("fire_res", 12, "+12% to Fire Resistance"),
      ("cold_res", 10, "+10% to Cold Resistance"),
      ("lightning_res", 11, "+11% to Lightning Resistance")]),

    ("V3", "Life + resist ring — HIGH rolls",
     {"C"},
     dict(name="Inferno Loop", base="Ruby Ring", cls="Rings"),
     [("life", 78, "+78 to maximum Life"),
      ("fire_res", 45, "+45% to Fire Resistance"),
      ("cold_res", 44, "+44% to Cold Resistance")]),

    ("V4", "Life + resist ring — LOW rolls",
     {"C", "JUNK"},
     dict(name="Dim Loop", base="Ruby Ring", cls="Rings"),
     [("life", 25, "+25 to maximum Life"),
      ("fire_res", 14, "+14% to Fire Resistance"),
      ("cold_res", 12, "+12% to Cold Resistance")]),

    ("V5", "Gloves T1 mods perfect rolls",
     {"S"},
     dict(name="Divine Grip", base="Plated Gauntlets", cls="Gloves"),
     [("life", 145, "+145 to maximum Life"),
      ("atk_spd", 24, "24% increased Attack Speed"),
      ("crit_multi", 42, "42% increased Critical Damage Bonus"),
      ("fire_res", 45, "+45% to Fire Resistance")]),

    ("V6", "Gloves same tiers bottom rolls",
     {"B", "A", "C"},
     dict(name="Worn Grip", base="Plated Gauntlets", cls="Gloves"),
     [("life", 100, "+100 to maximum Life"),
      ("atk_spd", 13, "13% increased Attack Speed"),
      ("crit_multi", 25, "25% increased Critical Damage Bonus"),
      ("fire_res", 20, "+20% to Fire Resistance")]),

    # ── Edge Cases ────────────────────────────────
    ("E1", "Unknown item class (falls back to any ladder match)",
     {"S", "A", "B", "C"},
     dict(name="Mystery Box", base="Unknown Base", cls="FooBarBaz"),
     [("life", 100, "+100 to maximum Life"),
      ("crit_multi", 30, "30% increased Critical Damage Bonus")]),

    ("E3", "Empty mod list",
     {"JUNK"},
     dict(name="Bare Plate", base="Full Plate", cls="Body Armours"),
     []),

    ("E4", "Single god mod only (T1 crit multi — capped at C by mod count)",
     {"C"},
     dict(name="Havoc Loop", base="Ruby Ring", cls="Rings"),
     [("crit_multi", 42, "42% increased Critical Damage Bonus")]),

    ("E5", "Value exceeding T1 max (sanctified/corrupted body armour)",
     {"S", "A", "B"},
     dict(name="Transcendent Mail", base="Full Plate", cls="Body Armours"),
     [("life", 250, "+250 to maximum Life"),
      ("%life", 15, "15% increased maximum Life"),
      ("fire_res", 45, "+45% to Fire Resistance"),
      ("cold_res", 42, "+42% to Cold Resistance")]),

    ("E6", "Body armour mid-tier life (+145 is T5 on body, not T1)",
     {"C", "B"},
     dict(name="Fortress Plate", base="Full Plate", cls="Body Armours"),
     [("life", 145, "+145 to maximum Life"),
      ("all_res", 14, "+14% to all Elemental Resistances"),
      ("str", 40, "+40 to Strength")]),

    ("E7", "Dagger with crit + spell + attack hybrid (no +skill levels)",
     {"C", "B"},
     dict(name="Soul Fang", base="Stiletto", cls="Daggers"),
     [("spell_dmg", 70, "70% increased Spell Damage"),
      ("crit_chance", 30, "30% increased Critical Hit Chance"),
      ("atk_spd", 20, "20% increased Attack Speed")]),

    ("E8", "Focus with ES + spirit + cast speed",
     {"A", "S", "B", "C"},
     dict(name="Omen Lens", base="Bone Focus", cls="Foci"),
     [("es", 80, "+80 to maximum Energy Shield"),
      ("spirit", 25, "+25 to Spirit"),
      ("cast_spd", 20, "20% increased Cast Speed")]),

    # ── DPS/Defense Factor Tests ─────────────────
    ("D1", "Low DPS bow (100 dps, ilvl 80) — crushed to JUNK",
     {"JUNK", "C"},
     dict(name="Trash Bow", base="Recurve Bow", cls="Bows", ilvl=80, total_dps=100),
     [("%phys", 170, "170% increased Physical Damage"),
      ("phys_dmg", 50, "Adds 30 to 50 Physical Damage"),
      ("crit_chance", 35, "35% increased Critical Hit Chance"),
      ("atk_spd", 24, "24% increased Attack Speed")]),

    ("D2", "Good DPS bow (400 dps, ilvl 80) — no +skill levels penalty",
     {"B", "A"},
     dict(name="Storm Thirst", base="Recurve Bow", cls="Bows", ilvl=80, total_dps=400),
     [("%phys", 170, "170% increased Physical Damage"),
      ("phys_dmg", 50, "Adds 30 to 50 Physical Damage"),
      ("crit_chance", 35, "35% increased Critical Hit Chance"),
      ("atk_spd", 24, "24% increased Attack Speed")]),

    ("D3", "Low defense body armour (150 total) — penalized",
     {"C", "JUNK"},
     dict(name="Weak Plate", base="Full Plate", cls="Body Armours", total_defense=150),
     [("life", 70, "+70 to maximum Life"),
      ("armour", 200, "+200 to Armour"),
      ("fire_res", 28, "+28% to Fire Resistance")]),

    ("D4", "Wand — DPS factor = 1.0 (excluded from DPS scoring)",
     {"B", "C", "A"},
     dict(name="Caster Wand", base="Bone Wand", cls="Wands", total_dps=50),
     [("spell_dmg", 55, "55% increased Spell Damage"),
      ("cast_spd", 14, "14% increased Cast Speed"),
      ("mana", 50, "+50 to maximum Mana")]),

    ("D5", "Ring — both factors = 1.0 (excluded from both)",
     {"S", "A"},
     dict(name="Plain Ring", base="Ruby Ring", cls="Rings", total_dps=0, total_defense=0),
     [("crit_multi", 38, "38% increased Critical Damage Bonus"),
      ("life", 70, "+70 to maximum Life"),
      ("lightning_res", 30, "+30% to Lightning Resistance")]),
]


@pytest.mark.parametrize(
    "test_id,desc,expected_grades,item_kwargs,mods_spec",
    GRADE_CASES,
    ids=[c[0] for c in GRADE_CASES],
)
def test_grade(mod_database, stat_ids, test_id, desc, expected_grades, item_kwargs, mods_spec):
    """Parametrized: score_item returns grade within expected set."""
    test_item = make_item(**item_kwargs)
    test_mods = []
    for spec in mods_spec:
        name, value, raw = spec
        test_mods.append(make_mod(stat_ids, name, value, raw))

    score = mod_database.score_item(test_item, test_mods)
    grade_str = score.grade.value
    assert grade_str in expected_grades, (
        f"[{test_id}] {desc}: got {grade_str}, expected one of {expected_grades} "
        f"(score={score.normalized_score:.3f}, somv={score.somv_factor:.3f}, "
        f"dps_f={score.dps_factor:.3f}, def_f={score.defense_factor:.3f})"
    )


# ── E2: Fake stat ID edge case (needs raw ParsedMod) ────

def test_grade_fake_stat_id(mod_database, stat_ids):
    """Mod not in bridge (fake stat ID + life)."""
    test_item = make_item(name="Enigma Grip", base="Chain Gloves", cls="Gloves")
    test_mods = [
        ParsedMod(raw_text="50% increased Foo Power", stat_id="explicit.stat_fake_123",
                  value=50.0, mod_type="explicit"),
        make_mod(stat_ids, "life", 60, "+60 to maximum Life"),
    ]
    score = mod_database.score_item(test_item, test_mods)
    assert score.grade.value in {"B", "C"}


# ── SOMV factor range ────────────────────────────────────

def test_somv_factor_range(mod_database, stat_ids):
    """SOMV factor is always in [0.90, 1.10]."""
    # Test with various roll qualities
    for fire_val in [10, 20, 30, 40, 46]:
        test_item = make_item(name="Test Ring", base="Ruby Ring", cls="Rings")
        test_mods = [make_mod(stat_ids, "fire_res", fire_val,
                              f"+{fire_val}% to Fire Resistance")]
        score = mod_database.score_item(test_item, test_mods)
        assert 0.90 <= score.somv_factor <= 1.10, (
            f"SOMV {score.somv_factor} out of range for fire_res={fire_val}"
        )


def test_somv_perfect_beats_bottom(mod_database, stat_ids):
    """Perfect rolls score higher than bottom rolls due to SOMV."""
    perfect_item = make_item(name="Perfect", base="Ruby Ring", cls="Rings")
    perfect_mods = [
        make_mod(stat_ids, "fire_res", 45, "+45% to Fire Resistance"),
        make_mod(stat_ids, "cold_res", 43, "+43% to Cold Resistance"),
    ]

    bottom_item = make_item(name="Bottom", base="Ruby Ring", cls="Rings")
    bottom_mods = [
        make_mod(stat_ids, "fire_res", 12, "+12% to Fire Resistance"),
        make_mod(stat_ids, "cold_res", 10, "+10% to Cold Resistance"),
    ]

    perfect_score = mod_database.score_item(perfect_item, perfect_mods)
    bottom_score = mod_database.score_item(bottom_item, bottom_mods)

    assert perfect_score.normalized_score > bottom_score.normalized_score
    assert perfect_score.somv_factor > bottom_score.somv_factor


# ── Chaos resistance weight ──────────────────────────────

def test_chaos_res_weight(mod_database, stat_ids):
    """Chaos res weight > elemental res weight."""
    if "chaos_res" not in stat_ids or "fire_res" not in stat_ids:
        pytest.skip("Missing stat IDs for chaos/fire res")

    # Score items with identical values but different resist types
    chaos_item = make_item(name="Chaos Ring", base="Ruby Ring", cls="Rings")
    chaos_mods = [make_mod(stat_ids, "chaos_res", 30, "+30% to Chaos Resistance")]

    fire_item = make_item(name="Fire Ring", base="Ruby Ring", cls="Rings")
    fire_mods = [make_mod(stat_ids, "fire_res", 30, "+30% to Fire Resistance")]

    chaos_score = mod_database.score_item(chaos_item, chaos_mods)
    fire_score = mod_database.score_item(fire_item, fire_mods)

    # Chaos res has weight 0.5, fire res has weight 0.3
    chaos_ws = sum(ms.weight for ms in chaos_score.mod_scores)
    fire_ws = sum(ms.weight for ms in fire_score.mod_scores)
    assert chaos_ws > fire_ws, (
        f"Chaos res weight ({chaos_ws}) should be > fire res weight ({fire_ws})"
    )


# ── _assign_grade boundaries ─────────────────────────────

def test_assign_grade_junk_no_key_mods():
    """No key mods + low score = JUNK."""
    grade = ModDatabase._assign_grade(0.2, key_mods=[], high_tier_key=[],
                                      total_mods=3)
    assert grade == Grade.JUNK


def test_assign_grade_c_with_high_score_no_keys():
    """No key mods but very high score + enough mods = C (not JUNK)."""
    grade = ModDatabase._assign_grade(0.70, key_mods=[], high_tier_key=[],
                                      total_mods=3)
    assert grade == Grade.C


def test_assign_grade_s_requires_2_high_tier():
    """S requires 2+ T1/T2 key mods."""
    # Mock key_mods and high_tier_key as simple lists
    from unittest.mock import MagicMock
    key1 = MagicMock()
    key2 = MagicMock()
    key3 = MagicMock()

    # 1 high tier key → not S
    grade = ModDatabase._assign_grade(0.80, key_mods=[key1, key2, key3],
                                      high_tier_key=[key1],
                                      total_mods=4)
    assert grade != Grade.S  # Should be A

    # 2 high tier keys + high score + enough mods → S
    grade = ModDatabase._assign_grade(0.80, key_mods=[key1, key2, key3],
                                      high_tier_key=[key1, key2],
                                      total_mods=4)
    assert grade == Grade.S


def test_assign_grade_special_affix_bonus():
    """Special affixes (fractured/desecrated) lower thresholds."""
    from unittest.mock import MagicMock
    key1 = MagicMock()
    key2 = MagicMock()
    key3 = MagicMock()

    # Without special affixes, 0.65 + 2 high tier + 3 keys ≠ S (needs >=0.75)
    grade_no_special = ModDatabase._assign_grade(
        0.65, key_mods=[key1, key2, key3],
        high_tier_key=[key1, key2], total_mods=4,
        special_affix_count=0)

    # With 1 special affix, threshold drops by 0.10 → 0.65 >= 0.65 → S possible
    grade_with_special = ModDatabase._assign_grade(
        0.65, key_mods=[key1, key2, key3],
        high_tier_key=[key1, key2], total_mods=4,
        special_affix_count=1)

    # The special affix version should be equal or higher grade
    grade_order = {Grade.JUNK: 0, Grade.C: 1, Grade.B: 2, Grade.A: 3, Grade.S: 4}
    assert grade_order[grade_with_special] >= grade_order[grade_no_special]


# ── DPS/Defense factor functions ─────────────────────────

def test_dps_factor_non_weapon():
    """Non-weapon classes return 1.0."""
    assert _dps_factor(200, "Rings", 80) == 1.0
    assert _dps_factor(200, "Body Armours", 80) == 1.0
    assert _dps_factor(200, "Wands", 80) == 1.0


def test_dps_factor_zero_dps():
    """Zero DPS returns 1.0."""
    assert _dps_factor(0, "Bows", 80) == 1.0
    assert _dps_factor(0.0, "Bows", 80) == 1.0


def test_dps_factor_low_penalizes():
    """Low DPS on attack weapon penalizes heavily."""
    f = _dps_factor(50, "Bows", 80)
    assert f < 0.5, f"Low DPS bow should be penalized, got {f}"


def test_dps_factor_high_rewards():
    """High DPS on attack weapon gets factor >= 1.0."""
    f = _dps_factor(600, "Bows", 80)
    assert f >= 1.0, f"High DPS bow should be rewarded, got {f}"


def test_defense_factor_non_armor():
    """Non-armor classes return 1.0."""
    assert _defense_factor(500, "Rings", ) == 1.0
    assert _defense_factor(500, "Bows") == 1.0
    assert _defense_factor(500, "Wands") == 1.0


def test_defense_factor_zero():
    """Zero defense returns 1.0."""
    assert _defense_factor(0, "Body Armours") == 1.0


def test_defense_factor_low_penalizes():
    """Low defense penalizes."""
    f = _defense_factor(100, "Body Armours")
    assert f < 1.0, f"Low defense body should be penalized, got {f}"


def test_defense_factor_high():
    """High defense gets factor >= 1.0."""
    f = _defense_factor(1000, "Body Armours")
    assert f >= 1.0, f"High defense body should get factor >= 1.0, got {f}"


# ── Skill level factor tests ─────────────────────────

def _make_mod_score(mod_group="SomeGroup"):
    """Minimal ModScore for skill level factor testing."""
    return ModScore(
        raw_text="", stat_id="", value=0, mod_group=mod_group,
        generation_type="prefix", tier=None, tier_label="",
        percentile=0.5, weight=1.0, weighted_score=0.5, is_key_mod=True,
    )


def test_skill_level_factor_non_weapon():
    """Non-weapon classes always return 1.0."""
    mods = [_make_mod_score("PhysicalDamage")]
    assert _skill_level_factor(mods, "Rings") == 1.0
    assert _skill_level_factor(mods, "Body Armours") == 1.0
    assert _skill_level_factor(mods, "Gloves") == 1.0
    assert _skill_level_factor(mods, "Boots") == 1.0


def test_skill_level_factor_weapon_without_skills():
    """Weapons without +skill level mods get 0.5 penalty."""
    mods = [_make_mod_score("PhysicalDamage"), _make_mod_score("AttackSpeed")]
    assert _skill_level_factor(mods, "Bows") == 0.5
    assert _skill_level_factor(mods, "Two Hand Swords") == 0.5
    assert _skill_level_factor(mods, "Wands") == 0.5
    assert _skill_level_factor(mods, "Sceptres") == 0.5
    assert _skill_level_factor(mods, "Staves") == 0.5
    assert _skill_level_factor(mods, "Daggers") == 0.5


def test_skill_level_factor_weapon_with_skills():
    """Weapons WITH +skill level mods return 1.0."""
    for group in ("AllSkillLevels", "AddedGemLevels", "AddedSkillLevels"):
        mods = [_make_mod_score("PhysicalDamage"), _make_mod_score(group)]
        assert _skill_level_factor(mods, "Bows") == 1.0, f"Group {group} should bypass penalty"
        assert _skill_level_factor(mods, "Wands") == 1.0, f"Group {group} should bypass penalty"
