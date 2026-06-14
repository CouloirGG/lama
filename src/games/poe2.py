"""
POE2 game configuration factory.

Creates a GameConfig populated with all Path of Exile 2 constants.
Imports from the existing config.py during transition to stay in sync.
"""

from pathlib import Path
from typing import Optional

from core.game_config import GameConfig


def create_poe2_config(
    league: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> GameConfig:
    """Create a GameConfig for Path of Exile 2.

    Args:
        league: Override league name. Defaults to config.DEFAULT_LEAGUE.
        cache_dir: Override cache directory. Defaults to config.CACHE_DIR.

    Returns:
        Fully populated GameConfig for POE2.
    """
    from config import (
        DEFAULT_LEAGUE,
        CACHE_DIR,
        REPOE_BASE_URL,
        REPOE_CACHE_DIR,
        REPOE_CACHE_TTL,
        DPS_ITEM_CLASSES,
        TWO_HAND_CLASSES,
        DEFENSE_ITEM_CLASSES,
        DPS_BRACKETS_2H,
        DPS_BRACKETS_1H,
        DEFENSE_THRESHOLDS,
        POE2SCOUT_BASE_URL,
        PRICE_REFRESH_INTERVAL,
        GRADE_TIER_MAP,
    )

    _league = league or DEFAULT_LEAGUE
    _cache_dir = cache_dir or CACHE_DIR

    return GameConfig(
        # Identity
        game_id="poe2",
        default_league=_league,
        cache_dir=_cache_dir,

        # Mod Database (RePoE)
        repoe_base_url=REPOE_BASE_URL,
        repoe_cache_dir=REPOE_CACHE_DIR,
        repoe_cache_ttl=REPOE_CACHE_TTL,
        dps_item_classes=DPS_ITEM_CLASSES,
        two_hand_classes=TWO_HAND_CLASSES,
        defense_item_classes=DEFENSE_ITEM_CLASSES,
        dps_brackets_2h=DPS_BRACKETS_2H,
        dps_brackets_1h=DPS_BRACKETS_1H,
        defense_thresholds=DEFENSE_THRESHOLDS,

        # Price Sources
        price_source_url=POE2SCOUT_BASE_URL,
        price_refresh_interval=PRICE_REFRESH_INTERVAL,

        # Grade mapping
        grade_tier_map=GRADE_TIER_MAP,

        # ── Mod Database scoring (from mod_database.py) ──────
        weight_table=[
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
                "chaosresist",
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
        ],
        defence_group_markers=("reductionrating", "evasionrating", "energyshield"),
        display_names=[
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
            ("thorns", "Thorns"),
            ("lightradius", "Light"),
            ("itemrarity", "Rarity"),
            ("itemfoundrarity", "Rarity"),
        ],

        # ── Item Parser classification (from item_parser.py) ──
        currency_keywords=frozenset({
            "divine orb", "exalted orb", "chaos orb", "mirror of kalandra",
            "orb of alchemy", "orb of alteration", "orb of chance", "orb of fusing",
            "orb of regret", "orb of scouring", "blessed orb", "regal orb",
            "vaal orb", "jeweller's orb", "chromatic orb", "glassblower's bauble",
            "gemcutter's prism", "cartographer's chisel", "orb of annulment",
            "orb of binding", "engineer's orb", "harbinger's orb",
            "ancient orb", "orb of horizons", "orb of unmaking",
            "scroll of wisdom", "portal scroll", "armourer's scrap",
            "blacksmith's whetstone", "silver coin",
        }),
        valuable_bases=frozenset({
            "stellar amulet", "astral plate", "vaal regalia", "hubris circlet",
            "sorcerer boots", "sorcerer gloves", "titanium spirit shield",
            "imbued wand", "profane wand", "opal ring", "steel ring",
            "crystal belt", "stygian vise", "two-toned boots",
            "fingerless silk gloves", "gripped gloves", "bone helmet",
            "spiked gloves", "marble amulet", "agate amulet",
        }),

        # ── Price Cache endpoints & categories (from price_cache.py) ──
        poe_ninja_exchange_url="https://poe.ninja/poe2/api/economy/exchange/current/overview",
        exchange_categories=[
            "Currency", "Fragments", "Essences", "Runes", "Expedition",
            "SoulCores", "Idols", "UncutGems", "LineageSupportGems",
            "Ultimatum", "Breach", "Delirium", "Ritual", "Abyss",
        ],
        poe2scout_unique_categories=[
            "accessory", "armour", "flask", "jewel", "map", "weapon", "sanctum",
        ],
        poe2scout_currency_categories=[
            "currency", "fragments", "runes", "talismans", "essences", "ultimatum",
            "expedition", "ritual", "vaultkeys", "breach", "abyss", "uncutgems",
            "lineagesupportgems", "delirium", "incursion", "idol",
        ],
        price_request_delay=0.3,
    )
