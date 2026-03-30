"""Decode Path of Building export codes from poe.ninja into structured data.

POB codes are base64 (URL-safe) + zlib compressed XML strings containing
build info, pre-calculated stats, passive tree, skill gems, and items.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import base64
import math
import zlib
import re


@dataclass
class PobStats:
    """Pre-calculated stats from POB export."""
    # Defensive
    life: int = 0
    energy_shield: int = 0
    mana: int = 0
    spirit: int = 0
    total_ehp: float = 0.0
    evasion: int = 0
    armour: int = 0

    # Resists
    fire_resist: int = 0
    fire_resist_overcap: int = 0
    cold_resist: int = 0
    cold_resist_overcap: int = 0
    lightning_resist: int = 0
    lightning_resist_overcap: int = 0
    chaos_resist: int = 0
    chaos_resist_overcap: int = 0

    # Max hit taken
    physical_max_hit: int = 0
    fire_max_hit: int = 0
    cold_max_hit: int = 0
    lightning_max_hit: int = 0
    chaos_max_hit: int = 0
    lowest_max_hit: int = 0

    # Mitigation
    block_chance: int = 0
    spell_block_chance: int = 0
    spell_suppression: int = 0
    deflect_chance: int = 0
    evade_chance: int = 0

    # Offense
    combined_dps: float = 0.0
    total_dps: float = 0.0
    total_dot: float = 0.0
    hit_chance: float = 0.0
    crit_multiplier: float = 0.0
    speed: float = 0.0

    # Attributes
    strength: int = 0
    dexterity: int = 0
    intelligence: int = 0

    # Charges
    power_charges_max: int = 0
    frenzy_charges_max: int = 0
    endurance_charges_max: int = 0

    # Movement
    movement_speed_mod: float = 0.0

    # Raw dict of ALL stats for anything not explicitly mapped
    all_stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class PobSkillGem:
    name: str
    level: int = 0
    quality: int = 0
    enabled: bool = True


@dataclass
class PobSkillGroup:
    label: str = ""
    main_skill: str = ""
    gems: List[PobSkillGem] = field(default_factory=list)
    enabled: bool = True
    slot: str = ""  # equipment slot this is socketed in


@dataclass
class PobData:
    """Full decoded POB export."""
    class_name: str = ""
    ascendancy: str = ""
    level: int = 0
    stats: PobStats = field(default_factory=PobStats)
    passive_nodes: List[int] = field(default_factory=list)
    skill_groups: List[PobSkillGroup] = field(default_factory=list)
    tree_version: str = ""
    raw_xml: str = ""  # kept for debugging


# Mapping from POB stat names to PobStats fields
_STAT_MAP = {
    "Life": "life",
    "EnergyShield": "energy_shield",
    "Mana": "mana",
    "Spirit": "spirit",
    "TotalEHP": "total_ehp",
    "Evasion": "evasion",
    "Armour": "armour",
    "FireResist": "fire_resist",
    "FireResistOverCap": "fire_resist_overcap",
    "ColdResist": "cold_resist",
    "ColdResistOverCap": "cold_resist_overcap",
    "LightningResist": "lightning_resist",
    "LightningResistOverCap": "lightning_resist_overcap",
    "ChaosResist": "chaos_resist",
    "ChaosResistOverCap": "chaos_resist_overcap",
    "PhysicalMaximumHitTaken": "physical_max_hit",
    "FireMaximumHitTaken": "fire_max_hit",
    "ColdMaximumHitTaken": "cold_max_hit",
    "LightningMaximumHitTaken": "lightning_max_hit",
    "ChaosMaximumHitTaken": "chaos_max_hit",
    "LowestMaximumHitTaken": "lowest_max_hit",
    "BlockChance": "block_chance",
    "SpellBlockChance": "spell_block_chance",
    "SpellSuppressionChance": "spell_suppression",
    "DeflectChance": "deflect_chance",
    "EvadeChance": "evade_chance",
    "CombinedDPS": "combined_dps",
    "TotalDPS": "total_dps",
    "TotalDot": "total_dot",
    "HitChance": "hit_chance",
    "CritMultiplier": "crit_multiplier",
    "Speed": "speed",
    "Str": "strength",
    "Dex": "dexterity",
    "Int": "intelligence",
    "PowerChargesMax": "power_charges_max",
    "FrenzyChargesMax": "frenzy_charges_max",
    "EnduranceChargesMax": "endurance_charges_max",
    "EffectiveMovementSpeedMod": "movement_speed_mod",
}


def decode_pob_code(pob_code: str) -> Optional[PobData]:
    """Decode a POB export code into structured data.

    The code is base64 (URL-safe) encoded, zlib compressed XML.
    Returns None if decoding fails.
    """
    if not pob_code:
        return None
    try:
        # Fix URL-safe base64
        code = pob_code.replace("-", "+").replace("_", "/")
        # Add padding
        code += "=" * (4 - len(code) % 4) if len(code) % 4 else ""
        raw = zlib.decompress(base64.b64decode(code))
        xml_str = raw.decode("utf-8", errors="replace")
    except Exception:
        return None

    return _parse_pob_xml(xml_str)


def _parse_pob_xml(xml_str: str) -> PobData:
    """Parse POB XML into PobData. Uses regex for speed (no xml.etree dependency issues with malformed XML)."""
    data = PobData(raw_xml=xml_str)

    # Build element
    build_match = re.search(
        r'<Build\s+([^>]+)>', xml_str
    )
    if build_match:
        attrs = build_match.group(1)
        data.level = int(_attr(attrs, "level") or "0")
        data.class_name = _attr(attrs, "className") or ""
        data.ascendancy = _attr(attrs, "ascendClassName") or ""

    # PlayerStats
    stats = PobStats()
    for m in re.finditer(r'<PlayerStat\s+stat="([^"]+)"\s+value="([^"]+)"', xml_str):
        stat_name, value_str = m.group(1), m.group(2)
        try:
            value = float(value_str)
        except (ValueError, OverflowError):
            continue
        if not math.isfinite(value):
            continue
        stats.all_stats[stat_name] = value
        field_name = _STAT_MAP.get(stat_name)
        if field_name:
            if isinstance(getattr(stats, field_name), int):
                try:
                    setattr(stats, field_name, int(value))
                except (OverflowError, ValueError):
                    pass
            else:
                setattr(stats, field_name, value)
    data.stats = stats

    # Passive tree — extract from <Spec> nodes attribute
    spec_match = re.search(r'<Spec[^>]*\bnodes="([^"]+)"', xml_str)
    if spec_match:
        node_str = spec_match.group(1)
        data.passive_nodes = [int(n) for n in node_str.split(",") if n.strip()]

    # Tree version
    tv_match = re.search(r'treeVersion="([^"]+)"', xml_str)
    if tv_match:
        data.tree_version = tv_match.group(1)

    # Skill groups
    for sg_match in re.finditer(
        r'<Skill\s+([^>]*)>(.*?)</Skill>', xml_str, re.DOTALL
    ):
        sg_attrs = sg_match.group(1)
        sg_body = sg_match.group(2)

        group = PobSkillGroup(
            label=_attr(sg_attrs, "label") or "",
            enabled=_attr(sg_attrs, "enabled") != "false",
            slot=_attr(sg_attrs, "slot") or "",
        )

        # Find main active skill index
        main_idx_str = _attr(sg_attrs, "mainActiveSkill") or "1"

        # Parse gems
        for gem_match in re.finditer(r'<Gem\s+([^/]*)/?>', sg_body):
            gem_attrs = gem_match.group(1)
            gem = PobSkillGem(
                name=_attr(gem_attrs, "nameSpec") or _attr(gem_attrs, "skillId") or "",
                level=int(_attr(gem_attrs, "level") or "1"),
                quality=int(_attr(gem_attrs, "quality") or "0"),
                enabled=_attr(gem_attrs, "enabled") != "false",
            )
            if gem.name:
                group.gems.append(gem)

        # Set main skill name
        try:
            idx = int(main_idx_str) - 1
            active_gems = [g for g in group.gems if g.enabled]
            if 0 <= idx < len(active_gems):
                group.main_skill = active_gems[idx].name
        except (ValueError, IndexError):
            pass

        if group.gems:
            data.skill_groups.append(group)

    return data


def _attr(attrs_str: str, name: str) -> Optional[str]:
    """Extract an XML attribute value by name from an attribute string."""
    m = re.search(rf'{name}="([^"]*)"', attrs_str)
    return m.group(1) if m else None
