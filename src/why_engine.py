"""
why_engine.py -- Plain-language explanation generator for LAMA.

Consumes population data (search protobuf), character data (profile API),
POB stats (decoded export), and game knowledge to produce contextual
explanations of WHY a build uses specific items, keystones, and skills.

Usage:
    engine = WhyEngine(builds_client)
    explanations = engine.explain_character(char_data, archetype)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from builds_client import BuildsClient, CharacterData, BuildArchetype
from pob_decoder import decode_pob_code, PobData, PobStats
from game_knowledge import (
    KEYSTONES, DEFENSE_MECHANICS, STAT_THRESHOLDS, MOD_SYNERGIES,
    UNIQUE_JEWELS,
    KeystoneInfo, StatThreshold,
)

logger = logging.getLogger("why_engine")


# ---------------------------------------------------------------------------
# Explanation data structures
# ---------------------------------------------------------------------------

@dataclass
class Explanation:
    """A single plain-language explanation."""
    context: str        # where this applies: "keystone", "gear", "stat", "meta", "action"
    title: str          # short label: "Sanguimancy", "Fire Resistance", "Helm Slot"
    text: str           # the actual explanation
    severity: str = "info"  # "info", "warning", "critical", "positive"
    stat_delta: Optional[str] = None   # e.g., "+40K EHP" or "-15% DPS"
    source: str = ""    # data source: "population", "game_knowledge", "pob"
    slot: str = ""      # equipment slot if gear-related
    adoption_pct: float = 0.0  # population adoption % if relevant


@dataclass
class SynergyContributor:
    """A single item/keystone/mod that contributes to a stat category."""
    name: str
    source_type: str    # "keystone", "gear", "skill", "passive", "missing"
    contribution: str   # e.g., "+30% more spell damage", "Chain explosions on kill"
    slot: str = ""      # equipment slot if gear
    severity: str = "info"      # "positive" if has it, "critical"/"warning" if missing
    adoption_pct: float = 0.0
    detail: str = ""    # full explanation text
    estimated_value: float = 0.0   # estimated raw DPS or EHP contribution
    estimated_pct: float = 0.0     # estimated % of total


@dataclass
class SynergyCategory:
    """A stat category with its contributors for the synergy map."""
    category: str       # "dps", "survival", "clear_speed"
    label: str          # "DPS", "Survival", "Clear Speed"
    icon: str
    value: str          # "184K Comet", "18,057 EHP", etc.
    status: str         # severity color
    contributors: List[SynergyContributor] = field(default_factory=list)
    missing: List[SynergyContributor] = field(default_factory=list)


@dataclass
class InsightGroup:
    """A group of related explanations under one impact category."""
    category: str       # "survival", "damage", "clear_speed", "quality"
    label: str          # "Survival", "Damage", "Clear Speed", "Quality of Life"
    icon: str           # material icon name
    severity: str       # worst severity in group
    summary: str        # 1-2 sentence summary of the group
    items: List[Explanation] = field(default_factory=list)
    stat_value: Optional[str] = None   # e.g., "18,057 EHP" or "4.5M DPS"
    stat_status: str = "info"          # color of the stat: "critical", "warning", "positive"


@dataclass
class Scorecard:
    """Top-level build scorecard — the first thing the player sees."""
    ehp: int = 0
    ehp_status: str = "info"        # "critical", "warning", "positive"
    ehp_context: str = ""           # "Bottom 15% of this build"
    dps: float = 0.0
    dps_label: str = ""             # "4.5M Poisonburst Arrow"
    dps_status: str = "info"
    resist_summary: str = ""        # "All capped" or "Fire 45% / Cold 75% / Light 75% / Chaos -12%"
    resist_status: str = "info"
    critical_count: int = 0         # number of critical issues
    warning_count: int = 0
    positive_count: int = 0


@dataclass
class CharacterExplanations:
    """All explanations for a character."""
    keystones: List[Explanation] = field(default_factory=list)
    gear: Dict[str, List[Explanation]] = field(default_factory=dict)
    stats: List[Explanation] = field(default_factory=list)
    actions: List[Explanation] = field(default_factory=list)
    meta: List[Explanation] = field(default_factory=list)
    # Summarized output
    scorecard: Optional[Scorecard] = None
    insight_groups: List[InsightGroup] = field(default_factory=list)
    synergy_map: List[SynergyCategory] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for API response."""
        result = {
            "keystones": [_exp_dict(e) for e in self.keystones],
            "gear": {slot: [_exp_dict(e) for e in exps] for slot, exps in self.gear.items()},
            "stats": [_exp_dict(e) for e in self.stats],
            "actions": [_exp_dict(e) for e in self.actions],
            "meta": [_exp_dict(e) for e in self.meta],
        }
        if self.scorecard:
            result["scorecard"] = {
                "ehp": self.scorecard.ehp,
                "ehpStatus": self.scorecard.ehp_status,
                "ehpContext": self.scorecard.ehp_context,
                "dps": self.scorecard.dps,
                "dpsLabel": self.scorecard.dps_label,
                "dpsStatus": self.scorecard.dps_status,
                "resistSummary": self.scorecard.resist_summary,
                "resistStatus": self.scorecard.resist_status,
                "criticalCount": self.scorecard.critical_count,
                "warningCount": self.scorecard.warning_count,
                "positiveCount": self.scorecard.positive_count,
            }
        if self.insight_groups:
            result["insightGroups"] = [
                {
                    "category": g.category,
                    "label": g.label,
                    "icon": g.icon,
                    "severity": g.severity,
                    "summary": g.summary,
                    "statValue": g.stat_value,
                    "statStatus": g.stat_status,
                    "items": [_exp_dict(e) for e in g.items],
                }
                for g in self.insight_groups
            ]
        if self.synergy_map:
            result["synergyMap"] = [
                {
                    "category": sc.category,
                    "label": sc.label,
                    "icon": sc.icon,
                    "value": sc.value,
                    "status": sc.status,
                    "contributors": [
                        {"name": c.name, "sourceType": c.source_type, "contribution": c.contribution,
                         "slot": c.slot, "severity": c.severity, "adoptionPct": c.adoption_pct,
                         "detail": c.detail, "estimatedValue": c.estimated_value, "estimatedPct": c.estimated_pct}
                        for c in sc.contributors
                    ],
                    "missing": [
                        {"name": c.name, "sourceType": c.source_type, "contribution": c.contribution,
                         "slot": c.slot, "severity": c.severity, "adoptionPct": c.adoption_pct,
                         "detail": c.detail, "estimatedValue": c.estimated_value, "estimatedPct": c.estimated_pct}
                        for c in sc.missing
                    ],
                }
                for sc in self.synergy_map
            ]
        return result


def _exp_dict(e: Explanation) -> dict:
    d = {"context": e.context, "title": e.title, "text": e.text, "severity": e.severity}
    if e.stat_delta:
        d["statDelta"] = e.stat_delta
    if e.source:
        d["source"] = e.source
    if e.slot:
        d["slot"] = e.slot
    if e.adoption_pct:
        d["adoptionPct"] = e.adoption_pct
    return d


# ---------------------------------------------------------------------------
# Why Engine
# ---------------------------------------------------------------------------

class WhyEngine:
    """Generates plain-language explanations for a character's build choices."""

    def __init__(self, builds_client: BuildsClient):
        self._client = builds_client

    def explain_character(
        self,
        char_data: CharacterData,
        archetype: BuildArchetype,
    ) -> CharacterExplanations:
        """Generate all explanations for a character.

        This is the main entry point. Fetches population data, decodes POB,
        and produces contextual explanations across all categories.
        """
        result = CharacterExplanations()

        # Decode POB stats if available
        pob: Optional[PobData] = None
        if char_data.pob_code:
            pob = decode_pob_code(char_data.pob_code)

        # Fetch population data for this archetype
        char_class = char_data.ascendancy or char_data.char_class
        skill = archetype.main_skill
        profile = None
        popular_keystones = []

        popular_rare_mods = {}

        if char_class and skill:
            profile = self._client.fetch_archetype_profile(char_class, skill)
            popular_keystones = self._client.fetch_popular_keystones(char_class, skill)
            # Rare mod analysis — use only 5 chars to keep it fast
            try:
                popular_rare_mods = self._client.fetch_popular_rare_mods(
                    char_class, skill, max_chars=5
                )
            except Exception as e:
                logger.debug(f"Popular rare mods fetch failed: {e}")

        # Generate explanations by category
        result.stats = self._explain_stats(pob, profile, char_data)
        result.keystones = self._explain_keystones(
            char_data.keystones, popular_keystones, archetype, pob, profile,
            ascendancy_points=getattr(char_data, 'ascendancy_points', 0),
        )
        result.actions = self._generate_actions(
            char_data, archetype, pob, profile, popular_keystones
        )
        result.gear = self._explain_gear(char_data, archetype, profile)
        result.meta = self._explain_meta(archetype, profile)

        # Summarize: build scorecard + deduped insight groups
        result.scorecard = self._build_scorecard(pob, profile, result, char_data)
        result.insight_groups = self._build_insight_groups(result)
        result.synergy_map = self._build_synergy_map(
            char_data, archetype, pob, profile, popular_keystones, result,
            popular_rare_mods=popular_rare_mods,
        )

        return result

    # ------------------------------------------------------------------
    # Stat explanations
    # ------------------------------------------------------------------

    def _explain_stats(
        self, pob: Optional[PobData], profile: Optional[dict],
        char_data: CharacterData,
    ) -> List[Explanation]:
        explanations = []
        if not pob:
            return explanations

        stats = pob.stats
        ranges = profile.get("statRanges", {}) if profile else {}
        total = profile.get("totalCount", 0) if profile else 0

        # EHP
        if stats.total_ehp > 0:
            ehp_range = ranges.get("ehp", {})
            threshold = STAT_THRESHOLDS.get("ehp")
            explanations.append(self._stat_position_explanation(
                "Effective HP", int(stats.total_ehp), ehp_range, threshold,
                desc=DEFENSE_MECHANICS.get("ehp", None),
            ))

        # Life
        if stats.life > 0:
            life_range = ranges.get("life", {})
            threshold = STAT_THRESHOLDS.get("life")
            explanations.append(self._stat_position_explanation(
                "Life", stats.life, life_range, threshold,
            ))

        # Energy Shield
        if stats.energy_shield > 0:
            es_range = ranges.get("energyshield", {})
            threshold = STAT_THRESHOLDS.get("energy_shield")
            explanations.append(self._stat_position_explanation(
                "Energy Shield", stats.energy_shield, es_range, threshold,
            ))

        # Resists
        for resist_name, stat_field, range_key in [
            ("Fire Resistance", stats.fire_resist, "fireres"),
            ("Cold Resistance", stats.cold_resist, "coldres"),
            ("Lightning Resistance", stats.lightning_resist, "lightningres"),
            ("Chaos Resistance", stats.chaos_resist, "chaosres"),
        ]:
            r_range = ranges.get(range_key, {})
            explanations.append(self._explain_resist(
                resist_name, stat_field, r_range,
                is_chaos=(range_key == "chaosres"),
            ))

        # Armour / Evasion — only show if they're a meaningful part of the build
        # (above 1000 suggests intentional investment, not just incidental gear)
        if stats.armour > 1000:
            a_range = ranges.get("armour", {})
            explanations.append(self._stat_position_explanation(
                "Armour", stats.armour, a_range,
                desc=DEFENSE_MECHANICS.get("armour", None),
            ))

        if stats.evasion > 1000:
            ev_range = ranges.get("evasion", {})
            explanations.append(self._stat_position_explanation(
                "Evasion", stats.evasion, ev_range,
                desc=DEFENSE_MECHANICS.get("evasion", None),
            ))

        return [e for e in explanations if e is not None]

    def _stat_position_explanation(
        self, name: str, value: int, pop_range: dict,
        threshold: Optional[StatThreshold] = None,
        desc=None,
    ) -> Optional[Explanation]:
        """Generate a stat position explanation against population range."""
        parts = []
        severity = "info"

        # Population position
        pop_min = pop_range.get("min", 0)
        pop_max = pop_range.get("max", 0)
        if pop_max > pop_min:
            pct = ((value - pop_min) / (pop_max - pop_min)) * 100
            pct = max(0, min(100, pct))
            if pct < 15:
                parts.append(f"Your {name} ({value:,}) is near the bottom of this build's range ({pop_min:,} - {pop_max:,}).")
                severity = "warning"
            elif pct < 40:
                parts.append(f"Your {name} ({value:,}) is below average for this build ({pop_min:,} - {pop_max:,}).")
                severity = "warning"
            elif pct > 80:
                parts.append(f"Your {name} ({value:,}) is excellent for this build (top 20% of {pop_min:,} - {pop_max:,} range).")
                severity = "positive"
            else:
                parts.append(f"Your {name} ({value:,}) is solid for this build (range: {pop_min:,} - {pop_max:,}).")

        # Threshold context
        if threshold:
            if value < threshold.endgame_min:
                parts.append(f"Below endgame minimum ({threshold.endgame_min:,}). {threshold.description}")
                severity = "critical"
            elif value < threshold.boss_ready:
                parts.append(f"Adequate for mapping but below boss-ready ({threshold.boss_ready:,}). {threshold.description}")
                if severity != "critical":
                    severity = "warning"

        # Mechanic description
        if desc and hasattr(desc, 'description'):
            parts.append(desc.description)

        if not parts:
            return None

        return Explanation(
            context="stat", title=name, text=" ".join(parts),
            severity=severity, source="population",
        )

    def _explain_resist(
        self, name: str, value: int, pop_range: dict,
        is_chaos: bool = False,
    ) -> Explanation:
        cap = 75
        parts = []
        severity = "info"

        if value >= cap:
            overcap = value - cap
            if overcap > 0:
                parts.append(f"{name} is capped at {cap}% with {overcap}% overcap (helps against curses).")
                severity = "positive"
            else:
                parts.append(f"{name} is capped at {cap}%.")
                severity = "positive"
        elif is_chaos:
            if value < 0:
                parts.append(f"{name} is negative ({value}%). Chaos damage bypasses Energy Shield by default — this is dangerous.")
                severity = "critical"
            elif value < 30:
                parts.append(f"{name} is {value}% (cap is {cap}%). Chaos damage bypasses ES, so even life builds want positive chaos res.")
                severity = "warning"
            else:
                parts.append(f"{name} is {value}% ({cap - value}% below cap). Getting closer — each point matters.")
                severity = "info"
        else:
            gap = cap - value
            parts.append(f"{name} is {value}% — {gap}% below the {cap}% cap. Uncapped resists mean taking significantly more elemental damage.")
            severity = "critical" if gap > 30 else "warning"

        resist_mech = DEFENSE_MECHANICS.get("resist_caps")
        if resist_mech and value < cap:
            parts.append(resist_mech.description)

        return Explanation(
            context="stat", title=name, text=" ".join(parts),
            severity=severity, source="game_knowledge",
        )

    # ------------------------------------------------------------------
    # Keystone explanations
    # ------------------------------------------------------------------

    def _explain_keystones(
        self,
        player_keystones: List[str],
        popular_keystones: List[dict],
        archetype: BuildArchetype,
        pob: Optional[PobData],
        profile: Optional[dict],
        ascendancy_points: int = 0,
    ) -> List[Explanation]:
        explanations = []
        player_ks_set = set(player_keystones)
        # If player has allocated ascendancy points, they likely have the
        # major ascendancy passives — don't flag those as missing
        has_ascendancy_allocated = ascendancy_points >= 4

        # Explain keystones the player HAS
        for ks_name in player_keystones:
            info = KEYSTONES.get(ks_name)
            pop = next((pk for pk in popular_keystones if pk["name"] == ks_name), None)
            pct = pop["percentage"] if pop else 0

            if info:
                text = f"{info.benefits}"
                if pct > 0:
                    text += f" {pct:.0f}% of players in this build use it."
                severity = "positive" if pct > 50 else "info"
            else:
                text = f"You have {ks_name} allocated."
                if pct > 0:
                    text += f" {pct:.0f}% of similar builds use it."
                severity = "info"

            explanations.append(Explanation(
                context="keystone", title=ks_name, text=text,
                severity=severity, source="game_knowledge",
                adoption_pct=pct,
            ))

        # Flag popular keystones the player is MISSING — lead with IMPACT
        # Separate ascendancy passives from tree keystones
        for pk in popular_keystones:
            if pk["name"] in player_ks_set:
                continue
            if pk["percentage"] < 50:
                continue

            info = KEYSTONES.get(pk["name"])
            pct = pk["percentage"]
            node_type = pk.get("type", "")  # "Ascendancy" or "Keystone"
            is_ascendancy = node_type == "Ascendancy"

            # CRITICAL: If player has ascendancy points allocated, they likely
            # already have the major ascendancy passives. Don't tell them to
            # get something they already have.
            if is_ascendancy and has_ascendancy_allocated:
                continue  # Skip — player has ascendancy points, likely has this

            # Build the explanation text
            if info:
                text = info.impact
                text += f" ({pct:.0f}% of this build uses {pk['name']}.)"
            elif is_ascendancy:
                if pct >= 90:
                    text = (
                        f"{pk['name']} is a core ascendancy passive used by {pct:.0f}% of this build. "
                        f"You earn ascendancy passives by completing ascendancy trials. "
                        f"This is likely essential to how the build functions."
                    )
                else:
                    text = (
                        f"{pk['name']} is an ascendancy passive used by {pct:.0f}% of this build. "
                        f"Check if you have ascendancy points to allocate it."
                    )
            else:
                if pct >= 95:
                    text = (
                        f"{pk['name']} is a passive tree keystone used by {pct:.0f}% of this build — "
                        f"essentially mandatory. You need to path to it on the passive tree."
                    )
                elif pct >= 70:
                    text = (
                        f"{pk['name']} is a passive tree keystone used by {pct:.0f}% of this build. "
                        f"Pathing to it on the passive tree requires investing points in that region."
                    )
                else:
                    text = (
                        f"{pk['name']} is used by {pct:.0f}% of this build. "
                        f"A popular choice that complements this archetype."
                    )

            severity = "critical" if pct > 70 else "warning"
            # Ascendancy passives with very high adoption are less "critical" since
            # the player probably has them and we just can't see them
            if is_ascendancy and pct > 90:
                severity = "warning"

            title_prefix = "Ascendancy: " if is_ascendancy else "Keystone: "

            explanations.append(Explanation(
                context="keystone", title=f"{title_prefix}{pk['name']}",
                text=text, severity=severity,
                source="population" if not info else "game_knowledge",
                adoption_pct=pct,
            ))

        return explanations

    # ------------------------------------------------------------------
    # Action recommendations
    # ------------------------------------------------------------------

    def _generate_actions(
        self,
        char_data: CharacterData,
        archetype: BuildArchetype,
        pob: Optional[PobData],
        profile: Optional[dict],
        popular_keystones: List[dict],
    ) -> List[Explanation]:
        actions = []

        # Uncapped resists — consolidate into single action
        if pob:
            uncapped = []
            for resist_name, val in [
                ("Fire", pob.stats.fire_resist),
                ("Cold", pob.stats.cold_resist),
                ("Lightning", pob.stats.lightning_resist),
                ("Chaos", pob.stats.chaos_resist),
            ]:
                if val < 75:
                    uncapped.append((resist_name, val, 75 - val))

            if uncapped:
                if len(uncapped) == 1:
                    rn, rv, gap = uncapped[0]
                    text = (
                        f"Your {rn} resistance is {rv}% — {gap}% below the 75% cap. "
                        f"You take {_damage_increase(rv)}% more {rn.lower()} damage than intended. "
                        f"Look for {rn.lower()} res on rings, belt, or gloves."
                    )
                else:
                    parts = [f"{rn} {rv}%" for rn, rv, _ in uncapped]
                    text = (
                        f"You have {len(uncapped)} uncapped elemental resistances: {', '.join(parts)}. "
                        f"Each uncapped resist means taking significantly more damage of that type. "
                        f"Prioritize all-resistance mods on rings, belt, or amulet to cap multiple at once."
                    )
                actions.append(Explanation(
                    context="action",
                    title=f"Cap {'Resistances' if len(uncapped) > 1 else uncapped[0][0] + ' Resistance'}",
                    text=text,
                    severity="critical",
                    source="game_knowledge",
                ))

            # Note: chaos resist is now included in the uncapped check above

        # Missing high-adoption keystones — lead with impact, distinguish type
        player_ks = set(char_data.keystones)
        has_asc = getattr(char_data, 'ascendancy_points', 0) >= 4
        for pk in popular_keystones:
            if pk["name"] in player_ks or pk["percentage"] < 70:
                continue
            info = KEYSTONES.get(pk["name"])
            pct = pk["percentage"]
            node_type = pk.get("type", "")
            is_ascendancy = node_type == "Ascendancy"

            # Skip ascendancy passives if player has ascendancy points allocated
            if is_ascendancy and has_asc:
                continue

            if info:
                text = f"{info.impact} ({pct:.0f}% of this build uses it.)"
            elif is_ascendancy:
                text = (
                    f"{pk['name']} is a core ascendancy passive ({pct:.0f}% adoption). "
                    f"Earn it by completing ascendancy trials."
                )
            else:
                text = (
                    f"{pk['name']} is used by {pct:.0f}% of this build. "
                    f"Path to it on the passive tree for a significant boost."
                )

            # Ascendancy passives at 90%+ are likely already taken — downgrade severity
            if is_ascendancy and pct > 90:
                sev = "info"
                title = f"Check Ascendancy: {pk['name']}"
            else:
                sev = "critical" if pct > 90 else "warning"
                title = f"Allocate {pk['name']}" if not is_ascendancy else f"Ascendancy: {pk['name']}"

            actions.append(Explanation(
                context="action",
                title=title,
                text=text,
                severity=sev,
                source="population" if not info else "game_knowledge",
                adoption_pct=pct,
            ))

        # Low EHP warning
        if pob and pob.stats.total_ehp > 0:
            ehp_threshold = STAT_THRESHOLDS.get("ehp")
            if ehp_threshold and pob.stats.total_ehp < ehp_threshold.endgame_min:
                actions.append(Explanation(
                    context="action", title="Increase Effective HP",
                    text=(
                        f"Your EHP ({int(pob.stats.total_ehp):,}) is below the endgame minimum ({ehp_threshold.endgame_min:,}). "
                        f"{ehp_threshold.description} "
                        f"Focus on life/ES on gear, resist cap, and defensive keystones."
                    ),
                    severity="critical",
                    source="game_knowledge",
                ))

        # Dead mods warning
        if archetype.dead_mods:
            dead_count = len(archetype.dead_mods)
            slots = set(dm.get("slot", "") for dm in archetype.dead_mods)
            actions.append(Explanation(
                context="action", title=f"{dead_count} Dead Mods Found",
                text=(
                    f"You have {dead_count} mods across {', '.join(slots)} that don't benefit your "
                    f"{archetype.damage_type} {archetype.main_skill} build. "
                    f"Replacing these with build-relevant mods is a direct upgrade."
                ),
                severity="warning",
                source="game_knowledge",
            ))

        # ── Gear-based improvement recommendations ──────────
        total_dps = 0
        for sg in char_data.skill_groups:
            for d in (sg.dps if hasattr(sg, "dps") and sg.dps else []):
                total_dps = max(total_dps, d.dps or 0, d.dot_dps or 0, d.damage or 0)

        _add_gear_improvement_actions(char_data, archetype, pob, total_dps, actions)

        # Sort by severity
        sev_order = {"critical": 0, "warning": 1, "info": 2, "positive": 3}
        actions.sort(key=lambda a: sev_order.get(a.severity, 9))

        return actions

    # ------------------------------------------------------------------
    # Gear explanations
    # ------------------------------------------------------------------

    def _explain_gear(
        self,
        char_data: CharacterData,
        archetype: BuildArchetype,
        profile: Optional[dict],
    ) -> Dict[str, List[Explanation]]:
        gear: Dict[str, List[Explanation]] = {}

        for item in char_data.equipment:
            slot = item.slot
            if slot in ("Flask", "Flask2"):
                continue
            slot_exps = []

            # Dead mods on this slot
            dead = [dm for dm in (archetype.dead_mods or []) if dm.get("slot") == slot]
            for dm in dead:
                mod_text = dm.get("mod", "")
                reason = dm.get("reason", "doesn't benefit this build")
                slot_exps.append(Explanation(
                    context="gear", title="Wasted Mod",
                    text=f"\"{mod_text}\" — {reason}. Consider replacing with a mod that scales your {archetype.damage_type} damage or defenses.",
                    severity="warning", source="game_knowledge", slot=slot,
                ))

            # Item is unique — check if it's popular
            if item.rarity == "Unique" and item.name:
                slot_exps.append(Explanation(
                    context="gear", title=item.name,
                    text=f"You're using the unique {item.name}. Unique items provide fixed, build-defining stats that can't be found on rares.",
                    severity="info", source="game_knowledge", slot=slot,
                ))

            if slot_exps:
                gear[slot] = slot_exps

        return gear

    # ------------------------------------------------------------------
    # Meta context
    # ------------------------------------------------------------------

    def _explain_meta(
        self, archetype: BuildArchetype, profile: Optional[dict],
    ) -> List[Explanation]:
        explanations = []

        if profile:
            total = profile.get("totalCount", 0)
            if total > 0:
                explanations.append(Explanation(
                    context="meta",
                    title="Build Population",
                    text=(
                        f"There are {total:,} characters running {archetype.main_skill} "
                        f"as {archetype.tags[0] if archetype.tags else 'this class'} on the ladder. "
                        f"Analysis is based on this population's gear, keystones, and stats."
                    ),
                    severity="info",
                    source="population",
                ))

        # Build type summary
        parts = []
        if archetype.damage_type != "unknown":
            parts.append(f"{archetype.damage_type} damage")
        if archetype.defense_type:
            parts.append(f"{archetype.defense_type} defense")
        if archetype.is_crit:
            parts.append("crit scaling")
        if archetype.is_coc:
            parts.append("Cast on Crit")
        if archetype.elements:
            parts.append(f"{'/'.join(archetype.elements)} element(s)")

        if parts:
            explanations.append(Explanation(
                context="meta",
                title="Build Classification",
                text=f"LAMA classifies your build as: {', '.join(parts)}. Scoring and recommendations are tuned for this archetype.",
                severity="info",
                source="game_knowledge",
            ))

        return explanations

    # ------------------------------------------------------------------
    # Summarization: Scorecard + Insight Groups
    # ------------------------------------------------------------------

    def _build_scorecard(
        self, pob: Optional[PobData], profile: Optional[dict],
        exps: "CharacterExplanations", char_data: Optional[CharacterData] = None,
    ) -> Scorecard:
        """Build the top-level scorecard with big numbers."""
        sc = Scorecard()

        if pob:
            # EHP
            sc.ehp = int(pob.stats.total_ehp)
            ranges = profile.get("statRanges", {}) if profile else {}
            ehp_range = ranges.get("ehp", {})
            ehp_max = ehp_range.get("max", 0)
            ehp_thresh = STAT_THRESHOLDS.get("ehp")
            if ehp_thresh and sc.ehp < ehp_thresh.endgame_min:
                sc.ehp_status = "critical"
                sc.ehp_context = f"Below endgame minimum ({ehp_thresh.endgame_min:,})"
            elif ehp_thresh and sc.ehp < ehp_thresh.boss_ready:
                sc.ehp_status = "warning"
                sc.ehp_context = f"OK for mapping, below boss-ready ({ehp_thresh.boss_ready:,})"
            elif ehp_max > 0:
                pct = min(100, max(0, (sc.ehp / ehp_max) * 100))
                if pct > 50:
                    sc.ehp_status = "positive"
                    sc.ehp_context = f"Top {100 - int(pct)}% of this build"
                else:
                    sc.ehp_status = "info"
                    sc.ehp_context = f"Average for this build"

            # DPS — from character skill groups (poe.ninja calculates these)
            best_dps = 0.0
            best_skill = ""
            if char_data:
                for sg in char_data.skill_groups:
                    for d in (sg.dps if hasattr(sg, "dps") and sg.dps else []):
                        total = max(d.dps or 0, d.dot_dps or 0, d.damage or 0)
                        if total > best_dps:
                            best_dps = total
                            best_skill = d.name or ""

            # Fall back to POB stats if character data didn't have DPS
            if best_dps == 0:
                if pob.stats.combined_dps > 0:
                    best_dps = pob.stats.combined_dps
                if pob.stats.total_dot > best_dps:
                    best_dps = pob.stats.total_dot

            if best_dps > 0:
                sc.dps = best_dps
                if best_dps >= 1_000_000:
                    label = f"{best_dps / 1_000_000:.1f}M"
                elif best_dps >= 1_000:
                    label = f"{best_dps / 1_000:.0f}K"
                else:
                    label = f"{best_dps:,.0f}"
                sc.dps_label = f"{label} {best_skill}".strip()
                sc.dps_status = "info"

                # DPS ceiling + percentile from featured characters
                if profile:
                    featured = profile.get("featuredCharacters", [])
                    pop_dps_values = []
                    # Extract DPS from featured characters
                    for fch in featured:
                        for fk, fv in fch.items():
                            if fk.startswith("dps") and fv:
                                try:
                                    dps_str = str(fv)
                                    m = _re.match(r"([\d.]+)\s*([KkMm])?", dps_str)
                                    if m:
                                        v = float(m.group(1))
                                        mult = m.group(2) or ""
                                        if mult.upper() == "K": v *= 1000
                                        elif mult.upper() == "M": v *= 1_000_000
                                        if v > 100:
                                            pop_dps_values.append(v)
                                except (ValueError, TypeError):
                                    pass

                    if pop_dps_values:
                        pop_dps_values.sort()
                        dps_max = max(pop_dps_values)
                        dps_median = pop_dps_values[len(pop_dps_values) // 2]
                        # Percentile: what % of population is below this player
                        below = sum(1 for v in pop_dps_values if v <= best_dps)
                        dps_pct = int(below / len(pop_dps_values) * 100)

                        top_label = _format_number(dps_max)
                        median_label = _format_number(dps_median)

                        if dps_pct < 25:
                            sc.dps_status = "warning"
                            sc.dps_label += f" (top players hit {top_label})"
                        elif dps_pct > 75:
                            sc.dps_status = "positive"

            # Resists
            resists = {
                "Fire": pob.stats.fire_resist,
                "Cold": pob.stats.cold_resist,
                "Lightning": pob.stats.lightning_resist,
                "Chaos": pob.stats.chaos_resist,
            }
            all_capped = all(v >= 75 for v in resists.values())
            ele_capped = all(resists[r] >= 75 for r in ["Fire", "Cold", "Lightning"])
            if all_capped:
                sc.resist_summary = "All resistances capped"
                sc.resist_status = "positive"
            elif ele_capped:
                sc.resist_summary = f"Elemental capped, Chaos {resists['Chaos']}%"
                sc.resist_status = "warning" if resists["Chaos"] < 0 else "info"
            else:
                uncapped = [f"{n} {v}%" for n, v in resists.items() if v < 75]
                sc.resist_summary = f"Uncapped: {', '.join(uncapped)}"
                sc.resist_status = "critical"

        # Count severities across all explanations
        all_exps = (exps.actions + exps.keystones + exps.stats
                    + exps.meta + [e for sl in exps.gear.values() for e in sl])
        sc.critical_count = sum(1 for e in all_exps if e.severity == "critical")
        sc.warning_count = sum(1 for e in all_exps if e.severity == "warning")
        sc.positive_count = sum(1 for e in all_exps if e.severity == "positive")

        return sc

    def _build_insight_groups(
        self, exps: "CharacterExplanations",
    ) -> List[InsightGroup]:
        """Dedup and group explanations into impact categories."""

        # Collect all explanations, tagging each with an impact category
        seen_titles = set()  # for dedup
        survival_items = []
        damage_items = []
        clear_items = []
        quality_items = []

        # Classify keywords for routing
        SURVIVAL_KW = {"life", "ehp", "health", "resist", "armour", "armor",
                       "evasion", "block", "suppress", "deflect", "es ", "energy shield",
                       "sanguimancy", "mind over matter", "chaos inoculation",
                       "sustain", "leech", "regen", "recovery", "survive", "die",
                       "damage taken", "hit taken", "defensive"}
        DAMAGE_KW = {"dps", "damage", "multiplier", "crit", "penetration",
                     "pain attunement", "crimson power", "elemental overload",
                     "more damage", "spell damage", "attack damage", "dot",
                     "grasping wounds", "vitality siphon"}
        CLEAR_KW = {"clear", "explosion", "chain", "area", "aoe",
                    "sunder the flesh", "movement", "speed"}

        def _categorize(e: Explanation) -> str:
            text_lower = (e.title + " " + e.text).lower()
            # Score each category
            surv = sum(1 for kw in SURVIVAL_KW if kw in text_lower)
            dmg = sum(1 for kw in DAMAGE_KW if kw in text_lower)
            clr = sum(1 for kw in CLEAR_KW if kw in text_lower)
            if clr > surv and clr > dmg:
                return "clear_speed"
            if dmg > surv:
                return "damage"
            if surv > 0:
                return "survival"
            return "quality"

        def _dedup_add(e: Explanation, target: list):
            # Dedup by title (strip "Missing: " prefix for keystone matching)
            key = e.title.replace("Missing: ", "").replace("Allocate ", "")
            if key in seen_titles:
                return
            seen_titles.add(key)
            target.append(e)

        # Process actions first (highest priority), then keystones, stats, gear
        for e in exps.actions:
            cat = _categorize(e)
            _dedup_add(e, {"survival": survival_items, "damage": damage_items,
                           "clear_speed": clear_items, "quality": quality_items}[cat])

        for e in exps.keystones:
            cat = _categorize(e)
            _dedup_add(e, {"survival": survival_items, "damage": damage_items,
                           "clear_speed": clear_items, "quality": quality_items}[cat])

        # Stats: only include non-positive (skip "Fire Resistance is capped" type noise)
        for e in exps.stats:
            if e.severity == "positive":
                continue  # Don't clutter with "all good" items
            cat = _categorize(e)
            _dedup_add(e, {"survival": survival_items, "damage": damage_items,
                           "clear_speed": clear_items, "quality": quality_items}[cat])

        for slot_exps in exps.gear.values():
            for e in slot_exps:
                if e.severity in ("info",) and "unique" in e.text.lower():
                    continue  # Skip generic "you're using unique X" noise
                cat = _categorize(e)
                _dedup_add(e, {"survival": survival_items, "damage": damage_items,
                               "clear_speed": clear_items, "quality": quality_items}[cat])

        # Build groups
        groups = []

        if survival_items:
            worst = _worst_severity(survival_items)
            crit_count = sum(1 for e in survival_items if e.severity == "critical")
            warn_count = sum(1 for e in survival_items if e.severity == "warning")
            summary_parts = []
            if crit_count:
                summary_parts.append(f"{crit_count} critical issue{'s' if crit_count > 1 else ''}")
            if warn_count:
                summary_parts.append(f"{warn_count} improvement{'s' if warn_count > 1 else ''}")
            summary = f"Survival: {' and '.join(summary_parts)} found." if summary_parts else "Survival looks solid."
            # Add specific context
            if crit_count:
                top_issue = next(e for e in survival_items if e.severity == "critical")
                summary += f" Top priority: {top_issue.title.replace('Missing: ', '').replace('Allocate ', '')}."

            sc = exps.scorecard
            groups.append(InsightGroup(
                category="survival", label="Survival", icon="shield",
                severity=worst, summary=summary,
                items=survival_items,
                stat_value=f"{sc.ehp:,} EHP" if sc and sc.ehp else None,
                stat_status=sc.ehp_status if sc else "info",
            ))

        if damage_items:
            worst = _worst_severity(damage_items)
            crit_count = sum(1 for e in damage_items if e.severity == "critical")
            warn_count = sum(1 for e in damage_items if e.severity == "warning")
            summary_parts = []
            if crit_count:
                summary_parts.append(f"{crit_count} critical")
            if warn_count:
                summary_parts.append(f"{warn_count} improvement{'s' if warn_count > 1 else ''}")
            summary = f"Damage: {' and '.join(summary_parts)} found." if summary_parts else "Damage scaling looks solid."
            if crit_count:
                top_issue = next(e for e in damage_items if e.severity == "critical")
                summary += f" Top priority: {top_issue.title.replace('Missing: ', '').replace('Allocate ', '')}."

            sc = exps.scorecard
            groups.append(InsightGroup(
                category="damage", label="Damage", icon="bolt",
                severity=worst, summary=summary,
                items=damage_items,
                stat_value=sc.dps_label if sc and sc.dps_label else None,
                stat_status=sc.dps_status if sc else "info",
            ))

        if clear_items:
            worst = _worst_severity(clear_items)
            summary = f"Clear Speed: {len(clear_items)} item{'s' if len(clear_items) > 1 else ''} to address."
            groups.append(InsightGroup(
                category="clear_speed", label="Clear Speed", icon="speed",
                severity=worst, summary=summary,
                items=clear_items,
            ))

        if quality_items:
            worst = _worst_severity(quality_items)
            summary = f"{len(quality_items)} additional consideration{'s' if len(quality_items) > 1 else ''}."
            groups.append(InsightGroup(
                category="quality", label="Quality of Life", icon="tune",
                severity=worst, summary=summary,
                items=quality_items,
            ))

        # Sort groups by severity (critical first)
        sev_order = {"critical": 0, "warning": 1, "info": 2, "positive": 3}
        groups.sort(key=lambda g: sev_order.get(g.severity, 9))

        return groups

    # ------------------------------------------------------------------
    # Synergy Map: contribution trees per stat category
    # ------------------------------------------------------------------

    def _build_synergy_map(
        self, char_data: CharacterData, archetype: BuildArchetype,
        pob: Optional[PobData], profile: Optional[dict],
        popular_keystones: List[dict], exps: "CharacterExplanations",
        popular_rare_mods: Optional[dict] = None,
    ) -> List[SynergyCategory]:
        """Build contribution trees for DPS, Survival, and Clear Speed."""
        categories = []
        sc = exps.scorecard
        player_ks = set(char_data.keystones)
        total_dps = sc.dps if sc else 0

        # ── Analyze gear mods for DPS/survival contributions ──
        gear_dps_mods = []   # (slot, item_name, mod_text, category, est_pct)
        gear_surv_mods = []

        for item in char_data.equipment:
            if item.slot in ("Flask", "Flask2"):
                continue
            all_mods = (item.explicit_mods or []) + (item.implicit_mods or []) + (item.crafted_mods or [])
            slot_label = item.slot
            item_label = item.name or item.type_line or slot_label

            for mod in all_mods:
                mod_clean = _strip_ninja_brackets(mod)
                analysis = _analyze_mod_contribution(mod_clean, archetype)
                if analysis:
                    cat, desc, est_pct = analysis
                    if cat == "dps":
                        gear_dps_mods.append((slot_label, item_label, mod_clean, desc, est_pct))
                    elif cat == "survival":
                        gear_surv_mods.append((slot_label, item_label, mod_clean, desc, est_pct))

        # Sort by estimated impact
        gear_dps_mods.sort(key=lambda x: x[4], reverse=True)
        gear_surv_mods.sort(key=lambda x: x[4], reverse=True)

        # ── DPS Category ────────────────────────────────
        dps_contributors = []
        dps_missing = []

        # Main skill with DPS value
        for sg in char_data.skill_groups:
            for d in (sg.dps if hasattr(sg, "dps") and sg.dps else []):
                skill_total = max(d.dps or 0, d.dot_dps or 0, d.damage or 0)
                if skill_total > 500:
                    pct = (skill_total / total_dps * 100) if total_dps > 0 else 0
                    dps_contributors.append(SynergyContributor(
                        name=d.name or "Unknown Skill",
                        source_type="skill",
                        contribution=f"{_format_number(skill_total)} DPS",
                        severity="positive",
                        detail=f"Active skill dealing {_format_number(skill_total)} damage per second.",
                        estimated_value=skill_total,
                        estimated_pct=round(pct, 1),
                    ))

        # Support gems — each is a damage multiplier
        main_skill_group = None
        for sg in char_data.skill_groups:
            if any(d.name == archetype.main_skill for d in (sg.dps if hasattr(sg, 'dps') and sg.dps else [])):
                main_skill_group = sg
                break
        if not main_skill_group:
            for sg in char_data.skill_groups:
                if archetype.main_skill in sg.gems:
                    main_skill_group = sg
                    break

        if main_skill_group:
            support_count = 0
            for gem in main_skill_group.gems:
                gem_name = gem if isinstance(gem, str) else getattr(gem, "name", "")
                if gem_name and gem_name != archetype.main_skill:
                    support_count += 1
                    # Each support is roughly a 30-40% MORE multiplier
                    est_pct = 25.0  # conservative per-support estimate
                    dps_contributors.append(SynergyContributor(
                        name=gem_name, source_type="support",
                        contribution=f"Support gem (~{est_pct:.0f}% more multiplier)",
                        severity="positive",
                        detail=f"Linked to {archetype.main_skill}. Each support gem multiplies damage — {support_count} supports means roughly {support_count}x multiplied base damage.",
                        estimated_pct=est_pct,
                        estimated_value=total_dps * est_pct / 100 if total_dps > 0 else 0,
                    ))

        # Cross-build comparison: what do top featured characters have that we don't?
        if profile:
            featured = profile.get("featuredCharacters", [])
            # Get top 3 DPS characters' names for reference
            top_chars = []
            for fch in featured[:50]:  # check more featured chars for DPS ceiling
                for fk, fv in fch.items():
                    if fk.startswith("dps") and fv:
                        try:
                            dps_str = str(fv)
                            m = _re.match(r"([\d.]+)\s*([KkMm])?", dps_str)
                            if m:
                                v = float(m.group(1))
                                mult = m.group(2) or ""
                                if mult.upper() == "K": v *= 1000
                                elif mult.upper() == "M": v *= 1_000_000
                                if v > total_dps * 1.5:
                                    top_chars.append({
                                        "name": fch.get("name", "?"),
                                        "account": fch.get("account", "?"),
                                        "dps": v,
                                    })
                        except (ValueError, TypeError):
                            pass
                        break

            if top_chars and total_dps > 0:
                best = max(top_chars, key=lambda x: x["dps"])
                ratio = best["dps"] / total_dps
                dps_missing.append(SynergyContributor(
                    name=f"DPS Ceiling: {_format_number(best['dps'])}",
                    source_type="missing",
                    contribution=f"Top player ({best['name']}) achieves {_format_number(best['dps'])} DPS — {ratio:.1f}x your current {_format_number(total_dps)}",
                    severity="info",
                    detail=f"Look up {best['name']} on poe.ninja to see their full gear and gem setup. The gap comes from gear quality, gem levels, jewels, and support gems.",
                    estimated_value=best["dps"] - total_dps,
                    estimated_pct=round((ratio - 1) * 100, 0),
                ))

        # Gear mods contributing to DPS (top items — +gem levels, % damage, etc.)
        for slot, item_name, mod_text, desc, est_pct in gear_dps_mods[:8]:
            est_val = total_dps * est_pct / 100 if total_dps > 0 else 0
            dps_contributors.append(SynergyContributor(
                name=f"{item_name}",
                source_type="gear", slot=slot,
                contribution=f"{desc} ({mod_text})",
                severity="positive",
                detail=f"On {slot}: {mod_text}. Estimated ~{est_pct:.0f}% of total DPS.",
                estimated_value=est_val,
                estimated_pct=est_pct,
            ))

        # Jewels — analyze DPS contribution from each jewel
        jewels = getattr(char_data, 'jewels', [])
        jewel_total_dps_pct = 0
        has_adorned = False
        has_megalomaniac = False
        has_from_nothing = False
        has_historic = False
        magic_jewel_count = 0

        for jewel in jewels:
            jname = jewel.name or jewel.type_line or "Unknown Jewel"
            all_j_mods = (jewel.explicit_mods or []) + (jewel.desecrated_mods or []) + (jewel.implicit_mods or [])
            j_dps_pct = 0
            j_key_mods = []

            # Check for known unique jewels
            unique_info = UNIQUE_JEWELS.get(jname)
            if unique_info:
                est_pct = 20.0 if jname == "The Adorned" else 15.0
                dps_contributors.append(SynergyContributor(
                    name=jname,
                    source_type="gear", slot="Jewel",
                    contribution=unique_info.description[:80],
                    severity="positive",
                    detail=f"{unique_info.description} {unique_info.impact}",
                    estimated_pct=est_pct,
                    estimated_value=total_dps * est_pct / 100 if total_dps > 0 else 0,
                ))
                if jname == "The Adorned":
                    has_adorned = True
                elif jname == "Megalomaniac":
                    has_megalomaniac = True
                elif jname == "From Nothing":
                    has_from_nothing = True
                elif jname in ("Heroic Tragedy", "Undying Hate"):
                    has_historic = True
                jewel_total_dps_pct += est_pct
                continue

            # Track magic jewels (relevant for The Adorned)
            if jewel.rarity in ("Magic", "magic"):
                magic_jewel_count += 1

            # Analyze mod contributions
            for mod in all_j_mods:
                mc = _strip_ninja_brackets(mod)
                a = _analyze_mod_contribution(mc, archetype)
                if a and a[0] == "dps":
                    j_dps_pct += a[2]
                    j_key_mods.append(a[1])
                ml = mc.lower()
                if "cooldown recovery" in ml:
                    j_key_mods.append(f"Cooldown Recovery ({mc})")

            if j_dps_pct > 2 or j_key_mods:
                # If The Adorned is present and this is a magic jewel, mods are multiplied
                if has_adorned and jewel.rarity in ("Magic", "magic"):
                    j_dps_pct *= 2.0  # rough estimate of Adorned multiplier effect
                    j_key_mods.insert(0, "Adorned-multiplied")

                jewel_total_dps_pct += j_dps_pct
                est_val = total_dps * j_dps_pct / 100 if total_dps > 0 else 0
                dps_contributors.append(SynergyContributor(
                    name=jname,
                    source_type="gear", slot="Jewel",
                    contribution=f"~{j_dps_pct:.0f}% DPS: {', '.join(j_key_mods[:2])}",
                    severity="positive",
                    detail=f"Jewel with {len(all_j_mods)} mods. {', '.join(j_key_mods[:3])}",
                    estimated_value=est_val,
                    estimated_pct=j_dps_pct,
                ))

        # Missing jewel recommendations
        if len(jewels) < 8 and total_dps > 50000:
            per_slot = 7.0 if has_adorned else 5.0
            slots_missing = max(0, 8 - len(jewels))
            dps_missing.append(SynergyContributor(
                name=f"Fill Jewel Slots ({len(jewels)}/8+)",
                source_type="missing",
                contribution=f"Only {len(jewels)} jewels. Top players use 8-10. Each adds ~{per_slot:.0f}% DPS.",
                severity="warning",
                detail=(
                    f"Each jewel with spell damage, crit, and elemental damage mods adds significant scaling. "
                    f"{'The Adorned multiplies magic jewel mods — prioritize well-rolled corrupted magic jewels. ' if has_adorned else ''}"
                    f"Look for: Megalomaniac (3 random notables), or rare/magic jewels with your build's damage type + crit."
                ),
                estimated_pct=min(slots_missing * per_slot, 35),
                estimated_value=total_dps * min(slots_missing * per_slot / 100, 0.35),
            ))

        # Suggest The Adorned if player doesn't have it and uses magic jewels
        if not has_adorned and magic_jewel_count >= 3:
            dps_missing.append(SynergyContributor(
                name="The Adorned",
                source_type="missing",
                contribution=f"You have {magic_jewel_count} magic jewels — The Adorned would multiply ALL their mods by ~2.5x.",
                severity="warning",
                detail=UNIQUE_JEWELS["The Adorned"].impact,
                estimated_pct=magic_jewel_count * 5.0,
                estimated_value=total_dps * magic_jewel_count * 0.05 if total_dps > 0 else 0,
            ))

        # Suggest Megalomaniac if player doesn't have one
        if not has_megalomaniac and total_dps > 100000:
            dps_missing.append(SynergyContributor(
                name="Megalomaniac",
                source_type="missing",
                contribution="3 random notable passives — search for combos matching your build.",
                severity="info",
                detail=UNIQUE_JEWELS["Megalomaniac"].impact,
                estimated_pct=10.0,
                estimated_value=total_dps * 0.10 if total_dps > 0 else 0,
            ))

        # DPS keystones the player has
        DPS_KEYSTONES = {"Pain Attunement", "Elemental Overload", "Avatar of Fire",
                         "Crimson Power", "Grasping Wounds", "Point Blank",
                         "Iron Will", "Elemental Equilibrium", "Crimson Dance",
                         "Overwhelming Toxicity"}
        for ks in char_data.keystones:
            if ks in DPS_KEYSTONES:
                info = KEYSTONES.get(ks)
                # Estimate keystone DPS contribution
                est_pct = _estimate_keystone_dps_pct(ks)
                est_val = total_dps * est_pct / 100 if total_dps > 0 else 0
                dps_contributors.append(SynergyContributor(
                    name=ks, source_type="keystone",
                    contribution=f"~{est_pct:.0f}% DPS ({info.benefits.split('.')[0]})" if info else f"~{est_pct:.0f}% DPS",
                    severity="positive",
                    detail=info.description if info else "",
                    estimated_value=est_val,
                    estimated_pct=est_pct,
                ))

        # Popular rare mod comparison — show what mods top players use vs what player has
        if popular_rare_mods:
            for item in char_data.equipment:
                if item.rarity != "Rare" or item.slot in ("Flask", "Flask2"):
                    continue
                slot_mods = popular_rare_mods.get(item.slot, [])
                if not slot_mods:
                    continue
                # Check which popular mods the player is missing
                player_mods = set()
                for mod in (item.explicit_mods or []) + (item.implicit_mods or []):
                    clean = _strip_ninja_brackets(mod)
                    normalized = _re.sub(r"[\d,.]+", "#", clean).strip()
                    player_mods.add(normalized)

                missing_popular = []
                for mod_norm, pct in slot_mods[:5]:
                    if mod_norm not in player_mods and pct >= 30:
                        missing_popular.append(f"{mod_norm.replace('#', 'X')} ({pct:.0f}%)")

                if missing_popular:
                    item_name = item.name or item.type_line or item.slot
                    slot_name = item.slot.lower().rstrip("s") + "s" if not item.slot.lower().endswith("s") else item.slot.lower()
                    detail_lines = [f"Top players' rare {slot_name} commonly have:"]
                    for mod_norm, pct in slot_mods[:6]:
                        has_it = "✓" if mod_norm in player_mods else "✗"
                        detail_lines.append(f"  {has_it} {mod_norm.replace('#', 'X')} ({pct:.0f}%)")

                    dps_missing.append(SynergyContributor(
                        name=f"{item_name}: Missing Key Mods",
                        source_type="missing", slot=item.slot,
                        contribution=f"Missing: {', '.join(missing_popular[:3])}",
                        severity="warning",
                        detail="\n".join(detail_lines),
                        estimated_pct=len(missing_popular) * 5.0,
                        estimated_value=total_dps * len(missing_popular) * 0.05,
                    ))

        # Missing DPS keystones (skip ascendancy passives if player has them allocated)
        has_asc_points = getattr(char_data, 'ascendancy_points', 0) >= 4
        for pk in popular_keystones:
            if pk["name"] in player_ks or pk["percentage"] < 50:
                continue
            if pk.get("type") == "Ascendancy" and has_asc_points:
                continue
            if pk["name"] in DPS_KEYSTONES:
                info = KEYSTONES.get(pk["name"])
                est_pct = _estimate_keystone_dps_pct(pk["name"])
                est_val = total_dps * est_pct / 100 if total_dps > 0 else 0
                dps_missing.append(SynergyContributor(
                    name=pk["name"], source_type="missing",
                    contribution=f"+~{_format_number(est_val)} DPS (~{est_pct:.0f}%)" if est_val > 0 else (info.impact.split(".")[0] + "." if info else ""),
                    severity="critical" if pk["percentage"] > 80 else "warning",
                    adoption_pct=pk["percentage"],
                    detail=info.impact if info else "",
                    estimated_value=est_val,
                    estimated_pct=est_pct,
                ))

        # Missing gear upgrades — suggest +gem levels if player doesn't have max
        _add_gear_upgrade_suggestions(char_data, archetype, total_dps, dps_missing)

        # Sort contributors by estimated value
        dps_contributors.sort(key=lambda c: c.estimated_value, reverse=True)

        categories.append(SynergyCategory(
            category="dps", label="DPS", icon="bolt",
            value=sc.dps_label if sc else "?",
            status=sc.dps_status if sc else "info",
            contributors=dps_contributors,
            missing=dps_missing,
        ))

        # ── Survival Category ───────────────────────────
        surv_contributors = []
        surv_missing = []
        total_ehp = sc.ehp if sc else 0

        if pob:
            if pob.stats.life > 0:
                pct = (pob.stats.life / total_ehp * 100) if total_ehp > 0 else 0
                surv_contributors.append(SynergyContributor(
                    name="Life Pool", source_type="stat",
                    contribution=f"{pob.stats.life:,} life",
                    severity="positive" if pob.stats.life >= 3000 else "warning",
                    estimated_value=float(pob.stats.life),
                    estimated_pct=round(pct, 1),
                ))
            if pob.stats.energy_shield > 0:
                pct = (pob.stats.energy_shield / total_ehp * 100) if total_ehp > 0 else 0
                surv_contributors.append(SynergyContributor(
                    name="Energy Shield", source_type="stat",
                    contribution=f"{pob.stats.energy_shield:,} ES",
                    severity="positive" if pob.stats.energy_shield >= 3000 else "info",
                    estimated_value=float(pob.stats.energy_shield),
                    estimated_pct=round(pct, 1),
                ))
            if pob.stats.block_chance > 0:
                surv_contributors.append(SynergyContributor(
                    name="Block", source_type="stat",
                    contribution=f"{pob.stats.block_chance}% block chance",
                    severity="positive",
                ))

        # Gear mods contributing to survival (top items)
        for slot, item_name, mod_text, desc, est_pct in gear_surv_mods[:6]:
            est_val = total_ehp * est_pct / 100 if total_ehp > 0 else 0
            surv_contributors.append(SynergyContributor(
                name=f"{item_name}",
                source_type="gear", slot=slot,
                contribution=f"{desc} ({mod_text})",
                severity="positive",
                estimated_value=est_val,
                estimated_pct=est_pct,
            ))

        SURV_KEYSTONES = {"Sanguimancy", "Mind Over Matter", "Chaos Inoculation",
                          "Iron Reflexes", "Acrobatics", "Unwavering Stance",
                          "Ghost Reaver", "Eldritch Battery", "Blood Magic",
                          "Vitality Siphon"}
        for ks in char_data.keystones:
            if ks in SURV_KEYSTONES:
                info = KEYSTONES.get(ks)
                surv_contributors.append(SynergyContributor(
                    name=ks, source_type="keystone",
                    contribution=info.benefits.split(".")[0] + "." if info else "Survival keystone",
                    severity="positive",
                ))

        for pk in popular_keystones:
            if pk["name"] in player_ks or pk["percentage"] < 50:
                continue
            if pk.get("type") == "Ascendancy" and has_asc_points:
                continue
            if pk["name"] in SURV_KEYSTONES:
                info = KEYSTONES.get(pk["name"])
                est_pct = _estimate_keystone_ehp_pct(pk["name"])
                est_val = total_ehp * est_pct / 100 if total_ehp > 0 else 0
                surv_missing.append(SynergyContributor(
                    name=pk["name"], source_type="missing",
                    contribution=f"+~{_format_number(est_val)} EHP (~{est_pct:.0f}%)" if est_val > 0 else (info.impact.split(".")[0] + "." if info else ""),
                    severity="critical" if pk["percentage"] > 80 else "warning",
                    adoption_pct=pk["percentage"],
                    detail=info.impact if info else "",
                    estimated_value=est_val,
                    estimated_pct=est_pct,
                ))

        # ── Sustain sub-layer for survival ──────────────
        # Build-specific sustain: mana for MoM, life cost for Blood Mage, etc.
        if pob:
            mana = pob.stats.all_stats.get("Mana", 0)
            mana_regen = pob.stats.all_stats.get("ManaRegenRecovery", 0)
            if mana > 0 and ("Mind Over Matter" in player_ks or archetype.defense_type == "mom"):
                surv_contributors.append(SynergyContributor(
                    name="Mana (MoM Buffer)", source_type="stat",
                    contribution=f"{int(mana):,} mana ({int(mana_regen)}/s regen)",
                    severity="positive" if mana > 1000 else "warning",
                    detail=f"Mind Over Matter uses mana as a damage buffer. Your {int(mana):,} mana pool absorbs 40% of damage taken. Regen: {mana_regen:.0f}/s.",
                    estimated_value=mana * 0.4,
                    estimated_pct=round(mana * 0.4 / max(total_ehp, 1) * 100, 1),
                ))

            life_regen = pob.stats.all_stats.get("LifeRegenRecovery", 0)
            if life_regen > 0:
                surv_contributors.append(SynergyContributor(
                    name="Life Regeneration", source_type="stat",
                    contribution=f"{life_regen:.0f}/s life regen",
                    severity="positive" if life_regen > 100 else "info",
                ))

        # Sort survival by estimated value
        surv_contributors.sort(key=lambda c: c.estimated_value, reverse=True)

        categories.append(SynergyCategory(
            category="survival", label="Survival", icon="shield",
            value=f"{sc.ehp:,} EHP" if sc else "?",
            status=sc.ehp_status if sc else "info",
            contributors=surv_contributors,
            missing=surv_missing,
        ))

        # ── Clear Speed Category ────────────────────────
        clear_contributors = []
        clear_missing = []

        CLEAR_KEYSTONES = {"Sunder the Flesh", "Running Assault", "Relentless Pursuit",
                           "Wildsurge Incantation", "Path Seeker"}
        for ks in char_data.keystones:
            if ks in CLEAR_KEYSTONES:
                info = KEYSTONES.get(ks)
                clear_contributors.append(SynergyContributor(
                    name=ks, source_type="keystone",
                    contribution=info.benefits.split(".")[0] + "." if info else "Clear speed keystone",
                    severity="positive",
                    estimated_pct=15.0,
                ))

        # Speed stats from POB
        if pob:
            if pob.stats.movement_speed_mod > 0:
                speed_pct = int(pob.stats.movement_speed_mod * 100)
                clear_contributors.append(SynergyContributor(
                    name="Movement Speed", source_type="stat",
                    contribution=f"{speed_pct}% movement speed",
                    severity="positive" if speed_pct >= 130 else "info",
                ))
            cast_speed = pob.stats.speed
            if cast_speed > 0:
                clear_contributors.append(SynergyContributor(
                    name="Cast/Attack Speed", source_type="stat",
                    contribution=f"{cast_speed:.2f} casts/sec",
                    severity="positive",
                ))

        # Gear mods affecting clear speed (cast speed, attack speed, AoE, proj speed)
        for item in char_data.equipment:
            if item.slot in ("Flask", "Flask2"):
                continue
            all_mods = (item.explicit_mods or []) + (item.implicit_mods or [])
            item_label = item.name or item.type_line or item.slot
            for mod in all_mods:
                mc = _strip_ninja_brackets(mod).lower()
                m = _re.search(r"(\d+)% increased cast speed", mc)
                if m and archetype.damage_type == "spell":
                    val = int(m.group(1))
                    clear_contributors.append(SynergyContributor(
                        name=item_label, source_type="gear", slot=item.slot,
                        contribution=f"+{val}% cast speed",
                        severity="positive", estimated_pct=val * 0.3,
                    ))
                m = _re.search(r"(\d+)% increased attack speed", mc)
                if m and archetype.damage_type == "attack":
                    val = int(m.group(1))
                    clear_contributors.append(SynergyContributor(
                        name=item_label, source_type="gear", slot=item.slot,
                        contribution=f"+{val}% attack speed",
                        severity="positive", estimated_pct=val * 0.3,
                    ))
                if "area of effect" in mc or "increased area" in mc:
                    clear_contributors.append(SynergyContributor(
                        name=item_label, source_type="gear", slot=item.slot,
                        contribution="Increased AoE",
                        severity="positive",
                    ))

        # Skill gem level as a DPS AND clear speed factor
        main_gem_level = 0
        for sg in char_data.skill_groups:
            if hasattr(sg, "gems"):
                for g in sg.gems:
                    gname = g if isinstance(g, str) else getattr(g, "name", "")
                    if gname == archetype.main_skill:
                        # We don't have the actual gem level from char_data
                        # but we can infer from +gem level gear
                        main_gem_level = 20  # base
                        break

        # Count +gem levels from gear
        total_gem_bonus = 0
        for item in char_data.equipment:
            all_mods = (item.explicit_mods or []) + (item.implicit_mods or []) + (item.crafted_mods or [])
            for mod in all_mods:
                mc = _strip_ninja_brackets(mod).lower()
                m = _re.search(r"\+(\d+) to level of all .*(spell|skill)", mc)
                if m:
                    total_gem_bonus += int(m.group(1))

        if total_gem_bonus > 0:
            effective_level = 20 + total_gem_bonus
            clear_contributors.append(SynergyContributor(
                name=f"{archetype.main_skill} Gem Level",
                source_type="stat",
                contribution=f"Effective Lv{effective_level} (+{total_gem_bonus} from gear)",
                severity="positive",
                detail=f"Each gem level is ~10-12% more base damage. Your +{total_gem_bonus} from gear takes {archetype.main_skill} from Lv20 to Lv{effective_level}. Getting a Lv21 base gem (requires character Lv97+) would push this to Lv{effective_level + 1}.",
                estimated_pct=total_gem_bonus * 11.0,
            ))

        for pk in popular_keystones:
            if pk["name"] in player_ks or pk["percentage"] < 50:
                continue
            if pk.get("type") == "Ascendancy" and has_asc_points:
                continue
            if pk["name"] in CLEAR_KEYSTONES:
                info = KEYSTONES.get(pk["name"])
                clear_missing.append(SynergyContributor(
                    name=pk["name"], source_type="missing",
                    contribution=info.impact.split(".")[0] + "." if info else f"{pk['percentage']:.0f}% of builds use this",
                    severity="critical" if pk["percentage"] > 80 else "warning",
                    adoption_pct=pk["percentage"],
                    detail=info.impact if info else "",
                    estimated_pct=15.0,
                ))

        # Always show clear speed category
        categories.append(SynergyCategory(
            category="clear_speed", label="Clear Speed", icon="speed",
            value=f"Lv{20 + total_gem_bonus} {archetype.main_skill}" if total_gem_bonus > 0 else "—",
            status="info",
            contributors=clear_contributors,
            missing=clear_missing,
        ))

        return categories


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _worst_severity(items: List[Explanation]) -> str:
    """Return the most severe severity in a list."""
    sev_order = {"critical": 0, "warning": 1, "info": 2, "positive": 3}
    return min((e.severity for e in items), key=lambda s: sev_order.get(s, 9), default="info")

def _matches_archetype(synergy_text: str, archetype: BuildArchetype) -> bool:
    """Check if a synergy description is relevant to the archetype."""
    text = synergy_text.lower()
    if archetype.damage_type in text:
        return True
    if archetype.defense_type in text:
        return True
    for tag in archetype.tags:
        if tag.lower() in text:
            return True
    return False


def _format_number(n: float) -> str:
    """Format a large number for display: 1234567 -> 1.2M, 12345 -> 12.3K."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return f"{n:,.0f}"


def _add_gear_improvement_actions(
    char_data, archetype, pob, total_dps: float, actions: list,
):
    """Add gear-based improvement recommendations to the actions list."""
    if not total_dps:
        return

    # Analyze each slot for DPS and survival contribution
    slot_dps = {}  # slot -> total estimated DPS %
    slot_items = {}  # slot -> item

    for item in char_data.equipment:
        if item.slot in ("Flask", "Flask2"):
            continue
        all_mods = (item.explicit_mods or []) + (item.implicit_mods or []) + (item.crafted_mods or [])
        total_pct = 0
        slot_items[item.slot] = item
        for mod in all_mods:
            mc = _strip_ninja_brackets(mod)
            a = _analyze_mod_contribution(mc, archetype)
            if a and a[0] == "dps":
                total_pct += a[2]
        slot_dps[item.slot] = total_pct

    # ── Skill gem level upgrade ──────────────────────
    # Count total +gem levels from gear
    total_gem_bonus = 0
    for item in char_data.equipment:
        all_mods = (item.explicit_mods or []) + (item.implicit_mods or []) + (item.crafted_mods or [])
        for mod in all_mods:
            mc = _strip_ninja_brackets(mod).lower()
            m = _re.search(r"\+(\d+) to level of all .*(spell|skill)", mc)
            if m:
                total_gem_bonus += int(m.group(1))

    effective_gem_level = 20 + total_gem_bonus
    char_level = char_data.level or 0

    # Lv21 gem upgrade
    if char_level >= 97:
        est_dps_gain = total_dps * 0.11
        actions.append(Explanation(
            context="action",
            title=f"Lv21 {archetype.main_skill} Gem",
            text=(
                f"At character Lv{char_level}, you can use a Lv21 {archetype.main_skill} gem. "
                f"This takes your effective gem level from {effective_gem_level} to {effective_gem_level + 1}. "
                f"Each gem level is ~11% more base damage — estimated +{_format_number(est_dps_gain)} DPS. "
                f"Lv21 gems drop from corrupting Lv20 gems with a Vaal Orb (1-in-8 chance)."
            ),
            severity="warning",
            source="game_knowledge",
        ))
    elif char_level >= 90:
        levels_needed = 97 - char_level
        actions.append(Explanation(
            context="action",
            title=f"Level to 97 for Lv21 Gem",
            text=(
                f"You're Lv{char_level} — {levels_needed} levels from being able to use a Lv21 {archetype.main_skill} gem. "
                f"This would add ~11% more base damage (~{_format_number(total_dps * 0.11)} DPS). "
                f"One of the biggest single upgrades available."
            ),
            severity="info",
            source="game_knowledge",
        ))

    # ── Gem quality upgrade ──────────────────────────
    # Quality on skill gems typically gives 0.5-1% more damage per quality point
    # A 20% quality gem vs 0% quality is ~10-20% more DPS
    actions.append(Explanation(
        context="action",
        title=f"Quality {archetype.main_skill} Gem",
        text=(
            f"If your {archetype.main_skill} gem isn't 20% quality, use Gemcutter's Prisms to max it. "
            f"20% quality typically adds 10-20% more damage or AoE depending on the gem. "
            f"For Cast on Crit builds, also quality your trigger skill (attack gem) for crit chance."
        ),
        severity="info",
        source="game_knowledge",
    ))

    # ── Slots with low DPS contribution ──────────────
    # Identify gear slots that could have DPS mods but don't
    DPS_ELIGIBLE_SLOTS = {
        "spell": ["Weapon", "Weapon2", "Amulet", "Ring", "Ring2", "Helm", "Gloves"],
        "attack": ["Weapon", "Weapon2", "Amulet", "Ring", "Ring2", "Gloves", "Belt"],
    }
    eligible = DPS_ELIGIBLE_SLOTS.get(archetype.damage_type, [])

    for slot in eligible:
        if slot not in slot_dps:
            continue
        dps_pct = slot_dps[slot]
        item = slot_items.get(slot)
        if not item:
            continue

        # Skip uniques — can't easily change their mods
        if item.rarity == "Unique":
            continue

        item_name = item.name or item.type_line or slot

        if dps_pct < 3 and archetype.damage_type == "spell":
            # This slot has almost no DPS mods — suggest adding some
            suggestions = []
            if slot in ("Ring", "Ring2"):
                suggestions = ["+% spell damage", "+% cast speed", "+% cold/fire/lightning damage"]
            elif slot == "Helm":
                suggestions = ["+% spell damage", "+to gem levels", "+% crit chance"]
            elif slot == "Gloves":
                suggestions = ["+% cast speed", "+% spell damage", "added damage"]
            elif slot == "Belt":
                suggestions = ["+% damage", "+% elemental damage"]

            if suggestions:
                est_gain = total_dps * 0.08  # conservative estimate for adding DPS mods
                actions.append(Explanation(
                    context="action",
                    title=f"Add DPS Mods: {item_name} ({slot})",
                    text=(
                        f"Your {slot.lower()} ({item_name}) has almost no damage mods. "
                        f"For a {archetype.damage_type} build, look for: {', '.join(suggestions)}. "
                        f"Even mid-tier DPS mods here could add ~{_format_number(est_gain)} DPS."
                    ),
                    severity="warning",
                    source="game_knowledge",
                    slot=slot,
                ))

    # ── Survival gear upgrades ───────────────────────
    if pob and pob.stats.life < 3000 and archetype.defense_type == "life":
        # Find slots without life mods
        slots_without_life = []
        for item in char_data.equipment:
            if item.slot in ("Flask", "Flask2", "Weapon", "Weapon2"):
                continue
            all_mods = (item.explicit_mods or []) + (item.implicit_mods or []) + (item.crafted_mods or [])
            has_life = any("maximum life" in _strip_ninja_brackets(m).lower() for m in all_mods)
            if not has_life and item.rarity != "Unique":
                slots_without_life.append(item.slot)

        if slots_without_life:
            actions.append(Explanation(
                context="action",
                title=f"Add Life to {len(slots_without_life)} Slots",
                text=(
                    f"Your life pool ({pob.stats.life:,}) is low. "
                    f"Slots without +life mods: {', '.join(slots_without_life)}. "
                    f"Adding T2+ life (80+) to each would add ~{len(slots_without_life) * 80} life, "
                    f"significantly boosting your survivability."
                ),
                severity="critical" if pob.stats.life < 2000 else "warning",
                source="game_knowledge",
            ))

    # ── Weapon upgrade for spell builds ──────────────
    if archetype.damage_type == "spell":
        for wslot in ("Weapon", "Weapon2"):
            item = slot_items.get(wslot)
            if not item or item.rarity == "Unique":
                continue
            dpct = slot_dps.get(wslot, 0)
            if dpct < 40:
                # Weapon should be the highest DPS contributor for spell builds
                actions.append(Explanation(
                    context="action",
                    title=f"Upgrade {wslot}: {item.name or item.type_line}",
                    text=(
                        f"Your {wslot.lower()} contributes ~{dpct:.0f}% of DPS. "
                        f"For spell builds, weapons should provide +gem levels, % spell damage, "
                        f"% as extra elemental damage, and cast speed. "
                        f"A well-rolled weapon with +3 gem levels and 100%+ spell damage "
                        f"is typically the single biggest DPS upgrade."
                    ),
                    severity="warning",
                    source="game_knowledge",
                    slot=wslot,
                ))


def _damage_increase(current_resist: int) -> int:
    """Calculate how much more damage you take with uncapped resist vs 75%."""
    if current_resist >= 75:
        return 0
    baseline = 100 - 75
    actual = 100 - max(current_resist, -100)
    return int(((actual - baseline) / baseline) * 100)


import re as _re

_NINJA_BRACKET = _re.compile(r"\[([^|\]]*\|)?([^\]]*)\]")

def _strip_ninja_brackets(text: str) -> str:
    """Strip poe.ninja bracket formatting: [tag|display] -> display."""
    return _NINJA_BRACKET.sub(r"\2", text)


def _analyze_mod_contribution(mod_text: str, archetype) -> Optional[tuple]:
    """Analyze a mod for its DPS or survival contribution.
    Returns (category, description, estimated_pct) or None if not relevant.
    """
    ml = mod_text.lower()

    # +N to level of all spell/skill gems — massive DPS (~10-12% per level)
    m = _re.search(r"\+(\d+) to level of all .*(spell|skill)", ml)
    if m:
        levels = int(m.group(1))
        est = levels * 11  # ~11% per gem level
        return ("dps", f"+{levels} gem levels (~{est}% DPS)", est)

    # +N to level of specific skill gems
    m = _re.search(r"\+(\d+) to level of all .*(cold|fire|lightning|chaos|physical)", ml)
    if m:
        levels = int(m.group(1))
        est = levels * 8
        return ("dps", f"+{levels} element gem levels (~{est}% DPS)", est)

    # % increased spell damage
    m = _re.search(r"(\d+)% increased .*spell.*damage", ml)
    if m and archetype.damage_type == "spell":
        val = int(m.group(1))
        est = val * 0.15  # rough: 100% inc spell = ~15% more DPS (diminishing)
        return ("dps", f"+{val}% spell damage", round(est, 1))

    # Gain X% of damage as extra element
    m = _re.search(r"(\d+)% of damage as extra", ml)
    if m:
        val = int(m.group(1))
        est = val * 0.7  # extra damage conversion is very efficient
        return ("dps", f"+{val}% as extra elemental", round(est, 1))

    # % increased critical hit chance
    m = _re.search(r"(\d+)% increased .*critical.*hit.*chance", ml)
    if m and archetype.is_crit:
        val = int(m.group(1))
        est = val * 0.08  # crit chance is good but diminishing
        return ("dps", f"+{val}% crit chance", round(est, 1))

    # % increased cast speed — DPS for spell builds
    m = _re.search(r"(\d+)% increased cast speed", ml)
    if m and archetype.damage_type == "spell":
        val = int(m.group(1))
        est = val * 0.5
        return ("dps", f"+{val}% cast speed", round(est, 1))

    # % increased attack speed — DPS for attack builds AND Cast on Crit builds
    m = _re.search(r"(\d+)% increased (?:local )?attack speed", ml)
    if m and (archetype.damage_type == "attack" or archetype.is_coc):
        val = int(m.group(1))
        est = val * 0.6 if archetype.is_coc else val * 0.5  # CoC: more attacks = more triggers
        label = f"+{val}% attack speed (CoC trigger rate)" if archetype.is_coc else f"+{val}% attack speed"
        return ("dps", label, round(est, 1))

    # % increased elemental/cold/fire/lightning damage
    m = _re.search(r"(\d+)% increased (?:cold|fire|lightning|elemental) damage", ml)
    if m:
        val = int(m.group(1))
        est = val * 0.12
        return ("dps", f"+{val}% elemental damage", round(est, 1))

    # +N to maximum life
    m = _re.search(r"\+(\d+) to maximum life", ml)
    if m:
        val = int(m.group(1))
        if val >= 30:
            return ("survival", f"+{val} life", round(val / 50, 1))

    # +N to maximum energy shield / % increased energy shield
    m = _re.search(r"\+(\d+) to maximum energy shield", ml)
    if m:
        val = int(m.group(1))
        if val >= 30:
            return ("survival", f"+{val} ES", round(val / 50, 1))

    m = _re.search(r"(\d+)% increased energy shield", ml)
    if m:
        val = int(m.group(1))
        if val >= 20:
            return ("survival", f"+{val}% ES", round(val * 0.3, 1))

    # % to resistances
    m = _re.search(r"\+(\d+)% to all .*resist", ml)
    if m:
        val = int(m.group(1))
        return ("survival", f"+{val}% all res", round(val * 0.2, 1))

    return None


def _estimate_keystone_dps_pct(name: str) -> float:
    """Rough DPS contribution estimate for a keystone, based on game knowledge."""
    estimates = {
        "Pain Attunement": 30.0,
        "Elemental Overload": 25.0,
        "Crimson Power": 20.0,
        "Grasping Wounds": 15.0,
        "Avatar of Fire": 12.0,
        "Point Blank": 20.0,
        "Iron Will": 15.0,
        "Elemental Equilibrium": 18.0,
        "Crimson Dance": 25.0,
        "Overwhelming Toxicity": 30.0,
    }
    return estimates.get(name, 10.0)


def _estimate_keystone_ehp_pct(name: str) -> float:
    """Rough EHP contribution estimate for a keystone."""
    estimates = {
        "Sanguimancy": 80.0,  # enables low-life which transforms EHP
        "Mind Over Matter": 40.0,
        "Chaos Inoculation": 50.0,
        "Eldritch Battery": 25.0,
        "Vitality Siphon": 30.0,  # sustain = effective EHP
        "Iron Reflexes": 20.0,
        "Acrobatics": 15.0,
        "Unwavering Stance": 10.0,
        "Ghost Reaver": 20.0,
        "Blood Magic": 15.0,
    }
    return estimates.get(name, 10.0)


def _add_gear_upgrade_suggestions(
    char_data, archetype, total_dps: float, missing: list,
):
    """Add gear upgrade suggestions based on what could improve DPS."""
    # Check for +gem level opportunities
    total_gem_levels = 0
    has_amulet_gem = False
    for item in char_data.equipment:
        all_mods = (item.explicit_mods or []) + (item.implicit_mods or []) + (item.crafted_mods or [])
        for mod in all_mods:
            ml = _strip_ninja_brackets(mod).lower()
            m = _re.search(r"\+(\d+) to level of all .*(spell|skill)", ml)
            if m:
                total_gem_levels += int(m.group(1))
                if item.slot == "Amulet":
                    has_amulet_gem = True

    # Suggest +gem level upgrade if player has low total
    if total_gem_levels < 5 and archetype.damage_type == "spell":
        est_per_level = total_dps * 0.11 if total_dps > 0 else 0
        missing.append(SynergyContributor(
            name="+Gem Level Gear",
            source_type="missing",
            contribution=f"+~{_format_number(est_per_level * 3)} DPS — get +gem levels on weapon/amulet",
            severity="warning",
            detail=(
                f"You have +{total_gem_levels} total gem levels from gear. "
                f"Top players often have +8 to +10. Each gem level is ~11% more DPS. "
                f"Look for weapons with '+N to level of all Spell Skills' or amulets with '+N to all Skill Gems'."
            ),
            estimated_value=est_per_level * 3,
            estimated_pct=33.0,
        ))

    if not has_amulet_gem and archetype.damage_type == "spell":
        est = total_dps * 0.22 if total_dps > 0 else 0
        missing.append(SynergyContributor(
            name="Amulet: +Gem Levels",
            source_type="missing",
            contribution=f"+~{_format_number(est)} DPS — amulet with +2-3 gem levels",
            severity="warning",
            detail=(
                "Your amulet doesn't have +gem levels. A +2 or +3 amulet is one of the biggest "
                "single-slot DPS upgrades for spell builds. Also consider corrupting for +1 gem levels."
            ),
            estimated_value=est,
            estimated_pct=22.0,
        ))

    # Lv21 gem upgrade suggestion
    char_level = char_data.level or 0
    if char_level >= 97 and total_dps > 0:
        est = total_dps * 0.11
        missing.append(SynergyContributor(
            name=f"Lv21 {archetype.main_skill} Gem",
            source_type="missing",
            contribution=f"+~{_format_number(est)} DPS — Lv21 gem (+1 base level)",
            severity="warning",
            detail=(
                f"At Lv{char_level}, you can equip a Lv21 gem. "
                f"Corrupt a Lv20 gem with a Vaal Orb for a 1-in-8 chance at Lv21. "
                f"Each gem level is ~11% more base damage."
            ),
            estimated_value=est,
            estimated_pct=11.0,
        ))

    # Slots with no DPS contribution that could have some
    DPS_ELIGIBLE = {"spell": ["Ring", "Ring2"], "attack": ["Ring", "Ring2", "Belt"]}
    eligible = DPS_ELIGIBLE.get(archetype.damage_type, [])
    for slot in eligible:
        item = next((eq for eq in char_data.equipment if eq.slot == slot), None)
        if not item or item.rarity == "Unique":
            continue
        all_mods = (item.explicit_mods or []) + (item.implicit_mods or []) + (item.crafted_mods or [])
        has_dps = False
        for mod in all_mods:
            mc = _strip_ninja_brackets(mod).lower()
            if any(kw in mc for kw in ["spell damage", "cast speed", "attack speed", "critical", "added", "gem"]):
                has_dps = True
                break
        if not has_dps and total_dps > 0:
            est = total_dps * 0.08
            missing.append(SynergyContributor(
                name=f"{item.name or item.type_line} ({slot})",
                source_type="missing",
                contribution=f"+~{_format_number(est)} DPS — add damage mods to {slot.lower()}",
                severity="info",
                detail=f"Your {slot.lower()} has no DPS mods. Adding spell damage, cast speed, or crit could add ~8% DPS.",
                estimated_value=est,
                estimated_pct=8.0,
            ))
