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
class CharacterExplanations:
    """All explanations for a character."""
    keystones: List[Explanation] = field(default_factory=list)
    gear: Dict[str, List[Explanation]] = field(default_factory=dict)  # slot -> explanations
    stats: List[Explanation] = field(default_factory=list)
    actions: List[Explanation] = field(default_factory=list)
    meta: List[Explanation] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for API response."""
        return {
            "keystones": [_exp_dict(e) for e in self.keystones],
            "gear": {slot: [_exp_dict(e) for e in exps] for slot, exps in self.gear.items()},
            "stats": [_exp_dict(e) for e in self.stats],
            "actions": [_exp_dict(e) for e in self.actions],
            "meta": [_exp_dict(e) for e in self.meta],
        }


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

        if char_class and skill:
            profile = self._client.fetch_archetype_profile(char_class, skill)
            popular_keystones = self._client.fetch_popular_keystones(char_class, skill)

        # Generate explanations by category
        result.stats = self._explain_stats(pob, profile, char_data)
        result.keystones = self._explain_keystones(
            char_data.keystones, popular_keystones, archetype, pob, profile
        )
        result.actions = self._generate_actions(
            char_data, archetype, pob, profile, popular_keystones
        )
        result.gear = self._explain_gear(char_data, archetype, profile)
        result.meta = self._explain_meta(archetype, profile)

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

        # Armour
        if stats.armour > 0:
            a_range = ranges.get("armour", {})
            explanations.append(self._stat_position_explanation(
                "Armour", stats.armour, a_range,
                desc=DEFENSE_MECHANICS.get("armour", None),
            ))

        # Evasion
        if stats.evasion > 0:
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
    ) -> List[Explanation]:
        explanations = []
        player_ks_set = set(player_keystones)

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
        for pk in popular_keystones:
            if pk["name"] in player_ks_set:
                continue
            if pk["percentage"] < 20:
                continue  # Only flag widely-used keystones

            info = KEYSTONES.get(pk["name"])
            pct = pk["percentage"]

            if info:
                # Lead with what the player is LOSING by not having this
                text = info.impact
                text += f" ({pct:.0f}% of this build uses {pk['name']}.)"
            else:
                # No game knowledge — infer impact from adoption rate
                if pct >= 95:
                    text = (
                        f"{pk['name']} is used by {pct:.0f}% of players in this build — "
                        f"it's essentially mandatory. Without it, you're at a significant "
                        f"disadvantage in either damage or survivability compared to "
                        f"virtually every other player running this build."
                    )
                elif pct >= 70:
                    text = (
                        f"{pk['name']} is used by {pct:.0f}% of players in this build. "
                        f"The high adoption rate suggests it provides a major damage or "
                        f"survivability boost that most players consider essential for "
                        f"this archetype."
                    )
                else:
                    text = (
                        f"{pk['name']} is used by {pct:.0f}% of players in this build. "
                        f"It's a popular but not universal choice — it likely provides "
                        f"a meaningful boost that complements this archetype."
                    )

            severity = "critical" if pct > 70 else "warning"

            explanations.append(Explanation(
                context="keystone", title=f"Missing: {pk['name']}",
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

        # Uncapped resists action
        if pob:
            for resist_name, val in [
                ("Fire", pob.stats.fire_resist),
                ("Cold", pob.stats.cold_resist),
                ("Lightning", pob.stats.lightning_resist),
            ]:
                if val < 75:
                    gap = 75 - val
                    actions.append(Explanation(
                        context="action", title=f"Cap {resist_name} Resistance",
                        text=(
                            f"Your {resist_name} resistance is {val}% — {gap}% below the 75% cap. "
                            f"Uncapped resists mean you take {_damage_increase(val)}% more {resist_name.lower()} damage than intended. "
                            f"Look for {resist_name.lower()} res on rings, belt, or gloves. "
                            f"A single T2 {resist_name.lower()} res mod (35%+) would close most of this gap."
                        ),
                        severity="critical",
                        source="game_knowledge",
                    ))

            # Chaos resist warning
            if pob.stats.chaos_resist < 0:
                actions.append(Explanation(
                    context="action", title="Fix Chaos Resistance",
                    text=(
                        f"Your chaos resistance is negative ({pob.stats.chaos_resist}%). "
                        f"Chaos damage bypasses Energy Shield by default. "
                        f"Prioritize chaos res on rings or amulet — even getting to 0% is a significant survivability boost."
                    ),
                    severity="critical",
                    source="game_knowledge",
                ))

        # Missing high-adoption keystones — lead with impact
        player_ks = set(char_data.keystones)
        for pk in popular_keystones:
            if pk["name"] in player_ks or pk["percentage"] < 70:
                continue
            info = KEYSTONES.get(pk["name"])
            pct = pk["percentage"]
            if info:
                text = f"{info.impact} ({pct:.0f}% of this build uses it.)"
            else:
                text = (
                    f"{pk['name']} is used by {pct:.0f}% of this build — "
                    f"without it, you're at a significant disadvantage in either "
                    f"damage or survivability."
                )
            actions.append(Explanation(
                context="action",
                title=f"Allocate {pk['name']}",
                text=text,
                severity="critical" if pct > 90 else "warning",
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _damage_increase(current_resist: int) -> int:
    """Calculate how much more damage you take with uncapped resist vs 75%."""
    # At 75% resist you take 25% of damage
    # At current_resist you take (100 - current_resist)% of damage
    if current_resist >= 75:
        return 0
    baseline = 100 - 75  # 25% damage taken at cap
    actual = 100 - max(current_resist, -100)
    return int(((actual - baseline) / baseline) * 100)
