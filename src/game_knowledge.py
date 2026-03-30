"""
Static game knowledge database for Path of Exile 2.

Used by the why-engine to explain WHY players make specific build choices.
Maps game mechanics to plain-language explanations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class KeystoneInfo:
    description: str
    benefits: str
    synergies: List[str]
    anti_synergies: List[str]
    build_types: List[str]


@dataclass
class DefenseMechanicInfo:
    description: str
    how_it_works: str
    strengths: List[str]
    weaknesses: List[str]


@dataclass
class ModSynergyInfo:
    mods: List[str]
    explanation: str
    build_types: List[str]


@dataclass
class StatThreshold:
    stat: str
    mapping_min: Optional[float] = None
    endgame_min: Optional[float] = None
    boss_ready: Optional[float] = None
    tanky: Optional[float] = None
    description: str = ""


# ---------------------------------------------------------------------------
# KEYSTONES
# ---------------------------------------------------------------------------

KEYSTONES: Dict[str, KeystoneInfo] = {

    "Blood Magic": KeystoneInfo(
        description="Removes all mana. Skills cost life instead of mana.",
        benefits=(
            "Eliminates mana problems entirely. Lets you invest fully into "
            "life, resulting in a massive life pool since mana nodes and gear "
            "become irrelevant."
        ),
        synergies=["Life stacking gear", "Life regeneration", "Life leech",
                   "Life flask sustain", "Strength stacking"],
        anti_synergies=["Energy Shield builds", "Mind Over Matter",
                        "Mana-based auras", "Archmage"],
        build_types=["life_stacking", "attack_physical", "melee"],
    ),

    "Pain Attunement": KeystoneInfo(
        description="30% more spell damage while on low life (below 50% life).",
        benefits=(
            "Massive damage multiplier for spell builds. Achieved by "
            "reserving life with auras or using Petrified Blood to stay "
            "at low life permanently."
        ),
        synergies=["Sanguimancy", "Energy Shield gear", "Life reservation",
                   "Petrified Blood", "Low-life builds"],
        anti_synergies=["Life stacking", "Blood Magic (unless reserving life)",
                        "Builds without ES backup"],
        build_types=["spell_damage", "low_life", "energy_shield"],
    ),

    "Sanguimancy": KeystoneInfo(
        description="Life costs are paid from Energy Shield instead of Life.",
        benefits=(
            "Enables low-life builds safely — you can reserve life for "
            "Pain Attunement without dying to skill costs. Your ES absorbs "
            "the skill costs while your life stays reserved."
        ),
        synergies=["Pain Attunement", "Energy Shield gear",
                   "Life reservation", "ES recharge"],
        anti_synergies=["Chaos Inoculation", "Builds with no ES",
                        "Blood Magic"],
        build_types=["low_life", "energy_shield", "spell_damage"],
    ),

    "Elemental Overload": KeystoneInfo(
        description=(
            "Your hits can't critically strike. If you've dealt a critical "
            "strike recently, 40% more elemental damage."
        ),
        benefits=(
            "Huge damage boost for builds that invest minimally in crit. "
            "You only need enough crit chance to proc it once every 4 seconds, "
            "then you get a free 40% more multiplier."
        ),
        synergies=["Elemental damage skills", "Fast-hitting skills",
                   "Orb of Storms (for easy crit procs)", "Minimal crit gear"],
        anti_synergies=["Crit multi stacking", "Crit chance stacking",
                        "Assassin ascendancy", "Brittle"],
        build_types=["elemental_non_crit", "spell_damage", "elemental_attack"],
    ),

    "Avatar of Fire": KeystoneInfo(
        description=(
            "50% of physical, lightning, and cold damage is converted to fire. "
            "You can only deal fire damage."
        ),
        benefits=(
            "Converts all your damage to fire, letting you scale everything "
            "with fire damage modifiers. Simplifies gear and passive choices "
            "to a single element."
        ),
        synergies=["Fire penetration", "Fire damage gear",
                   "Conversion gloves/weapons", "Elemental Equilibrium (from allies)"],
        anti_synergies=["Chaos damage", "Non-fire elemental builds",
                        "Poison", "Bleed (physical)"],
        build_types=["fire_damage", "conversion", "elemental_attack"],
    ),

    "Chaos Inoculation": KeystoneInfo(
        description="Maximum life becomes 1. You are immune to chaos damage.",
        benefits=(
            "Complete chaos immunity removes an entire damage type from the "
            "game. You rely entirely on Energy Shield as your health pool, "
            "freeing you from needing any chaos resistance."
        ),
        synergies=["Energy Shield gear", "ES recharge", "Ghost Reaver",
                   "ES leech", "Discipline aura", "Intelligence stacking"],
        anti_synergies=["Life stacking", "Life leech", "Life flasks",
                        "Blood Magic", "Mind Over Matter", "Pain Attunement"],
        build_types=["energy_shield", "chaos_immune", "intelligence_stacking"],
    ),

    "Mind Over Matter": KeystoneInfo(
        description="40% of damage taken from hits is deducted from mana before life.",
        benefits=(
            "Effectively adds 40% of your mana pool as extra EHP. Great for "
            "builds that naturally have a large mana pool and good mana "
            "recovery."
        ),
        synergies=["Large mana pool", "Mana regeneration", "Clarity aura",
                   "Mana flask", "Mana on hit"],
        anti_synergies=["Blood Magic", "Low mana builds",
                        "Heavy mana reservation", "Energy Shield focus"],
        build_types=["mana_stacking", "hybrid_life_mana", "spell_damage"],
    ),

    "Ancestral Bond": KeystoneInfo(
        description="You can't deal damage directly. +1 to maximum number of summoned totems.",
        benefits=(
            "Extra totem lets totem builds scale their clear and single-target. "
            "The 'no damage' downside is irrelevant because totems deal "
            "the damage for you."
        ),
        synergies=["Totem skills", "Totem placement speed", "Totem life",
                   "Multiple totems support"],
        anti_synergies=["Self-cast builds", "Attack builds", "Minion builds",
                        "Any non-totem playstyle"],
        build_types=["totem"],
    ),

    "Iron Reflexes": KeystoneInfo(
        description="Converts all evasion rating to armour rating.",
        benefits=(
            "Lets you stack armour from both armour and evasion sources. "
            "Dexterity-based gear with high evasion becomes armour, giving "
            "huge physical damage reduction."
        ),
        synergies=["Armour stacking", "Grace aura (becomes armour)",
                   "Determination aura", "Molten Shell", "Granite Flask"],
        anti_synergies=["Evasion builds", "Acrobatics", "Dodge chance",
                        "Blind synergies"],
        build_types=["armour_stacking", "physical_mitigation", "melee"],
    ),

    "Acrobatics": KeystoneInfo(
        description="Grants dodge chance but reduces armour and energy shield.",
        benefits=(
            "Strong layer of avoidance for evasion-based characters. Dodge "
            "is checked independently of evasion, giving two chances to avoid "
            "hits entirely."
        ),
        synergies=["Evasion gear", "Grace aura", "Blind", "Jade Flask",
                   "Dexterity stacking"],
        anti_synergies=["Iron Reflexes", "Armour stacking",
                        "Energy Shield builds", "Determination aura"],
        build_types=["evasion", "dodge", "ranged_attack"],
    ),

    "Ghost Reaver": KeystoneInfo(
        description="Life leech applies to Energy Shield instead of life.",
        benefits=(
            "Gives ES builds access to leech-based sustain, which is normally "
            "life-only. Essential for attack-based ES characters who need "
            "instant recovery during fights."
        ),
        synergies=["Energy Shield gear", "Attack builds", "Life leech sources",
                   "Chaos Inoculation", "Vaal Pact"],
        anti_synergies=["Life builds", "ES recharge builds (leech stops recharge)",
                        "Spell-only builds without leech"],
        build_types=["energy_shield", "attack_es"],
    ),

    "Resolute Technique": KeystoneInfo(
        description="Your hits always connect (100% hit chance). You can never critically strike.",
        benefits=(
            "Eliminates accuracy as a stat requirement entirely. Perfect "
            "for builds that don't invest in crit and just want reliable, "
            "consistent damage."
        ),
        synergies=["Non-crit attack builds", "Elemental Overload (conflict — "
                   "can't crit to proc)", "Strength-based melee",
                   "Bleed builds", "Ignite via other means"],
        anti_synergies=["Critical strike builds", "Elemental Overload",
                        "Assassin ascendancy", "Brittle"],
        build_types=["non_crit_attack", "melee", "physical_attack"],
    ),

    "Unwavering Stance": KeystoneInfo(
        description="Cannot be stunned. Cannot evade attacks.",
        benefits=(
            "Stun immunity is critical for builds that channel or have long "
            "cast animations. The 'cannot evade' downside is irrelevant "
            "for armour-based characters."
        ),
        synergies=["Armour stacking", "Iron Reflexes", "Channelling skills",
                   "Determination aura"],
        anti_synergies=["Evasion builds", "Acrobatics", "Dodge builds",
                        "Grace aura"],
        build_types=["armour_stacking", "melee", "channelling"],
    ),

    "Crimson Dance": KeystoneInfo(
        description="Bleeding you inflict can stack up to 8 times on an enemy.",
        benefits=(
            "Massively increases bleed DPS by allowing stacking. Normally "
            "only the strongest bleed counts — with Crimson Dance, fast "
            "attacks apply many bleeds simultaneously."
        ),
        synergies=["Bleed chance", "Physical damage", "Attack speed",
                   "Multistrike", "Lacerate", "Puncture"],
        anti_synergies=["Elemental builds", "Spell builds",
                        "Slow-hitting builds (less stacking)"],
        build_types=["bleed", "physical_attack", "melee"],
    ),

    "Point Blank": KeystoneInfo(
        description=(
            "Projectiles deal up to 30% more damage to close targets, "
            "scaling down to 30% less damage at far range."
        ),
        benefits=(
            "Huge damage boost for close-range projectile builds. "
            "Perfect for bow or wand characters who fight at melee range "
            "or use shotgunning skills."
        ),
        synergies=["Close-range projectile skills", "Barrage", "Rain of Arrows",
                   "Short-range wand skills", "Shotgunning mechanics"],
        anti_synergies=["Long-range playstyles", "Off-screen clearing",
                        "Artillery-style skills"],
        build_types=["projectile_close_range", "bow_attack", "wand_attack"],
    ),

    "Iron Will": KeystoneInfo(
        description=(
            "Strength's damage bonus applies to spell damage instead of "
            "only melee physical damage."
        ),
        benefits=(
            "Lets strength-stacking characters scale spell damage. "
            "Every point of strength gives both life (survivability) and "
            "spell damage (offense), making gearing very efficient."
        ),
        synergies=["Strength stacking", "Hybrid spell/melee gear",
                   "Battlemage", "Shaper of Storms"],
        anti_synergies=["Dexterity stacking", "Intelligence stacking",
                        "Pure caster builds that don't stack STR"],
        build_types=["strength_stacking", "spell_damage", "hybrid_melee_caster"],
    ),
}


# ---------------------------------------------------------------------------
# DEFENSE_MECHANICS
# ---------------------------------------------------------------------------

DEFENSE_MECHANICS: Dict[str, DefenseMechanicInfo] = {

    "resist_caps": DefenseMechanicInfo(
        description=(
            "Elemental resistances cap at 75% by default. Each point of "
            "resistance reduces elemental damage taken by that percentage."
        ),
        how_it_works=(
            "Capping resists (75% fire/cold/lightning) is the #1 defensive "
            "priority. Overcapping by 15-30% protects against curse maps "
            "and Elemental Weakness which lower your resists temporarily."
        ),
        strengths=["Massive damage reduction", "Required for all builds",
                   "Relatively easy to cap with gear"],
        weaknesses=["Curses can lower resists below cap",
                    "Does not protect against physical or chaos damage"],
    ),

    "ehp": DefenseMechanicInfo(
        description=(
            "Effective Health Pool (EHP) is the total damage you can take "
            "before dying, factoring in life, ES, armour, resists, and all "
            "mitigation layers."
        ),
        how_it_works=(
            "Raw life/ES alone doesn't tell the full story. A character "
            "with 4000 life and 50% physical reduction has 8000 EHP vs "
            "physical. Multiple layers multiply together for much higher "
            "effective survivability."
        ),
        strengths=["Best single metric for tankiness",
                   "Accounts for all mitigation"],
        weaknesses=["Complex to calculate", "Varies by damage type",
                    "Doesn't account for recovery or avoidance"],
    ),

    "armour": DefenseMechanicInfo(
        description=(
            "Armour reduces physical damage taken from hits. It is more "
            "effective against many small hits than a few large ones."
        ),
        how_it_works=(
            "Armour uses a formula where its effectiveness scales inversely "
            "with hit size. Against small hits it can mitigate 80%+, but "
            "against huge slam attacks it might only reduce 20-30%. "
            "Endurance charges and other flat phys reduction help vs big hits."
        ),
        strengths=["Excellent vs many small hits", "Scales with flasks",
                   "Molten Shell scales off armour"],
        weaknesses=["Weak vs large single hits", "No effect on elemental damage",
                    "No effect on damage over time"],
    ),

    "evasion": DefenseMechanicInfo(
        description=(
            "Evasion gives a chance to entirely avoid attack hits. "
            "Uses an entropy-based system so it's not purely random."
        ),
        how_it_works=(
            "The entropy system ensures that if you have 50% evade chance, "
            "you will evade exactly every other hit — no unlucky streaks. "
            "Only works against attacks, not spells. Blind greatly boosts "
            "evasion effectiveness."
        ),
        strengths=["Avoids damage completely when it procs",
                   "Entropy system prevents bad luck streaks",
                   "Grace aura provides large flat evasion"],
        weaknesses=["Does not work against spells",
                    "Doesn't prevent damage over time",
                    "Big hits still hurt when they land"],
    ),

    "energy_shield": DefenseMechanicInfo(
        description=(
            "Energy Shield acts as a secondary health pool that sits on "
            "top of life. It recharges after not taking damage for a short "
            "period. By default, chaos damage bypasses ES."
        ),
        how_it_works=(
            "ES starts recharging after a delay (typically 2 seconds of no "
            "damage taken). It can be very large with the right gear — "
            "CI builds can have 8000+ ES. Chaos damage bypasses ES unless "
            "you have CI or specific mods."
        ),
        strengths=["Can reach very high values",
                   "Recharges without flasks",
                   "CI makes you chaos immune"],
        weaknesses=["Chaos damage bypasses it by default",
                    "Recharge delay can be deadly",
                    "Stuns interrupt recharge"],
    ),

    "block": DefenseMechanicInfo(
        description=(
            "Block gives a chance to take zero damage from a hit. "
            "Applies to attacks and can apply to spells with specific mods."
        ),
        how_it_works=(
            "When you block, the hit deals zero damage. Block chance caps "
            "at 75%. Spell block is separate and usually lower. Shields "
            "are the primary source of block chance."
        ),
        strengths=["Negates damage entirely on proc",
                   "Works against both physical and elemental",
                   "Shield-based builds can reach 75% easily"],
        weaknesses=["Probabilistic — bad luck can kill",
                    "Spell block harder to cap",
                    "Requires a shield (no two-handed weapons)"],
    ),

    "spell_suppression": DefenseMechanicInfo(
        description=(
            "Spell Suppression gives a chance to halve incoming spell "
            "damage. At 100% suppression chance, all spell hits are halved."
        ),
        how_it_works=(
            "When suppression triggers, the spell hit deals 50% reduced "
            "damage. At 100% chance, this is a consistent 50% less spell "
            "damage taken — extremely powerful. Primarily found on "
            "evasion-based gear."
        ),
        strengths=["50% spell damage reduction at cap",
                   "Consistent and reliable at 100%",
                   "Naturally available on evasion gear"],
        weaknesses=["Only affects spells, not attacks",
                    "Hard to cap without evasion gear",
                    "Below 100% it's unreliable"],
    ),

    "deflection": DefenseMechanicInfo(
        description=(
            "Deflection is a POE2 defense mechanic that provides damage "
            "reduction against deflected hits, scaling with shield defenses."
        ),
        how_it_works=(
            "When wielding a shield, deflection provides a baseline damage "
            "reduction layer. It scales with the shield's defensive stats "
            "and is checked separately from block, giving shield users "
            "an additional mitigation layer."
        ),
        strengths=["Additional layer for shield users",
                   "Scales with shield investment",
                   "Works alongside block"],
        weaknesses=["Requires a shield",
                    "Less effective without shield investment",
                    "Does not apply to unshielded characters"],
    ),
}


# ---------------------------------------------------------------------------
# MOD_SYNERGIES
# ---------------------------------------------------------------------------

MOD_SYNERGIES: List[ModSynergyInfo] = [

    ModSynergyInfo(
        mods=["flat physical damage", "% increased physical damage"],
        explanation=(
            "Flat phys and percent phys scale multiplicatively. Flat adds "
            "the base, percent scales it up. Having both is far stronger "
            "than stacking just one."
        ),
        build_types=["physical_attack", "melee", "bow_physical"],
    ),

    ModSynergyInfo(
        mods=["maximum life", "life regeneration"],
        explanation=(
            "Raw life and life regen together create passive sustain. "
            "Higher max life also makes % regen more effective since "
            "it's usually a percentage of max life."
        ),
        build_types=["life_stacking", "melee", "RF"],
    ),

    ModSynergyInfo(
        mods=["critical strike chance", "critical strike multiplier"],
        explanation=(
            "Crit chance and crit multi are a scaling pair — chance is "
            "worthless without multi, and multi is worthless without "
            "chance. You need both to make crit builds work."
        ),
        build_types=["crit_attack", "crit_spell", "assassin"],
    ),

    ModSynergyInfo(
        mods=["+level to spell skill gems", "spell damage"],
        explanation=(
            "Gem levels scale the base damage of spells, while spell "
            "damage % multiplies the result. Together they compound "
            "heavily — +1 gem level can be 10-15% more base damage."
        ),
        build_types=["spell_damage", "caster"],
    ),

    ModSynergyInfo(
        mods=["attack speed", "flat added damage"],
        explanation=(
            "Attack speed multiplies how often you hit, and flat damage "
            "is added to every hit. More hits per second means more "
            "value from every point of flat damage."
        ),
        build_types=["attack_speed", "elemental_attack", "physical_attack"],
    ),

    ModSynergyInfo(
        mods=["energy shield", "ES recharge rate"],
        explanation=(
            "Higher ES pool means more to recharge, and faster recharge "
            "rate means less downtime after taking damage. Together they "
            "make ES recovery feel seamless."
        ),
        build_types=["energy_shield", "CI", "low_life"],
    ),

    ModSynergyInfo(
        mods=["elemental resistances", "chaos resistance"],
        explanation=(
            "Capping all elemental resists is mandatory. Adding chaos "
            "resist on the same gear slot is highly efficient because "
            "chaos damage is common in endgame and hard to mitigate "
            "otherwise."
        ),
        build_types=["all_builds"],
    ),
]


# ---------------------------------------------------------------------------
# STAT_THRESHOLDS
# ---------------------------------------------------------------------------

STAT_THRESHOLDS: Dict[str, StatThreshold] = {

    "life": StatThreshold(
        stat="life",
        mapping_min=3000,
        endgame_min=4000,
        boss_ready=5000,
        tanky=6500,
        description=(
            "3000+ life is the minimum for comfortable mapping. "
            "5000+ is expected for endgame bosses. Below 3000 you will "
            "get one-shot regularly."
        ),
    ),

    "energy_shield": StatThreshold(
        stat="energy_shield",
        mapping_min=3000,
        endgame_min=4000,
        boss_ready=6000,
        tanky=8000,
        description=(
            "CI builds need at least 4000 ES to feel safe. 6000+ for "
            "bosses. Hybrid builds can get by with less since life backs "
            "them up."
        ),
    ),

    "ehp": StatThreshold(
        stat="ehp",
        mapping_min=10000,
        endgame_min=15000,
        boss_ready=30000,
        tanky=50000,
        description=(
            "EHP accounts for all mitigation. 15K+ is comfortable for "
            "general mapping, 30K+ is tanky, and 50K+ is very hard to "
            "kill. Anything below 10K will feel fragile."
        ),
    ),

    "elemental_resists": StatThreshold(
        stat="elemental_resists",
        mapping_min=75,
        endgame_min=75,
        boss_ready=75,
        tanky=75,
        description=(
            "All elemental resists must be capped at 75%. This is not "
            "optional — uncapped resists mean you take 2-4x more "
            "elemental damage. Overcap by 15-30% for curse protection."
        ),
    ),

    "chaos_resist": StatThreshold(
        stat="chaos_resist",
        mapping_min=0,
        endgame_min=20,
        boss_ready=40,
        tanky=75,
        description=(
            "Chaos resist is often neglected but matters in endgame. "
            "Positive chaos resist (above 0%) is good, 40%+ is solid, "
            "and 75% (capped) is ideal but hard to achieve."
        ),
    ),

    "dps": StatThreshold(
        stat="dps",
        mapping_min=100000,
        endgame_min=500000,
        boss_ready=1000000,
        tanky=None,
        description=(
            "DPS varies enormously by skill and build. Roughly: 100K+ "
            "clears maps, 500K+ handles endgame content, 1M+ melts "
            "bosses. These are very approximate — some skills have "
            "higher effective DPS than tooltip shows."
        ),
    ),

    "armour": StatThreshold(
        stat="armour",
        mapping_min=5000,
        endgame_min=15000,
        boss_ready=30000,
        tanky=50000,
        description=(
            "5000 armour helps with trash mobs. 15K+ starts feeling "
            "tanky. 30K+ with flasks up gives strong physical mitigation. "
            "50K+ and you barely feel physical hits."
        ),
    ),

    "evasion": StatThreshold(
        stat="evasion",
        mapping_min=5000,
        endgame_min=15000,
        boss_ready=25000,
        tanky=40000,
        description=(
            "Evasion builds want 15K+ for reliable avoidance. Pair with "
            "spell suppression and some life to avoid getting one-shot "
            "by the hits that land."
        ),
    ),
}
