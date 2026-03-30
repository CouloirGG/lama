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
    impact: str  # What the player LOSES without this — the "why you'll struggle" line
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

    # ------------------------------------------------------------------
    # Core passive tree keystones
    # ------------------------------------------------------------------

    "Blood Magic": KeystoneInfo(
        description="Removes all mana. Skills cost life instead of mana.",
        benefits="Eliminates mana problems entirely. Lets you invest fully into life since mana nodes and gear become irrelevant.",
        impact="Without Blood Magic, you need to manage mana as a resource — mana gear, mana flask, and mana regen all compete with damage and survival stats on your gear.",
        synergies=["Life stacking gear", "Life regeneration", "Strength stacking"],
        anti_synergies=["Energy Shield builds", "Mind Over Matter", "Mana-based auras"],
        build_types=["life_stacking", "attack_physical", "melee"],
    ),

    "Pain Attunement": KeystoneInfo(
        description="30% more spell damage while on low life (below 50% life).",
        benefits="Massive 30% MORE damage multiplier for spell builds. Achieved by reserving life with auras to stay at low life permanently.",
        impact="Without Pain Attunement, you're missing a 30% more damage multiplier — one of the biggest single damage boosts in the game. Your spell DPS will be roughly 30% lower than other players running this build.",
        synergies=["Sanguimancy", "Energy Shield gear", "Life reservation"],
        anti_synergies=["Life stacking", "Builds without ES backup"],
        build_types=["spell_damage", "low_life", "energy_shield"],
    ),

    "Sanguimancy": KeystoneInfo(
        description="Life costs are paid from Energy Shield instead of Life.",
        benefits="Enables low-life builds safely — you can reserve life for Pain Attunement without dying to skill costs. Your ES absorbs the skill costs while your life stays reserved.",
        impact="Without Sanguimancy, every skill cast drains your life pool. On a low-life build, this means you'll constantly dip below the threshold and die to skill costs. This keystone is what makes the entire low-life archetype functional.",
        synergies=["Pain Attunement", "Energy Shield gear", "Life reservation", "ES recharge"],
        anti_synergies=["Chaos Inoculation", "Builds with no ES", "Blood Magic"],
        build_types=["low_life", "energy_shield", "spell_damage"],
    ),

    "Elemental Overload": KeystoneInfo(
        description="Your hits can't critically strike. If you've dealt a critical strike recently, 40% more elemental damage.",
        benefits="Huge 40% more elemental damage for builds that invest minimally in crit. You only need enough crit chance to proc it once every 4 seconds.",
        impact="Without Elemental Overload, non-crit builds lose a free 40% more damage multiplier. You'd need massive investment in crit chance AND crit multiplier gear to match this damage through actual crits.",
        synergies=["Elemental damage skills", "Fast-hitting skills", "Minimal crit gear"],
        anti_synergies=["Crit multi stacking", "Crit chance stacking"],
        build_types=["elemental_non_crit", "spell_damage", "elemental_attack"],
    ),

    "Avatar of Fire": KeystoneInfo(
        description="50% of physical, lightning, and cold damage is converted to fire. You can only deal fire damage.",
        benefits="Converts all damage to fire, letting you scale everything with fire damage modifiers. Simplifies gearing to a single element.",
        impact="Without Avatar of Fire, your damage is split across multiple elements, making it harder to scale with penetration and modifiers. Fire-focused builds lose significant damage scaling efficiency.",
        synergies=["Fire penetration", "Fire damage gear", "Conversion"],
        anti_synergies=["Chaos damage", "Non-fire elemental builds", "Poison"],
        build_types=["fire_damage", "conversion", "elemental_attack"],
    ),

    "Chaos Inoculation": KeystoneInfo(
        description="Maximum life becomes 1. You are immune to chaos damage.",
        benefits="Complete chaos immunity removes an entire damage type. You rely entirely on Energy Shield, freeing all chaos resistance gear slots for other stats.",
        impact="Without CI, ES builds must invest heavily in chaos resistance on gear (which competes with ES and damage stats), or risk being one-shot by chaos damage which bypasses ES by default.",
        synergies=["Energy Shield gear", "ES recharge", "Ghost Reaver", "Discipline aura"],
        anti_synergies=["Life stacking", "Life leech", "Blood Magic"],
        build_types=["energy_shield", "chaos_immune", "intelligence_stacking"],
    ),

    "Mind Over Matter": KeystoneInfo(
        description="40% of damage taken from hits is deducted from mana before life.",
        benefits="Effectively adds 40% of your mana pool as extra EHP. Great for builds with large mana pools and good mana recovery.",
        impact="Without Mind Over Matter, all hit damage goes directly to your life pool. You're missing a significant defensive layer — potentially thousands of extra effective HP that your unused mana could be providing.",
        synergies=["Large mana pool", "Mana regeneration", "Clarity aura"],
        anti_synergies=["Blood Magic", "Low mana builds", "Heavy mana reservation"],
        build_types=["mana_stacking", "hybrid_life_mana", "spell_damage"],
    ),

    "Ancestral Bond": KeystoneInfo(
        description="You can't deal damage directly. +1 to maximum number of summoned totems.",
        benefits="Extra totem scales clear speed and single-target damage. The 'no damage' downside is irrelevant because totems deal the damage for you.",
        impact="Without Ancestral Bond, totem builds have one fewer totem, directly reducing your damage output and coverage. For totem builds, this is a pure damage gain.",
        synergies=["Totem skills", "Totem placement speed"],
        anti_synergies=["Self-cast builds", "Attack builds"],
        build_types=["totem"],
    ),

    "Iron Reflexes": KeystoneInfo(
        description="Converts all evasion rating to armour rating.",
        benefits="Stack armour from both armour and evasion sources. Dexterity-based gear with high evasion becomes armour, giving huge physical damage reduction.",
        impact="Without Iron Reflexes, your evasion and armour are split defenses — neither reaches its full potential. Converting evasion to armour gives much more consistent physical mitigation than split defenses.",
        synergies=["Armour stacking", "Grace aura (becomes armour)", "Determination aura"],
        anti_synergies=["Evasion builds", "Acrobatics", "Dodge chance"],
        build_types=["armour_stacking", "physical_mitigation", "melee"],
    ),

    "Acrobatics": KeystoneInfo(
        description="Grants dodge chance but reduces armour and energy shield.",
        benefits="Strong avoidance layer for evasion-based characters. Dodge is checked independently of evasion, giving two chances to avoid hits.",
        impact="Without Acrobatics, evasion-based characters lack a secondary avoidance layer. When evasion fails (and it will — it's entropy-based), you take the full hit with minimal mitigation.",
        synergies=["Evasion gear", "Grace aura", "Dexterity stacking"],
        anti_synergies=["Iron Reflexes", "Armour stacking", "Energy Shield builds"],
        build_types=["evasion", "dodge", "ranged_attack"],
    ),

    "Ghost Reaver": KeystoneInfo(
        description="Life leech applies to Energy Shield instead of life.",
        benefits="Gives ES builds access to leech-based sustain. Essential for attack-based ES characters who need instant recovery during fights.",
        impact="Without Ghost Reaver, ES builds can't leech — your only recovery is ES recharge (which has a delay). In sustained combat, you'll have no way to recover ES until you stop taking damage.",
        synergies=["Energy Shield gear", "Attack builds", "Life leech sources"],
        anti_synergies=["Life builds", "ES recharge builds"],
        build_types=["energy_shield", "attack_es"],
    ),

    "Resolute Technique": KeystoneInfo(
        description="Your hits always connect (100% hit chance). You can never critically strike.",
        benefits="Eliminates accuracy as a stat requirement. Perfect for non-crit builds that want reliable, consistent damage without investing in accuracy gear.",
        impact="Without Resolute Technique, non-crit attack builds need significant accuracy investment on gear and passives. Missing attacks is a direct DPS loss — even 90% hit chance means 10% of your attacks deal zero damage.",
        synergies=["Non-crit attack builds", "Strength-based melee"],
        anti_synergies=["Critical strike builds", "Elemental Overload"],
        build_types=["non_crit_attack", "melee", "physical_attack"],
    ),

    "Unwavering Stance": KeystoneInfo(
        description="Cannot be stunned. Cannot evade attacks.",
        benefits="Stun immunity is critical for builds that channel or have long cast animations. The 'cannot evade' downside is irrelevant for armour-based characters.",
        impact="Without Unwavering Stance, you can be stunned mid-cast or mid-channel, interrupting your damage and leaving you vulnerable. In boss fights, a stun at the wrong moment means death.",
        synergies=["Armour stacking", "Iron Reflexes", "Channelling skills"],
        anti_synergies=["Evasion builds", "Acrobatics", "Grace aura"],
        build_types=["armour_stacking", "melee", "channelling"],
    ),

    "Crimson Dance": KeystoneInfo(
        description="Bleeding you inflict can stack up to 8 times on an enemy.",
        benefits="Massively increases bleed DPS by allowing stacking. Normally only the strongest bleed counts — with Crimson Dance, fast attacks apply many bleeds simultaneously.",
        impact="Without Crimson Dance, only your single strongest bleed applies. You're losing up to 7x bleed damage potential. For bleed builds, this is the difference between viable and non-functional boss damage.",
        synergies=["Bleed chance", "Physical damage", "Attack speed"],
        anti_synergies=["Elemental builds", "Spell builds"],
        build_types=["bleed", "physical_attack", "melee"],
    ),

    "Point Blank": KeystoneInfo(
        description="Projectiles deal up to 30% more damage to close targets, scaling down to 30% less at far range.",
        benefits="Huge damage boost for close-range projectile builds. Perfect for bow or wand characters who fight at short range.",
        impact="Without Point Blank, close-range projectile builds miss a 30% more multiplier. That's a massive DPS loss for builds designed to fight up close.",
        synergies=["Close-range projectile skills", "Barrage", "Rain of Arrows"],
        anti_synergies=["Long-range playstyles", "Off-screen clearing"],
        build_types=["projectile_close_range", "bow_attack", "wand_attack"],
    ),

    "Iron Will": KeystoneInfo(
        description="Strength's damage bonus applies to spell damage instead of only melee physical damage.",
        benefits="Lets strength-stacking characters scale spell damage. Every point of strength gives both life (survivability) and spell damage (offense).",
        impact="Without Iron Will, strength-stacking casters get zero spell damage from their primary attribute. You'd need to dual-scale both STR (for life) and INT (for damage), making gearing much harder.",
        synergies=["Strength stacking", "Battlemage"],
        anti_synergies=["Dexterity stacking", "Intelligence stacking"],
        build_types=["strength_stacking", "spell_damage", "hybrid_melee_caster"],
    ),

    "Eldritch Battery": KeystoneInfo(
        description="Energy Shield protects mana instead of life.",
        benefits="Lets you spend ES as mana, enabling high-cost skills without mana investment. Combined with Mind Over Matter, your ES also becomes a defensive layer via mana absorption.",
        impact="Without Eldritch Battery, high-cost skill builds struggle with mana sustain. You'd need heavy mana investment on gear and passives that could otherwise go toward damage or defenses.",
        synergies=["Mind Over Matter", "ES gear", "High mana cost skills"],
        anti_synergies=["CI", "ES as primary defense pool"],
        build_types=["mana_stacking", "spell_damage"],
    ),

    "Elemental Equilibrium": KeystoneInfo(
        description="Enemies you hit gain resistance to the element you hit them with, but lose resistance to other elements.",
        benefits="Lets you debuff enemy resistance to your main damage element by hitting with a different element first. Effectively adds penetration for free.",
        impact="Without Elemental Equilibrium, you're missing a significant resistance debuff on enemies. Builds that use this gain what amounts to 25-50% penetration for free — a huge DPS multiplier especially against bosses.",
        synergies=["Multi-element setups", "Trigger skills with off-element"],
        anti_synergies=["Single-element self-hit builds"],
        build_types=["elemental_damage", "spell_damage"],
    ),

    # ------------------------------------------------------------------
    # Blood Mage (Witch) ascendancy keystones
    # ------------------------------------------------------------------

    "Sunder the Flesh": KeystoneInfo(
        description="Your critical strikes cause enemies to explode on death, dealing a percentage of their life as physical damage to nearby enemies.",
        benefits="Massive clear speed boost — every enemy you kill becomes a chain explosion. In dense packs, one kill cascades into wiping the entire screen.",
        impact="Without Sunder the Flesh, your clear speed drops dramatically. You'll need to hit every enemy individually instead of relying on chain explosions. This is the primary reason Blood Mage clears faster than other ascendancies.",
        synergies=["Crit builds", "High pack density", "Physical damage scaling"],
        anti_synergies=["Non-crit builds", "Single-target focused"],
        build_types=["spell_damage", "crit", "clear_speed"],
    ),

    "Vitality Siphon": KeystoneInfo(
        description="Life and ES leech from spell damage. Spell hits recover a percentage of damage dealt as life and energy shield.",
        benefits="Gives spell builds sustain they normally can't get — life and ES leech from every spell hit. Keeps you alive during sustained combat without relying on flasks or regen.",
        impact="Without Vitality Siphon, your spell build has no leech. You'll rely entirely on regen, flasks, and ES recharge for sustain. In prolonged boss fights, you'll run out of recovery and die.",
        synergies=["High DPS spells", "ES builds", "Sanguimancy"],
        anti_synergies=["Low hit-rate builds", "DoT-only builds"],
        build_types=["spell_damage", "energy_shield", "sustain"],
    ),

    "Grasping Wounds": KeystoneInfo(
        description="Your hits have a chance to inflict Grasping Wounds, slowing enemies and causing them to take increased damage.",
        benefits="Enemies take more damage from all sources AND are slowed. This is both a damage multiplier and a defensive layer — slowed enemies are easier to avoid and kite.",
        impact="Without Grasping Wounds, you're missing both a damage multiplier on enemies AND a slow defensive layer. Your effective DPS is lower and enemies reach you faster. Nearly all Blood Mage builds take this for good reason.",
        synergies=["High hit rate", "Multi-hit spells", "Comet", "Cast on Crit"],
        anti_synergies=["DoT-only builds"],
        build_types=["spell_damage", "attack_damage", "crit"],
    ),

    "Crimson Power": KeystoneInfo(
        description="Gain increased spell damage based on your missing life percentage. More missing life = more damage.",
        benefits="Synergizes perfectly with low-life builds — the more life you reserve, the more spell damage you get. Combined with Pain Attunement, this creates massive damage scaling.",
        impact="Without Crimson Power, low-life Blood Mage builds lose a significant damage multiplier that scales with reserved life. You're leaving potentially 40-80% increased spell damage on the table.",
        synergies=["Low-life builds", "Pain Attunement", "Sanguimancy", "Life reservation"],
        anti_synergies=["Full life builds", "Life stacking"],
        build_types=["low_life", "spell_damage"],
    ),

    "Gore Spike": KeystoneInfo(
        description="Corpse skills deal additional damage and have increased area of effect.",
        benefits="Direct damage boost and AoE increase for corpse-based skills like Unearth, Detonate Dead, and Volatile Dead.",
        impact="Without Gore Spike, corpse-based builds deal less damage per corpse and have smaller explosion radius. If you're running corpse skills, this is free damage and coverage.",
        synergies=["Unearth", "Detonate Dead", "Volatile Dead", "Corpse skills"],
        anti_synergies=["Non-corpse builds"],
        build_types=["spell_damage", "corpse_skills"],
    ),

    "Sanguine Tides": KeystoneInfo(
        description="Blood skills have increased damage and reduced cost.",
        benefits="Makes blood-themed skills cheaper to cast and deal more damage. Efficiency boost for builds centered on blood skills.",
        impact="Without Sanguine Tides, blood skill builds pay higher costs and deal less damage. The cost reduction is especially important for sustain on skills that cost life.",
        synergies=["Blood skills", "Life cost management"],
        anti_synergies=["Non-blood skill builds"],
        build_types=["spell_damage", "blood_skills"],
    ),

    "Sacrifice of Blood": KeystoneInfo(
        description="Sacrifice a portion of life to gain a powerful temporary buff to spell damage and cast speed.",
        benefits="Massive burst damage window — sacrifice life for a significant spell damage and speed boost. Used for burst phases on bosses.",
        impact="Without Sacrifice of Blood, you lack an on-demand burst damage cooldown. Boss phases that require high burst DPS will take longer, and some DPS checks become harder to meet.",
        synergies=["Sanguimancy (ES pays the life cost)", "Burst damage builds"],
        anti_synergies=["Builds without ES backup for the life cost"],
        build_types=["spell_damage", "burst_damage"],
    ),

    # ------------------------------------------------------------------
    # Other common ascendancy keystones (cross-class)
    # ------------------------------------------------------------------

    "Running Assault": KeystoneInfo(
        description="Gain a damage buff after using a movement skill. Attacks and spells deal more damage for a short duration after dashing.",
        benefits="Free damage multiplier triggered by your movement skill. Since you're already dashing to dodge mechanics, this is essentially permanent uptime in most content.",
        impact="Without Running Assault, you're missing a more damage multiplier that has near-100% uptime in active gameplay. Every other player in your build is doing more damage simply by dashing before attacking.",
        synergies=["Movement skills", "Active playstyle", "Boss fights"],
        anti_synergies=["Stationary playstyles", "Totem builds"],
        build_types=["attack_damage", "spell_damage"],
    ),

    "Path Seeker": KeystoneInfo(
        description="Increased effect of buffs on you. Auras, heralds, and other buffs are more powerful.",
        benefits="Multiplies the effect of all your buffs — auras, heralds, flasks, temporary buffs. The more buffs you run, the more value this gives.",
        impact="Without Path Seeker, every aura, herald, and buff you run is less effective. If you're running 3+ buffs (and most builds do), you're losing a compounding efficiency bonus across all of them.",
        synergies=["Multiple auras", "Heralds", "Flask effect", "Buff stacking"],
        anti_synergies=["Builds with few buffs"],
        build_types=["aura_stacking", "buff_scaling"],
    ),

    "Path of the Sorceress": KeystoneInfo(
        description="Increased spell damage and cast speed based on intelligence.",
        benefits="Scales your spell damage and cast speed from intelligence. Since casters already stack INT, this gives free damage from a stat you're already investing in.",
        impact="Without Path of the Sorceress, your intelligence only gives mana and ES — you're not extracting damage value from your primary attribute. Other casters are getting both defense AND offense from every point of INT.",
        synergies=["Intelligence stacking", "Spell builds", "ES gear"],
        anti_synergies=["Attack builds", "Strength-based casters"],
        build_types=["spell_damage", "intelligence_stacking"],
    ),

    "Overwhelming Toxicity": KeystoneInfo(
        description="Poisons you inflict deal damage faster and have increased damage.",
        benefits="Dramatically increases poison DPS by making poisons tick faster and harder. Core keystone for any poison build — it's the difference between poisons being a minor DOT and being your primary damage source.",
        impact="Without Overwhelming Toxicity, your poisons tick slowly and deal less damage. Poison builds without this do a fraction of the DPS. If your build relies on poison at all, this is non-negotiable.",
        synergies=["Poison chance", "Chaos damage", "Fast-hitting skills"],
        anti_synergies=["Non-poison builds", "Pure elemental builds"],
        build_types=["poison", "chaos_damage", "dot"],
    ),

    "Relentless Pursuit": KeystoneInfo(
        description="Increased damage while moving. Bonus damage and speed when you haven't stopped recently.",
        benefits="Rewards aggressive, mobile playstyle with a persistent damage bonus. Since most ranged builds are always repositioning, this has near-permanent uptime.",
        impact="Without Relentless Pursuit, you're dealing less damage than players who take it — and it's essentially free since you're already moving to dodge mechanics. It's a damage multiplier with no real downside.",
        synergies=["Ranged builds", "Mobile playstyle", "Bow skills"],
        anti_synergies=["Stationary channelling", "Totem builds"],
        build_types=["ranged_attack", "mobile_playstyle"],
    ),

    "Wildsurge Incantation": KeystoneInfo(
        description="Spell skills fire an additional projectile or chain, with increased spell damage.",
        benefits="Extra projectile/chain means more hits per cast, more coverage, and more damage. Directly scales multi-hit spell builds.",
        impact="Without Wildsurge Incantation, your spells hit fewer targets per cast. In both clear (fewer enemies hit) and single-target (fewer hits per cast), you lose damage throughput.",
        synergies=["Projectile spells", "Chain spells", "Multi-hit builds"],
        anti_synergies=["Single-target only builds", "Non-projectile spells"],
        build_types=["spell_damage", "projectile"],
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
