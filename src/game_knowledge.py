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
        description="You can't deal damage directly. Reworked in 0.5.0: now DOUBLES your maximum totem limit (instead of a flat +1), placing totems costs nothing and consumes no charges, and each totem reserves 75 spirit.",
        benefits="Doubling the totem limit is a far bigger damage multiplier than the old +1 for invested totem builds, and free/no-charge placement removes the old casting friction. The 'no direct damage' downside is irrelevant — totems deal the damage.",
        impact="Without Ancestral Bond, totem builds lose the doubled limit (so far fewer totems) and pay placement costs — directly cutting damage and coverage. For a totem build it's a large, near-mandatory gain.",
        synergies=["Totem skills", "Spirit reservation efficiency", "Totem placement speed"],
        anti_synergies=["Self-cast builds", "Attack builds", "Low spirit pools"],
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
        benefits="Gives ES builds leech-based sustain — more important after 0.5.0, which nerfed ES recharge. Note 0.5.0 also removed instant leech: leech now applies gradually (single highest instance per resource, capped at 40k damage), so it's steady recovery rather than the old instant burst.",
        impact="Without Ghost Reaver, ES builds can't leech at all — recovery falls back on ES recharge, which 0.5.0 nerfed (slower start, weaker rate). In sustained fights you'll have no reliable way to refill ES while taking hits.",
        synergies=["Energy Shield gear", "Attack builds", "Life leech sources"],
        anti_synergies=["Life builds", "Pure ES-recharge builds"],
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
        benefits="Gives spell builds sustain they normally can't get. Since 0.5.0 removed instant leech (single highest instance, 40k cap), it's steady recovery during sustained combat rather than instant burst — still valuable given ES recharge was also nerfed.",
        impact="Without Vitality Siphon, your spell build has no leech and leans on regen, flasks, and the now-weaker ES recharge. In prolonged boss fights you'll run low on recovery and die.",
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
            "CI builds can have 8000+ ES. 0.5.0 nerfed ES recharge (slower "
            "start and reduced recharge-rate notables), so ES builds now "
            "lean more on leech (Ghost Reaver) and the new Runic Ward layer "
            "for in-combat recovery. Chaos damage bypasses ES unless you "
            "have CI or specific mods."
        ),
        strengths=["Can reach very high values",
                   "Recharges without flasks",
                   "CI makes you chaos immune"],
        weaknesses=["Chaos damage bypasses it by default",
                    "Recharge delay can be deadly",
                    "Recharge was nerfed in 0.5.0 (weaker recovery)",
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
            "Deflection is a POE2 shield defense that gives a CHANCE to "
            "Deflect incoming hits. Reformulated in 0.5.0 to scale better "
            "with Deflection Rating and capped at 95% — mirroring how "
            "Evasion's chance-to-Evade is capped at 95%."
        ),
        how_it_works=(
            "Deflection is an entropy-style chance to deflect a hit, scaling "
            "with your Deflection Rating against the attacker's accuracy "
            "(0.5.0 formula), up to a 95% cap. More shield/Deflection "
            "investment raises the chance. It is checked separately from "
            "block, giving shield users an additional avoidance layer."
        ),
        strengths=["Additional avoidance layer for shield users",
                   "Scales with Deflection Rating investment",
                   "Can reach a 95% deflect chance",
                   "Works alongside block"],
        weaknesses=["Requires a shield",
                    "Chance-based — unreliable at low investment",
                    "Does not apply to unshielded characters"],
    ),

    "runic_ward": DefenseMechanicInfo(
        description=(
            "Runic Ward is a new 0.5.0 (Kalguuran) defensive layer — a "
            "separate pool that keeps you alive once you hit 1 life, "
            "absorbing damage while it lasts and regenerating independently "
            "of your life and energy shield."
        ),
        how_it_works=(
            "Granted by Kalguuran armour via Verisium Runeforging: armour "
            "bases below item level 55 gain Runic Ward for free, while "
            "higher-level bases trade some conventional defenses (armour/"
            "evasion/ES) for it. It activates at 1 life as a last-stand "
            "buffer and regenerates on its own, independent of life recovery."
        ),
        strengths=["Independent regeneration (no flasks needed)",
                   "A genuine last-stand layer at 1 life",
                   "Free on low-level (sub-ilvl-55) armour bases"],
        weaknesses=["Requires Kalguuran/Verisium-crafted armour",
                    "Higher-level bases trade away conventional defenses",
                    "New mechanic — limited gear support so far"],
    ),
}


# ---------------------------------------------------------------------------
# MOD_SYNERGIES
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# UNIQUE JEWELS — build-defining jewels with special mechanics
# ---------------------------------------------------------------------------

@dataclass
class UniqueJewelInfo:
    description: str
    impact: str  # what the build gains/loses
    build_types: List[str]
    restriction: str = ""  # e.g., "only one Historic jewel at a time"


UNIQUE_JEWELS: Dict[str, UniqueJewelInfo] = {
    "The Adorned": UniqueJewelInfo(
        description="Increases the effect of all Corrupted Magic Jewel Socket Passive Skills by its effect value (e.g., 148% = 2.48x multiplier on all magic jewel mods).",
        impact="Build-defining for magic jewel stacking builds. Every corrupted magic jewel's mods get multiplied — a 15% spell damage mod becomes 37%. With 8+ magic jewels, this is often 30-50% of total DPS.",
        build_types=["spell_damage", "attack_damage", "crit"],
    ),
    "Megalomaniac": UniqueJewelInfo(
        description="Grants 3 random notable passives from the passive tree. The most universally accessible power jewel.",
        impact="Any 2-3 notable combo that aligns with your build is essentially free passive points without pathing. Top players search for specific combos that synergize with their build.",
        build_types=["any"],
    ),
    "From Nothing": UniqueJewelInfo(
        description="Allows allocating a Keystone without pathing to it on the passive tree.",
        impact="Sleeper strong for builds that want an off-tree Keystone without spending 10-15 passive points pathing across the tree. Saves passive points for damage or defense nodes.",
        build_types=["any"],
    ),
    "Heart of the Well": UniqueJewelInfo(
        description="Grants extra damage as Lightning, cooldown recovery rate, and crit scaling.",
        impact="Strong for Cast on Crit and spell builds. The cooldown recovery rate directly increases CoC trigger frequency, and extra Lightning damage is a direct DPS multiplier.",
        build_types=["spell_damage", "crit", "coc"],
    ),
    "Heroic Tragedy": UniqueJewelInfo(
        description="Historic jewel that transforms passives within its radius based on its seed number. Can place powerful Keystones near your socket.",
        impact="Build-defining IF the seed rolls the right Keystone near your jewel socket. A gamble on placement, but the ceiling is very high.",
        build_types=["any"],
        restriction="Only one Historic jewel can be socketed at a time (mutually exclusive with Undying Hate)",
    ),
    "Undying Hate": UniqueJewelInfo(
        description="Historic jewel that transforms passives within its radius. Alternative to Heroic Tragedy with different transformation options.",
        impact="Similar to Heroic Tragedy — seed-dependent passive transformation. Can enable unique passive combinations not available otherwise.",
        build_types=["any"],
        restriction="Only one Historic jewel can be socketed at a time (mutually exclusive with Heroic Tragedy)",
    ),
}


# ---------------------------------------------------------------------------
# CLASS_SCALING — Per-class DPS scaling weights derived from 410-build analysis.
# Values represent the correlation ratio (top 1/3 vs bottom 1/3 DPS).
# Higher = more important for that class.
# ---------------------------------------------------------------------------

@dataclass
class ClassScaling:
    primary_factor: str  # name of #1 scaling factor
    weights: Dict[str, float]  # factor_name -> correlation_ratio
    defense_meta: str  # dominant defense type
    key_keystones: List[str]  # most important keystones
    keystone_combo: str  # description of the keystone pattern
    top_unique: str  # most popular unique item
    top_unique_pct: float  # adoption %

CLASS_SCALING: Dict[str, ClassScaling] = {
    "Warrior": ClassScaling(
        primary_factor="extra_elemental",
        weights={
            "extra_as": 3.3, "atk_speed": 3.2, "evasion_pct": 3.6,
            "crit_chance": 2.6, "crit_multi": 2.3, "armour_pct": 1.9,
            "cast_speed": 4.1, "gem_levels": 1.0,
        },
        defense_meta="life",
        key_keystones=["Blood Magic", "Giant's Blood", "Sacrifice of Flesh"],
        keystone_combo="Blood Magic + Giant's Blood for life scaling",
        top_unique="Headhunter",
        top_unique_pct=57.0,
    ),
    "Witch": ClassScaling(
        primary_factor="crit_multi",
        weights={
            "crit_multi": 11.7, "crit_chance": 3.5, "extra_as": 1.8,
            "evasion_pct": 1.7, "jewel_count": 1.6, "es_pct": 1.4,
            "gem_levels": 1.4, "cast_speed": 1.2,
        },
        defense_meta="es",
        key_keystones=["Eldritch Battery", "Mind Over Matter", "Blood Magic"],
        keystone_combo="EB + MoM for ES-as-mana defense, or Blood Magic for life builds",
        top_unique="The Vertex",
        top_unique_pct=49.0,
    ),
    "Ranger": ClassScaling(
        primary_factor="crit_chance",
        weights={
            "crit_chance": 2.6, "gem_levels": 1.6, "crit_multi": 1.5,
            "cast_speed": 3.0, "evasion_pct": 1.2, "es_pct": 1.2,
            "atk_speed": 1.0, "extra_as": 1.0,
        },
        defense_meta="es",
        key_keystones=[],
        keystone_combo="Minimal keystones — Ranger scales through gear and gems",
        top_unique="Headhunter",
        top_unique_pct=92.0,
    ),
    "Sorceress": ClassScaling(
        primary_factor="extra_elemental",
        weights={
            "extra_as": 1.5, "crit_multi": 1.5, "es_flat": 1.4,
            "crit_chance": 1.3, "gem_levels": 1.3, "cast_speed": 1.2,
            "spell_dmg": 1.2, "jewel_count": 1.2,
        },
        defense_meta="es",
        key_keystones=["Chaos Inoculation", "Elemental Equilibrium"],
        keystone_combo="CI for chaos immunity + ES as primary defense",
        top_unique="Maligaro's Virtuosity",
        top_unique_pct=54.0,
    ),
    "Monk": ClassScaling(
        primary_factor="crit_multi",
        weights={
            "crit_multi": 2.7, "extra_as": 2.7, "armour_pct": 5.2,
            "phys_dmg": 1.7, "crit_chance": 1.6, "jewel_count": 1.5,
            "cast_speed": 5.0, "gem_levels": 1.0,
        },
        defense_meta="es",
        key_keystones=["Chaos Inoculation"],
        keystone_combo="CI for chaos immunity, some builds use Resonance",
        top_unique="Headhunter",
        top_unique_pct=69.0,
    ),
    "Mercenary": ClassScaling(
        primary_factor="crit_chance",
        weights={
            "crit_chance": 3.6, "crit_multi": 2.9, "cast_speed": 2.7,
            "extra_as": 2.5, "jewel_count": 2.4, "es_flat": 1.9,
            "gem_levels": 1.4, "es_pct": 1.4,
        },
        defense_meta="life",
        key_keystones=["Mind Over Matter", "Eldritch Battery", "Blood Magic"],
        keystone_combo="EB + MoM or Blood Magic depending on build",
        top_unique="Headhunter",
        top_unique_pct=58.0,
    ),
    "Huntress": ClassScaling(
        primary_factor="spell_damage",
        weights={
            "spell_dmg": 5.4, "crit_multi": 3.8, "cast_speed": 3.6,
            "gem_levels": 2.4, "extra_as": 1.8, "es_pct": 1.6,
            "jewel_count": 1.5, "es_flat": 1.4,
        },
        defense_meta="es",
        key_keystones=["Chaos Inoculation", "Mind Over Matter", "Eldritch Battery"],
        keystone_combo="CI + MoM + EB triple defense layer",
        top_unique="Headhunter",
        top_unique_pct=76.0,
    ),
    "Druid": ClassScaling(
        primary_factor="crit_multi",
        weights={
            "crit_multi": 10.7, "cast_speed": 2.8, "es_flat": 1.8,
            "atk_speed": 1.6, "extra_as": 1.6, "crit_chance": 1.5,
            "evasion_pct": 1.4, "jewel_count": 1.4,
        },
        defense_meta="mom",
        key_keystones=["Mind Over Matter", "Wildsurge Incantation", "Eldritch Battery", "Conduit", "Resonance", "Hollow Palm Technique", "Blackflame Covenant"],
        keystone_combo="MoM + EB + Wildsurge + Conduit (64-71% of top builds). Most complex keystone stacking of any class.",
        top_unique="The Covenant",
        top_unique_pct=55.0,
    ),
}


# ---------------------------------------------------------------------------
# TOP_SUPPORT_GEMS — Support gems that separate top from bottom builds.
# These appear in top 1/3 but rarely in bottom 1/3.
# ---------------------------------------------------------------------------

@dataclass
class SupportGemInfo:
    description: str
    impact: str
    best_for: List[str]  # class names or "all"

TOP_SUPPORT_GEMS: Dict[str, SupportGemInfo] = {
    "Cast on Critical": SupportGemInfo(
        description="Triggers linked spells when you critically strike with an attack.",
        impact="Enables Cast on Crit builds — the highest DPS archetype this meta. Turns attack speed and crit chance into spell DPS.",
        best_for=["Witch", "Sorceress", "Mercenary", "Huntress"],
    ),
    "Rakiata's Flow": SupportGemInfo(
        description="Grants additional projectile and damage scaling.",
        impact="Top-only support gem — 54 top builds use it vs 0 bottom builds. Core for projectile and multi-hit builds.",
        best_for=["all"],
    ),
    "Boundless Energy II": SupportGemInfo(
        description="Increases energy and damage scaling for linked skills.",
        impact="39 top builds vs 0 bottom. Provides both damage and sustain scaling.",
        best_for=["Witch", "Mercenary", "Huntress"],
    ),
    "Garukhan's Resolve": SupportGemInfo(
        description="Grants evasion-based defense bonuses and damage.",
        impact="37 top builds vs 0 bottom. Defensive + offensive support.",
        best_for=["Warrior", "Ranger", "Monk", "Huntress"],
    ),
    "Dialla's Desire": SupportGemInfo(
        description="Increases damage based on gem levels.",
        impact="Top Sorceress support — scales with the +gem level stacking strategy.",
        best_for=["Sorceress"],
    ),
    "Uul-Netol's Embrace": SupportGemInfo(
        description="Physical damage scaling and life conversion.",
        impact="Warrior top-only support — core for melee physical builds.",
        best_for=["Warrior"],
    ),
    "Uhtred's Augury": SupportGemInfo(
        description="Spell augmentation and damage multiplier.",
        impact="Top Witch and Mercenary support for spell builds.",
        best_for=["Witch", "Mercenary"],
    ),
    "Concentrated Area": SupportGemInfo(
        description="Reduces area but increases area damage.",
        impact="DPS boost for AoE skills — trades clear speed for single-target damage.",
        best_for=["Monk", "Sorceress", "Druid"],
    ),
}


# ---------------------------------------------------------------------------
# POPULAR_UNIQUES — Build-defining unique items with adoption rates and
# class affinity from 410-build analysis.
# ---------------------------------------------------------------------------

@dataclass
class UniqueItemInfo:
    description: str
    impact: str
    slot: str
    global_adoption: float  # % of all builds
    best_classes: List[str]
    class_adoption: Dict[str, float]  # class -> adoption %

POPULAR_UNIQUES: Dict[str, UniqueItemInfo] = {
    "Headhunter": UniqueItemInfo(
        description="Steals rare monster mods on kill, granting massive temporary buffs.",
        impact="THE endgame belt. Stolen mods can multiply your DPS by 10-100x during mapping. Near-mandatory for endgame clear speed.",
        slot="Belt",
        global_adoption=67.0,
        best_classes=["all"],
        class_adoption={"Ranger": 92, "Huntress": 76, "Monk": 69, "Druid": 67, "Mercenary": 58, "Warrior": 57, "Witch": 40},
    ),
    "The Vertex": UniqueItemInfo(
        description="Helmet with +gem levels, ES, and removes attribute requirements from gems.",
        impact="Frees up attribute investment and adds gem levels. Core for ES/spell builds.",
        slot="Helm",
        global_adoption=34.0,
        best_classes=["Witch", "Sorceress", "Mercenary"],
        class_adoption={"Sorceress": 68, "Witch": 49, "Mercenary": 29, "Huntress": 27, "Druid": 26, "Monk": 24},
    ),
    "Kalandra's Touch": UniqueItemInfo(
        description="Ring that mirrors the mods of your other ring.",
        impact="Doubles the mods of your best ring. If your other ring has strong mods, this is massive value.",
        slot="Ring2",
        global_adoption=29.0,
        best_classes=["all"],
        class_adoption={"Huntress": 49, "Ranger": 39, "Mercenary": 34, "Druid": 33, "Witch": 32},
    ),
    "The Covenant": UniqueItemInfo(
        description="Body armour that adds chaos damage to spells and life cost.",
        impact="Build-defining for Druid (55% adoption). Adds significant chaos damage to spells at the cost of life per cast. Synergizes with Sanguimancy (ES pays the life cost).",
        slot="BodyArmour",
        global_adoption=18.0,
        best_classes=["Druid", "Sorceress", "Mercenary", "Huntress"],
        class_adoption={"Druid": 55, "Sorceress": 32, "Mercenary": 21, "Huntress": 16},
    ),
    "Hyrri's Ire": UniqueItemInfo(
        description="Body armour with high evasion, cold damage, and dodge.",
        impact="Best-in-slot for evasion-based ranged builds. Adds cold damage and dodge chance.",
        slot="BodyArmour",
        global_adoption=11.0,
        best_classes=["Ranger", "Mercenary"],
        class_adoption={"Ranger": 45, "Mercenary": 37},
    ),
    "Maligaro's Virtuosity": UniqueItemInfo(
        description="Gloves with massive crit multiplier bonus.",
        impact="54% of Sorceress builds use these — the crit multi bonus is one of the biggest single DPS multipliers in the game.",
        slot="Gloves",
        global_adoption=9.0,
        best_classes=["Sorceress"],
        class_adoption={"Sorceress": 54},
    ),
    "Choir of the Storm": UniqueItemInfo(
        description="Amulet that adds lightning damage based on critical strikes.",
        impact="Strong for crit spell builds — each crit adds lightning bolts. Synergizes with high crit chance builds.",
        slot="Amulet",
        global_adoption=7.0,
        best_classes=["Druid", "Witch"],
        class_adoption={"Druid": 24, "Witch": 14},
    ),
    "Palm of the Dreamer": UniqueItemInfo(
        description="Offhand that grants extra elemental damage.",
        impact="27% extra elemental damage as a shield slot — massive DPS for spell builds that don't need a second weapon.",
        slot="Offhand",
        global_adoption=5.0,
        best_classes=["Witch", "Druid"],
        class_adoption={"Druid": 14, "Witch": 11, "Warrior": 11},
    ),
    "Essentia Sanguis": UniqueItemInfo(
        description="Gloves with ES leech and attack bonuses.",
        impact="Core for ES-based attack builds (Monk, Huntress). Provides ES sustain through leech.",
        slot="Gloves",
        global_adoption=6.0,
        best_classes=["Monk", "Huntress"],
        class_adoption={"Huntress": 29, "Monk": 18},
    ),
    "Plaguefinger": UniqueItemInfo(
        description="Gloves that enhance poison and chaos damage.",
        impact="43% of Rangers use these — core for Poisonburst Arrow builds.",
        slot="Gloves",
        global_adoption=5.0,
        best_classes=["Ranger"],
        class_adoption={"Ranger": 43},
    ),
}


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
# KEYSTONE_COMBOS — synergistic keystone pairings from data analysis
# ---------------------------------------------------------------------------

@dataclass
class KeystoneCombo:
    keystones: List[str]
    description: str
    impact: str
    best_classes: List[str]
    adoption_pct: float  # % of top builds using this combo


KEYSTONE_COMBOS: List[KeystoneCombo] = [
    KeystoneCombo(
        keystones=["Eldritch Battery", "Mind Over Matter"],
        description="ES protects mana (EB), then mana absorbs 40% of damage (MoM). Your Energy Shield becomes a massive damage buffer without needing life investment.",
        impact="The #1 defensive combo for top spell builds. 32% of top Witch builds and 36% of top Mercenary builds use this. Turns ES into an EHP multiplier — your ES pool effectively becomes extra life.",
        best_classes=["Witch", "Mercenary", "Huntress", "Druid"],
        adoption_pct=25.0,
    ),
    KeystoneCombo(
        keystones=["Mind Over Matter", "Eldritch Battery", "Wildsurge Incantation", "Conduit"],
        description="The Druid mega-combo: EB+MoM for defense, Wildsurge for extra projectile/chain, Conduit for charge sharing. 64-71% of top Druid builds use ALL of these.",
        impact="This 4-keystone combo defines the Druid meta. Without all 4, you're fundamentally weaker than other top Druid builds. Requires pathing across the passive tree.",
        best_classes=["Druid"],
        adoption_pct=65.0,
    ),
    KeystoneCombo(
        keystones=["Blood Magic", "Giant's Blood"],
        description="Blood Magic removes mana (skills cost life), Giant's Blood increases life pool and regen. Together they create a massive life pool with strong sustain.",
        impact="Core Warrior combo — 47-62% of top Warriors use both. Your entire resource system is life-based, making gearing simpler and life scaling exponential.",
        best_classes=["Warrior"],
        adoption_pct=47.0,
    ),
    KeystoneCombo(
        keystones=["Chaos Inoculation"],
        description="CI alone transforms your build: 1 life, immune to chaos, ES is your only health pool. Not a combo — it's a standalone build archetype.",
        impact="The most popular keystone globally (26% of all builds). Removes chaos resistance as a gear concern entirely. Requires heavy ES investment on gear.",
        best_classes=["Sorceress", "Monk", "Huntress"],
        adoption_pct=26.0,
    ),
    KeystoneCombo(
        keystones=["Blood Magic", "Sanguimancy", "Pain Attunement"],
        description="Blood Magic (skills cost life) + Sanguimancy (life costs paid from ES) + Pain Attunement (30% more spell damage on low life). The low-life spell archetype.",
        impact="Enables low-life builds: reserve life for Pain Attunement's 30% more damage, while Sanguimancy ensures skill costs come from ES instead of draining your reserved life.",
        best_classes=["Witch"],
        adoption_pct=20.0,
    ),
    KeystoneCombo(
        keystones=["Mind Over Matter", "Blackflame Covenant"],
        description="MoM for defense (mana absorbs damage) + Blackflame Covenant for chaos/fire damage scaling. 100% of top Druid builds use both.",
        impact="Blackflame Covenant converts fire damage to chaos — combined with MoM for defense, this is the Druid's offensive + defensive identity.",
        best_classes=["Druid"],
        adoption_pct=79.0,
    ),
    KeystoneCombo(
        keystones=["Blood Magic", "Sacrifice of Flesh"],
        description="Blood Magic (skills cost life) + Sacrifice of Flesh (sacrifice life for power). The Warrior burst damage combo.",
        impact="42% of top Warriors use Sacrifice of Flesh — it provides burst damage windows at the cost of life, which Blood Magic builds have in abundance.",
        best_classes=["Warrior"],
        adoption_pct=42.0,
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
