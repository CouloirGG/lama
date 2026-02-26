# Meta Efficiency Analysis — Fate of the Vaal League

**Date**: 2026-02-25
**Data Source**: poe.ninja builds API (124,276 indexed characters), 11 build guide sites, YouTube creators
**Currency Rates**: 1 Divine = 288.1 Exalted = 27.47 Chaos

---

## 1. League Snapshot

| Class | % Share | Trend |
|---|---|---|
| Blood Mage | 17.94% | Down |
| Oracle | 16.79% | Up |
| Pathfinder | 12.63% | Down |
| Shaman | 8.0% | Down |
| Titan | 5.38% | Up |

---

## 2. Blood Mage Build Variants — Cost-to-Enable Analysis

**Finding**: The 70/30 split between MoM+EB and Blood Magic builds reflects affordability, not player ignorance.

| Variant | Cost to Enable | Key Items |
|---|---|---|
| MoM+EB (mainstream) | 5-15 div | Budget-friendly, works with any gear |
| Blood Magic + Atziri's Acuity | 30-80 div | Requires Acuity (underpriced at ~5-10 div) + specific rare mods |

**Atziri's Acuity** is underpriced relative to its power for Blood Magic builds — it's the keystone enabler that makes the variant work. Players who can't afford the full suite default to MoM+EB, which is correct budget optimization.

---

## 3. CI Mechanics in POE2 (Corrected)

**CRITICAL**: POE2 CI works differently from POE1:
- Non-poison chaos damage deals **2x to ES** but does **NOT bypass it** (POE1: chaos bypassed ES entirely)
- CI grants **bleed immunity** (added in 0.2.0e patch)
- CI makes you permanently "full life" which triggers **Pain Attunement's penalty** (-30% crit damage at full life)
- Only 6.5% of Pathfinders take CI — this is **correct** because:
  - Pathfinders have flask identity (chaos flasks cover chaos damage naturally)
  - Pain Attunement penalty is severe for crit builds
  - Chaos resistance is cheaper to solve via gear/flasks than giving up crit

---

## 4. Pain Attunement & Low-Life Builds

**Pain Attunement in POE2**: -30% crit damage at full life, +30% at low life
**Low life threshold**: 35% (vs 50% in POE1)

Low-life builds exist and are a valid archetype:
- **Coward's Legacy** belt: forces perpetual low-life state
- **Crown of Thorns**: low-life trigger helmet
- **Blood Mage** life-spending: naturally goes to low life via Blood Magic
- No Petrified Blood in POE2, so sustain is harder — requires specific investment

**Dashboard implication**: Don't flag Pain Attunement as a "trap" — check if the character has low-life enablers.

---

## 5. Anoint Crisis — Build Guide Copy-Paste Problem

**Key Data Points**:
- Most anoint adoption rates are **below 20%** across classes
- Exception: Blood Mage "Fast Metabolism" at **69.6%** adoption
- Players overwhelmingly copy build guides verbatim, including anoints

**Build Guide Sources Driving Anoint Choices**:
1. **Mobalytics** — most popular, detailed gear/tree/anoint recommendations
2. **Maxroll** — comprehensive tier lists with anoint sections
3. **PoE Vault** — deep guides with anoint reasoning
4. **Game8** — JP audience, copied to EN
5. **AOEAH, Epiccarry, MMOGAH, Odealo, Boosting-Ground, Boostmatch** — SEO-farm guides, often copy each other

**YouTube Creator Influence** (by follower impact on prices):
- **Tier 1** (market movers): Zizaran, Fubgun, Alkaizer
- **Tier 2** (significant influence): PhazePlays, Asmodeus, DEADRABB1T
- **Tier 3** (niche): Medieval Marty, others

**Opportunity**: Low anoint adoption = massive untapped efficiency. A dashboard that shows "optimal anoint for your build + cost" would help players AND drive demand for underpriced anoint materials.

---

## 6. Suffix Economy — Exponential Gear Pricing

**Key Insight** (from Stu): Resistance nodes on the passive tree are **strategic**, not wasteful. They free gear suffix slots.

| Gear Scenario | Approximate Price |
|---|---|
| Triple T1 prefix ring (damage mods) | 5-10 div |
| Same + T1 resistance suffix | 20-40 div |
| Same + T1 res + T1 attack speed suffix | 50-100+ div |
| Triple T1 prefix + triple T1 suffix | Near-mirror territory |

Each additional good suffix mod scales price **exponentially**, not linearly. This means:
- Tree resistance = saves 10-100x on gear
- Our mod weight table should reflect suffix scarcity premium
- "Resistance is filler" (weight 0.3) is **wrong for pricing** — on a triple-T1-prefix item, a resistance suffix adds enormous value

**Dashboard implication**: Show "suffix budget freed" by tree resistance investment.

---

## 7. Cost of Upgrade (CoU) Framework

| Investment Tier | Divine Range | What You Get | Best ROI |
|---|---|---|---|
| Starter | 0-5 div | Cheap uniques, leveling gear | Foundation |
| Core | 5-15 div | Build-defining uniques + gem levels | High |
| **Lineage Gems** | **15-50 div** | **Multiplicative "More" damage** | **BEST DPS/div** |
| Variant Switch | 50-100 div | Different build variant (e.g., MoM→Blood Magic) | Situational |
| Endgame Craft | 100-500 div | Self-craft + suffix optimization | Diminishing |

**Lineage support gems provide multiplicative "More" damage** vs gear's additive "Increased" — they are consistently the best DPS investment per divine spent in the 15-50 div range.

---

## 8. HC vs SC Defensive Techniques

HC players use defensive techniques that SC players largely ignore, creating pricing arbitrage:

| Technique | Mechanic | HC Adoption | SC Adoption |
|---|---|---|---|
| Disciple of Varashta | 3-node defensive system (Sorceress ascendancy) | High | Low |
| Witchhunter Sorcery Ward | All-damage absorption barrier (Mercenary) | High | Low |
| Lich Eternal Life bypass | 25% less damage taken exploit | High | Very Low |
| Infinite Healing Titan | 15k life/sec via 100% reduced duration | Medium | Very Low |
| Iron Reflexes | Deterministic phys mitigation (no RNG evasion) | High | Medium |
| Shield Wall | Projectile blocking | High | Low |
| 75/50 Block Investment | Max block/spell block | High | Medium |

**Pricing Impact**: HC defensive items are underpriced in SC because demand is lower — but they provide enormous survivability that SC players would benefit from.

---

## 9. Passive Tree Notable Findings

### Free Efficiency Nodes (commonly missed)
- **Jewel Socket #7960**: Location needs verification — if easily accessible, provides free socket for build-specific jewels
- **Cower Before the First Ones**: Oracle-only node (Paths Not Taken / The Unseen Path ascendancy) — NOT universally available, class-locked

### US+Bulwark Combo
- Potentially exploitable defensive combo — needs mobalytics deep dive
- Could be an efficiency outlier that build guides haven't caught

---

## 10. Meta Influence vs Price Following

**Conclusion**: Current prices are ~80% driven by build guide consensus and ~20% by actual optimization.

**Evidence**:
- Build guides converge on the same items/skills → demand spike → price rise
- When a YouTube creator (especially Tier 1) features an item, price moves within hours
- Underpriced items exist where guides haven't recommended them (Atziri's Acuity, HC defensive items in SC)
- The meta is self-reinforcing: guides → players copy → demand → price → guides cite price as validation

**Opportunity for LAMA**: By surfacing CoU analysis, optimal anoints, and build efficiency data, we can:
1. Help individual players optimize their builds (immediate value)
2. Aggregate optimization signals into pricing intelligence (identify underpriced items before guides catch up)
3. Potentially influence the meta by being the first source to recommend optimizations

---

## Dashboard Feature Requirements

### Build Efficiency Tab (new)
1. **Cost of Upgrade Calculator** — For a given character, show the next best investment tier and specific items
2. **Optimal Anoint Finder** — Given class/skill, show the highest-impact anoint with cost and adoption %
3. **Build Variant Comparison** — Show alternative variants with cost-to-enable and DPS/defense tradeoffs
4. **Suffix Economy Analyzer** — Show how many suffix slots tree investment frees and the divine value saved
5. **HC Technique Scout** — Surface HC defensive techniques applicable to the character's build
6. **Lineage Gem ROI** — Show which lineage gems give the most DPS per divine for the character's skills
7. **Trend Predictor** — Track build guide mentions and YouTube features to predict price movements

### Data Sources Needed
- poe.ninja builds API (already integrated)
- poe2scout prices (already integrated)
- Build guide scraping (new — periodic, not real-time)
- Anoint → notable mapping (partially in builds_client.py)
- Lineage gem pricing (poe2scout has this)
- Passive tree data (needs integration)
