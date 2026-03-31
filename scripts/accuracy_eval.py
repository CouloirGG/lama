"""
accuracy_eval.py — Ground-truth accuracy evaluation for LAMA.

Claude (this script) acts as the expert build analyst. It deep-analyzes
builds from poe.ninja, then runs the same builds through LAMA's why-engine,
and compares every recommendation. Discrepancies are logged as bugs.

Usage:
    python scripts/accuracy_eval.py

Output: JSON report at scripts/accuracy_report.json
"""

import sys, os, json, time, re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from builds_client import (
    BuildsClient, classify_build, ASCENDANCY_MAP, ATTACK_SKILLS,
    SPELL_SKILLS, MINION_SKILLS,
)
from why_engine import WhyEngine, _strip_ninja_brackets, _analyze_mod_contribution
from pob_decoder import decode_pob_code


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExpertAnalysis:
    """Claude's deep analysis of a build."""
    name: str = ""
    account: str = ""
    ascendancy: str = ""
    level: int = 0
    tier: str = ""  # "top", "mid", "low"

    # Core classification
    main_skill: str = ""
    damage_type: str = ""  # spell, attack, minion
    defense_type: str = ""  # life, es, hybrid, mom
    is_coc: bool = False
    is_crit: bool = False

    # DPS analysis
    main_dps: float = 0.0
    dps_sources: List[Dict] = field(default_factory=list)  # what contributes to DPS
    dps_bottlenecks: List[str] = field(default_factory=list)  # what's holding DPS back

    # Survival analysis
    ehp: float = 0.0
    life: int = 0
    es: int = 0
    resists_capped: bool = False
    uncapped_resists: List[str] = field(default_factory=list)
    survival_strengths: List[str] = field(default_factory=list)
    survival_weaknesses: List[str] = field(default_factory=list)

    # Gear analysis
    gear_strengths: List[Dict] = field(default_factory=list)  # slot, item, why
    gear_weaknesses: List[Dict] = field(default_factory=list)
    gear_upgrades: List[Dict] = field(default_factory=list)  # specific upgrade suggestions

    # Keystone/passive analysis
    keystones_allocated: List[str] = field(default_factory=list)
    ascendancy_passives_count: int = 0
    keystone_recommendations: List[str] = field(default_factory=list)

    # Overall recommendations
    top_recommendations: List[str] = field(default_factory=list)


@dataclass
class LamaAnalysis:
    """LAMA's analysis of the same build."""
    main_skill: str = ""
    damage_type: str = ""
    defense_type: str = ""
    is_coc: bool = False
    is_crit: bool = False

    scorecard_ehp: int = 0
    scorecard_dps: str = ""

    actions: List[Dict] = field(default_factory=list)
    keystones: List[Dict] = field(default_factory=list)
    synergy_dps_contributors: List[Dict] = field(default_factory=list)
    synergy_dps_missing: List[Dict] = field(default_factory=list)
    synergy_survival_missing: List[Dict] = field(default_factory=list)

    gear_missing_mods: List[Dict] = field(default_factory=list)


@dataclass
class Discrepancy:
    """A difference between Claude's analysis and LAMA's output."""
    category: str  # "classification", "dps", "survival", "gear", "keystone", "recommendation"
    severity: str  # "critical", "major", "minor"
    description: str
    claude_says: str
    lama_says: str
    build_name: str = ""
    build_tier: str = ""


# ---------------------------------------------------------------------------
# Expert build analysis (Claude as analyst)
# ---------------------------------------------------------------------------

def expert_analyze(char, pob, arch) -> ExpertAnalysis:
    """Deep analysis of a build — Claude's ground truth."""
    analysis = ExpertAnalysis(
        name=char.name,
        account=char.account or "",
        ascendancy=char.ascendancy,
        level=char.level,
        ascendancy_passives_count=getattr(char, 'ascendancy_points', 0),
    )

    # ── Main skill + DPS ──
    best_dps = 0
    best_skill = ""
    all_skills = []
    for sg in char.skill_groups:
        for d in (sg.dps if hasattr(sg, 'dps') and sg.dps else []):
            total = max(d.dps or 0, d.dot_dps or 0, d.damage or 0)
            if total > 0:
                all_skills.append({"name": d.name, "dps": total,
                                   "is_dot": (d.dot_dps or 0) > (d.dps or 0)})
            if total > best_dps:
                best_dps = total
                best_skill = d.name or ""

    analysis.main_skill = best_skill
    analysis.main_dps = best_dps
    analysis.dps_sources = all_skills

    # ── Damage type ──
    if best_skill in MINION_SKILLS:
        analysis.damage_type = "minion"
    elif best_skill in SPELL_SKILLS:
        analysis.damage_type = "spell"
    elif best_skill in ATTACK_SKILLS:
        analysis.damage_type = "attack"
    else:
        # Heuristic from gear
        spell_mods = 0
        attack_mods = 0
        for eq in char.equipment:
            for mod in (eq.explicit_mods or []) + (eq.implicit_mods or []):
                ml = _strip_ninja_brackets(mod).lower()
                if "spell" in ml or "cast speed" in ml:
                    spell_mods += 1
                if "attack speed" in ml or "added" in ml and "attack" in ml:
                    attack_mods += 1
        analysis.damage_type = "spell" if spell_mods > attack_mods else "attack" if attack_mods > spell_mods else "unknown"

    # ── CoC detection ──
    all_gems = [g for sg in char.skill_groups for g in sg.gems]
    has_coc_gem = any("Cast on Crit" in g or "Cast when Crit" in g or "Cast on Critical" in g for g in all_gems)
    if has_coc_gem and best_skill in SPELL_SKILLS:
        has_attack = any(g in ATTACK_SKILLS for g in all_gems)
        if has_attack:
            analysis.is_coc = True
    # Also detect from skill group structure: spell DPS + attack in same group
    if not analysis.is_coc and best_skill in SPELL_SKILLS:
        for sg in char.skill_groups:
            has_dps = any(d.name == best_skill for d in (sg.dps if hasattr(sg, 'dps') and sg.dps else []))
            has_attack = any(g in ATTACK_SKILLS for g in sg.gems)
            if has_dps and has_attack:
                analysis.is_coc = True
                break

    # ── Defense type ──
    ds = char.defensive_stats or {}
    char_life = ds.get("life", 0) or 0
    char_es = ds.get("energyShield", 0) or 0
    has_ci = "Chaos Inoculation" in char.keystones
    has_mom = "Mind Over Matter" in char.keystones

    if has_ci:
        analysis.defense_type = "es"
    elif has_mom:
        analysis.defense_type = "mom"
    elif char_es > char_life * 2 and char_es > 2000:
        analysis.defense_type = "es"
    elif char_es > char_life:
        analysis.defense_type = "hybrid"
    else:
        analysis.defense_type = "life"

    # ── Crit detection ──
    analysis.is_crit = analysis.is_coc or any(
        "crit" in _strip_ninja_brackets(mod).lower()
        for eq in char.equipment
        for mod in (eq.explicit_mods or [])
    )

    # ── POB stats ──
    if pob:
        analysis.ehp = pob.stats.total_ehp
        analysis.life = pob.stats.life
        analysis.es = pob.stats.energy_shield

        # Resist check
        resists = {
            "Fire": pob.stats.fire_resist,
            "Cold": pob.stats.cold_resist,
            "Lightning": pob.stats.lightning_resist,
            "Chaos": pob.stats.chaos_resist,
        }
        uncapped = [f"{n} ({v}%)" for n, v in resists.items() if v < 75]
        analysis.resists_capped = len(uncapped) == 0
        analysis.uncapped_resists = uncapped

        if analysis.ehp < 15000:
            analysis.survival_weaknesses.append(f"Low EHP ({int(analysis.ehp):,})")
        if analysis.life < 3000 and analysis.defense_type == "life":
            analysis.survival_weaknesses.append(f"Low life ({analysis.life:,})")
        if analysis.es < 4000 and analysis.defense_type == "es":
            analysis.survival_weaknesses.append(f"Low ES ({analysis.es:,})")
        if uncapped:
            analysis.survival_weaknesses.append(f"Uncapped resists: {', '.join(uncapped)}")

    # ── Keystones ──
    analysis.keystones_allocated = list(char.keystones)

    # ── Gear analysis ──
    total_gem_levels = 0
    for item in char.equipment:
        if item.slot in ("Flask", "Flask2"):
            continue
        all_mods = (item.explicit_mods or []) + (item.implicit_mods or []) + (item.crafted_mods or [])
        item_name = item.name or item.type_line or item.slot

        dps_pct = 0
        key_dps_mods = []
        for mod in all_mods:
            mc = _strip_ninja_brackets(mod)
            a = _analyze_mod_contribution(mc, arch)
            if a and a[0] == "dps":
                dps_pct += a[2]
                key_dps_mods.append(a[1])
            ml = mc.lower()
            m = re.search(r"\+(\d+) to level of all .*(spell|skill)", ml)
            if m:
                total_gem_levels += int(m.group(1))

        if dps_pct > 20:
            analysis.gear_strengths.append({
                "slot": item.slot, "item": item_name,
                "why": f"~{dps_pct:.0f}% DPS contribution: {', '.join(key_dps_mods[:3])}"
            })

        # Check for wasted mods
        wasted = []
        for mod in all_mods:
            mc = _strip_ninja_brackets(mod).lower()
            if analysis.damage_type == "spell" and ("attack speed" in mc and "local" in mc):
                wasted.append(mod)
            if analysis.damage_type == "attack" and "spell damage" in mc:
                wasted.append(mod)
        if wasted:
            analysis.gear_weaknesses.append({
                "slot": item.slot, "item": item_name,
                "why": f"Wasted mods: {', '.join(_strip_ninja_brackets(w)[:40] for w in wasted)}"
            })

    # ── Recommendations ──
    if analysis.main_dps > 0 and total_gem_levels < 5 and analysis.damage_type == "spell":
        analysis.top_recommendations.append(
            f"Get more +gem levels on gear (currently +{total_gem_levels}). "
            f"Weapons and amulet can have +3-5 each."
        )

    if not analysis.resists_capped:
        analysis.top_recommendations.append(
            f"Cap resistances first: {', '.join(analysis.uncapped_resists)}"
        )

    if analysis.level >= 97 and analysis.main_dps > 0:
        analysis.top_recommendations.append("Use Lv21 main skill gem (corrupt Lv20 with Vaal Orb)")

    if analysis.ehp < 15000 and analysis.defense_type == "life":
        analysis.top_recommendations.append("Increase life on gear — target 4000+ life pool")

    return analysis


# ---------------------------------------------------------------------------
# LAMA analysis extraction
# ---------------------------------------------------------------------------

def lama_analyze(char, arch, engine) -> LamaAnalysis:
    """Run LAMA's why-engine and extract structured results."""
    exps = engine.explain_character(char, arch)
    la = LamaAnalysis()
    la.main_skill = arch.main_skill
    la.damage_type = arch.damage_type
    la.defense_type = arch.defense_type
    la.is_coc = arch.is_coc
    la.is_crit = arch.is_crit

    if exps.scorecard:
        la.scorecard_ehp = exps.scorecard.ehp
        la.scorecard_dps = exps.scorecard.dps_label

    la.actions = [{"title": e.title, "severity": e.severity, "text": e.text[:100]}
                  for e in exps.actions]
    la.keystones = [{"title": e.title, "severity": e.severity}
                    for e in exps.keystones]

    for cat in exps.synergy_map:
        if cat.category == "dps":
            la.synergy_dps_contributors = [
                {"name": c.name, "type": c.source_type, "pct": c.estimated_pct}
                for c in cat.contributors
            ]
            la.synergy_dps_missing = [
                {"name": c.name, "contribution": c.contribution[:80]}
                for c in cat.missing
            ]
        if cat.category == "survival":
            la.synergy_survival_missing = [
                {"name": c.name, "contribution": c.contribution[:80]}
                for c in cat.missing
            ]

    return la


# ---------------------------------------------------------------------------
# Comparison: Claude vs LAMA
# ---------------------------------------------------------------------------

def compare(expert: ExpertAnalysis, lama: LamaAnalysis) -> List[Discrepancy]:
    """Compare expert analysis against LAMA output. Return discrepancies."""
    bugs = []
    build_name = expert.name
    build_tier = expert.tier

    # ── Classification ──
    if expert.damage_type != lama.damage_type and expert.damage_type != "unknown":
        bugs.append(Discrepancy(
            category="classification", severity="critical",
            description=f"Damage type mismatch",
            claude_says=expert.damage_type, lama_says=lama.damage_type,
            build_name=build_name, build_tier=build_tier,
        ))

    if expert.defense_type != lama.defense_type:
        bugs.append(Discrepancy(
            category="classification", severity="major",
            description=f"Defense type mismatch",
            claude_says=expert.defense_type, lama_says=lama.defense_type,
            build_name=build_name, build_tier=build_tier,
        ))

    if expert.is_coc != lama.is_coc:
        bugs.append(Discrepancy(
            category="classification", severity="major",
            description=f"CoC detection mismatch",
            claude_says=str(expert.is_coc), lama_says=str(lama.is_coc),
            build_name=build_name, build_tier=build_tier,
        ))

    if expert.main_skill and lama.main_skill and expert.main_skill.lower() != lama.main_skill.lower():
        bugs.append(Discrepancy(
            category="classification", severity="major",
            description=f"Main skill mismatch",
            claude_says=f"{expert.main_skill} ({expert.main_dps:,.0f} DPS)",
            lama_says=lama.main_skill,
            build_name=build_name, build_tier=build_tier,
        ))

    # ── Ascendancy passive false positives ──
    if expert.ascendancy_passives_count >= 4:
        for action in lama.actions:
            title = action["title"]
            if "Ascendancy:" in title or "Check Ascendancy:" in title:
                node = title.split(": ", 1)[1] if ": " in title else title
                bugs.append(Discrepancy(
                    category="keystone", severity="critical",
                    description=f"LAMA recommends ascendancy passive that player likely already has",
                    claude_says=f"Player has {expert.ascendancy_passives_count} ascendancy points (likely has {node})",
                    lama_says=f"Action: {title}",
                    build_name=build_name, build_tier=build_tier,
                ))
        for ks in lama.keystones:
            if "Ascendancy:" in ks["title"]:
                bugs.append(Discrepancy(
                    category="keystone", severity="critical",
                    description=f"LAMA flags ascendancy passive as missing",
                    claude_says=f"Player has {expert.ascendancy_passives_count} ascendancy points",
                    lama_says=f"Keystone: {ks['title']}",
                    build_name=build_name, build_tier=build_tier,
                ))

    # ── False "Add Life" on ES builds ──
    if expert.defense_type in ("es", "hybrid"):
        for action in lama.actions:
            if "Add Life" in action["title"] and action["severity"] == "critical":
                bugs.append(Discrepancy(
                    category="survival", severity="major",
                    description=f"LAMA tells ES/hybrid build to add life as critical",
                    claude_says=f"Defense type is {expert.defense_type} (life={expert.life}, ES={expert.es})",
                    lama_says=f"Action: {action['title']} ({action['severity']})",
                    build_name=build_name, build_tier=build_tier,
                ))

    # ── DPS analysis accuracy ──
    if expert.main_dps > 0 and lama.scorecard_dps:
        # Parse LAMA's DPS label
        dps_match = re.match(r"([\d.]+)([KkMm])?", lama.scorecard_dps.split()[0] if lama.scorecard_dps else "")
        if dps_match:
            lama_dps = float(dps_match.group(1))
            mult = dps_match.group(2) or ""
            if mult.upper() == "K": lama_dps *= 1000
            elif mult.upper() == "M": lama_dps *= 1_000_000

            # Check if LAMA's DPS is within 20% of expert's
            if abs(lama_dps - expert.main_dps) / max(expert.main_dps, 1) > 0.5:
                bugs.append(Discrepancy(
                    category="dps", severity="major",
                    description=f"DPS value significantly different",
                    claude_says=f"{expert.main_dps:,.0f} DPS ({expert.main_skill})",
                    lama_says=f"{lama.scorecard_dps}",
                    build_name=build_name, build_tier=build_tier,
                ))

    # ── Resist accuracy ──
    if not expert.resists_capped:
        has_resist_action = any("Resist" in a["title"] for a in lama.actions)
        if not has_resist_action:
            bugs.append(Discrepancy(
                category="survival", severity="major",
                description=f"Uncapped resists not flagged",
                claude_says=f"Uncapped: {', '.join(expert.uncapped_resists)}",
                lama_says="No resist action in recommendations",
                build_name=build_name, build_tier=build_tier,
            ))

    # ── Gear analysis gaps ──
    # Check if LAMA identifies gear strengths that expert identified
    if expert.gear_strengths and not lama.synergy_dps_contributors:
        bugs.append(Discrepancy(
            category="gear", severity="minor",
            description=f"Expert found {len(expert.gear_strengths)} strong gear pieces but LAMA shows no DPS contributors",
            claude_says=f"Gear strengths: {[g['slot'] for g in expert.gear_strengths]}",
            lama_says="Empty DPS contributors",
            build_name=build_name, build_tier=build_tier,
        ))

    # ── Too many critical actions for top-tier builds ──
    if expert.tier == "top":
        crit_actions = [a for a in lama.actions if a["severity"] == "critical"]
        if len(crit_actions) > 2:
            bugs.append(Discrepancy(
                category="recommendation", severity="minor",
                description=f"Top-tier build gets {len(crit_actions)} critical actions (should be minimal)",
                claude_says=f"This is a top-tier build ({expert.main_dps:,.0f} DPS)",
                lama_says=f"Critical actions: {[a['title'] for a in crit_actions][:3]}",
                build_name=build_name, build_tier=build_tier,
            ))

    return bugs


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("LAMA ACCURACY EVALUATION")
    print("Claude as expert analyst vs LAMA why-engine")
    print("=" * 60)
    print()

    client = BuildsClient()
    client._snapshot_version = None
    client._snapshot_name = None
    client._cache.clear()
    client._fetch_snapshot_info()
    engine = WhyEngine(client)

    # Build selection: 60% top, 30% mid, 10% low
    # Use featured characters from search — first few are top, middle are mid, last are low
    all_asc = sorted(set(ASCENDANCY_MAP.keys()) - {"Witch Hunter"})

    all_builds = []
    for asc in all_asc:
        data = client._fetch_search(asc, "Comet")
        if not data or not data.get("featuredCharacters"):
            continue
        chars = data["featuredCharacters"]
        total = len(chars)
        if total == 0:
            continue

        # Top 60%: first 2 chars
        for ch in chars[:2]:
            all_builds.append((asc, ch, "top"))
        # Mid 30%: chars from middle
        mid_idx = total // 2
        for ch in chars[mid_idx:mid_idx+1]:
            all_builds.append((asc, ch, "mid"))
        # Low 10%: last char
        if total > 3:
            all_builds.append((asc, chars[-1], "low"))

        time.sleep(0.5)

    print(f"Selected {len(all_builds)} builds across {len(all_asc)} ascendancies")
    print(f"  Top tier: {sum(1 for _, _, t in all_builds if t == 'top')}")
    print(f"  Mid tier: {sum(1 for _, _, t in all_builds if t == 'mid')}")
    print(f"  Low tier: {sum(1 for _, _, t in all_builds if t == 'low')}")
    print()

    all_discrepancies = []
    builds_analyzed = 0
    crashes = 0
    seen = set()

    for asc, ch_info, tier in all_builds:
        acct = ch_info.get("account", "")
        name = ch_info.get("name", "")
        if not acct or not name:
            continue
        key = (acct, name)
        if key in seen:
            continue
        seen.add(key)

        char = client.lookup_character(acct, name)
        if not char:
            continue

        builds_analyzed += 1
        arch = classify_build(char)
        pob = decode_pob_code(char.pob_code) if char.pob_code else None

        # Expert analysis
        expert = expert_analyze(char, pob, arch)
        expert.tier = tier

        # LAMA analysis
        try:
            lama = lama_analyze(char, arch, engine)
        except Exception as e:
            crashes += 1
            all_discrepancies.append(Discrepancy(
                category="crash", severity="critical",
                description=f"LAMA crashed on {name}",
                claude_says="Build should be analyzable",
                lama_says=str(e)[:100],
                build_name=name, build_tier=tier,
            ))
            continue

        # Compare
        bugs = compare(expert, lama)
        all_discrepancies.extend(bugs)

        # Status
        name_safe = name.encode("ascii", "replace").decode()
        status = f"{len(bugs)} issues" if bugs else "MATCH"
        print(f"  {builds_analyzed:3d}. [{tier:3s}] {asc:25s} {name_safe:25s} {status}")

        time.sleep(1)

    # ── Report ──
    print()
    print("=" * 60)
    print(f"RESULTS: {builds_analyzed} builds, {len(all_discrepancies)} discrepancies, {crashes} crashes")
    print("=" * 60)
    print()

    # Group by category
    by_category = defaultdict(list)
    for d in all_discrepancies:
        by_category[d.category].append(d)

    for cat in sorted(by_category.keys()):
        items = by_category[cat]
        # Dedup similar issues
        unique_descriptions = set()
        deduped = []
        for d in items:
            desc_key = f"{d.description}|{d.severity}"
            if desc_key not in unique_descriptions:
                unique_descriptions.add(desc_key)
                deduped.append(d)

        print(f"{cat.upper()} ({len(items)} total, {len(deduped)} unique):")
        for d in deduped[:5]:
            print(f"  [{d.severity:>8}] {d.description}")
            print(f"    Claude: {d.claude_says[:80]}")
            print(f"    LAMA:   {d.lama_says[:80]}")
            print(f"    Build:  {d.build_name} ({d.build_tier})")
        if len(deduped) > 5:
            print(f"  ... +{len(deduped) - 5} more unique issues")
        print()

    # Save full report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "builds_analyzed": builds_analyzed,
        "crashes": crashes,
        "total_discrepancies": len(all_discrepancies),
        "by_category": {
            cat: len(items) for cat, items in by_category.items()
        },
        "discrepancies": [asdict(d) for d in all_discrepancies],
    }

    report_path = os.path.join(os.path.dirname(__file__), "accuracy_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full report saved to: {report_path}")


if __name__ == "__main__":
    main()
