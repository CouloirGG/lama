"""
scaling_analysis.py — Find game-wide scaling themes across all build types.

Pulls builds across all tiers and ascendancies, correlates what actually
drives DPS, survival, and clear speed. Identifies GGG's design patterns
and player optimization patterns.
"""
import sys, os, json, time, re
from collections import defaultdict

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from builds_client import BuildsClient, classify_build, ASCENDANCY_MAP
from pob_decoder import decode_pob_code
from why_engine import _strip_ninja_brackets


def main():
    client = BuildsClient()
    client._fetch_snapshot_info()

    archetypes = [
        'Blood Mage', 'Pathfinder', 'Oracle', 'Stormweaver', 'Invoker',
        'Titan', 'Deadeye', 'Ritualist', 'Amazon', 'Witchhunter',
        'Infernalist', 'Lich', 'Acolyte of Chayula', 'Gemling Legionnaire',
    ]

    all_builds = []
    print("Collecting builds across all tiers...")

    for asc in archetypes:
        data = client._fetch_search(asc, 'Comet')
        if not data or not data.get('featuredCharacters'):
            continue

        chars = data['featuredCharacters']
        total = len(chars)
        indices = sorted(set(min(i, total-1) for i in [0, total//4, total//2, total*3//4, total-1]))

        for idx in indices:
            ch = chars[idx]
            acct, name = ch.get('account', ''), ch.get('name', '')
            if not acct or not name:
                continue

            char = client.lookup_character(acct, name)
            if not char:
                continue

            pob = decode_pob_code(char.pob_code) if char.pob_code else None
            arch = classify_build(char)

            main_dps, main_skill = 0, ''
            for sg in char.skill_groups:
                for d in (sg.dps if hasattr(sg, 'dps') and sg.dps else []):
                    t = max(d.dps or 0, d.dot_dps or 0, d.damage or 0)
                    if t > main_dps:
                        main_dps, main_skill = t, d.name or ''

            gem_levels = 0
            extra_as_pct = 0
            spell_dmg_pct = 0
            crit_chance_pct = 0
            cast_speed_pct = 0
            for item in char.equipment:
                if item.slot in ('Flask', 'Flask2'):
                    continue
                for mod in (item.explicit_mods or []) + (item.implicit_mods or []) + (item.crafted_mods or []):
                    mc = _strip_ninja_brackets(mod).lower()
                    m = re.search(r'\+(\d+) to level of all .*(spell|skill)', mc)
                    if m: gem_levels += int(m.group(1))
                    m = re.search(r'(\d+)% of damage as extra', mc)
                    if m: extra_as_pct += int(m.group(1))
                    m = re.search(r'(\d+)% increased .*spell.*damage', mc)
                    if m: spell_dmg_pct += int(m.group(1))
                    m = re.search(r'(\d+)% increased .*critical.*hit.*chance', mc)
                    if m: crit_chance_pct += int(m.group(1))
                    m = re.search(r'(\d+)% increased cast speed', mc)
                    if m: cast_speed_pct += int(m.group(1))

            jewels = getattr(char, 'jewels', [])
            has_adorned = any('adorned' in (j.name or '').lower() for j in jewels)

            support_count = 0
            for sg in char.skill_groups:
                if any(d.name == main_skill for d in (sg.dps if hasattr(sg, 'dps') and sg.dps else [])):
                    support_count = len([g for g in sg.gems if g != main_skill])
                    break

            ehp = int(pob.stats.total_ehp) if pob else 0
            life = pob.stats.life if pob else 0
            es = pob.stats.energy_shield if pob else 0
            unique_count = sum(1 for eq in char.equipment if eq.rarity == 'Unique' and eq.slot not in ('Flask', 'Flask2'))

            tier = 'top' if idx == 0 else 'high' if idx <= total//4 else 'mid' if idx <= total//2 else 'low'

            all_builds.append({
                'name': name, 'ascendancy': asc, 'level': char.level, 'tier': tier,
                'main_skill': main_skill, 'main_dps': main_dps,
                'damage_type': arch.damage_type, 'defense_type': arch.defense_type,
                'ehp': ehp, 'life': life, 'es': es,
                'gem_levels': gem_levels, 'extra_as_pct': extra_as_pct,
                'spell_dmg_pct': spell_dmg_pct, 'crit_chance_pct': crit_chance_pct,
                'cast_speed_pct': cast_speed_pct,
                'jewel_count': len(jewels), 'has_adorned': has_adorned,
                'support_count': support_count, 'unique_count': unique_count,
                'keystones': list(char.keystones),
                'ascendancy_points': getattr(char, 'ascendancy_points', 0),
            })
            time.sleep(0.5)

        time.sleep(0.5)

    print(f"\nCollected {len(all_builds)} builds across {len(archetypes)} ascendancies")

    # ═══════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════

    builds_with_dps = [b for b in all_builds if b['main_dps'] > 1000]
    builds_with_dps.sort(key=lambda x: x['main_dps'], reverse=True)

    if len(builds_with_dps) < 8:
        print("Not enough builds with DPS data for analysis")
        return

    q = len(builds_with_dps) // 4
    top_q = builds_with_dps[:q]
    bot_q = builds_with_dps[-q:]

    def avg(lst, key):
        vals = [b[key] for b in lst if b.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0

    def pct_true(lst, key):
        return sum(1 for b in lst if b.get(key)) / len(lst) * 100 if lst else 0

    print("\n" + "=" * 70)
    print("WHAT DRIVES DPS: TOP 25% vs BOTTOM 25%")
    print("=" * 70)
    fmt = "{:<35s} {:>12s} {:>12s} {:>12s}"
    print(fmt.format("Metric", "Top 25%", "Bot 25%", "Ratio"))
    print("-" * 71)

    for label, key in [
        ("Avg DPS", "main_dps"),
        ("Avg +Gem Levels from Gear", "gem_levels"),
        ("Avg % Extra As Elemental", "extra_as_pct"),
        ("Avg % Spell Damage", "spell_dmg_pct"),
        ("Avg % Crit Chance", "crit_chance_pct"),
        ("Avg % Cast Speed", "cast_speed_pct"),
        ("Avg Jewel Count", "jewel_count"),
        ("Avg Support Gems", "support_count"),
        ("Avg Unique Items", "unique_count"),
        ("Avg Level", "level"),
        ("Avg EHP", "ehp"),
    ]:
        t = avg(top_q, key)
        b = avg(bot_q, key)
        ratio = f"{t/b:.1f}x" if b > 0 else "inf"
        print(fmt.format(label, f"{t:,.1f}", f"{b:,.1f}", ratio))

    print()
    for label, key in [
        ("% with The Adorned", "has_adorned"),
    ]:
        t = pct_true(top_q, key)
        b = pct_true(bot_q, key)
        print(fmt.format(label, f"{t:.0f}%", f"{b:.0f}%", f"+{t-b:.0f}%"))

    # Keystone analysis
    print("\n" + "=" * 70)
    print("KEYSTONE USAGE: TOP vs BOTTOM")
    print("=" * 70)
    for label, group in [("Top 25%", top_q), ("Bottom 25%", bot_q)]:
        ks = defaultdict(int)
        for b in group:
            for k in b['keystones']:
                ks[k] += 1
        print(f"\n{label} ({len(group)} builds):")
        for k, c in sorted(ks.items(), key=lambda x: -x[1])[:8]:
            print(f"  {k:35s} {c}/{len(group)} ({c/len(group)*100:.0f}%)")

    # Per-ascendancy DPS range
    print("\n" + "=" * 70)
    print("DPS RANGE BY ASCENDANCY")
    print("=" * 70)
    by_asc = defaultdict(list)
    for b in builds_with_dps:
        by_asc[b['ascendancy']].append(b['main_dps'])
    for asc in sorted(by_asc.keys()):
        vals = sorted(by_asc[asc], reverse=True)
        if vals:
            print(f"  {asc:25s} max={vals[0]:>12,.0f}  min={vals[-1]:>10,.0f}  spread={vals[0]/max(vals[-1],1):.0f}x")

    # ═══════════════════════════════════════════════════
    # FINDINGS: Game-wide themes
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("GAME-WIDE SCALING THEMES")
    print("=" * 70)

    # Calculate correlation strength
    themes = []

    top_gem = avg(top_q, 'gem_levels')
    bot_gem = avg(bot_q, 'gem_levels')
    if bot_gem > 0:
        themes.append((top_gem/bot_gem, "+Gem Levels", f"Top={top_gem:.1f} vs Bot={bot_gem:.1f}"))

    top_extra = avg(top_q, 'extra_as_pct')
    bot_extra = avg(bot_q, 'extra_as_pct')
    if bot_extra > 0:
        themes.append((top_extra/bot_extra, "% Extra As Elemental Damage", f"Top={top_extra:.0f}% vs Bot={bot_extra:.0f}%"))

    top_jewel = avg(top_q, 'jewel_count')
    bot_jewel = avg(bot_q, 'jewel_count')
    if bot_jewel > 0:
        themes.append((top_jewel/bot_jewel, "Jewel Count", f"Top={top_jewel:.1f} vs Bot={bot_jewel:.1f}"))

    top_spell = avg(top_q, 'spell_dmg_pct')
    bot_spell = avg(bot_q, 'spell_dmg_pct')
    if bot_spell > 0:
        themes.append((top_spell/bot_spell, "% Spell Damage on Gear", f"Top={top_spell:.0f}% vs Bot={bot_spell:.0f}%"))

    top_crit = avg(top_q, 'crit_chance_pct')
    bot_crit = avg(bot_q, 'crit_chance_pct')
    if bot_crit > 0:
        themes.append((top_crit/bot_crit, "% Crit Chance on Gear", f"Top={top_crit:.0f}% vs Bot={bot_crit:.0f}%"))

    themes.sort(reverse=True)

    print("\nScaling factors ranked by correlation with DPS (top vs bottom):\n")
    for i, (ratio, name, detail) in enumerate(themes):
        print(f"  {i+1}. {name:40s} {ratio:.1f}x  ({detail})")

    # Save
    with open(os.path.join(os.path.dirname(__file__), 'scaling_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump({'builds': all_builds, 'count': len(all_builds)}, f, indent=2, ensure_ascii=False)
    print(f"\nData saved to scripts/scaling_analysis.json")


if __name__ == "__main__":
    main()
