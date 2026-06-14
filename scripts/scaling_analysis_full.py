"""
scaling_analysis_full.py — Comprehensive scaling analysis across ALL classes.

Analyzes what drives DPS, survival, and clear speed for each base class
and archetype separately, not just spell builds. Includes gear mods,
keystones, jewels, and perk tree patterns.
"""
import sys, os, json, time, re
from collections import defaultdict, Counter

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from builds_client import BuildsClient, classify_build, ASCENDANCY_MAP
from pob_decoder import decode_pob_code
from why_engine import _strip_ninja_brackets


def extract_build_data(client, char):
    """Extract all measurable stats from a character."""
    pob = decode_pob_code(char.pob_code) if char.pob_code else None
    arch = classify_build(char)

    main_dps, main_skill, is_dot = 0, '', False
    all_skill_dps = []
    for sg in char.skill_groups:
        for d in (sg.dps if hasattr(sg, 'dps') and sg.dps else []):
            t = max(d.dps or 0, d.dot_dps or 0, d.damage or 0)
            if t > 0:
                all_skill_dps.append((d.name or '?', t, (d.dot_dps or 0) > (d.dps or 0)))
            if t > main_dps:
                main_dps, main_skill = t, d.name or ''
                is_dot = (d.dot_dps or 0) > (d.dps or 0)

    # Gear mod analysis
    gem_levels = 0
    extra_as_pct = 0
    spell_dmg_pct = 0
    phys_dmg_pct = 0
    attack_speed_pct = 0
    cast_speed_pct = 0
    crit_chance_pct = 0
    crit_multi_pct = 0
    life_flat = 0
    es_flat = 0
    res_total = 0

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
            m = re.search(r'(\d+)% increased .*physical.*damage', mc)
            if m: phys_dmg_pct += int(m.group(1))
            m = re.search(r'(\d+)% increased attack speed', mc)
            if m: attack_speed_pct += int(m.group(1))
            m = re.search(r'(\d+)% increased cast speed', mc)
            if m: cast_speed_pct += int(m.group(1))
            m = re.search(r'(\d+)% increased .*critical.*hit.*chance', mc)
            if m: crit_chance_pct += int(m.group(1))
            m = re.search(r'(\d+)% increased .*critical.*damage', mc)
            if m: crit_multi_pct += int(m.group(1))
            m = re.search(r'\+(\d+) to maximum life', mc)
            if m: life_flat += int(m.group(1))
            m = re.search(r'\+(\d+) to maximum energy shield', mc)
            if m: es_flat += int(m.group(1))
            m = re.search(r'\+(\d+)% to .*resist', mc)
            if m: res_total += int(m.group(1))

    jewels = getattr(char, 'jewels', [])
    has_adorned = any('adorned' in (j.name or '').lower() for j in jewels)
    has_megalomaniac = any('megalomaniac' in (j.name or '').lower() for j in jewels)
    unique_count = sum(1 for eq in char.equipment if eq.rarity == 'Unique' and eq.slot not in ('Flask', 'Flask2'))

    # Support gem count for main skill
    support_count = 0
    for sg in char.skill_groups:
        if any(d.name == main_skill for d in (sg.dps if hasattr(sg, 'dps') and sg.dps else [])):
            support_count = len([g for g in sg.gems if g != main_skill])
            break

    return {
        'name': char.name,
        'ascendancy': char.ascendancy,
        'base_class': ASCENDANCY_MAP.get(char.ascendancy, '?'),
        'level': char.level,
        'main_skill': main_skill,
        'main_dps': main_dps,
        'is_dot': is_dot,
        'damage_type': arch.damage_type,
        'defense_type': arch.defense_type,
        'is_coc': arch.is_coc,
        'is_crit': arch.is_crit,
        'ehp': int(pob.stats.total_ehp) if pob else 0,
        'life': pob.stats.life if pob else 0,
        'es': pob.stats.energy_shield if pob else 0,
        'gem_levels': gem_levels,
        'extra_as_pct': extra_as_pct,
        'spell_dmg_pct': spell_dmg_pct,
        'phys_dmg_pct': phys_dmg_pct,
        'attack_speed_pct': attack_speed_pct,
        'cast_speed_pct': cast_speed_pct,
        'crit_chance_pct': crit_chance_pct,
        'crit_multi_pct': crit_multi_pct,
        'life_flat': life_flat,
        'es_flat': es_flat,
        'res_total': res_total,
        'jewel_count': len(jewels),
        'has_adorned': has_adorned,
        'has_megalomaniac': has_megalomaniac,
        'support_count': support_count,
        'unique_count': unique_count,
        'keystones': list(char.keystones),
        'asc_points': getattr(char, 'ascendancy_points', 0),
    }


def analyze_group(builds, label):
    """Analyze a group of builds and print findings."""
    if len(builds) < 4:
        return

    builds_with_dps = sorted([b for b in builds if b['main_dps'] > 1000], key=lambda x: -x['main_dps'])
    if len(builds_with_dps) < 4:
        return

    q = max(1, len(builds_with_dps) // 3)
    top = builds_with_dps[:q]
    bot = builds_with_dps[-q:]

    def avg(lst, key):
        vals = [b[key] for b in lst if b.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0

    print(f"\n{'=' * 70}")
    print(f"{label} ({len(builds_with_dps)} builds with DPS)")
    print(f"DPS range: {builds_with_dps[0]['main_dps']:,.0f} — {builds_with_dps[-1]['main_dps']:,.0f}")
    print(f"{'=' * 70}")

    # Find which metrics have the biggest top/bottom difference
    metrics = [
        ("Cast Speed %", "cast_speed_pct"),
        ("Attack Speed %", "attack_speed_pct"),
        ("Crit Chance %", "crit_chance_pct"),
        ("Crit Multi %", "crit_multi_pct"),
        ("Spell Damage %", "spell_dmg_pct"),
        ("Physical Damage %", "phys_dmg_pct"),
        ("Extra As Elemental %", "extra_as_pct"),
        ("+Gem Levels", "gem_levels"),
        ("Jewel Count", "jewel_count"),
        ("Support Gems", "support_count"),
        ("Unique Items", "unique_count"),
        ("Flat Life on Gear", "life_flat"),
        ("Flat ES on Gear", "es_flat"),
        ("EHP", "ehp"),
    ]

    ratios = []
    for label_m, key in metrics:
        t = avg(top, key)
        b = avg(bot, key)
        ratio = t / b if b > 0 else (99 if t > 0 else 1)
        ratios.append((ratio, label_m, t, b))

    ratios.sort(reverse=True)

    print(f"\nTop scaling factors (top 1/3 vs bottom 1/3):")
    fmt = "  {:<30s} {:>10s} {:>10s} {:>8s}"
    print(fmt.format("Factor", "Top", "Bottom", "Ratio"))
    for ratio, name, t, b in ratios[:8]:
        if ratio > 1.05:
            print(fmt.format(name, f"{t:.1f}", f"{b:.1f}", f"{ratio:.1f}x"))

    # Keystones
    ks_top = Counter()
    ks_bot = Counter()
    for b in top:
        for k in b['keystones']: ks_top[k] += 1
    for b in bot:
        for k in b['keystones']: ks_bot[k] += 1

    if ks_top:
        print(f"\nTop builds keystones:")
        for k, c in ks_top.most_common(5):
            bot_c = ks_bot.get(k, 0)
            print(f"  {k:35s} top={c}/{len(top)} ({c/len(top)*100:.0f}%)  bot={bot_c}/{len(bot)} ({bot_c/len(bot)*100:.0f}%)")

    # Main skills
    skill_counts = Counter(b['main_skill'] for b in builds_with_dps if b['main_skill'])
    print(f"\nMain skills used:")
    for sk, c in skill_counts.most_common(5):
        avg_dps = sum(b['main_dps'] for b in builds_with_dps if b['main_skill'] == sk) / max(c, 1)
        print(f"  {sk:30s} {c} builds  avg DPS={avg_dps:,.0f}")


def main():
    client = BuildsClient()
    client._fetch_snapshot_info()

    # Search per base class with class-appropriate skills
    class_skills = {
        'Warrior': [('Titan', 'Slam'), ('Titan', 'Earthquake'), ('Warbringer', 'Slam'),
                    ('Smith of Kitava', 'Slam'), ('Smith of Kitava', 'Cyclone')],
        'Witch': [('Blood Mage', 'Comet'), ('Blood Mage', 'Spark'),
                  ('Infernalist', 'Comet'), ('Lich', 'Bone Storm')],
        'Ranger': [('Pathfinder', 'Poisonburst Arrow'), ('Pathfinder', 'Vine Arrow'),
                   ('Deadeye', 'Lightning Arrow'), ('Deadeye', 'Barrage')],
        'Sorceress': [('Stormweaver', 'Spark'), ('Stormweaver', 'Comet'),
                      ('Chronomancer', 'Comet'), ('Disciple of Varashta', 'Comet')],
        'Monk': [('Invoker', 'Bell'), ('Invoker', 'Flicker Strike'),
                 ('Acolyte of Chayula', 'Bell'), ('Acolyte of Chayula', 'Ice Strike')],
        'Mercenary': [('Tactician', 'Power Shot'), ('Witchhunter', 'Power Shot'),
                      ('Gemling Legionnaire', 'Comet')],
        'Huntress': [('Amazon', 'Barrage'), ('Amazon', 'Lightning Arrow'),
                     ('Ritualist', 'Bone Storm')],
        'Druid': [('Oracle', 'Bone Storm'), ('Oracle', 'Storm Call'),
                  ('Shaman', 'Tornado'), ('Shaman', 'Storm Call')],
    }

    all_builds = []
    builds_by_class = defaultdict(list)
    builds_by_type = defaultdict(list)  # spell, attack, etc.

    print("Collecting builds across all base classes...")

    for base_class, combos in class_skills.items():
        class_builds = 0
        for asc, skill in combos:
            data = client._fetch_search(asc, skill)
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

                bd = extract_build_data(client, char)
                all_builds.append(bd)
                builds_by_class[base_class].append(bd)
                builds_by_type[bd['damage_type']].append(bd)
                class_builds += 1
                time.sleep(0.5)

            time.sleep(0.5)

        ns = base_class
        print(f"  {ns}: {class_builds} builds collected")

    print(f"\nTotal: {len(all_builds)} builds across {len(builds_by_class)} base classes")

    # ═══════════════════════════════════════════════════
    # GLOBAL ANALYSIS
    # ═══════════════════════════════════════════════════
    analyze_group(all_builds, "ALL BUILDS (GLOBAL)")

    # ═══════════════════════════════════════════════════
    # PER-CLASS ANALYSIS
    # ═══════════════════════════════════════════════════
    for base_class in ['Warrior', 'Witch', 'Ranger', 'Sorceress', 'Monk', 'Mercenary', 'Huntress', 'Druid']:
        builds = builds_by_class.get(base_class, [])
        if builds:
            analyze_group(builds, f"{base_class.upper()} CLASS")

    # ═══════════════════════════════════════════════════
    # PER-DAMAGE-TYPE ANALYSIS
    # ═══════════════════════════════════════════════════
    for dmg_type in ['spell', 'attack', 'minion']:
        builds = builds_by_type.get(dmg_type, [])
        if builds:
            analyze_group(builds, f"{dmg_type.upper()} BUILDS")

    # ═══════════════════════════════════════════════════
    # CROSS-CLASS THEMES
    # ═══════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("CROSS-CLASS THEMES")
    print(f"{'=' * 70}")

    # Defense meta
    def_counts = Counter(b['defense_type'] for b in all_builds)
    print(f"\nDefense type distribution:")
    for dt, c in def_counts.most_common():
        print(f"  {dt:15s} {c}/{len(all_builds)} ({c/len(all_builds)*100:.0f}%)")

    # Keystone meta across ALL builds
    ks_all = Counter()
    for b in all_builds:
        for k in b['keystones']:
            ks_all[k] += 1
    print(f"\nMost popular keystones (all classes):")
    for k, c in ks_all.most_common(10):
        print(f"  {k:35s} {c}/{len(all_builds)} ({c/len(all_builds)*100:.0f}%)")

    # Jewel count by tier
    dps_sorted = sorted([b for b in all_builds if b['main_dps'] > 1000], key=lambda x: -x['main_dps'])
    if len(dps_sorted) >= 6:
        top_third = dps_sorted[:len(dps_sorted)//3]
        bot_third = dps_sorted[-(len(dps_sorted)//3):]
        t_jwl = sum(b['jewel_count'] for b in top_third) / len(top_third)
        b_jwl = sum(b['jewel_count'] for b in bot_third) / len(bot_third)
        print(f"\nJewel count: top 1/3 avg={t_jwl:.1f}, bottom 1/3 avg={b_jwl:.1f} ({t_jwl/max(b_jwl,0.1):.1f}x)")

    # Save
    with open(os.path.join(os.path.dirname(__file__), 'scaling_analysis_full.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'builds': all_builds,
            'count': len(all_builds),
            'by_class': {k: len(v) for k, v in builds_by_class.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"\nData saved to scripts/scaling_analysis_full.json")


if __name__ == "__main__":
    main()
