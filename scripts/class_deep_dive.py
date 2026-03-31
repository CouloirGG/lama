"""
class_deep_dive.py — Deep analysis of every base class and ascendancy.

Pulls extensive builds per class with class-appropriate skills,
analyzes scaling factors, keystones, unique items, support gems,
and jewel patterns specific to each class.
"""
import sys, os, json, time, re
from collections import defaultdict, Counter

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from builds_client import BuildsClient, classify_build, ASCENDANCY_MAP
from pob_decoder import decode_pob_code
from why_engine import _strip_ninja_brackets


def extract_full(client, char):
    """Extract comprehensive stats from a character."""
    pob = decode_pob_code(char.pob_code) if char.pob_code else None
    arch = classify_build(char)

    main_dps, main_skill = 0, ''
    for sg in char.skill_groups:
        for d in (sg.dps if hasattr(sg, 'dps') and sg.dps else []):
            t = max(d.dps or 0, d.dot_dps or 0, d.damage or 0)
            if t > main_dps:
                main_dps, main_skill = t, d.name or ''

    # Gear mod totals
    stats = defaultdict(int)
    for item in char.equipment:
        if item.slot in ('Flask', 'Flask2'):
            continue
        for mod in (item.explicit_mods or []) + (item.implicit_mods or []) + (item.crafted_mods or []):
            mc = _strip_ninja_brackets(mod).lower()
            for pattern, key in [
                (r'\+(\d+) to level of all .*(spell|skill)', 'gem_levels'),
                (r'(\d+)% of damage as extra', 'extra_as'),
                (r'(\d+)% increased .*spell.*damage', 'spell_dmg'),
                (r'(\d+)% increased .*physical.*damage', 'phys_dmg'),
                (r'(\d+)% increased attack speed', 'atk_speed'),
                (r'(\d+)% increased cast speed', 'cast_speed'),
                (r'(\d+)% increased .*critical.*hit.*chance', 'crit_chance'),
                (r'(\d+)% increased .*critical.*damage', 'crit_multi'),
                (r'\+(\d+) to maximum life', 'life_flat'),
                (r'\+(\d+) to maximum energy shield', 'es_flat'),
                (r'(\d+)% increased.*energy shield', 'es_pct'),
                (r'\+(\d+)% to .*resist', 'res'),
                (r'\+(\d+) to spirit', 'spirit'),
                (r'(\d+)% increased .*evasion', 'evasion_pct'),
                (r'(\d+)% increased .*armour', 'armour_pct'),
            ]:
                m = re.search(pattern, mc)
                if m:
                    stats[key] += int(m.group(1))

    jewels = getattr(char, 'jewels', [])
    jewel_names = [j.name or j.type_line for j in jewels if j.name or j.type_line]

    supports = []
    for sg in char.skill_groups:
        if any(d.name == main_skill for d in (sg.dps if hasattr(sg, 'dps') and sg.dps else [])):
            supports = [g for g in sg.gems if g != main_skill]
            break

    uniques = [(eq.slot, eq.name) for eq in char.equipment
               if eq.rarity == 'Unique' and eq.slot not in ('Flask', 'Flask2') and eq.name]

    return {
        'name': char.name, 'ascendancy': char.ascendancy,
        'base_class': ASCENDANCY_MAP.get(char.ascendancy, '?'),
        'level': char.level, 'main_skill': main_skill, 'main_dps': main_dps,
        'damage_type': arch.damage_type, 'defense_type': arch.defense_type,
        'is_coc': arch.is_coc, 'is_crit': arch.is_crit,
        'ehp': int(pob.stats.total_ehp) if pob else 0,
        'life': pob.stats.life if pob else 0,
        'es': pob.stats.energy_shield if pob else 0,
        'jewel_count': len(jewels), 'jewel_names': jewel_names,
        'has_adorned': any('adorned' in n.lower() for n in jewel_names),
        'support_count': len(supports), 'supports': supports[:6],
        'unique_count': len(uniques), 'uniques': uniques,
        'keystones': list(char.keystones),
        'asc_points': getattr(char, 'ascendancy_points', 0),
        **{k: v for k, v in stats.items()},
    }


def analyze_class(builds, class_name):
    """Deep analysis of a single class."""
    builds_dps = sorted([b for b in builds if b['main_dps'] > 1000], key=lambda x: -x['main_dps'])
    if len(builds_dps) < 4:
        return None

    q = max(2, len(builds_dps) // 3)
    top, bot = builds_dps[:q], builds_dps[-q:]

    def avg(lst, k):
        vals = [b.get(k, 0) for b in lst]
        return sum(vals) / len(vals) if vals else 0

    # Find top scaling factors
    stat_keys = [
        'gem_levels', 'extra_as', 'spell_dmg', 'phys_dmg', 'atk_speed',
        'cast_speed', 'crit_chance', 'crit_multi', 'life_flat', 'es_flat',
        'es_pct', 'spirit', 'evasion_pct', 'armour_pct', 'jewel_count',
        'support_count', 'unique_count',
    ]

    factors = []
    for key in stat_keys:
        t, b = avg(top, key), avg(bot, key)
        ratio = t / b if b > 0 else (99.0 if t > 0 else 1.0)
        if ratio > 1.05:
            factors.append((ratio, key, t, b))
    factors.sort(reverse=True)

    # Keystones
    ks_top, ks_bot = Counter(), Counter()
    for b in top:
        for k in b['keystones']: ks_top[k] += 1
    for b in bot:
        for k in b['keystones']: ks_bot[k] += 1

    # Unique items
    unique_counter = Counter()
    for b in builds:
        for slot, uname in b.get('uniques', []):
            unique_counter[f"{slot}: {uname}"] += 1

    # Support gems
    sup_top, sup_bot = Counter(), Counter()
    for b in top:
        for s in b.get('supports', []): sup_top[s] += 1
    for b in bot:
        for s in b.get('supports', []): sup_bot[s] += 1

    # Main skills
    skill_counter = Counter()
    skill_dps = defaultdict(list)
    for b in builds_dps:
        if b['main_skill']:
            skill_counter[b['main_skill']] += 1
            skill_dps[b['main_skill']].append(b['main_dps'])

    # Defense types
    def_counter = Counter(b['defense_type'] for b in builds)

    return {
        'class': class_name,
        'total_builds': len(builds),
        'dps_builds': len(builds_dps),
        'dps_range': (builds_dps[0]['main_dps'], builds_dps[-1]['main_dps']),
        'top_factors': [(r, k, t, b) for r, k, t, b in factors[:8]],
        'keystones_top': dict(ks_top.most_common(8)),
        'keystones_bot': dict(ks_bot.most_common(8)),
        'top_count': len(top),
        'bot_count': len(bot),
        'popular_uniques': dict(unique_counter.most_common(10)),
        'supports_top': dict(sup_top.most_common(8)),
        'supports_bot': dict(sup_bot.most_common(8)),
        'main_skills': {sk: {'count': c, 'avg_dps': sum(skill_dps[sk])/c}
                        for sk, c in skill_counter.most_common(8)},
        'defense_types': dict(def_counter),
        'avg_jewels_top': avg(top, 'jewel_count'),
        'avg_jewels_bot': avg(bot, 'jewel_count'),
    }


def print_class_report(analysis):
    """Print formatted class report."""
    a = analysis
    print(f"\n{'='*70}")
    print(f"{a['class'].upper()} ({a['dps_builds']} builds with DPS)")
    print(f"DPS range: {a['dps_range'][0]:,.0f} — {a['dps_range'][1]:,.0f} ({a['dps_range'][0]/max(a['dps_range'][1],1):.0f}x spread)")
    print(f"{'='*70}")

    print(f"\nScaling factors (top vs bottom 1/3):")
    for ratio, key, t, b in a['top_factors']:
        label = key.replace('_', ' ').title()
        if ratio >= 99:
            print(f"  {label:30s} top={t:.1f}  bot=0  (top-only)")
        else:
            print(f"  {label:30s} {ratio:.1f}x  (top={t:.1f}, bot={b:.1f})")

    print(f"\nKeystones (top {a['top_count']} vs bottom {a['bot_count']}):")
    all_ks = set(list(a['keystones_top'].keys()) + list(a['keystones_bot'].keys()))
    for k in sorted(all_ks, key=lambda x: -(a['keystones_top'].get(x,0))):
        tc = a['keystones_top'].get(k, 0)
        bc = a['keystones_bot'].get(k, 0)
        print(f"  {k:35s} top={tc} ({tc/a['top_count']*100:.0f}%)  bot={bc} ({bc/a['bot_count']*100:.0f}%)")

    print(f"\nMain skills:")
    for sk, info in a['main_skills'].items():
        print(f"  {sk:30s} {info['count']:3d} builds  avg DPS={info['avg_dps']:>12,.0f}")

    print(f"\nDefense types: {a['defense_types']}")

    print(f"\nPopular uniques:")
    for item, count in list(a['popular_uniques'].items())[:8]:
        pct = count / a['total_builds'] * 100
        print(f"  {item:45s} {count:3d} ({pct:.0f}%)")

    print(f"\nTop support gems (top builds):")
    for gem, count in list(a['supports_top'].items())[:6]:
        bc = a['supports_bot'].get(gem, 0)
        print(f"  {gem:35s} top={count}  bot={bc}")

    print(f"\nJewels: top avg={a['avg_jewels_top']:.1f}  bot avg={a['avg_jewels_bot']:.1f}")


def main():
    client = BuildsClient()
    client._fetch_snapshot_info()

    # Comprehensive skill lists per class
    class_skills = {
        'Warrior': [
            ('Titan', 'Slam'), ('Titan', 'Earthquake'), ('Titan', 'Cyclone'),
            ('Warbringer', 'Slam'), ('Warbringer', 'Cyclone'), ('Warbringer', 'Punch'),
            ('Smith of Kitava', 'Slam'), ('Smith of Kitava', 'Cyclone'),
            ('Smith of Kitava', 'Comet'),
        ],
        'Witch': [
            ('Blood Mage', 'Comet'), ('Blood Mage', 'Spark'), ('Blood Mage', 'Unearth'),
            ('Infernalist', 'Comet'), ('Infernalist', 'Spark'), ('Infernalist', 'Detonate Dead'),
            ('Lich', 'Bone Storm'), ('Lich', 'Comet'), ('Lich', 'Unearth'),
        ],
        'Ranger': [
            ('Pathfinder', 'Poisonburst Arrow'), ('Pathfinder', 'Vine Arrow'),
            ('Pathfinder', 'Comet'), ('Pathfinder', 'Gas Arrow'),
            ('Deadeye', 'Lightning Arrow'), ('Deadeye', 'Barrage'),
            ('Deadeye', 'Ice Shot'),
        ],
        'Sorceress': [
            ('Stormweaver', 'Spark'), ('Stormweaver', 'Comet'), ('Stormweaver', 'Arc'),
            ('Chronomancer', 'Comet'), ('Chronomancer', 'Spark'),
            ('Disciple of Varashta', 'Comet'), ('Disciple of Varashta', 'Fire Storm'),
            ('Disciple of Varashta', 'Spark'),
        ],
        'Monk': [
            ('Invoker', 'Bell'), ('Invoker', 'Flicker Strike'), ('Invoker', 'Comet'),
            ('Invoker', 'Ice Strike'),
            ('Acolyte of Chayula', 'Bell'), ('Acolyte of Chayula', 'Ice Strike'),
            ('Acolyte of Chayula', 'Flicker Strike'),
        ],
        'Mercenary': [
            ('Tactician', 'Power Shot'), ('Tactician', 'Lightning Arrow'),
            ('Witchhunter', 'Power Shot'), ('Witchhunter', 'Ice Shot'),
            ('Gemling Legionnaire', 'Comet'), ('Gemling Legionnaire', 'Spark'),
        ],
        'Huntress': [
            ('Amazon', 'Barrage'), ('Amazon', 'Lightning Arrow'),
            ('Amazon', 'Poisonburst Arrow'), ('Amazon', 'Cyclone'),
            ('Ritualist', 'Bone Storm'), ('Ritualist', 'Comet'),
            ('Ritualist', 'Unearth'),
        ],
        'Druid': [
            ('Oracle', 'Bone Storm'), ('Oracle', 'Storm Call'), ('Oracle', 'Comet'),
            ('Shaman', 'Tornado'), ('Shaman', 'Storm Call'), ('Shaman', 'Lightning Bolt'),
        ],
    }

    all_results = {}
    all_builds = []

    for base_class, combos in class_skills.items():
        class_builds = []
        print(f"\nCollecting {base_class}...")

        for asc, skill in combos:
            data = client._fetch_search(asc, skill)
            if not data or not data.get('featuredCharacters'):
                continue

            chars = data['featuredCharacters']
            total = len(chars)
            indices = sorted(set(min(i, total-1) for i in
                [0, 1, 2, total//4, total//2, total*3//4, total-1]))

            seen = set()
            for idx in indices:
                ch = chars[idx]
                acct, name = ch.get('account', ''), ch.get('name', '')
                if not acct or not name or name in seen:
                    continue
                seen.add(name)

                char = client.lookup_character(acct, name)
                if not char:
                    continue

                bd = extract_full(client, char)
                class_builds.append(bd)
                all_builds.append(bd)
                time.sleep(0.4)

            time.sleep(0.3)

        print(f"  {base_class}: {len(class_builds)} builds")
        analysis = analyze_class(class_builds, base_class)
        if analysis:
            all_results[base_class] = analysis
            print_class_report(analysis)

    # Global analysis
    global_analysis = analyze_class(all_builds, "ALL CLASSES (GLOBAL)")
    if global_analysis:
        all_results['GLOBAL'] = global_analysis
        print_class_report(global_analysis)

    # Save
    out_path = os.path.join(os.path.dirname(__file__), 'class_deep_dive.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'build_count': len(all_builds),
            'classes': {k: {kk: vv for kk, vv in v.items() if kk != 'builds'}
                        for k, v in all_results.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
