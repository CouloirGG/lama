"""
meta_harvester.py — Meta data harvester for LAMA.

Collects build data from poe.ninja for all ascendancies, runs scaling
analysis (top 1/3 vs bottom 1/3), and outputs a compressed meta_shard.json.gz
that the LAMA app loads at runtime to drive data-driven analysis.

Usage:
    python scripts/meta_harvester.py
    python scripts/meta_harvester.py --league "Runes of Aldur" --output shard.json.gz
    python scripts/meta_harvester.py --dry-run
"""
import sys, os, json, gzip, time, re, argparse, statistics
from collections import defaultdict, Counter
from datetime import datetime, timezone

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from builds_client import BuildsClient, classify_build, ASCENDANCY_MAP
from pob_decoder import decode_pob_code

# Import just the bracket stripper without loading the full why-engine
# (which triggers MetaData.load() and a 30s download attempt)
import re as _re
_NINJA_BRACKET = _re.compile(r"\[([^|\]]*\|)?([^\]]*)\]")
def _strip_ninja_brackets(text):
    return _NINJA_BRACKET.sub(r"\2", text)

# ── Class skill combos (same as class_deep_dive.py) ──────────────────────

CLASS_SKILLS = {
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
        # 0.5.0 Runes of Aldur — probe skills best-effort, refine once meta observed
        ('Martial Artist', 'Tempest Bell'), ('Martial Artist', 'Ice Strike'),
        ('Martial Artist', 'Flicker Strike'),
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
        # 0.5.0 Runes of Aldur — probe skills best-effort, refine once meta observed
        ('Spirit Walker', 'Lightning Arrow'), ('Spirit Walker', 'Barrage'),
    ],
    'Druid': [
        ('Oracle', 'Bone Storm'), ('Oracle', 'Storm Call'), ('Oracle', 'Comet'),
        ('Shaman', 'Tornado'), ('Shaman', 'Storm Call'), ('Shaman', 'Lightning Bolt'),
    ],
}

# Stat keys used in mod extraction and scaling analysis
STAT_KEYS = [
    'gem_levels', 'extra_as', 'spell_dmg', 'phys_dmg', 'atk_speed',
    'cast_speed', 'crit_chance', 'crit_multi', 'life_flat', 'es_flat',
    'es_pct', 'spirit', 'evasion_pct', 'armour_pct', 'jewel_count',
    'support_count', 'unique_count',
]

# Canonical set of ascendancies to iterate (excludes alternate spellings)
CANONICAL_ASCENDANCIES = [
    asc for asc in ASCENDANCY_MAP
    if asc not in ("Witch Hunter", "Abyssal Lich")
]


# ── Character extraction (from class_deep_dive.py) ───────────────────────

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

    # Jewel mods
    jewel_mods = []
    for j in jewels:
        for mod in (getattr(j, 'explicit_mods', None) or []):
            jewel_mods.append(_strip_ninja_brackets(mod))

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
        'jewel_mods': jewel_mods,
        'has_adorned': any('adorned' in n.lower() for n in jewel_names),
        'support_count': len(supports), 'supports': supports[:6],
        'unique_count': len(uniques), 'uniques': uniques,
        'keystones': list(char.keystones),
        'asc_points': getattr(char, 'ascendancy_points', 0),
        **{k: v for k, v in stats.items()},
    }


# ── Scaling analysis ─────────────────────────────────────────────────────

def compute_class_scaling(builds, class_name):
    """Compute scaling weights for a class by comparing top vs bottom 1/3."""
    builds_dps = sorted([b for b in builds if b['main_dps'] > 1000],
                        key=lambda x: -x['main_dps'])
    if len(builds_dps) < 4:
        return None

    q = max(2, len(builds_dps) // 3)
    top, bot = builds_dps[:q], builds_dps[-q:]

    def avg(lst, k):
        vals = [b.get(k, 0) for b in lst]
        return sum(vals) / len(vals) if vals else 0

    # Scaling weights
    weights = {}
    for key in STAT_KEYS:
        t, b = avg(top, key), avg(bot, key)
        ratio = t / b if b > 0 else (99.0 if t > 0 else 1.0)
        if ratio > 1.05:
            weights[key] = round(ratio, 1)

    # Defense distribution
    def_counter = Counter(b['defense_type'] for b in builds)
    total = len(builds) or 1
    defense_dist = {k: round(v / total * 100) for k, v in def_counter.most_common()}
    defense_meta = def_counter.most_common(1)[0][0] if def_counter else 'life'

    # Top skills
    skill_counter = Counter()
    skill_dps = defaultdict(list)
    for b in builds_dps:
        if b['main_skill']:
            skill_counter[b['main_skill']] += 1
            skill_dps[b['main_skill']].append(b['main_dps'])
    top_skills = [
        {"name": sk, "count": c, "avg_dps": int(sum(skill_dps[sk]) / c)}
        for sk, c in skill_counter.most_common(8)
    ]

    # DPS range
    dps_vals = [b['main_dps'] for b in builds_dps]
    dps_range = {
        "min": int(dps_vals[-1]) if dps_vals else 0,
        "max": int(dps_vals[0]) if dps_vals else 0,
        "spread": int(dps_vals[0] / max(dps_vals[-1], 1)) if dps_vals else 0,
    }

    # Primary factor (highest weight)
    primary_factor = max(weights, key=weights.get) if weights else ''

    return {
        "sample_size": len(builds),
        "primary_factor": primary_factor,
        "weights": weights,
        "defense_meta": defense_meta,
        "defense_distribution": defense_dist,
        "top_skills": top_skills,
        "dps_range": dps_range,
    }


def compute_top_support_gems(all_builds):
    """Find support gems that separate top from bottom builds."""
    builds_dps = sorted([b for b in all_builds if b['main_dps'] > 1000],
                        key=lambda x: -x['main_dps'])
    if len(builds_dps) < 4:
        return {}

    q = max(2, len(builds_dps) // 3)
    top, bot = builds_dps[:q], builds_dps[-q:]

    sup_top, sup_bot = Counter(), Counter()
    sup_classes = defaultdict(Counter)
    for b in top:
        for s in b.get('supports', []):
            sup_top[s] += 1
            sup_classes[s][b['base_class']] += 1
    for b in bot:
        for s in b.get('supports', []):
            sup_bot[s] += 1

    result = {}
    for gem in set(list(sup_top.keys()) + list(sup_bot.keys())):
        tc, bc = sup_top.get(gem, 0), sup_bot.get(gem, 0)
        if tc <= bc:
            continue  # Only gems that appear more in top
        ratio = tc / bc if bc > 0 else float('inf')
        if ratio < 1.5 and bc > 0:
            continue
        best = [cls for cls, _ in sup_classes[gem].most_common(4)] if gem in sup_classes else []
        result[gem] = {
            "top_count": tc,
            "bottom_count": bc,
            "ratio": round(ratio, 1) if ratio != float('inf') else "inf",
            "best_classes": best if best else ["all"],
        }

    return dict(sorted(result.items(), key=lambda x: -(x[1]['top_count'])))


def compute_popular_uniques(all_builds):
    """Compute unique item adoption rates globally and per class."""
    result = {}
    # Count per unique per class
    unique_class = defaultdict(lambda: defaultdict(int))
    unique_slot = {}
    unique_global = Counter()
    total_per_class = Counter()

    for b in all_builds:
        base = b['base_class']
        total_per_class[base] += 1
        seen = set()
        for slot, uname in b.get('uniques', []):
            if uname in seen:
                continue
            seen.add(uname)
            unique_global[uname] += 1
            unique_class[uname][base] += 1
            unique_slot[uname] = slot

    total = len(all_builds) or 1
    for uname, count in unique_global.most_common(30):
        if count < 2:
            continue
        class_adoption = {}
        for cls, cnt in unique_class[uname].items():
            cls_total = total_per_class[cls] or 1
            class_adoption[cls] = round(cnt / cls_total * 100)
        result[uname] = {
            "slot": unique_slot.get(uname, "?"),
            "global_adoption_pct": round(count / total * 100, 1),
            "class_adoption": dict(sorted(class_adoption.items(),
                                          key=lambda x: -x[1])),
        }

    return result


def compute_keystone_combos(all_builds):
    """Detect keystone combos from co-occurrence in top builds."""
    builds_dps = sorted([b for b in all_builds if b['main_dps'] > 1000],
                        key=lambda x: -x['main_dps'])
    if len(builds_dps) < 4:
        return []

    q = max(2, len(builds_dps) // 3)
    top, bot = builds_dps[:q], builds_dps[-q:]

    # Find pairs that co-occur in top builds
    pair_top = Counter()
    pair_bot = Counter()
    pair_classes = defaultdict(Counter)

    for b in top:
        ks = sorted(set(b['keystones']))
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                pair = (ks[i], ks[j])
                pair_top[pair] += 1
                pair_classes[pair][b['base_class']] += 1

    for b in bot:
        ks = sorted(set(b['keystones']))
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                pair_bot[(ks[i], ks[j])] += 1

    total_top = len(top) or 1
    total_bot = len(bot) or 1
    combos = []
    for pair, count in pair_top.most_common(20):
        if count < 2:
            continue
        top_pct = round(count / total_top * 100)
        bot_pct = round(pair_bot.get(pair, 0) / total_bot * 100)
        if top_pct <= bot_pct:
            continue  # Only combos that appear more in top
        best = [cls for cls, _ in pair_classes[pair].most_common(4)]
        total_all = len(all_builds) or 1
        # Count in all builds
        all_count = sum(1 for b in all_builds
                        if pair[0] in b['keystones'] and pair[1] in b['keystones'])
        combos.append({
            "keystones": list(pair),
            "adoption_pct": round(all_count / total_all * 100, 1),
            "best_classes": best,
            "top_pct": top_pct,
            "bottom_pct": bot_pct,
        })

    combos.sort(key=lambda x: -x['top_pct'])
    return combos[:15]


def compute_mod_weights(all_builds):
    """Compute global and per-class mod weights."""
    def _weights_for(builds):
        builds_dps = sorted([b for b in builds if b['main_dps'] > 1000],
                            key=lambda x: -x['main_dps'])
        if len(builds_dps) < 4:
            return {}
        q = max(2, len(builds_dps) // 3)
        top, bot = builds_dps[:q], builds_dps[-q:]

        def avg(lst, k):
            vals = [b.get(k, 0) for b in lst]
            return sum(vals) / len(vals) if vals else 0

        weights = {}
        mod_keys = ['gem_levels', 'extra_as', 'spell_dmg', 'phys_dmg',
                     'atk_speed', 'cast_speed', 'crit_chance', 'crit_multi']
        for key in mod_keys:
            t, b = avg(top, key), avg(bot, key)
            ratio = t / b if b > 0 else (99.0 if t > 0 else 1.0)
            if ratio > 1.05:
                weights[key] = round(ratio, 1)
        return dict(sorted(weights.items(), key=lambda x: -x[1]))

    global_weights = _weights_for(all_builds)

    per_class = {}
    class_builds = defaultdict(list)
    for b in all_builds:
        class_builds[b['base_class']].append(b)
    for cls, builds in class_builds.items():
        w = _weights_for(builds)
        if w:
            # Only include weights that differ meaningfully from global
            filtered = {k: v for k, v in w.items()
                        if abs(v - global_weights.get(k, 1.0)) > 0.5 or v > 3.0}
            if filtered:
                per_class[cls] = filtered

    return {"global": global_weights, "per_class": per_class}


def compute_dps_ceilings(all_builds):
    """Compute DPS ceilings per ascendancy."""
    asc_dps = defaultdict(list)
    for b in all_builds:
        if b['main_dps'] > 1000:
            asc_dps[b['ascendancy']].append(b['main_dps'])

    result = {}
    for asc, dps_vals in asc_dps.items():
        if len(dps_vals) < 2:
            continue
        dps_vals.sort(reverse=True)
        result[asc] = {
            "max": int(dps_vals[0]),
            "p90": int(dps_vals[max(0, len(dps_vals) // 10)]),
            "median": int(statistics.median(dps_vals)),
        }

    return result


# ── Data collection ──────────────────────────────────────────────────────

def collect_builds(client, dry_run=False):
    """Collect builds for all classes. Returns (all_builds, class_builds_map)."""
    all_builds = []
    class_builds_map = defaultdict(list)

    for base_class, combos in CLASS_SKILLS.items():
        print(f"\n[{base_class}] Collecting builds...")
        class_count = 0

        for asc, skill in combos:
            if dry_run:
                print(f"  [dry-run] Would fetch: {asc} / {skill}")
                continue

            try:
                data = client._fetch_search(asc, skill)
            except Exception as e:
                print(f"  [WARN] Search failed for {asc}/{skill}: {e}")
                time.sleep(0.5)
                continue

            if not data or not data.get('featuredCharacters'):
                time.sleep(0.5)
                continue

            chars = data['featuredCharacters']
            total = len(chars)

            # Sample 5-7 characters: top, some from middle, one from bottom
            indices = sorted(set(min(i, total - 1) for i in
                [0, 1, 2, total // 4, total // 2, total * 3 // 4, total - 1]))

            seen = set()
            for idx in indices:
                ch = chars[idx]
                acct = ch.get('account', '')
                name = ch.get('name', '')
                if not acct or not name or name in seen:
                    continue
                seen.add(name)

                try:
                    char = client.lookup_character(acct, name)
                except Exception as e:
                    print(f"  [WARN] Lookup failed for {acct}/{name}: {e}")
                    time.sleep(0.5)
                    continue

                if not char:
                    time.sleep(0.5)
                    continue

                try:
                    bd = extract_full(client, char)
                except Exception as e:
                    print(f"  [WARN] Extract failed for {name}: {e}")
                    time.sleep(0.5)
                    continue

                all_builds.append(bd)
                class_builds_map[base_class].append(bd)
                class_count += 1
                time.sleep(0.5)

            time.sleep(0.5)

        print(f"  [{base_class}] {class_count} builds collected")

    return all_builds, class_builds_map


# ── Shard generation ─────────────────────────────────────────────────────

def generate_shard(client, all_builds, class_builds_map):
    """Generate the meta shard dict from collected builds."""
    # Class scaling
    class_scaling = {}
    for base_class, builds in class_builds_map.items():
        scaling = compute_class_scaling(builds, base_class)
        if scaling:
            class_scaling[base_class] = scaling

    shard = {
        "version": 2,
        "league": client._snapshot_name or "unknown",
        "league_url": client._snapshot_league_url or "",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "snapshot_version": client._snapshot_version or "",
        "total_builds_sampled": len(all_builds),
        "total_characters_on_ladder": 0,  # filled if available

        "class_scaling": class_scaling,
        "top_support_gems": compute_top_support_gems(all_builds),
        "popular_uniques": compute_popular_uniques(all_builds),
        "keystone_combos": compute_keystone_combos(all_builds),
        "mod_weights": compute_mod_weights(all_builds),
        "dps_ceilings": compute_dps_ceilings(all_builds),
    }

    return shard


def write_shard(shard, output_path):
    """Write compressed shard to disk."""
    json_bytes = json.dumps(shard, indent=2, ensure_ascii=False).encode('utf-8')
    with gzip.open(output_path, 'wb') as f:
        f.write(json_bytes)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"\nShard written: {output_path} ({size_kb:.1f} KB compressed)")
    print(f"  Builds sampled: {shard['total_builds_sampled']}")
    print(f"  Classes with scaling data: {len(shard['class_scaling'])}")
    print(f"  Top support gems: {len(shard['top_support_gems'])}")
    print(f"  Popular uniques: {len(shard['popular_uniques'])}")
    print(f"  Keystone combos: {len(shard['keystone_combos'])}")
    print(f"  DPS ceilings for: {len(shard['dps_ceilings'])} ascendancies")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LAMA Meta Harvester — collect meta data and generate meta_shard.json.gz")
    parser.add_argument('--league', type=str, default=None,
                        help='Override league name (default: auto-detect from poe.ninja)')
    parser.add_argument('--output', type=str, default='meta_shard.json.gz',
                        help='Output file path (default: meta_shard.json.gz)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be fetched without making API calls')
    args = parser.parse_args()

    print("LAMA Meta Harvester")
    print("=" * 50)

    client = BuildsClient()

    if not args.dry_run:
        print("Fetching snapshot info from poe.ninja...")
        if not client._fetch_snapshot_info():
            print("[ERROR] Failed to fetch snapshot info. Aborting.")
            sys.exit(1)

        league = args.league or client._snapshot_name or "unknown"
        print(f"  League: {league}")
        print(f"  Snapshot: {client._snapshot_version}")
        print(f"  League URL: {client._snapshot_league_url}")

        if args.league:
            client._snapshot_name = args.league
    else:
        print("[dry-run] Skipping snapshot fetch")
        print(f"  Ascendancies to process: {len(CANONICAL_ASCENDANCIES)}")
        for asc in CANONICAL_ASCENDANCIES:
            base = ASCENDANCY_MAP[asc]
            print(f"    {asc} ({base})")

    print(f"\nClasses to scan: {len(CLASS_SKILLS)}")
    print(f"Skill combos: {sum(len(v) for v in CLASS_SKILLS.values())}")
    print()

    # Collect
    all_builds, class_builds_map = collect_builds(client, dry_run=args.dry_run)

    if args.dry_run:
        print("\n[dry-run] No data collected. Exiting.")
        return

    if not all_builds:
        print("\n[ERROR] No builds collected. Aborting.")
        sys.exit(1)

    print(f"\nTotal builds collected: {len(all_builds)}")
    for cls, builds in sorted(class_builds_map.items()):
        print(f"  {cls}: {len(builds)}")

    # Analyze
    print("\nRunning scaling analysis...")
    shard = generate_shard(client, all_builds, class_builds_map)

    # Write
    write_shard(shard, args.output)
    print("\nDone.")


if __name__ == "__main__":
    main()
