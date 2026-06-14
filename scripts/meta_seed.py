"""
meta_seed.py — Per-ascendancy meta seeder for LAMA.

For every ascendancy, pulls the top builds from poe.ninja (skill-agnostic, so the
meta is discovered empirically rather than assumed from a curated probe list),
runs extract_full on each, and aggregates the current meta per ascendancy and per
base class: top skills, uniques, keystones, defense/damage split, DPS range, EHP.

Outputs:
    scripts/meta_seed.json                       — full structured data (+ raw builds)
    docs/meta-report-<league-slug>.md            — readable per-ascendancy tables

Usage:
    python scripts/meta_seed.py                  # all ascendancies, 12 builds each
    python scripts/meta_seed.py --per 8 --only Monk,Huntress
"""
import sys, os, json, time, argparse, statistics
from collections import Counter, defaultdict

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from builds_client import BuildsClient, ASCENDANCY_MAP
from meta_harvester import extract_full, CANONICAL_ASCENDANCIES

OUT_JSON = os.path.join(os.path.dirname(__file__), "meta_seed.json")


def _top(counter, n):
    return [{"name": k, "count": v} for k, v in counter.most_common(n)]


def aggregate(asc, builds):
    skills, uniques, keystones, supports = Counter(), Counter(), Counter(), Counter()
    defense, damage = Counter(), Counter()
    dps_vals, ehp_vals = [], []
    crit = coc = 0
    for b in builds:
        if b.get('main_skill'):
            skills[b['main_skill']] += 1
        for _slot, name in b.get('uniques', []):
            if name:
                uniques[name] += 1
        for ks in b.get('keystones', []):
            keystones[ks] += 1
        for g in b.get('supports', []):
            supports[g] += 1
        if b.get('defense_type'):
            defense[b['defense_type']] += 1
        if b.get('damage_type'):
            damage[b['damage_type']] += 1
        if b.get('main_dps', 0) > 1000:
            dps_vals.append(b['main_dps'])
        if b.get('ehp', 0) > 0:
            ehp_vals.append(b['ehp'])
        crit += 1 if b.get('is_crit') else 0
        coc += 1 if b.get('is_coc') else 0
    n = len(builds)
    dps_vals.sort()
    dps = {}
    if dps_vals:
        dps = {
            "median": int(statistics.median(dps_vals)),
            "p90": int(dps_vals[min(len(dps_vals) - 1, int(len(dps_vals) * 0.9))]),
            "max": int(dps_vals[-1]),
        }
    return {
        "ascendancy": asc,
        "base_class": ASCENDANCY_MAP.get(asc, "?"),
        "sample_size": n,
        "top_skills": _top(skills, 6),
        "top_uniques": _top(uniques, 10),
        "top_keystones": _top(keystones, 10),
        "common_supports": _top(supports, 8),
        "defense_meta": dict(defense.most_common()),
        "damage_meta": dict(damage.most_common()),
        "crit_pct": round(crit / n, 2) if n else 0,
        "coc_pct": round(coc / n, 2) if n else 0,
        "dps": dps,
        "ehp_median": int(statistics.median(ehp_vals)) if ehp_vals else 0,
        "builds": [
            {"name": b.get('name'), "level": b.get('level'),
             "main_skill": b.get('main_skill'), "main_dps": b.get('main_dps'),
             "defense_type": b.get('defense_type'),
             "uniques": [n for _s, n in b.get('uniques', [])],
             "keystones": b.get('keystones', [])}
            for b in builds
        ],
    }


def collect(client, asc, per):
    """Fetch top `per` builds for an ascendancy (skill-agnostic) and extract each."""
    try:
        data = client._fetch_search(asc, "")
    except Exception as e:
        print(f"  [{asc}] search failed: {e}", flush=True)
        return []
    chars = (data or {}).get("featuredCharacters", []) or []
    out, seen = [], set()
    for ch in chars:
        if len(out) >= per:
            break
        acct, name = ch.get("account", ""), ch.get("name", "")
        if not acct or not name or name in seen:
            continue
        seen.add(name)
        try:
            cd = client.lookup_character(acct, name)
            if not cd:
                time.sleep(0.4); continue
            out.append(extract_full(client, cd))
        except Exception as e:
            print(f"  [{asc}] {name}: {e}", flush=True)
        time.sleep(0.4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=12, help="builds per ascendancy")
    ap.add_argument("--only", type=str, default="", help="comma list of ascendancies/classes to limit to")
    args = ap.parse_args()

    client = BuildsClient()
    if not client._fetch_snapshot_info():
        print("Could not resolve current league from poe.ninja. Aborting.")
        sys.exit(1)
    league = client._snapshot_name
    print(f"League: {league} | per-ascendancy: {args.per}", flush=True)

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    ascendancies = [a for a in CANONICAL_ASCENDANCIES
                    if not only or a in only or ASCENDANCY_MAP.get(a) in only]

    per_asc, all_builds = {}, []
    for i, asc in enumerate(ascendancies, 1):
        print(f"[{i}/{len(ascendancies)}] {asc} ...", flush=True)
        builds = collect(client, asc, args.per)
        print(f"    {len(builds)} builds", flush=True)
        if builds:
            per_asc[asc] = aggregate(asc, builds)
            all_builds.extend(builds)
        # checkpoint after each ascendancy
        json.dump({"league": league, "per_ascendancy": per_asc},
                  open(OUT_JSON, "w", encoding="utf-8"), indent=2)

    # roll up per base class
    per_class = {}
    by_class = defaultdict(list)
    for asc, agg in per_asc.items():
        for b in agg["builds"]:
            by_class[agg["base_class"]].append(b)
    for cls, builds in by_class.items():
        # reuse aggregate by faking ascendancy=class on raw rows we still have features for
        # (re-aggregate from the trimmed rows we kept)
        skills, uniques, keystones = Counter(), Counter(), Counter()
        defense = Counter()
        dps_vals = []
        for b in builds:
            if b.get("main_skill"): skills[b["main_skill"]] += 1
            for n in b.get("uniques", []): uniques[n] += 1
            for ks in b.get("keystones", []): keystones[ks] += 1
            if b.get("defense_type"): defense[b["defense_type"]] += 1
            if b.get("main_dps", 0) > 1000: dps_vals.append(b["main_dps"])
        per_class[cls] = {
            "base_class": cls, "sample_size": len(builds),
            "top_skills": _top(skills, 8), "top_uniques": _top(uniques, 10),
            "top_keystones": _top(keystones, 10),
            "defense_meta": dict(defense.most_common()),
            "dps_median": int(statistics.median(dps_vals)) if dps_vals else 0,
        }

    result = {"league": league, "total_builds": len(all_builds),
              "per_ascendancy": per_asc, "per_class": per_class}
    json.dump(result, open(OUT_JSON, "w", encoding="utf-8"), indent=2)
    print(f"\nWrote {OUT_JSON}: {len(all_builds)} builds across {len(per_asc)} ascendancies", flush=True)
    write_report(result, league)


def write_report(result, league):
    slug = league.replace(" ", "-").lower()
    path = os.path.join(os.path.dirname(__file__), "..", "docs", f"meta-report-{slug}.md")
    L = [f"# PoE2 Meta Report — {league}", "",
         f"Seeded from poe.ninja top builds per ascendancy. Total builds: {result['total_builds']}.", ""]
    L.append("## Per base class\n")
    L.append("| Class | n | Top skills | Top keystones | Defense | DPS median |")
    L.append("|---|---|---|---|---|---|")
    for cls, a in sorted(result["per_class"].items()):
        sk = ", ".join(f"{x['name']}({x['count']})" for x in a["top_skills"][:4])
        ks = ", ".join(f"{x['name']}" for x in a["top_keystones"][:3])
        df = ", ".join(f"{k}:{v}" for k, v in a["defense_meta"].items())
        L.append(f"| {cls} | {a['sample_size']} | {sk} | {ks} | {df} | {a['dps_median']:,} |")
    L.append("\n## Per ascendancy\n")
    for asc, a in sorted(result["per_ascendancy"].items(), key=lambda kv: kv[1]["base_class"]):
        L.append(f"### {asc} ({a['base_class']}) — n={a['sample_size']}")
        sk = ", ".join(f"{x['name']} ({x['count']})" for x in a["top_skills"])
        un = ", ".join(f"{x['name']} ({x['count']})" for x in a["top_uniques"][:6])
        ks = ", ".join(f"{x['name']} ({x['count']})" for x in a["top_keystones"][:6])
        dps = a.get("dps", {})
        L.append(f"- **Top skills:** {sk or '—'}")
        L.append(f"- **Top uniques:** {un or '—'}")
        L.append(f"- **Top keystones:** {ks or '—'}")
        L.append(f"- **Defense:** {a['defense_meta']} | **Crit:** {int(a['crit_pct']*100)}% | "
                 f"**DPS** med {dps.get('median',0):,} / max {dps.get('max',0):,} | EHP med {a['ehp_median']:,}")
        L.append("")
    open(path, "w", encoding="utf-8").write("\n".join(L))
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
