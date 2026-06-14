"""
fixture_eval.py — Ground-truth accuracy evaluation using curated fixtures.

Unlike accuracy_eval.py which compares LAMA against itself (shared code),
this script compares LAMA against independently-sourced ground truth from
poe.ninja's own data and manually annotated expectations.

Fixtures are in tests/fixtures/accuracy_fixtures.json.

Usage:
    python scripts/fixture_eval.py
"""

import sys, os, json, time
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from builds_client import BuildsClient, classify_build
from why_engine import WhyEngine
from pob_decoder import decode_pob_code


def main():
    print("=" * 60)
    print("FIXTURE-BASED ACCURACY EVALUATION")
    print("LAMA vs independent poe.ninja ground truth")
    print("=" * 60)
    print()

    # Load fixtures
    fixture_path = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "accuracy_fixtures.json")
    with open(fixture_path, encoding="utf-8") as f:
        fixtures = json.load(f)

    print(f"Loaded {len(fixtures)} fixtures")
    print()

    client = BuildsClient()
    engine = WhyEngine(client)

    bugs = []
    tested = 0

    for fix in fixtures:
        acct = fix["account"]
        name = fix["name"]

        char = client.lookup_character(acct, name)
        if not char:
            print(f"  SKIP: {name} — not found")
            continue

        tested += 1
        arch = classify_build(char)

        try:
            exps = engine.explain_character(char, arch)
        except Exception as e:
            bugs.append({
                "build": name, "category": "crash", "severity": "critical",
                "description": f"LAMA crashed: {str(e)[:80]}",
            })
            continue

        name_safe = name.encode("ascii", "replace").decode()
        build_bugs = []

        # ── Check 1: Defense type vs poe.ninja stats ──
        expected_def = fix["expected_defense_type"]
        if arch.defense_type != expected_def:
            # Check if it's a reasonable disagreement (hybrid vs es is close)
            if not ({arch.defense_type, expected_def} == {"hybrid", "es"}):
                build_bugs.append({
                    "category": "classification", "severity": "major",
                    "description": f"Defense type: LAMA={arch.defense_type}, ground_truth={expected_def} (life={fix['pn_life']}, ES={fix['pn_es']})",
                })

        # ── Check 2: Ascendancy false positives ──
        if "ascendancy_passives" in fix.get("should_not_recommend", []):
            for action in (exps.actions or []):
                title = action.title
                if "Ascendancy:" in title or "Check Ascendancy:" in title:
                    build_bugs.append({
                        "category": "false_positive", "severity": "critical",
                        "description": f"Recommends ascendancy passive '{title}' but player has {fix['pn_ascendancy_points']} asc points",
                    })

        # ── Check 3: Resist false positives ──
        if "cap_resists" in fix.get("should_not_recommend", []):
            for action in (exps.actions or []):
                if "Cap" in action.title and "Resist" in action.title:
                    build_bugs.append({
                        "category": "false_positive", "severity": "major",
                        "description": f"Recommends '{action.title}' but all resists are capped per poe.ninja",
                    })

        # ── Check 4: ES build told to add life ──
        if "add_life_critical" in fix.get("should_not_recommend", []):
            for action in (exps.actions or []):
                if "Add Life" in action.title and action.severity == "critical":
                    build_bugs.append({
                        "category": "false_positive", "severity": "major",
                        "description": f"ES build gets critical 'Add Life' (life={fix['pn_life']}, ES={fix['pn_es']})",
                    })

        # ── Check 5: Resist accuracy vs poe.ninja ──
        # LAMA should flag uncapped resists that poe.ninja shows as uncapped
        for resist_name, pn_key in [
            ("Fire", "pn_fire_res"), ("Cold", "pn_cold_res"),
            ("Lightning", "pn_lightning_res"), ("Chaos", "pn_chaos_res"),
        ]:
            pn_val = fix.get(pn_key, 75)
            if pn_val < 75:
                # poe.ninja says uncapped — does LAMA flag it?
                has_resist_action = any(
                    resist_name.lower() in a.title.lower() or "resist" in a.title.lower()
                    for a in (exps.actions or [])
                )
                if not has_resist_action:
                    build_bugs.append({
                        "category": "missed_issue", "severity": "major",
                        "description": f"{resist_name} resist is {pn_val}% (uncapped) per poe.ninja but LAMA doesn't flag it",
                    })

        # ── Check 6: Jewel count accuracy ──
        lama_jewel_count = len(getattr(char, "jewels", []))
        pn_jewel_count = fix.get("pn_jewel_count", 0)
        if abs(lama_jewel_count - pn_jewel_count) > 1:
            build_bugs.append({
                "category": "data", "severity": "minor",
                "description": f"Jewel count mismatch: LAMA={lama_jewel_count}, poe.ninja={pn_jewel_count}",
            })

        # ── Check 7: Known unique jewels detected ──
        pn_jewel_names = fix.get("pn_jewel_names", [])
        important_jewels = {"The Adorned", "Megalomaniac", "From Nothing",
                           "Heart of the Well", "Heroic Tragedy", "Undying Hate"}
        for jn in pn_jewel_names:
            if jn in important_jewels:
                # Check if LAMA's synergy map mentions this jewel
                dps_cat = next((c for c in exps.synergy_map if c.category == "dps"), None)
                if dps_cat:
                    jewel_mentioned = any(jn.lower() in c.name.lower() for c in dps_cat.contributors)
                    if not jewel_mentioned:
                        build_bugs.append({
                            "category": "missed_issue", "severity": "major",
                            "description": f"Important jewel '{jn}' exists but not detected in synergy map",
                        })

        # ── Check 8: Scorecard EHP matches poe.ninja ──
        if exps.scorecard and fix.get("pn_ehp", 0) > 0:
            lama_ehp = exps.scorecard.ehp
            pn_ehp = fix["pn_ehp"]
            if pn_ehp > 0 and abs(lama_ehp - pn_ehp) / max(pn_ehp, 1) > 0.1:
                build_bugs.append({
                    "category": "data", "severity": "minor",
                    "description": f"EHP mismatch: LAMA={lama_ehp:,}, poe.ninja={pn_ehp:,} ({abs(lama_ehp-pn_ehp)/pn_ehp*100:.0f}% diff)",
                })

        # Log results
        bugs.extend([{**b, "build": name_safe, "tier": fix["tier"]} for b in build_bugs])
        status = f"{len(build_bugs)} issues" if build_bugs else "PASS"
        print(f"  {tested:2d}. [{fix['tier']:3s}] {fix['ascendancy']:25s} {name_safe:25s} {status}")

        time.sleep(1)

    # ── Report ──
    print()
    print("=" * 60)
    print(f"RESULTS: {tested} builds, {len(bugs)} discrepancies")
    print("=" * 60)
    print()

    if bugs:
        from collections import defaultdict
        by_cat = defaultdict(list)
        for b in bugs:
            by_cat[b["category"]].append(b)

        for cat in sorted(by_cat.keys()):
            items = by_cat[cat]
            # Dedup
            seen = set()
            unique = []
            for b in items:
                key = b["description"][:60]
                if key not in seen:
                    seen.add(key)
                    unique.append(b)

            print(f"{cat.upper()} ({len(items)} total, {len(unique)} unique):")
            for b in unique[:5]:
                print(f"  [{b['severity']:>8}] {b['description']}")
                print(f"    Build: {b['build']} ({b['tier']})")
            if len(unique) > 5:
                print(f"  ... +{len(unique)-5} more")
            print()
    else:
        print("ALL CHECKS PASS!")

    # Save report
    report_path = os.path.join(os.path.dirname(__file__), "fixture_eval_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "tested": tested,
            "total_bugs": len(bugs),
            "bugs": bugs,
        }, f, indent=2)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
