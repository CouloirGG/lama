"""Run why-engine audit across all 21 ascendancies."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from builds_client import BuildsClient, classify_build, ASCENDANCY_MAP
from why_engine import WhyEngine
from pob_decoder import decode_pob_code

client = BuildsClient()
client._snapshot_version = None
client._snapshot_name = None
client._cache.clear()
client._fetch_snapshot_info()
print(f"Snapshot: {client._snapshot_version}")

engine = WhyEngine(client)
all_asc = sorted(set(ASCENDANCY_MAP.keys()) - {"Witch Hunter"})
bugs = []
tested = 0

for asc in all_asc:
    data = client._fetch_search(asc, "Comet")
    if not data or not data.get("featuredCharacters"):
        bugs.append(f"NO_DATA: {asc}")
        continue

    ch = data["featuredCharacters"][0]
    char = client.lookup_character(ch["account"], ch["name"])
    if not char:
        continue

    tested += 1
    arch = classify_build(char)
    try:
        exps = engine.explain_character(char, arch)
        dps = exps.scorecard.dps_label if exps.scorecard else "?"
        syn = sum(len(c.missing) for c in exps.synergy_map)
        # Avoid unicode issues in print
        name_safe = char.name.encode("ascii", "replace").decode()
        print(f"{tested:2d}/{len(all_asc)} {asc:25s} {name_safe:25s} DPS={dps:>12s} missing={syn} OK")
    except Exception as e:
        bugs.append(f"CRASH: {asc}: {str(e)[:80]}")

    time.sleep(1.5)

print(f"\nTested: {tested}/{len(all_asc)} | Bugs: {len(bugs)}")
if bugs:
    for b in bugs:
        print(f"  {b}")
else:
    print("ALL CLEAN!")
