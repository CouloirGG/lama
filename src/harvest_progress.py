#!/usr/bin/env python3
"""Quick progress view for running calibration harvester."""
import json, sys, time
from pathlib import Path

CACHE = Path.home() / ".poe2-price-overlay" / "cache"
QUERIES_PER_PASS = 136

def show():
    today = time.strftime("%Y-%m-%d")
    print(f"=== Harvester Progress ({today}) ===\n")

    total_samples = 0
    total_done = 0
    total_planned = 0

    for p in range(1, 16):
        state_file = CACHE / f"harvester_state_p{p}.json"
        shard_file = CACHE / f"calibration_shard_fate_of_the_vaal_{today}_p{p}.jsonl"

        if not state_file.exists():
            continue

        with open(state_file) as f:
            state = json.load(f)

        done = len(state.get("completed_queries", []))
        samples = state.get("total_samples", 0)
        dead = len(state.get("dead_combos", []))

        # Check shard file line count for actual records
        shard_lines = 0
        if shard_file.exists():
            with open(shard_file) as f:
                shard_lines = sum(1 for _ in f)

        pct = done / QUERIES_PER_PASS * 100 if QUERIES_PER_PASS else 0
        bar_width = 30
        filled = int(bar_width * done / QUERIES_PER_PASS)
        bar = "#" * filled + "-" * (bar_width - filled)

        status = "DONE" if done >= QUERIES_PER_PASS else "RUNNING" if shard_lines > 0 or done > 0 else "pending"
        print(f"  Pass {p:2d}  [{bar}] {done:3d}/{QUERIES_PER_PASS} ({pct:5.1f}%)  "
              f"{shard_lines:5d} samples  {dead:3d} dead  {status}")

        total_samples += shard_lines
        total_done += done
        total_planned += QUERIES_PER_PASS

    print(f"\n  Total samples across all shards: {total_samples}")

    # Estimate remaining time based on rate (~10s/query when throttled)
    remaining = total_planned - total_done
    if remaining > 0:
        est_min = remaining * 10 / 60
        print(f"  Remaining queries: {remaining} (~{est_min:.0f} min at 10s/req)")

if __name__ == "__main__":
    if "--watch" in sys.argv:
        try:
            while True:
                print("\033[2J\033[H", end="")  # clear screen
                show()
                print("\n  (Ctrl+C to stop, refreshing every 30s)")
                time.sleep(30)
        except KeyboardInterrupt:
            print()
    else:
        show()
