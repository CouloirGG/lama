"""
LAMA - Harvest Scheduler

Runs the calibration harvester continuously with periodic accuracy checks.
After each harvest cycle, runs the full_pipeline accuracy experiment and
logs results to a tracking file so you can see how accuracy changes as
data volume grows.

The harvester uses a date-based seed per pass, so running the same pass
number on the same day produces the same query plan (skipping completed
queries). Each new cycle resets state and starts fresh passes, getting new
data from different API result offsets.

Usage:
    python harvest_scheduler.py                    # default: 15 passes/cycle, check after each
    python harvest_scheduler.py --passes 5         # 5 passes per cycle
    python harvest_scheduler.py --cooldown 300     # 5 min cooldown between cycles
    python harvest_scheduler.py --no-accuracy      # skip accuracy checks (harvest only)
    python harvest_scheduler.py --once             # single cycle then exit
    python harvest_scheduler.py --accuracy-only    # just run accuracy check
    python harvest_scheduler.py --history          # show accuracy tracking history
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Paths
SRC_DIR = Path(__file__).resolve().parent
CACHE_DIR = Path(os.path.expanduser("~")) / ".poe2-price-overlay" / "cache"
ACCURACY_LOG = CACHE_DIR / "accuracy_tracking.jsonl"
HARVESTER_SCRIPT = SRC_DIR / "calibration_harvester.py"
ACCURACY_SCRIPT = SRC_DIR / "accuracy_lab.py"


def count_records() -> int:
    """Count total calibration records across all shard files."""
    total = 0
    for f in CACHE_DIR.glob("calibration_shard_*.jsonl"):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                total += sum(1 for _ in fh)
        except OSError:
            pass
    return total


def find_next_pass() -> int:
    """Find the next incomplete pass number by checking state files."""
    for p in range(1, 100):
        state_file = CACHE_DIR / f"harvester_state_p{p}.json"
        if not state_file.exists():
            return p
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                st = json.load(f)
            completed = len(st.get("completed_queries", []))
            # 136 = 17 categories * 8 brackets (primary), may vary
            # If fewer than expected are done, this pass has work left
            if completed < 100:
                return p
        except (json.JSONDecodeError, OSError):
            return p
    return 1


def check_rate_limit() -> int:
    """Check if trade API has an active rate limit penalty.

    Makes a lightweight request to the trade API and reads the
    X-Rate-Limit-Ip-State header. Returns penalty seconds remaining,
    or 0 if no penalty.
    """
    import requests
    try:
        resp = requests.get(
            "https://www.pathofexile.com/api/trade2/data/leagues",
            timeout=10,
            headers={"User-Agent": "LAMA-HarvestScheduler/1.0"},
        )
        state_raw = resp.headers.get("X-Rate-Limit-Ip-State", "")
        if state_raw:
            for part in state_raw.split(","):
                try:
                    _hits, _window, penalty_rem = part.strip().split(":")
                    if int(penalty_rem) > 0:
                        return int(penalty_rem)
                except (ValueError, AttributeError):
                    pass
    except Exception:
        pass
    return 0


def run_harvest(passes: int, start_pass: int = 1) -> bool:
    """Run the calibration harvester. Returns True on success."""
    end_pass = start_pass + passes - 1
    cmd = [
        sys.executable, str(HARVESTER_SCRIPT),
        "--passes", str(end_pass),
        "--start-pass", str(start_pass),
        "--resume",
    ]

    print(f"\n{'='*60}")
    print(f"  HARVESTER: passes {start_pass}-{end_pass}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd, cwd=str(SRC_DIR),
            timeout=14400,  # 4 hour timeout per cycle
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("  WARNING: Harvest cycle timed out (4h). Continuing...")
        return False
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"  ERROR: Harvest failed: {e}")
        return False


def run_accuracy_check() -> dict | None:
    """Run the full_pipeline accuracy experiment. Returns metrics dict or None."""
    print(f"\n{'='*60}")
    print(f"  ACCURACY CHECK")
    print(f"{'='*60}")

    cmd = [
        sys.executable, str(ACCURACY_SCRIPT),
        "--experiment", "full_pipeline",
    ]

    try:
        result = subprocess.run(
            cmd, cwd=str(SRC_DIR),
            capture_output=True, text=True,
            timeout=600,  # 10 min timeout
        )
        output = result.stdout
        print(output)
        if result.stderr:
            for line in result.stderr.strip().split("\n"):
                if line.strip():
                    print(f"  [stderr] {line}")

        # Parse key metrics from output
        metrics = {}
        for line in output.split("\n"):
            stripped = line.strip()
            m = re.match(r"Within 2x:\s+\d+/\d+\s+\((\d+\.\d+)%\)", stripped)
            if m:
                metrics["pct_2x"] = float(m.group(1))
                continue
            m = re.match(r"Within 3x:\s+\d+/\d+\s+\((\d+\.\d+)%\)", stripped)
            if m:
                metrics["pct_3x"] = float(m.group(1))
                continue
            m = re.match(r"Median error:\s+(\d+\.\d+)x", stripped)
            if m:
                metrics["median_error"] = float(m.group(1))
                continue
            m = re.match(r"Total prepared records:\s+(\d+)", stripped)
            if m:
                metrics["total_records"] = int(m.group(1))
                continue
            m = re.match(r"Train:\s+(\d+),\s+Test:\s+(\d+)", stripped)
            if m:
                metrics["train"] = int(m.group(1))
                metrics["test"] = int(m.group(2))
                continue

        return metrics if metrics else None

    except subprocess.TimeoutExpired:
        print("  WARNING: Accuracy check timed out (10min)")
        return None
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"  ERROR: Accuracy check failed: {e}")
        return None


def log_accuracy(metrics: dict, cycle: int):
    """Append accuracy results to the tracking log."""
    entry = {
        "ts": int(time.time()),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "cycle": cycle,
        "shard_records": count_records(),
        **metrics,
    }
    ACCURACY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ACCURACY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"\n  Logged to {ACCURACY_LOG}")


def print_tracking_history():
    """Print the accuracy tracking history so far."""
    if not ACCURACY_LOG.exists():
        print("  No accuracy history yet.")
        return

    entries = []
    with open(ACCURACY_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not entries:
        print("  No accuracy history yet.")
        return

    print(f"\n  Accuracy Tracking History ({len(entries)} entries):")
    print(f"  {'Time':>16s}  {'Cycle':>5s}  {'Records':>8s}  "
          f"{'Within 2x':>9s}  {'Within 3x':>9s}  {'Median':>7s}")
    print(f"  {'-'*16}  {'-'*5}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*7}")
    for e in entries:
        t = e.get("time", "?")
        c = e.get("cycle", "?")
        r = e.get("shard_records", "?")
        p2 = e.get("pct_2x", 0)
        p3 = e.get("pct_3x", 0)
        m = e.get("median_error", 0)
        print(f"  {t:>16s}  {c:>5}  {r:>8}  "
              f"{p2:>8.1f}%  {p3:>8.1f}%  {m:>6.2f}x")

    if len(entries) >= 2:
        first, last = entries[0], entries[-1]
        dp = last.get("pct_2x", 0) - first.get("pct_2x", 0)
        dr = last.get("shard_records", 0) - first.get("shard_records", 0)
        sign = "+" if dp >= 0 else ""
        print(f"\n  Delta since start: {sign}{dp:.1f}% accuracy, +{dr} records")


def main():
    parser = argparse.ArgumentParser(
        description="LAMA Harvest Scheduler - continuous harvesting with accuracy tracking")
    parser.add_argument("--passes", type=int, default=15,
                        help="Passes per harvest cycle (default: 15)")
    parser.add_argument("--cooldown", type=int, default=60,
                        help="Seconds between cycles (default: 60)")
    parser.add_argument("--no-accuracy", action="store_true",
                        help="Skip accuracy checks (harvest only)")
    parser.add_argument("--once", action="store_true",
                        help="Run one cycle then exit")
    parser.add_argument("--accuracy-only", action="store_true",
                        help="Run accuracy check only (no harvesting)")
    parser.add_argument("--history", action="store_true",
                        help="Print accuracy tracking history and exit")
    parser.add_argument("--max-cycles", type=int, default=0,
                        help="Stop after N cycles (0 = unlimited)")

    args = parser.parse_args()

    if args.history:
        print_tracking_history()
        return

    if args.accuracy_only:
        print(f"Current shard records: {count_records()}")
        metrics = run_accuracy_check()
        if metrics:
            log_accuracy(metrics, cycle=0)
        print_tracking_history()
        return

    # Find where to start
    start_pass = find_next_pass()

    print(f"{'='*60}")
    print(f"  LAMA Harvest Scheduler")
    print(f"{'='*60}")
    print(f"  Passes per cycle: {args.passes}")
    print(f"  Starting from pass: {start_pass}")
    print(f"  Cooldown: {args.cooldown}s between cycles")
    print(f"  Accuracy checks: {'OFF' if args.no_accuracy else 'after each cycle'}")
    print(f"  Mode: {'single cycle' if args.once else 'continuous'}")
    if args.max_cycles:
        print(f"  Max cycles: {args.max_cycles}")
    print(f"  Current shard records: {count_records()}")
    print(f"\n  Press Ctrl+C to stop gracefully.\n")

    print_tracking_history()

    cycle = 0
    next_start = start_pass
    try:
        while True:
            cycle += 1
            cycle_start = time.time()

            if args.max_cycles and cycle > args.max_cycles:
                print(f"\nReached max cycles ({args.max_cycles}). Stopping.")
                break

            print(f"\n{'#'*60}")
            print(f"  CYCLE {cycle} - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            print(f"  Passes {next_start}-{next_start + args.passes - 1}")
            print(f"{'#'*60}")

            # Pre-check rate limit before starting harvester
            penalty = check_rate_limit()
            if penalty > 120:
                print(f"\n  API rate limit active: {penalty}s remaining. "
                      f"Skipping this cycle.")
                if args.once:
                    print("  (Run again after the penalty expires.)")
                    break
                print(f"  Will retry in {args.cooldown}s...")
                time.sleep(args.cooldown)
                cycle -= 1  # Don't count skipped cycle
                continue

            records_before = count_records()

            success = run_harvest(passes=args.passes, start_pass=next_start)
            records_after = count_records()
            new_records = records_after - records_before

            print(f"\n  Cycle {cycle} harvest complete: "
                  f"+{new_records} records ({records_after} total)")

            # Advance pass counter for next cycle
            next_start += args.passes

            if not args.no_accuracy:
                metrics = run_accuracy_check()
                if metrics:
                    log_accuracy(metrics, cycle=cycle)
                    print_tracking_history()

            if args.once:
                print("\n  Single cycle complete. Exiting.")
                break

            elapsed = time.time() - cycle_start
            print(f"\n  Cycle took {elapsed/60:.1f} min. "
                  f"Cooling down {args.cooldown}s...")
            time.sleep(args.cooldown)

    except KeyboardInterrupt:
        print(f"\n\n  Stopped after {cycle} cycle(s).")
        print_tracking_history()


if __name__ == "__main__":
    main()
