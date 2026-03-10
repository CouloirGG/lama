"""
LAMA - Disappearance Tracker

Tracks whether harvested trade listings have disappeared (likely sold)
or are still listed (possibly overpriced). Items that disappear from
trade = confirmed sales = trustworthy training data.

Workflow:
1. Reads JSONL records that have `listing_id` fields (from harvester)
2. After configurable delay (4+ hours), batch-checks listing IDs via trade fetch API
3. IDs that return null = delisted (likely sold) -> sale_confidence = 3.0
4. IDs still present = still listed (possibly overpriced) -> sale_confidence = 0.3
5. IDs not yet checked -> sale_confidence = 1.0 (neutral)
6. Writes sale_confidence back to records

Usage:
    python disappearance_tracker.py --recheck --min-age 4h
    python disappearance_tracker.py --recheck --min-age 24h
    python disappearance_tracker.py --stats
"""

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import CACHE_DIR, DEFAULT_LEAGUE, TRADE_API_BASE

logger = logging.getLogger(__name__)

# Sale confidence values
CONFIDENCE_SOLD = 3.0       # Listing disappeared -> likely sold
CONFIDENCE_UNKNOWN = 1.0    # Not yet checked
CONFIDENCE_STALE = 0.3      # Still listed after 24h+ -> likely overpriced

# API constraints
FETCH_BATCH_SIZE = 10       # Trade API caps fetches at 10 IDs
BURST_SIZE = 4              # API calls before pausing
BURST_PAUSE = 8.0           # Seconds between bursts
MIN_AGE_DEFAULT = 1 * 3600  # 1 hour minimum before checking

# State file
TRACKER_STATE_FILE = CACHE_DIR / "disappearance_state.json"


def _parse_duration(s: str) -> int:
    """Parse a duration string like '4h', '24h', '30m' into seconds."""
    m = re.match(r'^(\d+)([hm])$', s.strip().lower())
    if not m:
        raise ValueError(f"Invalid duration: {s} (use e.g. '4h' or '30m')")
    val = int(m.group(1))
    unit = m.group(2)
    if unit == 'h':
        return val * 3600
    elif unit == 'm':
        return val * 60
    return val


def load_records_with_listing_ids(input_paths: List[str],
                                  min_age_sec: int) -> List[dict]:
    """Load JSONL records that have listing_id fields and are old enough.

    Returns records where:
    - listing_id is present and non-empty
    - record timestamp is at least min_age_sec seconds ago
    - sale_confidence has not already been set
    """
    import glob as _glob
    now = time.time()
    records = []
    expanded = []
    for p in input_paths:
        expanded.extend(_glob.glob(p))

    if not expanded:
        print(f"No input files found matching: {input_paths}")
        return []

    for path in expanded:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    listing_id = rec.get("listing_id", "")
                    if not listing_id:
                        continue

                    # Skip already-checked records
                    if "sale_confidence" in rec:
                        continue

                    # Check age
                    ts = rec.get("ts", 0)
                    if ts > 0 and (now - ts) < min_age_sec:
                        continue

                    rec["_source_file"] = path
                    records.append(rec)
        except Exception as e:
            print(f"Warning: could not read {path}: {e}")

    return records


def _get_query_id(session, league: str) -> Optional[str]:
    """Do a minimal search to obtain a valid query_id for fetch calls.

    The trade API requires a query parameter on fetch endpoints.
    We do a broad search for any rare ring (always has results) to get one.
    """
    league_slug = league.replace(" ", "+")
    search_url = f"{TRADE_API_BASE}/search/poe2/{league_slug}"
    query_body = {
        "query": {
            "status": {"option": "online"},
            "stats": [{"type": "and", "filters": []}],
            "filters": {
                "type_filters": {
                    "filters": {
                        "category": {"option": "accessory.ring"},
                        "rarity": {"option": "rare"},
                    }
                },
            },
        },
        "sort": {"price": "asc"},
    }
    try:
        resp = session.post(search_url, json=query_body, timeout=10)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 60))
            print(f"  Rate limited on search, waiting {retry_after}s...")
            time.sleep(retry_after + 1)
            resp = session.post(search_url, json=query_body, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            qid = data.get("id", "")
            if qid:
                return qid
        print(f"  Search for query_id failed: HTTP {resp.status_code}")
    except Exception as e:
        print(f"  Search for query_id error: {e}")
    return None


def batch_check_listings(listing_ids: List[str],
                         session=None,
                         query_id: str = "") -> Dict[str, bool]:
    """Check which listing IDs still exist on the trade API.

    Returns {listing_id: True} for IDs that still exist (still listed),
    and {listing_id: False} for IDs that returned null (delisted/sold).

    Requires a valid query_id from a prior search (trade API requirement).
    """
    import requests

    if session is None:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "LAMA-DisappearanceTracker/1.0 (contact: hello@couloir.gg)",
        })

    results = {}
    total_batches = (len(listing_ids) + FETCH_BATCH_SIZE - 1) // FETCH_BATCH_SIZE
    for batch_start in range(0, len(listing_ids), FETCH_BATCH_SIZE):
        batch = listing_ids[batch_start:batch_start + FETCH_BATCH_SIZE]
        batch_num = batch_start // FETCH_BATCH_SIZE + 1
        ids_str = ",".join(batch)
        url = f"{TRADE_API_BASE}/fetch/{ids_str}"
        params = {"query": query_id} if query_id else {}

        try:
            resp = session.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                print(f"  Rate limited, waiting {retry_after}s...")
                time.sleep(retry_after + 1)
                resp = session.get(url, params=params, timeout=15)

            if resp.status_code != 200:
                print(f"  Fetch HTTP {resp.status_code} for batch {batch_num}/{total_batches}")
                # Mark all as unknown on error
                for lid in batch:
                    results[lid] = None  # unknown
                continue

            data = resp.json()
            fetched = data.get("result", [])

            for i, lid in enumerate(batch):
                if i < len(fetched):
                    entry = fetched[i]
                    # null = delisted, dict = still present
                    results[lid] = entry is not None
                else:
                    results[lid] = None  # unknown

            if batch_num % 10 == 0 or batch_num == total_batches:
                sold_so_far = sum(1 for v in results.values() if v is False)
                listed_so_far = sum(1 for v in results.values() if v is True)
                print(f"  Batch {batch_num}/{total_batches}: "
                      f"{sold_so_far} sold, {listed_so_far} still listed")

        except Exception as e:
            print(f"  Fetch error at batch {batch_num}/{total_batches}: {e}")
            for lid in batch:
                results[lid] = None

        # Burst pacing
        if batch_num % BURST_SIZE == 0:
            time.sleep(BURST_PAUSE)

    return results


def _load_tracker_state() -> dict:
    """Load cumulative tracker state from disk."""
    if TRACKER_STATE_FILE.exists():
        try:
            with open(TRACKER_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"runs": 0, "total_checked": 0, "total_sold": 0, "total_stale": 0}


def _save_tracker_state(state: dict):
    """Save cumulative tracker state to disk."""
    TRACKER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRACKER_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _write_back_to_file(src_path: str, file_records: List[dict],
                         statuses: Dict[str, bool]) -> int:
    """Write sale_confidence back to a single source file.

    Returns the number of records updated.
    """
    try:
        with open(src_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  Error reading {src_path}: {e}")
        return 0

    # Build a set of listing IDs we need to update in this file
    lid_confidence = {}
    for rec in file_records:
        lid = rec.get("listing_id", "")
        if not lid:
            continue
        status = statuses.get(lid)
        if status is False:
            lid_confidence[lid] = CONFIDENCE_SOLD
        elif status is True:
            lid_confidence[lid] = CONFIDENCE_STALE
        # None (unknown) -> skip, don't write confidence

    if not lid_confidence:
        return 0

    # Rewrite the file, injecting sale_confidence into matching records
    updated_count = 0
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            new_lines.append(line)
            continue
        try:
            rec = json.loads(stripped)
            lid = rec.get("listing_id", "")
            if lid in lid_confidence:
                rec["sale_confidence"] = lid_confidence[lid]
                updated_count += 1
            new_lines.append(json.dumps(rec) + "\n")
        except json.JSONDecodeError:
            new_lines.append(line)

    # Atomic write: write to temp file in same directory, then rename.
    # This prevents data loss if the process crashes mid-write.
    src_dir = os.path.dirname(src_path) or "."
    try:
        fd, tmp_path = tempfile.mkstemp(dir=src_dir, suffix=".tmp",
                                        prefix=".disappearance_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        os.replace(tmp_path, src_path)
    except Exception as e:
        print(f"  Error writing {src_path}: {e}")
        # Clean up temp file if rename failed
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return 0

    return updated_count


def recheck_records(input_paths: List[str], min_age_sec: int,
                    dry_run: bool = False, league: str = DEFAULT_LEAGUE,
                    max_ids: int = 0):
    """Main recheck flow: load records, check listings, write back confidence."""
    import requests

    records = load_records_with_listing_ids(input_paths, min_age_sec)
    if not records:
        print("No eligible records to check (need listing_id, "
              f"min age {min_age_sec/3600:.1f}h, no existing sale_confidence).")
        return

    print(f"Found {len(records)} records to check")

    # Deduplicate listing IDs (same listing can appear in multiple records)
    lid_to_records: Dict[str, List[dict]] = {}
    for rec in records:
        lid = rec["listing_id"]
        if lid not in lid_to_records:
            lid_to_records[lid] = []
        lid_to_records[lid].append(rec)

    unique_ids = list(lid_to_records.keys())

    # Sort by newest first — items that sell tend to sell quickly,
    # so recently-listed items have the best signal for detecting sales.
    def _newest_ts(lid):
        recs = lid_to_records[lid]
        return max(r.get("ts", 0) for r in recs)
    unique_ids.sort(key=_newest_ts, reverse=True)

    # Batch limiting: cap IDs checked per run
    if max_ids > 0 and len(unique_ids) > max_ids:
        deferred = len(unique_ids) - max_ids
        print(f"Batch limit: checking {max_ids} of {len(unique_ids)} IDs "
              f"({deferred} deferred to future runs)")
        unique_ids = unique_ids[:max_ids]
        # Filter records to only those with IDs we're actually checking
        checked_set = set(unique_ids)
        records = [r for r in records if r["listing_id"] in checked_set]

    n_batches = (len(unique_ids) + FETCH_BATCH_SIZE - 1) // FETCH_BATCH_SIZE
    print(f"Unique listing IDs: {len(unique_ids)}")
    print(f"API calls needed: ~{n_batches + 1} (1 search + {n_batches} fetches)")

    if dry_run:
        print("Dry run — not making API calls.")
        return

    # Check listings
    session = requests.Session()
    session.headers.update({
        "User-Agent": "LAMA-DisappearanceTracker/1.0 (contact: hello@couloir.gg)",
    })

    # Get a query_id first (trade API requires it for fetch calls)
    print(f"Getting query_id from search (league: {league})...")
    query_id = _get_query_id(session, league)
    if not query_id:
        print("ERROR: Could not obtain query_id from trade API. Aborting.")
        session.close()
        return
    print(f"Got query_id: {query_id[:16]}...")

    print("Checking listing status...")
    statuses = batch_check_listings(unique_ids, session, query_id=query_id)

    # Compute results
    sold = sum(1 for v in statuses.values() if v is False)
    still_listed = sum(1 for v in statuses.values() if v is True)
    unknown = sum(1 for v in statuses.values() if v is None)
    print(f"\nResults:")
    print(f"  Sold (delisted): {sold}")
    print(f"  Still listed:    {still_listed}")
    print(f"  Unknown/error:   {unknown}")

    # Write sale_confidence back to source files
    # Group records by source file for efficient rewrite
    by_file: Dict[str, List[dict]] = {}
    for rec in records:
        src = rec.pop("_source_file", "")
        if src:
            if src not in by_file:
                by_file[src] = []
            by_file[src].append(rec)

    updated_count = 0
    for src_path, file_records in by_file.items():
        updated_count += _write_back_to_file(src_path, file_records, statuses)

    print(f"\nUpdated {updated_count} records with sale_confidence")

    # Update cumulative tracker state
    state = _load_tracker_state()
    state["runs"] += 1
    state["total_checked"] += len(unique_ids)
    state["total_sold"] += sold
    state["total_stale"] += still_listed
    state["last_run"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save_tracker_state(state)
    print(f"Tracker state saved ({state['total_checked']} cumulative IDs checked)")

    session.close()


def show_stats(input_paths: List[str]):
    """Show statistics about sale_confidence in records."""
    import glob as _glob
    expanded = []
    for p in input_paths:
        expanded.extend(_glob.glob(p))

    total = 0
    with_lid = 0
    with_sc = 0
    sc_counts = {CONFIDENCE_SOLD: 0, CONFIDENCE_UNKNOWN: 0, CONFIDENCE_STALE: 0}

    for path in expanded:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    total += 1
                    if rec.get("listing_id"):
                        with_lid += 1
                    sc = rec.get("sale_confidence")
                    if sc is not None:
                        with_sc += 1
                        # Bucket by closest known value
                        if sc >= 2.0:
                            sc_counts[CONFIDENCE_SOLD] += 1
                        elif sc <= 0.5:
                            sc_counts[CONFIDENCE_STALE] += 1
                        else:
                            sc_counts[CONFIDENCE_UNKNOWN] += 1
        except Exception:
            pass

    print(f"Records: {total}")
    print(f"With listing_id: {with_lid}")
    print(f"With sale_confidence: {with_sc}")
    if with_sc:
        print(f"  Sold (confidence={CONFIDENCE_SOLD}): {sc_counts[CONFIDENCE_SOLD]}")
        print(f"  Stale (confidence={CONFIDENCE_STALE}): {sc_counts[CONFIDENCE_STALE]}")
        print(f"  Unknown (confidence={CONFIDENCE_UNKNOWN}): {sc_counts[CONFIDENCE_UNKNOWN]}")


def main():
    parser = argparse.ArgumentParser(
        description="Track disappearing trade listings for confirmed sale data")
    parser.add_argument("--input", "-i", nargs="+",
                        default=[str(CACHE_DIR / "calibration_shard_*.jsonl")],
                        help="Input JSONL file(s) from harvester (supports globs)")
    parser.add_argument("--recheck", action="store_true",
                        help="Check listing IDs and write sale_confidence")
    parser.add_argument("--min-age", default="1h",
                        help="Minimum record age before checking (e.g. '1h', '4h', '24h')")
    parser.add_argument("--max-ids", type=int, default=5000,
                        help="Max listing IDs to check per run (default: 5000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be checked without API calls")
    parser.add_argument("--stats", action="store_true",
                        help="Show sale_confidence statistics")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.stats:
        show_stats(args.input)
        return

    if args.recheck:
        min_age = _parse_duration(args.min_age)
        print(f"Min age: {min_age/3600:.1f}h")
        recheck_records(args.input, min_age, dry_run=args.dry_run,
                        max_ids=args.max_ids)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
