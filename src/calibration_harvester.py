"""
POE2 Price Overlay - Calibration Harvester

Standalone CLI script that queries the trade API for listed rare items,
scores them locally using ModDatabase, and records (score, actual_price)
calibration pairs to calibration.jsonl.

Target: 400-600 samples across all equipment categories in ~10 minutes.

Usage:
    python calibration_harvester.py
    python calibration_harvester.py --dry-run
    python calibration_harvester.py --categories rings,amulets --max-queries 8
    python calibration_harvester.py --league "Fate of the Vaal"
"""

import argparse
import json
import logging
import random
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import (
    CALIBRATION_LOG_FILE,
    CACHE_DIR,
    DEFAULT_LEAGUE,
    HARVESTER_STATE_FILE,
    TRADE_API_BASE,
)
from item_parser import ParsedItem
from mod_database import ModDatabase
from mod_parser import ModParser, ParsedMod
from trade_client import TradeClient

logger = logging.getLogger(__name__)

# ─── Category Definitions ────────────────────────────
# Maps short name → (trade API category filter, item_class for scoring)
# Trade API category values: https://www.pathofexile.com/api/trade2/data/filters
CATEGORIES: Dict[str, Tuple[str, str]] = {
    "rings":            ("accessory.ring",        "Rings"),
    "amulets":          ("accessory.amulet",      "Amulets"),
    "belts":            ("accessory.belt",         "Belts"),
    "body_armours":     ("armour.chest",           "Body Armours"),
    "boots":            ("armour.boots",           "Boots"),
    "gloves":           ("armour.gloves",          "Gloves"),
    "helmets":          ("armour.helmet",          "Helmets"),
    "shields":          ("armour.shield",          "Shields"),
    "quivers":          ("armour.quiver",          "Quivers"),
    "foci":             ("armour.focus",           "Foci"),
    "bows":             ("weapon.bow",             "Bows"),
    "crossbows":        ("weapon.crossbow",        "Crossbows"),
    "one_hand_swords":  ("weapon.onesword",        "One Hand Swords"),
    "two_hand_swords":  ("weapon.twosword",        "Two Hand Swords"),
    "one_hand_axes":    ("weapon.oneaxe",          "One Hand Axes"),
    "two_hand_axes":    ("weapon.twoaxe",          "Two Hand Axes"),
    "one_hand_maces":   ("weapon.onemace",         "One Hand Maces"),
    "two_hand_maces":   ("weapon.twomace",         "Two Hand Maces"),
    "daggers":          ("weapon.dagger",          "Daggers"),
    "claws":            ("weapon.claw",            "Claws"),
    "wands":            ("weapon.wand",            "Wands"),
    "staves":           ("weapon.staff",           "Staves"),
    "sceptres":         ("weapon.sceptre",         "Sceptres"),
    "flails":           ("weapon.flail",           "Flails"),
}

# Price brackets: (label, min_price, max_price, currency)
# Using exalted for cheap bracket, divine for the rest
PRICE_BRACKETS = [
    ("cheap",   50,  100, "exalted"),
    ("mid",      1,    5, "divine"),
    ("high",     5,   50, "divine"),
    ("premium", 50, None, "divine"),
]

RESULTS_PER_QUERY = 10
MIN_INTERVAL = 2.5  # seconds between API calls
LONG_PENALTY_THRESHOLD = 300  # seconds — bail if penalty exceeds this


# ─── Query Building ──────────────────────────────────

def build_harvester_query(category_filter: str, price_min: float,
                          price_max: Optional[float],
                          price_currency: str,
                          league: str) -> Tuple[str, dict]:
    """Build a trade API search query for rare items in a category + price range.

    Returns (search_url, query_body).
    """
    price_filter = {"min": price_min}
    if price_max is not None:
        price_filter["max"] = price_max

    query = {
        "query": {
            "status": {"option": "online"},
            "stats": [{"type": "and", "filters": []}],
            "filters": {
                "type_filters": {
                    "filters": {
                        "category": {"option": category_filter},
                        "rarity": {"option": "rare"},
                    }
                },
                "trade_filters": {
                    "filters": {
                        "price": {
                            **price_filter,
                            "option": price_currency,
                        }
                    }
                },
            },
        },
        "sort": {"price": "asc"},
    }

    url = f"{TRADE_API_BASE}/search/poe2/{league}"
    return url, query


# ─── Item Construction from Listing ──────────────────

def listing_to_parsed_item(listing: dict, item_class: str) -> Optional[ParsedItem]:
    """Convert a trade API listing into a ParsedItem + mod tuples.

    Returns (ParsedItem, [(mod_type, text), ...]) or (None, []) on failure.
    """
    item_data = listing.get("item", {})
    if not item_data:
        return None

    item = ParsedItem()
    item.name = item_data.get("name", "")
    item.base_type = item_data.get("typeLine", "")
    item.item_class = item_class
    item.rarity = "rare"
    item.item_level = item_data.get("ilvl", 0)

    # Combat stats from extended data
    extended = item_data.get("extended", {})
    if extended:
        dps = extended.get("dps", 0) or 0
        pdps = extended.get("pdps", 0) or 0
        edps = extended.get("edps", 0) or 0
        item.total_dps = float(dps) if dps else float(pdps) + float(edps)
        item.physical_dps = float(pdps) if pdps else 0.0
        item.elemental_dps = float(edps) if edps else 0.0

        ar = extended.get("ar", 0) or 0
        ev = extended.get("ev", 0) or 0
        es = extended.get("es", 0) or 0
        item.armour = int(ar)
        item.evasion = int(ev)
        item.energy_shield = int(es)
        item.total_defense = item.armour + item.evasion + item.energy_shield

    # Collect all mod lines as (mod_type, text) tuples
    mods = []
    for mod_text in item_data.get("explicitMods", []):
        mods.append(("explicit", mod_text))
    for mod_text in item_data.get("implicitMods", []):
        mods.append(("implicit", mod_text))
    for mod_text in item_data.get("fracturedMods", []):
        mods.append(("fractured", mod_text))
    for mod_text in item_data.get("enchantMods", []):
        mods.append(("enchant", mod_text))
    for mod_text in item_data.get("craftedMods", []):
        mods.append(("crafted", mod_text))

    item.mods = mods
    return item


def extract_price_divine(listing: dict, trade_client: TradeClient) -> Optional[float]:
    """Extract the listed price from a listing and normalize to divine."""
    price_info = listing.get("listing", {}).get("price", {})
    amount = price_info.get("amount", 0)
    currency = price_info.get("currency", "")

    if amount <= 0 or not currency:
        return None

    divine_to_chaos = trade_client._divine_to_chaos_fn()
    return trade_client._normalize_to_divine(amount, currency, divine_to_chaos)


# ─── State Management ────────────────────────────────

def load_state() -> dict:
    """Load harvester state from disk."""
    if HARVESTER_STATE_FILE.exists():
        try:
            with open(HARVESTER_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed_queries": [], "total_samples": 0,
            "query_plan_seed": ""}


def save_state(state: dict):
    """Persist harvester state to disk."""
    HARVESTER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HARVESTER_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def make_query_key(cat_name: str, bracket_label: str) -> str:
    """Deterministic key for a category+bracket pair."""
    return f"{cat_name}:{bracket_label}"


def build_query_plan(categories: Dict[str, Tuple[str, str]],
                     seed: str) -> List[Tuple[str, str, str, str]]:
    """Build the full query plan: list of (cat_name, item_class, cat_filter, bracket_label).

    Deterministically shuffled by seed so re-runs on the same day
    process items in the same order.
    """
    plan = []
    for cat_name, (cat_filter, item_class) in categories.items():
        for bracket_label, _, _, _ in PRICE_BRACKETS:
            plan.append((cat_name, item_class, cat_filter, bracket_label))

    rng = random.Random(seed)
    rng.shuffle(plan)
    return plan


# ─── Calibration Record Writing ──────────────────────

def write_calibration_record(score_result, price_divine: float,
                             item_class: str):
    """Append a calibration record in the same format as _log_calibration()."""
    record = {
        "ts": int(time.time()),
        "grade": score_result.grade.value,
        "score": round(score_result.normalized_score, 3),
        "item_class": item_class,
        "top_mods": score_result.top_mods_summary,
        "min_divine": round(price_divine, 4),
        "max_divine": round(price_divine, 4),
        "results": 1,
        "estimate": False,
        "total_dps": round(score_result.total_dps, 1),
        "total_defense": score_result.total_defense,
        "dps_factor": round(score_result.dps_factor, 3),
        "defense_factor": round(score_result.defense_factor, 3),
        "somv_factor": round(score_result.somv_factor, 3),
    }

    CALIBRATION_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ─── Main Harvester Loop ────────────────────────────

def run_harvester(league: str, categories: Dict[str, Tuple[str, str]],
                  dry_run: bool = False, max_queries: int = 0):
    """Execute the harvester: query trade API, score items, write calibration data."""

    # Initialize components
    print(f"League: {league}")
    print(f"Categories: {len(categories)}")
    print(f"Brackets: {len(PRICE_BRACKETS)}")
    total_queries = len(categories) * len(PRICE_BRACKETS)
    print(f"Total queries: {total_queries}")

    if max_queries > 0:
        print(f"Max queries: {max_queries}")

    # Build deterministic query plan (same seed = same order per day)
    today_seed = date.today().isoformat()
    plan = build_query_plan(categories, today_seed)

    # Load state for resumability
    state = load_state()
    if state.get("query_plan_seed") != today_seed:
        # New day — reset state
        state = {"completed_queries": [], "total_samples": 0,
                 "query_plan_seed": today_seed}
        save_state(state)

    completed = set(state["completed_queries"])
    remaining = [(cn, ic, cf, bl) for cn, ic, cf, bl in plan
                 if make_query_key(cn, bl) not in completed]

    if not remaining:
        print(f"All {total_queries} queries already completed today. "
              f"({state['total_samples']} samples collected)")
        return

    print(f"Remaining queries: {len(remaining)} "
          f"(skipping {len(completed)} already completed)")

    if dry_run:
        print("\n--- DRY RUN: Query Plan ---")
        for i, (cn, ic, cf, bl) in enumerate(remaining):
            bracket = next(b for b in PRICE_BRACKETS if b[0] == bl)
            price_str = f"{bracket[1]}-{bracket[2] or 'max'} {bracket[3]}"
            print(f"  {i+1:3d}. {cn:20s} {bl:8s} ({price_str})")
            if max_queries > 0 and i + 1 >= max_queries:
                print(f"  ... (capped at {max_queries})")
                break
        print(f"\nTotal: {min(len(remaining), max_queries or len(remaining))} queries")
        return

    # Initialize ModParser + ModDatabase
    print("\nLoading mod parser and database...")
    mod_parser = ModParser()
    mod_parser.load_stats()
    if not mod_parser.loaded:
        print("ERROR: ModParser failed to load stats. Check network connection.")
        sys.exit(1)

    mod_db = ModDatabase()
    if not mod_db.load(mod_parser):
        print("ERROR: ModDatabase failed to load. Check network connection.")
        sys.exit(1)

    db_stats = mod_db.get_stats()
    print(f"ModDatabase ready: bridge={db_stats['bridge_size']}, "
          f"ladders={db_stats['ladder_count']}")

    # Initialize TradeClient (no rate conversion needed — we read prices directly)
    trade_client = TradeClient(league=league)

    print(f"\nStarting harvest...")
    session = trade_client._session
    queries_done = 0
    samples_this_run = 0
    skipped_no_mods = 0
    skipped_low_price = 0
    skipped_fake = 0
    errors = 0
    t_start = time.time()

    for cat_name, item_class, cat_filter, bracket_label in remaining:
        if max_queries > 0 and queries_done >= max_queries:
            print(f"\nReached max queries ({max_queries}), stopping.")
            break

        query_key = make_query_key(cat_name, bracket_label)
        bracket = next(b for b in PRICE_BRACKETS if b[0] == bracket_label)
        _, price_min, price_max, price_currency = bracket

        print(f"\n[{queries_done+1}/{min(len(remaining), max_queries or len(remaining))}] "
              f"{cat_name} / {bracket_label} "
              f"({price_min}-{price_max or 'max'} {price_currency})")

        # Rate limit
        trade_client._rate_limit()
        time.sleep(max(0, MIN_INTERVAL - trade_client._min_interval))

        # Check for long penalty
        if trade_client._is_rate_limited():
            wait = trade_client._rate_limited_until - time.time()
            if wait > LONG_PENALTY_THRESHOLD:
                print(f"  Rate limited for {wait:.0f}s (>{LONG_PENALTY_THRESHOLD}s) "
                      f"— saving state and exiting.")
                save_state(state)
                sys.exit(0)
            print(f"  Rate limited, waiting {wait:.0f}s...")
            time.sleep(wait + 1)

        # Step 1: Search
        search_url, query_body = build_harvester_query(
            cat_filter, price_min, price_max, price_currency, league)

        try:
            trade_client._rate_limit()
            resp = session.post(search_url, json=query_body, timeout=10)
            trade_client._parse_rate_limit_headers(resp)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 5))
                if retry_after > LONG_PENALTY_THRESHOLD:
                    print(f"  429 with penalty {retry_after}s — saving and exiting.")
                    save_state(state)
                    sys.exit(0)
                print(f"  429 — waiting {retry_after}s...")
                time.sleep(retry_after)
                trade_client._rate_limit()
                resp = session.post(search_url, json=query_body, timeout=10)
                trade_client._parse_rate_limit_headers(resp)

            if resp.status_code != 200:
                print(f"  Search HTTP {resp.status_code}, skipping")
                errors += 1
                queries_done += 1
                state["completed_queries"].append(query_key)
                continue

            search_data = resp.json()
            query_id = search_data.get("id")
            result_ids = search_data.get("result", [])
            total = search_data.get("total", 0)

            if not query_id or not result_ids:
                print(f"  0 results")
                queries_done += 1
                state["completed_queries"].append(query_key)
                save_state(state)
                continue

            print(f"  {total} total results, fetching {min(len(result_ids), RESULTS_PER_QUERY)}...")

        except Exception as e:
            print(f"  Search error: {e}")
            errors += 1
            queries_done += 1
            state["completed_queries"].append(query_key)
            continue

        # Step 2: Fetch listings
        fetch_ids = result_ids[:RESULTS_PER_QUERY]
        try:
            # Rate limit before fetch
            trade_client._rate_limit()
            time.sleep(max(0, MIN_INTERVAL - trade_client._min_interval))

            listings = trade_client._do_fetch(query_id, fetch_ids)
            if not listings:
                print(f"  Fetch returned no listings")
                queries_done += 1
                state["completed_queries"].append(query_key)
                save_state(state)
                continue

        except Exception as e:
            print(f"  Fetch error: {e}")
            errors += 1
            queries_done += 1
            state["completed_queries"].append(query_key)
            continue

        # Step 3: Score each listing and write calibration records
        batch_samples = 0
        last_grade = "-"
        for listing in listings:
            # Extract price
            price_div = extract_price_divine(listing, trade_client)
            if price_div is None or price_div <= 0:
                skipped_low_price += 1
                continue

            # Build ParsedItem from listing
            item = listing_to_parsed_item(listing, item_class)
            if item is None:
                continue

            # Parse mods
            parsed_mods = mod_parser.parse_mods(item)
            if not parsed_mods:
                skipped_no_mods += 1
                continue

            # Score item
            try:
                score = mod_db.score_item(item, parsed_mods)
            except Exception as e:
                logger.debug(f"Score error: {e}")
                continue

            # Sanity filter: reject listings where score and price
            # wildly disagree — these are fake/price-fixer listings.
            # JUNK items listed at 5+ divine, or C items at 50+ divine
            # are almost certainly not real.
            grade = score.grade.value
            if grade == "JUNK" and price_div >= 5:
                skipped_fake += 1
                continue
            if grade == "C" and price_div >= 50:
                skipped_fake += 1
                continue
            if grade in ("JUNK", "C") and score.normalized_score < 0.3 and price_div >= 20:
                skipped_fake += 1
                continue

            # Write calibration record
            try:
                write_calibration_record(score, price_div, item_class)
                last_grade = score.grade.value
                batch_samples += 1
                samples_this_run += 1
                state["total_samples"] += 1
            except Exception as e:
                logger.debug(f"Write error: {e}")

        print(f"  Scored {batch_samples}/{len(listings)} items -> "
              f"{last_grade} (running total: {samples_this_run})")

        queries_done += 1
        state["completed_queries"].append(query_key)
        save_state(state)

    # Final summary
    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"Harvest complete!")
    print(f"  Queries: {queries_done}")
    print(f"  Samples collected: {samples_this_run}")
    print(f"  Total samples (all runs today): {state['total_samples']}")
    print(f"  Skipped (no mods): {skipped_no_mods}")
    print(f"  Skipped (bad price): {skipped_low_price}")
    print(f"  Skipped (fake/price-fixer): {skipped_fake}")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Output: {CALIBRATION_LOG_FILE}")
    print(f"{'='*50}")

    save_state(state)


# ─── CLI Entry Point ─────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Harvest calibration data from the POE2 trade API")
    parser.add_argument("--league", default=DEFAULT_LEAGUE,
                        help=f"League name (default: {DEFAULT_LEAGUE})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print query plan without making API calls")
    parser.add_argument("--categories",
                        help="Comma-separated category names to harvest "
                             "(default: all). Use --dry-run to see names.")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Max number of queries to run (0 = unlimited)")
    parser.add_argument("--reset", action="store_true",
                        help="Reset today's state and start fresh")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    # Quiet down noisy loggers unless verbose
    if not args.verbose:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("mod_parser").setLevel(logging.WARNING)
        logging.getLogger("mod_database").setLevel(logging.WARNING)
        logging.getLogger("trade_client").setLevel(logging.WARNING)

    # Filter categories
    cats = CATEGORIES
    if args.categories:
        requested = [c.strip().lower() for c in args.categories.split(",")]
        cats = {}
        for name in requested:
            if name in CATEGORIES:
                cats[name] = CATEGORIES[name]
            else:
                print(f"Unknown category: {name}")
                print(f"Available: {', '.join(sorted(CATEGORIES.keys()))}")
                sys.exit(1)

    # Reset state if requested
    if args.reset:
        if HARVESTER_STATE_FILE.exists():
            HARVESTER_STATE_FILE.unlink()
            print("State reset.")

    run_harvester(
        league=args.league,
        categories=cats,
        dry_run=args.dry_run,
        max_queries=args.max_queries,
    )


if __name__ == "__main__":
    main()
