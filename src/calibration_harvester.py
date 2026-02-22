"""
LAMA - Calibration Harvester

Standalone CLI script that queries the trade API for listed rare items,
scores them locally using ModDatabase, and records (score, actual_price)
calibration pairs to a shard output file.

Target: 3,600+ samples per run across all equipment categories.

Usage:
    python calibration_harvester.py
    python calibration_harvester.py --dry-run
    python calibration_harvester.py --categories rings,amulets --max-queries 8
    python calibration_harvester.py --league "Fate of the Vaal"
    python calibration_harvester.py --passes 3
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
    CACHE_DIR,
    DEFAULT_LEAGUE,
    HARVESTER_STATE_FILE,
    TRADE_API_BASE,
    CALIBRATION_MAX_PRICE_DIVINE,
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
    "one_hand_maces":   ("weapon.onemace",         "One Hand Maces"),
    "two_hand_maces":   ("weapon.twomace",         "Two Hand Maces"),
    "wands":            ("weapon.wand",            "Wands"),
    "staves":           ("weapon.staff",           "Staves"),
    "sceptres":         ("weapon.sceptre",         "Sceptres"),
}

# Price brackets: (label, min_price, max_price, currency)
# Primary brackets (passes 1-5)
_PRIMARY_BRACKETS = [
    ("very_cheap",  10,   50, "exalted"),
    ("cheap",       50,  100, "exalted"),
    ("low_mid",      1,    2, "divine"),
    ("mid",          2,    5, "divine"),
    ("high_low",     5,   15, "divine"),
    ("high",        15,   50, "divine"),
    ("premium",     50,  150, "divine"),
    ("ultra",      150,  300, "divine"),
]

# Stagger brackets: offset ranges that sample gaps between primary brackets
_STAGGER_BRACKETS = [
    ("stag_exalt",  25,   75, "exalted"),
    ("stag_cheap",  75,  150, "exalted"),
    ("stag_low",     1,    3, "divine"),
    ("stag_mid",     3,    8, "divine"),
    ("stag_high",    8,   25, "divine"),
    ("stag_prem",   25,   80, "divine"),
    ("stag_ultra",  80,  200, "divine"),
    ("stag_top",   200,  300, "divine"),
]

# Micro brackets: fine-grained resolution in the 1-50d range where most
# real pricing action happens, plus extra exalted coverage
_MICRO_BRACKETS = [
    ("micro_ex1",    5,   30, "exalted"),
    ("micro_ex2",   30,   60, "exalted"),
    ("micro_ex3",   60,  100, "exalted"),
    ("micro_1",      1,    2, "divine"),
    ("micro_2",      2,    4, "divine"),
    ("micro_3",      4,    7, "divine"),
    ("micro_4",      7,   12, "divine"),
    ("micro_5",     12,   20, "divine"),
    ("micro_6",     20,   35, "divine"),
    ("micro_7",     35,   60, "divine"),
    ("micro_8",     60,  100, "divine"),
    ("micro_9",    100,  200, "divine"),
]

_BRACKET_SETS = [_PRIMARY_BRACKETS, _STAGGER_BRACKETS, _MICRO_BRACKETS]

def get_brackets_for_pass(pass_num: int):
    """Return the bracket set for a given pass number.
    Cycles through primary -> stagger -> micro, 5 passes each."""
    set_index = (pass_num - 1) // 5
    if set_index >= len(_BRACKET_SETS):
        # Wrap around for very long runs
        set_index = set_index % len(_BRACKET_SETS)
    return _BRACKET_SETS[set_index]

def bracket_set_name(pass_num: int) -> str:
    """Human-readable name for the bracket set used by a pass."""
    set_index = ((pass_num - 1) // 5) % len(_BRACKET_SETS)
    return ["primary", "stagger", "micro"][set_index]

# For backward compat and dry-run display
PRICE_BRACKETS = _PRIMARY_BRACKETS

RESULTS_PER_QUERY = 50
FETCH_BATCH_SIZE = 10  # Trade API caps fetches at 10 IDs per request
BURST_SIZE = 4  # API calls per burst before pausing
BURST_PAUSE = 8.0  # seconds to pause between bursts
LONG_PENALTY_THRESHOLD = 300  # seconds — log warning for long penalties


# ─── Output File ──────────────────────────────────────

def get_shard_output_path(league: str, pass_num: int = 1) -> Path:
    """Return path for harvester output: calibration_shard_{league}_{date}_p{pass}.jsonl"""
    league_slug = league.lower().replace(" ", "_")
    today = date.today().isoformat()
    return CACHE_DIR / f"calibration_shard_{league_slug}_{today}_p{pass_num}.jsonl"


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

def load_state(pass_num: int = 1) -> dict:
    """Load harvester state from disk."""
    state_file = HARVESTER_STATE_FILE.parent / f"harvester_state_p{pass_num}.json"
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed_queries": [], "total_samples": 0,
            "query_plan_seed": "", "dead_combos": []}


def save_state(state: dict, pass_num: int = 1):
    """Persist harvester state to disk."""
    state_file = HARVESTER_STATE_FILE.parent / f"harvester_state_p{pass_num}.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def make_query_key(cat_name: str, bracket_label: str) -> str:
    """Deterministic key for a category+bracket pair."""
    return f"{cat_name}:{bracket_label}"


def build_query_plan(categories: Dict[str, Tuple[str, str]],
                     seed: str,
                     brackets=None) -> List[Tuple[str, str, str, str]]:
    """Build the full query plan: list of (cat_name, item_class, cat_filter, bracket_label).

    Deterministically shuffled by seed so re-runs on the same day
    process items in the same order.
    """
    if brackets is None:
        brackets = PRICE_BRACKETS
    plan = []
    for cat_name, (cat_filter, item_class) in categories.items():
        for bracket_label, _, _, _ in brackets:
            plan.append((cat_name, item_class, cat_filter, bracket_label))

    rng = random.Random(seed)
    rng.shuffle(plan)
    return plan


# ─── Fake/Price-Fixer Detection ──────────────────────

def is_fake_listing(grade: str, score: float, price_div: float,
                    n_explicit_mods: int) -> bool:
    """Return True if this listing looks like a price-fixer or fake.

    Balanced thresholds: strict enough to filter obvious fakes,
    loose enough to capture legitimate high-price items where
    our grading disagrees with the market (these are valuable
    calibration data points).
    """
    # JUNK items listed at 5+ divine
    if grade == "JUNK" and price_div >= 5:
        return True
    # C-grade items listed at 50+ divine
    if grade == "C" and price_div >= 50:
        return True
    # Very low-score JUNK/C at 30+ divine
    if grade in ("JUNK", "C") and score < 0.2 and price_div >= 30:
        return True
    # B-grade with very low score at 150+ divine
    if grade == "B" and score < 0.35 and price_div >= 150:
        return True
    # Items with only 1 explicit mod listed at 50+ divine — suspicious
    if n_explicit_mods <= 1 and price_div >= 50:
        return True
    # Price above absolute cap
    if price_div > CALIBRATION_MAX_PRICE_DIVINE:
        return True
    return False


# ─── Calibration Record Writing ──────────────────────

def write_calibration_record(score_result, price_divine: float,
                             item_class: str, league: str,
                             output_file: Path):
    """Append a calibration record to the shard output file."""
    record = {
        "ts": int(time.time()),
        "league": league,
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

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ─── Main Harvester Loop ────────────────────────────

def run_harvester(league: str, categories: Dict[str, Tuple[str, str]],
                  dry_run: bool = False, max_queries: int = 0,
                  pass_num: int = 1, resume: bool = False):
    """Execute the harvester: query trade API, score items, write calibration data."""

    output_file = get_shard_output_path(league, pass_num)
    brackets = get_brackets_for_pass(pass_num)

    # Initialize components
    print(f"League: {league}")
    print(f"Categories: {len(categories)}")
    print(f"Brackets: {len(brackets)} ({bracket_set_name(pass_num)})")
    total_queries = len(categories) * len(brackets)
    print(f"Total queries: {total_queries}")
    print(f"Pass: {pass_num}")
    print(f"Output: {output_file}")

    if max_queries > 0:
        print(f"Max queries: {max_queries}")

    # Build deterministic query plan (seed varies by pass for different offsets)
    today_seed = f"{date.today().isoformat()}:p{pass_num}"

    # Load state for resumability
    state = load_state(pass_num)

    if resume and state.get("query_plan_seed"):
        # --resume: reuse the existing seed so the query plan matches,
        # allowing continuation of an interrupted run across days
        seed = state["query_plan_seed"]
        print(f"  Resuming with saved seed: {seed}")
    else:
        seed = today_seed
        if state.get("query_plan_seed") != seed:
            # New day or new pass — reset state
            state = {"completed_queries": [], "total_samples": 0,
                     "query_plan_seed": seed, "dead_combos": []}
            save_state(state, pass_num)

    plan = build_query_plan(categories, seed, brackets)

    completed = set(state["completed_queries"])
    dead_combos = set(state.get("dead_combos", []))
    remaining = [(cn, ic, cf, bl) for cn, ic, cf, bl in plan
                 if make_query_key(cn, bl) not in completed
                 and make_query_key(cn, bl) not in dead_combos]

    if not remaining:
        print(f"All {total_queries} queries already completed for pass {pass_num}. "
              f"({state['total_samples']} samples collected)")
        return

    skipped_dead = len(dead_combos - completed)
    print(f"Remaining queries: {len(remaining)} "
          f"(skipping {len(completed)} completed"
          f"{f', {skipped_dead} dead' if skipped_dead else ''})")

    if dry_run:
        print("\n--- DRY RUN: Query Plan ---")
        for i, (cn, ic, cf, bl) in enumerate(remaining):
            bracket = next(b for b in brackets if b[0] == bl)
            price_str = f"{bracket[1]}-{bracket[2] or 'max'} {bracket[3]}"
            print(f"  {i+1:3d}. {cn:20s} {bl:10s} ({price_str})")
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

    print(f"\nStarting harvest (pass {pass_num})...")
    session = trade_client._session
    queries_done = 0
    samples_this_run = 0
    skipped_no_mods = 0
    skipped_low_price = 0
    skipped_fake = 0
    errors = 0
    burst_count = 0  # API calls in current burst
    t_start = time.time()
    effective_total = min(len(remaining), max_queries or len(remaining))

    for cat_name, item_class, cat_filter, bracket_label in remaining:
        if max_queries > 0 and queries_done >= max_queries:
            print(f"\nReached max queries ({max_queries}), stopping.")
            break

        query_key = make_query_key(cat_name, bracket_label)
        bracket = next(b for b in brackets if b[0] == bracket_label)
        _, price_min, price_max, price_currency = bracket

        elapsed = time.time() - t_start
        pct = queries_done / effective_total * 100 if effective_total else 0
        eta_str = ""
        if queries_done > 0:
            eta_sec = elapsed / queries_done * (effective_total - queries_done)
            eta_str = f" | ETA {eta_sec/60:.0f}m"

        print(f"\n[{queries_done+1}/{effective_total}] ({pct:.0f}%) "
              f"{cat_name} / {bracket_label} "
              f"({price_min}-{price_max or 'max'} {price_currency}) "
              f"| {samples_this_run} samples | {elapsed/60:.1f}m{eta_str}")

        # Burst pacing: pause after every BURST_SIZE API calls
        if burst_count >= BURST_SIZE:
            time.sleep(BURST_PAUSE)
            burst_count = 0

        # Rate limit
        trade_client._rate_limit()

        # Check for long penalty
        if trade_client._is_rate_limited():
            wait = trade_client._rate_limited_until - time.time()
            if wait > LONG_PENALTY_THRESHOLD:
                print(f"  Rate limited for {wait:.0f}s (>{LONG_PENALTY_THRESHOLD}s) "
                      f"— long wait, saving state first...")
                save_state(state, pass_num)
            print(f"  Rate limited, waiting {wait:.0f}s...")
            time.sleep(wait + 1)

        # Step 1: Search
        search_url, query_body = build_harvester_query(
            cat_filter, price_min, price_max, price_currency, league)

        try:
            trade_client._rate_limit()
            resp = session.post(search_url, json=query_body, timeout=10)
            trade_client._parse_rate_limit_headers(resp)
            burst_count += 1

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 5))
                if retry_after > LONG_PENALTY_THRESHOLD:
                    print(f"  429 with penalty {retry_after}s — long wait, "
                          f"saving state first...")
                    save_state(state, pass_num)
                print(f"  429 — waiting {retry_after}s...")
                time.sleep(retry_after + 1)
                burst_count = 0  # Reset burst after penalty wait
                trade_client._rate_limit()
                resp = session.post(search_url, json=query_body, timeout=10)
                trade_client._parse_rate_limit_headers(resp)
                burst_count += 1

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
                print(f"  0 results (marking dead)")
                queries_done += 1
                state["completed_queries"].append(query_key)
                if query_key not in state.get("dead_combos", []):
                    state.setdefault("dead_combos", []).append(query_key)
                save_state(state, pass_num)
                continue

            # For multi-pass: offset into results to get different items
            # Offset resets per bracket set (every 5 passes) so stagger/micro
            # passes start from offset 0, not from the global pass number.
            pass_within_set = (pass_num - 1) % 5
            offset = pass_within_set * RESULTS_PER_QUERY
            available_ids = result_ids[offset:offset + RESULTS_PER_QUERY]
            if not available_ids:
                # No more results at this offset
                print(f"  {total} total results, no new results at offset {offset}")
                queries_done += 1
                state["completed_queries"].append(query_key)
                save_state(state, pass_num)
                continue

            print(f"  {total} total results, fetching {len(available_ids)} "
                  f"(offset {offset})...")

        except Exception as e:
            print(f"  Search error: {e}")
            errors += 1
            queries_done += 1
            state["completed_queries"].append(query_key)
            continue

        # Step 2: Fetch listings (batched — API caps at 10 IDs per request)
        listings = []
        fetch_failed = False
        for batch_start in range(0, len(available_ids), FETCH_BATCH_SIZE):
            batch_ids = available_ids[batch_start:batch_start + FETCH_BATCH_SIZE]
            try:
                if burst_count >= BURST_SIZE:
                    time.sleep(BURST_PAUSE)
                    burst_count = 0
                trade_client._rate_limit()

                batch_listings = trade_client._do_fetch(query_id, batch_ids)
                burst_count += 1
                if batch_listings:
                    listings.extend(batch_listings)
            except Exception as e:
                print(f"  Fetch error (batch {batch_start}): {e}")
                errors += 1
                fetch_failed = True
                break

        if not listings:
            if fetch_failed:
                queries_done += 1
                state["completed_queries"].append(query_key)
                continue
            print(f"  Fetch returned no listings")
            queries_done += 1
            state["completed_queries"].append(query_key)
            save_state(state, pass_num)
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

            # Count explicit mods for fake detection
            n_explicit = len(listing.get("item", {}).get("explicitMods", []))

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

            # Fake/price-fixer detection
            grade = score.grade.value
            if is_fake_listing(grade, score.normalized_score, price_div,
                               n_explicit):
                skipped_fake += 1
                continue

            # Write calibration record to shard output file
            try:
                write_calibration_record(score, price_div, item_class,
                                         league, output_file)
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
        save_state(state, pass_num)

    # Final summary
    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"Harvest complete (pass {pass_num})!")
    print(f"  Queries: {queries_done}")
    print(f"  Samples collected: {samples_this_run}")
    print(f"  Total samples (all runs today): {state['total_samples']}")
    print(f"  Skipped (no mods): {skipped_no_mods}")
    print(f"  Skipped (bad price): {skipped_low_price}")
    print(f"  Skipped (fake/price-fixer): {skipped_fake}")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Output: {output_file}")
    print(f"{'='*50}")

    save_state(state, pass_num)


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
    parser.add_argument("--passes", type=int, default=1,
                        help="Number of passes with different offsets "
                             "(default: 1). Each pass samples deeper into listings.")
    parser.add_argument("--start-pass", type=int, default=1,
                        help="First pass number to run (default: 1). "
                             "Use with --passes to skip already-completed passes.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an interrupted run (reuses saved seed, "
                             "skips completed queries even across days)")
    parser.add_argument("--reset", action="store_true",
                        help="Reset today's state and start fresh")
    parser.add_argument("--reset-dead", action="store_true",
                        help="Clear dead combo lists from all pass state files")
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
        for p in range(1, args.passes + 1):
            sf = HARVESTER_STATE_FILE.parent / f"harvester_state_p{p}.json"
            if sf.exists():
                sf.unlink()
        # Also reset legacy state file
        if HARVESTER_STATE_FILE.exists():
            HARVESTER_STATE_FILE.unlink()
        print("State reset.")

    # Clear dead combos if requested
    if args.reset_dead:
        import glob as _glob
        pattern = str(HARVESTER_STATE_FILE.parent / "harvester_state_p*.json")
        cleared = 0
        for sf_path in _glob.glob(pattern):
            try:
                with open(sf_path, "r", encoding="utf-8") as f:
                    st = json.load(f)
                if st.get("dead_combos"):
                    n = len(st["dead_combos"])
                    st["dead_combos"] = []
                    with open(sf_path, "w", encoding="utf-8") as f:
                        json.dump(st, f, indent=2)
                    cleared += n
                    print(f"  Cleared {n} dead combos from {Path(sf_path).name}")
            except Exception:
                pass
        print(f"Dead combos cleared ({cleared} total).")

    # Run each pass
    for pass_num in range(args.start_pass, args.passes + 1):
        if args.passes > 1:
            print(f"\n{'#'*50}")
            print(f"  PASS {pass_num} of {args.passes}")
            print(f"{'#'*50}")

        run_harvester(
            league=args.league,
            categories=cats,
            dry_run=args.dry_run,
            max_queries=args.max_queries,
            pass_num=pass_num,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
