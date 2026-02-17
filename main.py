"""
POE2 Price Overlay - Main Application
Orchestrates all components:
    Cursor Stop → Ctrl+C → Clipboard Parse → Price Lookup → Overlay Display

Usage:
    python main.py                    # Default league
    python main.py --league "Dawn"    # Specific league
    python main.py --console          # Console output (no GUI overlay)
    python main.py --debug            # Verbose logging
"""

import sys
import os
import re
import time
import queue
import logging
import argparse
import threading
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEFAULT_LEAGUE,
    LOG_LEVEL,
    LOG_FILE,
)
from item_detection import ItemDetector
from item_parser import ItemParser
from price_cache import PriceCache
from overlay import PriceOverlay, ConsoleOverlay
from mod_parser import ModParser
from trade_client import TradeClient
from filter_updater import FilterUpdater, find_template_filter
from mod_database import ModDatabase
from calibration import CalibrationEngine
from bug_reporter import BugReporter

logger = logging.getLogger("poe2-overlay")


class POE2PriceOverlay:
    """
    Main application class.

    Pipeline:
    1. ItemDetector sends Ctrl+C when cursor stops over POE2
    2. ItemParser.parse_clipboard() structures the clipboard text
    3. PriceCache looks up the price
    4. Overlay displays the result
    """

    def __init__(self, league: str = DEFAULT_LEAGUE, use_console: bool = False,
                 no_filter_update: bool = False, test_filter_update: bool = False):
        self.league = league.strip()
        self._no_filter_update = no_filter_update
        self._test_filter_update = test_filter_update

        # Initialize components
        logger.info("Initializing POE2 Price Overlay...")
        logger.info(f"League: {self.league}")

        self.price_cache = PriceCache(league=self.league)
        self.item_detector = ItemDetector()
        self.item_parser = ItemParser()
        self.mod_parser = ModParser()
        self.trade_client = TradeClient(
            league=self.league,
            divine_to_chaos_fn=lambda: self.price_cache.divine_to_chaos,
            divine_to_exalted_fn=lambda: self.price_cache.divine_to_exalted,
        )
        self.mod_database = ModDatabase()
        self.calibration = CalibrationEngine()

        # Deep query state: last scored item (for Ctrl+Shift+C trade lookup)
        self._last_scored_item = None
        self._last_scored_mods = None
        self._last_scored_result = None
        self._last_scored_cursor = (0, 0)
        self._last_scored_lock = threading.Lock()

        # Auto-calibration queue: A/S graded items get trade API lookups in background
        self._calibration_queue = queue.Queue()

        # Filter updater
        project_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        template = find_template_filter(project_dir)
        self.filter_updater = FilterUpdater(
            self.price_cache, template,
            test_mode=self._test_filter_update,
        )
        if template:
            logger.info(f"Filter template: {template.name}")
        else:
            logger.info("No .filter template found — filter updater disabled")

        if use_console:
            self.overlay = ConsoleOverlay()
        else:
            self.overlay = PriceOverlay()

        # Trade query cancellation: increment on each new detection so
        # stale trade queries abort before wasting API calls.
        self._trade_generation = 0
        self._trade_gen_lock = threading.Lock()

        # Statistics
        self.stats = {
            "triggers": 0,
            "successful_lookups": 0,
            "failed_read": 0,
            "failed_parse": 0,
            "not_found": 0,
            "start_time": 0,
        }

        # Bug reporter (Ctrl+Shift+B)
        self.bug_reporter = BugReporter(
            root_fn=lambda: self.overlay._root,
            stats_fn=lambda: self.stats,
            overlay=self.overlay,
        )

        # Wire up detection callbacks
        self.item_detector.set_callback(self._on_change_detected)
        self.item_detector.set_hide_callback(self.overlay.hide)

    def start(self):
        """
        Start all components and begin monitoring.
        """
        logger.info("=" * 50)
        logger.info("  POE2 Price Overlay - Starting")
        logger.info("=" * 50)

        self.stats["start_time"] = time.time()

        # 1. Start price cache (background refresh)
        logger.info("Loading price data...")
        self.price_cache.start()

        # Wait briefly for initial price data
        time.sleep(1)
        cache_stats = self.price_cache.get_stats()
        logger.info(f"Price cache: {cache_stats['total_items']} items loaded")

        if cache_stats['total_items'] == 0:
            logger.warning(
                "No price data loaded yet. Prices will appear once "
                "poe.ninja data finishes downloading."
            )

        # 1b. Load mod parser stat definitions (for rare item pricing)
        logger.info("Loading trade stat definitions...")
        self.mod_parser.load_stats()
        if self.mod_parser.loaded:
            logger.info("Rare item pricing enabled")
        else:
            logger.warning("Rare item pricing disabled (no stat data)")

        # 1b2. Load mod tier database for local scoring
        if self.mod_parser.loaded:
            logger.info("Loading mod tier database...")
            if self.mod_database.load(self.mod_parser):
                stats = self.mod_database.get_stats()
                logger.info(f"Local scoring ready (bridge={stats['bridge_size']}, ladders={stats['ladder_count']})")
                # Load calibration data for score→price estimation
                from config import CALIBRATION_LOG_FILE
                n = self.calibration.load(CALIBRATION_LOG_FILE)
                if n:
                    logger.info(f"Calibration: {n} samples loaded")
            else:
                logger.warning("Local scoring disabled — falling back to trade API")

        # 1c. Handle filter update
        if self._test_filter_update:
            logger.info("Running filter update (test mode: hidden items show as tiny text)...")
            self.filter_updater.update_now(dry_run=False)
            logger.info("Test filter written. Reload in-game to verify.")
            return
        if not self._no_filter_update:
            self.filter_updater.start()

        # 2. Start item detection in background thread
        logger.info("Starting item detection (clipboard mode)...")
        detect_thread = threading.Thread(
            target=self.item_detector.run_detection_loop,
            daemon=True,
            name="ItemDetector"
        )
        detect_thread.start()

        # 2b. Deep query hotkey listener (Ctrl+Shift+C)
        if self.mod_database.loaded:
            threading.Thread(
                target=self._deep_query_hotkey_loop,
                daemon=True,
                name="DeepQueryHotkey",
            ).start()

        # 2c. Bug report hotkey listener (Ctrl+Shift+B)
        threading.Thread(
            target=self._bug_report_hotkey_loop,
            daemon=True,
            name="BugReportHotkey",
        ).start()

        # 2d. Auto-calibration queue processor
        if self.mod_database.loaded:
            threading.Thread(
                target=self._calibration_queue_loop,
                daemon=True,
                name="CalibrationQueue",
            ).start()

        # 3. Start status reporting
        status_thread = threading.Thread(
            target=self._status_loop,
            daemon=True,
            name="StatusReporter"
        )
        status_thread.start()

        logger.info("Ready! Hover over items in POE2 to see prices.")
        logger.info("Close this window to stop.\n")

        # 4. Run overlay on main thread (tkinter requirement)
        try:
            self.overlay.run()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Shut down all components."""
        logger.info("\nShutting down...")
        self.filter_updater.stop()
        self.price_cache.stop()
        self.overlay.shutdown()
        self._print_session_stats()

    # ─── Core Pipeline ───────────────────────────────

    def _on_change_detected(self, item_text: str, cursor_x: int, cursor_y: int):
        """
        Called by ItemDetector when Ctrl+C returns item data.
        This is the core pipeline that parses the text and shows a price.
        """
        self.stats["triggers"] += 1
        start_time = time.time()

        try:
            # Debug: save clipboard text
            if logger.isEnabledFor(logging.DEBUG):
                self._save_debug_text(item_text, cursor_x, cursor_y)

            # Step 1: Parse clipboard-format item data
            item = self.item_parser.parse_clipboard(item_text)

            if not item:
                self.stats["failed_parse"] += 1
                logger.info(f"Parse failed: {item_text.split(chr(10))[0]}")
                return

            logger.info(
                f"Item: {item.name} ({item.rarity})"
                + (f" base={item.base_type}" if item.base_type else "")
                + (" [unidentified]" if item.unidentified else "")
            )

            # Skip worthless currency shards — not worth displaying
            item_lower = (item.name or "").lower()
            if any(s in item_lower for s in self._WORTHLESS_ITEMS):
                logger.info(f"Worthless item: {item.name}")
                self.overlay.show_price(
                    text="\u2717", tier="low",
                    cursor_x=cursor_x, cursor_y=cursor_y,
                )
                return

            # Chanceable bases: normal items that can become valuable uniques
            base_lower = (item.base_type or "").lower()
            if item.rarity == "normal" and base_lower in self._CHANCEABLE_BASES:
                unique_name = self._CHANCEABLE_BASES[base_lower]
                # Try to pull the unique's price from cache
                unique_result = self.price_cache.lookup(unique_name)
                if unique_result:
                    price_str = unique_result["display"]
                    tier = unique_result["tier"]
                    divine = unique_result.get("divine_value", 0)
                else:
                    price_str = "valuable"
                    tier = "good"
                    divine = 0
                logger.info(f"Chanceable base: {item.base_type} → {unique_name} ({price_str})")
                self.overlay.show_price(
                    text=f"Chance \u2192 {unique_name} ({price_str})",
                    tier=tier,
                    cursor_x=cursor_x, cursor_y=cursor_y,
                    price_divine=divine,
                )
                self.stats["successful_lookups"] += 1
                return

            # Unidentified items: can't price rares/magic without mods
            if item.unidentified:
                base = item.base_type or item.name

                # Only look up possible uniques when the rarity IS unique
                if item.rarity == "unique":
                    result = self.price_cache.lookup_unidentified(base)
                    if result:
                        logger.info(
                            f">>> PRICE [unid] {base}: {result['display']} "
                            f"({result['name']})"
                        )
                        self.overlay.show_price(
                            text=f"{base} (unid): {result['display']}",
                            tier=result["tier"],
                            cursor_x=cursor_x,
                            cursor_y=cursor_y,
                            price_divine=result.get("divine_value", 0),
                        )
                        self.stats["successful_lookups"] += 1
                        return

                # Rare/magic/unknown unidentified — no mods to price
                logger.info(f"Unidentified {item.rarity}: {base}")
                self.overlay.show_price(
                    text="\u2717", tier="low",
                    cursor_x=cursor_x, cursor_y=cursor_y,
                )
                return

            # Step 1c: Corrupted uniques → trade API for Vaal-outcome-aware pricing
            if item.rarity == "unique" and getattr(item, "corrupted", False):
                static_result = self.price_cache.lookup(
                    item_name=item.lookup_key,
                    base_type=item.base_type,
                    item_level=item.item_level,
                )
                self._price_unique_async(item, cursor_x, cursor_y,
                                         static_result=static_result)
                return

            # Step 2: Non-unique items with mods → local scoring (or trade API fallback)
            if (item.rarity in ("rare", "magic") and item.mods
                    and self.mod_parser.loaded):
                # Resolve magic item base_type if missing
                if not item.base_type and item.name:
                    resolved = self.mod_parser.resolve_base_type(item.name)
                    if resolved:
                        item.base_type = resolved

                parsed_mods = self.mod_parser.parse_mods(item)
                if not parsed_mods:
                    self.overlay.show_price(
                        text="\u2717", tier="low",
                        cursor_x=cursor_x, cursor_y=cursor_y,
                    )
                    return

                # Primary path: local scoring (instant, no API calls)
                if self.mod_database.loaded:
                    self._score_and_display(item, parsed_mods, cursor_x, cursor_y)
                    return

                # Fallback: trade API (if mod database failed to load)
                if self._has_only_common_mods(item.mods):
                    self.overlay.show_price(
                        text="\u2717", tier="low",
                        cursor_x=cursor_x, cursor_y=cursor_y,
                    )
                    return
                self._price_rare_async(item, cursor_x, cursor_y)
                return

            # Step 2b: Normal/magic items with sockets → trade API for base pricing
            if (item.rarity in ("normal", "magic") and item.sockets >= 2):
                self._price_base_async(item, cursor_x, cursor_y)
                return

            # Step 3: Static price lookup (uniques, currency, gems)
            result = self.price_cache.lookup(
                item_name=item.lookup_key,
                base_type=item.base_type,
                item_level=item.item_level,
            )

            if not result:
                # Step 3b: Fallback — search clipboard text directly against cache
                result = self.price_cache.lookup_from_text(item_text)

            if not result:
                self.stats["not_found"] += 1
                logger.info(f"No price: {item.lookup_key} (base: {item.base_type})")
                self.overlay.show_price(
                    text="\u2717", tier="low",
                    cursor_x=cursor_x, cursor_y=cursor_y,
                )
                return

            # Step 3: Display the price
            elapsed = (time.time() - start_time) * 1000
            matched_name = result.get("name", item.name)
            logger.info(
                f">>> PRICE [{elapsed:.0f}ms] {matched_name}: "
                f"{result['display']}"
            )

            self.overlay.show_price(
                text=f"{matched_name}: {result['display']}",
                tier=result["tier"],
                cursor_x=cursor_x,
                cursor_y=cursor_y,
                price_divine=result.get("divine_value", 0),
            )
            self.stats["successful_lookups"] += 1

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)

    def _price_rare_async(self, item, cursor_x: int, cursor_y: int):
        """
        Price a rare item via the trade API in a background thread.
        Shows animated "Checking." indicator, then updates with result.
        Cancels any previous in-flight trade query via generation counter.
        """
        display_name = item.name or item.base_type

        # Increment generation — any in-flight trade query with an older
        # generation will abort before making more API calls.
        with self._trade_gen_lock:
            self._trade_generation += 1
            my_gen = self._trade_generation

        def _is_stale():
            """Check if a newer item detection has superseded us."""
            with self._trade_gen_lock:
                return my_gen != self._trade_generation

        # Show initial checking indicator
        self.overlay.show_price(
            text=f"{display_name}: Checking.",
            tier="low",
            cursor_x=cursor_x,
            cursor_y=cursor_y,
        )

        # Animated dots: cycles ". → .. → ..." while searching
        search_done = threading.Event()

        def _animate_dots():
            dots = 1
            while not search_done.wait(0.4):
                if _is_stale():
                    return
                dots = (dots % 3) + 1
                self.overlay.show_price(
                    text=f"{display_name}: Checking{'.' * dots}",
                    tier="low",
                    cursor_x=cursor_x,
                    cursor_y=cursor_y,
                )

        def _do_price():
            anim = threading.Thread(
                target=_animate_dots, daemon=True, name="PriceAnim")
            anim.start()
            try:
                # Resolve missing base_type for magic items (name includes
                # prefix + base + suffix, e.g. "Mystic Stellar Amulet of the Fox")
                if not item.base_type and item.name:
                    resolved = self.mod_parser.resolve_base_type(item.name)
                    if resolved:
                        item.base_type = resolved
                        logger.info(f"Resolved base type: '{item.name}' → '{resolved}'")
                    else:
                        logger.info(f"Could not resolve base type for '{item.name}'")

                if _is_stale():
                    logger.debug(f"Trade query cancelled (stale): {display_name}")
                    return

                parsed_mods = self.mod_parser.parse_mods(item)
                if not parsed_mods:
                    logger.info(f"No mods matched for {display_name}")
                    self.overlay.show_price(
                        text="\u2717", tier="low",
                        cursor_x=cursor_x, cursor_y=cursor_y,
                    )
                    self.stats["not_found"] += 1
                    return

                logger.info(f"Matched {len(parsed_mods)} mods for {display_name}")

                result = self.trade_client.price_rare_item(
                    item, parsed_mods, is_stale=_is_stale)
                if _is_stale():
                    logger.debug(f"Trade result discarded (stale): {display_name}")
                    return
                if result:
                    self.overlay.show_price(
                        text=f"{display_name}: {result.display}",
                        tier=result.tier,
                        cursor_x=cursor_x,
                        cursor_y=cursor_y,
                        estimate=result.estimate,
                        price_divine=result.min_price,
                    )
                    self.stats["successful_lookups"] += 1
                else:
                    self.overlay.show_price(
                        text="\u2717", tier="low",
                        cursor_x=cursor_x, cursor_y=cursor_y,
                    )
                    self.stats["not_found"] += 1
            except Exception as e:
                logger.error(f"Rare pricing error: {e}", exc_info=True)
                self.overlay.show_price(
                    text=f"{display_name}: ?", tier="low",
                    cursor_x=cursor_x, cursor_y=cursor_y,
                )
                self.stats["not_found"] += 1
            finally:
                search_done.set()

        thread = threading.Thread(target=_do_price, daemon=True, name="RarePricer")
        thread.start()

    def _price_base_async(self, item, cursor_x: int, cursor_y: int):
        """
        Price a normal/magic base item via the trade API in a background thread.
        Used for items valued by their base type + sockets (e.g., 3-socket bases).
        """
        display_name = item.base_type or item.name
        sockets = item.sockets
        ilvl = getattr(item, "item_level", 0) or 0
        tag = f"{sockets}S, ilvl {ilvl}" if ilvl > 0 else f"{sockets}S"

        with self._trade_gen_lock:
            self._trade_generation += 1
            my_gen = self._trade_generation

        def _is_stale():
            with self._trade_gen_lock:
                return my_gen != self._trade_generation

        # Show initial checking indicator
        self.overlay.show_price(
            text=f"{display_name} ({tag}): Checking.",
            tier="low",
            cursor_x=cursor_x,
            cursor_y=cursor_y,
        )

        # Animated dots: cycles ". → .. → ..." while searching
        search_done = threading.Event()

        def _animate_dots():
            dots = 1
            while not search_done.wait(0.4):
                if _is_stale():
                    return
                dots = (dots % 3) + 1
                self.overlay.show_price(
                    text=f"{display_name} ({tag}): Checking{'.' * dots}",
                    tier="low",
                    cursor_x=cursor_x,
                    cursor_y=cursor_y,
                )

        def _do_price():
            anim = threading.Thread(
                target=_animate_dots, daemon=True, name="BaseAnim")
            anim.start()
            try:
                if _is_stale():
                    return

                result = self.trade_client.price_base_item(
                    item, is_stale=_is_stale)
                if _is_stale():
                    return
                if result:
                    self.overlay.show_price(
                        text=f"{display_name} ({tag}): {result.display}",
                        tier=result.tier,
                        cursor_x=cursor_x,
                        cursor_y=cursor_y,
                        price_divine=result.min_price,
                    )
                    self.stats["successful_lookups"] += 1
                else:
                    self.overlay.show_price(
                        text="\u2717", tier="low",
                        cursor_x=cursor_x, cursor_y=cursor_y,
                    )
                    self.stats["not_found"] += 1
            except Exception as e:
                logger.error(f"Base pricing error: {e}", exc_info=True)
                self.overlay.show_price(
                    text=f"{display_name}: ?", tier="low",
                    cursor_x=cursor_x, cursor_y=cursor_y,
                )
                self.stats["not_found"] += 1
            finally:
                search_done.set()

        thread = threading.Thread(target=_do_price, daemon=True, name="BasePricer")
        thread.start()

    def _price_unique_async(self, item, cursor_x: int, cursor_y: int,
                            static_result: dict = None):
        """Price a corrupted unique via the trade API in a background thread.

        Shows static price immediately if available, then upgrades with
        trade API result that reflects actual Vaal outcomes + sockets.
        """
        display_name = item.name or item.base_type
        sockets = getattr(item, "sockets", 0) or 0
        tag = f"{sockets}S corrupted" if sockets else "corrupted"

        # Show static price while trade API loads
        if static_result:
            static_display = static_result.get("display", "?")
            self.overlay.show_price(
                text=f"{display_name} ({tag}): Checking.",
                tier=static_result.get("tier", "low"),
                cursor_x=cursor_x, cursor_y=cursor_y,
                price_divine=static_result.get("divine_value", 0),
            )
        else:
            self.overlay.show_price(
                text=f"{display_name} ({tag}): Checking.",
                tier="low",
                cursor_x=cursor_x, cursor_y=cursor_y,
            )

        with self._trade_gen_lock:
            self._trade_generation += 1
            my_gen = self._trade_generation

        def _is_stale():
            with self._trade_gen_lock:
                return my_gen != self._trade_generation

        search_done = threading.Event()

        def _animate_dots():
            dots = 1
            while not search_done.wait(0.4):
                if _is_stale():
                    return
                dots = (dots % 3) + 1
                self.overlay.show_price(
                    text=f"{display_name} ({tag}): Checking{'.' * dots}",
                    tier="low",
                    cursor_x=cursor_x, cursor_y=cursor_y,
                )

        def _do_price():
            anim = threading.Thread(
                target=_animate_dots, daemon=True, name="UniqueAnim")
            anim.start()
            try:
                if _is_stale():
                    return

                result = self.trade_client.price_unique_item(
                    item, is_stale=_is_stale)
                if _is_stale():
                    return
                if result and result.min_price > 0:
                    self.overlay.show_price(
                        text=f"{display_name} ({tag}): {result.display}",
                        tier=result.tier,
                        cursor_x=cursor_x, cursor_y=cursor_y,
                        price_divine=result.min_price,
                    )
                    self.stats["successful_lookups"] += 1
                elif static_result:
                    # Trade API returned nothing — fall back to static
                    static_display = static_result.get("display", "?")
                    self.overlay.show_price(
                        text=f"{display_name}: {static_display}",
                        tier=static_result.get("tier", "low"),
                        cursor_x=cursor_x, cursor_y=cursor_y,
                        price_divine=static_result.get("divine_value", 0),
                    )
                    self.stats["successful_lookups"] += 1
                else:
                    self.overlay.show_price(
                        text="\u2717", tier="low",
                        cursor_x=cursor_x, cursor_y=cursor_y,
                    )
                    self.stats["not_found"] += 1
            except Exception as e:
                logger.error(f"Unique pricing error: {e}", exc_info=True)
                if static_result:
                    self.overlay.show_price(
                        text=f"{display_name}: {static_result.get('display', '?')}",
                        tier=static_result.get("tier", "low"),
                        cursor_x=cursor_x, cursor_y=cursor_y,
                        price_divine=static_result.get("divine_value", 0),
                    )
                else:
                    self.overlay.show_price(
                        text=f"{display_name}: ?", tier="low",
                        cursor_x=cursor_x, cursor_y=cursor_y,
                    )
                self.stats["not_found"] += 1
            finally:
                search_done.set()

        thread = threading.Thread(target=_do_price, daemon=True,
                                  name="UniquePricer")
        thread.start()

    # ─── Local Scoring + Deep Query ──────────────────

    def _score_and_display(self, item, parsed_mods, cursor_x, cursor_y):
        """Score item locally, display grade, store state for deep query."""
        from config import GRADE_TIER_MAP
        score = self.mod_database.score_item(item, parsed_mods)
        display_name = item.name or item.base_type

        # Store for deep query hotkey
        with self._last_scored_lock:
            self._last_scored_item = item
            self._last_scored_mods = parsed_mods
            self._last_scored_result = score
            self._last_scored_cursor = (cursor_x, cursor_y)

        overlay_tier = GRADE_TIER_MAP.get(score.grade.value, "low")

        # Query calibration for price estimate
        price_est = self.calibration.estimate(
            score.normalized_score, getattr(item, "item_class", "") or "",
            grade=score.grade.value)
        d2c = self.price_cache.divine_to_chaos
        text = score.format_overlay_text(price_estimate=price_est,
                                         divine_to_chaos=d2c)

        # If we have a price estimate, use price-based tier instead of grade-based.
        # Only override for B+ grades — C/JUNK calibration estimates are unreliable
        # and would show misleading colors (e.g. orange "C") with few samples.
        if price_est is not None and score.grade.value not in ("C", "JUNK"):
            chaos_val = price_est * d2c
            if chaos_val >= 25:
                overlay_tier = "high"
            elif chaos_val >= 5:
                overlay_tier = "good"
            elif chaos_val >= 1:
                overlay_tier = "decent"
            else:
                overlay_tier = "low"

        # Log combat factors when they modify the score
        if score.dps_factor != 1.0:
            logger.info(f"DPS factor: {score.dps_factor:.2f} (dps={score.total_dps:.0f})")
        if score.defense_factor != 1.0:
            logger.info(f"Defense factor: {score.defense_factor:.2f} (def={score.total_defense})")

        log_extra = f" est~{price_est:.1f}d" if price_est else ""
        logger.info(f"Grade {score.grade.value}: {display_name} "
                     f"(score={score.normalized_score:.3f}{log_extra}) "
                     f"{score.top_mods_summary}")

        self.overlay.show_price(text=text, tier=overlay_tier,
                                cursor_x=cursor_x, cursor_y=cursor_y)
        if score.grade.value not in ("C", "JUNK"):
            self.stats["successful_lookups"] += 1

        # Auto-queue for background trade API calibration.
        # A/S always, B sampled 1-in-3, C/JUNK sampled 1-in-10.
        # Sampling rates are for collection only — analysis weights
        # results equally regardless of sample rate.
        import random
        grade = score.grade.value
        if grade in ("A", "S"):
            self._calibration_queue.put((item, parsed_mods, score))
        elif grade == "B" and random.random() < 0.33:
            self._calibration_queue.put((item, parsed_mods, score))
        elif grade in ("C", "JUNK") and random.random() < 0.10:
            self._calibration_queue.put((item, parsed_mods, score))

    def _deep_query_hotkey_loop(self):
        """Poll for Ctrl+Shift+C to trigger trade API lookup on last scored item."""
        import ctypes
        VK_SHIFT, VK_CONTROL, VK_C = 0x10, 0x11, 0x43
        _gaks = ctypes.windll.user32.GetAsyncKeyState
        was_pressed = False

        while True:
            time.sleep(0.05)  # 20 Hz
            pressed = bool(_gaks(VK_CONTROL) & 0x8000
                           and _gaks(VK_SHIFT) & 0x8000
                           and _gaks(VK_C) & 0x8000)
            if pressed and not was_pressed:
                was_pressed = True
                self._trigger_deep_query()
            elif not pressed:
                was_pressed = False

    def _bug_report_hotkey_loop(self):
        """Poll for Ctrl+Shift+B to trigger bug report dialog."""
        import ctypes
        VK_SHIFT, VK_CONTROL, VK_B = 0x10, 0x11, 0x42
        _gaks = ctypes.windll.user32.GetAsyncKeyState
        was_pressed = False

        while True:
            time.sleep(0.05)  # 20 Hz
            pressed = bool(_gaks(VK_CONTROL) & 0x8000
                           and _gaks(VK_SHIFT) & 0x8000
                           and _gaks(VK_B) & 0x8000)
            if pressed and not was_pressed:
                was_pressed = True
                self.bug_reporter.report()
            elif not pressed:
                was_pressed = False

    def _trigger_deep_query(self):
        """Execute trade API lookup on the last locally-scored item."""
        with self._last_scored_lock:
            item = self._last_scored_item
            mods = self._last_scored_mods
            score = self._last_scored_result
            cx, cy = self._last_scored_cursor

        if not item or not mods:
            return
        if not self.item_detector.game_window.is_poe2_foreground():
            return

        logger.info(f"Deep query: {item.name or item.base_type}")
        self._price_rare_deep_async(item, mods, score, cx, cy)

    def _price_rare_deep_async(self, item, parsed_mods, score_result, cursor_x, cursor_y):
        """Trade API lookup for a locally-scored item (triggered by Ctrl+Shift+C)."""
        display_name = item.name or item.base_type
        grade_str = score_result.grade.value

        with self._trade_gen_lock:
            self._trade_generation += 1
            my_gen = self._trade_generation

        def _is_stale():
            with self._trade_gen_lock:
                return my_gen != self._trade_generation

        self.overlay.show_price(text=f"{grade_str}: Checking.",
                                tier="low",
                                cursor_x=cursor_x, cursor_y=cursor_y)
        search_done = threading.Event()

        def _animate_dots():
            dots = 1
            while not search_done.wait(0.4):
                if _is_stale():
                    return
                dots = (dots % 3) + 1
                self.overlay.show_price(
                    text=f"{grade_str}: Checking{'.' * dots}",
                    tier="low",
                    cursor_x=cursor_x, cursor_y=cursor_y)

        def _do_deep():
            threading.Thread(target=_animate_dots, daemon=True,
                             name="DeepAnim").start()
            try:
                if _is_stale():
                    return

                result = self.trade_client.price_rare_item(
                    item, parsed_mods, is_stale=_is_stale)
                if _is_stale():
                    return
                if result:
                    self.overlay.show_price(
                        text=f"{display_name}: {result.display}",
                        tier=result.tier,
                        cursor_x=cursor_x, cursor_y=cursor_y,
                        estimate=result.estimate,
                        price_divine=result.min_price)
                    self.stats["successful_lookups"] += 1
                    self._log_calibration(score_result, result, item)
                else:
                    self.overlay.show_price(
                        text=f"{grade_str}: No listings", tier="low",
                        cursor_x=cursor_x, cursor_y=cursor_y)
                    self.stats["not_found"] += 1
            except Exception as e:
                logger.error(f"Deep query error: {e}", exc_info=True)
                self.overlay.show_price(
                    text=f"{grade_str}: ?", tier="low",
                    cursor_x=cursor_x, cursor_y=cursor_y)
            finally:
                search_done.set()

        threading.Thread(target=_do_deep, daemon=True,
                         name="DeepQueryPricer").start()

    def _log_calibration(self, score_result, trade_result, item):
        """Append grade-vs-price calibration record and live-update engine."""
        import json
        from config import CALIBRATION_LOG_FILE
        try:
            record = {
                "ts": int(time.time()),
                "grade": score_result.grade.value,
                "score": round(score_result.normalized_score, 3),
                "item_class": getattr(item, "item_class", ""),
                "top_mods": score_result.top_mods_summary,
                "min_divine": trade_result.min_price,
                "max_divine": trade_result.max_price,
                "results": trade_result.num_results,
                "estimate": trade_result.estimate,
                "total_dps": round(score_result.total_dps, 1),
                "total_defense": score_result.total_defense,
                "dps_factor": round(score_result.dps_factor, 3),
                "defense_factor": round(score_result.defense_factor, 3),
            }
            CALIBRATION_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CALIBRATION_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            # Live-update calibration engine so estimates improve within session
            if trade_result.min_price > 0:
                self.calibration.add_sample(
                    score_result.normalized_score,
                    trade_result.min_price,
                    getattr(item, "item_class", ""),
                    grade=score_result.grade.value)
        except Exception as e:
            logger.warning(f"Calibration log failed: {e}")

    def _calibration_queue_loop(self):
        """Process auto-queued items for trade API calibration."""
        while True:
            item, parsed_mods, score_result = self._calibration_queue.get()
            display_name = item.name or item.base_type
            try:
                # Wait out any active rate limit before attempting
                while self.trade_client._is_rate_limited():
                    wait = self.trade_client._rate_limited_until - time.time()
                    if wait > 0:
                        time.sleep(min(wait + 1, 65))

                result = self.trade_client.price_rare_item(
                    item, parsed_mods, is_stale=lambda: False)
                if result and "Rate limited" in result.display:
                    # Still rate limited — re-queue and back off
                    self._calibration_queue.put((item, parsed_mods, score_result))
                    time.sleep(10)
                elif result and result.min_price > 0:
                    self._log_calibration(score_result, result, item)
                    logger.info(
                        f"Auto-cal: {display_name} "
                        f"grade={score_result.grade.value} → {result.display}")
                else:
                    logger.info(
                        f"Auto-cal: {display_name} "
                        f"grade={score_result.grade.value} → no listings")
            except Exception as e:
                logger.warning(f"Auto-cal failed ({display_name}): {e}")

    # Items that should always show ✗ (too cheap to bother pricing)
    _WORTHLESS_ITEMS = (
        "chance shard", "transmutation shard", "regal shard",
        "artificer's shard",
    )

    # Normal base types that can be chanced into valuable uniques.
    # Maps base_type (lowercase) → unique name for price lookup.
    _CHANCEABLE_BASES = {
        "heavy belt": "Headhunter",
        "tribal mask": "The Vertex",
    }

    # High-roll detection: don't dismiss common mods with exceptional values
    _MOD_VALUE_RE = re.compile(r'[+-]?(\d+(?:\.\d+)?)')
    _HIGH_ROLL_THRESHOLD = 100       # flat values: +158 mana → keep
    _HIGH_ROLL_PCT_THRESHOLD = 50    # percentage values: 60% regen → keep

    def _has_only_common_mods(self, mods: list) -> bool:
        """Check if all mods are common filler (not worth trade API lookup).
        Implicit mods are always considered common (inherent to base type).
        High-roll common mods (e.g., +158 mana) are NOT considered common.
        Uses the canonical pattern list from TradeClient."""
        patterns = TradeClient._COMMON_MOD_PATTERNS
        for mod_type, mod_text in mods:
            if mod_type == "implicit":
                continue  # Implicits don't drive item value
            text_lower = mod_text.lower()
            if not any(pat in text_lower for pat in patterns):
                return False
            # Common pattern matched — but check if the roll is high enough
            # to be valuable despite being a "common" mod type
            m = self._MOD_VALUE_RE.search(mod_text)
            if m:
                value = float(m.group(1))
                is_pct = "%" in mod_text or "increased" in text_lower or "reduced" in text_lower
                threshold = self._HIGH_ROLL_PCT_THRESHOLD if is_pct else self._HIGH_ROLL_THRESHOLD
                if value >= threshold:
                    return False  # High roll — don't dismiss
        return True

    def _save_debug_text(self, text: str, cx: int, cy: int):
        """Save clipboard text for debugging."""
        try:
            debug_dir = Path(os.path.expanduser("~")) / ".poe2-price-overlay" / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            txt_path = debug_dir / f"clipboard_{timestamp}_{cx}_{cy}.txt"
            txt_path.write_text(text or "(empty)", encoding="utf-8")

            logger.debug(f"Debug text saved: {txt_path}")
        except Exception as e:
            logger.warning(f"Failed to save debug text: {e}")

    # ─── Status & Diagnostics ────────────────────────

    def _status_loop(self):
        """Periodically log status information."""
        while True:
            time.sleep(60)  # Every minute
            try:
                cache_stats = self.price_cache.get_stats()
                uptime = time.time() - self.stats["start_time"]
                total = self.stats["triggers"]
                hits = self.stats["successful_lookups"]
                hit_rate = (hits / total * 100) if total > 0 else 0

                logger.info(
                    f"[Status] Uptime: {uptime/60:.0f}min | "
                    f"Triggers: {total} | Prices shown: {hits} ({hit_rate:.0f}%) | "
                    f"Cache: {cache_stats['total_items']} items | "
                    f"Last refresh: {cache_stats['last_refresh']}"
                )
            except Exception:
                pass

    def _print_session_stats(self):
        """Print session summary on exit."""
        uptime = time.time() - self.stats["start_time"]
        total = self.stats["triggers"]

        print("\n" + "=" * 50)
        print("  Session Summary")
        print("=" * 50)
        print(f"  Uptime:           {uptime/60:.1f} minutes")
        print(f"  Total triggers:   {total}")
        print(f"  Prices shown:     {self.stats['successful_lookups']}")
        print(f"  Parse failures:   {self.stats['failed_parse']}")
        print(f"  Items not priced: {self.stats['not_found']}")
        if total > 0:
            rate = self.stats['successful_lookups'] / total * 100
            print(f"  Success rate:     {rate:.1f}%")
        print("=" * 50)


# ─── Entry Point ─────────────────────────────────────

def setup_logging(debug: bool = False):
    """Configure logging.

    Console always shows INFO+ only (prices, status, errors).
    File gets DEBUG when --debug is used (full OCR text, captures, etc.).
    This keeps the terminal clean so prices are visible.
    """
    # Ensure log directory exists
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Console handler — always INFO level so prices stand out
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "%(asctime)s %(message)s",
        datefmt="%H:%M:%S"
    )
    console.setFormatter(console_fmt)

    # File handler — gets all detail when debug is enabled
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    file_handler.setFormatter(file_fmt)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)


def main():
    # Disable console Ctrl+C handling FIRST — before any threads start.
    # We send Ctrl+C via keybd_event to copy items from POE2, and the
    # Windows console would otherwise treat it as a terminate signal.
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleCtrlHandler(None, True)
        # Resize console window — find by title for Windows Terminal compatibility
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, "POE2 Price Overlay")
        if hwnd:
            user32.MoveWindow(hwnd, 100, 100, 650, 500, True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="POE2 Price Overlay - Real-time item pricing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run with default settings
  python main.py --league "Dawn"          # Use "Dawn" league
  python main.py --console --debug        # Debug mode, console output
        """
    )
    parser.add_argument(
        "--league", "-l",
        default=DEFAULT_LEAGUE,
        help=f"League name (default: {DEFAULT_LEAGUE})"
    )
    parser.add_argument(
        "--console", "-c",
        action="store_true",
        help="Use console output instead of GUI overlay"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--no-filter-update",
        action="store_true",
        help="Disable automatic loot filter updating"
    )
    parser.add_argument(
        "--test-filter-update",
        action="store_true",
        help="Dry-run filter update (parse + compute + print diff, no write)"
    )

    args = parser.parse_args()

    setup_logging(debug=args.debug)

    try:
        app = POE2PriceOverlay(
            league=args.league,
            use_console=args.console,
            no_filter_update=args.no_filter_update,
            test_filter_update=args.test_filter_update,
        )
        app.start()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
