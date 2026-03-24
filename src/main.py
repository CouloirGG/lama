"""
LAMA (Live Auction Market Assessor) - Main Application
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
import time
import logging
import argparse
import threading
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bundle_paths import APP_DIR
from config import (
    DEFAULT_LEAGUE,
    LOG_LEVEL,
    LOG_FILE,
    OVERLAY_REFERENCE_HEIGHT,
)
from item_detection import ItemDetector
from item_parser import ItemParser
from price_cache import PriceCache
from overlay import PriceOverlay, ConsoleOverlay
from mod_parser import ModParser
from filter_updater import FilterUpdater, find_template_filter
from mod_database import ModDatabase
from bug_reporter import BugReporter

logger = logging.getLogger("poe2-overlay")

# Currency below this chaos threshold → silent skip (no overlay)
_CURRENCY_SKIP_CHAOS = 2


class LAMA:
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
        logger.info("Initializing LAMA...")
        logger.info(f"League: {self.league}")

        self.price_cache = PriceCache(league=self.league)
        self.item_detector = ItemDetector()
        self.item_parser = ItemParser()
        self.mod_parser = ModParser()
        self.mod_database = ModDatabase()

        # Pipeline dedup: skip re-processing the same item within a short window
        self._last_pipeline_item: str = ""
        self._last_pipeline_time: float = 0
        self._PIPELINE_DEDUP_TTL: float = 2.0  # seconds

        # Filter updater
        template = find_template_filter(APP_DIR / "resources")
        self.filter_updater = FilterUpdater(
            self.price_cache, template,
            test_mode=self._test_filter_update,
        )
        if template:
            logger.info(f"Filter template: {template.name}")
        else:
            logger.info("No .filter template found — filter updater disabled")

        # Load overlay display settings from dashboard config
        self._display_settings = self._load_display_settings()

        if use_console:
            self.overlay = ConsoleOverlay()
        else:
            # Compute overlay scale factor from game window resolution
            scale = 1.0
            rect = self.item_detector.game_window._find_poe2_rect()
            if rect:
                game_h = rect[3] - rect[1]
                scale = max(0.6, min(1.5, game_h / OVERLAY_REFERENCE_HEIGHT))
                logger.info(f"Game window height {game_h}px -> overlay scale {scale:.2f}")
            else:
                logger.info("Game window not found — overlay scale defaulting to 1.0")

            theme = self._display_settings.get("overlay_theme", "poe2")
            pulse_style = self._display_settings.get("overlay_pulse_style", "sheen")
            self.overlay = PriceOverlay(theme=theme, pulse_style=pulse_style,
                                        scale_factor=scale, game_rect=rect)

        # Set overlay mode (stars_only suppresses popup overlays)
        if hasattr(self.overlay, 'set_overlay_mode'):
            self.overlay.set_overlay_mode(
                self._display_settings.get("overlay_mode", "stars_only"))

        # Apply custom tier styles to overlay (if any)
        if hasattr(self.overlay, 'load_custom_styles'):
            self.overlay.load_custom_styles(
                self._display_settings.get("overlay_tier_styles", {}))

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
            item_context_fn=lambda: None,
        )

        # Wire up detection callbacks
        self.item_detector.set_callback(self._on_change_detected)
        self.item_detector.set_hide_callback(self.overlay.hide)
        self.item_detector.set_reshow_callback(self._on_reshow)

    # Overlay mode → format_overlay_text flags
    _MODE_FLAGS = {
        "stars_only": {"show_grade": False, "show_price": False, "show_stars": False, "show_mods": False, "show_dps": False},
        "minimal":    {"show_grade": False, "show_price": True,  "show_stars": False, "show_mods": False, "show_dps": False},
        "standard":   {"show_grade": True,  "show_price": True,  "show_stars": True,  "show_mods": False, "show_dps": True},
        "detailed":   {"show_grade": True,  "show_price": True,  "show_stars": True,  "show_mods": True,  "show_dps": True},
    }

    @staticmethod
    def _load_display_settings() -> dict:
        """Load overlay display flags from dashboard settings file."""
        import json
        settings_file = Path(os.path.expanduser("~")) / ".poe2-price-overlay" / "dashboard_settings.json"
        defaults = {
            "overlay_mode": "stars_only",
            "overlay_show_low_value": False,
            "overlay_tier_styles": {},
            "overlay_theme": "poe2",
            "overlay_pulse_style": "sheen",
        }
        try:
            if settings_file.exists():
                with open(settings_file) as f:
                    saved = json.load(f)
                for key in defaults:
                    if key in saved:
                        defaults[key] = saved[key]
                # Migrate legacy per-toggle settings → overlay_mode
                if "overlay_mode" not in saved:
                    if saved.get("overlay_show_mods"):
                        defaults["overlay_mode"] = "detailed"
                    elif saved.get("overlay_show_grade", True) and saved.get("overlay_show_dps", True):
                        defaults["overlay_mode"] = "standard"
                    elif saved.get("overlay_show_price", True):
                        defaults["overlay_mode"] = "minimal"
                    else:
                        defaults["overlay_mode"] = "stars_only"
        except Exception:
            pass
        return defaults

    def start(self):
        """
        Start all components and begin monitoring.
        """
        logger.info("=" * 50)
        logger.info("  LAMA - Starting")
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
            else:
                logger.warning("Local scoring disabled")

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

        # 2b. Bug report hotkey listener (Ctrl+Shift+B)
        threading.Thread(
            target=self._bug_report_hotkey_loop,
            daemon=True,
            name="BugReportHotkey",
        ).start()

        # 2c. Star clear hotkey listener (Ctrl+Shift+X)
        threading.Thread(
            target=self._star_clear_hotkey_loop,
            daemon=True,
            name="StarClearHotkey",
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

    def _on_reshow(self, cursor_x: int, cursor_y: int):
        """Called when the user re-hovers the same item after moving away.

        Repositions the existing overlay at the new cursor position without
        re-running the pricing pipeline (no re-parse, re-score, or API calls).
        """
        self.overlay.reshow(cursor_x, cursor_y)

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

            # Pipeline dedup: skip if same item was just processed
            # Prevents duplicate processing from clipboard read races
            item_ident = f"{item.name}|{item.base_type}|{item.rarity}"
            now = time.time()
            if (item_ident == self._last_pipeline_item
                    and (now - self._last_pipeline_time) < self._PIPELINE_DEDUP_TTL):
                return
            self._last_pipeline_item = item_ident
            self._last_pipeline_time = now

            logger.info(
                f"Item: {item.name} ({item.rarity})"
                + (f" base={item.base_type}" if item.base_type else "")
                + (" [unidentified]" if item.unidentified else "")
            )

            # Currency: silent skip for low-value, clean overlay for valuable
            if item.rarity == "currency":
                result = self.price_cache.lookup(
                    item_name=item.lookup_key,
                    base_type=item.base_type,
                    item_level=item.item_level,
                )
                if not result:
                    logger.info(f"Unknown currency, skipping: {item.name}")
                    self.overlay.hide()
                    self.item_detector.suppress_reshow()
                    return  # unknown currency → skip silently
                chaos_val = result.get("divine_value", 0) * self.price_cache.divine_to_chaos
                if chaos_val < _CURRENCY_SKIP_CHAOS:
                    logger.info(f"Low-value currency ({chaos_val:.0f}c), skipping: {item.name}")
                    self.overlay.hide()
                    self.item_detector.suppress_reshow()
                    return  # low-value currency → skip silently
                # Show clean overlay for valuable currency
                static_text = result["display"]
                logger.info(f">>> PRICE {item.name}: {static_text}")
                self.overlay.show_price(
                    text=static_text,
                    tier=result["tier"],
                    cursor_x=cursor_x,
                    cursor_y=cursor_y,
                    price_divine=result.get("divine_value", 0),
                )
                # Star for valuable currency
                if chaos_val >= 5:
                    star_key = item.name or item.base_type or ""
                    if chaos_val >= 500:
                        self.overlay.place_star(cursor_x, cursor_y, "gold3", item_key=star_key, item_class=getattr(item, "item_class", "") or "")
                    elif chaos_val >= 100:
                        self.overlay.place_star(cursor_x, cursor_y, "gold2", item_key=star_key, item_class=getattr(item, "item_class", "") or "")
                    elif chaos_val >= 25:
                        self.overlay.place_star(cursor_x, cursor_y, "gold1", item_key=star_key, item_class=getattr(item, "item_class", "") or "")
                    else:
                        self.overlay.place_star(cursor_x, cursor_y, "silver1", item_key=star_key, item_class=getattr(item, "item_class", "") or "")
                self.stats["successful_lookups"] += 1
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
                chanceable_text = f"{price_str} Chance \u2192 {unique_name}"
                logger.info(f"Chanceable base: {item.base_type} -> {unique_name} ({price_str})")
                self.overlay.show_price(
                    text=chanceable_text,
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
                        unid_text = f"{result['display']} unid"
                        logger.info(
                            f">>> PRICE [unid] {base}: {result['display']} "
                            f"({result['name']})"
                        )
                        self.overlay.show_price(
                            text=unid_text,
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
                    text="UNID", tier="low",
                    cursor_x=cursor_x, cursor_y=cursor_y,
                )
                return

            # Step 2: Non-unique items with mods → local scoring
            if (item.rarity in ("rare", "magic") and item.mods
                    and self.mod_parser.loaded):
                # Resolve magic item base_type if missing
                if not item.base_type and item.name:
                    resolved = self.mod_parser.resolve_base_type(item.name)
                    if resolved:
                        item.base_type = resolved

                parsed_mods = self.mod_parser.parse_mods(item)
                if not parsed_mods:
                    self._show_dismiss(item, cursor_x, cursor_y)
                    return

                # Local scoring (instant, no API calls)
                if self.mod_database.loaded:
                    self._score_and_display(item, parsed_mods, cursor_x, cursor_y,
                                            clipboard_text=item_text)
                    return

                # No mod database — can't score
                self._show_dismiss(item, cursor_x, cursor_y)
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
                self._show_dismiss(item, cursor_x, cursor_y)
                return

            # Step 3: Display the price
            elapsed = (time.time() - start_time) * 1000
            matched_name = result.get("name", item.name)
            static_text = result['display']
            logger.info(
                f">>> PRICE [{elapsed:.0f}ms] {matched_name}: "
                f"{result['display']}"
            )

            self.overlay.show_price(
                text=static_text,
                tier=result["tier"],
                cursor_x=cursor_x,
                cursor_y=cursor_y,
                price_divine=result.get("divine_value", 0),
            )
            # Star for valuable static-priced items (uniques, gems, etc.)
            divine_val = result.get("divine_value", 0)
            d2c = self.price_cache.divine_to_chaos or 1
            chaos_val = divine_val * d2c
            if chaos_val >= 5:
                star_key = matched_name or item.base_type or ""
                if chaos_val >= 500:
                    self.overlay.place_star(cursor_x, cursor_y, "gold3", item_key=star_key, item_class=getattr(item, "item_class", "") or "")
                elif chaos_val >= 100:
                    self.overlay.place_star(cursor_x, cursor_y, "gold2", item_key=star_key, item_class=getattr(item, "item_class", "") or "")
                elif chaos_val >= 25:
                    self.overlay.place_star(cursor_x, cursor_y, "gold1", item_key=star_key, item_class=getattr(item, "item_class", "") or "")
                else:
                    self.overlay.place_star(cursor_x, cursor_y, "silver1", item_key=star_key, item_class=getattr(item, "item_class", "") or "")
            self.stats["successful_lookups"] += 1

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)

    # ─── Local Scoring ────────────────────────────────

    def _score_and_display(self, item, parsed_mods, cursor_x, cursor_y,
                           clipboard_text=None):
        """Score item locally and display grade overlay."""
        from config import GRADE_TIER_MAP
        score = self.mod_database.score_item(item, parsed_mods)
        display_name = item.name or item.base_type

        overlay_tier = GRADE_TIER_MAP.get(score.grade.value, "low")

        d2c = self.price_cache.divine_to_chaos
        d2e = self.price_cache.divine_to_exalted
        ds = self._display_settings
        mode = ds.get("overlay_mode", "stars_only")
        flags = self._MODE_FLAGS.get(mode, self._MODE_FLAGS["stars_only"])
        text = score.format_overlay_text(
            price_estimate=None,
            estimate_low=None,
            estimate_high=None,
            confidence_tier=None,
            value_tier=None,
            divine_to_chaos=d2c,
            divine_to_exalted=d2e,
            show_grade=flags["show_grade"],
            show_price=flags["show_price"],
            show_stars=flags["show_stars"],
            show_mods=flags["show_mods"],
            show_dps=flags["show_dps"],
        )

        # Scrap override: JUNK/C items with quality/sockets
        if text == "SCRAP":
            overlay_tier = "scrap"

        # Log factors when they modify the score
        if score.dps_factor != 1.0:
            logger.info(f"DPS factor: {score.dps_factor:.2f} (dps={score.total_dps:.0f})")
        if score.defense_factor != 1.0:
            logger.info(f"Defense factor: {score.defense_factor:.2f} (def={score.total_defense})")
        if score.somv_factor != 1.0:
            logger.info(f"SOMV factor: {score.somv_factor:.3f} (roll quality)")

        logger.info(f"Grade {score.grade.value}: {display_name} "
                     f"(score={score.normalized_score:.3f}) "
                     f"{score.top_mods_summary}")

        # Show popup overlay (unless stars_only mode)
        if mode != "stars_only":
            is_borderless = (text == "\u2605")
            if text == "\u2717" and not ds.get("overlay_show_low_value", False):
                pass  # suppress junk overlay
            else:
                self.overlay.show_price(text=text, tier=overlay_tier,
                                        cursor_x=cursor_x, cursor_y=cursor_y,
                                        borderless=is_borderless)

        if score.grade.value not in ("C", "JUNK"):
            self.stats["successful_lookups"] += 1

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

    def _star_clear_hotkey_loop(self):
        """Poll for Ctrl+Shift+X to clear all star indicators."""
        import ctypes
        VK_SHIFT, VK_CONTROL, VK_X = 0x10, 0x11, 0x58
        _gaks = ctypes.windll.user32.GetAsyncKeyState
        was_pressed = False
        focus_lost_since = None

        while True:
            time.sleep(0.05)  # 20 Hz

            # Hotkey: Ctrl+Shift+X
            pressed = bool(_gaks(VK_CONTROL) & 0x8000
                           and _gaks(VK_SHIFT) & 0x8000
                           and _gaks(VK_X) & 0x8000)
            if pressed and not was_pressed:
                was_pressed = True
                logger.info("Star clear: Ctrl+Shift+X pressed")
                self.overlay.clear_stars()
            elif not pressed:
                was_pressed = False

            # Auto-clear: if POE2 loses focus for >30 seconds, clear stars
            if self.item_detector.game_window.is_poe2_foreground():
                focus_lost_since = None
            else:
                if focus_lost_since is None:
                    focus_lost_since = time.time()
                elif time.time() - focus_lost_since > 30:
                    logger.info("Star clear: POE2 lost focus for 30s")
                    self.overlay.clear_stars()
                    focus_lost_since = None  # Reset so we don't spam

    # Items that should always show ✗ (too cheap to bother pricing)
    def _show_dismiss(self, item, cursor_x, cursor_y):
        """Show dismiss (✗) or scrap hammer if the item has quality/sockets."""
        if self._display_settings.get("overlay_mode", "stars_only") == "stars_only":
            return  # no popup overlays in stars-only mode
        has_scrap = (
            getattr(item, "quality", 0) > 0 or
            getattr(item, "sockets", 0) > 0
        )
        if has_scrap:
            self.overlay.show_price(
                text="SCRAP", tier="scrap",
                cursor_x=cursor_x, cursor_y=cursor_y,
            )
        elif self._display_settings.get("overlay_show_low_value", False):
            self.overlay.show_price(
                text="\u2717", tier="low",
                cursor_x=cursor_x, cursor_y=cursor_y,
            )

    # Normal base types that can be chanced into valuable uniques.
    # Maps base_type (lowercase) → unique name for price lookup.
    _CHANCEABLE_BASES = {
        "heavy belt": "Headhunter",
        "tribal mask": "The Vertex",
    }

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

                d2c = cache_stats.get('divine_to_chaos', 0)
                d2e = cache_stats.get('divine_to_exalted', 0)
                m2d = cache_stats.get('mirror_to_divine', 0)

                logger.info(
                    f"[Status] Uptime: {uptime/60:.0f}min | "
                    f"Triggers: {total} | Prices shown: {hits} ({hit_rate:.0f}%) | "
                    f"Cache: {cache_stats['total_items']} items | "
                    f"Last refresh: {cache_stats['last_refresh']} | "
                    f"D2C: {d2c:.1f} | D2E: {d2e:.1f} | M2D: {m2d:.1f}"
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
    File gets DEBUG when --debug is used (full item text, captures, etc.).
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
        import setproctitle
        setproctitle.setproctitle("LAMA-overlay")
    except ImportError:
        pass

    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleCtrlHandler(None, True)
        # Resize console window — find by title for Windows Terminal compatibility
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, "LAMA")
        if hwnd:
            user32.MoveWindow(hwnd, 100, 100, 650, 500, True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="LAMA - Live Auction Market Assessor",
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
        app = LAMA(
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
