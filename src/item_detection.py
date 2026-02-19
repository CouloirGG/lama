"""
LAMA - Item Detection via Clipboard
Monitors cursor position, sends Ctrl+C when cursor stops over POE2,
and reads the resulting item data from clipboard.

Replaces the OCR pipeline (screen_capture + ocr_engine) with a single
clipboard read — same approach used by Awakened PoE Trade, ExileExchange2, etc.
"""

import time
import logging
from typing import Optional, Tuple, Callable

from config import (
    SCAN_FPS,
    DETECTION_COOLDOWN,
    CURSOR_STILL_RADIUS,
    CURSOR_STILL_FRAMES,
)
from screen_capture import CursorTracker, GameWindowDetector
from clipboard_reader import ClipboardReader

logger = logging.getLogger(__name__)


class ItemDetector:
    """
    Detects items under the cursor by sending Ctrl+C when the cursor stops.
    Same cursor-stop logic as ScreenCapture, but triggers a clipboard read
    instead of a screen capture + OCR.
    """

    def __init__(self):
        self.cursor = CursorTracker()
        self.game_window = GameWindowDetector()
        self.clipboard = ClipboardReader()

        self._prev_cursor_pos: Tuple[int, int] = (0, 0)
        self._last_trigger_time: float = 0
        self._cursor_still_count: int = 0

        # Position-based cooldown: don't re-fire at the same spot
        self._last_trigger_pos: Optional[Tuple[int, int]] = None

        # Content-based dedup: don't fire callback for the same item text
        self._last_item_text: str = ""
        self._last_item_time: float = 0
        self._DEDUP_TTL: float = 30.0  # seconds before same item can re-trigger

        # Callbacks
        self._on_change: Optional[Callable] = None
        self._on_hide: Optional[Callable] = None
        # When True, the overlay was hidden by cursor movement and the
        # next detection of the same item should re-fire the callback
        # instead of being blocked by content dedup.
        self._awaiting_reshow: bool = False
        self._reshow_origin_pos: Optional[Tuple[int, int]] = None

    def set_callback(self, callback: Callable):
        """
        Set callback for when item data is detected.
        Callback receives: (item_text: str, cursor_x: int, cursor_y: int)
        """
        self._on_change = callback

    def set_hide_callback(self, callback: Callable):
        """Set callback for when cursor moves away from a priced item."""
        self._on_hide = callback

    def _is_same_position(self, pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> bool:
        """Check if two positions are within CURSOR_STILL_RADIUS of each other."""
        dx = abs(pos_a[0] - pos_b[0])
        dy = abs(pos_a[1] - pos_b[1])
        return dx <= CURSOR_STILL_RADIUS and dy <= CURSOR_STILL_RADIUS

    def run_detection_loop(self):
        """
        Main detection loop. Call this in a dedicated thread.
        Sends Ctrl+C when cursor stops moving over POE2.
        """
        frame_interval = 1.0 / SCAN_FPS
        logger.info(
            f"Detection loop started ({SCAN_FPS} fps, "
            f"still threshold: {CURSOR_STILL_FRAMES} frames, "
            f"cooldown: {DETECTION_COOLDOWN}s)"
        )

        while True:
            loop_start = time.time()

            try:
                # Get cursor position
                cx, cy = self.cursor.get_position()

                # Only process when cursor is over the POE2 window
                if not self.game_window.is_cursor_over_poe2(cx, cy):
                    self._cursor_still_count = 0
                    self._prev_cursor_pos = (cx, cy)
                    continue

                # Track cursor stillness
                cursor_moved = not self._is_same_position((cx, cy), self._prev_cursor_pos)

                if cursor_moved:
                    self._cursor_still_count = 0
                    # Clear position cooldown when cursor moves away from last trigger spot
                    # (but keep _last_item_text for content dedup — the game can return
                    # stale item data from a previous tooltip, so we must prevent
                    # the same item text from re-triggering at every new position)
                    if self._last_trigger_pos and not self._is_same_position((cx, cy), self._last_trigger_pos):
                        # Save origin before clearing — needed for distance-guarded reshow
                        self._reshow_origin_pos = self._last_trigger_pos
                        self._last_trigger_pos = None
                        # Hide overlay when cursor leaves the item
                        if self._on_hide:
                            self._on_hide()
                            self._awaiting_reshow = True
                else:
                    self._cursor_still_count += 1

                self._prev_cursor_pos = (cx, cy)

                # Only trigger after cursor has been still long enough
                if self._cursor_still_count != CURSOR_STILL_FRAMES:
                    continue

                # Position-based cooldown: skip if we already triggered at this spot
                if self._last_trigger_pos and self._is_same_position((cx, cy), self._last_trigger_pos):
                    continue

                # Time-based cooldown
                now = time.time()
                if (now - self._last_trigger_time) < DETECTION_COOLDOWN:
                    continue

                # Only send Ctrl+C if POE2 is the focused window
                if not self.game_window.is_poe2_foreground():
                    logger.debug(f"POE2 not focused, skipping Ctrl+C")
                    continue

                # Send Ctrl+C and read clipboard
                logger.debug(f"Cursor stopped at ({cx}, {cy}), reading clipboard...")
                item_text = self.clipboard.copy_item_under_cursor()

                if item_text is None:
                    logger.debug("No item data from clipboard")
                    self._last_trigger_pos = (cx, cy)
                    continue

                # Content-based dedup: skip if same item text was just processed
                now_dedup = time.time()
                if (item_text == self._last_item_text
                        and (now_dedup - self._last_item_time) < self._DEDUP_TTL):
                    if self._awaiting_reshow:
                        if self._reshow_origin_pos and self._is_same_position(
                                (cx, cy), self._reshow_origin_pos):
                            # Genuine jitter — cursor returned to same item
                            logger.debug(f"Re-showing item at ({cx}, {cy})")
                            self._awaiting_reshow = False
                            self._last_trigger_pos = (cx, cy)
                            if self._on_change:
                                self._on_change(item_text, cx, cy)
                        else:
                            # Stale cached data at a different position — suppress
                            logger.debug(f"Suppressing stale reshow at ({cx}, {cy})")
                            self._last_trigger_pos = (cx, cy)
                            self._awaiting_reshow = False
                    else:
                        logger.debug(f"Skipping duplicate item at ({cx}, {cy})")
                        self._last_trigger_pos = (cx, cy)
                    continue

                logger.info(f"Clipboard item at ({cx}, {cy}): {item_text.split(chr(10))[0]}")

                # Fire callback with item text
                if self._on_change:
                    self._on_change(item_text, cx, cy)
                    self._last_trigger_time = now
                    self._last_trigger_pos = (cx, cy)
                    self._last_item_text = item_text
                    self._last_item_time = now_dedup
                    self._awaiting_reshow = False

            except Exception as e:
                logger.error(f"Detection loop error: {e}", exc_info=True)

            finally:
                # Always maintain target FPS — even when `continue` is used.
                elapsed = time.time() - loop_start
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)


# ─── Quick Test ──────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    detector = ItemDetector()

    def on_item(text, cx, cy):
        print(f"\n[ITEM at ({cx}, {cy})]")
        print(text)
        print()

    detector.set_callback(on_item)
    print("Starting detection loop... (Ctrl+C to stop)")
    print("Hover over items in POE2 to see clipboard data.")
    detector.run_detection_loop()
