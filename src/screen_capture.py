"""
LAMA - Cursor Tracking & Game Window Detection
Provides cursor position tracking and POE2 window detection for the
clipboard-based item detection pipeline.
"""

import time
import logging
import ctypes
from typing import Optional, Tuple

from config import POE2_WINDOW_TITLE

logger = logging.getLogger(__name__)


class CursorTracker:
    """
    Tracks cursor position using Windows API.
    Falls back to a fixed position for non-Windows testing.
    """

    def __init__(self):
        self._use_win32 = False
        try:
            # Try Windows API
            self._user32 = ctypes.windll.user32
            self._use_win32 = True
            logger.info("Using Win32 cursor tracking")
        except (AttributeError, OSError):
            logger.info("Win32 not available, using fallback cursor tracking")

    def get_position(self) -> Tuple[int, int]:
        """Return current cursor (x, y) position."""
        if self._use_win32:
            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
            pt = POINT()
            self._user32.GetCursorPos(ctypes.byref(pt))
            return (pt.x, pt.y)
        else:
            # Fallback for non-Windows (testing)
            return (960, 540)


class GameWindowDetector:
    """
    Detects if the cursor is over the POE2 window.
    Uses cursor-in-window-bounds instead of foreground check so the overlay
    works on multi-monitor setups where POE2 may not be the "focused" window.
    """

    def __init__(self):
        self._use_win32 = False
        self._poe2_rect: Optional[Tuple[int, int, int, int]] = None
        self._rect_update_time: float = 0
        self._RECT_CACHE_SECS = 2.0  # re-scan windows every 2s
        try:
            self._user32 = ctypes.windll.user32
            self._use_win32 = True
        except (AttributeError, OSError):
            pass

    def is_poe2_foreground(self) -> bool:
        """Check if POE2 is the foreground (focused) window."""
        if not self._use_win32:
            return True
        fg_hwnd = self._user32.GetForegroundWindow()
        length = self._user32.GetWindowTextLengthW(fg_hwnd)
        if length <= 0:
            return False
        buf = ctypes.create_unicode_buffer(length + 1)
        self._user32.GetWindowTextW(fg_hwnd, buf, length + 1)
        return buf.value == POE2_WINDOW_TITLE

    def is_cursor_over_poe2(self, cx: int, cy: int) -> bool:
        """Check if cursor position (cx, cy) is inside the POE2 window."""
        if not self._use_win32:
            return True  # Assume yes for testing

        rect = self._get_poe2_rect_cached()
        if rect is None:
            return False

        left, top, right, bottom = rect
        return left <= cx <= right and top <= cy <= bottom

    def _get_poe2_rect_cached(self) -> Optional[Tuple[int, int, int, int]]:
        """Get POE2 window rect, refreshing from OS every few seconds."""
        now = time.time()
        if now - self._rect_update_time > self._RECT_CACHE_SECS:
            self._poe2_rect = self._find_poe2_rect()
            self._rect_update_time = now
            if self._poe2_rect:
                logger.debug(f"POE2 window rect: {self._poe2_rect}")
        return self._poe2_rect

    def _find_poe2_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """Find POE2 window rectangle using FindWindowW (no callback needed)."""
        if not self._use_win32:
            return (0, 0, 1920, 1080)

        hwnd = self._user32.FindWindowW(None, POE2_WINDOW_TITLE)
        if not hwnd:
            return None

        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                        ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
        r = RECT()
        self._user32.GetWindowRect(hwnd, ctypes.byref(r))
        return (r.left, r.top, r.right, r.bottom)
