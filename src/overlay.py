"""
POE2 Price Overlay - Overlay Window
Draws a transparent, always-on-top price tag near the cursor.
Uses tkinter for zero-dependency transparent window rendering.

On Windows, this creates a layered window that:
- Is always on top of other windows
- Is click-through (doesn't steal focus from POE2)
- Has a transparent background
- Shows a small price tag with color coding
"""

import time
import logging
import threading
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import tkinter as tk
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False
    logger.warning("tkinter not available - overlay disabled")

from config import (
    OVERLAY_OFFSET_X,
    OVERLAY_OFFSET_Y,
    OVERLAY_BG_COLOR,
    OVERLAY_FONT_SIZE,
    OVERLAY_PADDING,
    PRICE_COLOR_HIGH,
    PRICE_COLOR_GOOD,
    PRICE_COLOR_DECENT,
    PRICE_COLOR_LOW,
)


class PriceOverlay:
    """
    Transparent overlay window that shows item prices.
    
    Architecture:
    - tkinter runs on the main thread (requirement on macOS, best practice elsewhere)
    - Other threads schedule UI updates via root.after()
    - The overlay window is always present but hidden when not showing a price
    """

    _NORMAL_BORDER_COLOR = "#333355"

    # Value-based border effects:
    #   (min_divine, colors, pulse_ms, border_width, text_cycle)
    # pulse_ms=0 means static (no pulse). Checked highest-first.
    # text_cycle=True cycles label text color through the palette too.
    _VALUE_TIERS = [
        (5000, ("#FF4040", "#FF8C00", "#FFD700", "#40FF40", "#40CCFF", "#FF69B4"),
               150, 4, True),    # Mirror: rainbow, very fast, text cycles
        (1000, ("#FF4500", "#FFD700"), 250, 3, False),   # Red-orange/gold, fast
        (500,  ("#FF6600", "#FFD700"), 350, 3, False),    # Orange/gold, fast
        (250,  ("#FFD700", "#B8860B"), 450, 3, False),    # Gold, medium
        (100,  ("#FFD700", "#8B7536"), 600, 2, False),    # Gold, slow
        (50,   ("#DAA520",),           0,   2, False),    # Dark gold, static
        (25,   ("#A0A0B0",),           0,   2, False),    # Silver, static
    ]
    # Fallback for estimates below 25 divine
    _ESTIMATE_BORDER_COLORS = ("#FFD700", "#B8860B")
    _ESTIMATE_PULSE_MS = 600

    def __init__(self):
        self._root: Optional[tk.Tk] = None
        self._label: Optional[tk.Label] = None
        self._visible = False
        self._hide_timer: Optional[str] = None
        self._pulse_timer: Optional[str] = None
        self._pulse_index: int = 0
        self._pulse_colors: tuple = ()
        self._pulse_ms: int = 0
        self._text_pulse: bool = False
        self._is_estimate: bool = False
        self._ready = threading.Event()
        self._pending_updates = []
        self._lock = threading.Lock()
        self._hwnd: int = 0  # Top-level Win32 HWND for SetWindowPos calls

    def initialize(self):
        """
        Initialize the tkinter overlay window.
        MUST be called from the main thread.
        """
        if not TK_AVAILABLE:
            logger.error("Cannot initialize overlay: tkinter not available")
            return

        self._root = tk.Tk()

        # Window configuration
        self._root.title("POE2 Price")
        self._root.overrideredirect(True)          # No title bar, borders
        self._root.attributes("-topmost", True)     # Always on top
        self._root.attributes("-alpha", 0.92)       # Slight transparency

        # Transparent background (Windows-specific)
        transparent_color = "#010101"  # Nearly black, used as transparency key
        self._root.configure(bg=transparent_color)
        try:
            self._root.attributes("-transparentcolor", transparent_color)
        except tk.TclError:
            # Not supported on all platforms
            pass

        # Border frame provides an outline around the label
        # (2px so estimate pulse is clearly visible)
        self._frame = tk.Frame(
            self._root,
            bg=self._NORMAL_BORDER_COLOR,
            padx=2,
            pady=2,
        )
        self._frame.pack()

        # Create the price label inside the border frame
        self._label = tk.Label(
            self._frame,
            text="",
            font=("Segoe UI", OVERLAY_FONT_SIZE, "bold"),
            fg=PRICE_COLOR_GOOD,
            bg=OVERLAY_BG_COLOR,
            padx=OVERLAY_PADDING,
            pady=OVERLAY_PADDING // 2,
            relief="flat",
            borderwidth=0,
        )
        self._label.pack()

        # Force window realization so we get a valid HWND
        self._root.update_idletasks()

        # Make click-through on Windows (must be after update_idletasks)
        self._make_click_through()

        # Start hidden
        self._root.withdraw()
        self._visible = False

        # Process pending updates periodically
        self._root.after(50, self._process_pending)

        self._ready.set()
        logger.info("Overlay window initialized")

    def show_price(self, text: str, tier: str, cursor_x: int, cursor_y: int,
                   estimate: bool = False, price_divine: float = 0):
        """
        Show a price tag near the cursor position.
        Thread-safe - can be called from any thread.

        Args:
            text: Price text (e.g., "~8 Exalted")
            tier: Price tier ("high", "good", "decent", "low")
            cursor_x: Cursor X position on screen
            cursor_y: Cursor Y position on screen
            estimate: If True, price is a conservative estimate
            price_divine: Price in divine orbs (drives border effects)
        """
        with self._lock:
            self._pending_updates.append(
                ("show", text, tier, cursor_x, cursor_y, estimate, price_divine))

    def hide(self):
        """Hide the price overlay. Thread-safe."""
        with self._lock:
            self._pending_updates.append(("hide",))

    def run(self):
        """
        Run the tkinter main loop.
        MUST be called from the main thread. Blocks until shutdown.
        """
        if not self._root:
            self.initialize()
        if self._root:
            self._root.mainloop()

    def shutdown(self):
        """Shut down the overlay."""
        if self._root:
            try:
                self._root.quit()
                self._root.destroy()
            except Exception:
                pass

    # ─── Internal Methods ────────────────────────────

    def _process_pending(self):
        """Process pending UI updates from other threads."""
        with self._lock:
            updates = self._pending_updates[:]
            self._pending_updates.clear()

        for update in updates:
            try:
                if update[0] == "show":
                    _, text, tier, cx, cy, estimate, price_divine = update
                    self._do_show(text, tier, cx, cy, estimate, price_divine)
                elif update[0] == "hide":
                    self._do_hide()
            except Exception as e:
                logger.error(f"Overlay update error: {e}")

        # Schedule next check
        if self._root:
            self._root.after(50, self._process_pending)

    def _do_show(self, text: str, tier: str, cursor_x: int, cursor_y: int,
                 estimate: bool = False, price_divine: float = 0):
        """Actually show the price tag (must be on main thread)."""
        if not self._root or not self._label:
            return

        # Set text color based on tier
        color = {
            "high": PRICE_COLOR_HIGH,
            "good": PRICE_COLOR_GOOD,
            "decent": PRICE_COLOR_DECENT,
            "low": PRICE_COLOR_LOW,
        }.get(tier, PRICE_COLOR_LOW)

        # Update label
        self._label.configure(text=f" {text} ", fg=color)

        # Stop any existing pulse animation
        if self._pulse_timer:
            self._root.after_cancel(self._pulse_timer)
            self._pulse_timer = None

        # Determine border effect from divine price
        border_colors = None
        pulse_ms = 0
        border_width = 2
        text_cycle = False

        for min_div, colors, pms, bw, tc in self._VALUE_TIERS:
            if price_divine >= min_div:
                border_colors = colors
                pulse_ms = pms
                border_width = bw
                text_cycle = tc
                break

        # Fallback: estimates below value tiers still get gold pulse
        if not border_colors and estimate:
            border_colors = self._ESTIMATE_BORDER_COLORS
            pulse_ms = self._ESTIMATE_PULSE_MS

        # Apply border effect
        self._frame.configure(padx=border_width, pady=border_width)
        self._text_pulse = text_cycle
        if border_colors:
            self._pulse_colors = border_colors
            self._pulse_ms = pulse_ms
            self._pulse_index = 0
            self._frame.configure(bg=border_colors[0])
            if pulse_ms > 0 and len(border_colors) > 1:
                self._is_estimate = True  # reuse flag to keep pulse running
                self._start_pulse()
            else:
                self._is_estimate = False
        else:
            self._frame.configure(bg=self._NORMAL_BORDER_COLOR)
            self._is_estimate = False

        # Position near cursor
        x = cursor_x + OVERLAY_OFFSET_X
        y = cursor_y + OVERLAY_OFFSET_Y

        # Ensure it stays on screen
        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()
        self._root.update_idletasks()
        label_w = self._label.winfo_reqwidth()
        label_h = self._label.winfo_reqheight()

        if x + label_w > screen_w:
            x = cursor_x - label_w - 10
        if y < 0:
            y = cursor_y + 30
        if y + label_h > screen_h:
            y = screen_h - label_h

        self._root.geometry(f"+{x}+{y}")

        # Show the window
        self._root.deiconify()
        self._root.lift()
        self._root.attributes("-topmost", True)

        # Win32: force window above borderless fullscreen games
        # tkinter's -topmost alone isn't sufficient against game windows
        if self._hwnd:
            try:
                import ctypes
                HWND_TOPMOST = -1
                SWP_NOMOVE = 0x0002
                SWP_NOSIZE = 0x0001
                SWP_NOACTIVATE = 0x0010
                SWP_SHOWWINDOW = 0x0040
                ctypes.windll.user32.SetWindowPos(
                    self._hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                    SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
                )
            except Exception:
                pass

        self._visible = True

        # Cancel any leftover hide timer from a previous show
        if self._hide_timer:
            self._root.after_cancel(self._hide_timer)
            self._hide_timer = None

        # No auto-hide timer — the overlay stays visible until the cursor
        # moves off the item, which triggers overlay.hide() via the
        # ItemDetector's on_hide callback (item_detection.py:114-116).

    def _start_pulse(self):
        """Pulse the border between colors at the configured speed."""
        if not self._root or not self._is_estimate or not self._visible:
            return
        self._pulse_index = (self._pulse_index + 1) % len(self._pulse_colors)
        self._frame.configure(bg=self._pulse_colors[self._pulse_index])
        # Mirror tier: cycle text color too (offset by half for contrast)
        if self._text_pulse and self._label:
            n = len(self._pulse_colors)
            text_idx = (self._pulse_index + n // 2) % n
            self._label.configure(fg=self._pulse_colors[text_idx])
        self._pulse_timer = self._root.after(self._pulse_ms, self._start_pulse)

    def _do_hide(self):
        """Actually hide the overlay (must be on main thread)."""
        if self._root and self._visible:
            # Cancel auto-hide timer so it can't fire later and kill a
            # subsequent show_price (the root cause of the "flash and vanish" bug)
            if self._hide_timer:
                self._root.after_cancel(self._hide_timer)
                self._hide_timer = None
            # Stop pulse animation
            if self._pulse_timer:
                self._root.after_cancel(self._pulse_timer)
                self._pulse_timer = None
            self._is_estimate = False
            self._frame.configure(bg=self._NORMAL_BORDER_COLOR, padx=2, pady=2)
            self._root.withdraw()
            self._visible = False

    def _make_click_through(self):
        """
        Make the window click-through on Windows.
        Uses WS_EX_NOACTIVATE so the overlay never steals focus from POE2.
        WS_EX_TOOLWINDOW hides it from the taskbar/Alt-Tab.

        Note: WS_EX_TRANSPARENT is intentionally NOT used here — it can
        make the window completely invisible on some setups. NOACTIVATE
        is sufficient to prevent focus stealing.
        """
        try:
            import ctypes

            user32 = ctypes.windll.user32

            # winfo_id() returns the tkinter child widget HWND.
            # GetParent() gives the actual top-level Win32 window.
            # (wm_frame() returns '0x0' when overrideredirect is True)
            child_hwnd = self._root.winfo_id()
            hwnd = user32.GetParent(child_hwnd) or child_hwnd

            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_TOPMOST = 0x00000008
            WS_EX_TOOLWINDOW = 0x00000080      # Hide from taskbar
            WS_EX_NOACTIVATE = 0x08000000       # Never steal focus

            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style |= WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)

            # Force Windows to apply the style change immediately
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_NOZORDER = 0x0004
            SWP_FRAMECHANGED = 0x0020
            user32.SetWindowPos(
                hwnd, 0, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED,
            )

            self._hwnd = hwnd
            logger.info(f"Click-through enabled (hwnd=0x{hwnd:X})")
        except Exception as e:
            logger.warning(f"Click-through not available: {e}")


class ConsoleOverlay:
    """
    Fallback overlay that prints to console.
    Used for testing on non-Windows systems.
    """

    def __init__(self):
        self._ready = threading.Event()
        self._ready.set()

    def initialize(self):
        pass

    def show_price(self, text: str, tier: str, cursor_x: int, cursor_y: int,
                   estimate: bool = False, price_divine: float = 0):
        tier_symbols = {"high": "$$", "good": ">>", "decent": "- ", "low": "  "}
        symbol = tier_symbols.get(tier, "  ")
        flag = " [est]" if estimate else ""
        try:
            print(f"  {symbol} {text}{flag}  (at {cursor_x}, {cursor_y})")
        except (UnicodeEncodeError, OSError):
            pass  # Windows cp1252 terminal can't encode some chars

    def hide(self):
        pass

    def run(self):
        # Non-blocking for console mode
        self._ready.set()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    def shutdown(self):
        pass


# ─── Quick Test ──────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    def _get_cursor_pos():
        """Get current mouse cursor position via Win32 API (thread-safe)."""
        import ctypes

        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        pt = POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    if TK_AVAILABLE:
        overlay = PriceOverlay()

        def test_sequence():
            """Show test prices near the actual mouse cursor."""
            import time as t
            overlay._ready.wait()
            t.sleep(1)

            test_prices = [
                ("12 Divine", "high"),
                ("8 Exalted", "good"),
                ("1.5 Exalted", "decent"),
                ("45 Chaos", "low"),
            ]

            for text, tier in test_prices:
                cx, cy = _get_cursor_pos()
                print(f"Showing: {text} at ({cx}, {cy})")
                overlay.show_price(text, tier, cx, cy)
                t.sleep(3)

            overlay.shutdown()

        thread = threading.Thread(target=test_sequence, daemon=True)
        thread.start()
        overlay.run()
    else:
        print("tkinter not available, using console overlay")
        overlay = ConsoleOverlay()
        overlay.show_price("12 Divine", "high", 500, 400)
        overlay.show_price("8 Exalted", "good", 600, 300)
