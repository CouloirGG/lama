"""
LAMA - Overlay Window
Draws a transparent, always-on-top price tag near the cursor.
Uses tkinter for zero-dependency transparent window rendering.

On Windows, this creates a layered window that:
- Is always on top of other windows
- Is click-through (doesn't steal focus from POE2)
- Has a transparent background
- Shows a small price tag with color coding
"""

import re
import time
import logging
import threading
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# PIL for currency icon loading (optional — falls back to text-only)
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import tkinter as tk
    import tkinter.font as tkfont
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
    OVERLAY_THEME,
    OVERLAY_PULSE_STYLE,
    OVERLAY_REFERENCE_HEIGHT,
    PRICE_COLOR_HIGH,
    PRICE_COLOR_GOOD,
    PRICE_COLOR_DECENT,
    PRICE_COLOR_LOW,
    PRICE_COLOR_SCRAP,
)

# ─── Theme Constants ──────────────────────────────────
THEME_CLASSIC = "classic"
THEME_POE2 = "poe2"

# POE2 gothic theme color palette
_POE2_BG = "#1a120c"               # Darker aged parchment (more contrast)
_POE2_BORDER_NORMAL = "#3d2e1e"    # Dark weathered leather
_POE2_BORDER_ACCENT = "#6b5530"    # Tarnished brass accent lines
_POE2_CORNER_GOLD = "#c4a456"      # Bright gold corner diamonds
_POE2_TEXT_MUTED = "#7a6b55"       # Secondary text (stars, mods)
_POE2_BLOOD_DARK = "#3a0a08"       # Dark dried blood (splatters)
_POE2_BLOOD_MID = "#5c1510"        # Mid-tone blood (scratches)
_POE2_VIGNETTE = "#0d0804"         # Near-black edge darkening

# Grunge decoration positions (deterministic, pre-computed)
# Each: (x_frac, y_frac, w, h) — fractions of canvas width/height
_GRUNGE_SPLATTERS = [
    (0.08, 0.15, 3, 2),   # top-left area
    (0.85, 0.20, 2, 3),   # top-right area
    (0.12, 0.80, 2, 2),   # bottom-left area
    (0.90, 0.75, 3, 2),   # bottom-right area
    (0.45, 0.10, 2, 1),   # top-center nick
    (0.55, 0.88, 1, 2),   # bottom-center nick
]
# Scratch marks: (x1_frac, y1_frac, x2_frac, y2_frac)
_GRUNGE_SCRATCHES = [
    (0.15, 0.05, 0.22, 0.12),   # top-left scratch
    (0.78, 0.08, 0.88, 0.02),   # top-right scratch
    (0.10, 0.92, 0.18, 0.98),   # bottom-left scratch
    (0.82, 0.90, 0.92, 0.96),   # bottom-right scratch
]

# Sheen sweep animation constants
_SHEEN_WIDTH_FRAC = 0.28    # Band = 28% of overlay width
_SHEEN_SWEEP_MS = 1800      # Full left→right sweep duration
_SHEEN_FPS = 30             # ~33ms per frame
_SHEEN_FRAME_MS = 1000 // _SHEEN_FPS
_SHEEN_PAUSE_MS = 400       # Gap between sweeps

# Serif font fallback chain (Palatino Linotype ships with every Windows since XP)
_POE2_FONT_CHAIN = ("Palatino Linotype", "Book Antiqua", "Georgia", "Segoe UI")


class PriceOverlay:
    """
    Transparent overlay window that shows item prices.
    
    Architecture:
    - tkinter runs on the main thread (requirement on macOS, best practice elsewhere)
    - Other threads schedule UI updates via root.after()
    - The overlay window is always present but hidden when not showing a price
    """

    _NORMAL_BORDER_COLOR = "#333355"

    # Currency detection patterns for inline icon rendering
    _CURRENCY_SHORT_RE = re.compile(r'(?<=\d)(d|c)\b')
    _CURRENCY_LONG_RE = re.compile(r'\b(Divine|Chaos|Exalted|Mirror)s?\b', re.IGNORECASE)

    # Maps icon keys → resource filenames
    _CURRENCY_ICON_FILES = {
        "divine": "resources/img/divine_orb.png",
        "chaos": "resources/img/chaos_orb.png",
        "exalted": "resources/img/exalted_orb.png",
        "mirror": "resources/img/mirror_of_kalandra.png",
        "scrap": "resources/img/scrap_hammer.png",
    }

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

    # POE2 Gothic: blood/crimson border palette — darker, grittier
    _VALUE_TIERS_POE2 = [
        (5000, ("#8b1a1a", "#a85520", "#c4a456", "#5c1510", "#6b5a2e", "#7a2020"),
               150, 4, True),    # Mirror: blood rainbow, very fast
        (1000, ("#8b1a1a", "#a85520"), 250, 3, False),   # Crimson/burnt, fast
        (500,  ("#7a2020", "#a85520"), 350, 3, False),    # Blood/amber, fast
        (250,  ("#a85520", "#6b5a2e"), 450, 3, False),    # Burnt orange/bronze
        (100,  ("#6b5a2e", "#5a4a30"), 600, 2, False),    # Tarnished bronze
        (50,   ("#5a4a30",),           0,   2, False),    # Aged bronze, static
        (25,   ("#4a3a28",),           0,   2, False),    # Dark leather, static
    ]

    # Fallback for estimates below 25 divine
    _ESTIMATE_BORDER_COLORS = ("#FFD700", "#B8860B")
    _ESTIMATE_BORDER_COLORS_POE2 = ("#a85520", "#6b5a2e")
    _ESTIMATE_PULSE_MS = 600

    # Tier ID to divine threshold mapping (matches dashboard tier definitions)
    _TIER_ID_MAP = {
        "mirror": 5000, "jackpot": 1000, "big_hit": 500,
        "great_find": 250, "good_find": 100, "worth_sell": 50,
        "marginal": 25, "vendor": 0,
    }

    def __init__(self, theme: str = OVERLAY_THEME, pulse_style: str = OVERLAY_PULSE_STYLE,
                 scale_factor: float = 1.0):
        self._theme = theme if theme in (THEME_CLASSIC, THEME_POE2) else THEME_POE2
        self._pulse_style = pulse_style if pulse_style in ("border", "sheen", "both", "none") else "sheen"
        # Resolution scale factor (1.0 = 1080p baseline)
        self._scale = max(0.6, min(1.5, scale_factor))
        self._font_size = max(9, round(OVERLAY_FONT_SIZE * self._scale))
        self._padding = max(4, round(OVERLAY_PADDING * self._scale))
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
        self._transparent_color = "#010101"  # Nearly black, used as transparency key
        # Custom tier styles from dashboard settings (keyed by threshold)
        self._custom_text_colors: dict = {}
        self._custom_bg_colors: dict = {}
        self._custom_border_colors: dict = {}
        # Currency icon labels (created in initialize, packed on demand)
        self._icon_label: Optional[tk.Label] = None
        self._suffix_label: Optional[tk.Label] = None
        self._currency_icons: dict = {}  # key → ImageTk.PhotoImage (must persist)
        # POE2 Canvas theme widgets (created in _initialize_poe2)
        self._canvas: Optional[tk.Canvas] = None
        self._cv_items: dict = {}  # tag → canvas item ID
        self._poe2_font: Optional[tkfont.Font] = None
        self._poe2_font_small: Optional[tkfont.Font] = None
        # Sheen sweep animation state
        self._sheen_timer: Optional[str] = None
        self._sheen_x: float = 0
        self._sheen_dir: int = 1          # 1 = right, -1 = left (ping-pong)
        self._sheen_strips: list = []     # Canvas item IDs for the 3 sheen strips
        self._sheen_color: str = "#c4a456"
        self._sheen_active: bool = False

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
        self._root.configure(bg=self._transparent_color)
        try:
            self._root.attributes("-transparentcolor", self._transparent_color)
        except tk.TclError:
            # Not supported on all platforms
            pass

        # Theme-specific widget construction
        if self._theme == THEME_POE2:
            self._initialize_poe2()
        else:
            self._initialize_classic()

        # Pre-load currency icons
        self._load_currency_icons()

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
        logger.info(f"Overlay window initialized (theme={self._theme}, scale={self._scale:.2f})")

    def _initialize_classic(self):
        """Set up classic Frame+Label widgets (original UI)."""
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
            font=("Segoe UI", self._font_size, "bold"),
            fg=PRICE_COLOR_GOOD,
            bg=OVERLAY_BG_COLOR,
            padx=self._padding,
            pady=self._padding // 2,
            relief="flat",
            borderwidth=0,
        )
        self._label.pack(side="left")

        # Currency icon label (packed between prefix and suffix on demand)
        self._icon_label = tk.Label(
            self._frame,
            bg=OVERLAY_BG_COLOR,
            padx=1,
            pady=0,
            borderwidth=0,
        )
        # Suffix label (text after the currency icon)
        self._suffix_label = tk.Label(
            self._frame,
            text="",
            font=("Segoe UI", self._font_size, "bold"),
            fg=PRICE_COLOR_GOOD,
            bg=OVERLAY_BG_COLOR,
            padx=self._padding,
            pady=self._padding // 2,
            relief="flat",
            borderwidth=0,
        )
        # Neither icon_label nor suffix_label packed initially

    def _initialize_poe2(self):
        """Set up POE2 gothic Canvas widget with serif font."""
        # Resolve serif font from fallback chain
        available = set()
        try:
            available = set(tkfont.families())
        except Exception:
            pass
        chosen_family = _POE2_FONT_CHAIN[-1]  # fallback
        for family in _POE2_FONT_CHAIN:
            if family in available:
                chosen_family = family
                break
        logger.info(f"POE2 theme font: {chosen_family}")

        self._poe2_font = tkfont.Font(
            family=chosen_family, size=self._font_size + 2, weight="bold")
        self._poe2_font_small = tkfont.Font(
            family=chosen_family, size=max(9, self._font_size - 2))

        # Canvas is the single child of root; bg = transparent color key
        self._canvas = tk.Canvas(
            self._root,
            bg=self._transparent_color,
            highlightthickness=0,
            borderwidth=0,
        )
        self._canvas.pack()

        # Also create classic widgets (needed for _parse_currency icon rendering)
        # They won't be packed in POE2 mode, but _load_currency_icons references them
        self._frame = tk.Frame(self._root)  # hidden, never packed
        self._icon_label = tk.Label(self._frame)
        self._suffix_label = tk.Label(self._frame)
        self._label = tk.Label(self._frame)

    def load_custom_styles(self, overlay_tier_styles: dict):
        """Apply custom tier colors from dashboard settings.

        Args:
            overlay_tier_styles: dict mapping tier IDs (e.g. "mirror", "jackpot")
                to style dicts with keys: text_color, border_color, bg_color.
        """
        self._custom_text_colors.clear()
        self._custom_bg_colors.clear()
        self._custom_border_colors.clear()
        if not overlay_tier_styles:
            return
        for tier_id, style in overlay_tier_styles.items():
            threshold = self._TIER_ID_MAP.get(tier_id)
            if threshold is None:
                continue
            if style.get("text_color"):
                self._custom_text_colors[threshold] = style["text_color"]
            if style.get("bg_color"):
                self._custom_bg_colors[threshold] = style["bg_color"]
            if style.get("border_color"):
                self._custom_border_colors[threshold] = style["border_color"]
        if self._custom_text_colors or self._custom_bg_colors or self._custom_border_colors:
            logger.info(f"Loaded custom overlay styles for {len(overlay_tier_styles)} tier(s)")

    def show_price(self, text: str, tier: str, cursor_x: int, cursor_y: int,
                   estimate: bool = False, price_divine: float = 0,
                   borderless: bool = False):
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
            borderless: If True, render without border/background (floating icon)
        """
        with self._lock:
            self._pending_updates.append(
                ("show", text, tier, cursor_x, cursor_y, estimate, price_divine,
                 borderless))

    def reshow(self, cursor_x: int, cursor_y: int):
        """Reposition and re-display the overlay without re-rendering.

        Used when the user re-hovers the same item — avoids the cost of
        re-running the full pricing pipeline and prevents "Checking..."
        flashes that replace an already-displayed result.
        Thread-safe.
        """
        with self._lock:
            self._pending_updates.append(("reshow", cursor_x, cursor_y))

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

    @staticmethod
    def _get_screen_size() -> Tuple[int, int]:
        """Get screen size in physical pixels via Win32 API.

        Uses GetSystemMetrics which returns the same physical-pixel coordinate
        space as GetCursorPos, avoiding DPI mismatch with tkinter's
        winfo_screenwidth/height on HiDPI displays.
        """
        try:
            import ctypes
            SM_CXSCREEN, SM_CYSCREEN = 0, 1
            w = ctypes.windll.user32.GetSystemMetrics(SM_CXSCREEN)
            h = ctypes.windll.user32.GetSystemMetrics(SM_CYSCREEN)
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass
        # Fallback (non-Windows or error)
        return 1920, 1080

    def _load_currency_icons(self):
        """Pre-load currency icon PNGs, resized to match font size."""
        if not PIL_AVAILABLE:
            logger.debug("PIL not available — currency icons disabled")
            return

        from bundle_paths import get_resource

        icon_size = self._font_size + 4  # ~18px at default font size
        scrap_size = self._font_size * 3  # Larger — standalone overlay icon (42px)
        for key, rel_path in self._CURRENCY_ICON_FILES.items():
            try:
                img_path = get_resource(rel_path)
                if not img_path.exists():
                    continue
                img = Image.open(img_path).convert("RGBA")
                size = scrap_size if key == "scrap" else icon_size
                img = img.resize((size, size), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._currency_icons[key] = photo
            except Exception as e:
                logger.debug(f"Failed to load currency icon {key}: {e}")

        if self._currency_icons:
            logger.info(f"Loaded {len(self._currency_icons)} currency icon(s)")

    def _parse_currency(self, text: str):
        """Find a currency token in text and return (prefix, icon_key, suffix) or None."""
        # Short forms: "~130d" → ("~130", "divine", " ★3: ..."),  "~45c" → chaos
        m = self._CURRENCY_SHORT_RE.search(text)
        if m:
            key = "divine" if m.group(1) == "d" else "chaos"
            if key in self._currency_icons:
                return text[:m.start()], key, text[m.end():]

        # Long forms: "2-5 Divine", "100 Chaos", "2 Exalted", "1 Mirror"
        m = self._CURRENCY_LONG_RE.search(text)
        if m:
            word = m.group(1).lower()
            if word in self._currency_icons:
                return text[:m.start()], word, text[m.end():]

        return None

    def _render_with_icon(self, prefix: str, icon_key: str, suffix: str,
                          color: str, bg_color: str):
        """Render as [prefix text][icon][suffix text] labels packed left-to-right."""
        # Unpack all to ensure correct ordering
        self._label.pack_forget()
        self._icon_label.pack_forget()
        self._suffix_label.pack_forget()

        self._label.configure(text=f" {prefix}", fg=color, bg=bg_color,
                              padx=(OVERLAY_PADDING, 2))
        self._label.pack(side="left")

        icon = self._currency_icons.get(icon_key)
        if icon:
            self._icon_label.configure(image=icon, bg=bg_color)
            self._icon_label.pack(side="left")

        if suffix and suffix.strip():
            self._suffix_label.configure(text=f"{suffix} ", fg=color, bg=bg_color,
                                         padx=(2, OVERLAY_PADDING))
            self._suffix_label.pack(side="left")

    def _render_text_only(self, text: str, color: str, bg_color: str):
        """Render as a single text label (original behavior)."""
        self._icon_label.pack_forget()
        self._suffix_label.pack_forget()
        self._label.pack_forget()

        self._label.configure(text=f" {text} ", fg=color, bg=bg_color,
                              padx=OVERLAY_PADDING)
        self._label.pack(side="left")

    # ─── POE2 Canvas Rendering ─────────────────────────

    def _poe2_render(self, text: str, color: str, bg_color: str,
                     border_color: str, border_width: int, estimate: bool):
        """Draw the POE2 gothic overlay on the Canvas widget.

        Layout:  ◆──[ [A]  ~130d  ★★★ ]──◆
        - Grade badge (small colored rect + letter) on the left
        - Price text (large serif) center
        - Currency icon inline after price
        - Secondary text (stars, mods) in muted color
        - Corner diamonds at all four corners
        - Decorative rule lines near top/bottom edges
        """
        c = self._canvas
        c.delete("all")
        self._cv_items.clear()
        # Reset canvas size to prevent old geometry from flashing when
        # shrinking (e.g. full overlay with pips → small ✗ tag)
        c.configure(width=1, height=1)

        pad_x = round(12 * self._scale)
        pad_y = round(8 * self._scale)
        diamond_size = max(2, round(4 * self._scale))
        rule_inset = round(6 * self._scale)

        # Measure text to compute canvas size
        font = self._poe2_font
        font_sm = self._poe2_font_small
        line_h = font.metrics("linespace")

        # Parse text into segments: grade badge, price, secondary
        # Expected formats: "[A] ~130d ★★★", "~130d ★★★", "SCRAP", "✗"
        grade_letter = ""
        grade_color = color
        price_part = text
        secondary_part = ""

        # Extract grade badge: "[A]", "[S]", etc.
        grade_match = re.match(r'^\[([A-Z]+)\]\s*', text)
        if grade_match:
            grade_letter = grade_match.group(1)
            price_part = text[grade_match.end():]

        # Split secondary info (stars, mod summary) after price
        # Stars appear as ★★★ or ★3 etc.
        star_count = 0
        star_match = re.search(r'\s+(★.*)$', price_part)
        if star_match:
            secondary_part = star_match.group(1)
            price_part = price_part[:star_match.start()]

        # Extract star count from secondary_part for pip rendering
        star_only = re.search(r'★+', secondary_part)
        if star_only:
            star_count = len(star_only.group())
            # Remove stars from secondary_part (keep any non-star text)
            secondary_part = secondary_part[:star_only.start()] + \
                secondary_part[star_only.end():]
            secondary_part = secondary_part.strip()

        # Check for JUNK/C/SCRAP — plain small tag, no ornate frame
        is_plain = text in ("SCRAP", "\u2717") or (
            grade_letter in ("C", "JUNK") and not estimate)

        if is_plain:
            # SCRAP with hammer icon — show icon-only tag
            scrap_icon = self._currency_icons.get("scrap") if text == "SCRAP" else None
            if scrap_icon:
                icon_size = scrap_icon.width()
                w = icon_size + pad_x * 2
                h = icon_size + pad_y * 2

                c.configure(width=w, height=h)
                c.create_rectangle(0, 0, w, h, fill=_POE2_BG,
                                   outline=_POE2_BORDER_NORMAL, width=1,
                                   tags="bg")
                c.create_image(w // 2, h // 2, image=scrap_icon,
                               anchor="center", tags="scrap_icon")
                self._sheen_strips = []
                return

            # Simple small tag — no ornate decorations
            tw = font_sm.measure(text) + 4
            th = font_sm.metrics("linespace")
            w = tw + pad_x * 2
            h = th + pad_y * 2

            c.configure(width=w, height=h)
            # Background
            c.create_rectangle(0, 0, w, h, fill=_POE2_BG,
                               outline=_POE2_BORDER_NORMAL, width=1,
                               tags="bg")
            c.create_text(w // 2, h // 2, text=text, fill=color,
                          font=font_sm, anchor="center", tags="price")
            self._sheen_strips = []
            return

        # ── Compute layout widths ───────────────────────
        x_cursor = pad_x

        # Grade badge
        badge_w = 0
        if grade_letter:
            badge_w = font_sm.measure(grade_letter) + round(12 * self._scale)  # rect padding
            x_cursor += badge_w + 6

        # Price text
        price_w = font.measure(price_part) + 4
        x_cursor += price_w

        # Currency icon (if applicable)
        icon_key = None
        icon_w = 0
        currency = self._parse_currency(price_part) if self._currency_icons else None
        if currency:
            _, icon_key, _ = currency
            # Recalculate: price_part is just the prefix before currency
            price_part = currency[0]
            price_w = font.measure(price_part) + 4
            icon_w = self._font_size + 6  # icon size + gap
            if currency[2] and currency[2].strip():
                secondary_part = currency[2].strip() + (
                    ("  " + secondary_part) if secondary_part else "")

        # Secondary text
        secondary_w = 0
        if secondary_part:
            secondary_w = font_sm.measure(secondary_part) + 8

        total_content_w = (badge_w + 6 if badge_w else 0) + price_w + icon_w + secondary_w
        w = total_content_w + pad_x * 2
        h = line_h + pad_y * 2

        # Minimum width for aesthetics
        w = max(w, max(round(80 * self._scale), 60))

        c.configure(width=w, height=h)

        # ── Draw layers (back to front) ─────────────────

        # 1. Background fill
        c.create_rectangle(1, 1, w - 1, h - 1, fill=bg_color,
                           outline="", tags="bg_fill")

        # 1b. Edge vignette — dark strips along each edge for depth
        vig_w = max(3, w // 12)
        c.create_rectangle(1, 1, vig_w, h - 1, fill=_POE2_VIGNETTE,
                           outline="", stipple="gray25", tags="vignette")
        c.create_rectangle(w - vig_w, 1, w - 1, h - 1, fill=_POE2_VIGNETTE,
                           outline="", stipple="gray25", tags="vignette")

        # 1c. Blood splatters — small dark ovals for worn/gritty look
        for xf, yf, sw, sh in _GRUNGE_SPLATTERS:
            sx = int(xf * w)
            sy = int(yf * h)
            c.create_oval(sx, sy, sx + sw, sy + sh,
                          fill=_POE2_BLOOD_DARK, outline="", tags="grunge")

        # 1d. Scratch marks — short angled lines
        for x1f, y1f, x2f, y2f in _GRUNGE_SCRATCHES:
            c.create_line(int(x1f * w), int(y1f * h),
                          int(x2f * w), int(y2f * h),
                          fill=_POE2_BLOOD_MID, width=1, tags="grunge")

        # 1e. Sheen sweep strips — 3 translucent bands (leading, center, trailing)
        # Placed here so they render under the border and text layers
        sheen_band = max(4, int(w * _SHEEN_WIDTH_FRAC))
        strip_w = sheen_band // 3
        off = -sheen_band - 10  # start off-screen left
        self._sheen_strips = [
            c.create_rectangle(off, 2, off + strip_w, h - 2,
                               fill="#ffffff", outline="", stipple="gray25",
                               tags="sheen"),
            c.create_rectangle(off + strip_w, 2, off + strip_w * 2, h - 2,
                               fill="#ffffff", outline="", stipple="gray50",
                               tags="sheen"),
            c.create_rectangle(off + strip_w * 2, 2, off + sheen_band, h - 2,
                               fill="#ffffff", outline="", stipple="gray25",
                               tags="sheen"),
        ]

        # 2. Border — double-line effect: outer dark, inner accent
        # Outer border
        self._cv_items["border"] = c.create_rectangle(
            1, 1, w - 1, h - 1,
            outline=border_color, width=border_width, fill="",
            tags="border")
        # Inner accent line (inset 1px from outer)
        if border_width >= 2:
            c.create_rectangle(
                border_width + 1, border_width + 1,
                w - border_width - 1, h - border_width - 1,
                outline=_POE2_BORDER_ACCENT, width=1, fill="",
                tags="border_inner")

        # 3. Decorative rule lines (thin accent near top/bottom)
        c.create_line(rule_inset, 3, w - rule_inset, 3,
                      fill=_POE2_BORDER_ACCENT, width=1, tags="rule")
        c.create_line(rule_inset, h - 3, w - rule_inset, h - 3,
                      fill=_POE2_BORDER_ACCENT, width=1, tags="rule")

        # 4. Corner diamonds
        self._draw_corner_diamond(c, diamond_size + 1, diamond_size + 1,
                                  diamond_size)
        self._draw_corner_diamond(c, w - diamond_size - 1, diamond_size + 1,
                                  diamond_size)
        self._draw_corner_diamond(c, diamond_size + 1, h - diamond_size - 1,
                                  diamond_size)
        self._draw_corner_diamond(c, w - diamond_size - 1, h - diamond_size - 1,
                                  diamond_size)

        # 5. Grade badge (small colored rect with letter)
        x = pad_x
        cy = h // 2
        if grade_letter:
            badge_h = font_sm.metrics("linespace") + 4
            grade_bg = {
                "S": "#8b2500", "A": "#6b4c00",
                "B": "#2e4a3a", "JUNK": "#3a2a2a",
            }.get(grade_letter, "#333333")
            c.create_rectangle(x, cy - badge_h // 2,
                               x + badge_w, cy + badge_h // 2,
                               fill=grade_bg, outline=_POE2_BORDER_ACCENT,
                               width=1, tags="badge")
            c.create_text(x + badge_w // 2, cy, text=grade_letter,
                          fill=color, font=font_sm, anchor="center",
                          tags="badge_text")
            x += badge_w + 6

        # 6. Price text
        self._cv_items["price_text"] = c.create_text(
            x, cy, text=price_part, fill=color, font=font,
            anchor="w", tags="price")
        x += price_w

        # 7. Currency icon
        if icon_key and icon_key in self._currency_icons:
            icon = self._currency_icons[icon_key]
            c.create_image(x + 2, cy, image=icon, anchor="w", tags="icon")
            x += icon_w

        # 8. Secondary text (non-star remainder, e.g. currency suffix)
        if secondary_part:
            c.create_text(x + 4, cy, text=secondary_part,
                          fill=_POE2_TEXT_MUTED, font=font_sm,
                          anchor="w", tags="secondary")

        # 9. Star pips — small gold diamonds sitting on the bottom edge
        if star_count > 0:
            pip_size = max(1, round(2 * self._scale))
            pip_gap = max(2, round(4 * self._scale))
            total_pip_w = star_count * (pip_size * 2) + \
                (star_count - 1) * pip_gap
            pip_x = (w - total_pip_w) // 2 + pip_size
            pip_y = h - 1  # straddle the bottom border
            for _ in range(star_count):
                self._draw_corner_diamond(c, pip_x, pip_y, pip_size)
                pip_x += pip_size * 2 + pip_gap

    @staticmethod
    def _draw_corner_diamond(canvas, cx, cy, size=4):
        """Draw a small gold diamond polygon at (cx, cy)."""
        canvas.create_polygon(
            cx, cy - size,
            cx + size, cy,
            cx, cy + size,
            cx - size, cy,
            fill=_POE2_CORNER_GOLD, outline="", tags="diamond"
        )

    def _process_pending(self):
        """Process pending UI updates from other threads.

        Only the final show or hide is executed — intermediate updates are
        discarded to prevent flicker and stale-artifact rendering when
        multiple detections queue up within the same 50ms polling window.
        """
        with self._lock:
            updates = self._pending_updates[:]
            self._pending_updates.clear()

        if updates:
            # Find the last show/reshow and last hide in the batch
            last_show = None
            last_reshow = None
            last_show_idx = -1
            last_hide_idx = -1
            for i, update in enumerate(updates):
                if update[0] == "show":
                    last_show = update
                    last_show_idx = i
                elif update[0] == "reshow":
                    last_reshow = update
                    last_show_idx = i  # reshow counts as a show for ordering
                elif update[0] == "hide":
                    last_hide_idx = i

            # Execute only the final action
            try:
                if last_show_idx > last_hide_idx:
                    # Prefer full show over reshow when both are pending
                    if last_show and (not last_reshow or
                            updates.index(last_show) > updates.index(last_reshow)):
                        _, text, tier, cx, cy, estimate, price_divine, borderless = last_show
                        self._do_show(text, tier, cx, cy, estimate, price_divine,
                                      borderless)
                    elif last_reshow:
                        _, cx, cy = last_reshow
                        self._do_reshow(cx, cy)
                elif last_hide_idx >= 0:
                    self._do_hide()
            except Exception as e:
                logger.error(f"Overlay update error: {e}")

        # Schedule next check
        if self._root:
            self._root.after(50, self._process_pending)

    def _do_show(self, text: str, tier: str, cursor_x: int, cursor_y: int,
                 estimate: bool = False, price_divine: float = 0,
                 borderless: bool = False):
        """Actually show the price tag (must be on main thread)."""
        if not self._root:
            return
        # Classic needs _label; POE2 needs _canvas
        if self._theme == THEME_CLASSIC and not self._label:
            return
        if self._theme == THEME_POE2 and not self._canvas:
            return

        # Make fully transparent immediately so DWM composites nothing at
        # the old position.  withdraw() alone is unreliable on some Windows
        # setups for WS_EX_LAYERED + transparentcolor windows — DWM can
        # leave "ghost" pixels at the old position.  Setting alpha=0 is a
        # direct instruction to the compositor and always works.
        self._root.attributes('-alpha', 0)
        self._root.update_idletasks()

        # Set text color based on tier
        color = {
            "high": PRICE_COLOR_HIGH,
            "good": PRICE_COLOR_GOOD,
            "decent": PRICE_COLOR_DECENT,
            "low": PRICE_COLOR_LOW,
            "scrap": PRICE_COLOR_SCRAP,
        }.get(tier, PRICE_COLOR_LOW)

        # Stop any existing pulse/sheen animation
        if self._pulse_timer:
            self._root.after_cancel(self._pulse_timer)
            self._pulse_timer = None
        if self._sheen_timer:
            self._root.after_cancel(self._sheen_timer)
            self._sheen_timer = None
        self._sheen_active = False

        # ── Compute border effect (shared by both themes) ──
        bg_color = _POE2_BG if self._theme == THEME_POE2 else OVERLAY_BG_COLOR
        border_color = _POE2_BORDER_NORMAL if self._theme == THEME_POE2 else self._NORMAL_BORDER_COLOR
        border_colors = None
        pulse_ms = 0
        border_width = 2
        text_cycle = False
        matched_threshold = None

        # Select theme-appropriate tier table
        value_tiers = (self._VALUE_TIERS_POE2 if self._theme == THEME_POE2
                       else self._VALUE_TIERS)
        est_border = (self._ESTIMATE_BORDER_COLORS_POE2 if self._theme == THEME_POE2
                      else self._ESTIMATE_BORDER_COLORS)

        if borderless:
            bg_color = self._transparent_color
            if self._theme == THEME_CLASSIC:
                self._frame.configure(
                    bg=self._transparent_color, padx=0, pady=0)
            self._is_estimate = False
        else:
            for min_div, colors, pms, bw, tc in value_tiers:
                if price_divine >= min_div:
                    border_colors = colors
                    pulse_ms = pms
                    border_width = bw
                    text_cycle = tc
                    matched_threshold = min_div
                    break

            # Apply custom tier overrides from dashboard settings
            if matched_threshold is not None:
                custom_text = self._custom_text_colors.get(matched_threshold)
                if custom_text:
                    color = custom_text
                custom_bg = self._custom_bg_colors.get(matched_threshold)
                if custom_bg:
                    bg_color = custom_bg
                custom_border = self._custom_border_colors.get(matched_threshold)
                if custom_border:
                    border_colors = (custom_border,)
                    pulse_ms = 0  # static when custom

            # Fallback: estimates below value tiers still get pulse
            if not border_colors and estimate:
                border_colors = est_border
                pulse_ms = self._ESTIMATE_PULSE_MS

            # Apply border effect
            self._text_pulse = text_cycle
            if border_colors:
                self._pulse_colors = border_colors
                self._pulse_ms = pulse_ms
                self._pulse_index = 0
                border_color = border_colors[0]
                if pulse_ms > 0 and len(border_colors) > 1:
                    self._is_estimate = True
                else:
                    self._is_estimate = False
            else:
                self._is_estimate = False

        # ── Theme-specific rendering ───────────────────────
        # When sheen-only, suppress the colored border — keep default neutral
        if self._pulse_style == "sheen" and self._theme == THEME_POE2 and not borderless:
            sheen_border = border_color  # save tier color for the sheen tint
            border_color = _POE2_BORDER_NORMAL
            border_width = 2
        else:
            sheen_border = border_color

        if self._theme == THEME_POE2:
            self._poe2_render(text, color, bg_color, border_color,
                              border_width, estimate)
            if self._is_estimate:
                want_border = self._pulse_style in ("border", "both")
                want_sheen = (self._pulse_style in ("sheen", "both")
                              and self._theme == THEME_POE2)
                if want_border:
                    self._start_pulse()
                if want_sheen and self._sheen_strips:
                    self._sheen_color = sheen_border or _POE2_CORNER_GOLD
                    self._sheen_x = 3  # start at left inset
                    self._sheen_dir = 1
                    self._sheen_active = True
                    self._position_sheen_strips()
                    self._start_sheen()
        else:
            # Classic theme: apply border to frame
            if not borderless:
                self._frame.configure(padx=border_width, pady=border_width,
                                      bg=border_color)
            # Render label content (with or without currency icon)
            scrap_icon = self._currency_icons.get("scrap") if text == "SCRAP" else None
            if scrap_icon:
                # Show hammer icon instead of "SCRAP" text
                self._icon_label.pack_forget()
                self._suffix_label.pack_forget()
                self._label.pack_forget()
                self._icon_label.configure(image=scrap_icon, bg=bg_color)
                self._icon_label.pack(side="left")
            elif self._currency_icons:
                currency = self._parse_currency(text)
                if currency:
                    prefix, icon_key, suffix = currency
                    self._render_with_icon(prefix, icon_key, suffix, color, bg_color)
                else:
                    self._render_text_only(text, color, bg_color)
            else:
                self._render_text_only(text, color, bg_color)
            if self._is_estimate:
                # Classic theme: only border pulse (sheen is POE2-only)
                want_border = self._pulse_style in ("border", "both", "sheen")
                if want_border:
                    self._start_pulse()

        # Position near cursor (offsets are absolute, not scaled)
        x = cursor_x + OVERLAY_OFFSET_X
        y = cursor_y + OVERLAY_OFFSET_Y

        # Use Win32 screen metrics (same physical-pixel coordinate space as
        # GetCursorPos) instead of tkinter's winfo_screenwidth which returns
        # DPI-scaled logical pixels and causes misplacement on HiDPI displays.
        screen_w, screen_h = self._get_screen_size()
        self._root.update_idletasks()
        if self._theme == THEME_POE2:
            widget_w = self._canvas.winfo_reqwidth()
            widget_h = self._canvas.winfo_reqheight()
        else:
            widget_w = self._frame.winfo_reqwidth()
            widget_h = self._frame.winfo_reqheight()

        # Slide (don't flip) to keep overlay on-screen
        if x + widget_w > screen_w:
            x = screen_w - widget_w
        if x < 0:
            x = 0
        if y < 0:
            y = cursor_y + 30
        if y + widget_h > screen_h:
            y = screen_h - widget_h

        self._root.geometry(f"+{x}+{y}")

        # Ensure window is in shown state, then restore opacity.
        # The window was made alpha=0 at the top of _do_show.
        self._root.deiconify()
        self._root.lift()
        self._root.attributes("-topmost", True)
        self._root.attributes('-alpha', 0.92)

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
        pulse_color = self._pulse_colors[self._pulse_index]

        if self._theme == THEME_POE2 and self._canvas:
            # POE2: update canvas border outline
            border_id = self._cv_items.get("border")
            if border_id:
                self._canvas.itemconfigure(border_id, outline=pulse_color)
            # Mirror tier: cycle price text color too
            if self._text_pulse:
                n = len(self._pulse_colors)
                text_idx = (self._pulse_index + n // 2) % n
                pulse_fg = self._pulse_colors[text_idx]
                price_id = self._cv_items.get("price_text")
                if price_id:
                    self._canvas.itemconfigure(price_id, fill=pulse_fg)
        else:
            # Classic: update frame background
            self._frame.configure(bg=pulse_color)
            if self._text_pulse:
                n = len(self._pulse_colors)
                text_idx = (self._pulse_index + n // 2) % n
                pulse_fg = self._pulse_colors[text_idx]
                if self._label:
                    self._label.configure(fg=pulse_fg)
                if self._suffix_label and self._suffix_label.winfo_ismapped():
                    self._suffix_label.configure(fg=pulse_fg)
        self._pulse_timer = self._root.after(self._pulse_ms, self._start_pulse)

    def _start_sheen(self):
        """Ping-pong the sheen band back and forth inside the overlay."""
        if not self._root or not self._sheen_active or not self._visible:
            return
        if not self._sheen_strips or not self._canvas:
            return

        c = self._canvas
        w = c.winfo_reqwidth()
        sheen_band = max(4, int(w * _SHEEN_WIDTH_FRAC))
        inset = 3  # border inset
        step = (w - sheen_band - inset * 2) / (_SHEEN_SWEEP_MS / _SHEEN_FRAME_MS)

        self._sheen_x += step * self._sheen_dir

        # Bounce at edges
        max_x = w - sheen_band - inset
        if self._sheen_x >= max_x:
            self._sheen_x = max_x
            self._sheen_dir = -1
            self._position_sheen_strips()
            self._sheen_timer = self._root.after(
                _SHEEN_PAUSE_MS, self._start_sheen)
            return
        if self._sheen_x <= inset:
            self._sheen_x = inset
            self._sheen_dir = 1
            self._position_sheen_strips()
            self._sheen_timer = self._root.after(
                _SHEEN_PAUSE_MS, self._start_sheen)
            return

        self._position_sheen_strips()
        self._sheen_timer = self._root.after(
            _SHEEN_FRAME_MS, self._start_sheen)

    def _position_sheen_strips(self):
        """Move the 3 sheen strips to the current _sheen_x position, clamped to border."""
        if not self._canvas or len(self._sheen_strips) < 3:
            return
        c = self._canvas
        h = c.winfo_reqheight()
        w = c.winfo_reqwidth()
        sheen_band = max(4, int(w * _SHEEN_WIDTH_FRAC))
        strip_w = sheen_band // 3
        x = self._sheen_x
        inset = 3  # stay inside the border
        for i, sid in enumerate(self._sheen_strips):
            x0 = max(inset, x + strip_w * i)
            x1 = min(w - inset, x + strip_w * (i + 1))
            c.coords(sid, x0, inset, x1, h - inset)
            c.itemconfigure(sid, fill=_POE2_CORNER_GOLD)

    def _do_reshow(self, cursor_x: int, cursor_y: int):
        """Reposition and re-display the overlay without re-rendering.

        Reuses the existing canvas/label content — just moves the window.
        """
        if not self._root:
            return

        # Make transparent before repositioning (same ghost-prevention as _do_show)
        self._root.attributes('-alpha', 0)
        self._root.update_idletasks()

        # Position near cursor (same logic as _do_show)
        x = cursor_x + OVERLAY_OFFSET_X
        y = cursor_y + OVERLAY_OFFSET_Y

        screen_w, screen_h = self._get_screen_size()
        self._root.update_idletasks()
        if self._theme == THEME_POE2 and self._canvas:
            widget_w = self._canvas.winfo_reqwidth()
            widget_h = self._canvas.winfo_reqheight()
        elif self._frame:
            widget_w = self._frame.winfo_reqwidth()
            widget_h = self._frame.winfo_reqheight()
        else:
            return

        if x + widget_w > screen_w:
            x = screen_w - widget_w
        if x < 0:
            x = 0
        if y < 0:
            y = cursor_y + 30
        if y + widget_h > screen_h:
            y = screen_h - widget_h

        self._root.geometry(f"+{x}+{y}")
        self._root.deiconify()
        self._root.lift()
        self._root.attributes("-topmost", True)
        self._root.attributes('-alpha', 0.92)

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
            # Stop sheen animation
            if self._sheen_timer:
                self._root.after_cancel(self._sheen_timer)
                self._sheen_timer = None
            self._sheen_active = False
            self._is_estimate = False
            # Reset theme-specific state
            if self._theme == THEME_POE2 and self._canvas:
                border_id = self._cv_items.get("border")
                if border_id:
                    self._canvas.itemconfigure(border_id,
                                               outline=_POE2_BORDER_NORMAL)
            else:
                self._frame.configure(bg=self._NORMAL_BORDER_COLOR,
                                      padx=2, pady=2)
            # Alpha=0 first to guarantee no ghost pixels, then withdraw
            self._root.attributes('-alpha', 0)
            self._root.update_idletasks()
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
                   estimate: bool = False, price_divine: float = 0,
                   borderless: bool = False):
        tier_symbols = {"high": "$$", "good": ">>", "decent": "- ", "low": "  "}
        symbol = tier_symbols.get(tier, "  ")
        flag = " [est]" if estimate else ""
        try:
            print(f"  {symbol} {text}{flag}  (at {cursor_x}, {cursor_y})")
        except (UnicodeEncodeError, OSError):
            pass  # Windows cp1252 terminal can't encode some chars

    def reshow(self, cursor_x: int, cursor_y: int):
        pass

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
    import argparse as _ap

    logging.basicConfig(level=logging.DEBUG)

    _parser = _ap.ArgumentParser(description="Overlay visual test")
    _parser.add_argument("--theme", choices=["classic", "poe2"], default="poe2",
                         help="Theme to test (default: poe2)")
    _parser.add_argument("--pulse-style", choices=["border", "sheen", "both", "none"],
                         default="sheen", help="Pulse style to test (default: sheen)")
    _args = _parser.parse_args()

    def _get_cursor_pos():
        """Get current mouse cursor position via Win32 API (thread-safe)."""
        import ctypes

        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        pt = POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    if TK_AVAILABLE:
        overlay = PriceOverlay(theme=_args.theme, pulse_style=_args.pulse_style)

        def test_sequence():
            """Show test prices near the actual mouse cursor."""
            import time as t
            overlay._ready.wait()
            t.sleep(1)

            test_prices = [
                # (text, tier, estimate, price_divine)
                ("[S] ~130d ★★★", "high", True, 130),
                ("[A] ~45d ★★", "good", True, 45),
                ("[B] ~8d ★", "decent", False, 8),
                ("45 Chaos", "low", False, 0.5),
                ("SCRAP", "scrap", False, 0),
                ("\u2717", "low", False, 0),
                # High-value pulse tests
                ("[S] ~6000d ★★★", "high", True, 6000),  # Mirror rainbow
                ("[S] ~300d ★★★", "high", True, 300),     # Gold pulse
            ]

            for entry in test_prices:
                text, tier = entry[0], entry[1]
                estimate = entry[2] if len(entry) > 2 else False
                price_div = entry[3] if len(entry) > 3 else 0
                cx, cy = _get_cursor_pos()
                try:
                    print(f"Showing ({_args.theme}): {text!r} tier={tier} "
                          f"est={estimate} divine={price_div}")
                except UnicodeEncodeError:
                    safe = text.encode("ascii", errors="replace").decode()
                    print(f"Showing ({_args.theme}): {safe!r} tier={tier} "
                          f"est={estimate} divine={price_div}")
                overlay.show_price(text, tier, cx, cy,
                                   estimate=estimate,
                                   price_divine=price_div)
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
