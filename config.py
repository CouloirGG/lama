"""
POE2 Price Overlay - Configuration
All tunable constants in one place.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# POE2 Game Settings
# ─────────────────────────────────────────────
POE2_WINDOW_TITLE = "Path of Exile 2"
POE2_PROCESS_NAME = "PathOfExile2.exe"

# Default league - user can change this
DEFAULT_LEAGUE = "Fate of the Vaal"

# ─────────────────────────────────────────────
# Item Detection
# ─────────────────────────────────────────────
# How often to check cursor position (fps)
SCAN_FPS = 8  # 8 checks per second

# Cooldown after a successful detection (seconds)
DETECTION_COOLDOWN = 1.0

# Cursor must be within this many pixels for N frames before triggering
# This filters out camera panning (cursor moves = camera moves = everything changes)
CURSOR_STILL_RADIUS = 20  # pixels
CURSOR_STILL_FRAMES = 3   # must be still for 3 consecutive frames (~375ms)

# ─────────────────────────────────────────────
# Clipboard Reading (Ctrl+C)
# ─────────────────────────────────────────────
# Seconds to wait after sending Ctrl+C for the game to write to clipboard
CTRL_C_DELAY = 0.05

# ─────────────────────────────────────────────
# Price Data
# ─────────────────────────────────────────────
# poe2scout.com — pre-aggregated prices for uniques, currencies, gems, maps, etc.
POE2SCOUT_BASE_URL = "https://poe2scout.com/api"

# How often to refresh price data (seconds)
PRICE_REFRESH_INTERVAL = 900  # 15 minutes

# Local cache directory
CACHE_DIR = Path(os.path.expanduser("~")) / ".poe2-price-overlay" / "cache"

# ─────────────────────────────────────────────
# Overlay Display
# ─────────────────────────────────────────────
# Price tag offset from cursor (pixels)
OVERLAY_OFFSET_X = 20
OVERLAY_OFFSET_Y = -40

# Price tag styling
OVERLAY_BG_COLOR = "#1a1a2e"      # Dark background
OVERLAY_BG_ALPHA = 0.85            # Background opacity
OVERLAY_FONT_SIZE = 14
OVERLAY_PADDING = 8

# Price tier colors
PRICE_COLOR_HIGH = "#ff6b35"       # Orange - very valuable (>= 50 Exalted)
PRICE_COLOR_GOOD = "#ffd700"       # Gold - worth picking up (>= 5 Exalted)
PRICE_COLOR_DECENT = "#4ecdc4"     # Teal - decent value (>= 1 Exalted)
PRICE_COLOR_LOW = "#95a5a6"        # Grey - low value (< 1 Exalted)

# Price display thresholds (in Exalted Orb equivalent)
PRICE_TIER_HIGH = 50.0
PRICE_TIER_GOOD = 5.0
PRICE_TIER_DECENT = 1.0

# How long the price tag stays visible (seconds)
OVERLAY_DISPLAY_DURATION = 4.0

# ─────────────────────────────────────────────
# Trade API (for rare item pricing)
# ─────────────────────────────────────────────
TRADE_API_BASE = "https://www.pathofexile.com/api/trade2"
TRADE_STATS_URL = f"{TRADE_API_BASE}/data/stats"
TRADE_MAX_REQUESTS_PER_SECOND = 2
TRADE_RESULT_COUNT = 8
TRADE_CACHE_TTL = 300  # 5 minutes
TRADE_MOD_MIN_MULTIPLIER = 0.8  # 80% of actual value as min filter
TRADE_STATS_CACHE_FILE = CACHE_DIR / "trade_stats.json"

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE = Path(os.path.expanduser("~")) / ".poe2-price-overlay" / "overlay.log"
