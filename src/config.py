"""
POE2 Price Overlay - Configuration
All tunable constants in one place.
"""

import os
from pathlib import Path

from bundle_paths import get_resource

# ─────────────────────────────────────────────
# Version
# ─────────────────────────────────────────────
_version_file = get_resource("resources/VERSION")
APP_VERSION = _version_file.read_text().strip() if _version_file.exists() else "dev"

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
OVERLAY_DISPLAY_DURATION = 2.0

# ─────────────────────────────────────────────
# Trade API (for rare item pricing)
# ─────────────────────────────────────────────
TRADE_API_BASE = "https://www.pathofexile.com/api/trade2"
TRADE_STATS_URL = f"{TRADE_API_BASE}/data/stats"
TRADE_MAX_REQUESTS_PER_SECOND = 1  # Conservative to avoid 60s trade API bans
TRADE_RESULT_COUNT = 8
TRADE_CACHE_TTL = 300  # 5 minutes
TRADE_MOD_MIN_MULTIPLIER = 0.8  # 80% of actual value as min filter
TRADE_STATS_CACHE_FILE = CACHE_DIR / "trade_stats.json"
TRADE_ITEMS_URL = f"{TRADE_API_BASE}/data/items"
TRADE_ITEMS_CACHE_FILE = CACHE_DIR / "trade_items.json"

# ─────────────────────────────────────────────
# Trade Watchlist
# ─────────────────────────────────────────────
WATCHLIST_DEFAULT_POLL_INTERVAL = 300   # 5 min between polls per query
WATCHLIST_MIN_REQUEST_INTERVAL = 2.0    # seconds between any watchlist API calls
WATCHLIST_FETCH_COUNT = 10              # listings to fetch per query
WATCHLIST_MAX_QUERIES = 6

# ─────────────────────────────────────────────
# Loot Filter Updater
# ─────────────────────────────────────────────
# POE2 game filter directory (OneDrive-synced Documents)
FILTER_OUTPUT_DIR = Path(os.path.expanduser("~")) / "OneDrive" / "Documents" / "My Games" / "Path of Exile 2"

# How often to re-tier the filter (seconds)
FILTER_UPDATE_INTERVAL = 86400  # 24 hours

# Tier thresholds in CHAOS values (converted to divine at runtime using
# the current divine:chaos exchange rate so tiers stay meaningful
# regardless of divine orb price fluctuations).
#
# Currency-style tiers (s/a/b/c/d/e):
FILTER_CURRENCY_CHAOS_THRESHOLDS = {
    "s": 25.0,     # >= 25 chaos (roughly 1 divine)
    "a": 5.0,      # >= 5 chaos
    "b": 2.0,      # >= 2 chaos
    "c": 1.0,      # >= 1 chaos
    "d": 1.0,      # same as c (d-tier styling still has colored fill, skip it)
    "e": 0.0,      # everything under 1c — text only, no fill
}

# Unique tiers (t1/t2/t3/hideable):
FILTER_UNIQUE_CHAOS_THRESHOLDS = {
    "t1": 25.0,    # >= 25 chaos (~1 divine)
    "t2": 3.0,     # >= 3 chaos
    "t3": 0.5,     # >= 0.5 chaos
    "hideable": 0.0,
}

# Fragment tiers (a/b/c):
FILTER_FRAGMENT_CHAOS_THRESHOLDS = {
    "a": 5.0,      # >= 5 chaos
    "b": 1.0,      # >= 1 chaos
    "c": 0.0,
}

# Timestamp file for tracking last update
FILTER_LAST_UPDATE_FILE = CACHE_DIR / "filter_last_update"

# Strictness presets: multiplied against chaos thresholds
# Lower = show more items, higher = show only top items
STRICTNESS_PRESETS = {
    "relaxed": 0.5,
    "normal": 1.0,
    "strict": 2.0,
    "very_strict": 4.0,
}

# ─────────────────────────────────────────────
# DPS & Defense Classification
# ─────────────────────────────────────────────
# Attack weapons where DPS drives value (staves/wands are caster, excluded)
DPS_ITEM_CLASSES = frozenset({
    "Bows", "Crossbows",
    "One Hand Swords", "Two Hand Swords",
    "One Hand Axes", "Two Hand Axes",
    "One Hand Maces", "Two Hand Maces",
    "Daggers", "Claws", "Flails", "Spears",
})

TWO_HAND_CLASSES = frozenset({
    "Bows", "Crossbows",
    "Two Hand Swords", "Two Hand Axes", "Two Hand Maces",
})

DEFENSE_ITEM_CLASSES = frozenset({
    "Body Armours", "Boots", "Gloves", "Helmets", "Shields",
    "Bucklers", "Foci",
})

# DPS thresholds: (terrible, low, decent, good) total DPS values
# Keyed by minimum ilvl bracket
DPS_BRACKETS_2H = {
    68: (150, 250, 400, 600),   # endgame
    0:  (30, 60, 120, 200),     # leveling
}
DPS_BRACKETS_1H = {
    68: (80, 150, 250, 400),    # endgame
    0:  (15, 35, 70, 120),      # leveling
}

# Defense thresholds per slot: (terrible, low, decent, good)
DEFENSE_THRESHOLDS = {
    "Body Armours": (200, 400, 700, 1000),
    "Shields":      (150, 300, 500, 750),
    "Helmets":      (100, 200, 350, 500),
    "Gloves":       (80, 160, 280, 400),
    "Boots":        (80, 160, 280, 400),
    "Bucklers":     (100, 200, 350, 500),
    "Foci":         (50, 100, 200, 350),
}

# Trade API filter multipliers (search for items with >= X% of this item's stats)
TRADE_DPS_FILTER_MULT = 0.75
TRADE_DEFENSE_FILTER_MULT = 0.70

# ─────────────────────────────────────────────
# RePoE Mod Database (local scoring engine)
# ─────────────────────────────────────────────
REPOE_BASE_URL = "https://repoe-fork.github.io/poe2"
REPOE_CACHE_DIR = CACHE_DIR / "repoe"
REPOE_CACHE_TTL = 7 * 86400  # 7 days

# Grade-to-overlay tier mapping (local scoring → overlay colors)
GRADE_TIER_MAP = {
    "S": "high",
    "A": "good",
    "B": "decent",
    "C": "low",     # C and JUNK both show as ✗ (not worth reselling)
    "JUNK": "low",
}

# Calibration log file (grade vs actual trade price)
CALIBRATION_LOG_FILE = CACHE_DIR / "calibration.jsonl"

# Harvester state file (resumability across runs)
HARVESTER_STATE_FILE = CACHE_DIR / "harvester_state.json"

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE = Path(os.path.expanduser("~")) / ".poe2-price-overlay" / "overlay.log"

# ─────────────────────────────────────────────
# Bug Reporting (Discord webhook)
# ─────────────────────────────────────────────
DISCORD_WEBHOOK_URL = ""
BUG_REPORT_LOG_LINES = 200        # Tail of overlay.log to include
BUG_REPORT_MAX_CLIPBOARDS = 5     # Most recent clipboard debug files
BUG_REPORT_DB = CACHE_DIR / "bug_reports.jsonl"
DEBUG_DIR = Path(os.path.expanduser("~")) / ".poe2-price-overlay" / "debug"
