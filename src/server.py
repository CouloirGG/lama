"""
server.py — FastAPI backend for LAMA dashboard.

Manages the overlay subprocess (main.py), streams logs over WebSocket,
exposes status/settings APIs, and serves the dashboard HTML.

Endpoints:
  GET  /dashboard        → serves dashboard.html
  GET  /api/status       → overlay state + parsed stats
  POST /api/start        → launch main.py subprocess
  POST /api/stop         → graceful shutdown via CTRL_BREAK_EVENT
  POST /api/restart      → stop + start
  GET  /api/settings     → read dashboard_settings.json
  POST /api/settings     → write dashboard_settings.json
  GET  /api/leagues      → fetch leagues from poe2scout
  GET  /api/log          → recent log lines (initial load)
  WS   /ws               → real-time log + status streaming
"""

import asyncio
import json
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import platform

import requests
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from builds_client import (BuildsClient, enrich_item_mods, classify_build,
                           strip_ninja_brackets, _SKIP_SLOTS,
                           detect_current_anoint, compute_upgrade_priority,
                           compute_cost_tiers, find_lineage_upgrades,
                           get_anoint_description, compute_build_comparison,
                           compute_improvement_package,
                           SLOT_DISPLAY, SLOT_TO_UNIQUE_SLUG)
import guide_scraper
from bundle_paths import IS_FROZEN, APP_DIR, get_resource
from config import TRADE_STATS_CACHE_FILE, TRADE_ITEMS_CACHE_FILE
from item_lookup import ItemLookup
from oauth import OAuthManager
from price_cache import PriceCache
from game_commands import GameCommander
from stash_client import StashClient
from stash_scorer import StashScorer, TabSummary
from telemetry import TelemetryUploader
from trade_actions import TradeActions
from watchlist import WatchlistWorker

logger = logging.getLogger("dashboard")

# Hidden subprocess helper — suppresses console windows on Windows
_HIDDEN_SI = subprocess.STARTUPINFO()
_HIDDEN_SI.dwFlags |= subprocess.STARTF_USESHOWWINDOW
_HIDDEN_SI.wShowWindow = 0  # SW_HIDE
_HIDDEN_FLAGS = subprocess.CREATE_NO_WINDOW

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PORT = int(os.environ.get("POE2_DASHBOARD_PORT", "8450"))
SETTINGS_DIR = Path(os.path.expanduser("~")) / ".poe2-price-overlay"
SETTINGS_FILE = SETTINGS_DIR / "dashboard_settings.json"
POE2SCOUT_API = "https://poe2scout.com/api"

# Bug report (mirrors config.py constants)
DISCORD_WEBHOOK_URL = os.environ.get(
    "DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1476088582786519193/_GYenGzCpnxosoP_bvKYNbELYs5-rIIsfvcFenNUxr59GAQcwwvkJzC-Jt0rvmMAMftL"
).strip()
LOG_FILE = SETTINGS_DIR / "overlay.log"
DEBUG_DIR = SETTINGS_DIR / "debug"
BUG_REPORT_LOG_LINES = 200
BUG_REPORT_MAX_CLIPBOARDS = 5
BUG_REPORT_DB = SETTINGS_DIR / "cache" / "bug_reports.jsonl"

# Status line regex — matches main.py status format
STATUS_RE = re.compile(
    r"\[Status\] Uptime: (\d+)min \| "
    r"Triggers: (\d+) \| Prices shown: (\d+) \((\d+)%\) \| "
    r"Cache: (\d+) items \| "
    r"Last refresh: (.+?) \| "
    r"D2C: ([\d.]+) \| D2E: ([\d.]+) \| M2D: ([\d.]+) \| Cal: (\d+)"
)


# ---------------------------------------------------------------------------
# Settings manager
# ---------------------------------------------------------------------------
DEFAULT_SETTINGS = {
    "league": "Fate of the Vaal",
    "no_filter_update": False,
    "auto_start": True,
    "font_size": 14,
    "scan_fps": 8,
    "detection_cooldown": 1.0,
    "overlay_duration": 2.0,
    "cursor_still_radius": 20,
    "cursor_still_frames": 3,
    "filter_strictness": "normal",
    "filter_tier_styles": {},
    "filter_section_visibility": {},
    "filter_gear_classes": {},
    "filter_color_preset": "default",
    "watchlist_queries": [],
    "watchlist_poll_interval": 300,
    "watchlist_online_only": True,
    "start_with_windows": False,
    "overlay_show_grade": True,
    "overlay_show_price": True,
    "overlay_show_stars": True,
    "overlay_show_mods": False,
    "overlay_show_dps": True,
    "overlay_show_low_value": False,
    "overlay_display_preset": "standard",
    "overlay_tier_styles": {},
    "overlay_theme": "poe2",
    "overlay_pulse_style": "sheen",
    "telemetry_enabled": False,
    "poesessid": "",
    "nux_completed": False,
    "window_width": 1100,
    "window_height": 750,
    "window_maximized": False,
    "companion_enabled": False,
    "companion_pin": "",
}


# ---------------------------------------------------------------------------
# Windows auto-start (registry)
# ---------------------------------------------------------------------------
AUTOSTART_REG_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
AUTOSTART_REG_VALUE = "LAMA"


def _get_autostart_command() -> str:
    """Return the command string for the auto-start registry value."""
    if IS_FROZEN:
        return f'"{sys.executable}"'
    # Dev mode: launch via pythonw / python + app.py
    app_py = str(Path(__file__).parent / "app.py")
    return f'"{sys.executable}" "{app_py}"'


def set_autostart(enabled: bool):
    """Add or remove LAMA from Windows startup (HKCU\\...\\Run)."""
    try:
        import winreg
        if enabled:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, AUTOSTART_REG_KEY,
                                 0, winreg.KEY_SET_VALUE)
            winreg.SetValueEx(key, AUTOSTART_REG_VALUE, 0, winreg.REG_SZ,
                              _get_autostart_command())
            winreg.CloseKey(key)
            logger.info("Auto-start enabled (registry key set)")
        else:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, AUTOSTART_REG_KEY,
                                 0, winreg.KEY_SET_VALUE)
            try:
                winreg.DeleteValue(key, AUTOSTART_REG_VALUE)
                logger.info("Auto-start disabled (registry key removed)")
            except FileNotFoundError:
                pass  # already absent
            winreg.CloseKey(key)
    except Exception as e:
        logger.warning(f"Failed to update auto-start registry: {e}")


def get_autostart() -> bool:
    """Check if the LAMA auto-start registry key exists."""
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, AUTOSTART_REG_KEY,
                             0, winreg.KEY_QUERY_VALUE)
        try:
            winreg.QueryValueEx(key, AUTOSTART_REG_VALUE)
            return True
        except FileNotFoundError:
            return False
        finally:
            winreg.CloseKey(key)
    except Exception:
        return False


def deep_merge(base: dict, updates: dict) -> dict:
    """Deep merge updates into base dict. Mutates and returns base."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


# ---------------------------------------------------------------------------
# Companion mode utilities
# ---------------------------------------------------------------------------
import random
import socket
import io

COMPANION_CHARSET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # no 0/O/1/I/L


def generate_pin(length: int = 4) -> str:
    """Generate a random PIN from the safe charset."""
    return "".join(random.choice(COMPANION_CHARSET) for _ in range(length))


def get_lan_ip() -> str:
    """Get the LAN IP address via UDP trick (no actual packet sent)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def load_settings() -> dict:
    """Load settings from disk, merging with defaults."""
    settings = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as f:
                saved = json.load(f)
            settings.update(saved)
        except Exception as e:
            logger.warning(f"Failed to load settings: {e}")
    return settings


def save_settings(settings: dict):
    """Persist settings to disk."""
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save settings: {e}")


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------
class ConnectionManager:
    """Manages active WebSocket connections and broadcasts events."""

    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)
        logger.info(f"WebSocket connected ({len(self.connections)} active)")

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)
        logger.info(f"WebSocket disconnected ({len(self.connections)} active)")

    async def broadcast(self, event: dict):
        """Send a JSON event to all connected clients."""
        dead = []
        for ws in self.connections:
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Overlay subprocess manager
# ---------------------------------------------------------------------------
class OverlayProcess:
    """Manages the main.py overlay subprocess lifecycle."""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.state = "stopped"  # stopped | starting | running | error
        self.started_at: Optional[float] = None
        self.reader_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Stats parsed from [Status] lines
        self.stats = {
            "uptime_min": 0,
            "triggers": 0,
            "prices_shown": 0,
            "success_rate": 0,
            "cache_items": 0,
            "last_refresh": "never",
            "divine_to_chaos": 0,
            "divine_to_exalted": 0,
            "mirror_to_divine": 0,
            "calibration_samples": 0,
        }

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def start(self, league: str, no_filter_update: bool = False):
        """Spawn main.py as a subprocess."""
        if self.process and self.process.poll() is None:
            return {"error": "Overlay is already running"}

        self.state = "starting"

        if IS_FROZEN:
            cmd = [sys.executable, "--overlay-worker", "--league", league]
        else:
            cmd = [sys.executable, str(Path(__file__).parent / "main.py"), "--league", league]
        if no_filter_update:
            cmd.append("--no-filter-update")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(APP_DIR),
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | _HIDDEN_FLAGS,
                startupinfo=_HIDDEN_SI,
            )
            self.started_at = time.time()
            self.state = "running"

            # Reset stats
            self.stats = {
                "uptime_min": 0,
                "triggers": 0,
                "prices_shown": 0,
                "success_rate": 0,
                "cache_items": 0,
                "last_refresh": "never",
                "divine_to_chaos": 0,
                "divine_to_exalted": 0,
                "calibration_samples": 0,
            }

            # Start output reader thread
            self.reader_thread = threading.Thread(
                target=self._read_output, daemon=True
            )
            self.reader_thread.start()

            logger.info(f"Overlay started: PID {self.process.pid}, league={league}")
            return {"status": "started", "pid": self.process.pid}

        except Exception as e:
            self.state = "error"
            logger.error(f"Failed to start overlay: {e}")
            return {"error": str(e)}

    def stop(self):
        """Gracefully stop the overlay subprocess."""
        if not self.process or self.process.poll() is not None:
            self.state = "stopped"
            return {"status": "not_running"}

        pid = self.process.pid
        logger.info(f"Stopping overlay PID {pid}...")

        try:
            # Send CTRL_BREAK_EVENT — bypasses SetConsoleCtrlHandler(None, True)
            # in main.py:1110, delivers KeyboardInterrupt for graceful shutdown
            os.kill(pid, signal.CTRL_BREAK_EVENT)

            try:
                self.process.wait(timeout=5)
                logger.info(f"Overlay stopped gracefully (PID {pid})")
            except subprocess.TimeoutExpired:
                logger.warning(f"Overlay didn't stop in 5s, killing PID {pid}")
                self.process.kill()
                self.process.wait(timeout=3)

        except Exception as e:
            logger.error(f"Error stopping overlay: {e}")
            try:
                self.process.kill()
            except Exception:
                pass
            return {"error": str(e)}
        finally:
            self.process = None
            self.started_at = None
            self.state = "stopped"

        return {"status": "stopped", "pid": pid}

    def get_status(self) -> dict:
        """Return current overlay status and stats."""
        # Check if process crashed
        if self.process and self.process.poll() is not None:
            self.state = "error"
            self.process = None

        uptime = 0
        if self.started_at and self.state == "running":
            uptime = int(time.time() - self.started_at)

        from config import APP_VERSION, GIT_BRANCH, IS_DEV_BUILD
        return {
            "state": self.state,
            "uptime": uptime,
            "stats": dict(self.stats),
            "version": APP_VERSION,
            "branch": GIT_BRANCH,
            "is_dev": IS_DEV_BUILD,
        }

    def _classify_line(self, line: str) -> str:
        """Assign a color to a log line based on content."""
        lower = line.lower()
        if "[status]" in lower:
            return "#818cf8"  # purple for status
        if "error" in lower or "failed" in lower or "exception" in lower:
            return "#ef4444"  # red
        if "warning" in lower or "warn" in lower:
            return "#f59e0b"  # amber
        if "price:" in lower or "divine" in lower or "exalted" in lower:
            return "#34d399"  # green for prices
        if "cache" in lower or "refresh" in lower:
            return "#22d3ee"  # cyan
        if "session summary" in lower or "=====" in lower:
            return "#fbbf24"  # yellow
        return "#94a3b8"  # default grey

    def _read_output(self):
        """Read stdout from subprocess and queue lines for broadcast."""
        try:
            for raw_line in iter(self.process.stdout.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue

                # Parse [Status] lines for stats
                m = STATUS_RE.search(line)
                if m:
                    self.stats = {
                        "uptime_min": int(m.group(1)),
                        "triggers": int(m.group(2)),
                        "prices_shown": int(m.group(3)),
                        "success_rate": int(m.group(4)),
                        "cache_items": int(m.group(5)),
                        "last_refresh": m.group(6),
                        "divine_to_chaos": float(m.group(7)),
                        "divine_to_exalted": float(m.group(8)),
                        "mirror_to_divine": float(m.group(9)),
                        "calibration_samples": int(m.group(10)),
                    }

                color = self._classify_line(line)

                # Extract timestamp if present (HH:MM:SS format from logger)
                ts_match = re.match(r"^(\d{2}:\d{2}:\d{2})\s+(.*)$", line)
                if ts_match:
                    ts = ts_match.group(1)
                    msg = ts_match.group(2)
                else:
                    ts = time.strftime("%H:%M:%S")
                    msg = line

                log_entry = {"time": ts, "message": msg, "color": color}

                # Add to log buffer
                log_buffer.append(log_entry)

                # Broadcast to WebSocket clients
                if self._loop:
                    asyncio.run_coroutine_threadsafe(
                        ws_manager.broadcast({"type": "log", **log_entry}),
                        self._loop,
                    )

        except Exception as e:
            logger.error(f"Output reader error: {e}")
        finally:
            # Process has ended
            if self.state == "running":
                self.state = "error"
                if self._loop:
                    asyncio.run_coroutine_threadsafe(
                        ws_manager.broadcast({
                            "type": "state_change",
                            "state": "error",
                        }),
                        self._loop,
                    )


overlay = OverlayProcess()
log_buffer: deque[dict] = deque(maxlen=500)
watchlist_worker: Optional[WatchlistWorker] = None
price_cache: Optional[PriceCache] = None
item_lookup: Optional[ItemLookup] = None
telemetry_uploader: Optional[TelemetryUploader] = None
game_commander = GameCommander()
trade_actions: Optional[TradeActions] = None

# Character viewer
builds_client = BuildsClient()

# Stash viewer
oauth_manager: Optional[OAuthManager] = None
stash_client: Optional[StashClient] = None
stash_scorer: Optional[StashScorer] = None
stash_data: dict = {
    "tabs": [],           # List[TabSummary serialized]
    "last_refresh": None,
    "total_value": 0.0,
    "refreshing": False,
}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Set up the event loop reference and background tasks."""
    global watchlist_worker, price_cache, item_lookup, telemetry_uploader, trade_actions
    global oauth_manager, stash_client, stash_scorer

    loop = asyncio.get_running_loop()
    overlay.set_loop(loop)

    # Background task: periodic status push
    status_task = asyncio.create_task(status_broadcast_loop())

    # Initialize watchlist worker
    settings = load_settings()
    league = settings.get("league", "Fate of the Vaal")
    watchlist_worker = WatchlistWorker(league, broadcast_fn=ws_manager.broadcast,
                                       log_buffer=log_buffer)
    watchlist_worker.update_queries(
        settings.get("watchlist_queries", []),
        settings.get("watchlist_poll_interval", 300),
        online_only=settings.get("watchlist_online_only", True),
    )
    # Propagate POESESSID to watchlist worker if configured
    poesessid = settings.get("poesessid", "")
    if poesessid:
        watchlist_worker.set_session_id(poesessid)
    watchlist_worker.start(loop)

    # Initialize trade actions (authenticated API calls)
    trade_actions = TradeActions(lambda: load_settings().get("poesessid", ""))

    # Server-side PriceCache for Markets tab (works without overlay running)
    price_cache = PriceCache(league=league)
    price_cache.start()

    # Initialize item lookup in background thread (loads RePoE data)
    item_lookup = ItemLookup()
    _il = item_lookup
    def _init_lookup():
        try:
            _il.initialize()
        except Exception as e:
            logger.warning(f"Item lookup init failed: {e}")
    threading.Thread(target=_init_lookup, daemon=True).start()

    # Background task: check for updates after a short delay
    update_task = asyncio.create_task(check_for_updates())

    # Sync auto-start setting with actual registry state
    actual_autostart = get_autostart()
    if settings.get("start_with_windows", False) != actual_autostart:
        settings["start_with_windows"] = actual_autostart
        save_settings(settings)

    # Initialize telemetry uploader (opt-in)
    telemetry_uploader = TelemetryUploader(league=league)
    if settings.get("telemetry_enabled", False):
        telemetry_uploader.start_schedule()

    # Initialize OAuth + Stash viewer
    oauth_manager = OAuthManager()
    stash_client = StashClient(oauth_manager)
    stash_scorer = StashScorer()
    def _init_scorer():
        try:
            stash_scorer.initialize()
        except Exception as e:
            logger.warning(f"StashScorer init failed: {e}")
    threading.Thread(target=_init_scorer, daemon=True).start()

    logger.info("LAMA dashboard server ready")
    try:
        yield
    finally:
        status_task.cancel()
        update_task.cancel()
        if watchlist_worker:
            watchlist_worker.stop()
        if price_cache:
            price_cache.stop()
        if telemetry_uploader:
            telemetry_uploader.stop_schedule()
        # Stop overlay if running
        overlay.stop()


app = FastAPI(title="LAMA API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_github_headers() -> dict:
    """Build GitHub API headers, including auth token if available."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "POE2-Price-Overlay",
    }
    # Try gh CLI token (works on dev machines with gh installed)
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True, text=True, timeout=5,
            creationflags=_HIDDEN_FLAGS, startupinfo=_HIDDEN_SI,
        )
        token = result.stdout.strip()
        if token:
            headers["Authorization"] = f"token {token}"
    except Exception:
        pass
    # Also accept explicit env var
    env_token = os.environ.get("GITHUB_TOKEN", "")
    if env_token:
        headers["Authorization"] = f"token {env_token}"
    return headers


async def check_for_updates():
    """After a short delay, check GitHub for a newer release."""
    await asyncio.sleep(5)
    try:
        from config import APP_VERSION
        if APP_VERSION == "dev":
            return
        loop = asyncio.get_running_loop()
        gh_headers = _get_github_headers()
        resp = await loop.run_in_executor(None, lambda: requests.get(
            "https://api.github.com/repos/CouloirGG/lama/releases/latest",
            timeout=10,
            headers=gh_headers,
        ))
        if resp.status_code != 200:
            return
        data = resp.json()
        latest_tag = data.get("tag_name", "").lstrip("v")
        release_url = data.get("html_url", "")
        if not latest_tag:
            return
        # Simple version comparison (major.minor.patch)
        def _ver_tuple(v):
            parts = v.split(".")
            return tuple(int(p) for p in parts if p.isdigit())
        if _ver_tuple(latest_tag) > _ver_tuple(APP_VERSION):
            logger.info(f"Update available: v{latest_tag} (current: v{APP_VERSION})")
            # Find Setup exe asset for one-click update
            # Use API url (not browser_download_url) — works for private repos
            setup_url = ""
            for asset in data.get("assets", []):
                name = asset.get("name", "")
                if "Setup" in name and name.endswith(".exe"):
                    setup_url = asset.get("url", "") or asset.get("browser_download_url", "")
                    break
            await ws_manager.broadcast({
                "type": "update_available",
                "current": APP_VERSION,
                "latest": latest_tag,
                "url": release_url,
                "setup_url": setup_url,
            })
    except Exception as e:
        logger.debug(f"Update check failed: {e}")


def _merge_cache_rates(status: dict):
    """Fill KPI exchange rates from server-side price_cache.

    The overlay subprocess reports rates via [Status] log lines, but those
    only appear when the overlay is running.  The server-side price_cache
    refreshes independently, so we always have fresh rates available.
    """
    cache_stats = price_cache.get_stats()
    stats = status.setdefault("stats", {})
    for key in ("divine_to_chaos", "divine_to_exalted", "mirror_to_divine"):
        if cache_stats.get(key):
            stats[key] = cache_stats[key]
    # Also fill cache metadata (items count, last refresh time)
    if cache_stats.get("total_items"):
        stats["cache_items"] = cache_stats["total_items"]
    if cache_stats.get("last_refresh") and cache_stats["last_refresh"] != "Never":
        stats["last_refresh"] = cache_stats["last_refresh"]


async def status_broadcast_loop():
    """Push status updates to WebSocket clients every 5 seconds."""
    while True:
        await asyncio.sleep(5)
        if ws_manager.connections:
            status = overlay.get_status()
            if price_cache:
                _merge_cache_rates(status)
            await ws_manager.broadcast({"type": "status", **status})


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class StartRequest(BaseModel):
    league: Optional[str] = None
    no_filter_update: Optional[bool] = None


class SettingsRequest(BaseModel):
    league: Optional[str] = None
    no_filter_update: Optional[bool] = None
    auto_start: Optional[bool] = None
    font_size: Optional[int] = None
    scan_fps: Optional[int] = None
    detection_cooldown: Optional[float] = None
    overlay_duration: Optional[float] = None
    cursor_still_radius: Optional[int] = None
    cursor_still_frames: Optional[int] = None
    filter_strictness: Optional[str] = None
    filter_tier_styles: Optional[dict] = None
    filter_section_visibility: Optional[dict] = None
    filter_gear_classes: Optional[dict] = None
    filter_color_preset: Optional[str] = None
    watchlist_queries: Optional[list] = None
    watchlist_poll_interval: Optional[int] = None
    watchlist_online_only: Optional[bool] = None
    start_with_windows: Optional[bool] = None
    overlay_show_grade: Optional[bool] = None
    overlay_show_price: Optional[bool] = None
    overlay_show_stars: Optional[bool] = None
    overlay_show_mods: Optional[bool] = None
    overlay_show_dps: Optional[bool] = None
    overlay_show_low_value: Optional[bool] = None
    overlay_display_preset: Optional[str] = None
    overlay_tier_styles: Optional[dict] = None
    overlay_theme: Optional[str] = None
    overlay_pulse_style: Optional[str] = None
    telemetry_enabled: Optional[bool] = None
    poesessid: Optional[str] = None
    nux_completed: Optional[bool] = None


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------
@app.get("/api/status")
async def get_status():
    status = overlay.get_status()
    # Merge server-side price_cache rates so KPIs work without overlay running
    if price_cache:
        _merge_cache_rates(status)
    return status


@app.post("/api/start")
async def start_overlay(req: StartRequest = StartRequest()):
    settings = load_settings()
    league = req.league or settings.get("league", "Fate of the Vaal")
    no_filter = req.no_filter_update if req.no_filter_update is not None else settings.get("no_filter_update", False)

    result = overlay.start(league, no_filter_update=no_filter)

    if "error" not in result:
        await ws_manager.broadcast({
            "type": "state_change",
            "state": "running",
        })

    return result


@app.post("/api/stop")
async def stop_overlay():
    result = overlay.stop()
    await ws_manager.broadcast({
        "type": "state_change",
        "state": "stopped",
    })
    return result


@app.post("/api/restart")
async def restart_overlay(req: StartRequest = StartRequest()):
    overlay.stop()
    await ws_manager.broadcast({"type": "state_change", "state": "stopped"})

    # Brief pause for cleanup
    await asyncio.sleep(0.5)

    settings = load_settings()
    league = req.league or settings.get("league", "Fate of the Vaal")
    no_filter = req.no_filter_update if req.no_filter_update is not None else settings.get("no_filter_update", False)

    result = overlay.start(league, no_filter_update=no_filter)
    if "error" not in result:
        await ws_manager.broadcast({"type": "state_change", "state": "running"})

    return result


def _redact_settings(settings: dict) -> dict:
    """Return settings with sensitive fields redacted for API responses."""
    out = dict(settings)
    if out.get("poesessid"):
        out["poesessid_set"] = True
        out["poesessid"] = ""
    else:
        out["poesessid_set"] = False
    out.pop("companion_pin", None)
    return out


@app.get("/api/settings")
async def get_settings():
    return _redact_settings(load_settings())


@app.post("/api/settings")
async def update_settings(req: SettingsRequest):
    settings = load_settings()
    updates = req.model_dump(exclude_none=True)
    # Keys that should be replaced wholesale (client sends full object, not partial)
    REPLACE_KEYS = {"filter_tier_styles", "filter_gear_classes", "watchlist_queries", "overlay_tier_styles"}
    for key in REPLACE_KEYS:
        if key in updates:
            settings[key] = updates.pop(key)
    deep_merge(settings, updates)
    save_settings(settings)
    await ws_manager.broadcast({"type": "settings", "settings": _redact_settings(settings)})

    # Update Windows auto-start registry if the setting changed
    if "start_with_windows" in updates:
        set_autostart(settings.get("start_with_windows", False))

    # Update server-side price cache if league changed
    if "league" in updates and price_cache:
        new_league = settings.get("league", "Fate of the Vaal")
        price_cache.league = new_league

    # Start/stop telemetry schedule if the setting changed
    if "telemetry_enabled" in updates and telemetry_uploader:
        if settings.get("telemetry_enabled", False):
            telemetry_uploader.start_schedule()
        else:
            telemetry_uploader.stop_schedule()

    # Propagate POESESSID to watchlist worker
    if "poesessid" in updates and watchlist_worker:
        watchlist_worker.set_session_id(settings.get("poesessid", ""))

    # Notify watchlist worker if queries, interval, or online filter changed
    if watchlist_worker and ("watchlist_queries" in updates or "watchlist_poll_interval" in updates or "watchlist_online_only" in updates):
        queries = settings.get("watchlist_queries", [])
        watchlist_worker.update_queries(
            queries,
            settings.get("watchlist_poll_interval", 300),
            online_only=settings.get("watchlist_online_only", True),
        )
        # Force-refresh all enabled queries so results appear immediately
        enabled = [q for q in queries if q.get("enabled", True) and q.get("id")]
        for q in enabled:
            watchlist_worker.force_refresh(q["id"])
        # Log to dashboard console
        log_entry = {
            "time": time.strftime("%H:%M:%S"),
            "message": f"Watchlist: {len(enabled)} queries queued for refresh",
            "color": "#6b8f71",
        }
        log_buffer.append(log_entry)
        await ws_manager.broadcast({"type": "log", **log_entry})

    return settings


@app.post("/api/window-geometry")
async def save_window_geometry(req: Request):
    """Persist window size/maximized state (called on resize, no broadcast)."""
    body = await req.json()
    settings = load_settings()
    if "width" in body and isinstance(body["width"], (int, float)):
        settings["window_width"] = max(900, int(body["width"]))
    if "height" in body and isinstance(body["height"], (int, float)):
        settings["window_height"] = max(600, int(body["height"]))
    if "maximized" in body:
        settings["window_maximized"] = bool(body["maximized"])
    save_settings(settings)
    return {"ok": True}


@app.get("/api/leagues")
async def get_leagues():
    """Fetch available leagues from poe2scout API."""
    try:
        resp = requests.get(
            f"{POE2SCOUT_API}/leagues",
            timeout=10,
            headers={"User-Agent": "POE2-Price-Overlay-Dashboard/1.0"},
        )
        if resp.status_code == 200:
            leagues = resp.json()
            # Extract league names (value field)
            return {
                "leagues": [
                    {"value": lg.get("value", ""), "label": lg.get("value", "")}
                    for lg in leagues
                    if lg.get("value")
                ]
            }
        return {"leagues": [], "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        logger.warning(f"Failed to fetch leagues: {e}")
        # Fallback
        return {
            "leagues": [
                {"value": "Fate of the Vaal", "label": "Fate of the Vaal"},
                {"value": "Standard", "label": "Standard"},
                {"value": "Hardcore Fate of the Vaal", "label": "Hardcore Fate of the Vaal"},
                {"value": "Hardcore", "label": "Hardcore"},
            ],
            "error": str(e),
        }


@app.get("/api/log")
async def get_log():
    """Return recent log lines for initial load."""
    return {"lines": list(log_buffer)}


# ---------------------------------------------------------------------------
# Item Lookup
# ---------------------------------------------------------------------------
class ItemLookupRequest(BaseModel):
    text: str

@app.post("/api/item-lookup")
async def post_item_lookup(req: ItemLookupRequest):
    """Parse and score pasted item text."""
    if not item_lookup or not item_lookup.ready:
        return JSONResponse(
            status_code=503,
            content={"error": "Item lookup is still initializing, try again in a moment"},
        )
    result = item_lookup.lookup(req.text)
    if result is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Could not parse item text"},
        )
    return result


# ---------------------------------------------------------------------------
# Watchlist endpoints
# ---------------------------------------------------------------------------
@app.get("/api/watchlist/results")
async def get_watchlist_results():
    """Return all cached watchlist results."""
    if not watchlist_worker:
        return {}
    return watchlist_worker.get_results()


@app.post("/api/watchlist/refresh/{query_id}")
async def refresh_watchlist_query(query_id: str):
    """Force-refresh a single watchlist query."""
    if not watchlist_worker:
        return {"error": "Watchlist not initialized"}
    watchlist_worker.force_refresh(query_id)
    return {"status": "queued", "query_id": query_id}


# ---------------------------------------------------------------------------
# Trade action endpoints (whisper, invite, hideout, trade, kick)
# ---------------------------------------------------------------------------
class TradeActionRequest(BaseModel):
    player: str = ""
    token: str = ""
    whisper: str = ""


@app.post("/api/trade/whisper")
async def trade_whisper(req: TradeActionRequest):
    """Send a trade whisper — via token API if available, else chat fallback."""
    settings = load_settings()
    poesessid = settings.get("poesessid", "")

    if req.token and poesessid and trade_actions:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, trade_actions.whisper_via_token, req.token
        )
        if result.get("status") == "sent":
            return {**result, "method": "api"}
        logger.warning(f"Whisper token API failed: {result.get('error')}, falling back to chat")

    # The trade API whisper field is the full ready-to-paste message
    # (e.g. "@CharName Hi, I would like to buy..."), so paste it directly.
    if req.whisper:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, game_commander.type_in_chat, req.whisper
        )
        return {**result, "method": "chat"}

    if not req.player:
        return {"error": "No whisper text or player name available"}

    # Bare fallback — just open whisper prompt to player (no message)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, game_commander.type_in_chat, f"@{req.player} ", False
    )
    return {**result, "method": "chat"}


@app.post("/api/trade/invite")
async def trade_invite(req: TradeActionRequest):
    """Send /invite <player> via chat."""
    if not req.player:
        return {"error": "Player name required"}
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, game_commander.invite, req.player)
    return result


@app.post("/api/trade/hideout")
async def trade_hideout(req: TradeActionRequest):
    """Visit a player's hideout — requires hideout_token + POESESSID.

    POE2 has no /hideout <player> chat command (unlike POE1).
    The only programmatic way is via the token API.
    """
    settings = load_settings()
    poesessid = settings.get("poesessid", "")

    if req.token and poesessid and trade_actions:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, trade_actions.hideout_via_token, req.token
        )
        if result.get("status") == "sent":
            return {**result, "method": "api"}
        return {**result, "method": "api"}

    return {"error": "Hideout requires POESESSID + hideout token (POE2 has no /hideout <player> command)"}


@app.post("/api/trade/tradewith")
async def trade_tradewith(req: TradeActionRequest):
    """Send /tradewith <player> via chat."""
    if not req.player:
        return {"error": "Player name required"}
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, game_commander.trade_with, req.player)
    return result


@app.post("/api/trade/kick")
async def trade_kick(req: TradeActionRequest):
    """Send /kick <player> via chat."""
    if not req.player:
        return {"error": "Player name required"}
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, game_commander.kick, req.player)
    return result


# ---------------------------------------------------------------------------
# Rate history endpoint (companion mobile app)
# ---------------------------------------------------------------------------
@app.get("/api/rate-history")
async def get_rate_history(since: int = 0):
    """Return rate history snapshots for the mobile companion app.

    Query params:
        since: Unix timestamp — only return entries newer than this (default 0 = all).
    """
    if not price_cache:
        return []
    history = price_cache._load_rate_history()
    if since > 0:
        history = [h for h in history if h.get("ts", 0) > since]
    return history


# ---------------------------------------------------------------------------
# Market data endpoint (Markets tab)
# ---------------------------------------------------------------------------
@app.get("/api/market-data")
async def get_market_data():
    """Return currency exchange data with sparklines for the Markets tab."""
    if not price_cache:
        return {"currencies": [], "rates": {}, "history": [], "last_refresh": "Never", "league": ""}
    return price_cache.get_market_data()


@app.get("/api/trade-data/stats")
async def get_trade_stats():
    """Return flattened stat definitions from cache for autocomplete."""
    if not TRADE_STATS_CACHE_FILE.exists():
        return []
    try:
        with open(TRADE_STATS_CACHE_FILE) as f:
            data = json.load(f)
        stats = []
        for group in data.get("result", []):
            group_label = group.get("label", "")
            for entry in group.get("entries", []):
                stats.append({
                    "id": entry.get("id", ""),
                    "text": entry.get("text", ""),
                    "type": group_label,
                })
        return stats
    except Exception as e:
        logger.warning(f"Failed to load trade stats: {e}")
        return []


@app.get("/api/trade-data/items")
async def get_trade_items():
    """Return base types grouped by category from cache."""
    if not TRADE_ITEMS_CACHE_FILE.exists():
        return []
    try:
        with open(TRADE_ITEMS_CACHE_FILE) as f:
            data = json.load(f)
        categories = []
        for group in data.get("result", []):
            entries = []
            for entry in group.get("entries", []):
                item = {"type": entry.get("type", "")}
                if entry.get("name"):
                    item["name"] = entry["name"]
                if entry.get("text"):
                    item["text"] = entry["text"]
                entries.append(item)
            categories.append({
                "label": group.get("label", ""),
                "entries": entries,
            })
        return categories
    except Exception as e:
        logger.warning(f"Failed to load trade items: {e}")
        return []



# ---------------------------------------------------------------------------
# Meta overview endpoints (class stats, popular skills, anoints)
# ---------------------------------------------------------------------------

@app.get("/api/meta/summary")
async def meta_summary():
    """Fetch class distribution / build summary from poe.ninja."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, builds_client.fetch_build_summary)
    if not result:
        return JSONResponse(status_code=502, content={"error": "Failed to fetch build summary"})
    return result


@app.get("/api/meta/skills")
async def meta_popular_skills():
    """Fetch popular skills ranked by usage."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, builds_client.fetch_popular_skills_list)
    return result


class AnointsRequest(BaseModel):
    characterClass: str = "all"
    skill: str


@app.post("/api/meta/anoints")
async def meta_anoints(req: AnointsRequest):
    """Fetch popular anoints for a class+skill combo."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, builds_client.fetch_popular_anoints,
        req.characterClass, req.skill
    )
    return result


# ---------------------------------------------------------------------------
# Character lookup endpoints (poe.ninja Builds API)
# ---------------------------------------------------------------------------
class CharacterLookupRequest(BaseModel):
    account: str
    character: str


@app.post("/api/character/lookup")
async def character_lookup(req: CharacterLookupRequest):
    """Look up a character by account + name via poe.ninja."""
    if not req.account.strip() or not req.character.strip():
        return JSONResponse(status_code=400, content={"error": "Account and character name required"})
    loop = asyncio.get_running_loop()
    char_data = await loop.run_in_executor(
        None, builds_client.lookup_character, req.account.strip(), req.character.strip()
    )
    if not char_data:
        return JSONResponse(status_code=404, content={"error": "Character not found. Make sure the account and character names are correct, and the character is on the current league's ladder."})

    # Auto-save to recent characters
    _save_recent_character(
        req.account.strip(), char_data.name,
        char_data.ascendancy or char_data.char_class, char_data.level,
    )

    result = builds_client.serialize_character(char_data)

    # Enrich equipment mods with tier data if mod engine is available
    if item_lookup and item_lookup.ready:
        try:
            mp = item_lookup._mod_parser
            mdb = item_lookup._mod_database
            for i, eq in enumerate(char_data.equipment):
                if eq.slot in _SKIP_SLOTS:
                    continue
                tier_data = enrich_item_mods(eq, mp, mdb)
                if tier_data and i < len(result.get("equipment", [])):
                    result["equipment"][i]["modTiers"] = tier_data
        except Exception as e:
            logger.debug(f"Mod tier enrichment failed: {e}")

    return result


@app.get("/api/character/saved")
async def get_saved_characters():
    """Return saved characters list (most recent first)."""
    settings = load_settings()
    return settings.get("saved_characters", [])


class CharacterDeleteRequest(BaseModel):
    account: str
    character: str = ""


@app.post("/api/character/delete")
async def delete_saved_character(req: CharacterDeleteRequest):
    """Remove a saved character (or entire account if no character specified)."""
    settings = load_settings()
    saved = settings.get("saved_characters", [])
    acct_lower = req.account.strip().lower()

    if req.character.strip():
        # Remove specific character
        char_lower = req.character.strip().lower()
        for acct in saved:
            if acct["accountName"].lower() == acct_lower:
                acct["characters"] = [
                    c for c in acct["characters"]
                    if c["name"].lower() != char_lower
                ]
                break
        # Remove accounts with no characters
        saved = [a for a in saved if a["characters"]]
    else:
        # Remove entire account
        saved = [a for a in saved if a["accountName"].lower() != acct_lower]

    settings["saved_characters"] = saved
    save_settings(settings)
    return {"status": "ok", "saved": saved}


def _save_recent_character(account: str, name: str, char_class: str, level: int):
    """Save a character to the recent characters list in settings."""
    settings = load_settings()
    saved = settings.get("saved_characters", [])

    char_entry = {
        "name": name,
        "class": char_class,
        "level": level,
        "lastLookup": int(time.time()),
    }

    # Find or create account
    acct = None
    for a in saved:
        if a["accountName"].lower() == account.lower():
            acct = a
            break

    if acct:
        # Upsert character
        existing_idx = next(
            (i for i, c in enumerate(acct["characters"])
             if c["name"].lower() == name.lower()), -1
        )
        if existing_idx >= 0:
            acct["characters"][existing_idx] = char_entry
        else:
            acct["characters"].append(char_entry)
        acct["lastUsed"] = int(time.time())
    else:
        saved.append({
            "accountName": account,
            "characters": [char_entry],
            "lastUsed": int(time.time()),
        })

    # Sort by most recent first
    saved.sort(key=lambda a: a.get("lastUsed", 0), reverse=True)

    settings["saved_characters"] = saved
    save_settings(settings)


class PopularItemsRequest(BaseModel):
    account: str
    character: str
    slot: str


@app.post("/api/character/popular-items")
async def character_popular_items(req: PopularItemsRequest):
    """Fetch popular items for a slot, relative to a character's class/skill."""
    if not req.account.strip() or not req.character.strip() or not req.slot.strip():
        return JSONResponse(status_code=400, content={"error": "Account, character, and slot required"})
    loop = asyncio.get_running_loop()

    # Look up character first (cached)
    char_data = await loop.run_in_executor(
        None, builds_client.lookup_character, req.account.strip(), req.character.strip()
    )
    if not char_data:
        return JSONResponse(status_code=404, content={"error": "Character not found"})

    # Fetch popular items for the slot
    result = await loop.run_in_executor(
        None, builds_client.get_popular_items_for_slot, char_data, req.slot.strip()
    )
    return result


class BuildInsightsRequest(BaseModel):
    account: str
    character: str


@app.post("/api/character/build-insights")
async def character_build_insights(req: BuildInsightsRequest):
    """Classify build archetype and compute per-slot tier summary."""
    if not req.account.strip() or not req.character.strip():
        return JSONResponse(status_code=400, content={"error": "Account and character required"})
    loop = asyncio.get_running_loop()

    # Look up character (cached)
    char_data = await loop.run_in_executor(
        None, builds_client.lookup_character, req.account.strip(), req.character.strip()
    )
    if not char_data:
        return JSONResponse(status_code=404, content={"error": "Character not found"})

    # Classify build
    archetype = classify_build(char_data)

    # Fetch popular keystones
    char_class = char_data.ascendancy or char_data.char_class
    main_skill = archetype.main_skill
    popular_ks = await loop.run_in_executor(
        None, builds_client.fetch_popular_keystones, char_class, main_skill
    )

    # Compare user's keystones vs popular
    user_ks_set = set(char_data.keystones)
    keystone_gaps = []
    for pks in popular_ks[:10]:
        keystone_gaps.append({
            "name": pks["name"],
            "percentage": pks["percentage"],
            "hasIt": pks["name"] in user_ks_set,
        })

    # Per-slot tier summary from enriched mod data
    slot_summary = []
    if item_lookup and item_lookup.ready:
        try:
            mp = item_lookup._mod_parser
            mdb = item_lookup._mod_database
            for eq in char_data.equipment:
                if eq.slot in _SKIP_SLOTS:
                    continue
                tier_data = enrich_item_mods(eq, mp, mdb)
                # Flatten all tier results across mod types
                all_tiers = []
                for mod_type_key, tiers in tier_data.items():
                    for t in tiers:
                        if t is not None:
                            all_tiers.append(t)

                # Only consider meaningful mods for avg/weakest (weight >= 0.5,
                # tier_count <= 15 to exclude absurdly deep defence ladders)
                meaningful = [t for t in all_tiers
                              if t["weight"] >= 0.5 and t["tier_count"] <= 15]

                t1_count = sum(1 for t in meaningful if t["tier_num"] == 1)
                mod_count = sum(
                    len(m_list) for m_list in [
                        eq.implicit_mods, eq.explicit_mods, eq.crafted_mods,
                        eq.fractured_mods, eq.desecrated_mods, eq.rune_mods,
                    ] if m_list
                )
                avg_tier = 0
                if meaningful:
                    avg_tier = round(sum(t["tier_num"] for t in meaningful) / len(meaningful), 1)

                # Find lowest-priority mod — highest tier number among meaningful mods
                weakest = None
                if meaningful:
                    worst = max(meaningful, key=lambda t: t["tier_num"])
                    if worst["tier_num"] >= 3:
                        weakest = {
                            "display_name": worst["display_name"],
                            "tier_num": worst["tier_num"],
                            "tier_count": worst["tier_count"],
                        }

                # 2A: Detailed improvement info — weak mods with next-tier targets
                weak_mods = []
                for t in sorted(meaningful, key=lambda t: -t["tier_num"]):
                    if t["tier_num"] >= 3:
                        current_range = t.get("tier_range", {})
                        next_t = t.get("next_tier")
                        wm = {
                            "name": t["display_name"],
                            "tier": t["tier_num"],
                            "tierCount": t["tier_count"],
                            "currentRange": f"{current_range.get('min', '?')}-{current_range.get('max', '?')}",
                        }
                        if next_t:
                            wm["nextTierRange"] = f"{next_t['min']}-{next_t['max']}"
                            wm["nextTier"] = t["tier_num"] - 1
                        weak_mods.append(wm)
                    if len(weak_mods) >= 3:
                        break

                # Dead mods on this slot (from archetype analysis)
                slot_dead = [dm for dm in (archetype.dead_mods or []) if dm.get("slot") == eq.slot]

                from builds_client import SLOT_DISPLAY
                slot_summary.append({
                    "slot": eq.slot,
                    "slotDisplay": SLOT_DISPLAY.get(eq.slot, eq.slot),
                    "itemName": eq.name or eq.type_line,
                    "avgTier": avg_tier,
                    "t1Count": t1_count,
                    "modCount": mod_count,
                    "enrichedCount": len(meaningful),
                    "weakest": weakest,
                    "weakMods": weak_mods,
                    "deadMods": [{"mod": dm["mod"], "reason": dm["reason"]} for dm in slot_dead[:2]],
                })
        except Exception as e:
            logger.debug(f"Slot summary failed: {e}")

    return {
        "archetype": {
            "tags": archetype.tags,
            "damageType": archetype.damage_type,
            "defenseType": archetype.defense_type,
            "mainSkill": archetype.main_skill,
            "isCrit": archetype.is_crit,
            "isCoc": archetype.is_coc,
            "elements": archetype.elements,
            "deadMods": archetype.dead_mods,
        },
        "keystoneGaps": {
            "user": list(char_data.keystones),
            "popular": keystone_gaps,
        },
        "slotSummary": slot_summary,
    }


class BuildEfficiencyRequest(BaseModel):
    account: str
    character: str


@app.post("/api/character/build-efficiency")
async def character_build_efficiency(req: BuildEfficiencyRequest):
    """Compute build efficiency analysis: upgrade priority, anoints, cost tiers, lineage ROI."""
    if not req.account.strip() or not req.character.strip():
        return JSONResponse(status_code=400, content={"error": "Account and character required"})
    loop = asyncio.get_running_loop()

    # Look up character (cached)
    char_data = await loop.run_in_executor(
        None, builds_client.lookup_character, req.account.strip(), req.character.strip()
    )
    if not char_data:
        return JSONResponse(status_code=404, content={"error": "Character not found"})

    # Classify build
    archetype = classify_build(char_data)
    char_class = char_data.ascendancy or char_data.char_class
    main_skill = archetype.main_skill

    # Build slot summary (reuse build-insights logic, include weakMods/deadMods for reasons)
    slot_summary = []
    if item_lookup and item_lookup.ready:
        try:
            mp = item_lookup._mod_parser
            mdb = item_lookup._mod_database
            for eq in char_data.equipment:
                if eq.slot in _SKIP_SLOTS:
                    continue
                tier_data = enrich_item_mods(eq, mp, mdb)
                all_tiers = []
                for _mk, tiers in tier_data.items():
                    for t in tiers:
                        if t is not None:
                            all_tiers.append(t)
                meaningful = [t for t in all_tiers
                              if t["weight"] >= 0.5 and t["tier_count"] <= 15]
                avg_tier = 0
                if meaningful:
                    avg_tier = round(sum(t["tier_num"] for t in meaningful) / len(meaningful), 1)
                # Weak mods (T3+) for upgrade reason text
                weak_mods = []
                for t in sorted(meaningful, key=lambda t: -t["tier_num"]):
                    if t["tier_num"] >= 3:
                        weak_mods.append({
                            "name": t["display_name"],
                            "tier": t["tier_num"],
                            "tierCount": t["tier_count"],
                        })
                    if len(weak_mods) >= 3:
                        break
                slot_dead = [dm for dm in (archetype.dead_mods or []) if dm.get("slot") == eq.slot]
                slot_summary.append({
                    "slot": eq.slot,
                    "slotDisplay": SLOT_DISPLAY.get(eq.slot, eq.slot),
                    "itemName": eq.name or eq.type_line,
                    "avgTier": avg_tier,
                    "enrichedCount": len(meaningful),
                    "weakMods": weak_mods,
                    "deadMods": [{"mod": dm["mod"], "reason": dm["reason"]} for dm in slot_dead[:2]],
                })
        except Exception as e:
            logger.debug(f"Efficiency slot summary failed: {e}")

    # Build price cache dict from unique prices across slots
    price_cache_dict = {}
    slots_to_check = set()
    for eq in char_data.equipment:
        if eq.slot not in _SKIP_SLOTS and eq.slot in SLOT_TO_UNIQUE_SLUG:
            slots_to_check.add(eq.slot)
    for slot in slots_to_check:
        try:
            prices = await loop.run_in_executor(
                None, builds_client.fetch_unique_prices, slot)
            price_cache_dict.update(prices)
        except Exception:
            pass

    # 1. Upgrade Priority
    upgrade_priority = await loop.run_in_executor(
        None, compute_upgrade_priority, char_data, slot_summary,
        builds_client, price_cache_dict
    )

    # 2. Anoint Optimizer
    current_anoint = detect_current_anoint(char_data)
    popular_anoints = await loop.run_in_executor(
        None, builds_client.fetch_popular_anoints, char_class, main_skill
    )
    anoint_optimal = False
    if current_anoint and popular_anoints:
        anoint_optimal = (popular_anoints[0]["name"].lower() == current_anoint.lower())

    # 3. Lineage Gem ROI
    lineage_gems = find_lineage_upgrades(char_data.skill_groups, price_cache_dict)

    # 4. Popular items by slot for cost tiers
    popular_by_slot = {}
    for eq in char_data.equipment:
        if eq.slot in _SKIP_SLOTS:
            continue
        try:
            items = await loop.run_in_executor(
                None, builds_client.fetch_popular_items,
                char_class, main_skill, eq.slot)
            if items:
                popular_by_slot[eq.slot] = items
        except Exception:
            pass

    # 5. Cost Tiers
    cost_tiers = compute_cost_tiers(archetype, popular_by_slot,
                                    price_cache_dict, lineage_gems)

    return {
        "upgradePriority": upgrade_priority,
        "anointOptimizer": {
            "current": current_anoint,
            "currentDesc": get_anoint_description(current_anoint) if current_anoint else None,
            "isOptimal": anoint_optimal,
            "popular": [
                {"name": a["name"], "percentage": a["percentage"],
                 "desc": get_anoint_description(a["name"])}
                for a in popular_anoints[:5]
            ],
        },
        "costTiers": cost_tiers,
        "lineageGemRoi": lineage_gems[:10],
        "archetype": {
            "defenseType": archetype.defense_type,
            "damageType": archetype.damage_type,
        },
    }


class ImprovementPackageRequest(BaseModel):
    account: str
    character: str


@app.post("/api/character/improvement-package")
async def character_improvement_package(req: ImprovementPackageRequest):
    """Compute structured improvement package: free changes, spend money, alternatives."""
    if not req.account.strip() or not req.character.strip():
        return JSONResponse(status_code=400, content={"error": "Account and character required"})
    loop = asyncio.get_running_loop()

    char_data = await loop.run_in_executor(
        None, builds_client.lookup_character, req.account.strip(), req.character.strip()
    )
    if not char_data:
        return JSONResponse(status_code=404, content={"error": "Character not found"})

    archetype = classify_build(char_data)
    char_class = char_data.ascendancy or char_data.char_class
    main_skill = archetype.main_skill

    # Reuse efficiency data
    slot_summary = []
    if item_lookup and item_lookup.ready:
        try:
            mp = item_lookup._mod_parser
            mdb = item_lookup._mod_database
            for eq in char_data.equipment:
                if eq.slot in _SKIP_SLOTS:
                    continue
                tier_data = enrich_item_mods(eq, mp, mdb)
                all_tiers = []
                for _mk, tiers in tier_data.items():
                    for t in tiers:
                        if t is not None:
                            all_tiers.append(t)
                meaningful = [t for t in all_tiers
                              if t["weight"] >= 0.5 and t["tier_count"] <= 15]
                avg_tier = 0
                if meaningful:
                    avg_tier = round(sum(t["tier_num"] for t in meaningful) / len(meaningful), 1)
                slot_summary.append({
                    "slot": eq.slot,
                    "slotDisplay": SLOT_DISPLAY.get(eq.slot, eq.slot),
                    "itemName": eq.name or eq.type_line,
                    "avgTier": avg_tier,
                    "enrichedCount": len(meaningful),
                })
        except Exception:
            pass

    price_cache_dict = {}
    for eq in char_data.equipment:
        if eq.slot not in _SKIP_SLOTS and eq.slot in SLOT_TO_UNIQUE_SLUG:
            try:
                prices = await loop.run_in_executor(
                    None, builds_client.fetch_unique_prices, eq.slot)
                price_cache_dict.update(prices)
            except Exception:
                pass

    upgrade_priority = await loop.run_in_executor(
        None, compute_upgrade_priority, char_data, slot_summary,
        builds_client, price_cache_dict
    )
    popular_anoints = await loop.run_in_executor(
        None, builds_client.fetch_popular_anoints, char_class, main_skill
    )
    lineage_gems = find_lineage_upgrades(char_data.skill_groups, price_cache_dict)

    package = compute_improvement_package(
        char_data, archetype, slot_summary, popular_anoints,
        lineage_gems, upgrade_priority
    )
    return package


class BuildCompareRequest(BaseModel):
    account: str
    character: str


@app.post("/api/character/build-compare")
async def character_build_compare(req: BuildCompareRequest):
    """Compare character build against aggregate top-build data."""
    if not req.account.strip() or not req.character.strip():
        return JSONResponse(status_code=400, content={"error": "Account and character required"})
    loop = asyncio.get_running_loop()

    char_data = await loop.run_in_executor(
        None, builds_client.lookup_character, req.account.strip(), req.character.strip()
    )
    if not char_data:
        return JSONResponse(status_code=404, content={"error": "Character not found"})

    archetype = classify_build(char_data)
    char_class = char_data.ascendancy or char_data.char_class
    main_skill = archetype.main_skill

    # Fetch popular keystones
    popular_keystones = await loop.run_in_executor(
        None, builds_client.fetch_popular_keystones, char_class, main_skill
    )

    # Fetch popular anoints
    popular_anoints = await loop.run_in_executor(
        None, builds_client.fetch_popular_anoints, char_class, main_skill
    )

    # Fetch popular items per slot
    popular_by_slot = {}
    for eq in char_data.equipment:
        if eq.slot in _SKIP_SLOTS:
            continue
        try:
            items = await loop.run_in_executor(
                None, builds_client.fetch_popular_items,
                char_class, main_skill, eq.slot)
            if items:
                popular_by_slot[eq.slot] = items
        except Exception:
            pass

    comparison = compute_build_comparison(
        char_data, popular_keystones, popular_anoints, popular_by_slot, []
    )
    return comparison


# ---------------------------------------------------------------------------
# Guide Companion endpoints
# ---------------------------------------------------------------------------
class GuideImportRequest(BaseModel):
    url: str


class GuideCompareRequest(BaseModel):
    character: Optional[dict] = None
    account: Optional[str] = None
    character_name: Optional[str] = None
    level: Optional[int] = None


class GuideStagePricesRequest(BaseModel):
    stage: str


@app.post("/api/guide/import")
async def guide_import(req: GuideImportRequest):
    """Fetch a build guide URL, parse it, save, and return summary."""
    url = req.url.strip()
    if not url:
        return JSONResponse(status_code=400, content={"error": "URL required"})
    loop = asyncio.get_running_loop()
    try:
        guide = await loop.run_in_executor(None, guide_scraper.import_guide, url)
    except ValueError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})
    except Exception as e:
        logger.error("Guide import failed: %s", e)
        return JSONResponse(status_code=500, content={"error": f"Failed to import guide: {e}"})
    return {
        "id": guide.id,
        "title": guide.title,
        "source": guide.source,
        "char_class": guide.char_class,
        "ascendancy": guide.ascendancy,
        "main_skill": guide.main_skill,
        "stages": len(guide.stages),
        "stage_list": [s.stage for s in guide.stages],
    }


@app.get("/api/guide/list")
async def guide_list():
    """Return all saved guides."""
    return guide_scraper.list_guides()


@app.get("/api/guide/{guide_id}")
async def guide_get(guide_id: str):
    """Return full parsed guide data."""
    guide = guide_scraper.load_guide(guide_id)
    if not guide:
        return JSONResponse(status_code=404, content={"error": "Guide not found"})
    return guide_scraper._guide_to_dict(guide)


@app.delete("/api/guide/{guide_id}")
async def guide_delete(guide_id: str):
    """Delete a saved guide."""
    ok = guide_scraper.delete_guide(guide_id)
    if not ok:
        return JSONResponse(status_code=404, content={"error": "Guide not found"})
    return {"ok": True}


@app.post("/api/guide/{guide_id}/compare")
async def guide_compare(guide_id: str, req: GuideCompareRequest):
    """Compare a character against a guide."""
    guide = guide_scraper.load_guide(guide_id)
    if not guide:
        return JSONResponse(status_code=404, content={"error": "Guide not found"})

    char_dict = req.character
    # If no inline character, look up via account/name
    if not char_dict and req.account and req.character_name:
        loop = asyncio.get_running_loop()
        char_data = await loop.run_in_executor(
            None, builds_client.lookup_character, req.account.strip(), req.character_name.strip()
        )
        if not char_data:
            return JSONResponse(status_code=404, content={"error": "Character not found"})
        char_dict = builds_client.serialize_character(char_data)

    if not char_dict:
        return JSONResponse(status_code=400, content={"error": "Provide character data or account+character_name"})

    if req.level:
        char_dict["level"] = req.level

    return guide_scraper.compare_character_to_guide(char_dict, guide, price_cache)


@app.post("/api/guide/{guide_id}/stage-prices")
async def guide_stage_prices(guide_id: str, req: GuideStagePricesRequest):
    """Get price estimates for all gear in a guide stage."""
    guide = guide_scraper.load_guide(guide_id)
    if not guide:
        return JSONResponse(status_code=404, content={"error": "Guide not found"})
    return guide_scraper.get_stage_prices(guide, req.stage, price_cache)


# ---------------------------------------------------------------------------
# OAuth endpoints (stash viewer authentication)
# ---------------------------------------------------------------------------
@app.post("/api/oauth/start")
async def oauth_start():
    """Initiate OAuth PKCE flow — opens browser for GGG authorization."""
    if not oauth_manager:
        return JSONResponse(status_code=503, content={"error": "OAuth not initialized"})
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, oauth_manager.authorize)
    if result.get("connected"):
        await ws_manager.broadcast({"type": "oauth_status", **oauth_manager.get_status()})
    return result


@app.get("/api/oauth/status")
async def oauth_status():
    """Return current OAuth connection status."""
    if not oauth_manager:
        return {"connected": False, "account_name": None}
    return oauth_manager.get_status()


@app.post("/api/oauth/disconnect")
async def oauth_disconnect():
    """Revoke tokens and clear OAuth connection."""
    if not oauth_manager:
        return {"status": "not_connected"}
    oauth_manager.disconnect()
    # Clear cached stash data
    stash_data["tabs"] = []
    stash_data["last_refresh"] = None
    stash_data["total_value"] = 0.0
    await ws_manager.broadcast({"type": "oauth_status", "connected": False, "account_name": None})
    return {"status": "disconnected"}


# ---------------------------------------------------------------------------
# Stash viewer endpoints
# ---------------------------------------------------------------------------
@app.get("/api/stash/status")
async def get_stash_status():
    """Return stash viewer status summary."""
    oauth = oauth_manager.get_status() if oauth_manager else {"connected": False}
    return {
        **oauth,
        "last_refresh": stash_data.get("last_refresh"),
        "tab_count": len(stash_data.get("tabs", [])),
        "total_value": stash_data.get("total_value", 0),
        "refreshing": stash_data.get("refreshing", False),
    }


@app.post("/api/stash/refresh")
async def refresh_stash():
    """Trigger a full stash fetch + score. Runs in background thread."""
    if not oauth_manager or not oauth_manager.connected:
        return JSONResponse(status_code=401, content={"error": "Not connected — use OAuth to connect first"})
    if not stash_client:
        return JSONResponse(status_code=503, content={"error": "Stash client not initialized"})
    if stash_data.get("refreshing"):
        return {"status": "already_refreshing"}

    settings = load_settings()
    league = settings.get("league", "Fate of the Vaal")
    loop = asyncio.get_running_loop()

    stash_data["refreshing"] = True

    def _run_refresh():
        try:
            # Update exchange rate for scoring
            if price_cache and stash_scorer:
                d2c = getattr(price_cache, "divine_to_chaos", 0)
                if d2c:
                    stash_scorer.set_divine_to_chaos(d2c)

            # Progress callback → WS broadcast
            def progress_cb(tab_name, done, total):
                asyncio.run_coroutine_threadsafe(
                    ws_manager.broadcast({
                        "type": "stash_progress",
                        "tab": tab_name,
                        "done": done,
                        "total": total,
                    }),
                    loop,
                )

            # Fetch all tabs
            tab_results = stash_client.fetch_all_tabs(league, progress_cb=progress_cb)

            # Score all items
            tab_summaries = []
            if stash_scorer and stash_scorer.ready:
                for tab, items in tab_results:
                    summary = stash_scorer.score_tab(tab, items)
                    tab_summaries.append(summary)
            else:
                # Without scorer, still show tab metadata
                for tab, items in tab_results:
                    tab_summaries.append(TabSummary(
                        id=tab.id, name=tab.name, type=tab.type,
                        colour=tab.colour, item_count=len(items),
                    ))

            total_value = sum(t.total_divine for t in tab_summaries)

            # Serialize tab summaries for API
            serialized_tabs = []
            for ts in tab_summaries:
                serialized_tabs.append({
                    "id": ts.id,
                    "name": ts.name,
                    "type": ts.type,
                    "colour": ts.colour,
                    "item_count": ts.item_count,
                    "scored_count": ts.scored_count,
                    "total_divine": ts.total_divine,
                    "items": [
                        {
                            "name": si.name,
                            "base_type": si.base_type,
                            "item_class": si.item_class,
                            "rarity": si.rarity,
                            "item_level": si.item_level,
                            "grade": si.grade,
                            "score": si.score,
                            "estimate_divine": si.estimate_divine,
                            "estimate_chaos": si.estimate_chaos,
                            "icon_url": si.icon_url,
                            "stack_size": si.stack_size,
                            "listed_price": si.listed_price,
                            "mods": si.mods,
                            "top_mods": si.top_mods,
                            "total_dps": si.total_dps,
                            "total_defense": si.total_defense,
                        }
                        for si in ts.items
                    ],
                })

            stash_data["tabs"] = serialized_tabs
            stash_data["total_value"] = round(total_value, 2)
            stash_data["last_refresh"] = time.strftime("%H:%M:%S")
            stash_data["refreshing"] = False

            # Save wealth snapshot
            StashScorer.save_wealth_snapshot(tab_summaries)

            # Broadcast completion
            asyncio.run_coroutine_threadsafe(
                ws_manager.broadcast({
                    "type": "stash_complete",
                    "total_value": round(total_value, 2),
                    "tab_count": len(tab_summaries),
                }),
                loop,
            )

            logger.info(f"Stash refresh complete: {total_value:.1f} divine across {len(tab_summaries)} tabs")

        except Exception as e:
            logger.error(f"Stash refresh failed: {e}")
            stash_data["refreshing"] = False
            asyncio.run_coroutine_threadsafe(
                ws_manager.broadcast({
                    "type": "stash_error",
                    "error": str(e),
                }),
                loop,
            )

    threading.Thread(target=_run_refresh, daemon=True).start()
    return {"status": "refreshing"}


@app.get("/api/stash/tabs")
async def get_stash_tabs():
    """Return stash tab summaries (without full item lists)."""
    return [
        {
            "id": t["id"],
            "name": t["name"],
            "type": t["type"],
            "colour": t["colour"],
            "item_count": t["item_count"],
            "scored_count": t.get("scored_count", 0),
            "total_divine": t.get("total_divine", 0),
        }
        for t in stash_data.get("tabs", [])
    ]


@app.get("/api/stash/tabs/{tab_id}")
async def get_stash_tab_items(tab_id: str):
    """Return scored items for a specific stash tab."""
    for tab in stash_data.get("tabs", []):
        if tab["id"] == tab_id:
            return tab
    return JSONResponse(status_code=404, content={"error": "Tab not found"})


@app.get("/api/stash/wealth-history")
async def get_wealth_history():
    """Return wealth history for sparkline display."""
    return StashScorer.load_wealth_history()


# ---------------------------------------------------------------------------
# Market Signals endpoint (coming soon — Discord integration)
# ---------------------------------------------------------------------------
@app.get("/api/market-signals")
async def get_market_signals():
    """Return market signals from trusted Discord analysts.

    Currently returns an empty feed. When the Discord bot is connected,
    this will read from a local cache of messages from #market-signals.
    Future WS event: {"type": "market_signal", "author": "...", "avatar": "...", "message": "...", "ts": ...}
    """
    return {"signals": [], "status": "coming_soon"}


# ---------------------------------------------------------------------------
# Bug report endpoint
# ---------------------------------------------------------------------------
class BugReportRequest(BaseModel):
    title: str = ""
    description: str = ""
    contact: str = ""


@app.post("/api/bug-report")
async def submit_bug_report(req: BugReportRequest):
    """Collect logs + system info and POST to Discord webhook."""

    title = req.title.strip() or f"Bug report {time.strftime('%Y-%m-%d %H:%M')}"
    description = req.description.strip()
    contact = req.contact.strip()

    # Collect data (mirrors bug_reporter.py._collect_data)
    log_tail = ""
    try:
        if LOG_FILE.exists():
            lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
            log_tail = "\n".join(lines[-BUG_REPORT_LOG_LINES:])
    except Exception as e:
        log_tail = f"(failed to read log: {e})"

    clipboards = []
    try:
        if DEBUG_DIR.exists():
            clips = sorted(
                DEBUG_DIR.glob("clipboard_*.txt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:BUG_REPORT_MAX_CLIPBOARDS]
            for clip_path in clips:
                try:
                    content = clip_path.read_text(encoding="utf-8", errors="replace")
                    clipboards.append((clip_path.name, content))
                except Exception:
                    pass
    except Exception:
        pass

    # System info
    screen_info = "unknown"
    try:
        import ctypes
        user32 = ctypes.windll.user32
        screen_info = f"{user32.GetSystemMetrics(0)}x{user32.GetSystemMetrics(1)}"
    except Exception:
        pass
    system_info = f"Python {sys.version.split()[0]}, {platform.platform()}, Screen {screen_info}"

    # Session stats from overlay
    status = overlay.get_status()
    stats = status["stats"]
    session_stats = (
        f"Uptime {status['uptime'] // 60}min, "
        f"{stats['triggers']} triggers, "
        f"{stats['prices_shown']} prices ({stats['success_rate']}%)"
    )

    # Save local record
    try:
        BUG_REPORT_DB.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": int(time.time()),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "title": title,
            "description": description,
            "system_info": system_info,
            "session_stats": session_stats,
            "contact": contact,
        }
        with open(BUG_REPORT_DB, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass

    # Build Discord message
    message = f"**Bug Report: {title}**"
    if description:
        message += f"\n{description}"
    message += f"\n\n**System:** {system_info}"
    message += f"\n**Session:** {session_stats}"
    if contact:
        message += f"\n**Contact:** {contact}"
    message += f"\n**Source:** Dashboard"
    if len(message) > 2000:
        message = message[:1997] + "..."

    # Build attachment
    parts = []
    if log_tail:
        parts.append(f"=== LOG TAIL (last {BUG_REPORT_LOG_LINES} lines) ===\n")
        parts.append(log_tail)
        parts.append("\n\n")
    for filename, content in clipboards:
        parts.append(f"=== {filename} ===\n")
        parts.append(content)
        parts.append("\n\n")
    combined = "".join(parts).encode("utf-8")

    # POST to Discord
    if not DISCORD_WEBHOOK_URL:
        logger.info("Bug report saved locally (no Discord webhook configured)")
        return {"status": "sent", "title": title, "note": "Saved locally"}

    try:
        resp = requests.post(
            DISCORD_WEBHOOK_URL,
            data={"content": message},
            files={"file": ("bug_report.txt", combined, "text/plain")},
            timeout=15,
        )
        if resp.status_code in range(200, 300):
            logger.info("Bug report sent successfully")
            return {"status": "sent", "title": title}
        else:
            logger.error(f"Bug report failed: HTTP {resp.status_code}")
            return {"error": f"Discord returned HTTP {resp.status_code}"}
    except Exception as e:
        logger.error(f"Bug report upload error: {e}")
        return {"error": "Failed to send report. Saved locally."}


# ---------------------------------------------------------------------------
# Telemetry endpoints (opt-in anonymous calibration data)
# ---------------------------------------------------------------------------
@app.post("/api/telemetry/upload")
async def telemetry_upload():
    """Manual trigger: upload pending calibration samples now."""
    if not telemetry_uploader:
        return {"error": "Telemetry not initialized"}
    settings = load_settings()
    if not settings.get("telemetry_enabled", False):
        return {"error": "Telemetry is disabled"}
    loop = asyncio.get_running_loop()
    success, reason = await loop.run_in_executor(None, telemetry_uploader.upload_now)
    if success:
        return {"status": "uploaded", "message": reason}
    return {"error": reason}


@app.get("/api/telemetry/status")
async def telemetry_status():
    """Return telemetry status for dashboard display."""
    if not telemetry_uploader:
        return {"last_upload": None, "pending_samples": 0, "enabled": False}
    status = telemetry_uploader.get_status()
    settings = load_settings()
    status["enabled"] = settings.get("telemetry_enabled", False)
    return status


# ---------------------------------------------------------------------------
# Feedback endpoint (sends to Discord webhook)
# ---------------------------------------------------------------------------
class FeedbackRequest(BaseModel):
    type: str = "feedback"  # "feedback" or "feature"
    title: str = ""
    description: str = ""


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Submit user feedback or feature request to Discord webhook."""
    kind = "Feature Request" if req.type == "feature" else "Feedback"
    title = req.title.strip() or f"{kind} {time.strftime('%Y-%m-%d %H:%M')}"
    description = req.description.strip()

    from config import APP_VERSION
    emoji = "\U0001f4a1" if req.type == "feature" else "\U0001f4ac"
    message = f"{emoji} **{kind}: {title}**"
    if description:
        message += f"\n{description}"
    message += f"\n\n**Source:** Dashboard v{APP_VERSION}"
    if len(message) > 2000:
        message = message[:1997] + "..."

    if not DISCORD_WEBHOOK_URL:
        logger.info(f"{kind} received but no Discord webhook configured")
        return {"status": "sent", "title": title, "note": "No webhook configured"}

    try:
        resp = requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": message},
            timeout=15,
        )
        if resp.status_code in range(200, 300):
            logger.info(f"{kind} sent: {title}")
            return {"status": "sent", "title": title}
        else:
            logger.error(f"{kind} failed: HTTP {resp.status_code}")
            return {"error": f"Discord returned HTTP {resp.status_code}"}
    except Exception as e:
        logger.error(f"{kind} upload error: {e}")
        return {"error": "Failed to send. Please try again later."}


# ---------------------------------------------------------------------------
# Filter items endpoint — returns items grouped by economy section and tier
# ---------------------------------------------------------------------------
_SECTION_CATEGORIES = {
    "currency": ["currency"],
    "currency->emotions": ["delirium"],
    "currency->catalysts": ["breach"],
    "currency->essence": ["essences"],
    "currency->omen": ["ritual"],
    "sockets->general": ["runes", "ultimatum", "idol", "abyss"],
    "fragments->generic": ["fragments", "vaultkeys"],
    "uniques": [
        "unique/accessory", "unique/armour", "unique/flask",
        "unique/jewel", "unique/map", "unique/weapon", "unique/sanctum",
    ],
}

_SECTION_THRESHOLD_TYPE = {
    "currency": "currency",
    "currency->emotions": "currency",
    "currency->catalysts": "currency",
    "currency->essence": "currency",
    "currency->omen": "currency",
    "sockets->general": "currency",
    "fragments->generic": "fragment",
    "uniques": "unique",
}

_CHAOS_THRESHOLDS = {
    "currency": {"s": 25.0, "a": 5.0, "b": 2.0, "c": 1.0, "d": 1.0, "e": 0.0},
    "unique": {"t1": 25.0, "t2": 3.0, "t3": 0.5, "hideable": 0.0},
    "fragment": {"a": 5.0, "b": 1.0, "c": 0.0},
}


@app.get("/api/filter-items")
async def get_filter_items():
    """Return items grouped by economy section and tier based on current prices."""
    settings = load_settings()
    league = settings.get("league", "Fate of the Vaal")
    cache_file = SETTINGS_DIR / "cache" / f"prices_{league.lower().replace(' ', '_')}.json"

    if not cache_file.exists():
        return {"items": {}, "divine_to_chaos": 0}

    try:
        with open(cache_file) as f:
            cache = json.load(f)
    except Exception:
        return {"items": {}, "divine_to_chaos": 0}

    prices = cache.get("prices", {})
    d2c = cache.get("divine_to_chaos", 68.0)

    # Convert chaos thresholds to divine
    divine_thresholds = {}
    for ttype, table in _CHAOS_THRESHOLDS.items():
        divine_thresholds[ttype] = {
            tier: (v / d2c if d2c > 0 else 0) for tier, v in table.items()
        }

    result = {}
    for section, cats in _SECTION_CATEGORIES.items():
        items_by_tier = {}
        ttype = _SECTION_THRESHOLD_TYPE[section]
        table = divine_thresholds[ttype]

        for key, data in prices.items():
            cat = data.get("category", "")
            if cat not in cats:
                continue
            dv = data.get("divine_value", 0)
            chaos = dv * d2c

            # Assign tier (highest matching threshold)
            assigned = list(table.keys())[-1]  # fallback to lowest
            for tier_name, threshold in sorted(table.items(), key=lambda x: -x[1]):
                if dv >= threshold:
                    assigned = tier_name
                    break

            if assigned not in items_by_tier:
                items_by_tier[assigned] = []
            items_by_tier[assigned].append({
                "name": data.get("name", key),
                "chaos": round(chaos, 1),
            })

        # Sort items within each tier by value descending
        for tier in items_by_tier:
            items_by_tier[tier].sort(key=lambda x: -x["chaos"])

        result[section] = items_by_tier

    return {"items": result, "divine_to_chaos": d2c}


# ---------------------------------------------------------------------------
# Filter update endpoint
# ---------------------------------------------------------------------------
@app.post("/api/update-filter")
async def update_filter():
    """Trigger a loot filter update by spawning a subprocess."""
    settings = load_settings()
    league = settings.get("league", "Fate of the Vaal")

    # Pass filter preferences via environment variable so the subprocess
    # can read them and forward to FilterUpdater.update_now()
    filter_prefs = {
        "filter_strictness": settings.get("filter_strictness", "normal"),
        "filter_tier_styles": settings.get("filter_tier_styles", {}),
        "filter_section_visibility": settings.get("filter_section_visibility", {}),
        "filter_gear_classes": settings.get("filter_gear_classes", {}),
    }

    if IS_FROZEN:
        cmd = [sys.executable, "--overlay-worker",
               "--league", league, "--test-filter-update"]
    else:
        cmd = [sys.executable, str(Path(__file__).parent / "main.py"),
               "--league", league, "--test-filter-update"]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["POE2_FILTER_PREFS"] = json.dumps(filter_prefs)

    def _run_update():
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(APP_DIR),
                env=env,
                timeout=120,
                creationflags=_HIDDEN_FLAGS, startupinfo=_HIDDEN_SI,
            )
            output = result.stdout + result.stderr
            return {"status": "completed", "output": output, "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            return {"error": "Filter update timed out after 120s"}
        except Exception as e:
            return {"error": str(e)}

    # Run in executor to avoid blocking
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _run_update)

    # Broadcast result to WebSocket clients
    if "error" not in result:
        await ws_manager.broadcast({
            "type": "log",
            "time": time.strftime("%H:%M:%S"),
            "message": "Loot filter updated successfully",
            "color": "#4a7c59",
        })
        log_buffer.append({
            "time": time.strftime("%H:%M:%S"),
            "message": "Loot filter updated successfully",
            "color": "#4a7c59",
        })
    else:
        await ws_manager.broadcast({
            "type": "log",
            "time": time.strftime("%H:%M:%S"),
            "message": f"Filter update failed: {result['error']}",
            "color": "#a83232",
        })

    return result


# ---------------------------------------------------------------------------
# App restart endpoint
# ---------------------------------------------------------------------------
@app.post("/api/restart-app")
async def restart_app():
    """Stop overlay and restart the entire app process."""
    overlay.stop()

    if IS_FROZEN:
        restart_cmd = [sys.executable, "--restart"]
    else:
        entry = Path(__file__).parent / "app.py"
        if not entry.exists():
            return {"error": "app.py not found — restart only works in standalone mode"}
        restart_cmd = [sys.executable, str(entry), "--restart"]

    # Spawn the new process FIRST — it has --restart which waits for the port
    # to be freed before binding.  This must happen before we tell the dashboard
    # to close, because closing pywebview triggers os._exit(0) in app.py which
    # would kill our daemon threads before Popen runs.
    subprocess.Popen(restart_cmd, cwd=str(APP_DIR),
                     creationflags=_HIDDEN_FLAGS, startupinfo=_HIDDEN_SI)

    # Now tell the dashboard to close the pywebview window
    await ws_manager.broadcast({"type": "app_restart"})

    def _kill_self():
        time.sleep(1.5)
        os._exit(0)

    # Use non-daemon thread so it survives even if main thread exits first
    threading.Thread(target=_kill_self, daemon=False).start()
    return {"status": "restarting"}


# ---------------------------------------------------------------------------
# One-click auto-update
# ---------------------------------------------------------------------------
@app.post("/api/apply-update")
async def apply_update():
    """Download the latest Setup exe from GitHub and launch it silently."""
    try:
        from config import APP_VERSION
        if APP_VERSION == "dev":
            return {"error": "Cannot auto-update dev builds"}
    except Exception:
        return {"error": "Cannot determine app version"}

    loop = asyncio.get_running_loop()

    # 1. Fetch the latest release to find the Setup exe asset
    try:
        gh_headers = _get_github_headers()
        resp = await loop.run_in_executor(None, lambda: requests.get(
            "https://api.github.com/repos/CouloirGG/lama/releases/latest",
            timeout=10,
            headers=gh_headers,
        ))
        if resp.status_code != 200:
            return {"error": f"GitHub API returned {resp.status_code}"}
        data = resp.json()
    except Exception as e:
        return {"error": f"Failed to fetch release info: {e}"}

    setup_url = ""
    setup_name = ""
    setup_size = 0
    for asset in data.get("assets", []):
        name = asset.get("name", "")
        if "Setup" in name and name.endswith(".exe"):
            setup_url = asset.get("url", "") or asset.get("browser_download_url", "")
            setup_name = name
            setup_size = asset.get("size", 0)
            break

    if not setup_url:
        return {"error": "No Setup exe found in latest release"}

    # 2. Download to temp dir with progress streaming
    dest = Path(tempfile.gettempdir()) / setup_name

    def _download():
        dl_headers = _get_github_headers()
        dl_headers["Accept"] = "application/octet-stream"
        r = requests.get(setup_url, stream=True, timeout=60,
                         headers=dl_headers)
        r.raise_for_status()
        total = setup_size or int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=256 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = min(int(downloaded * 100 / total), 100)
                    asyncio.run_coroutine_threadsafe(
                        ws_manager.broadcast({
                            "type": "update_progress",
                            "percent": pct,
                        }),
                        loop,
                    )
        return dest

    try:
        await ws_manager.broadcast({
            "type": "update_progress", "percent": 0,
        })
        installer_path = await loop.run_in_executor(None, _download)
    except Exception as e:
        return {"error": f"Download failed: {e}"}

    # 3. Launch installer silently and shut down
    logger.info(f"Launching installer: {installer_path}")
    await ws_manager.broadcast({
        "type": "update_progress", "percent": 100, "installing": True,
    })

    def _launch_and_exit():
        try:
            subprocess.Popen(
                [str(installer_path), "/VERYSILENT", "/FORCECLOSEAPPLICATIONS"],
                creationflags=subprocess.DETACHED_PROCESS,
            )
        except Exception as e:
            logger.error(f"Failed to launch installer: {e}")
            return
        time.sleep(0.5)
        os._exit(0)

    threading.Thread(target=_launch_and_exit, daemon=True).start()
    return {"status": "installing"}


# ---------------------------------------------------------------------------
# Dashboard serving
# ---------------------------------------------------------------------------
@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve dashboard.html for standalone app mode."""
    dashboard_path = get_resource("resources/dashboard.html")
    if not dashboard_path.exists():
        return HTMLResponse("<h1>dashboard.html not found</h1>", status_code=404)
    return HTMLResponse(
        dashboard_path.read_text(encoding="utf-8"),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/favicon.ico")
async def serve_favicon():
    """Serve favicon.ico — used by WebView2 for the taskbar icon."""
    from fastapi.responses import FileResponse
    ico_path = get_resource("resources/img/favicon.ico")
    if not ico_path.exists():
        return HTMLResponse("Not found", status_code=404)
    return FileResponse(ico_path, media_type="image/x-icon")


@app.get("/img/{filename}")
async def serve_image(filename: str):
    """Serve static images from resources/img/."""
    from fastapi.responses import FileResponse
    img_path = get_resource(f"resources/img/{filename}")
    if not img_path.exists():
        return HTMLResponse("Not found", status_code=404)
    return FileResponse(img_path, media_type="image/png")


# ---------------------------------------------------------------------------
# Companion mode endpoints (mobile app pairing)
# ---------------------------------------------------------------------------
@app.post("/api/companion/enable")
async def companion_enable():
    """Enable companion mode — generate PIN and return connection info."""
    settings = load_settings()
    pin = generate_pin()
    settings["companion_enabled"] = True
    settings["companion_pin"] = pin
    save_settings(settings)
    host = get_lan_ip()
    await ws_manager.broadcast({"type": "settings", "settings": _redact_settings(settings)})
    await ws_manager.broadcast({
        "type": "companion_status",
        "enabled": True,
        "host": host,
        "port": PORT,
        "pin": pin,
    })
    return {"host": host, "port": PORT, "pin": pin}


@app.post("/api/companion/disable")
async def companion_disable():
    """Disable companion mode — clear PIN."""
    settings = load_settings()
    settings["companion_enabled"] = False
    settings["companion_pin"] = ""
    save_settings(settings)
    await ws_manager.broadcast({"type": "settings", "settings": _redact_settings(settings)})
    await ws_manager.broadcast({"type": "companion_status", "enabled": False})
    return {"status": "disabled"}


@app.get("/api/companion/qr")
async def companion_qr():
    """Return QR code PNG for mobile pairing."""
    settings = load_settings()
    if not settings.get("companion_enabled") or not settings.get("companion_pin"):
        return JSONResponse({"error": "Companion mode not enabled"}, status_code=400)
    try:
        import qrcode
    except ImportError:
        return JSONResponse({"error": "qrcode library not installed"}, status_code=500)
    host = get_lan_ip()
    pin = settings["companion_pin"]
    payload = json.dumps({"host": host, "port": PORT, "pin": pin})
    img = qrcode.make(payload)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(buf, media_type="image/png")


@app.post("/api/companion/verify")
async def companion_verify(req: Request):
    """Verify a companion PIN — called by mobile app before connecting."""
    settings = load_settings()
    if not settings.get("companion_enabled"):
        return JSONResponse({"verified": False, "error": "Companion mode not enabled"}, status_code=403)
    pin = req.headers.get("X-LAMA-PIN", "")
    expected = settings.get("companion_pin", "")
    if not expected or pin != expected:
        return JSONResponse({"verified": False, "error": "Invalid PIN"}, status_code=401)
    from config import APP_VERSION
    return {"verified": True, "version": APP_VERSION}


@app.get("/api/companion/info")
async def companion_info():
    """Return companion mode status (no PIN unless enabled)."""
    settings = load_settings()
    enabled = settings.get("companion_enabled", False)
    result = {"enabled": enabled}
    if enabled:
        result["host"] = get_lan_ip()
        result["port"] = PORT
    return result


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    # Validate companion PIN if companion mode is enabled
    settings_raw = load_settings()
    if settings_raw.get("companion_enabled") and settings_raw.get("companion_pin"):
        pin = ws.query_params.get("pin", "")
        # Allow local connections without PIN (dashboard)
        client_host = ws.client.host if ws.client else ""
        is_local = client_host in ("127.0.0.1", "::1", "localhost")
        if not is_local and pin != settings_raw["companion_pin"]:
            await ws.accept()
            await ws.close(code=4001, reason="Invalid PIN")
            return

    await ws_manager.connect(ws)
    try:
        # Send initial state
        settings = _redact_settings(settings_raw)
        init_msg = {
            "type": "init",
            **overlay.get_status(),
            "settings": settings,
            "log": list(log_buffer),
        }
        if watchlist_worker:
            init_msg["watchlist_results"] = watchlist_worker.get_results()
            init_msg["watchlist_states"] = watchlist_worker.get_query_states()
        if oauth_manager:
            init_msg["oauth_status"] = oauth_manager.get_status()
        init_msg["stash_status"] = {
            "last_refresh": stash_data.get("last_refresh"),
            "tab_count": len(stash_data.get("tabs", [])),
            "total_value": stash_data.get("total_value", 0),
            "refreshing": stash_data.get("refreshing", False),
        }
        await ws.send_json(init_msg)
        # Keep alive — handle client messages
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await ws.send_json({"type": "pong"})
            except (json.JSONDecodeError, TypeError):
                pass
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info(f"Binding to 0.0.0.0:{PORT}")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
    )
