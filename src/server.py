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

# ---------------------------------------------------------------------------
# Sentry — error tracking
# ---------------------------------------------------------------------------
def _sentry_before_send(event, hint):
    if "extra" in event:
        for key in list(event["extra"].keys()):
            if any(s in key.lower() for s in ("token", "key", "secret", "password", "dsn")):
                event["extra"][key] = "[REDACTED]"
    return event

_sentry_dsn = os.environ.get("SENTRY_DSN", "")
if not _sentry_dsn:
    try:
        _settings_path = os.path.join(
            os.path.expanduser("~"), ".poe2-price-overlay", "dashboard_settings.json"
        )
        if os.path.exists(_settings_path):
            with open(_settings_path, "r") as f:
                _sentry_dsn = json.load(f).get("sentry_dsn", "")
    except Exception:
        pass

if _sentry_dsn:
    try:
        import sentry_sdk
        _release = "unknown"
        try:
            _release = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            pass
        sentry_sdk.init(
            dsn=_sentry_dsn,
            release=f"lama@{_release}",
            environment="backend",
            traces_sample_rate=0.1,
            before_send=_sentry_before_send,
        )
    except Exception as e:
        print(f"  Sentry: init failed ({e})")

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
from why_engine import WhyEngine
from bundle_paths import IS_FROZEN, APP_DIR, get_resource
from item_lookup import ItemLookup
from oauth import OAuthManager
from price_cache import PriceCache
from config import DEFAULT_LEAGUE, LEAGUE_OPTIONS
from game_commands import GameCommander
from character_client import CharacterClient
from stash_client import StashClient
from stash_scorer import StashScorer, TabSummary
import cloud_notify

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
    "league": DEFAULT_LEAGUE,
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
    "start_with_windows": False,
    "overlay_mode": "stars_only",
    "overlay_show_low_value": False,
    "overlay_tier_styles": {},
    "overlay_theme": "poe2",
    "nux_completed": False,
    "window_width": 1100,
    "window_height": 750,
    "window_maximized": False,
    "companion_enabled": False,
    "companion_pin": "",
    "cloud_enabled": False,
    "cloud_device_id": "",
    "cloud_secret": "",
    "cloud_relay_url": "",
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


_LEGACY_OVERLAY_KEYS = {
    "overlay_show_grade", "overlay_show_price", "overlay_show_stars",
    "overlay_show_mods", "overlay_show_dps", "overlay_display_preset",
    "overlay_pulse_style",
}


def load_settings() -> dict:
    """Load settings from disk, merging with defaults."""
    settings = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as f:
                saved = json.load(f)
            settings.update(saved)
            # Migrate legacy per-toggle overlay keys → overlay_mode
            if any(k in saved for k in _LEGACY_OVERLAY_KEYS) and "overlay_mode" not in saved:
                if saved.get("overlay_show_mods"):
                    settings["overlay_mode"] = "detailed"
                elif saved.get("overlay_show_grade", True) and saved.get("overlay_show_dps", True):
                    settings["overlay_mode"] = "standard"
                elif saved.get("overlay_show_price", True):
                    settings["overlay_mode"] = "minimal"
                else:
                    settings["overlay_mode"] = "stars_only"
            # Remove legacy keys so they don't pollute the settings
            for k in _LEGACY_OVERLAY_KEYS:
                settings.pop(k, None)
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
price_cache: Optional[PriceCache] = None
item_lookup: Optional[ItemLookup] = None
game_commander = GameCommander()

# Character viewer
builds_client = BuildsClient()
why_engine_instance = WhyEngine(builds_client)


def _lookup_character_with_fallback(account: str, character: str, force: bool = False):
    """Look up a character via poe.ninja (ladder + profile APIs).

    BuildsClient.lookup_character now tries the builds/ladder API first,
    then falls back to the profile API for non-ladder public characters.
    If both fail and the user is OAuth-connected, try GGG's API directly.
    ``force`` bypasses the local cache to re-fetch fresh from poe.ninja.
    """
    result = builds_client.lookup_character(account, character, force=force)
    if result:
        return result

    # Last resort: GGG OAuth API (only works for the authenticated user's own characters)
    if character_client and oauth_manager and oauth_manager.connected:
        logger.info(f"poe.ninja miss for {account}/{character}, trying GGG OAuth API")
        result = character_client.get_character(character)
        if result:
            if not result.account:
                result.account = account
            return result

    return None


# Stash viewer
oauth_manager: Optional[OAuthManager] = None
stash_client: Optional[StashClient] = None
character_client: Optional[CharacterClient] = None
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
    global price_cache, item_lookup
    global oauth_manager, stash_client, stash_scorer, character_client

    loop = asyncio.get_running_loop()
    overlay.set_loop(loop)

    # Background task: periodic status push
    status_task = asyncio.create_task(status_broadcast_loop())

    settings = load_settings()
    league = settings.get("league", DEFAULT_LEAGUE)

    # Auto-heal a stale challenge-league setting carried over from a past season:
    # if the saved league is neither the current default nor a permanent league,
    # fall back to the current default so pricing/meta don't load dead-league data.
    if league not in (DEFAULT_LEAGUE, "Standard", "Hardcore") and not league.startswith("Hardcore"):
        logger.info(f"Stale league setting '{league}' -> using current default '{DEFAULT_LEAGUE}'")
        league = DEFAULT_LEAGUE
        settings["league"] = league
        try:
            save_settings(settings)
        except Exception:
            pass

    # Configure cloud push notifications
    cloud_notify.configure(
        device_id=settings.get("cloud_device_id", ""),
        secret=settings.get("cloud_secret", ""),
        relay_url=settings.get("cloud_relay_url", ""),
        enabled=settings.get("cloud_enabled", False),
    )

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

    # Prefetch the GGG passive-tree export (data + sprite atlases) so the first
    # tree view renders instantly instead of cold-downloading ~5MB.
    def _prefetch_tree2():
        try:
            import tree2
            tree2.ensure_assets()
        except Exception as e:
            logger.warning(f"tree2 prefetch failed: {e}")
    threading.Thread(target=_prefetch_tree2, daemon=True).start()

    # Background task: check for updates after a short delay
    update_task = asyncio.create_task(check_for_updates())

    # Sync auto-start setting with actual registry state
    actual_autostart = get_autostart()
    if settings.get("start_with_windows", False) != actual_autostart:
        settings["start_with_windows"] = actual_autostart
        save_settings(settings)

    # Initialize OAuth + Stash viewer
    oauth_manager = OAuthManager()
    stash_client = StashClient(oauth_manager)
    character_client = CharacterClient(oauth_manager)
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
        if price_cache:
            price_cache.stop()
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


from fastapi.exceptions import RequestValidationError
@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logger.warning("422 Validation Error on %s %s — body: %s — errors: %s",
                    request.method, request.url.path, body.decode(errors="replace"), exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


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
    player_budget_div: Optional[float] = None   # player's spendable divine budget
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
    start_with_windows: Optional[bool] = None
    overlay_mode: Optional[str] = None
    overlay_show_low_value: Optional[bool] = None
    overlay_tier_styles: Optional[dict] = None
    overlay_theme: Optional[str] = None
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
    league = req.league or settings.get("league", DEFAULT_LEAGUE)
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
    league = req.league or settings.get("league", DEFAULT_LEAGUE)
    no_filter = req.no_filter_update if req.no_filter_update is not None else settings.get("no_filter_update", False)

    result = overlay.start(league, no_filter_update=no_filter)
    if "error" not in result:
        await ws_manager.broadcast({"type": "state_change", "state": "running"})

    return result


def _redact_settings(settings: dict) -> dict:
    """Return settings with sensitive fields redacted for API responses."""
    out = dict(settings)
    out.pop("companion_pin", None)
    if out.get("cloud_secret"):
        out["cloud_secret"] = ""
    return out


@app.get("/api/settings")
async def get_settings():
    return _redact_settings(load_settings())


@app.post("/api/settings")
async def update_settings(req: SettingsRequest):
    settings = load_settings()
    updates = req.model_dump(exclude_none=True)
    # Keys that should be replaced wholesale (client sends full object, not partial)
    REPLACE_KEYS = {"filter_tier_styles", "filter_gear_classes", "overlay_tier_styles"}
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
        new_league = settings.get("league", DEFAULT_LEAGUE)
        price_cache.league = new_league

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
        # Fallback (used only when the poe2scout league fetch fails)
        return {
            "leagues": LEAGUE_OPTIONS,
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
        None, _lookup_character_with_fallback, req.account.strip(), req.character.strip()
    )
    if not char_data:
        return JSONResponse(status_code=404, content={"error": "Character not found. Make sure the account and character names are correct. Non-ladder characters require OAuth login."})

    # Auto-save to recent characters
    _save_recent_character(
        req.account.strip(), char_data.name,
        char_data.ascendancy or char_data.char_class, char_data.level,
    )

    result = builds_client.serialize_character(char_data)
    _enrich_equipment(char_data, result)
    return result


class SmartLookupRequest(BaseModel):
    query: str


def _parse_ninja_url(url: str):
    """Extract account + character from a poe.ninja profile or builds URL.

    Formats:
      .../profile/Account-1234/league/character/CharName
      .../profile/Account-1234/character/CharName
      .../builds/league/character/Account-1234/CharName   (build-explorer link)
    """
    import re
    u = url.strip()
    # Builds explorer: /poe2/builds/{league}/character/{account}/{character}
    # NOTE: order differs from profile URLs — league precedes "character",
    # and the account comes AFTER the "character" segment.
    m = re.match(
        r"https?://poe\.ninja/poe2/builds/[^/]+/character/([^/?#]+)/([^/?#]+)",
        u,
    )
    if m:
        return m.group(1), m.group(2)
    # Profile with league segment
    m = re.match(
        r"https?://poe\.ninja/poe2/profile/([^/]+)/[^/]+/character/([^/?#]+)",
        u,
    )
    if m:
        return m.group(1), m.group(2)
    # Profile without league segment
    m = re.match(
        r"https?://poe\.ninja/poe2/profile/([^/]+)/character/([^/?#]+)",
        u,
    )
    if m:
        return m.group(1), m.group(2)
    return None, None


@app.post("/api/character/smart-lookup")
async def smart_character_lookup(req: SmartLookupRequest):
    """Flexible character lookup: accepts URLs, account/character, or partial names.

    Input formats:
      - poe.ninja URL → extract account + character
      - "account / character" or "account character" → direct lookup
      - partial text → fuzzy-match against saved characters, then try as character name
    """
    q = req.query.strip()
    if not q:
        return JSONResponse(status_code=400, content={"error": "Enter a character name, account/character, or poe.ninja URL"})

    loop = asyncio.get_running_loop()

    # 1. poe.ninja URL
    if "poe.ninja" in q:
        acct, char = _parse_ninja_url(q)
        if acct and char:
            char_data = await loop.run_in_executor(
                None, _lookup_character_with_fallback, acct, char
            )
            if char_data:
                _save_recent_character(acct, char_data.name,
                                       char_data.ascendancy or char_data.char_class, char_data.level)
                result = builds_client.serialize_character(char_data)
                _enrich_equipment(char_data, result)
                return result
            return JSONResponse(status_code=404, content={"error": f"Character not found: {char} (account: {acct})"})
        return JSONResponse(status_code=400, content={"error": "Could not parse poe.ninja URL. Expected format: poe.ninja/poe2/profile/Account/league/character/Name"})

    # 2. "account / character" or "account, character"
    for sep in ["/", ","]:
        if sep in q:
            parts = [p.strip() for p in q.split(sep, 1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                char_data = await loop.run_in_executor(
                    None, _lookup_character_with_fallback, parts[0], parts[1]
                )
                if char_data:
                    _save_recent_character(parts[0], char_data.name,
                                           char_data.ascendancy or char_data.char_class, char_data.level)
                    result = builds_client.serialize_character(char_data)
                    _enrich_equipment(char_data, result)
                    return result
                return JSONResponse(status_code=404, content={"error": f"Character not found: {parts[1]} (account: {parts[0]})"})

    # 3. Single term — fuzzy match saved characters, then try as character name across saved accounts
    settings = load_settings()
    saved = settings.get("saved_characters", [])
    q_lower = q.lower()

    # Check for exact or fuzzy character name match in saved list
    for acct_group in saved:
        acct_name = acct_group.get("accountName", "")
        for ch in acct_group.get("characters", []):
            if ch["name"].lower() == q_lower:
                char_data = await loop.run_in_executor(
                    None, _lookup_character_with_fallback, acct_name, ch["name"]
                )
                if char_data:
                    _save_recent_character(acct_name, char_data.name,
                                           char_data.ascendancy or char_data.char_class, char_data.level)
                    result = builds_client.serialize_character(char_data)
                    _enrich_equipment(char_data, result)
                    return result

    # Check if query matches an account name — return suggestions
    acct_matches = []
    for acct_group in saved:
        acct_name = acct_group.get("accountName", "")
        if q_lower in acct_name.lower():
            for ch in acct_group.get("characters", []):
                acct_matches.append({"account": acct_name, "name": ch["name"],
                                     "class": ch.get("class", ""), "level": ch.get("level", 0)})

    # Check if query partially matches any character name
    char_matches = []
    for acct_group in saved:
        acct_name = acct_group.get("accountName", "")
        for ch in acct_group.get("characters", []):
            if q_lower in ch["name"].lower() and ch["name"].lower() != q_lower:
                char_matches.append({"account": acct_name, "name": ch["name"],
                                     "class": ch.get("class", ""), "level": ch.get("level", 0)})

    suggestions = acct_matches + char_matches
    if suggestions:
        return JSONResponse(status_code=300, content={
            "error": "Multiple matches found. Select one:",
            "suggestions": suggestions,
        })

    return JSONResponse(status_code=404, content={
        "error": f"No character found for \"{q}\". Try: account/character, a poe.ninja URL, or a saved character name."
    })


def _enrich_equipment(char_data, result: dict):
    """Add mod tier data to equipment in result dict."""
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

    # Find or create account (normalize # ↔ - for poe.ninja compatibility)
    acct = None
    normalized = account.lower().replace("#", "-")
    for a in saved:
        if a["accountName"].lower().replace("#", "-") == normalized:
            acct = a
            break

    if acct:
        # Heal the stored account name to the one that just resolved. poe.ninja's
        # profile lookup is case-sensitive, and its data returns accounts in a
        # different case (e.g. "angrysmash#4212") than its URLs ("AngrySMASH-4212"),
        # so older saves can hold a name that no longer looks up. Overwrite it with
        # the account that just succeeded so the saved-character click keeps working.
        acct["accountName"] = account
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


@app.post("/api/character/budget-plan")
async def character_budget_plan(req: CharacterLookupRequest):
    """Budget planner: the costable (unique) upgrades top players run that you
    don't already have, priced via the live economy and ranked by adoption.
    Honest about un-priceable rare-gear slots (those go to 'check trade')."""
    if not req.account.strip() or not req.character.strip():
        return JSONResponse(status_code=400, content={"error": "Account and character required"})
    loop = asyncio.get_running_loop()

    char = await loop.run_in_executor(
        None, _lookup_character_with_fallback, req.account.strip(), req.character.strip()
    )
    if not char:
        return JSONResponse(status_code=404, content={"error": "Character not found. Non-ladder characters require OAuth login."})

    owned = {(e.name or "").lower() for e in (char.equipment or []) if getattr(e, "name", "")}

    SLOTS = ["Helm", "BodyArmour", "Gloves", "Boots", "Belt", "Amulet", "Ring", "Weapon"]
    # Bounded concurrency: 8 simultaneous cold fetches rate-limit poe.ninja and
    # return thin results, so cap at 3 in flight (the retry helper handles 429s).
    sem = asyncio.Semaphore(3)
    async def _fetch_slot(s):
        async with sem:
            try:
                return await loop.run_in_executor(None, builds_client.get_popular_items_for_slot, char, s)
            except Exception:
                return None
    results = await asyncio.gather(*[_fetch_slot(s) for s in SLOTS])

    upgrades, seen, rare_slots = [], set(), set()
    for res in results:
        if not isinstance(res, dict):
            continue
        slot, slot_display = res.get("slot", ""), res.get("slotDisplay", "")
        for it in res.get("items", []):
            name = it.get("name", "")
            if not name:
                continue
            if (it.get("rarity") or "").lower() != "unique":
                if slot_display:
                    rare_slots.add(slot_display)
                continue
            key = name.lower()
            if key in owned or key in seen:
                continue
            pd = price_cache.lookup(name, "", 0) if price_cache else None
            if not pd or not pd.get("divine_value"):
                continue
            seen.add(key)
            upgrades.append({
                "slot": slot, "slotDisplay": slot_display, "name": name,
                "priceDiv": round(pd["divine_value"], 3),
                "priceDisplay": pd.get("display", ""),
                "priceTier": pd.get("tier", ""),
                "quantity": pd.get("quantity", 0),
                "adoption": round(it.get("percentage", 0), 1),
            })

    # Rank by adoption (impact proxy); the client adds the budget cap + tiers.
    upgrades.sort(key=lambda u: (-u["adoption"], u["priceDiv"]))

    stats = price_cache.get_stats() if price_cache else {}
    return {
        "upgrades": upgrades,
        "rates": {
            "divToChaos": round(stats.get("divine_to_chaos", 0) or 0, 1),
            "divToExalted": round(stats.get("divine_to_exalted", 0) or 0, 0),
        },
        "pricedCount": len(upgrades),
        "rareSlots": sorted(rare_slots),
        "league": stats.get("league", ""),
    }


@app.post("/api/character/popular-items")
async def character_popular_items(req: PopularItemsRequest):
    """Fetch popular items for a slot, relative to a character's class/skill."""
    if not req.account.strip() or not req.character.strip() or not req.slot.strip():
        return JSONResponse(status_code=400, content={"error": "Account, character, and slot required"})
    loop = asyncio.get_running_loop()

    # Look up character first (cached)
    char_data = await loop.run_in_executor(
        None, _lookup_character_with_fallback, req.account.strip(), req.character.strip()
    )
    if not char_data:
        return JSONResponse(status_code=404, content={"error": "Character not found. Non-ladder characters require OAuth login."})

    # Fetch popular items for the slot
    result = await loop.run_in_executor(
        None, builds_client.get_popular_items_for_slot, char_data, req.slot.strip()
    )

    # Enrich rare items with popular mod breakdown
    slot = req.slot.strip()
    archetype = classify_build(char_data)
    char_class = char_data.ascendancy or char_data.char_class
    skill = archetype.main_skill

    try:
        rare_mods = await loop.run_in_executor(
            None, builds_client.fetch_popular_rare_mods, char_class, skill, 5
        )
        slot_mods = rare_mods.get(slot, [])

        if slot_mods:
            import re
            player_item = next((eq for eq in char_data.equipment if eq.slot == slot), None)
            player_mod_norms = set()
            if player_item:
                from why_engine import _strip_ninja_brackets
                for mod in (player_item.explicit_mods or []) + (player_item.implicit_mods or []):
                    clean = _strip_ninja_brackets(mod)
                    player_mod_norms.add(re.sub(r"[\d,.]+", "#", clean).strip())

            # Result is a dict with "items" key
            items_list = result.get("items", []) if isinstance(result, dict) else result
            for item in items_list:
                if isinstance(item, dict) and item.get("rarity") != "unique":
                    item["topMods"] = [
                        {
                            "name": mod_norm.replace("#", "X"),
                            "pct": round(pct),
                            "hasIt": mod_norm in player_mod_norms,
                        }
                        for mod_norm, pct in slot_mods[:6]
                        if pct >= 20
                    ]
    except Exception as e:
        logger.debug(f"Rare mod enrichment failed: {e}")

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
        None, _lookup_character_with_fallback, req.account.strip(), req.character.strip()
    )
    if not char_data:
        return JSONResponse(status_code=404, content={"error": "Character not found. Non-ladder characters require OAuth login."})

    # Classify build
    archetype = classify_build(char_data)

    # Persist archetype for overlay consumption (build-aware scoring)
    try:
        settings = load_settings()
        settings["build_archetype"] = archetype.to_dict()
        _save_settings(settings)
    except Exception as e:
        logger.debug(f"Failed to persist build archetype: {e}")

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

                # Gap analysis: what high-value mods are missing for this build?
                gap_analysis = []
                try:
                    present_groups = [t["group"] for t in all_tiers if t.get("group")]
                    # Resolve item class from type_line
                    gap_item_class = eq.type_line.split(" of ")[0].split(" Of ")[0].strip()
                    # Use the base item class category (e.g. "Gloves", "Amulet")
                    from item_parser import BASE_TYPE_TO_CLASS
                    gap_class = BASE_TYPE_TO_CLASS.get(gap_item_class, eq.slot)
                    # Map common slot names to item classes
                    _SLOT_TO_CLASS = {
                        "Helm": "Helmets", "BodyArmour": "Body Armours",
                        "Gloves": "Gloves", "Boots": "Boots",
                        "Amulet": "Amulets", "Ring": "Rings", "Ring2": "Rings",
                        "Belt": "Belts", "Weapon": "Weapons", "Weapon2": "Weapons",
                        "Offhand": "Shields",
                    }
                    gap_class = _SLOT_TO_CLASS.get(eq.slot, gap_class)
                    gap_analysis = mdb.compute_gap_analysis(
                        gap_class, present_groups, archetype=archetype)
                except Exception as e:
                    logger.debug(f"Gap analysis failed for {eq.slot}: {e}")

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
                    "gapAnalysis": gap_analysis[:5],
                })
        except Exception as e:
            logger.debug(f"Slot summary failed: {e}")

    # Anti-synergy detection
    anti_synergies = []
    try:
        from builds_client import detect_anti_synergies
        anti_synergies = detect_anti_synergies(
            archetype, set(char_data.keystones), char_data.equipment)
        # Promote critically missing keystones (>70% adoption) as anti-synergy
        for kg in keystone_gaps:
            if kg["percentage"] >= 70 and not kg["hasIt"]:
                anti_synergies.append({
                    "id": f"missing_keystone_{kg['name'].lower().replace(' ', '_')}",
                    "name": f"Missing popular keystone: {kg['name']}",
                    "severity": "info",
                    "message": (f"{kg['percentage']}% of top {archetype.main_skill} "
                                f"builds use {kg['name']}"),
                })
    except Exception as e:
        logger.debug(f"Anti-synergy detection failed: {e}")

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
            "level": archetype.level,
        },
        "keystoneGaps": {
            "user": list(char_data.keystones),
            "popular": keystone_gaps,
        },
        "slotSummary": slot_summary,
        "antiSynergies": anti_synergies,
    }


class WhyInsightsRequest(BaseModel):
    account: str
    character: str


@app.post("/api/character/why-insights")
async def character_why_insights(req: WhyInsightsRequest):
    """Generate plain-language explanations for a character's build choices."""
    if not req.account.strip() or not req.character.strip():
        return JSONResponse(status_code=400, content={"error": "Account and character required"})
    loop = asyncio.get_running_loop()

    char_data = await loop.run_in_executor(
        None, _lookup_character_with_fallback, req.account.strip(), req.character.strip()
    )
    if not char_data:
        return JSONResponse(status_code=404, content={"error": "Character not found"})

    archetype = classify_build(char_data)

    try:
        explanations = await loop.run_in_executor(
            None, why_engine_instance.explain_character, char_data, archetype
        )
        result = explanations.to_dict()
        # Tag each recommended (missing) item with its cost tier so the gear list
        # can show Free / Cheap / Chase, consistent with the coach.
        TIER_ORDER = {"free": 0, "cheap": 1, "chase": 2}
        for cat in (result.get("synergyMap") or []):
            for m in (cat.get("missing") or []):
                t = _cost_tier(m)
                m["costTier"] = t["tier"]
                m["costLabel"] = t["cost"]
            # Affordability-first: free/cheap before chase, then by impact.
            cat["missing"] = sorted(
                cat.get("missing") or [],
                key=lambda m: (TIER_ORDER.get(m.get("costTier"), 9), -(m.get("estimatedPct") or 0)))
        result["whatIf"] = _build_whatif(result)
        result["supportingStats"] = _supporting_stats(char_data)
        return result
    except Exception as e:
        logger.error(f"Why-engine failed: {e}")
        return {"keystones": [], "gear": {}, "stats": [], "actions": [], "meta": []}


@app.post("/api/character/refresh")
async def character_refresh(req: WhyInsightsRequest):
    """Force a fresh poe.ninja re-fetch (bypassing LAMA's cache) and drop the
    cached coach, so recent in-game changes show up. Note: LAMA is only as fresh
    as poe.ninja, which snapshots characters periodically — not live."""
    if not req.account.strip() or not req.character.strip():
        return JSONResponse(status_code=400, content={"error": "Account and character required"})
    loop = asyncio.get_running_loop()
    char = await loop.run_in_executor(
        None, _lookup_character_with_fallback, req.account.strip(), req.character.strip(), True)
    if not char:
        return JSONResponse(status_code=404, content={"error": "Character not found"})
    prefix = f"{req.account.strip().lower()}|{req.character.strip().lower()}|"
    with _coach_cache_lock:
        for k in [k for k in list(_coach_cache) if k.startswith(prefix)]:
            _coach_cache.pop(k, None)
    return {"ok": True, "name": char.name}


# ── AI Coach (grounded local LLM via Ollama) ──────────────────────────────
# Principle: LAMA computes the analysis AND ranks the priorities deterministically;
# the local model only EXPLAINS them in plain language. The model never decides
# priority or invents numbers — that keeps it accurate and hallucination-free.
OLLAMA_URL = "http://localhost:11434"
COACH_MODEL = "phi4:latest"

COACH_SYSTEM = (
    "You are LAMA, a friendly, concise Path of Exile 2 build coach. Below are FACTS about "
    "the player's character, computed by LAMA from live ladder data and the current economy "
    "— they are authoritative.\n"
    "RULES:\n"
    "- Use ONLY the facts given. NEVER invent or name a specific item, unique, flask, gem, "
    "keystone, passive, price, number, or mechanic that is not in the facts.\n"
    "- This is Path of Exile 2, NOT Path of Exile 1; never reference PoE1-only items or mechanics.\n"
    "- RESOURCE & DEFENCE GATES come first. If a FACT shows a mana/life-sustain deficit (spending the "
    "resource faster than it recovers), an Energy Shield with no leech/regen, or a Chaos Inoculation "
    "build that can't sustain or grow its ES, that is a HARD gate — address it before any damage "
    "advice, and do NOT tell the player to add attack speed or more DPS until it's fixed (that makes "
    "it worse). You can't deal damage when you're out of mana or dead. When a SUPPORTING STAT fact "
    "says a fix WON'T help (e.g. 'faster ES recharge nodes don't fix this'), never recommend that fix.\n"
    "- MONEY MATTERS — assume the player is on a tight budget (a few divine at most). ALWAYS "
    "lead with the FREE and CHEAP changes (passive/gem swaps, capping resistances, improving "
    "gear they already own). Cover those first and explain the impact.\n"
    "- Expensive 'chase' items are LONG-TERM GOALS only. Mention them last, briefly, and ALWAYS "
    "with their cost (e.g. 'down the line, the ~700 div Headhunter'). NEVER open with an "
    "expensive item and never imply the player should buy it now.\n"
    "- LAMA has already ranked the priorities by impact AND affordability; coach in that order.\n"
    "- If a FUNDING fact is given, close by telling the player HOW to start earning the currency "
    "for the chase goal — name the farming method and its rough income from that fact. Never "
    "invent a farming strategy not in the facts.\n"
    "Write 4-6 short sentences straight to the player: lead with the free/cheap priority #1 and "
    "WHY, cover the next free/cheap ones naming only the facts' items/stats, then name the chase "
    "goal with its cost as something to work toward and (if given) how to fund it. End with one "
    "line of encouragement. Plain friendly prose only — no markdown headers or bullet lists."
)


# Cost classification for recommendations -----------------------------------
GEAR_SLOTS = {"helm", "bodyarmour", "gloves", "boots", "belt", "amulet",
              "ring", "ring2", "weapon", "weapon2", "quiver", "shield", "focus"}
CHEAP_DIV_MAX = 5.0   # a priced unique at/under this many divine counts as 'cheap'


def _cost_tier(m: dict, budget: float = 0.0) -> dict:
    """Classify a missing recommendation by cost: free / cheap / chase.

    When the player gives a budget, anything they can actually afford counts as
    'cheap' and only what's out of reach is a 'chase' item."""
    name = m.get("name") or ""
    nl = name.lower()
    slot = (m.get("slot") or "").lower()
    # Free: only passive/keystone tree changes (a respec, genuinely no cost).
    # NB: PoE2 "Support: X" gems are TRADEABLE items — some (Rakiata's Flow etc.)
    # are very expensive — so they are NOT free.
    if m.get("sourceType") == "keystone" or "passive" in slot:
        return {"tier": "free", "cost": "free"}
    # Improving gear the player already owns (craft/buy a better-rolled rare) — affordable.
    if "missing key mods" in nl or (not slot and "(" in name):
        return {"tier": "cheap", "cost": "cheap — improve gear you already own"}
    # Price it against the live economy (strip a "Support:" prefix to the item name).
    lookup_name = name.split(":", 1)[1].strip() if nl.startswith("support:") else name
    div = None
    if price_cache:
        try:
            pd = price_cache.lookup(lookup_name, "", 0)
            if pd and pd.get("divine_value"):
                div = round(pd["divine_value"], 2)
        except Exception:
            div = None
    threshold = budget if budget and budget > 0 else CHEAP_DIV_MAX
    if div is not None:
        return {"tier": "cheap" if div <= threshold else "chase", "cost": f"~{div} div", "div": div}
    # Unpriced but tradeable (a support gem or a unique gear slot): can't call it
    # free or guess a tier — flag it honestly as something to price-check.
    if nl.startswith("support:") or slot in GEAR_SLOTS:
        return {"tier": "chase", "cost": "tradeable — check price"}
    return {"tier": "cheap", "cost": "minor upgrade"}


def _classify_missing(synergy: list, budget: float = 0.0) -> tuple:
    """Flatten synergy 'missing' items into (free, cheap, chase) buckets, dropping
    support-gem suggestions essentially no top build actually runs (noise)."""
    free, cheap, chase = [], [], []
    for cat in synergy:
        label = cat.get("label") or ""
        for m in (cat.get("missing") or []):
            name = m.get("name") or ""
            if not name:
                continue
            adopt = m.get("adoptionPct") or 0
            if name.lower().startswith("support:") and adopt < 10:
                continue
            t = _cost_tier(m, budget)
            rec = {"name": name, "cat": label, "impact": m.get("estimatedPct") or 0,
                   "adopt": adopt, "cost": t["cost"], "div": t.get("div")}
            (free if t["tier"] == "free" else cheap if t["tier"] == "cheap" else chase).append(rec)
    for b in (free, cheap, chase):
        b.sort(key=lambda r: (-(r["impact"] or 0), -(r["adopt"] or 0)))
    return free, cheap, chase


def _build_whatif(exp: dict) -> dict:
    """Estimated EHP/DPS/resists if the player applies the recommendations.

    DPS/EHP are ESTIMATES — the summed impact %s (capped for diminishing
    returns, and DPS bounded by the build's ceiling). Resist capping is the
    stated goal of the 'cap your resists' recommendation, so it's exact-ish."""
    sc = exp.get("scorecard", {}) or {}
    dps_pct = ehp_pct = 0.0
    for cat in (exp.get("synergyMap") or []):
        ck = cat.get("category")
        for m in (cat.get("missing") or []):
            p = m.get("estimatedPct") or 0
            if ck == "dps":
                dps_pct += p
            elif ck == "survival":
                ehp_pct += p
    dps_pct = min(dps_pct, 60.0)   # cap the aggregate — impacts overlap
    ehp_pct = min(ehp_pct, 40.0)
    cur_dps = sc.get("dps") or 0
    cur_ehp = sc.get("ehp") or 0
    ceiling = sc.get("dpsCeiling") or 0
    proj_dps = cur_dps * (1 + dps_pct / 100)
    if ceiling and proj_dps > ceiling:
        proj_dps = ceiling
    proj_ehp = cur_ehp * (1 + ehp_pct / 100)
    will_cap = bool(sc.get("resistStatus") and sc.get("resistStatus") != "positive")
    return {
        "dps": {"current": cur_dps, "projected": round(proj_dps), "deltaPct": round(dps_pct)},
        "ehp": {"current": cur_ehp, "projected": round(proj_ehp), "deltaPct": round(ehp_pct)},
        "resists": {"current": sc.get("resistSummary", ""),
                    "projected": "All capped" if will_cap else sc.get("resistSummary", ""),
                    "willCap": will_cap},
    }


_TREE_NODE_INDEX = None


def _tree_node_index() -> dict:
    """id -> {name, stats, keystone, notable} from the GGG tree export (cached)."""
    global _TREE_NODE_INDEX
    if _TREE_NODE_INDEX is not None:
        return _TREE_NODE_INDEX
    idx = {}
    try:
        import json
        p = os.path.join(os.path.dirname(__file__), "..", "resources", "data", "tree2", "data.json")
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        nodes = d.get("nodes", d)
        it = nodes.items() if isinstance(nodes, dict) else ((v.get("id"), v) for v in nodes)
        for nid, v in it:
            if isinstance(v, dict):
                idx[str(nid)] = {
                    "name": v.get("name") or "",
                    "stats": v.get("stats") or [],
                    "keystone": bool(v.get("isKeystone")),
                    "notable": bool(v.get("isNotable")),
                }
    except Exception as e:
        logger.debug(f"tree node index failed: {e}")
    _TREE_NODE_INDEX = idx
    return idx


# Tree text patterns that mark a node as carrying a real drawback (not just an upside).
_DRAWBACK_PATTERNS = ("you have no ", "deal no ", "no inherent ", "cannot ", "maximum life is 1",
                      "bypass", "50% more mana cost", "more mana cost of skills")


def _allocated_drawbacks(char) -> list:
    """Allocated keystones/notables that carry a build-defining tradeoff, read from
    GGG's own tree text. Includes nodes that TRANSFORM your gear's modifiers (e.g.
    Way of the Stonefist), since those can silently strip a stat that was on the gear."""
    out = []
    try:
        from pob_decoder import decode_pob_code
        pob = decode_pob_code(char.pob_code) if getattr(char, "pob_code", None) else None
        if not pob or not getattr(pob, "passive_nodes", None):
            return out
        idx = _tree_node_index()
        for nid in (str(n) for n in pob.passive_nodes):
            info = idx.get(nid)
            if not info or not (info["keystone"] or info["notable"]):
                continue
            lines, transform = [], False
            for s in info["stats"]:
                sl = s.lower()
                if any(p in sl for p in _DRAWBACK_PATTERNS):
                    lines.append(s.replace("\n", " "))
                if "transformed" in sl and "modifier" in sl:
                    transform = True
            if lines or transform:
                out.append({"name": info["name"], "drawbacks": lines,
                            "transforms_gear": transform, "keystone": info["keystone"]})
    except Exception as e:
        logger.debug(f"allocated drawbacks failed: {e}")
    return out


def _blame(drawbacks: list, keyword: str) -> str:
    """If an allocated node explicitly causes (or could strip) a `keyword` stat, name it."""
    for d in drawbacks:
        for line in d["drawbacks"]:
            if keyword in line.lower():
                return f" Likely cause: you've allocated {d['name']} — \"{line}\"."
    # The "rewrites your gear" fallback only makes sense for stats that live as
    # explicit gear mods (mana/life regen). ES sustain is a recharge mechanic, not
    # something a glove transform plausibly caused — don't speculate there.
    if keyword in ("mana", "life"):
        for d in drawbacks:
            if d.get("transforms_gear"):
                return (f" Likely cause: {d['name']} rewrites your gear's explicit modifiers, which can "
                        f"strip {keyword} that was on that gear — re-check that gear's mods.")
    return ""


def _supporting_stats(char) -> list:
    """DPS-supporting-stat problems from PoB: you can't deal damage if you run
    out of resource. Detects mana/life sustain deficits."""
    out = []
    try:
        from pob_decoder import decode_pob_code
        pob = decode_pob_code(char.pob_code) if getattr(char, "pob_code", None) else None
        if not pob:
            return out
        a = pob.stats.all_stats or {}
        drawbacks = _allocated_drawbacks(char)

        def g(k):
            try:
                return float(a.get(k, 0) or 0)
            except Exception:
                return 0.0

        # Mana sustain: cost/s vs regen + leech. A deficit is a GATE — you can't
        # deal damage when you're out of mana, and faster attacks drain it faster.
        cost = g("ManaPerSecondCost")
        regen = g("ManaRegenRecovery") + g("ManaLeechGainRate")
        pool = g("ManaUnreserved") or g("Mana")
        if cost > 0 and cost > regen * 1.05:
            secs = pool / (cost - regen) if cost > regen else 999
            ratio = (cost / regen) if regen > 0 else 99
            out.append({
                "label": "Mana sustain",
                "severity": "critical" if (ratio >= 1.8 or secs < 4) else "warning",
                "gate": True,
                "summary": (f"You spend ~{round(cost)} mana/s but only recover ~{round(regen)}/s "
                            f"(~{ratio:.1f}x your regen) — about {secs:.0f}s of attacking before you're dry. "
                            f"Add mana regeneration, mana leech, or reduce the skill's mana cost. Fix this BEFORE "
                            f"chasing more DPS — faster attacks just drain mana faster." + _blame(drawbacks, "mana")),
            })

        # Life-cost sustain (life-cost / blood-magic style skills).
        lcost = g("LifePerSecondCost")
        lregen = g("LifeRegenRecovery") + g("LifeLeechGainRate")
        if lcost > 0 and lcost > lregen * 1.05:
            lratio = (lcost / lregen) if lregen > 0 else 99
            lpool = g("LifeUnreserved") or g("Life")
            lsecs = lpool / (lcost - lregen) if lcost > lregen else 999
            out.append({
                "label": "Life sustain",
                "severity": "critical" if (lratio >= 1.8 or lsecs < 4) else "warning",
                "gate": True,
                "summary": (f"You spend ~{round(lcost)} life/s casting but only recover ~{round(lregen)}/s "
                            f"(~{lratio:.1f}x) — risky on a life-cost build. Add life leech/regen, or more max "
                            f"life to cast and survive. Fix this before chasing more DPS." + _blame(drawbacks, "life")),
            })

        # --- Defensive coherence: do your survival layers actually hold up? ---
        # Stat-by-stat advice can quietly steer a player into an incoherent
        # defence (e.g. take CI without the ES to back it). Catch the classics.
        life = float(getattr(pob.stats, "life", 0) or 0)
        es = float(getattr(pob.stats, "energy_shield", 0) or 0)
        es_leech = g("EnergyShieldLeechGainRate")
        es_regen = g("EnergyShieldRegenRecovery")
        level = float(getattr(char, "level", 0) or 0)
        is_ci = life <= 5 and es > 0  # Chaos Inoculation sets max life to 1

        # ES is a primary defence (CI, or ES is the bigger pool) but has no
        # in-combat sustain. Recharge only starts after ~2s of NOT being hit, so
        # under sustained fire it never refills. This is the CI trap, and "faster
        # ES recharge" nodes do nothing for it.
        es_primary = es > 0 and (is_ci or es >= max(life, 1))
        if es_primary and es_leech <= 0 and es_regen <= 0:
            if is_ci:
                lead = (f"You're running Chaos Inoculation — life is 1, so your ~{round(es)} "
                        f"Energy Shield is your ENTIRE health pool")
                tail = "or drop CI and keep your life pool"
            else:
                lead = (f"Energy Shield (~{round(es)}) is your main defence — bigger than your "
                        f"~{round(life)} life")
                tail = "or lean back on life"
            out.append({
                "label": "Energy Shield sustain",
                "severity": "critical" if is_ci else "warning",
                "gate": True,
                "summary": (f"{lead} — but you have no ES leech and no ES regen. ES only refills via "
                            f"recharge, which only starts after ~2s of NOT taking a hit, so in a sustained "
                            f"fight it never comes back. 'Faster ES recharge' nodes don't fix this — they "
                            f"change when recharge STARTS, not whether it can. Add Energy Shield leech "
                            f"(refills as you hit) or recharge-rate, {tail}." + _blame(drawbacks, "energy shield")),
            })

        # CI on a thin ES pool: ES is now your only HP and it's small for endgame.
        es_floor = max(4000.0, level * 55)
        if is_ci and level >= 65 and es < es_floor:
            out.append({
                "label": "Chaos Inoculation viability",
                "severity": "warning",
                "gate": True,
                "summary": (f"With CI your ~{round(es)} ES is your whole health pool — thin for level "
                            f"{round(level)} (aim for ~{round(es_floor/1000)}k+ at this stage). CI only pays "
                            f"off with a LARGE ES pool; below that you're squishier than you'd be on life+ES. "
                            f"Either grow ES hard (% increased ES, ES on every slot) or drop CI."),
            })
    except Exception as e:
        logger.debug(f"supporting stats failed: {e}")
    return out


def _build_coach_facts(exp: dict, char, swaps: list, budget: float = 0.0) -> str:
    sc = exp.get("scorecard", {}) or {}
    synergy = exp.get("synergyMap", []) or []
    free, cheap, chase = _classify_missing(synergy, budget)
    lines = [
        f"CHARACTER: {char.name} — {char.ascendancy or char.char_class}, Level {char.level}, main skill {sc.get('dpsSkill') or 'unknown'}.",
        f"DPS: {sc.get('dpsLabel','unknown')} ({sc.get('dpsPercentile','?')}th percentile among this build).",
        f"SURVIVAL: {sc.get('ehp','?')} EHP (status: {sc.get('ehpStatus','?')}).",
        f"RESISTANCES: {sc.get('resistSummary','?')}.",
    ]
    support = _supporting_stats(char)
    for s in support:
        lines.append(f"SUPPORTING STAT — {s['label']} ({s['severity']}): {s['summary']}")
    if budget and budget > 0:
        lines.append(f"PLAYER BUDGET: ~{budget:g} divine spendable right now — keep every recommendation within reach of this.")
    # Free actions: passive-tree swaps (no cost) then free gem/keystone changes.
    # Trim the verbose gain/lose detail to keep the prompt (and so the LLM
    # generation) short — keep the swap action + its top benefit.
    def _short_swap(s):
        action = s.split(". Gain:")[0].strip()
        gain = s.split("Gain:")[1].split("Lose:")[0].split(";")[0].strip() if "Gain:" in s else ""
        return f"{action} (gain {gain})" if gain else action
    free_lines = [f"passive tree swap — {_short_swap(s)}" for s in (swaps or [])[:2]]
    free_lines += [f"{r['name']} (+{round(r['impact'])}% {r['cat'].lower()}, free)" for r in free[:3]]
    if free_lines:
        lines.append("FREE CHANGES (do these first, no cost): " + "; ".join(free_lines))
    if cheap:
        lines.append("CHEAP UPGRADES: " + "; ".join(
            f"{r['name']} (+{round(r['impact'])}% {r['cat'].lower()}, {r['cost']})" for r in cheap[:4]))
    if chase:
        lines.append("LONG-TERM CHASE (save up — do NOT tell the player to buy now): " + "; ".join(
            f"{r['name']} ({r['cost']}, used by {round(r['adopt'])}% of top builds)" for r in chase[:3]))
        # Grounded funding guidance for this player's stage (so the coach can
        # tell them HOW to earn the currency, not just that it's expensive).
        try:
            import farming
            fst = farming.classify_stage(sc, char.level)
            strat = next((s for s in farming.STRATEGIES if fst["stage"] in s["stages"]), None)
            if strat:
                lines.append(f"FUNDING (the player is '{fst['label']}'): to earn currency for the chase items, "
                             f"their best fit is {strat['name']} — {strat['income']}")
        except Exception as e:
            logger.debug(f"coach funding hint failed: {e}")
    facts = "\n".join(lines)

    # Priority order — resource/defensive gates, then free, then cheap, chase last.
    pr = []
    for s in support:
        if s.get("severity") == "critical":
            pr.append(f"{s['label']} — fix first, you can't deal damage without it: {s['summary']}")
    if sc.get("ehpStatus") == "critical":
        pr.append(f"Survival is CRITICAL ({sc.get('ehp','?')} EHP) — add life and cap resistances on your gear (cheap). Fix this before anything else.")
    if sc.get("resistStatus") and sc.get("resistStatus") != "positive":
        pr.append(f"Cap your resistances ({sc.get('resistSummary','?')}) — cheap, and it stops one-shots.")
    if sc.get("ehpStatus") == "warning":
        pr.append(f"Shore up survival ({sc.get('ehp','?')} EHP) soon — cheap defensive gear.")
    for s in support:
        if s.get("severity") != "critical":
            pr.append(f"{s['label']}: {s['summary']}")
    if free_lines:
        pr.append("Make your FREE changes (the passive/gem swaps above) — biggest impact for zero cost.")
    if cheap:
        pr.append(f"Then the cheap upgrades: {', '.join(r['name'] for r in cheap[:2])}.")
    if chase:
        pr.append(f"Long-term goal only (expensive, don't buy yet): {chase[0]['name']} ({chase[0]['cost']}).")
    if pr:
        facts += "\n\nPRIORITY ORDER (ranked by LAMA — affordable + high-impact first; coach in this exact order):\n" + \
                 "\n".join(f"{i+1}. {p}" for i, p in enumerate(pr[:5]))
    return facts


async def _coach_tree_swaps(char, archetype, loop) -> list:
    """Free passive-tree swaps vs the top build for this archetype (summaries)."""
    try:
        from pob_decoder import decode_pob_code
        from tree_analyzer import TreeAnalyzer
        player_pob = decode_pob_code(char.pob_code) if char.pob_code else None
        if not player_pob or not player_pob.passive_nodes:
            return []
        char_class = char.ascendancy or char.char_class
        profile = await loop.run_in_executor(
            None, builds_client.fetch_archetype_profile, char_class, archetype.main_skill)
        top_nodes = None
        if profile and profile.get("featuredCharacters"):
            tc = profile["featuredCharacters"][0]
            top_char = await loop.run_in_executor(
                None, _lookup_character_with_fallback, tc.get("account", ""), tc.get("name", ""))
            if top_char and top_char.pob_code:
                tp = decode_pob_code(top_char.pob_code)
                top_nodes = tp.passive_nodes if tp else None
        analyzer = TreeAnalyzer()
        analysis = await loop.run_in_executor(None, analyzer.analyze, player_pob.passive_nodes, top_nodes)
        return [s.impact_summary for s in (analysis.swap_recommendations or [])
                if getattr(s, "impact_summary", "")][:3]
    except Exception as e:
        logger.debug(f"coach tree swaps failed: {e}")
        return []


def _ollama_available() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


def _ollama_chat(system: str, user: str, model: str = COACH_MODEL, timeout: int = 180) -> str:
    r = requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 260},
    }, timeout=timeout)
    r.raise_for_status()
    return ((r.json() or {}).get("message", {}) or {}).get("content", "").strip()


def _resolve_budget(req_budget) -> float:
    """Player's spendable divine budget: explicit request value, else saved setting."""
    try:
        b = float(req_budget or 0)
    except Exception:
        b = 0.0
    if b <= 0:
        try:
            b = float(load_settings().get("player_budget_div", 0) or 0)
        except Exception:
            b = 0.0
    return b if b > 0 else 0.0


class CoachRequest(BaseModel):
    account: str
    character: str
    model: str = ""
    budget: float = 0.0


async def _prepare_coach(req: "CoachRequest", loop):
    """Shared coach setup: lookup + analysis + tree swaps -> (facts, model).

    Returns {"error","status"} on failure, else {"facts","model"}."""
    char = await loop.run_in_executor(None, _lookup_character_with_fallback, req.account.strip(), req.character.strip())
    if not char:
        return {"error": "Character not found", "status": 404}
    archetype = classify_build(char)
    # Run the build analysis and the (free) tree-swap fetch concurrently — both
    # hit the network, so overlapping them shaves several seconds off the coach.
    swaps_task = asyncio.create_task(_coach_tree_swaps(char, archetype, loop))
    try:
        exp = (await loop.run_in_executor(None, why_engine_instance.explain_character, char, archetype)).to_dict()
    except Exception as e:
        swaps_task.cancel()
        logger.error(f"coach: analysis failed: {e}")
        return {"error": "Analysis failed", "status": 500}
    swaps = await swaps_task
    facts = _build_coach_facts(exp, char, swaps, _resolve_budget(req.budget))
    return {"facts": facts, "model": (req.model or COACH_MODEL).strip()}


# Coach response cache — generated in the background on character load and
# served instantly when the panel opens, so the slow model is hidden.
_coach_cache = {}
_coach_cache_lock = threading.Lock()
COACH_CACHE_TTL = 900  # 15 minutes


def _coach_key(account, character, budget):
    return f"{account.strip().lower()}|{character.strip().lower()}|{budget:g}"


def _coach_cache_get(key):
    with _coach_cache_lock:
        e = _coach_cache.get(key)
        if e and (time.time() - e.get("ts", 0)) < COACH_CACHE_TTL:
            return dict(e)
    return None


def _coach_cache_set(key, **kw):
    with _coach_cache_lock:
        _coach_cache[key] = {**kw, "ts": time.time()}


async def _generate_and_cache(req, loop, key):
    """Run the full coach (analysis + model) and store it in the cache."""
    _coach_cache_set(key, status="generating")
    prep = await _prepare_coach(req, loop)
    if "error" in prep:
        _coach_cache_set(key, status="error", error=prep["error"])
        return
    facts, model = prep["facts"], prep["model"]
    try:
        text = await loop.run_in_executor(None, _ollama_chat, COACH_SYSTEM, facts, model)
    except Exception as e:
        logger.error(f"coach prewarm generate failed: {e}")
        _coach_cache_set(key, status="error", error=str(e))
        return
    _coach_cache_set(key, status="done", text=text, model=model)


@app.post("/api/character/coach/prewarm")
async def coach_prewarm(req: CoachRequest):
    """Start generating the coach in the background so it's ready when opened."""
    if not _ollama_available() or not req.account.strip() or not req.character.strip():
        return {"status": "skip"}
    key = _coach_key(req.account, req.character, _resolve_budget(req.budget))
    e = _coach_cache_get(key)
    if e and e.get("status") in ("done", "generating"):
        return {"status": e["status"]}
    asyncio.create_task(_generate_and_cache(req, asyncio.get_running_loop(), key))
    return {"status": "warming"}


@app.post("/api/character/coach")
async def character_coach(req: CoachRequest):
    """Grounded plain-language 'what to focus on' from a local LLM (Ollama)."""
    if not _ollama_available():
        return JSONResponse(status_code=503, content={
            "available": False,
            "error": "Local AI coach unavailable. Install Ollama and pull a model (e.g. 'ollama pull phi4') to enable it.",
        })
    if not req.account.strip() or not req.character.strip():
        return JSONResponse(status_code=400, content={"error": "Account and character required"})
    loop = asyncio.get_running_loop()
    prep = await _prepare_coach(req, loop)
    if "error" in prep:
        return JSONResponse(status_code=prep["status"], content={"error": prep["error"]})
    facts, model = prep["facts"], prep["model"]
    try:
        coaching = await loop.run_in_executor(None, _ollama_chat, COACH_SYSTEM, facts, model)
    except Exception as e:
        logger.error(f"coach: ollama failed: {e}")
        return JSONResponse(status_code=502, content={"error": f"Coach model failed: {e}"})
    return {"coaching": coaching, "model": model, "facts": facts}


@app.post("/api/character/coach-stream")
async def character_coach_stream(req: CoachRequest):
    """Streaming coach: the analysis runs first, then the model's tokens stream
    in as plain text so the panel fills word-by-word instead of blank-spinning."""
    from fastapi.responses import StreamingResponse
    if not _ollama_available():
        return JSONResponse(status_code=503, content={
            "available": False,
            "error": "Local AI coach unavailable. Install Ollama and pull a model (e.g. 'ollama pull phi4') to enable it.",
        })
    if not req.account.strip() or not req.character.strip():
        return JSONResponse(status_code=400, content={"error": "Account and character required"})
    loop = asyncio.get_running_loop()
    key = _coach_key(req.account, req.character, _resolve_budget(req.budget))

    # If a prewarm is already generating this, wait for it rather than duplicate.
    e = _coach_cache_get(key)
    if e and e.get("status") == "generating":
        for _ in range(360):   # up to ~90s
            await asyncio.sleep(0.25)
            e = _coach_cache_get(key)
            if not e or e.get("status") != "generating":
                break

    # Cached + done -> serve instantly (the prewarm already paid the cost).
    if e and e.get("status") == "done" and e.get("text"):
        cached_text, cached_model = e["text"], e.get("model", COACH_MODEL)
        def replay():
            yield cached_text
        return StreamingResponse(replay(), media_type="text/plain; charset=utf-8",
                                 headers={"X-Coach-Model": cached_model, "X-Coach-Cached": "1", "X-Accel-Buffering": "no"})

    # Otherwise generate fresh, stream it, and cache the result for next time.
    prep = await _prepare_coach(req, loop)
    if "error" in prep:
        return JSONResponse(status_code=prep["status"], content={"error": prep["error"]})
    facts, model = prep["facts"], prep["model"]
    _coach_cache_set(key, status="generating")

    def generate():
        acc = []
        try:
            with requests.post(f"{OLLAMA_URL}/api/chat", json={
                "model": model,
                "messages": [{"role": "system", "content": COACH_SYSTEM},
                             {"role": "user", "content": facts}],
                "stream": True,
                "options": {"temperature": 0.3, "num_predict": 260},
            }, stream=True, timeout=240) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    delta = (obj.get("message") or {}).get("content", "")
                    if delta:
                        acc.append(delta)
                        yield delta
                    if obj.get("done"):
                        break
            _coach_cache_set(key, status="done", text="".join(acc), model=model)
        except Exception as e2:
            logger.error(f"coach stream failed: {e2}")
            _coach_cache_set(key, status="error", error=str(e2))

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8",
                             headers={"X-Coach-Model": model, "X-Accel-Buffering": "no"})


class FundingRequest(BaseModel):
    account: str
    character: str
    budget: float = 0.0


@app.post("/api/character/funding")
async def character_funding(req: FundingRequest):
    """Personalized, grounded 'how to fund it' currency-farming plan: classify the
    player's progression stage and surface the farming strategies that fit, plus
    the divine gap to their chase items."""
    if not req.account.strip() or not req.character.strip():
        return JSONResponse(status_code=400, content={"error": "Account and character required"})
    loop = asyncio.get_running_loop()
    char = await loop.run_in_executor(None, _lookup_character_with_fallback, req.account.strip(), req.character.strip())
    if not char:
        return JSONResponse(status_code=404, content={"error": "Character not found"})
    archetype = classify_build(char)
    try:
        exp = (await loop.run_in_executor(None, why_engine_instance.explain_character, char, archetype)).to_dict()
    except Exception as e:
        logger.error(f"funding: analysis failed: {e}")
        return JSONResponse(status_code=500, content={"error": "Analysis failed"})
    sc = exp.get("scorecard", {}) or {}
    budget = _resolve_budget(req.budget)
    _free, _cheap, chase = _classify_missing(exp.get("synergyMap", []) or [], budget)
    import farming
    return farming.funding_plan(sc, char.level, chase, budget)


class TreeAnalysisRequest(BaseModel):
    account: str
    character: str
    target_account: str = ""
    target_character: str = ""
    target_query: str = ""  # poe.ninja URL alternative


@app.post("/api/character/tree-analysis")
async def character_tree_analysis(req: TreeAnalysisRequest):
    """Analyze passive tree and recommend swaps vs a target build."""
    try:
        if not req.account.strip() or not req.character.strip():
            return {"swapRecommendations": [], "error": "Account and character required"}
        loop = asyncio.get_running_loop()

        player = await loop.run_in_executor(
            None, _lookup_character_with_fallback, req.account.strip(), req.character.strip()
        )
        if not player:
            return {"swapRecommendations": [], "error": "Player not found"}

        from pob_decoder import decode_pob_code
        from tree_analyzer import TreeAnalyzer

        player_pob = decode_pob_code(player.pob_code) if player.pob_code else None
        if not player_pob or not player_pob.passive_nodes:
            return {"swapRecommendations": [], "totalAllocated": 0}

        # Get target build (optional) — all wrapped in try/except
        top_nodes = None
        target_name = ""
        try:
            if req.target_query and "poe.ninja" in req.target_query:
                acct, char_name = _parse_ninja_url(req.target_query)
                if acct and char_name:
                    target = await loop.run_in_executor(
                        None, _lookup_character_with_fallback, acct, char_name
                    )
                    if target and target.pob_code:
                        top_pob = decode_pob_code(target.pob_code)
                        top_nodes = top_pob.passive_nodes if top_pob else None
                        target_name = target.name or ""
            elif req.target_account and req.target_character:
                target = await loop.run_in_executor(
                    None, _lookup_character_with_fallback, req.target_account.strip(), req.target_character.strip()
                )
                if target and target.pob_code:
                    top_pob = decode_pob_code(target.pob_code)
                    top_nodes = top_pob.passive_nodes if top_pob else None
                    target_name = target.name or ""

            # If no target specified, use the top featured character for this class
            if not top_nodes:
                archetype = classify_build(player)
                char_class = player.ascendancy or player.char_class
                profile = await loop.run_in_executor(
                    None, builds_client.fetch_archetype_profile, char_class, archetype.main_skill
                )
                if profile and profile.get("featuredCharacters"):
                    top_ch = profile["featuredCharacters"][0]
                    top_char = await loop.run_in_executor(
                        None, _lookup_character_with_fallback, top_ch.get("account", ""), top_ch.get("name", "")
                    )
                    if top_char and top_char.pob_code:
                        top_pob = decode_pob_code(top_char.pob_code)
                        top_nodes = top_pob.passive_nodes if top_pob else None
                        target_name = top_char.name or ""
        except Exception as e:
            logger.debug(f"Target lookup failed (non-fatal): {e}")

        analyzer = TreeAnalyzer()
        analysis = await loop.run_in_executor(
            None, analyzer.analyze, player_pob.passive_nodes, top_nodes
        )
        result = analysis.to_dict()
        result["targetName"] = target_name

        # Add tree visualization data
        # GGG node-id sets for the canvas renderer (small payload; the canvas
        # pulls geometry + sprites from the cached /tree2 export client-side).
        try:
            # The analyzer now runs on the same GGG 0.5.0 export the canvas uses,
            # so its swap node ids are 0.5.0-native — use them directly.
            result["treeNodes"] = {
                "player": [str(n) for n in player_pob.passive_nodes],
                "top": [str(n) for n in (top_nodes or [])],
                "swapTake": [s.take_id for s in analysis.swap_recommendations if s.take_id],
                "swapRefund": [s.refund_id for s in analysis.swap_recommendations if s.refund_id],
            }
            for sw in result.get("swapRecommendations", []):
                sw["takeNodes"] = [sw["takeId"]] if sw.get("takeId") else []
                sw["refundNodes"] = [sw["refundId"]] if sw.get("refundId") else []
        except Exception as e:
            logger.debug(f"treeNodes build failed (non-fatal): {e}")

        return result
    except Exception as e:
        logger.error(f"Tree analysis failed: {e}")
        return {"swapRecommendations": [], "error": str(e)}


class BuildCompareRequest2(BaseModel):
    player_account: str
    player_character: str
    target_query: str  # poe.ninja URL or account/character


@app.post("/api/character/compare")
async def character_compare(req: BuildCompareRequest2):
    """Compare player build against a target build (guide/reference)."""
    if not req.player_account.strip() or not req.player_character.strip() or not req.target_query.strip():
        return JSONResponse(status_code=400, content={"error": "Player and target build required"})
    loop = asyncio.get_running_loop()

    # Look up player
    player = await loop.run_in_executor(
        None, _lookup_character_with_fallback, req.player_account.strip(), req.player_character.strip()
    )
    if not player:
        return JSONResponse(status_code=404, content={"error": "Player character not found"})

    # Parse target — could be a poe.ninja URL or account/character
    target_query = req.target_query.strip()
    target = None

    if "poe.ninja" in target_query:
        acct, char_name = _parse_ninja_url(target_query)
        if acct and char_name:
            target = await loop.run_in_executor(
                None, _lookup_character_with_fallback, acct, char_name
            )
    else:
        # Accept account + character split on any of these separators
        # ("-" is excluded: it is the account discriminator, e.g. JCOLLINS510-4794).
        for sep in ("/", "#", ","):
            if sep in target_query:
                parts = [p.strip() for p in target_query.split(sep, 1)]
                if len(parts) == 2 and parts[0] and parts[1]:
                    target = await loop.run_in_executor(
                        None, _lookup_character_with_fallback, parts[0], parts[1]
                    )
                break

    if not target:
        return JSONResponse(status_code=404, content={"error": f"Target build not found: {target_query}"})

    try:
        result = await loop.run_in_executor(
            None, why_engine_instance.compare_builds, player, target
        )

        # Enrich shopping list with prices from price_cache
        if price_cache and result.get("shoppingList"):
            for item in result["shoppingList"]:
                target_name = item.get("targetItem", "")
                target_rarity = item.get("targetRarity", "")
                if target_rarity in ("Unique", "unique") and target_name:
                    try:
                        price_data = price_cache.lookup(target_name, "", 0)
                        if price_data:
                            item["price"] = price_data.get("display", "")
                            item["priceTier"] = price_data.get("tier", "")
                            item["priceValue"] = price_data.get("divine_value", 0) or price_data.get("chaos_value", 0)
                    except Exception:
                        pass

        # Enrich gear diffs with prices too
        if price_cache and result.get("diffs"):
            for diff in result["diffs"]:
                if diff.get("category") != "gear":
                    continue
                target_name = diff.get("target", "")
                target_rarity = diff.get("targetRarity", "")
                if target_rarity in ("Unique", "unique") and target_name:
                    try:
                        price_data = price_cache.lookup(target_name, "", 0)
                        if price_data:
                            diff["price"] = price_data.get("display", "")
                            diff["priceTier"] = price_data.get("tier", "")
                    except Exception:
                        pass

        # Enrich shopping list with popular alternatives per slot
        # (what top players actually use, with prices)
        if result.get("shoppingList"):
            try:
                for item in result["shoppingList"]:
                    slot = item.get("slot", "")
                    if not slot:
                        continue
                    popular_data = await loop.run_in_executor(
                        None, builds_client.get_popular_items_for_slot, player, slot
                    )
                    alternatives = []
                    current_name = item.get("currentItem", "").lower()
                    for pi in (popular_data.get("items") or [])[:8]:
                        pi_name = pi.get("name", "")
                        if pi_name.lower() == current_name:
                            continue  # skip what they already have
                        alt = {
                            "name": pi_name,
                            "usage": pi.get("percentage", 0),
                            "rarity": pi.get("rarity", ""),
                        }
                        # Add price if available
                        if pi.get("priceText"):
                            alt["price"] = pi["priceText"]
                        elif price_cache and pi.get("rarity") == "unique":
                            try:
                                pd = price_cache.lookup(pi_name, "", 0)
                                if pd:
                                    alt["price"] = pd.get("display", "")
                            except Exception:
                                pass
                        alternatives.append(alt)
                    item["alternatives"] = alternatives[:5]
            except Exception as e:
                logger.debug(f"Popular items enrichment failed (non-fatal): {e}")

        return result
    except Exception as e:
        logger.error(f"Build comparison failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


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
        None, _lookup_character_with_fallback, req.account.strip(), req.character.strip()
    )
    if not char_data:
        return JSONResponse(status_code=404, content={"error": "Character not found. Non-ladder characters require OAuth login."})

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
        None, _lookup_character_with_fallback, req.account.strip(), req.character.strip()
    )
    if not char_data:
        return JSONResponse(status_code=404, content={"error": "Character not found. Non-ladder characters require OAuth login."})

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
        None, _lookup_character_with_fallback, req.account.strip(), req.character.strip()
    )
    if not char_data:
        return JSONResponse(status_code=404, content={"error": "Character not found. Non-ladder characters require OAuth login."})

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
            None, _lookup_character_with_fallback, req.account.strip(), req.character_name.strip()
        )
        if not char_data:
            return JSONResponse(status_code=404, content={"error": "Character not found. Non-ladder characters require OAuth login."})
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
# OAuth character endpoints (GGG Character API)
# ---------------------------------------------------------------------------
@app.get("/api/oauth/characters")
async def oauth_characters():
    """List all characters for the OAuth-connected account."""
    if not oauth_manager or not oauth_manager.connected:
        return JSONResponse(status_code=401, content={"error": "Not connected"})
    if not character_client:
        return JSONResponse(status_code=503, content={"error": "Character client not initialized"})
    loop = asyncio.get_running_loop()
    chars = await loop.run_in_executor(None, character_client.list_characters)
    return {
        "characters": chars,
        "account_name": oauth_manager.account_name,
    }


class OAuthCharLookupRequest(BaseModel):
    character: str


@app.post("/api/oauth/character/lookup")
async def oauth_character_lookup(req: OAuthCharLookupRequest):
    """Fetch a specific character via GGG API and return enriched data."""
    if not oauth_manager or not oauth_manager.connected:
        return JSONResponse(status_code=401, content={"error": "Not connected"})
    if not character_client:
        return JSONResponse(status_code=503, content={"error": "Character client not initialized"})
    if not req.character.strip():
        return JSONResponse(status_code=400, content={"error": "Character name required"})

    loop = asyncio.get_running_loop()
    char_data = await loop.run_in_executor(
        None, character_client.get_character, req.character.strip()
    )
    if not char_data:
        return JSONResponse(status_code=404, content={"error": "Character not found. Non-ladder characters require OAuth login."})

    result = builds_client.serialize_character(char_data)
    result["source"] = "ggg"

    # Detect league from character list
    chars = await loop.run_in_executor(None, character_client.list_characters)
    for ch in chars:
        if ch.get("name") == char_data.name:
            result["detected_league"] = ch.get("league", "")
            break

    # Enrich equipment mods with tier data
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


class PriceGearRequest(BaseModel):
    character: str


@app.post("/api/oauth/character/price-gear")
async def oauth_price_gear(req: PriceGearRequest):
    """Price all equipped items on a character via GGG API."""
    if not oauth_manager or not oauth_manager.connected:
        return JSONResponse(status_code=401, content={"error": "Not connected"})
    if not character_client:
        return JSONResponse(status_code=503, content={"error": "Character client not initialized"})
    if not req.character.strip():
        return JSONResponse(status_code=400, content={"error": "Character name required"})

    loop = asyncio.get_running_loop()
    raw = await loop.run_in_executor(
        None, character_client.get_character_raw, req.character.strip()
    )
    if not raw:
        return JSONResponse(status_code=404, content={"error": "Character not found. Non-ladder characters require OAuth login."})

    # Extract equipment items from raw GGG response
    char_data = raw.get("character", raw) if isinstance(raw, dict) else raw
    equipment_items = char_data.get("equipment", []) + char_data.get("items", [])

    slots = []
    total_divine = 0.0
    d2c = 0.0
    if price_cache:
        d2c = getattr(price_cache, "divine_to_chaos", 0) or 0

    for api_item in equipment_items:
        slot = api_item.get("inventoryId", "")
        if not slot:
            continue

        frame_type = api_item.get("frameType", 0)
        rarity_map = {0: "normal", 1: "magic", 2: "rare", 3: "unique"}
        rarity = rarity_map.get(frame_type, "normal")
        item_name = api_item.get("name", "") or api_item.get("typeLine", "")
        icon = api_item.get("icon", "")
        slot_display = SLOT_DISPLAY.get(slot, slot)

        estimate = 0.0
        grade = ""

        if rarity == "unique" and price_cache:
            # Lookup unique price from cache
            try:
                lookup_name = item_name
                price_data = price_cache.lookup(lookup_name)
                if price_data and price_data.get("divine"):
                    estimate = price_data["divine"]
                elif price_data and price_data.get("chaos") and d2c > 0:
                    estimate = price_data["chaos"] / d2c
            except Exception:
                pass
        elif rarity == "rare":
            # Score via item parser + calibration
            try:
                from stash_client import StashClient
                parsed = StashClient.api_item_to_parsed(api_item)
                if parsed and item_lookup and item_lookup.ready:
                    score_result = item_lookup.score_item(parsed)
                    if score_result:
                        grade = score_result.get("grade", "")
                        est_divine = score_result.get("estimate_divine", 0)
                        if est_divine:
                            estimate = est_divine
            except Exception:
                pass

        total_divine += estimate

        slots.append({
            "slot": slot_display,
            "name": item_name,
            "rarity": rarity,
            "estimate_divine": round(estimate, 2),
            "grade": grade,
            "icon": icon,
        })

    total_chaos = round(total_divine * d2c, 0) if d2c > 0 else 0

    return {
        "character": req.character.strip(),
        "slots": slots,
        "total_divine": round(total_divine, 1),
        "total_chaos": int(total_chaos),
    }


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
    league = settings.get("league", DEFAULT_LEAGUE)
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
    league = settings.get("league", DEFAULT_LEAGUE)
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
    league = settings.get("league", DEFAULT_LEAGUE)

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


@app.get("/vendor/{filepath:path}")
async def serve_vendor(filepath: str):
    """Serve bundled vendor files (JS, CSS, fonts)."""
    from fastapi.responses import FileResponse
    vendor_path = get_resource(f"resources/vendor/{filepath}")
    if not vendor_path.exists():
        return HTMLResponse("Not found", status_code=404)
    media_types = {
        ".js": "application/javascript",
        ".css": "text/css",
        ".ttf": "font/ttf",
        ".woff2": "font/woff2",
        ".woff": "font/woff",
    }
    ext = "." + filepath.rsplit(".", 1)[-1] if "." in filepath else ""
    mt = media_types.get(ext, "application/octet-stream")
    return FileResponse(vendor_path, media_type=mt)


@app.get("/tree2/{filepath:path}")
async def serve_tree2(filepath: str):
    """Serve the cached GGG passive-tree export (data.json + WebP sprite atlases).

    Files download on first use; here we resolve from the local cache, fetching
    on demand if a requested file isn't present yet."""
    import tree2
    from fastapi.responses import FileResponse, JSONResponse
    p = tree2.file_path(filepath)
    if not p:
        await asyncio.get_running_loop().run_in_executor(None, tree2.ensure_assets)
        p = tree2.file_path(filepath)
    if not p:
        return JSONResponse(status_code=404, content={"error": "Tree asset not found"})
    media_types = {".json": "application/json", ".webp": "image/webp"}
    ext = "." + filepath.rsplit(".", 1)[-1] if "." in filepath else ""
    # The export is immutable per season (branch-pinned), so let the browser
    # cache it hard — avoids re-downloading ~6MB on every launch.
    return FileResponse(p, media_type=media_types.get(ext, "application/octet-stream"),
                        headers={"Cache-Control": "public, max-age=604800, immutable"})


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
# Cloud Alerts endpoints (push notifications via relay)
# ---------------------------------------------------------------------------
@app.post("/api/cloud/enable")
async def cloud_enable():
    """Enable cloud alerts — generate device_id + secret, save to settings."""
    import uuid
    import secrets as secrets_mod
    settings = load_settings()
    device_id = str(uuid.uuid4())
    secret = secrets_mod.token_hex(16)  # 32-char hex
    settings["cloud_enabled"] = True
    settings["cloud_device_id"] = device_id
    settings["cloud_secret"] = secret
    save_settings(settings)
    cloud_notify.configure(
        device_id=device_id, secret=secret,
        relay_url=settings.get("cloud_relay_url", ""),
        enabled=True,
    )
    await ws_manager.broadcast({"type": "settings", "settings": _redact_settings(settings)})
    return {"device_id": device_id, "relay_url": settings.get("cloud_relay_url", "")}


@app.post("/api/cloud/disable")
async def cloud_disable():
    """Disable cloud alerts — clear credentials."""
    settings = load_settings()
    settings["cloud_enabled"] = False
    settings["cloud_device_id"] = ""
    settings["cloud_secret"] = ""
    save_settings(settings)
    cloud_notify.configure(device_id="", secret="", relay_url="", enabled=False)
    await ws_manager.broadcast({"type": "settings", "settings": _redact_settings(settings)})
    return {"status": "disabled"}


@app.get("/api/cloud/qr")
async def cloud_qr():
    """Return QR code PNG for mobile cloud pairing."""
    settings = load_settings()
    if not settings.get("cloud_enabled") or not settings.get("cloud_device_id"):
        return JSONResponse({"error": "Cloud alerts not enabled"}, status_code=400)
    try:
        import qrcode
    except ImportError:
        return JSONResponse({"error": "qrcode library not installed"}, status_code=500)
    relay_url = settings.get("cloud_relay_url", "")
    payload = json.dumps({
        "relay": relay_url,
        "id": settings["cloud_device_id"],
        "key": settings["cloud_secret"],
    })
    img = qrcode.make(payload)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/cloud/info")
async def cloud_info():
    """Return cloud alerts status (no secret exposed)."""
    settings = load_settings()
    enabled = settings.get("cloud_enabled", False)
    result = {"enabled": enabled}
    if enabled:
        result["device_id"] = settings.get("cloud_device_id", "")
        result["relay_url"] = settings.get("cloud_relay_url", "")
    return result


@app.post("/api/cloud/relay-url")
async def cloud_set_relay_url(req: Request):
    """Update the relay URL."""
    body = await req.json()
    relay_url = body.get("relay_url", "").strip()
    settings = load_settings()
    settings["cloud_relay_url"] = relay_url
    save_settings(settings)
    cloud_notify.configure(
        device_id=settings.get("cloud_device_id", ""),
        secret=settings.get("cloud_secret", ""),
        relay_url=relay_url,
        enabled=settings.get("cloud_enabled", False),
    )
    await ws_manager.broadcast({"type": "settings", "settings": _redact_settings(settings)})
    return {"status": "ok", "relay_url": relay_url}


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
