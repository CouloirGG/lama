"""
app.py — Standalone desktop launcher for LAMA (Live Auction Market Assessor).

Starts the FastAPI server in a background thread and opens the
dashboard in a native window (no browser required).  The system tray icon
lets users hide/show the window, control the overlay, and quit.

Usage:
    python app.py

Requirements:
    pip install pywebview pystray Pillow
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from urllib.request import Request

# Ensure src/ is on sys.path so bare imports and uvicorn "server:app" work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
            environment="desktop",
            traces_sample_rate=0.1,
            before_send=_sentry_before_send,
        )
    except Exception as e:
        print(f"  Sentry: init failed ({e})")


def _log(msg):
    """Print that won't crash under pythonw (sys.stdout is None)."""
    try:
        print(msg)
    except Exception:
        pass


def _strip_window_border(hwnd):
    """Remove the thin frame border line (Windows 11) so the frameless window
    has no white edge, windowed or maximized. No-op on older Windows."""
    try:
        import ctypes
        from ctypes import wintypes
        fn = ctypes.windll.dwmapi.DwmSetWindowAttribute
        fn.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
        DWMWA_BORDER_COLOR = 34
        color = ctypes.c_uint(0xFFFFFFFE)  # DWMWA_COLOR_NONE
        fn(wintypes.HWND(hwnd), DWMWA_BORDER_COLOR, ctypes.byref(color), ctypes.sizeof(color))
    except Exception as e:
        _log(f"[Startup] border strip skipped: {e}")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PORT = int(os.environ.get("POE2_DASHBOARD_PORT", "8450"))
WINDOW_TITLE = "LAMA"
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 750


def _companion_enabled() -> bool:
    """Check if companion mode is enabled in settings."""
    settings_path = os.path.join(
        os.path.expanduser("~"), ".poe2-price-overlay", "dashboard_settings.json"
    )
    try:
        if os.path.exists(settings_path):
            with open(settings_path) as f:
                return json.load(f).get("companion_enabled", False)
    except Exception:
        pass
    return False


def start_server():
    """Run the FastAPI server in a background thread."""
    import uvicorn
    # Always bind to 0.0.0.0 so companion mode can be toggled at runtime
    # without a server restart.  PIN auth protects against unauthorised access.
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
    )


def wait_for_port_free(timeout=10):
    """Block until the port is no longer in use (for restart handoff)."""
    import socket
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", PORT))
                return True  # Port is free
            except OSError:
                time.sleep(0.3)
    return False


def wait_for_server(timeout=10):
    """Block until the server is accepting connections."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/status", timeout=1)
            return True
        except Exception:
            time.sleep(0.2)
    return False


class WindowApi:
    """JS-callable window controls for frameless mode."""

    def __init__(self):
        self._guard_until = 0.0
        self._original_proc = None
        self._hook_ref = None  # prevent garbage collection
        self._tray = None  # set by main() after tray starts
        self._ready_event = threading.Event()

    def _get_hwnd(self):
        import ctypes
        return ctypes.windll.user32.FindWindowW(None, WINDOW_TITLE)

    def _install_hook(self):
        """Install a Win32 hook for resize guard and edge-resize on frameless window."""
        import ctypes
        from ctypes import wintypes, WINFUNCTYPE, POINTER, c_int, c_uint

        hwnd = self._get_hwnd()
        if not hwnd or self._original_proc:
            return

        # Ensure WS_THICKFRAME is set so Windows sends WM_NCHITTEST for resize
        GWL_STYLE = -16
        WS_THICKFRAME = 0x00040000
        style = ctypes.windll.user32.GetWindowLongPtrW(hwnd, GWL_STYLE)
        if not (style & WS_THICKFRAME):
            ctypes.windll.user32.SetWindowLongPtrW(hwnd, GWL_STYLE, style | WS_THICKFRAME)
            # Refresh the frame without flashing
            SWP_FRAMECHANGED = 0x0020
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_NOZORDER = 0x0004
            ctypes.windll.user32.SetWindowPos(
                hwnd, 0, 0, 0, 0, 0,
                SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER
            )

        WM_WINDOWPOSCHANGING = 0x0046
        WM_NCHITTEST = 0x0084
        SWP_NOSIZE = 0x0001
        GWL_WNDPROC = -4

        # WM_NCHITTEST return values for resize edges
        HTCLIENT = 1
        HTLEFT = 10
        HTRIGHT = 11
        HTTOP = 12
        HTTOPLEFT = 13
        HTTOPRIGHT = 14
        HTBOTTOM = 15
        HTBOTTOMLEFT = 16
        HTBOTTOMRIGHT = 17

        RESIZE_BORDER = 8  # pixels from edge that trigger resize cursor

        class WINDOWPOS(ctypes.Structure):
            _fields_ = [
                ("hwnd", wintypes.HWND),
                ("hwndInsertAfter", wintypes.HWND),
                ("x", c_int), ("y", c_int),
                ("cx", c_int), ("cy", c_int),
                ("flags", c_uint),
            ]

        class RECT(ctypes.Structure):
            _fields_ = [("left", c_int), ("top", c_int),
                        ("right", c_int), ("bottom", c_int)]

        # LRESULT is pointer-sized (8 bytes on 64-bit Windows)
        LRESULT = wintypes.LPARAM
        WNDPROC = WINFUNCTYPE(LRESULT, wintypes.HWND, c_uint,
                              wintypes.WPARAM, wintypes.LPARAM)

        user32 = ctypes.windll.user32
        # Set restype to pointer-sized int (critical on 64-bit Windows).
        # Don't set argtypes on SetWindowLongPtrW — it needs to accept
        # a CFUNCTYPE callback which ctypes can't coerce to c_longlong.
        user32.SetWindowLongPtrW.restype = LRESULT
        user32.CallWindowProcW.restype = LRESULT
        user32.CallWindowProcW.argtypes = [
            ctypes.c_void_p, wintypes.HWND, c_uint, wintypes.WPARAM, wintypes.LPARAM
        ]

        api_ref = self

        @WNDPROC
        def hook_proc(hwnd, msg, wparam, lparam):
            if msg == WM_WINDOWPOSCHANGING and time.time() < api_ref._guard_until:
                pos = ctypes.cast(lparam, POINTER(WINDOWPOS)).contents
                pos.flags |= SWP_NOSIZE  # silently prevent resize

            # Edge resize: map cursor position near borders to resize handles
            if msg == WM_NCHITTEST:
                result = user32.CallWindowProcW(api_ref._original_proc,
                                                hwnd, msg, wparam, lparam)
                if result == HTCLIENT:
                    rc = RECT()
                    user32.GetWindowRect(hwnd, ctypes.byref(rc))
                    x = (lparam & 0xFFFF)
                    y = ((lparam >> 16) & 0xFFFF)
                    # Convert unsigned to signed (for multi-monitor negative coords)
                    if x >= 0x8000: x -= 0x10000
                    if y >= 0x8000: y -= 0x10000

                    left = x - rc.left < RESIZE_BORDER
                    right = rc.right - x < RESIZE_BORDER
                    top = y - rc.top < RESIZE_BORDER
                    bottom = rc.bottom - y < RESIZE_BORDER

                    if top and left:     return HTTOPLEFT
                    if top and right:    return HTTOPRIGHT
                    if bottom and left:  return HTBOTTOMLEFT
                    if bottom and right: return HTBOTTOMRIGHT
                    if left:             return HTLEFT
                    if right:            return HTRIGHT
                    if top:              return HTTOP
                    if bottom:           return HTBOTTOM
                return result

            return user32.CallWindowProcW(api_ref._original_proc,
                                          hwnd, msg, wparam, lparam)

        self._hook_ref = hook_proc
        self._original_proc = user32.SetWindowLongPtrW(hwnd, GWL_WNDPROC,
                                                        hook_proc)

    def on_ready(self):
        """Called from JS when dashboard has loaded initial data."""
        self._ready_event.set()

    def begin_guard(self):
        """Block window resizes for the next second (called from JS before actions)."""
        if not self._original_proc:
            self._install_hook()
        self._guard_until = time.time() + 1.0

    def minimize(self):
        import ctypes
        hwnd = self._get_hwnd()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 6)  # SW_MINIMIZE

    def toggle_maximize(self):
        """Toggle a work-area maximize and return the new state.

        We deliberately do NOT use SW_MAXIMIZE: a frameless window maximized
        that way fills the whole monitor and covers the taskbar. Instead we
        resize to the work area (taskbar excluded) and remember the prior rect
        to restore to."""
        import ctypes
        hwnd = self._get_hwnd()
        if not hwnd:
            return False

        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                        ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

        self._guard_until = 0  # allow the resize
        if getattr(self, "_maximized", False):
            r = getattr(self, "_restore_rect", None)
            if r:
                ctypes.windll.user32.MoveWindow(hwnd, r[0], r[1], r[2], r[3], True)
            self._maximized = False
        else:
            cur = RECT()
            ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(cur))
            self._restore_rect = (cur.left, cur.top, cur.right - cur.left, cur.bottom - cur.top)
            work = RECT()
            ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(work), 0)  # SPI_GETWORKAREA
            ctypes.windll.user32.MoveWindow(hwnd, work.left, work.top,
                                            work.right - work.left, work.bottom - work.top, True)
            self._maximized = True
        threading.Timer(0.3, self._persist_geometry).start()
        return self._maximized

    def resize_to(self, width, height):
        """Resize and center the window (called from JS size presets)."""
        import ctypes
        hwnd = self._get_hwnd()
        if not hwnd:
            return
        w, h = max(900, int(width)), max(600, int(height))
        # If maximized, restore first
        if ctypes.windll.user32.IsZoomed(hwnd):
            self._guard_until = 0
            ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            time.sleep(0.05)
        # Get screen work area (excludes taskbar)
        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                        ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
        work = RECT()
        ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(work), 0)  # SPI_GETWORKAREA
        sw = work.right - work.left
        sh = work.bottom - work.top
        w = min(w, sw)
        h = min(h, sh)
        x = work.left + (sw - w) // 2
        y = work.top + (sh - h) // 2
        self._guard_until = 0
        ctypes.windll.user32.MoveWindow(hwnd, x, y, w, h, True)
        self._maximized = False  # a preset size un-maximizes
        # Persist
        self._pending_geo = (w, h, False)
        threading.Timer(0.3, self._persist_geometry).start()

    def save_geometry(self, width, height, maximized=False):
        """Called from JS on window resize to persist geometry."""
        self._pending_geo = (int(width), int(height), bool(maximized))
        # Debounce: only persist after 500ms of no further calls
        if hasattr(self, "_geo_timer") and self._geo_timer:
            self._geo_timer.cancel()
        self._geo_timer = threading.Timer(0.5, self._persist_geometry)
        self._geo_timer.start()

    def _persist_geometry(self):
        """Write current window geometry to settings file."""
        import ctypes
        try:
            hwnd = self._get_hwnd()
            if not hwnd:
                return
            is_max = getattr(self, "_maximized", False) or bool(ctypes.windll.user32.IsZoomed(hwnd))
            # If maximized, don't overwrite the normal-size values
            if is_max:
                geo = getattr(self, "_pending_geo", None)
                if geo:
                    w, h = geo[0], geo[1]
                else:
                    w, h = WINDOW_WIDTH, WINDOW_HEIGHT
            else:
                # Read actual window rect
                class RECT(ctypes.Structure):
                    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
                rc = RECT()
                ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rc))
                w = rc.right - rc.left
                h = rc.bottom - rc.top
            settings_path = os.path.join(
                os.path.expanduser("~"), ".poe2-price-overlay", "dashboard_settings.json"
            )
            settings = {}
            if os.path.exists(settings_path):
                with open(settings_path) as f:
                    settings = json.load(f)
            if not is_max:
                settings["window_width"] = max(900, w)
                settings["window_height"] = max(600, h)
            settings["window_maximized"] = is_max
            with open(settings_path, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception:
            pass

    def close(self):
        """Hide to tray instead of quitting."""
        import ctypes
        hwnd = self._get_hwnd()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE

    def show(self):
        """Restore the window from tray."""
        import ctypes
        hwnd = self._get_hwnd()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            ctypes.windll.user32.SetForegroundWindow(hwnd)

    def force_close(self):
        """Actually destroy the window (used by restart and quit)."""
        import webview
        if webview.windows:
            webview.windows[0].destroy()

    def quit(self):
        """Stop tray icon and destroy the window — full exit."""
        if self._tray:
            self._tray.stop()
        self.force_close()

    def export_overlay_config(self, json_str):
        """Open native Save dialog and write overlay config JSON."""
        import webview
        win = webview.windows[0] if webview.windows else None
        if not win:
            return {"ok": False, "error": "No window"}
        result = win.create_file_dialog(
            webview.SAVE_DIALOG,
            save_filename="lama-overlay-config.json",
            file_types=("JSON files (*.json)",),
        )
        if not result:
            return {"ok": False}
        path = result if isinstance(result, str) else result[0]
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
            return {"ok": True, "path": path}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def import_overlay_config(self):
        """Open native Open dialog and read overlay config JSON."""
        import webview
        win = webview.windows[0] if webview.windows else None
        if not win:
            return {"ok": False, "error": "No window"}
        result = win.create_file_dialog(
            webview.OPEN_DIALOG,
            file_types=("JSON files (*.json)",),
        )
        if not result:
            return {"ok": False}
        path = result if isinstance(result, str) else result[0]
        try:
            with open(path, "r", encoding="utf-8") as f:
                return {"ok": True, "data": f.read()}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


def _ensure_deps():
    """Auto-install missing dependencies (runs silently under pythonw)."""
    try:
        import webview   # noqa: F401
        import pystray   # noqa: F401
        import PIL        # noqa: F401
        return
    except ImportError:
        pass
    import subprocess
    req = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "requirements.txt")
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0  # SW_HIDE
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", req,
         "--quiet", "--disable-pip-version-check"],
        creationflags=subprocess.CREATE_NO_WINDOW,
        startupinfo=si,
    )


def _tooltip_updater(tray):
    """Background thread: update tray tooltip every 10 seconds."""
    while tray._icon:
        try:
            raw = urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/api/status", timeout=2
            ).read()
            data = json.loads(raw)
            state = data.get("state", "stopped")
            triggers = data.get("stats", {}).get("triggers", 0)
            if state == "running":
                tray.update_tooltip(f"LAMA - Overlay running ({triggers} triggers)")
            else:
                tray.update_tooltip(f"LAMA - Overlay {state}")
        except Exception:
            tray.update_tooltip("LAMA")
        time.sleep(10)


def _set_icon_and_show(get_hwnd, show_fn, api_ref=None, _ms=None):
    """Background thread: set the taskbar icon, then reveal the window."""
    import ctypes
    from bundle_paths import get_resource

    # Wait for the hidden window's hwnd to exist
    hwnd = 0
    for _ in range(40):
        time.sleep(0.25)
        hwnd = get_hwnd()
        if hwnd:
            break
    if not hwnd:
        show_fn()  # show anyway even if icon fails
        return
    if _ms:
        _log(f"[Startup] hwnd found ({_ms()})")

    # Set the taskbar icon before the window is visible
    try:
        from win32com.propsys import propsys

        ico_path = str(get_resource("resources/img/favicon.ico"))
        store = propsys.SHGetPropertyStoreForWindow(hwnd)

        key_icon = propsys.PSGetPropertyKeyFromName(
            "System.AppUserModel.RelaunchIconResource")
        key_id = propsys.PSGetPropertyKeyFromName(
            "System.AppUserModel.ID")

        store.SetValue(key_icon, propsys.PROPVARIANTType(ico_path))
        store.SetValue(key_id, propsys.PROPVARIANTType("Couloir.LAMA"))
        store.Commit()
    except Exception:
        pass  # non-critical — falls back to executable icon

    # Install Win32 resize hook before showing the window
    if api_ref:
        api_ref._install_hook()

    # Wait for dashboard to signal it has received initial data
    if api_ref:
        ready = api_ref._ready_event.wait(timeout=5.0)
        _log(f"[Startup] {'Dashboard ready' if ready else 'Ready timeout (5s)'}")

    # Reveal the window (no-op when window is already visible)
    show_fn()
    if _ms:
        _log(f"[Startup] Window shown ({_ms()})")


def main():
    _t0 = time.time()
    def _ms(): return f"+{int((time.time() - _t0) * 1000)}ms"

    # Tell Windows this is its own app, not a generic Python process.
    # Must be called before any window is created.
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Couloir.LAMA")
    except Exception:
        pass

    try:
        import setproctitle
        setproctitle.setproctitle("LAMA")
    except ImportError:
        pass

    _ensure_deps()
    _log(f"[Startup] Deps OK ({_ms()})")
    try:
        import webview
    except ImportError:
        _log("=" * 50)
        _log("pywebview is required for standalone mode.")
        _log("Install it with:")
        _log("")
        _log("    pip install pywebview")
        _log("")
        _log("Then re-run: python app.py")
        _log("=" * 50)
        sys.exit(1)

    # Debug mode — enables WebView2 DevTools (right-click → Inspect)
    _debug_mode = "--debug" in sys.argv or "-d" in sys.argv

    # If launched with --restart, wait for the old process to release the port
    if "--restart" in sys.argv:
        _log("Restart requested — waiting for old process to release port...")
        if not wait_for_port_free():
            _log("ERROR: Port not freed within 10 seconds.")
            sys.exit(1)

    # Start the server in a daemon thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Wait for it to be ready
    _log(f"Starting LAMA on port {PORT}...")
    if not wait_for_server():
        _log("ERROR: Server failed to start within 10 seconds.")
        sys.exit(1)

    _log(f"[Startup] Server ready ({_ms()})")

    # Load saved window size from settings
    _win_w, _win_h = WINDOW_WIDTH, WINDOW_HEIGHT
    try:
        _settings_path = os.path.join(
            os.path.expanduser("~"), ".poe2-price-overlay", "dashboard_settings.json"
        )
        if os.path.exists(_settings_path):
            with open(_settings_path) as _sf:
                _saved = json.load(_sf)
            _win_w = max(WINDOW_WIDTH, int(_saved.get("window_width", WINDOW_WIDTH)))
            _win_h = max(WINDOW_HEIGHT, int(_saved.get("window_height", WINDOW_HEIGHT)))
    except Exception:
        pass

    # Clamp to the monitor work area (excludes the taskbar) and center the
    # window. We intentionally do NOT restore a maximized state: a frameless
    # pywebview window created maximized lands oversized and off-screen on
    # Windows. The title-bar maximize button still works at runtime.
    _win_x, _win_y = None, None
    try:
        import ctypes
        from ctypes import wintypes
        _rect = wintypes.RECT()
        # SPI_GETWORKAREA = 0x0030 → usable desktop minus taskbar
        if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(_rect), 0):
            _wa_w = _rect.right - _rect.left
            _wa_h = _rect.bottom - _rect.top
            _margin = 60
            _win_w = max(WINDOW_WIDTH, min(_win_w, _wa_w - _margin))
            _win_h = max(WINDOW_HEIGHT, min(_win_h, _wa_h - _margin))
            _win_x = _rect.left + (_wa_w - _win_w) // 2
            _win_y = _rect.top + (_wa_h - _win_h) // 2
    except Exception:
        pass

    # Open the native window pointing at the dashboard
    api = WindowApi()

    # --- System tray icon ---------------------------------------------------
    from tray import TrayIcon

    def _start_overlay():
        try:
            urllib.request.urlopen(
                Request(f"http://127.0.0.1:{PORT}/api/start",
                        method="POST", data=b"{}",
                        headers={"Content-Type": "application/json"}),
                timeout=5,
            )
        except Exception:
            pass

    def _stop_overlay():
        try:
            urllib.request.urlopen(
                Request(f"http://127.0.0.1:{PORT}/api/stop", method="POST"),
                timeout=5,
            )
        except Exception:
            pass

    def _get_overlay_state():
        try:
            raw = urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/api/status", timeout=2
            ).read()
            return json.loads(raw).get("state", "stopped")
        except Exception:
            return "stopped"

    tray = TrayIcon(
        on_show=api.show,
        on_start_overlay=_start_overlay,
        on_stop_overlay=_stop_overlay,
        on_quit=api.quit,
        get_overlay_state=_get_overlay_state,
    )
    tray.start()
    api._tray = tray
    _log(f"[Startup] Tray started ({_ms()})")

    # Tooltip updater (daemon thread)
    threading.Thread(target=_tooltip_updater, args=(tray,), daemon=True).start()

    # -------------------------------------------------------------------------
    # WebView2 refuses to initialise its render surface in a hidden window,
    # so we create the window visible.  The dashboard's own splash screen
    # covers the UI until data arrives, keeping the experience smooth.
    # A background thread still sets the taskbar icon via IPropertyStore.
    _win_kwargs = dict(
        url=f"http://127.0.0.1:{PORT}/dashboard?_t={int(time.time())}",
        width=_win_w,
        height=_win_h,
        min_size=(900, 600),
        background_color="#0d0b08",
        text_select=True,
        frameless=True,
        easy_drag=False,
        js_api=api,
    )
    if _win_x is not None and _win_y is not None:
        _win_kwargs["x"] = _win_x
        _win_kwargs["y"] = _win_y
    window = webview.create_window(WINDOW_TITLE, **_win_kwargs)

    def _on_shown():
        """Install resize hook and taskbar icon once the window is shown."""
        _log(f"[Startup] Window shown event fired ({_ms()})")
        # Install resize hook — must happen after window exists
        # Retry a few times since the HWND may not be ready immediately
        for attempt in range(10):
            hwnd = api._get_hwnd()
            if hwnd:
                api._install_hook()
                _strip_window_border(hwnd)
                _log(f"[Startup] Resize hook installed (hwnd={hwnd}, attempt={attempt})")
                break
            time.sleep(0.2)
        else:
            _log("[Startup] WARNING: Could not find window for resize hook")
        # Set taskbar icon in background
        threading.Thread(
            target=_set_icon_and_show, args=(api._get_hwnd, lambda: None, api, _ms), daemon=True
        ).start()

    window.events.shown += _on_shown

    # This blocks until the window is destroyed (force_close / quit)
    from bundle_paths import get_resource
    ico_path = str(get_resource("resources/img/favicon.ico"))
    webview.start(icon=ico_path, debug=_debug_mode)

    try:
        tray.stop()
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    if "--overlay-worker" in sys.argv:
        # Frozen-mode dispatch: server.py spawns this exe with --overlay-worker
        # to run the overlay subprocess within the single-exe bundle.
        sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a != "--overlay-worker"]
        from main import main as overlay_main
        overlay_main()
    elif "--restart" in sys.argv:
        # Strip --restart so it doesn't confuse webview, but handle it in main()
        main()
    else:
        main()
