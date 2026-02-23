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
import sys
import threading
import time
import urllib.request
from urllib.request import Request

# Ensure src/ is on sys.path so bare imports and uvicorn "server:app" work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _log(msg):
    """Print that won't crash under pythonw (sys.stdout is None)."""
    try:
        print(msg)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PORT = int(os.environ.get("POE2_DASHBOARD_PORT", "8450"))
WINDOW_TITLE = "LAMA"
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 750


def start_server():
    """Run the FastAPI server in a background thread."""
    import uvicorn
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
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

        RESIZE_BORDER = 6  # pixels from edge that trigger resize cursor

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
        import ctypes
        hwnd = self._get_hwnd()
        if hwnd:
            self._guard_until = 0  # allow maximize resize
            if ctypes.windll.user32.IsZoomed(hwnd):
                ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            else:
                ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE

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

    # Now reveal the window — user only ever sees the LAMA icon
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
    # Start hidden so the Python icon never flashes in the taskbar.
    # The icon thread sets IPropertyStore, then reveals the window.
    window = webview.create_window(
        WINDOW_TITLE,
        url=f"http://127.0.0.1:{PORT}/dashboard?_t={int(time.time())}",
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        min_size=(900, 600),
        background_color="#0d0b08",
        text_select=True,
        frameless=True,
        easy_drag=False,
        js_api=api,
        hidden=True,
    )

    # Set the taskbar icon, then show the window
    threading.Thread(
        target=_set_icon_and_show, args=(api._get_hwnd, api.show, api, _ms), daemon=True
    ).start()

    # This blocks until the window is destroyed (force_close / quit)
    from bundle_paths import get_resource
    ico_path = str(get_resource("resources/img/favicon.ico"))
    webview.start(icon=ico_path)

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
