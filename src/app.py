"""
app.py — Standalone desktop launcher for LAMA (Live Auction Market Assessor).

Starts the FastAPI server in a background thread and opens the
dashboard in a native window (no browser required).

Usage:
    python app.py

Requirements:
    pip install pywebview
"""

import os
import sys
import threading
import time

# Ensure src/ is on sys.path so bare imports and uvicorn "server:app" work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    import urllib.request
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

    def _get_hwnd(self):
        import ctypes
        return ctypes.windll.user32.FindWindowW(None, WINDOW_TITLE)

    def _install_hook(self):
        """Install a Win32 hook that silently blocks resize during guard periods."""
        import ctypes
        from ctypes import wintypes, WINFUNCTYPE, POINTER, c_int, c_uint

        hwnd = self._get_hwnd()
        if not hwnd or self._original_proc:
            return

        WM_WINDOWPOSCHANGING = 0x0046
        SWP_NOSIZE = 0x0001
        GWL_WNDPROC = -4

        class WINDOWPOS(ctypes.Structure):
            _fields_ = [
                ("hwnd", wintypes.HWND),
                ("hwndInsertAfter", wintypes.HWND),
                ("x", c_int), ("y", c_int),
                ("cx", c_int), ("cy", c_int),
                ("flags", c_uint),
            ]

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
            return user32.CallWindowProcW(api_ref._original_proc,
                                          hwnd, msg, wparam, lparam)

        self._hook_ref = hook_proc
        self._original_proc = user32.SetWindowLongPtrW(hwnd, GWL_WNDPROC,
                                                        hook_proc)

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
        import webview
        if webview.windows:
            webview.windows[0].destroy()


def _ensure_deps():
    """Auto-install missing dependencies (runs silently under pythonw)."""
    try:
        import webview  # noqa: F401
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


def main():
    _ensure_deps()
    try:
        import webview
    except ImportError:
        print("=" * 50)
        print("pywebview is required for standalone mode.")
        print("Install it with:")
        print()
        print("    pip install pywebview")
        print()
        print("Then re-run: python app.py")
        print("=" * 50)
        sys.exit(1)

    # If launched with --restart, wait for the old process to release the port
    if "--restart" in sys.argv:
        print("Restart requested — waiting for old process to release port...")
        if not wait_for_port_free():
            print("ERROR: Port not freed within 10 seconds.")
            sys.exit(1)

    # Start the server in a daemon thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Wait for it to be ready
    print(f"Starting LAMA on port {PORT}...")
    if not wait_for_server():
        print("ERROR: Server failed to start within 10 seconds.")
        sys.exit(1)

    print("Server ready. Opening window...")

    # Open the native window pointing at the dashboard
    api = WindowApi()
    window = webview.create_window(
        WINDOW_TITLE,
        url=f"http://127.0.0.1:{PORT}/dashboard",
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        min_size=(900, 600),
        background_color="#0d0b08",
        text_select=True,
        frameless=True,
        easy_drag=False,
        js_api=api,
    )

    # This blocks until the window is closed
    webview.start()

    print("Window closed. Shutting down.")
    # Daemon thread dies automatically when main exits
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
