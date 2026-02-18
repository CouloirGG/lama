"""
app.py — Standalone desktop launcher for POE2 Price Overlay.

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
WINDOW_TITLE = "POE2 Price Overlay"
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

    def minimize(self):
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, WINDOW_TITLE)
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 6)  # SW_MINIMIZE

    def toggle_maximize(self):
        import ctypes
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, WINDOW_TITLE)
        if hwnd:
            if user32.IsZoomed(hwnd):
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            else:
                user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE

    def close(self):
        import webview
        if webview.windows:
            webview.windows[0].destroy()


def main():
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
    print(f"Starting POE2 Price Overlay on port {PORT}...")
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
