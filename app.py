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
    window = webview.create_window(
        WINDOW_TITLE,
        url=f"http://127.0.0.1:{PORT}/dashboard",
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        min_size=(900, 600),
        background_color="#0d0b08",
        text_select=True,
    )

    # This blocks until the window is closed
    webview.start()

    print("Window closed. Shutting down.")
    # Daemon thread dies automatically when main exits
    os._exit(0)


if __name__ == "__main__":
    main()
