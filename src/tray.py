"""
tray.py -- System tray icon for LAMA.

Uses pystray with run_detached() so it coexists with pywebview's
main-thread event loop.  All callbacks run on pystray's background thread.
"""

import logging

logger = logging.getLogger("tray")


class TrayIcon:
    """Pystray wrapper with overlay-aware menu."""

    def __init__(self, on_show, on_start_overlay, on_stop_overlay, on_quit,
                 get_overlay_state):
        self._on_show = on_show
        self._on_start_overlay = on_start_overlay
        self._on_stop_overlay = on_stop_overlay
        self._on_quit = on_quit
        self._get_overlay_state = get_overlay_state
        self._icon = None

    # ------------------------------------------------------------------
    def start(self):
        """Create and start the tray icon (non-blocking)."""
        try:
            import pystray
            from PIL import Image
        except ImportError:
            logger.warning("pystray/Pillow not installed -- tray icon disabled")
            return

        from bundle_paths import get_resource

        # Load the LAMA icon, resize to 64x64 for the tray
        icon_path = get_resource("resources/img/lama_icon.png")
        try:
            image = Image.open(str(icon_path)).resize((64, 64), Image.LANCZOS)
        except Exception:
            # Fallback: solid orange square
            image = Image.new("RGB", (64, 64), (200, 150, 50))

        def _overlay_label(item):
            try:
                state = self._get_overlay_state()
            except Exception:
                state = "stopped"
            return "Stop Overlay" if state == "running" else "Start Overlay"

        def _toggle_overlay(icon, item):
            try:
                state = self._get_overlay_state()
            except Exception:
                state = "stopped"
            if state == "running":
                self._on_stop_overlay()
            else:
                self._on_start_overlay()

        menu = pystray.Menu(
            pystray.MenuItem("Show Dashboard", lambda icon, item: self._on_show(),
                             default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(_overlay_label, _toggle_overlay),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", lambda icon, item: self._on_quit()),
        )

        self._icon = pystray.Icon("LAMA", image, "LAMA", menu)
        self._icon.run_detached()
        logger.info("Tray icon started")

    # ------------------------------------------------------------------
    def stop(self):
        """Remove the tray icon."""
        if self._icon:
            try:
                self._icon.stop()
            except Exception:
                pass
            self._icon = None

    # ------------------------------------------------------------------
    def update_tooltip(self, text):
        """Update the hover tooltip text."""
        if self._icon:
            self._icon.title = text
