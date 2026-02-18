"""
bundle_paths.py — Frozen-mode-aware path resolution for PyInstaller.

Provides:
    IS_FROZEN  — True when running from a PyInstaller bundle
    APP_DIR    — sys._MEIPASS when frozen, source directory when running from source
    get_resource(relative_path) — resolves bundled resource files
"""

import sys
from pathlib import Path

IS_FROZEN = getattr(sys, "frozen", False)
APP_DIR = Path(sys._MEIPASS) if IS_FROZEN else Path(__file__).resolve().parent.parent


def get_resource(relative_path: str) -> Path:
    """Resolve a bundled resource file (dashboard.html, VERSION, etc.)."""
    return APP_DIR / relative_path
