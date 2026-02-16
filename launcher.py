"""
POE2 Price Overlay - Setup & Launcher
First-run: checks dependencies, creates shortcut.
Subsequent runs: launches the overlay directly.

Run this file to start the overlay:
    python launcher.py

Or double-click launcher.pyw (windowless version)
"""

import os
import sys
import subprocess
import ctypes
import logging
from pathlib import Path

# ─── Paths ───────────────────────────────────────────
APP_NAME = "POE2 Price Overlay"
APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = Path(os.path.expanduser("~")) / ".poe2-price-overlay"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
log = logging.getLogger(__name__)


def is_admin():
    """Check if running with admin privileges."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        return False


def check_python_deps():
    """Check and install Python dependencies."""
    required = ["requests"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        log.info(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            *missing, "--quiet"
        ])
        log.info("✓ Python dependencies installed")
    else:
        log.info("✓ Python dependencies OK")


def create_desktop_shortcut():
    """Create a desktop shortcut to the launcher."""
    try:
        desktop = Path(os.path.expanduser("~")) / "Desktop"
        if not desktop.exists():
            return

        # Create a .bat shortcut (simpler than .lnk, works everywhere)
        shortcut = desktop / "POE2 Price Overlay.bat"
        if shortcut.exists():
            return  # Don't overwrite

        bat_content = f'''@echo off
title {APP_NAME}
cd /d "{APP_DIR}"
python "{APP_DIR / 'main.py'}" %*
pause
'''
        shortcut.write_text(bat_content, encoding="utf-8")
        log.info(f"✓ Desktop shortcut created: {shortcut}")
    except Exception as e:
        log.warning(f"  Could not create shortcut: {e}")


def create_start_menu_entry():
    """Create Start Menu entry."""
    try:
        start_menu = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        if not start_menu.exists():
            return

        bat = start_menu / f"{APP_NAME}.bat"
        if bat.exists():
            return

        bat_content = f'''@echo off
cd /d "{APP_DIR}"
pythonw "{APP_DIR / 'main.py'}" %*
'''
        bat.write_text(bat_content, encoding="utf-8")
        log.info(f"✓ Start Menu entry created")
    except Exception:
        pass


def select_league():
    """Let user pick their league on first run."""
    league_file = DATA_DIR / "league.txt"

    if league_file.exists():
        return league_file.read_text(encoding="utf-8").strip()

    log.info("")
    log.info("  Which league are you playing?")
    log.info("  (You can change this later in Settings)")
    log.info("")
    log.info("  1. Fate of the Vaal (current temp league)")
    log.info("  2. Standard")
    log.info("  3. Hardcore Fate of the Vaal")
    log.info("  4. Hardcore")
    log.info("")

    leagues = {
        "1": "Fate of the Vaal",
        "2": "Standard",
        "3": "Hardcore Fate of the Vaal",
        "4": "Hardcore",
    }

    try:
        choice = input("  Enter number [1]: ").strip() or "1"
        league = leagues.get(choice, "Fate of the Vaal")
    except (EOFError, KeyboardInterrupt):
        league = "Fate of the Vaal"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    league_file.write_text(league, encoding="utf-8")
    log.info(f"  → Selected: {league}")
    return league


def first_run_setup():
    """Complete first-run setup."""
    marker = DATA_DIR / ".setup_complete"
    if marker.exists():
        return True

    log.info("")
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║       POE2 Price Overlay — First Run Setup       ║")
    log.info("╚══════════════════════════════════════════════════╝")
    log.info("")

    # 1. Check Python dependencies
    log.info("[1/3] Checking Python dependencies...")
    try:
        check_python_deps()
    except Exception as e:
        log.error(f"  Failed to install dependencies: {e}")
        log.error("  Try running: pip install -r requirements.txt")
        return False

    # 2. Select league
    log.info("[2/3] Configuring league...")
    select_league()

    # 3. Create shortcuts
    log.info("[3/3] Creating shortcuts...")
    create_desktop_shortcut()

    # Mark setup complete
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    marker.write_text("1", encoding="utf-8")

    log.info("")
    log.info("=" * 50)
    log.info("  ✓ Setup complete!")
    log.info("=" * 50)
    log.info("")
    log.info("  Tips:")
    log.info("  • Set POE2 to Windowed Fullscreen mode")
    log.info("  • Copy items with Ctrl+C in POE2 to get prices")
    log.info("  • Hover over items to see prices")
    log.info("")

    return True


def launch_overlay():
    """Launch the main overlay application."""
    # Read saved league
    league_file = DATA_DIR / "league.txt"
    league = "Fate of the Vaal"
    if league_file.exists():
        league = league_file.read_text(encoding="utf-8").strip()

    main_script = APP_DIR / "main.py"
    if not main_script.exists():
        log.error(f"Cannot find {main_script}")
        return 1

    log.info(f"Starting {APP_NAME} (League: {league})...")
    log.info("Hover over items in POE2 to see prices.")
    log.info("Press Ctrl+C to stop.\n")

    # Launch main.py with the selected league
    return subprocess.call([
        sys.executable, str(main_script),
        "--league", league,
    ])


def main():
    # Ensure we're on Windows
    if sys.platform != "win32":
        log.warning("This application is designed for Windows.")
        log.warning("Some features (overlay, cursor tracking) won't work on other platforms.")
        log.info("Running in console-only mode for testing...\n")

    try:
        # Run first-time setup if needed
        if not first_run_setup():
            input("\nPress Enter to exit...")
            return 1

        # Launch the overlay
        return launch_overlay()

    except KeyboardInterrupt:
        log.info("\nGoodbye!")
        return 0
    except Exception as e:
        log.error(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        return 1


if __name__ == "__main__":
    sys.exit(main())
