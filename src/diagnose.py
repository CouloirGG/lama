"""
LAMA - Diagnostics
Run this WITH POE2 open to test each pipeline step independently.
Reports exactly where the pipeline is failing.

Usage:
    python diagnose.py
"""

import sys
import os
import time
import ctypes
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Colors for console output ──────────────────────
class C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"

def ok(msg):   print(f"  {C.GREEN}[PASS]{C.END} {msg}")
def fail(msg): print(f"  {C.RED}[FAIL]{C.END} {msg}")
def warn(msg): print(f"  {C.YELLOW}[WARN]{C.END} {msg}")
def info(msg): print(f"  {C.CYAN}[INFO]{C.END} {msg}")


def header(title):
    print(f"\n{C.BOLD}{'='*55}{C.END}")
    print(f"{C.BOLD}  {title}{C.END}")
    print(f"{C.BOLD}{'='*55}{C.END}\n")


def test_python_environment():
    header("Step 1: Python Environment")

    print(f"  Python: {sys.version}")
    print(f"  Platform: {sys.platform}")
    print(f"  Executable: {sys.executable}")

    modules = {
        "requests": "HTTP client",
    }

    all_ok = True
    for mod, desc in modules.items():
        try:
            m = __import__(mod)
            version = getattr(m, "__version__", "?")
            ok(f"{desc}: {mod} v{version}")
        except ImportError:
            fail(f"{desc}: {mod} NOT INSTALLED")
            all_ok = False

    # Check tkinter
    try:
        import tkinter as tk
        ok(f"Overlay rendering: tkinter available")
    except ImportError:
        fail("Overlay rendering: tkinter NOT AVAILABLE")
        all_ok = False

    # Check pywin32
    if sys.platform == "win32":
        try:
            import win32clipboard
            ok("Clipboard access: pywin32 available")
        except ImportError:
            fail("Clipboard access: pywin32 NOT INSTALLED")
            all_ok = False

    return all_ok


def test_game_window():
    header("Step 2: POE2 Window Detection")

    if sys.platform != "win32":
        warn("Not on Windows - skipping window detection")
        return True

    try:
        user32 = ctypes.windll.user32

        # Check foreground window
        hwnd = user32.GetForegroundWindow()
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        fg_title = buf.value

        info(f"Current foreground window: '{fg_title}'")

        # Enumerate all windows to find POE2
        import ctypes.wintypes as wt

        poe2_found = False
        window_titles = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        def enum_callback(hwnd, lparam):
            nonlocal poe2_found
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    title = buf.value
                    if title.strip():
                        window_titles.append(title)
                        if "path of exile" in title.lower():
                            poe2_found = True
                            ok(f"POE2 window found: '{title}'")

                            # Get window rect
                            rect = wt.RECT()
                            user32.GetWindowRect(hwnd, ctypes.byref(rect))
                            info(f"Window position: ({rect.left}, {rect.top}) to ({rect.right}, {rect.bottom})")
                            info(f"Window size: {rect.right - rect.left}x{rect.bottom - rect.top}")
            return True

        user32.EnumWindows(enum_callback, 0)

        if not poe2_found:
            fail("POE2 window NOT FOUND")
            info("Visible windows containing 'path' or 'exile' or 'poe':")
            matches = [t for t in window_titles if any(
                k in t.lower() for k in ["path", "exile", "poe", "grinding"]
            )]
            if matches:
                for t in matches:
                    print(f"    → '{t}'")
            else:
                warn("No matching windows. Is POE2 running?")
                info("All visible window titles:")
                for t in sorted(window_titles)[:20]:
                    print(f"    → '{t}'")

            print()
            info("If POE2's window title is different than expected,")
            info("update POE2_WINDOW_TITLE in src/config.py")
            return False

        return True

    except Exception as e:
        fail(f"Window detection error: {e}")
        return False


def test_cursor_tracking():
    header("Step 3: Cursor Tracking")

    if sys.platform != "win32":
        warn("Not on Windows - skipping cursor tracking")
        return True

    try:
        user32 = ctypes.windll.user32

        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        pt = POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        ok(f"Cursor position: ({pt.x}, {pt.y})")

        print()
        info("Move your mouse and press Enter to test tracking...")
        input()

        positions = []
        for i in range(5):
            user32.GetCursorPos(ctypes.byref(pt))
            positions.append((pt.x, pt.y))
            time.sleep(0.1)

        if len(set(positions)) > 1:
            ok(f"Cursor tracking working: {positions}")
        else:
            ok(f"Cursor tracking working (cursor was still): {positions[0]}")

        return True

    except Exception as e:
        fail(f"Cursor tracking failed: {e}")
        return False


def test_price_cache():
    header("Step 4: Price Data (poe.ninja)")

    from config import DEFAULT_LEAGUE
    info(f"Configured league: '{DEFAULT_LEAGUE}'")

    # Check what league the user has saved
    league_file = os.path.join(os.path.expanduser("~"), ".poe2-price-overlay", "league.txt")
    if os.path.exists(league_file):
        with open(league_file) as f:
            saved_league = f.read().strip()
        info(f"Saved league file: '{saved_league}'")
    else:
        warn("No league.txt found - using default")
        saved_league = DEFAULT_LEAGUE

    # Try fetching from POE2 API (correct endpoint)
    import requests

    # Confirmed working endpoint (Feb 2026)
    poe2_url = "https://poe.ninja/poe2/api/economy/exchange/current/overview"

    info(f"Testing POE2 API: {poe2_url}")
    info(f"League parameter: '{saved_league}'")

    poe2_ok = False

    try:
        resp = requests.get(poe2_url,
            params={"league": saved_league, "type": "Currency"},
            timeout=15, headers={"User-Agent": "LAMA/1.0"})
        info(f"POE2 API HTTP status: {resp.status_code}")

        if resp.status_code == 200 and resp.content:
            data = resp.json()
            lines = data.get("lines", [])
            core = data.get("core", {})
            items = core.get("items", [])
            rates = core.get("rates", {})

            if lines:
                ok(f"POE2 API: {len(lines)} currencies, {len(items)} item details")
                if rates:
                    info(f"Rates: 1 Divine = {rates.get('chaos', '?')} Chaos, {rates.get('exalted', '?')} Exalted")

                id_map = {i.get("id", ""): i.get("name", "") for i in items}
                chaos_rate = rates.get("chaos", 68)
                for line in lines[:5]:
                    lid = line.get("id", "")
                    pv = line.get("primaryValue", 0)
                    name = id_map.get(lid, lid)
                    chaos = pv * chaos_rate
                    print(f"    {name}: {pv:.4f} div ({chaos:.1f} chaos)")
                poe2_ok = True
            else:
                warn(f"POE2 API returned 0 lines for '{saved_league}'")
        else:
            warn(f"POE2 API returned HTTP {resp.status_code}")
    except requests.ConnectionError:
        warn("Cannot connect to POE2 API")
    except Exception as e:
        warn(f"POE2 API error: {e}")

    if not poe2_ok:
        fail("POE2 API returned no data")
        warn(f"League name '{saved_league}' may be wrong")
        info("Check poe.ninja/poe2/economy for correct league names")

        # Try to discover available league names
        try:
            info("\nTrying to discover available leagues...")
            for test_league in ["Fate of the Vaal", "Standard", "Hardcore"]:
                r = requests.get(poe2_url,
                    params={"league": test_league, "type": "Currency"},
                    timeout=10, headers={"User-Agent": "LAMA/1.0"})
                if r.status_code == 200 and r.content:
                    d = r.json()
                    n = len(d.get("lines", []))
                    if n > 0:
                        print(f"    '{test_league}' → {n} currencies ✓")
                    else:
                        print(f"    '{test_league}' → 0 currencies ✗")
        except Exception:
            pass

        return False

    return True


def test_overlay():
    header("Step 5: Overlay Window")

    if sys.platform != "win32":
        warn("Not on Windows - overlay won't render")
        return True

    try:
        import tkinter as tk

        root = tk.Tk()
        root.title("POE2 Overlay Test")
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.92)

        # Test transparent window
        transparent_color = "#010101"
        root.configure(bg=transparent_color)
        try:
            root.attributes("-transparentcolor", transparent_color)
            ok("Transparent window supported")
        except tk.TclError:
            warn("Transparent window NOT supported on this system")

        # Show test label
        label = tk.Label(
            root,
            text=" ~ 8 Exalted ",
            font=("Segoe UI", 14, "bold"),
            fg="#ffd700",
            bg="#1a1a2e",
            padx=8,
            pady=4,
        )
        label.pack()

        # Position in center of screen
        root.update_idletasks()
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        x = sw // 2 - 50
        y = sh // 2 - 20
        root.geometry(f"+{x}+{y}")

        info(f"Showing test overlay at ({x}, {y}) for 3 seconds...")
        root.deiconify()
        root.update()

        time.sleep(3)
        root.destroy()

        ok("Overlay window rendered successfully")
        info("Did you see a gold '~ 8 Exalted' tag in the center of your screen?")

        return True

    except Exception as e:
        fail(f"Overlay test failed: {e}")
        return False


def main():
    print()
    print(f"{C.BOLD}╔══════════════════════════════════════════════════════╗{C.END}")
    print(f"{C.BOLD}║       LAMA — Diagnostic Tool          ║{C.END}")
    print(f"{C.BOLD}╚══════════════════════════════════════════════════════╝{C.END}")
    print()
    print("  This will test each component of the overlay pipeline.")
    print("  Keep POE2 running in Windowed Fullscreen while testing.")
    print()

    results = []

    results.append(("Python Environment", test_python_environment()))
    results.append(("POE2 Window Detection", test_game_window()))
    results.append(("Cursor Tracking", test_cursor_tracking()))
    results.append(("Price Data (poe.ninja)", test_price_cache()))
    results.append(("Overlay Window", test_overlay()))

    # ─── Summary ─────────────────────────────────────
    header("DIAGNOSTIC SUMMARY")

    all_pass = True
    for name, passed in results:
        if passed:
            print(f"  {C.GREEN}✓{C.END} {name}")
        else:
            print(f"  {C.RED}✗{C.END} {name}")
            all_pass = False

    print()
    if all_pass:
        print(f"  {C.GREEN}{C.BOLD}All checks passed!{C.END}")
        print("  If the overlay still isn't showing prices, run with:")
        print("    python src/main.py --debug --console")
        print("  to see detailed pipeline output.")
    else:
        print(f"  {C.RED}{C.BOLD}Some checks failed.{C.END} Fix the issues above and re-run.")

    print()
    input("Press Enter to exit...")
    return 0 if all_pass else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nDiagnostics cancelled.")
