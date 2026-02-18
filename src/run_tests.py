"""Spawn a PowerShell window per test module for visual monitoring.

Usage:
    python run_tests.py                  # all modules in parallel
    python run_tests.py --module mod_database  # single module
    python run_tests.py --sequential     # one at a time
"""

import os
import sys
import glob
import time
import argparse
import subprocess

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(PROJECT_DIR, "tests")


def discover_test_modules():
    """Find all test_*.py files in the tests/ directory."""
    pattern = os.path.join(TESTS_DIR, "test_*.py")
    modules = []
    for path in sorted(glob.glob(pattern)):
        basename = os.path.basename(path)
        name = basename.replace(".py", "")
        modules.append(name)
    return modules


def spawn_powershell(module_name, project_dir):
    """Spawn a PowerShell window running pytest for one test module.

    The window title is set to the module name, and the window
    stays open after completion with a pause prompt.
    """
    module_path = f"tests/{module_name}.py"
    title = f"POE2 Tests: {module_name}"

    # PowerShell command:
    # 1. Set window title
    # 2. Change to project directory
    # 3. Run pytest with verbose output
    # 4. Pause to keep window open
    ps_commands = (
        f"$Host.UI.RawUI.WindowTitle = '{title}'; "
        f"Set-Location '{project_dir}'; "
        f"Write-Host '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━' -ForegroundColor Cyan; "
        f"Write-Host '  {title}' -ForegroundColor Cyan; "
        f"Write-Host '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━' -ForegroundColor Cyan; "
        f"Write-Host ''; "
        f"python -m pytest {module_path} -v --tb=short --color=yes; "
        f"Write-Host ''; "
        f"Write-Host '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━' -ForegroundColor Cyan; "
        f"if ($LASTEXITCODE -eq 0) {{ "
        f"  Write-Host '  ALL PASSED' -ForegroundColor Green "
        f"}} else {{ "
        f"  Write-Host '  FAILURES DETECTED' -ForegroundColor Red "
        f"}}; "
        f"Write-Host '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━' -ForegroundColor Cyan; "
        f"Read-Host 'Press Enter to close'"
    )

    # CREATE_NEW_CONSOLE = 0x10
    CREATE_NEW_CONSOLE = 0x00000010

    proc = subprocess.Popen(
        ["powershell", "-NoExit", "-Command", ps_commands],
        creationflags=CREATE_NEW_CONSOLE,
    )
    return proc


def main():
    parser = argparse.ArgumentParser(description="POE2 OCR Test Runner")
    parser.add_argument(
        "--module", "-m",
        help="Run only this test module (e.g., 'mod_database' or 'test_mod_database')",
    )
    parser.add_argument(
        "--sequential", "-s",
        action="store_true",
        help="Run modules one at a time instead of in parallel",
    )
    args = parser.parse_args()

    project_dir = PROJECT_DIR
    modules = discover_test_modules()

    if not modules:
        print("No test modules found in tests/")
        sys.exit(1)

    # Filter to single module if requested
    if args.module:
        target = args.module
        if not target.startswith("test_"):
            target = f"test_{target}"
        matches = [m for m in modules if m == target]
        if not matches:
            print(f"Module '{args.module}' not found. Available: {', '.join(modules)}")
            sys.exit(1)
        modules = matches

    print(f"POE2 OCR Test Runner")
    print(f"{'=' * 50}")
    print(f"Launching {len(modules)} test module(s):")
    for m in modules:
        print(f"  - {m}")
    print()

    procs = []
    for module_name in modules:
        print(f"  Spawning: {module_name}")
        proc = spawn_powershell(module_name, project_dir)
        procs.append((module_name, proc))

        if args.sequential and len(modules) > 1:
            print(f"    Waiting for {module_name} to finish...")
            proc.wait()
            print(f"    {module_name} exited with code {proc.returncode}")
        else:
            # Small delay between spawns to stagger windows
            time.sleep(0.3)

    if not args.sequential:
        print(f"\n{len(procs)} PowerShell window(s) spawned.")
        print("Close them manually when done, or press Ctrl+C here.")

        try:
            for name, proc in procs:
                proc.wait()
        except KeyboardInterrupt:
            print("\nTerminating test windows...")
            for name, proc in procs:
                proc.terminate()


if __name__ == "__main__":
    main()
