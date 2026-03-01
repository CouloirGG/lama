"""
LAMA - Harvest Monitor

Thin tkinter wrapper around harvest_scheduler.py that shows a small
always-on-top status window during harvest runs. Designed to be called
by harvest_cycle.bat instead of harvest_scheduler.py directly.

Shows:
- Live progress (current pass, records collected, elapsed time)
- Green/red result on completion
- Recent run history from accuracy_tracking.jsonl
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

# Paths
SRC_DIR = Path(__file__).resolve().parent
CACHE_DIR = Path(os.path.expanduser("~")) / ".poe2-price-overlay" / "cache"
ACCURACY_LOG = CACHE_DIR / "accuracy_tracking.jsonl"

# Theme colors (POE2-style dark)
BG = "#1a120c"
BG_HEADER = "#140e08"
FG = "#c8b88a"
FG_DIM = "#7a6b55"
BORDER_COLOR = "#3d2e1e"
GREEN = "#4a2"
YELLOW = "#ca0"
RED = "#c33"

# Auto-close delay (ms)
AUTO_CLOSE_MS = 8000


def load_recent_runs(n=6):
    """Load last N runs from accuracy_tracking.jsonl."""
    if not ACCURACY_LOG.exists():
        return []
    entries = []
    try:
        with open(ACCURACY_LOG, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        pass
    return entries[-n:]


class HarvestMonitor:
    def __init__(self, test_mode=False):
        self.test_mode = test_mode

        self.root = tk.Tk()
        self.root.title("LAMA Harvester")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        # Drag state
        self._drag_x = 0
        self._drag_y = 0

        # State
        self.start_time = time.time()
        self.status = "STARTING"  # STARTING, RUNNING, ACCURACY, COMPLETE, FAILED
        self.current_pass = ""
        self.query_progress = ""
        self.records_delta = ""
        self.records_total = ""
        self.process = None
        self.return_code = None

        self._build_ui()
        self._position_window()
        if test_mode:
            self._start_test_sequence()
        else:
            self._start_harvest()
        self._tick_timer()

    def _build_ui(self):
        root = self.root
        W = 350

        # Header frame (draggable)
        hdr = tk.Frame(root, bg=BG_HEADER, padx=10, pady=6)
        hdr.pack(fill="x")
        hdr.bind("<Button-1>", self._drag_start)
        hdr.bind("<B1-Motion>", self._drag_motion)

        title = tk.Label(hdr, text="LAMA Harvester", font=("Consolas", 11, "bold"),
                         bg=BG_HEADER, fg=FG)
        title.pack(side="left")
        title.bind("<Button-1>", self._drag_start)
        title.bind("<B1-Motion>", self._drag_motion)

        # Close button
        close_btn = tk.Label(hdr, text="\u2715", font=("Consolas", 11),
                             bg=BG_HEADER, fg=FG_DIM, cursor="hand2")
        close_btn.pack(side="right")
        close_btn.bind("<Button-1>", lambda e: self._close())

        self.status_dot = tk.Label(hdr, text="\u2b24", font=("Consolas", 10),
                                   bg=BG_HEADER, fg=YELLOW)
        self.status_dot.pack(side="right", padx=(0, 6))
        self.status_label = tk.Label(hdr, text="STARTING",
                                     font=("Consolas", 9),
                                     bg=BG_HEADER, fg=YELLOW)
        self.status_label.pack(side="right", padx=(0, 4))

        # Separator
        tk.Frame(root, bg=BORDER_COLOR, height=1).pack(fill="x")

        # Body frame
        body = tk.Frame(root, bg=BG, padx=12, pady=8)
        body.pack(fill="x")

        self.line_pass = tk.Label(body, text="Initializing...",
                                  font=("Consolas", 10), bg=BG, fg=FG,
                                  anchor="w", width=38)
        self.line_pass.pack(anchor="w")

        self.line_records = tk.Label(body, text="",
                                     font=("Consolas", 10), bg=BG, fg=FG,
                                     anchor="w", width=38)
        self.line_records.pack(anchor="w")

        self.line_timer = tk.Label(body, text="Elapsed: 0m 00s",
                                   font=("Consolas", 10), bg=BG, fg=FG_DIM,
                                   anchor="w", width=38)
        self.line_timer.pack(anchor="w")

        # Right-click on any body label copies all status text
        for widget in (self.line_pass, self.line_records, self.line_timer, body):
            widget.bind("<Button-3>", self._copy_status)

        # Separator
        tk.Frame(root, bg=BORDER_COLOR, height=1).pack(fill="x")

        # History frame
        hist = tk.Frame(root, bg=BG, padx=12, pady=6)
        hist.pack(fill="x")

        runs = load_recent_runs(6)
        if runs:
            dots = ""
            ok_count = 0
            for r in runs:
                pct = r.get("pct_2x", 0)
                if pct >= 40:
                    dots += "\u25cf"
                    ok_count += 1
                else:
                    dots += "\u25cb"
            hist_text = f"Recent:  {dots}  {ok_count}/{len(runs)} ok"
        else:
            hist_text = "Recent:  no history yet"

        tk.Label(hist, text=hist_text, font=("Consolas", 9),
                 bg=BG, fg=FG_DIM, anchor="w").pack(anchor="w")

        # Force geometry
        root.update_idletasks()
        root.minsize(W, 0)

    def _drag_start(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _drag_motion(self, event):
        x = self.root.winfo_x() + event.x - self._drag_x
        y = self.root.winfo_y() + event.y - self._drag_y
        self.root.geometry(f"+{x}+{y}")

    def _copy_status(self, event=None):
        """Copy all visible status text to clipboard."""
        lines = [
            f"[{self.status}]",
            self.line_pass.cget("text"),
            self.line_records.cget("text"),
            self.line_timer.cget("text"),
        ]
        text = "\n".join(l for l in lines if l)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        # Brief visual feedback — flash timer line
        self.line_timer.config(fg=FG)
        self.root.after(300, lambda: self.line_timer.config(fg=FG_DIM))

    def _close(self):
        """Close window and kill subprocess if still running."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _position_window(self):
        """Place window in bottom-right corner, above taskbar."""
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        # Use actual size (accounts for minsize), with generous margin
        ww = max(self.root.winfo_width(), self.root.winfo_reqwidth(), 350)
        wh = max(self.root.winfo_height(), self.root.winfo_reqheight())
        x = sw - ww - 24
        y = sh - wh - 80  # clear the taskbar
        self.root.geometry(f"{ww}x{wh}+{x}+{y}")

    def _start_test_sequence(self):
        """Simulate a harvest run with fake progress updates."""
        def _fake():
            steps = [
                (1, "RUNNING", "PASS\t1 of 15", None),
                (2, None, "[5/136] (4%)", None),
                (2, None, "[30/136] (22%)", None),
                (2, None, "[80/136] (59%)", "Samples collected: 247"),
                (1, None, "PASS\t5 of 15", None),
                (2, None, "[60/136] (44%)", None),
                (2, None, "[136/136] (100%)", "Samples collected: 183"),
                (1, None, None, "Cycle 1 harvest complete: +430 records (48684 total)"),
                (2, "ACCURACY", "ACCURACY CHECK", None),
                (3, None, "Within 2x:\t500/900\t(55.6%)", None),
                (2, "COMPLETE", None, None),
            ]
            for delay, status, pass_line, rec_line in steps:
                time.sleep(delay)
                if status:
                    self.root.after(0, self._set_status, status)
                if pass_line:
                    self.root.after(0, self._parse_line, pass_line)
                if rec_line:
                    self.root.after(0, self._parse_line, rec_line)
            self.return_code = 0
            # No auto-close in test mode — use the X button
        t = threading.Thread(target=_fake, daemon=True)
        t.start()

    def _start_harvest(self):
        """Launch harvest_scheduler.py --once in a background thread."""
        t = threading.Thread(target=self._run_subprocess, daemon=True)
        t.start()

    def _run_subprocess(self):
        """Run harvest_scheduler and stream stdout."""
        cmd = [
            sys.executable, "-u",
            str(SRC_DIR / "harvest_scheduler.py"),
            "--once", "--passes", "15",
        ]
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(SRC_DIR),
            )
            self.root.after(0, self._set_status, "RUNNING")

            for line in self.process.stdout:
                self.root.after(0, self._parse_line, line)

            self.process.wait()
            self.return_code = self.process.returncode

            if self.return_code == 0:
                self.root.after(0, self._set_status, "COMPLETE")
            else:
                self.root.after(0, self._set_status, "FAILED")

        except Exception:
            self.root.after(0, self._set_status, "FAILED")

        # Schedule auto-close
        self.root.after(AUTO_CLOSE_MS, self._auto_close)

    def _parse_line(self, line):
        """Parse a stdout line and update UI."""
        line = line.strip()
        if not line:
            return

        # "  PASS X of Y"
        m = re.search(r"PASS\s+(\d+)\s+of\s+(\d+)", line)
        if m:
            self.current_pass = f"Pass {m.group(1)} of {m.group(2)}"
            self.line_pass.config(text=self.current_pass)
            return

        # "[X/Y] (Z%)" — query progress within a pass
        m = re.search(r"\[(\d+)/(\d+)\]\s*\((\d+)%\)", line)
        if m:
            self.query_progress = f"  Query {m.group(1)}/{m.group(2)} ({m.group(3)}%)"
            self.line_pass.config(text=self.current_pass + self.query_progress
                                  if self.current_pass else self.query_progress)
            return

        # "HARVESTER: passes X-Y"
        m = re.search(r"HARVESTER:\s+passes\s+(\d+)-(\d+)", line)
        if m:
            self.line_pass.config(text=f"Passes {m.group(1)}-{m.group(2)}")
            return

        # "+N records (M total)" or "Samples collected: N"
        m = re.search(r"\+(\d+)\s+records\s+\((\S+)\s+total\)", line)
        if m:
            self.records_delta = m.group(1)
            self.records_total = m.group(2)
            self.line_records.config(
                text=f"+{self.records_delta} records  ({self.records_total} total)")
            return

        m = re.search(r"Samples collected:\s+(\d+)", line)
        if m:
            self.line_records.config(text=f"+{m.group(1)} records this pass")
            return

        # "Current shard records: N"
        m = re.search(r"Current shard records:\s+(\d+)", line)
        if m:
            self.records_total = m.group(1)
            self.line_records.config(text=f"{m.group(1)} records in shards")
            return

        # "ACCURACY CHECK"
        if "ACCURACY CHECK" in line:
            self._set_status("ACCURACY")
            self.line_pass.config(text="Running accuracy check...")
            return

        # "Within 2x: N/M (P%)"
        m = re.search(r"Within 2x:\s+\d+/\d+\s+\((\d+\.\d+)%\)", line)
        if m:
            self.line_pass.config(text=f"Accuracy: {m.group(1)}% within 2x")
            return

        # Rate limit skip
        if "rate limit active" in line.lower() or "Skipping this cycle" in line:
            self._set_status("FAILED")
            self.line_pass.config(text="Rate limited — skipped")
            return

    def _set_status(self, status):
        """Update the status indicator."""
        self.status = status
        colors = {
            "STARTING": YELLOW,
            "RUNNING": YELLOW,
            "ACCURACY": YELLOW,
            "COMPLETE": GREEN,
            "FAILED": RED,
        }
        labels = {
            "STARTING": "STARTING",
            "RUNNING": "RUNNING",
            "ACCURACY": "CHECKING",
            "COMPLETE": "COMPLETE",
            "FAILED": "FAILED",
        }
        color = colors.get(status, YELLOW)
        self.status_dot.config(fg=color)
        self.status_label.config(text=labels.get(status, status), fg=color)

        if status in ("COMPLETE", "FAILED"):
            self.root.configure(bg=color)
            # Flash back to normal after 500ms
            self.root.after(500, lambda: self.root.configure(bg=BG))

    def _tick_timer(self):
        """Update the elapsed time display every second."""
        elapsed = int(time.time() - self.start_time)
        m, s = divmod(elapsed, 60)
        self.line_timer.config(text=f"Elapsed: {m}m {s:02d}s")
        self.root.after(1000, self._tick_timer)

    def _auto_close(self):
        """Auto-close the window after completion."""
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def run(self):
        self.root.mainloop()


def main():
    test_mode = "--test" in sys.argv
    monitor = HarvestMonitor(test_mode=test_mode)
    monitor.run()
    # Return the subprocess exit code
    sys.exit(monitor.return_code or 0)


if __name__ == "__main__":
    main()
