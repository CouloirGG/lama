"""
LAMA - Bug Reporter
Discord webhook-based bug reporting triggered by Ctrl+Shift+B.
Opens a dark-themed dialog, collects logs + system info, uploads to Discord.
"""

import sys
import os
import time
import platform
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import tkinter as tk
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False

from config import (
    DISCORD_WEBHOOK_URL,
    BUG_REPORT_LOG_LINES,
    BUG_REPORT_MAX_CLIPBOARDS,
    BUG_REPORT_DB,
    DEBUG_DIR,
    LOG_FILE,
)


class BugReporter:
    """Collects logs + description and uploads to Discord via webhook."""

    _COOLDOWN = 30  # seconds between reports

    def __init__(self, root_fn, stats_fn, overlay):
        """
        Args:
            root_fn: callable returning the tkinter Tk instance
            stats_fn: callable returning session stats dict
            overlay: PriceOverlay instance for confirmation display
        """
        self._root_fn = root_fn
        self._stats_fn = stats_fn
        self._overlay = overlay
        self._last_report_time = 0

    def report(self):
        """Called from hotkey thread. Checks cooldown, schedules dialog on main thread."""
        now = time.time()
        if now - self._last_report_time < self._COOLDOWN:
            remaining = int(self._COOLDOWN - (now - self._last_report_time))
            logger.info(f"Bug report cooldown: {remaining}s remaining")
            return

        try:
            root = self._root_fn()
            if root:
                root.after(0, self._show_dialog)
        except Exception as e:
            logger.error(f"Bug report dialog failed: {e}")

    def _show_dialog(self):
        """Create modal bug report dialog (runs on main thread)."""
        if not TK_AVAILABLE:
            return

        root = self._root_fn()
        if not root:
            return

        dialog = tk.Toplevel(root)
        dialog.title("Bug Report")
        dialog.configure(bg="#1a1a2e")
        dialog.resizable(False, False)

        # Size and center
        w, h = 400, 350
        sx = root.winfo_screenwidth()
        sy = root.winfo_screenheight()
        dialog.geometry(f"{w}x{h}+{(sx - w) // 2}+{(sy - h) // 2}")

        # NOT transient(root) — the overlay root has WS_EX_TOOLWINDOW which
        # hides children from the taskbar, making the dialog unfindable.
        dialog.grab_set()
        dialog.attributes("-topmost", True)

        # Force above borderless-fullscreen games via Win32 SetWindowPos
        dialog.update_idletasks()
        try:
            import ctypes
            child_hwnd = dialog.winfo_id()
            hwnd = ctypes.windll.user32.GetParent(child_hwnd) or child_hwnd
            HWND_TOPMOST = -1
            SWP_NOSIZE = 0x0001
            SWP_NOMOVE = 0x0002
            SWP_SHOWWINDOW = 0x0040
            ctypes.windll.user32.SetWindowPos(
                hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW,
            )
        except Exception:
            pass

        dialog.focus_force()
        dialog.lift()

        # --- Timestamp header (read-only context) ---
        tk.Label(
            dialog, text=f"Bug report {time.strftime('%Y-%m-%d %H:%M')}",
            bg="#1a1a2e", fg="#707090", font=("Segoe UI", 9),
        ).pack(anchor="w", padx=16, pady=(16, 8))

        # --- Title field ---
        tk.Label(
            dialog, text="Title:", bg="#1a1a2e", fg="#e0e0e0",
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=16, pady=(0, 4))

        title_var = tk.StringVar()
        title_entry = tk.Entry(
            dialog, textvariable=title_var,
            bg="#2a2a3e", fg="#e0e0e0", insertbackground="#e0e0e0",
            font=("Segoe UI", 10), relief="flat", bd=4,
        )
        title_entry.pack(fill="x", padx=16)

        # --- Description field ---
        tk.Label(
            dialog, text="What happened?", bg="#1a1a2e", fg="#e0e0e0",
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=16, pady=(12, 4))

        desc_text = tk.Text(
            dialog, height=5, wrap="word",
            bg="#2a2a3e", fg="#e0e0e0", insertbackground="#e0e0e0",
            font=("Segoe UI", 10), relief="flat", bd=4,
        )
        desc_text.pack(fill="x", padx=16)

        # --- Info label ---
        tk.Label(
            dialog, text="Logs + system info attached automatically",
            bg="#1a1a2e", fg="#707090", font=("Segoe UI", 9),
        ).pack(anchor="w", padx=16, pady=(8, 0))

        # --- Buttons ---
        def _send():
            title = title_var.get().strip() or f"Bug report {time.strftime('%Y-%m-%d %H:%M')}"
            description = desc_text.get("1.0", "end").strip()
            dialog.destroy()
            self._last_report_time = time.time()
            data = self._collect_data()
            threading.Thread(
                target=self._upload, args=(title, description, data),
                daemon=True, name="BugReportUpload",
            ).start()

        def _cancel():
            dialog.destroy()

        btn_frame = tk.Frame(dialog, bg="#1a1a2e")
        btn_frame.pack(pady=(16, 16))

        send_btn = tk.Button(
            btn_frame, text="Send", command=_send,
            bg="#2a4a2e", fg="#e0e0e0", activebackground="#3a5a3e",
            font=("Segoe UI", 10, "bold"), relief="flat", bd=0,
            padx=20, pady=6, cursor="hand2",
        )
        send_btn.pack(side="left", padx=8)

        cancel_btn = tk.Button(
            btn_frame, text="Cancel", command=_cancel,
            bg="#4a2a2e", fg="#e0e0e0", activebackground="#5a3a3e",
            font=("Segoe UI", 10), relief="flat", bd=0,
            padx=20, pady=6, cursor="hand2",
        )
        cancel_btn.pack(side="left", padx=8)

        # --- Keyboard bindings ---
        title_entry.bind("<Return>", lambda e: desc_text.focus_set())
        desc_text.bind("<Control-Return>", lambda e: _send())
        dialog.bind("<Escape>", lambda e: _cancel())

        title_entry.focus_force()
        title_entry.select_range(0, "end")

    def _collect_data(self):
        """Gather log tail, clipboard captures, system info, session stats."""
        result = {
            "log_tail": "",
            "clipboards": [],
            "system_info": "",
            "session_stats": "",
        }

        # Log tail
        try:
            if LOG_FILE.exists():
                lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
                result["log_tail"] = "\n".join(lines[-BUG_REPORT_LOG_LINES:])
        except Exception as e:
            result["log_tail"] = f"(failed to read log: {e})"

        # Recent clipboard debug files
        try:
            if DEBUG_DIR.exists():
                clips = sorted(
                    DEBUG_DIR.glob("clipboard_*.txt"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )[:BUG_REPORT_MAX_CLIPBOARDS]
                for clip_path in clips:
                    try:
                        content = clip_path.read_text(encoding="utf-8", errors="replace")
                        result["clipboards"].append((clip_path.name, content))
                    except Exception:
                        pass
        except Exception:
            pass

        # System info
        try:
            screen_info = "unknown"
            try:
                import ctypes
                user32 = ctypes.windll.user32
                screen_info = f"{user32.GetSystemMetrics(0)}x{user32.GetSystemMetrics(1)}"
            except Exception:
                pass
            result["system_info"] = (
                f"Python {sys.version.split()[0]}, "
                f"{platform.platform()}, "
                f"Screen {screen_info}"
            )
        except Exception as e:
            result["system_info"] = f"(failed: {e})"

        # Session stats
        try:
            stats = self._stats_fn()
            uptime = time.time() - stats.get("start_time", time.time())
            total = stats.get("triggers", 0)
            hits = stats.get("successful_lookups", 0)
            rate = (hits / total * 100) if total > 0 else 0
            result["session_stats"] = (
                f"Uptime {uptime / 60:.0f}min, "
                f"{total} triggers, "
                f"{hits} prices ({rate:.0f}%)"
            )
        except Exception as e:
            result["session_stats"] = f"(failed: {e})"

        return result

    def _save_local(self, title, description, data):
        """Append bug report to local JSONL database."""
        import json
        record = {
            "ts": int(time.time()),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "title": title,
            "description": description,
            "system_info": data["system_info"],
            "session_stats": data["session_stats"],
            "clipboard_count": len(data["clipboards"]),
        }
        try:
            BUG_REPORT_DB.parent.mkdir(parents=True, exist_ok=True)
            with open(BUG_REPORT_DB, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning(f"Failed to save bug report locally: {e}")

    def _upload(self, title, description, data):
        """Save locally and POST bug report to Discord webhook."""
        self._save_local(title, description, data)

        try:
            import requests
        except ImportError:
            logger.error("Bug report failed: 'requests' package not installed")
            self._show_result("Report failed (no requests lib)", "low")
            return

        # Build message content (max 2000 chars for Discord)
        message = f"**Bug Report: {title}**"
        if description:
            message += f"\n{description}"
        message += f"\n\n**System:** {data['system_info']}"
        message += f"\n**Session:** {data['session_stats']}"
        message += f"\n\n`Look at bug: {title}`"
        if len(message) > 2000:
            message = message[:1997] + "..."

        # Build combined attachment file
        parts = []
        if data["log_tail"]:
            parts.append(f"=== LOG TAIL (last {BUG_REPORT_LOG_LINES} lines) ===\n")
            parts.append(data["log_tail"])
            parts.append("\n\n")
        for filename, content in data["clipboards"]:
            parts.append(f"=== {filename} ===\n")
            parts.append(content)
            parts.append("\n\n")

        combined = "".join(parts).encode("utf-8")

        if not DISCORD_WEBHOOK_URL:
            logger.info("Bug report saved locally (no Discord webhook configured)")
            self._show_result("Saved locally", "decent")
            return

        try:
            resp = requests.post(
                DISCORD_WEBHOOK_URL,
                data={"content": message},
                files={"file": ("bug_report.txt", combined, "text/plain")},
                timeout=15,
            )
            if resp.status_code in range(200, 300):
                logger.info("Bug report sent successfully")
                self._show_result("Report sent!", "decent")
            else:
                logger.error(f"Bug report failed: HTTP {resp.status_code}")
                self._show_result("Report failed", "low")
        except Exception as e:
            logger.error(f"Bug report upload error: {e}")
            self._show_result("Report failed", "low")

    def _show_result(self, text, tier):
        """Show result in overlay."""
        try:
            import ctypes
            pt_class = type("POINT", (ctypes.Structure,),
                            {"_fields_": [("x", ctypes.c_long), ("y", ctypes.c_long)]})
            pt = pt_class()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
            self._overlay.show_price(
                text=text, tier=tier,
                cursor_x=pt.x, cursor_y=pt.y,
            )
        except Exception:
            pass
