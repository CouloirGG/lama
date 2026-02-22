"""
LAMA - Bug Reporter
Discord webhook-based bug reporting triggered by Ctrl+Shift+B.
Opens a dark-themed dialog, collects logs + system info, uploads to Discord.

When a recently priced item exists, defaults to "Price Inaccuracy" mode with
auto-populated item data. User can toggle to "General Bug" mode.
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

    def __init__(self, root_fn, stats_fn, overlay, item_context_fn=None):
        """
        Args:
            root_fn: callable returning the tkinter Tk instance
            stats_fn: callable returning session stats dict
            overlay: PriceOverlay instance for confirmation display
            item_context_fn: callable returning last priced item dict or None
        """
        self._root_fn = root_fn
        self._stats_fn = stats_fn
        self._overlay = overlay
        self._item_context_fn = item_context_fn
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

        # Fetch item context for price inaccuracy mode
        item_ctx = None
        if self._item_context_fn:
            try:
                item_ctx = self._item_context_fn()
            except Exception:
                pass

        dialog = tk.Toplevel(root)
        dialog.title("Bug Report")
        dialog.configure(bg="#1a1a2e")
        dialog.resizable(False, False)

        # Size and center — taller when showing item context
        w = 400
        h = 420 if item_ctx else 350
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

        # --- Category toggle state ---
        # "price" when Price Inaccuracy selected, "bug" when General Bug
        category = {"current": "price" if item_ctx else "bug"}

        # Container for everything below the pills (rebuilt on toggle)
        body_frame = tk.Frame(dialog, bg="#1a1a2e")
        body_frame.pack(fill="both", expand=True, padx=0, pady=0)

        # Mutable refs for title/desc widgets (rebuilt on toggle)
        widgets = {"title_var": None, "desc_text": None, "title_entry": None}

        def _build_body():
            """Build the body content based on current category."""
            for child in body_frame.winfo_children():
                child.destroy()

            title_var = tk.StringVar()
            widgets["title_var"] = title_var

            if category["current"] == "price" and item_ctx:
                # --- Item summary (read-only) ---
                item_name = item_ctx.get("item_name", "Unknown")
                base_type = item_ctx.get("base_type", "")
                name_line = item_name
                if base_type and base_type != item_name:
                    name_line = f"{item_name} -- {base_type}"

                tk.Label(
                    body_frame, text=name_line,
                    bg="#1a1a2e", fg="#ffd700", font=("Segoe UI", 11, "bold"),
                    wraplength=360, justify="left",
                ).pack(anchor="w", padx=16, pady=(4, 2))

                grade = item_ctx.get("grade", "?")
                display_text = item_ctx.get("display_text", "")
                summary = display_text or f"Grade: {grade}"
                tk.Label(
                    body_frame, text=summary,
                    bg="#1a1a2e", fg="#a0a0b0", font=("Segoe UI", 9),
                    wraplength=360, justify="left",
                ).pack(anchor="w", padx=16, pady=(0, 2))

                # Top mods (muted)
                mod_details = item_ctx.get("mod_details")
                if mod_details and isinstance(mod_details, list):
                    top_mods = [m.get("display", m.get("name", ""))
                                for m in mod_details[:3] if isinstance(m, dict)]
                    if top_mods:
                        tk.Label(
                            body_frame, text=", ".join(top_mods),
                            bg="#1a1a2e", fg="#707090", font=("Segoe UI", 8),
                            wraplength=360, justify="left",
                        ).pack(anchor="w", padx=16, pady=(0, 6))

                # Title pre-filled
                title_var.set(f"Price: {item_name}")
                desc_label_text = "What should the price be?"
                info_label_text = "Logs + item data attached automatically"
            else:
                # General bug mode — standard labels
                title_var.set("")
                desc_label_text = "What happened?"
                info_label_text = "Logs + system info attached automatically"

            # --- Title field ---
            tk.Label(
                body_frame, text="Title:", bg="#1a1a2e", fg="#e0e0e0",
                font=("Segoe UI", 10),
            ).pack(anchor="w", padx=16, pady=(0, 4))

            title_entry = tk.Entry(
                body_frame, textvariable=title_var,
                bg="#2a2a3e", fg="#e0e0e0", insertbackground="#e0e0e0",
                font=("Segoe UI", 10), relief="flat", bd=4,
            )
            title_entry.pack(fill="x", padx=16)
            widgets["title_entry"] = title_entry

            # --- Description field ---
            tk.Label(
                body_frame, text=desc_label_text, bg="#1a1a2e", fg="#e0e0e0",
                font=("Segoe UI", 10),
            ).pack(anchor="w", padx=16, pady=(12, 4))

            desc_text = tk.Text(
                body_frame, height=4 if (category["current"] == "price" and item_ctx) else 5,
                wrap="word",
                bg="#2a2a3e", fg="#e0e0e0", insertbackground="#e0e0e0",
                font=("Segoe UI", 10), relief="flat", bd=4,
            )
            desc_text.pack(fill="x", padx=16)
            widgets["desc_text"] = desc_text

            # --- Info label ---
            tk.Label(
                body_frame, text=info_label_text,
                bg="#1a1a2e", fg="#707090", font=("Segoe UI", 9),
            ).pack(anchor="w", padx=16, pady=(8, 0))

            # --- Keyboard bindings ---
            title_entry.bind("<Return>", lambda e: desc_text.focus_set())
            desc_text.bind("<Control-Return>", lambda e: _send())

            # Focus title
            title_entry.focus_force()
            title_entry.select_range(0, "end")

        # --- Category pills (only when item context available) ---
        if item_ctx:
            pill_frame = tk.Frame(dialog, bg="#1a1a2e")
            # Insert pill_frame right after the timestamp, before body_frame
            pill_frame.pack(after=dialog.winfo_children()[0], anchor="w",
                            padx=16, pady=(0, 8))

            active_bg = "#3a3a5e"
            active_fg = "#e0e0e0"
            inactive_bg = "#2a2a3e"
            inactive_fg = "#707090"

            price_pill = tk.Button(pill_frame, text="Price Inaccuracy",
                                   relief="flat", bd=0, padx=12, pady=4,
                                   font=("Segoe UI", 9, "bold"), cursor="hand2")
            price_pill.pack(side="left", padx=(0, 6))

            bug_pill = tk.Button(pill_frame, text="General Bug",
                                 relief="flat", bd=0, padx=12, pady=4,
                                 font=("Segoe UI", 9), cursor="hand2")
            bug_pill.pack(side="left")

            def _update_pill_styles():
                if category["current"] == "price":
                    price_pill.configure(bg=active_bg, fg=active_fg,
                                         activebackground=active_bg,
                                         font=("Segoe UI", 9, "bold"))
                    bug_pill.configure(bg=inactive_bg, fg=inactive_fg,
                                       activebackground=inactive_bg,
                                       font=("Segoe UI", 9))
                else:
                    price_pill.configure(bg=inactive_bg, fg=inactive_fg,
                                         activebackground=inactive_bg,
                                         font=("Segoe UI", 9))
                    bug_pill.configure(bg=active_bg, fg=active_fg,
                                       activebackground=active_bg,
                                       font=("Segoe UI", 9, "bold"))

            def _select_price():
                if category["current"] != "price":
                    category["current"] = "price"
                    _update_pill_styles()
                    _build_body()

            def _select_bug():
                if category["current"] != "bug":
                    category["current"] = "bug"
                    _update_pill_styles()
                    _build_body()

            price_pill.configure(command=_select_price)
            bug_pill.configure(command=_select_bug)
            _update_pill_styles()

        # Build initial body
        _build_body()

        # --- Send / Cancel ---
        def _send():
            title = widgets["title_var"].get().strip() or f"Bug report {time.strftime('%Y-%m-%d %H:%M')}"
            description = widgets["desc_text"].get("1.0", "end").strip()
            cat = category["current"]
            dialog.destroy()
            self._last_report_time = time.time()
            data = self._collect_data()
            # Attach item context for price inaccuracy reports
            if cat == "price" and item_ctx:
                data["item_context"] = item_ctx
            data["category"] = "price_inaccuracy" if cat == "price" else "bug"
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

        dialog.bind("<Escape>", lambda e: _cancel())

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
            "category": data.get("category", "bug"),
            "system_info": data["system_info"],
            "session_stats": data["session_stats"],
            "clipboard_count": len(data["clipboards"]),
        }
        # Add item context fields for price inaccuracy reports
        item_ctx = data.get("item_context")
        if item_ctx and data.get("category") == "price_inaccuracy":
            record["item_name"] = item_ctx.get("item_name")
            record["base_type"] = item_ctx.get("base_type")
            record["grade"] = item_ctx.get("grade")
            record["price_divine"] = item_ctx.get("price_divine")
            record["user_correction"] = description
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

        is_price = data.get("category") == "price_inaccuracy"
        item_ctx = data.get("item_context")
        prefix = "[PRICE]" if is_price else "[BUG]"

        # Build message content (max 2000 chars for Discord)
        message = f"**{prefix} {title}**"
        if is_price and item_ctx:
            grade = item_ctx.get("grade", "?")
            price = item_ctx.get("price_divine")
            price_str = f"{price:.1f}d" if price else "?"
            message += f"\nGrade: {grade} | Our estimate: {price_str}"
            if description:
                message += f"\nUser correction: **{description}**"
        elif description:
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
        # Attach item clipboard text for price inaccuracy reports
        if is_price and item_ctx:
            clipboard_text = item_ctx.get("clipboard_text")
            if clipboard_text:
                parts.append("=== ITEM CLIPBOARD ===\n")
                parts.append(clipboard_text)
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
