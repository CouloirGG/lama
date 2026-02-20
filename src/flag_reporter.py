"""
LAMA - Flag Reporter
Lets users flag inaccurate price estimates via Ctrl+Shift+F.
Opens a small dialog, saves locally (JSONL), and uploads to Discord.
"""

import json
import time
import logging
import threading

logger = logging.getLogger(__name__)

try:
    import tkinter as tk
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False

from config import (
    DISCORD_FLAG_WEBHOOK_URL,
    FLAG_REPORT_DB,
    FLAG_REPORT_COOLDOWN,
)


class FlagReporter:
    """Collects inaccurate-price feedback and uploads to Discord."""

    def __init__(self, root_fn, overlay):
        """
        Args:
            root_fn: callable returning the tkinter Tk instance
            overlay: PriceOverlay instance for result feedback
        """
        self._root_fn = root_fn
        self._overlay = overlay
        self._last_flag_time = 0

    def flag(self, data: dict):
        """Called from hotkey thread. Checks cooldown, schedules dialog on main thread."""
        if not data:
            logger.info("Flag: no item data to flag")
            return

        now = time.time()
        if now - self._last_flag_time < FLAG_REPORT_COOLDOWN:
            remaining = int(FLAG_REPORT_COOLDOWN - (now - self._last_flag_time))
            logger.info(f"Flag cooldown: {remaining}s remaining")
            return

        try:
            root = self._root_fn()
            if root:
                root.after(0, lambda: self._show_dialog(data))
        except Exception as e:
            logger.error(f"Flag dialog failed: {e}")

    def _show_dialog(self, data: dict):
        """Create flag dialog (runs on main thread)."""
        if not TK_AVAILABLE:
            return

        root = self._root_fn()
        if not root:
            return

        dialog = tk.Toplevel(root)
        dialog.title("Flag Inaccurate Price")
        dialog.configure(bg="#1a1a2e")
        dialog.resizable(False, False)

        # Size and center
        w, h = 400, 260
        sx = root.winfo_screenwidth()
        sy = root.winfo_screenheight()
        dialog.geometry(f"{w}x{h}+{(sx - w) // 2}+{(sy - h) // 2}")

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

        # --- Item info (read-only) ---
        item_name = data.get("item_name", "Unknown")
        grade = data.get("grade", "?")
        display_text = data.get("display_text", "")

        tk.Label(
            dialog, text=item_name,
            bg="#1a1a2e", fg="#ffd700", font=("Segoe UI", 11, "bold"),
            wraplength=360, justify="left",
        ).pack(anchor="w", padx=16, pady=(16, 4))

        tk.Label(
            dialog, text=display_text or f"Grade: {grade}",
            bg="#1a1a2e", fg="#a0a0b0", font=("Segoe UI", 9),
            wraplength=360, justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 12))

        # --- Correction input ---
        tk.Label(
            dialog, text="Actual price (optional, e.g. \"50d\"):",
            bg="#1a1a2e", fg="#e0e0e0", font=("Segoe UI", 10),
        ).pack(anchor="w", padx=16, pady=(0, 4))

        correction_var = tk.StringVar()
        correction_entry = tk.Entry(
            dialog, textvariable=correction_var,
            bg="#2a2a3e", fg="#e0e0e0", insertbackground="#e0e0e0",
            font=("Segoe UI", 10), relief="flat", bd=4,
        )
        correction_entry.pack(fill="x", padx=16)

        # --- Buttons ---
        def _submit():
            correction = correction_var.get().strip() or None
            dialog.destroy()
            self._last_flag_time = time.time()
            threading.Thread(
                target=self._process_flag, args=(data, correction),
                daemon=True, name="FlagReportUpload",
            ).start()

        def _cancel():
            dialog.destroy()

        btn_frame = tk.Frame(dialog, bg="#1a1a2e")
        btn_frame.pack(pady=(16, 16))

        flag_btn = tk.Button(
            btn_frame, text="Flag", command=_submit,
            bg="#4a3a2e", fg="#e0e0e0", activebackground="#5a4a3e",
            font=("Segoe UI", 10, "bold"), relief="flat", bd=0,
            padx=20, pady=6, cursor="hand2",
        )
        flag_btn.pack(side="left", padx=8)

        cancel_btn = tk.Button(
            btn_frame, text="Cancel", command=_cancel,
            bg="#4a2a2e", fg="#e0e0e0", activebackground="#5a3a3e",
            font=("Segoe UI", 10), relief="flat", bd=0,
            padx=20, pady=6, cursor="hand2",
        )
        cancel_btn.pack(side="left", padx=8)

        # --- Keyboard bindings ---
        correction_entry.bind("<Return>", lambda e: _submit())
        dialog.bind("<Escape>", lambda e: _cancel())

        correction_entry.focus_force()

    def _process_flag(self, data: dict, correction: str):
        """Save locally and upload to Discord (runs in background thread)."""
        record = {
            "ts": int(time.time()),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "item_name": data.get("item_name"),
            "base_type": data.get("base_type"),
            "rarity": data.get("rarity"),
            "item_class": data.get("item_class"),
            "display_text": data.get("display_text"),
            "tier": data.get("tier"),
            "price_divine": data.get("price_divine"),
            "user_correction": correction,
            "grade": data.get("grade"),
            "normalized_score": data.get("normalized_score"),
            "clipboard_text": data.get("clipboard_text"),
            "mod_details": data.get("mod_details"),
        }

        self._save_local(record)
        self._upload_discord(record)

    def _save_local(self, record: dict):
        """Append flag record to local JSONL file."""
        try:
            FLAG_REPORT_DB.parent.mkdir(parents=True, exist_ok=True)
            with open(FLAG_REPORT_DB, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            logger.info(f"Flag saved: {record.get('item_name')} "
                        f"(correction: {record.get('user_correction')})")
        except Exception as e:
            logger.warning(f"Failed to save flag locally: {e}")

    def _upload_discord(self, record: dict):
        """POST flag report to Discord webhook."""
        if not DISCORD_FLAG_WEBHOOK_URL:
            logger.debug("Flag: no Discord webhook configured, skipping upload")
            self._show_result("Flagged (local only)", "decent")
            return

        try:
            import requests
        except ImportError:
            logger.error("Flag upload failed: 'requests' package not installed")
            self._show_result("Flagged (local only)", "decent")
            return

        # Build Discord message
        item_name = record.get("item_name", "Unknown")
        correction = record.get("user_correction")
        grade = record.get("grade", "?")
        price = record.get("price_divine")
        price_str = f"{price:.1f}d" if price else "?"

        message = f"**Flag: {item_name}**"
        message += f"\nGrade: {grade} | Our estimate: {price_str}"
        if correction:
            message += f"\nUser correction: **{correction}**"
        else:
            message += "\n(no correction provided)"

        display_text = record.get("display_text", "")
        if display_text:
            message += f"\nDisplay: `{display_text}`"

        if len(message) > 2000:
            message = message[:1997] + "..."

        # Attach clipboard text as file
        files = {}
        clipboard = record.get("clipboard_text")
        if clipboard:
            files["file"] = ("item_data.txt", clipboard.encode("utf-8"), "text/plain")

        try:
            resp = requests.post(
                DISCORD_FLAG_WEBHOOK_URL,
                data={"content": message},
                files=files if files else None,
                timeout=15,
            )
            if resp.status_code in range(200, 300):
                logger.info("Flag uploaded to Discord")
                self._show_result("Flagged!", "decent")
            else:
                logger.error(f"Flag upload failed: HTTP {resp.status_code}")
                self._show_result("Flagged (upload failed)", "low")
        except Exception as e:
            logger.error(f"Flag upload error: {e}")
            self._show_result("Flagged (upload failed)", "low")

    def _show_result(self, text, tier):
        """Show result feedback in overlay."""
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
