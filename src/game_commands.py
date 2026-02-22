"""
game_commands.py — Send chat commands to the POE2 game window.

Uses ctypes keystroke simulation + clipboard paste to type commands
in the game chat. Reuses Windows API bindings from clipboard_reader.py.
"""

import ctypes
import ctypes.wintypes
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger("game_commands")

# ─── Windows API constants ────────────────────────
CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002
KEYEVENTF_KEYUP = 0x0002

VK_CONTROL = 0x11
VK_RETURN = 0x0D
VK_V = 0x56
VK_MENU = 0x12      # Alt
VK_SHIFT = 0x10
VK_LWIN = 0x5B

# ─── Windows API bindings ─────────────────────────
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

OpenClipboard = user32.OpenClipboard
OpenClipboard.argtypes = [ctypes.wintypes.HWND]
OpenClipboard.restype = ctypes.wintypes.BOOL

CloseClipboard = user32.CloseClipboard
CloseClipboard.argtypes = []
CloseClipboard.restype = ctypes.wintypes.BOOL

EmptyClipboard = user32.EmptyClipboard
EmptyClipboard.argtypes = []
EmptyClipboard.restype = ctypes.wintypes.BOOL

GetClipboardData = user32.GetClipboardData
GetClipboardData.argtypes = [ctypes.wintypes.UINT]
GetClipboardData.restype = ctypes.wintypes.HANDLE

SetClipboardData = user32.SetClipboardData
SetClipboardData.argtypes = [ctypes.wintypes.UINT, ctypes.wintypes.HANDLE]
SetClipboardData.restype = ctypes.wintypes.HANDLE

GlobalAlloc = kernel32.GlobalAlloc
GlobalAlloc.argtypes = [ctypes.wintypes.UINT, ctypes.c_size_t]
GlobalAlloc.restype = ctypes.wintypes.HANDLE

GlobalLock = kernel32.GlobalLock
GlobalLock.argtypes = [ctypes.wintypes.HANDLE]
GlobalLock.restype = ctypes.c_void_p

GlobalUnlock = kernel32.GlobalUnlock
GlobalUnlock.argtypes = [ctypes.wintypes.HANDLE]
GlobalUnlock.restype = ctypes.wintypes.BOOL

keybd_event = user32.keybd_event
keybd_event.argtypes = [
    ctypes.wintypes.BYTE,
    ctypes.wintypes.BYTE,
    ctypes.wintypes.DWORD,
    ctypes.POINTER(ctypes.c_ulong),
]
keybd_event.restype = None

GetAsyncKeyState = user32.GetAsyncKeyState
GetAsyncKeyState.argtypes = [ctypes.c_int]
GetAsyncKeyState.restype = ctypes.wintypes.SHORT

GetForegroundWindow = user32.GetForegroundWindow
GetForegroundWindow.restype = ctypes.wintypes.HWND

GetWindowTextW = user32.GetWindowTextW
GetWindowTextW.argtypes = [ctypes.wintypes.HWND, ctypes.wintypes.LPWSTR, ctypes.c_int]
GetWindowTextW.restype = ctypes.c_int

SetForegroundWindow = user32.SetForegroundWindow
SetForegroundWindow.argtypes = [ctypes.wintypes.HWND]
SetForegroundWindow.restype = ctypes.wintypes.BOOL

ShowWindow = user32.ShowWindow
ShowWindow.argtypes = [ctypes.wintypes.HWND, ctypes.c_int]
ShowWindow.restype = ctypes.wintypes.BOOL

EnumWindows = user32.EnumWindows
WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
EnumWindows.argtypes = [WNDENUMPROC, ctypes.wintypes.LPARAM]
EnumWindows.restype = ctypes.wintypes.BOOL

IsWindowVisible = user32.IsWindowVisible
IsWindowVisible.argtypes = [ctypes.wintypes.HWND]
IsWindowVisible.restype = ctypes.wintypes.BOOL

SW_RESTORE = 9


class GameCommander:
    """Send chat commands to the POE2 game window via keystroke simulation."""

    _lock = threading.Lock()

    def _find_poe2_hwnd(self) -> int:
        """Find the POE2 window handle by title."""
        found = [0]

        @WNDENUMPROC
        def callback(hwnd, lparam):
            buf = ctypes.create_unicode_buffer(256)
            GetWindowTextW(hwnd, buf, 256)
            if "path of exile" in buf.value.lower():
                found[0] = hwnd
                return False  # stop enumerating
            return True

        EnumWindows(callback, 0)
        return found[0]

    def _focus_poe2(self) -> bool:
        """Find and focus the POE2 window. Returns True if successful."""
        hwnd = self._find_poe2_hwnd()
        if not hwnd:
            return False
        ShowWindow(hwnd, SW_RESTORE)
        SetForegroundWindow(hwnd)
        # Wait for focus to settle
        deadline = time.time() + 1.0
        while time.time() < deadline:
            if GetForegroundWindow() == hwnd:
                time.sleep(0.05)  # small extra settle time
                return True
            time.sleep(0.03)
        logger.warning("Could not bring POE2 to foreground")
        return False

    def _wait_for_modifiers_released(self, timeout: float = 1.0):
        """Wait until no modifier keys are held down."""
        deadline = time.time() + timeout
        modifiers = [VK_CONTROL, VK_MENU, VK_SHIFT, VK_LWIN]
        while time.time() < deadline:
            if not any(GetAsyncKeyState(vk) & 0x8000 for vk in modifiers):
                return
            time.sleep(0.02)
        logger.warning("Timed out waiting for modifier keys to be released")

    def _get_clipboard_text(self) -> str:
        """Read CF_UNICODETEXT from the Windows clipboard."""
        text = ""
        if not OpenClipboard(None):
            return text
        try:
            handle = GetClipboardData(CF_UNICODETEXT)
            if handle:
                ptr = GlobalLock(handle)
                if ptr:
                    try:
                        text = ctypes.wstring_at(ptr)
                    finally:
                        GlobalUnlock(handle)
        finally:
            CloseClipboard()
        return text

    def _set_clipboard_text(self, text: str):
        """Write text to the Windows clipboard as CF_UNICODETEXT."""
        if not OpenClipboard(None):
            return
        try:
            EmptyClipboard()
            if text:
                encoded = text.encode("utf-16-le") + b"\x00\x00"
                h = GlobalAlloc(GMEM_MOVEABLE, len(encoded))
                if h:
                    ptr = GlobalLock(h)
                    if ptr:
                        try:
                            ctypes.memmove(ptr, encoded, len(encoded))
                        finally:
                            GlobalUnlock(h)
                        SetClipboardData(CF_UNICODETEXT, h)
        finally:
            CloseClipboard()

    def _send_key(self, vk: int, hold_ms: int = 10):
        """Send a single key press and release."""
        keybd_event(vk, 0, 0, None)
        time.sleep(hold_ms / 1000.0)
        keybd_event(vk, 0, KEYEVENTF_KEYUP, None)

    def _send_ctrl_v(self):
        """Send Ctrl+V paste."""
        keybd_event(VK_CONTROL, 0, 0, None)
        time.sleep(0.01)
        keybd_event(VK_V, 0, 0, None)
        time.sleep(0.01)
        keybd_event(VK_V, 0, KEYEVENTF_KEYUP, None)
        keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, None)

    def type_in_chat(self, text: str, send: bool = True) -> dict:
        """Type text in POE2 chat via clipboard paste.

        1. Save current clipboard
        2. Write text to clipboard
        3. Send Enter (opens chat)
        4. Send Ctrl+V (paste)
        5. If send: Send Enter (submit)
        6. Restore original clipboard

        Returns {"status": "sent"} or {"error": "..."}.
        """
        with self._lock:
            if not self._focus_poe2():
                return {"error": "POE2 window not found"}

            self._wait_for_modifiers_released()

            # Save clipboard
            original = self._get_clipboard_text()

            try:
                # Write command to clipboard
                self._set_clipboard_text(text)
                time.sleep(0.03)

                # Open chat
                self._send_key(VK_RETURN)
                time.sleep(0.08)

                # Paste
                self._send_ctrl_v()
                time.sleep(0.05)

                # Submit
                if send:
                    self._send_key(VK_RETURN)
                    time.sleep(0.05)

                logger.info(f"Sent chat command: {text[:60]}{'...' if len(text) > 60 else ''}")
                return {"status": "sent"}

            finally:
                # Restore clipboard
                time.sleep(0.05)
                if original:
                    self._set_clipboard_text(original)

    def whisper(self, player: str, message: str) -> dict:
        """Send a whisper message to a player."""
        return self.type_in_chat(f"@{player} {message}")

    def invite(self, player: str) -> dict:
        """Send /invite to a player."""
        return self.type_in_chat(f"/invite {player}")

    def visit_hideout(self, player: str) -> dict:
        """Visit a player's hideout.

        NOTE: POE2 does NOT support /hideout <player> (POE1 only).
        In POE2, visiting another player's hideout requires the token API
        or right-clicking their name in party → 'Visit Hideout'.
        This method is kept for POE1 compatibility but should not be called for POE2.
        """
        return self.type_in_chat(f"/hideout {player}")

    def trade_with(self, player: str) -> dict:
        """Open trade window with a player."""
        return self.type_in_chat(f"/tradewith {player}")

    def kick(self, player: str) -> dict:
        """Kick a player from party."""
        return self.type_in_chat(f"/kick {player}")

    def go_home(self) -> dict:
        """Return to own hideout."""
        return self.type_in_chat("/hideout")
