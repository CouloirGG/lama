"""
POE2 Price Overlay - Clipboard Reader
Sends Ctrl+C to POE2 to copy item data, then reads the clipboard.
Uses only ctypes (no extra dependencies).

POE2 (like POE1) copies structured item text to clipboard when Ctrl+C is
pressed while hovering over an item. This is instant and exact — no OCR needed.
"""

import time
import ctypes
import ctypes.wintypes
import logging
from typing import Optional

from config import CTRL_C_DELAY

logger = logging.getLogger(__name__)

# ─── Windows API constants ────────────────────────
CF_UNICODETEXT = 13

VK_CONTROL = 0x11
VK_C = 0x43

KEYEVENTF_KEYUP = 0x0002

GMEM_MOVEABLE = 0x0002

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

GlobalSize = kernel32.GlobalSize
GlobalSize.argtypes = [ctypes.wintypes.HANDLE]
GlobalSize.restype = ctypes.c_size_t

keybd_event = user32.keybd_event
keybd_event.argtypes = [
    ctypes.wintypes.BYTE,   # bVk
    ctypes.wintypes.BYTE,   # bScan
    ctypes.wintypes.DWORD,  # dwFlags
    ctypes.POINTER(ctypes.c_ulong),  # dwExtraInfo
]
keybd_event.restype = None


class ClipboardReader:
    """
    Reads item data from POE2 by sending Ctrl+C and reading the clipboard.
    Saves and restores the user's clipboard content.
    """

    def copy_item_under_cursor(self) -> Optional[str]:
        """
        Send Ctrl+C to POE2 and read the resulting clipboard text.

        Returns the item text if it looks like POE2 item data, else None.
        The user's original clipboard content is restored afterward.
        """
        # 1. Save current clipboard content
        original = self._get_clipboard_text()

        # 2. Clear clipboard so we can detect if Ctrl+C wrote something
        self._clear_clipboard()

        # 3. Send Ctrl+C
        self._send_ctrl_c()

        # 4. Wait for the game to process
        time.sleep(CTRL_C_DELAY)

        # 5. Read clipboard
        new_text = self._get_clipboard_text()

        # 6. Restore original clipboard (only if we got item data or clipboard was cleared)
        if original and original != new_text:
            self._set_clipboard_text(original)

        # 7. If the text didn't change, the clipboard clear failed or no
        #    new item was copied — treat as no detection to avoid stale data.
        if new_text == original:
            return None

        # 8. Validate — POE2 item data always contains "Rarity:" in the header
        if new_text and self._looks_like_item_data(new_text):
            return new_text

        return None

    def _send_ctrl_c(self):
        """Send Ctrl+C keypress via Windows keybd_event."""
        # Press Ctrl
        keybd_event(VK_CONTROL, 0, 0, None)
        # Press C
        keybd_event(VK_C, 0, 0, None)
        # Release C
        keybd_event(VK_C, 0, KEYEVENTF_KEYUP, None)
        # Release Ctrl
        keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, None)

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
                # Encode as UTF-16LE with null terminator
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

    def _clear_clipboard(self):
        """Empty the clipboard."""
        if not OpenClipboard(None):
            return
        try:
            EmptyClipboard()
        finally:
            CloseClipboard()

    def _looks_like_item_data(self, text: str) -> bool:
        """Check if clipboard text looks like POE2 item data."""
        # POE2 item clipboard always starts with "Item Class:" or "Rarity:"
        # and contains separator lines of "--------"
        return ("Rarity:" in text and "--------" in text)


# ─── Quick Test ──────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    reader = ClipboardReader()

    print("Hover over an item in POE2 and press Enter here...")
    input()

    text = reader.copy_item_under_cursor()
    if text:
        print("Got item data:")
        print(text)
    else:
        print("No item data found (cursor may not be over an item)")
