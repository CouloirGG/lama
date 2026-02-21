"""
trade_actions.py — Authenticated trade API actions (whisper + hideout via tokens).

When the user has a POESESSID configured, trade listings include whisper_token
and hideout_token fields. These can be submitted to the trade API to send
whispers or visit hideouts without keystroke simulation.
"""

import logging
from typing import Callable, Optional

import requests

from config import TRADE_API_BASE

logger = logging.getLogger("trade_actions")

WHISPER_URL = f"{TRADE_API_BASE}/whisper"
USER_AGENT = "POE2-Price-Overlay-TradeActions/1.0"


class TradeActions:
    """Handles authenticated trade API calls using tokens from listings."""

    def __init__(self, get_poesessid: Callable[[], str]):
        self._get_poesessid = get_poesessid
        self._session = requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT

    def _build_headers(self) -> Optional[dict]:
        """Build request headers with POESESSID cookie. Returns None if no session ID."""
        poesessid = self._get_poesessid()
        if not poesessid:
            return None
        return {
            "X-Requested-With": "XMLHttpRequest",
            "Content-Type": "application/json",
            "Cookie": f"POESESSID={poesessid}",
        }

    def whisper_via_token(self, token: str) -> dict:
        """Send a trade whisper using the whisper_token from a listing.

        Returns {"status": "sent"} or {"error": "..."}.
        """
        headers = self._build_headers()
        if not headers:
            return {"error": "No POESESSID configured"}

        try:
            resp = self._session.post(
                WHISPER_URL,
                json={"token": token},
                headers=headers,
                timeout=15,
            )

            if resp.status_code == 200:
                return {"status": "sent"}

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After", "60")
                logger.warning(f"Whisper rate limited, retry after {retry_after}s")
                return {"error": f"Rate limited ({retry_after}s)"}

            logger.warning(f"Whisper token API returned {resp.status_code}: {resp.text[:200]}")
            return {"error": f"API returned HTTP {resp.status_code}"}

        except Exception as e:
            logger.error(f"Whisper token API error: {e}")
            return {"error": str(e)}

    def hideout_via_token(self, token: str) -> dict:
        """Visit a seller's hideout using the hideout_token from a listing.

        Returns {"status": "sent"} or {"error": "..."}.
        """
        headers = self._build_headers()
        if not headers:
            return {"error": "No POESESSID configured"}

        try:
            resp = self._session.post(
                WHISPER_URL,
                json={"token": token},
                headers=headers,
                timeout=15,
            )

            if resp.status_code == 200:
                return {"status": "sent"}

            # Retry with "continue": true on failure
            if resp.status_code != 200:
                resp2 = self._session.post(
                    WHISPER_URL,
                    json={"token": token, "continue": True},
                    headers=headers,
                    timeout=15,
                )
                if resp2.status_code == 200:
                    return {"status": "sent"}

                if resp2.status_code == 429:
                    retry_after = resp2.headers.get("Retry-After", "60")
                    logger.warning(f"Hideout rate limited, retry after {retry_after}s")
                    return {"error": f"Rate limited ({retry_after}s)"}

                logger.warning(f"Hideout token API returned {resp2.status_code}: {resp2.text[:200]}")
                return {"error": f"API returned HTTP {resp2.status_code}"}

        except Exception as e:
            logger.error(f"Hideout token API error: {e}")
            return {"error": str(e)}
