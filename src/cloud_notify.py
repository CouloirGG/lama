"""
cloud_notify.py — Push notifications via cloud relay.

Sends push notifications to paired mobile devices through the Cloudflare
Worker relay. Used by the watchlist worker when matches are found.
"""

import asyncio
import logging
import time
from typing import Optional

import requests

logger = logging.getLogger("cloud_notify")

# Dedup window — don't re-notify the same query within this period (seconds)
_DEDUP_WINDOW = 300  # 5 minutes
_recent_pushes: dict[str, float] = {}  # query_id -> last push timestamp

# Module-level config — set by server.py on startup / settings change
_config: dict = {}


def configure(device_id: str, secret: str, relay_url: str, enabled: bool):
    """Update cloud notification config (called when settings change)."""
    _config["device_id"] = device_id
    _config["secret"] = secret
    _config["relay_url"] = relay_url.rstrip("/") if relay_url else ""
    _config["enabled"] = enabled


def is_enabled() -> bool:
    return bool(_config.get("enabled") and _config.get("device_id")
                and _config.get("secret") and _config.get("relay_url"))


def _send_push(title: str, body: str, data: Optional[dict] = None) -> dict:
    """Blocking POST to relay /push endpoint (run in executor)."""
    payload = {
        "device_id": _config["device_id"],
        "secret": _config["secret"],
        "title": title,
        "body": body,
    }
    if data:
        payload["data"] = data
    try:
        resp = requests.post(
            f"{_config['relay_url']}/push",
            json=payload,
            timeout=10,
            headers={"User-Agent": "LAMA-Desktop/1.0"},
        )
        result = resp.json()
        if resp.status_code != 200:
            logger.warning(f"Cloud push failed ({resp.status_code}): {result}")
        else:
            logger.debug(f"Cloud push sent: {title}")
        return result
    except Exception as e:
        logger.warning(f"Cloud push error: {e}")
        return {"error": str(e)}


async def push_notification(title: str, body: str, query_id: str = "",
                            data: Optional[dict] = None):
    """Send a push notification if cloud is enabled and not deduped.

    Args:
        title: Notification title (e.g. "Watchlist: Aegis Aurora")
        body: Notification body (e.g. "3 listed, cheapest 5 divine")
        query_id: For deduplication — skip if same query pushed recently
        data: Optional extra data payload
    """
    if not is_enabled():
        return

    # Dedup check
    if query_id:
        now = time.time()
        last_push = _recent_pushes.get(query_id, 0)
        if now - last_push < _DEDUP_WINDOW:
            return
        _recent_pushes[query_id] = now
        # Prune old entries
        cutoff = now - _DEDUP_WINDOW * 2
        for k in list(_recent_pushes):
            if _recent_pushes[k] < cutoff:
                del _recent_pushes[k]

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _send_push, title, body, data)
