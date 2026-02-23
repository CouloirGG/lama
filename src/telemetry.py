"""
LAMA - Telemetry Uploader
Opt-in anonymous calibration data upload via Discord webhook.
Uploads gzipped JSON with (score, grade, price) pairs — no PII.
"""

import gzip
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from config import (
    CALIBRATION_LOG_FILE,
    DISCORD_TELEMETRY_WEBHOOK_URL,
    TELEMETRY_UPLOAD_INTERVAL,
    TELEMETRY_LAST_UPLOAD_FILE,
    APP_VERSION,
)

logger = logging.getLogger(__name__)


class TelemetryUploader:
    """Collects calibration data and uploads to Discord on a 24h schedule."""

    def __init__(self, league: str):
        self.league = league
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    # ── Payload ──────────────────────────────────────

    def collect_payload(self, session_stats: dict | None = None) -> dict:
        """Read calibration.jsonl since last upload, strip PII, build payload."""
        last_ts = self._read_last_upload_ts()
        samples = []

        try:
            if CALIBRATION_LOG_FILE.exists():
                with open(CALIBRATION_LOG_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if rec.get("ts", 0) <= last_ts:
                            continue
                        # Strip PII fields — keep only scoring/pricing data
                        samples.append({
                            "ts": rec.get("ts", 0),
                            "s": rec.get("score", 0),
                            "g": rec.get("grade", ""),
                            "p": rec.get("min_divine", 0),
                            "c": rec.get("item_class", ""),
                            "d": rec.get("dps_factor", 1.0),
                            "f": rec.get("defense_factor", 1.0),
                            "v": rec.get("somv_factor", 1.0),
                            "r": rec.get("results", 0),
                        })
        except Exception as e:
            logger.warning(f"Telemetry: failed to read calibration log: {e}")

        payload = {
            "v": APP_VERSION,
            "league": self.league,
            "ts": datetime.now(timezone.utc).isoformat(),
            "samples": samples,
        }
        if session_stats:
            payload["stats"] = session_stats

        return payload

    # ── Upload ───────────────────────────────────────

    def upload(self, session_stats: dict | None = None) -> tuple[bool, str]:
        """Gzip payload and POST to Discord webhook. Returns (success, reason)."""
        url = DISCORD_TELEMETRY_WEBHOOK_URL
        if not url:
            logger.info("Telemetry upload: no webhook URL configured — skipping")
            return False, "No webhook URL configured"

        # Mask URL for safe logging (keep host + last 8 chars of path)
        try:
            masked = url.split("/webhooks/")[0] + "/webhooks/…" + url[-8:]
        except Exception:
            masked = url[:40] + "…"
        logger.info(f"Telemetry upload: starting (webhook: {masked})")

        payload = self.collect_payload(session_stats)
        n = len(payload["samples"])
        if not n:
            logger.info("Telemetry upload: no new samples since last upload")
            return True, "No new samples"

        logger.info(f"Telemetry upload: {n} samples, league={self.league}")

        # Gzip the JSON payload
        json_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        gzipped = gzip.compress(json_bytes)
        logger.info(f"Telemetry upload: payload {len(json_bytes)}B → {len(gzipped)}B gzipped")

        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"telemetry_{ts_str}.json.gz"
        message = (
            f"**Telemetry** v{APP_VERSION} | {self.league} | "
            f"{n} samples"
        )

        try:
            resp = requests.post(
                url,
                data={"content": message},
                files={"file": (filename, gzipped, "application/gzip")},
                timeout=15,
            )
            if resp.status_code in range(200, 300):
                logger.info(f"Telemetry upload: success — {n} samples uploaded")
                self._write_last_upload_ts()
                return True, f"Uploaded {n} samples"
            else:
                body = resp.text[:200] if resp.text else "(empty)"
                reason = f"HTTP {resp.status_code}"
                logger.warning(f"Telemetry upload: failed {reason} — {body}")
                return False, reason
        except Exception as e:
            logger.warning(f"Telemetry upload: exception — {e}")
            return False, str(e)

    # ── Scheduling ───────────────────────────────────

    def start_schedule(self):
        """Start background 24h daemon timer."""
        self.stop_schedule()
        self._schedule_next()
        logger.info("Telemetry: schedule started (24h interval)")

    def stop_schedule(self):
        """Cancel scheduled upload."""
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None

    def _schedule_next(self):
        """Schedule the next upload after TELEMETRY_UPLOAD_INTERVAL seconds."""
        with self._lock:
            self._timer = threading.Timer(
                TELEMETRY_UPLOAD_INTERVAL, self._scheduled_upload)
            self._timer.daemon = True
            self._timer.start()

    def _scheduled_upload(self):
        """Run upload and re-schedule."""
        try:
            self.upload()
        except Exception as e:
            logger.warning(f"Telemetry: scheduled upload failed: {e}")
        self._schedule_next()

    def upload_now(self) -> tuple[bool, str]:
        """Manual trigger from dashboard. Returns (success, reason)."""
        return self.upload()

    # ── Status ───────────────────────────────────────

    def get_status(self) -> dict:
        """Return telemetry status for dashboard display."""
        last_ts = self._read_last_upload_ts()
        pending = self._count_pending(last_ts)
        last_upload = None
        if last_ts > 0:
            last_upload = last_ts
        return {
            "last_upload": last_upload,
            "pending_samples": pending,
            "enabled": bool(DISCORD_TELEMETRY_WEBHOOK_URL),
        }

    # ── Internal ─────────────────────────────────────

    def _read_last_upload_ts(self) -> int:
        """Read the last upload timestamp from disk."""
        try:
            if TELEMETRY_LAST_UPLOAD_FILE.exists():
                data = json.loads(TELEMETRY_LAST_UPLOAD_FILE.read_text(encoding="utf-8"))
                return data.get("last_upload_ts", 0)
        except Exception:
            pass
        return 0

    def _write_last_upload_ts(self):
        """Write current time as last upload timestamp."""
        try:
            TELEMETRY_LAST_UPLOAD_FILE.parent.mkdir(parents=True, exist_ok=True)
            TELEMETRY_LAST_UPLOAD_FILE.write_text(
                json.dumps({"last_upload_ts": int(time.time())}),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Telemetry: failed to write last upload ts: {e}")

    def _count_pending(self, since_ts: int) -> int:
        """Count calibration records newer than since_ts."""
        count = 0
        try:
            if CALIBRATION_LOG_FILE.exists():
                with open(CALIBRATION_LOG_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if rec.get("ts", 0) > since_ts:
                            count += 1
        except Exception:
            pass
        return count
