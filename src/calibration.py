"""
LAMA - Calibration Engine

Reads calibration data (from shards and/or calibration.jsonl) and produces
divine-value estimates from local scoring data alone.

Uses k-NN inverse-distance-weighted interpolation in log-price space:
items with similar normalized scores tend to have similar market prices.
Log-space linearizes the exponential score->price relationship.

Grade-aware distance: neighbors with a different grade incur a penalty
so that S-grade items aren't dragged down by nearby C-grade data points
(which may have the same score but wildly different market value).

Expanded distance metric incorporates DPS and defense factors for
more accurate weapon/armor pricing.

Supports loading pre-built calibration shards (compact gzipped JSON)
for instant accuracy from first launch, with user data overlaid on top.
"""

import gzip
import json
import math
import logging
import time
from bisect import insort
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Numeric grade for distance calculation: higher = better
_GRADE_NUM = {"S": 4, "A": 3, "B": 2, "C": 1, "JUNK": 0}

# Sample tuple: (score, divine, grade_num, dps_factor, defense_factor, timestamp, is_user)
# timestamp: epoch seconds (0 for shard data)
# is_user: True for user's own data (recency-weighted), False for shard data
Sample = Tuple[float, float, int, float, float, int, bool]


class CalibrationEngine:
    MIN_CLASS_SAMPLES = 10
    MIN_GLOBAL_SAMPLES = 50
    _EPSILON = 1e-6  # prevents division by zero in distance weighting

    # Distance metric weights
    GRADE_PENALTY = 0.15       # per grade step
    DPS_WEIGHT = 0.3           # DPS factor difference weight
    DEFENSE_WEIGHT = 0.2       # defense factor difference weight

    # Sanity cap applied when loading data
    _MAX_PRICE_DIVINE = 300.0       # above this = likely price-fixer

    # Recency bonus: user data from last 7 days gets 2x weight
    _RECENCY_WINDOW = 7 * 86400    # 7 days in seconds
    _RECENCY_MULTIPLIER = 2.0

    def __init__(self):
        # class -> [Sample] sorted by score
        self._by_class: Dict[str, List[Sample]] = {}
        # all samples sorted by score
        self._global: List[Sample] = []

    @property
    def _k(self) -> int:
        """Scale k with data size: min(8, len/5), floor 5."""
        n = len(self._global)
        if n < 25:
            return 5
        return min(8, max(5, n // 5))

    def load(self, log_file: Path) -> int:
        """Read calibration JSONL, return total sample count.

        Applies sanity filters:
        - Skips records where min_divine <= 0
        - Skips records with price > _MAX_PRICE_DIVINE (price-fixers)
        - Skips estimates
        - Deduplicates identical (score, price, item_class) entries
        """
        if not log_file.exists():
            logger.debug("Calibration: no log file yet")
            return 0

        count = 0
        skipped = 0
        seen = set()  # dedup key: (score_rounded, price_rounded, item_class)
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    score = rec.get("score")
                    divine = rec.get("min_divine")
                    item_class = rec.get("item_class", "")
                    grade = rec.get("grade", "")

                    if score is None or divine is None:
                        continue
                    if divine <= 0:
                        continue
                    if rec.get("estimate", False):
                        skipped += 1
                        continue

                    # Sanity filter: extreme prices are price-fixers
                    if divine > self._MAX_PRICE_DIVINE:
                        skipped += 1
                        continue

                    # Dedup: same item scanned multiple times
                    dedup_key = (round(score, 3), round(divine, 2), item_class)
                    if dedup_key in seen:
                        skipped += 1
                        continue
                    seen.add(dedup_key)

                    grade_num = _GRADE_NUM.get(grade, 1)
                    ts = rec.get("ts", 0)
                    dps_factor = rec.get("dps_factor", 1.0)
                    defense_factor = rec.get("defense_factor", 1.0)
                    self._insert(float(score), float(divine),
                                 item_class, grade_num,
                                 dps_factor, defense_factor,
                                 ts=ts, is_user=True)
                    count += 1
        except Exception as e:
            logger.warning(f"Calibration: load error: {e}")

        if count:
            logger.info(f"Calibration: {count} user samples loaded "
                         f"({len(self._by_class)} item classes)"
                         + (f", {skipped} filtered" if skipped else ""))
        return count

    def load_shard(self, shard_path: Path) -> int:
        """Load a compact gzipped JSON calibration shard.

        Shard format:
        {
            "version": 1,
            "samples": [{"s": score, "g": grade_num, "p": price,
                          "c": class, "d": dps, "f": def, "v": somv}, ...]
        }

        Returns number of samples loaded.
        """
        if not shard_path.exists():
            logger.debug(f"Calibration: shard not found: {shard_path}")
            return 0

        try:
            if str(shard_path).endswith(".gz"):
                with gzip.open(shard_path, "rt", encoding="utf-8") as f:
                    shard = json.load(f)
            else:
                with open(shard_path, "r", encoding="utf-8") as f:
                    shard = json.load(f)

            samples = shard.get("samples", [])
            count = 0
            for s in samples:
                score = s.get("s")
                price = s.get("p")
                if score is None or price is None or price <= 0:
                    continue
                if price > self._MAX_PRICE_DIVINE:
                    continue

                grade_num = s.get("g", 1)
                item_class = s.get("c", "")
                dps_factor = s.get("d", 1.0)
                defense_factor = s.get("f", 1.0)

                self._insert(float(score), float(price),
                             item_class, grade_num,
                             dps_factor, defense_factor,
                             ts=0, is_user=False)
                count += 1

            if count:
                league = shard.get("league", "?")
                logger.info(f"Calibration: {count} shard samples loaded "
                             f"(league={league}, classes={len(self._by_class)})")
            return count

        except Exception as e:
            logger.warning(f"Calibration: shard load error ({shard_path}): {e}")
            return 0

    def load_remote_shard(self, league: str) -> int:
        """Check GitHub releases for a shard asset, download to cache, load it.

        Checks for assets named calibration-shard-{league_slug}-*.json.gz
        on the latest release. Downloads to SHARD_DIR with 24h cache TTL.

        Returns number of samples loaded (0 if no shard found or cached).
        """
        from config import SHARD_DIR, SHARD_REFRESH_INTERVAL, SHARD_GITHUB_REPO
        import requests

        SHARD_DIR.mkdir(parents=True, exist_ok=True)
        league_slug = league.lower().replace(" ", "-")

        # Check cache freshness
        cached = list(SHARD_DIR.glob(f"calibration-shard-{league_slug}-*.json.gz"))
        if cached:
            newest = max(cached, key=lambda p: p.stat().st_mtime)
            age = time.time() - newest.stat().st_mtime
            if age < SHARD_REFRESH_INTERVAL:
                logger.debug(f"Calibration: cached shard still fresh ({age/3600:.1f}h old)")
                return self.load_shard(newest)

        # Fetch latest release from GitHub
        try:
            from server import _get_github_headers
            headers = _get_github_headers()
        except Exception:
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "POE2-Price-Overlay",
            }

        try:
            resp = requests.get(
                f"https://api.github.com/repos/{SHARD_GITHUB_REPO}/releases/latest",
                timeout=10, headers=headers,
            )
            if resp.status_code != 200:
                logger.debug(f"Calibration: GitHub releases API returned {resp.status_code}")
                # Fall back to cached shard if available
                if cached:
                    return self.load_shard(max(cached, key=lambda p: p.stat().st_mtime))
                return 0

            data = resp.json()
            # Find shard asset matching our league
            shard_asset = None
            for asset in data.get("assets", []):
                name = asset.get("name", "")
                if (name.startswith(f"calibration-shard-{league_slug}-")
                        and name.endswith(".json.gz")):
                    shard_asset = asset
                    break

            if not shard_asset:
                logger.debug(f"Calibration: no shard asset for league '{league}' in latest release")
                if cached:
                    return self.load_shard(max(cached, key=lambda p: p.stat().st_mtime))
                return 0

            # Download the shard asset
            # Use API URL for private repos (requires auth), browser_download_url for public
            download_url = shard_asset.get("url", "") or shard_asset.get("browser_download_url", "")
            if not download_url:
                return 0

            dl_headers = dict(headers)
            dl_headers["Accept"] = "application/octet-stream"

            logger.info(f"Calibration: downloading shard {shard_asset['name']}...")
            dl_resp = requests.get(download_url, timeout=30, headers=dl_headers)
            if dl_resp.status_code != 200:
                logger.warning(f"Calibration: shard download failed ({dl_resp.status_code})")
                if cached:
                    return self.load_shard(max(cached, key=lambda p: p.stat().st_mtime))
                return 0

            # Save to cache
            shard_path = SHARD_DIR / shard_asset["name"]
            shard_path.write_bytes(dl_resp.content)
            logger.info(f"Calibration: shard saved to {shard_path}")
            return self.load_shard(shard_path)

        except Exception as e:
            logger.debug(f"Calibration: remote shard fetch failed: {e}")
            if cached:
                return self.load_shard(max(cached, key=lambda p: p.stat().st_mtime))
            return 0

    def add_sample(self, score: float, divine: float,
                   item_class: str, grade: str = ""):
        """Live-add a calibration point (called after each deep query)."""
        if divine <= 0:
            return
        if divine > self._MAX_PRICE_DIVINE:
            return
        grade_num = _GRADE_NUM.get(grade, 1)
        self._insert(score, divine, item_class, grade_num,
                     ts=int(time.time()), is_user=True)

    def estimate(self, score: float, item_class: str,
                 grade: str = "", dps_factor: float = 1.0,
                 defense_factor: float = 1.0) -> Optional[float]:
        """Return estimated divine value, or None if insufficient data.

        Tries class-specific data first, falls back to global.
        """
        grade_num = _GRADE_NUM.get(grade, 1)

        # Try class-specific
        class_samples = self._by_class.get(item_class)
        if class_samples and len(class_samples) >= self.MIN_CLASS_SAMPLES:
            return self._interpolate(score, class_samples, grade_num,
                                     dps_factor, defense_factor)

        # Fall back to global
        if len(self._global) >= self.MIN_GLOBAL_SAMPLES:
            return self._interpolate(score, self._global, grade_num,
                                     dps_factor, defense_factor)

        return None

    def sample_count(self, item_class: str = "") -> int:
        """How many samples for a class (or global total if empty)."""
        if item_class:
            return len(self._by_class.get(item_class, []))
        return len(self._global)

    def _insert(self, score: float, divine: float,
                item_class: str, grade_num: int,
                dps_factor: float = 1.0, defense_factor: float = 1.0,
                ts: int = 0, is_user: bool = False):
        """Insert a sample into class-specific and global lists (sorted)."""
        entry: Sample = (score, divine, grade_num, dps_factor, defense_factor,
                         ts, is_user)
        insort(self._global, entry)
        if item_class:
            if item_class not in self._by_class:
                self._by_class[item_class] = []
            insort(self._by_class[item_class], entry)

    def _interpolate(self, score: float, samples: List[Sample],
                     grade_num: int = 1, dps_factor: float = 1.0,
                     defense_factor: float = 1.0) -> float:
        """k-NN inverse-distance-weighted interpolation in log-price space.

        Distance = |score_diff| + 0.15*|grade_diff| + 0.3*|dps_diff| + 0.2*|def_diff|

        1. Compute multi-dimensional distance
        2. Sort by distance, take k nearest
        3. Weight by 1 / (distance + epsilon), with recency bonus for user data
        4. Weighted average of log(divine), then exp() back
        """
        k = self._k
        now = time.time()

        def _dist(s: Sample) -> float:
            score_d = abs(s[0] - score)
            grade_d = abs(s[2] - grade_num) * self.GRADE_PENALTY
            dps_d = abs(s[3] - dps_factor) * self.DPS_WEIGHT
            def_d = abs(s[4] - defense_factor) * self.DEFENSE_WEIGHT
            return score_d + grade_d + dps_d + def_d

        by_dist = sorted(samples, key=_dist)
        neighbors = by_dist[:k]

        total_weight = 0.0
        weighted_log_sum = 0.0

        for s in neighbors:
            s_score, s_divine, s_grade, s_dps, s_def, s_ts, s_user = s
            dist = _dist(s)
            w = 1.0 / (dist + self._EPSILON)

            # Recency bonus: recent user data gets extra weight
            if s_user and s_ts > 0:
                age = now - s_ts
                if age < self._RECENCY_WINDOW:
                    w *= self._RECENCY_MULTIPLIER

            weighted_log_sum += w * math.log(s_divine)
            total_weight += w

        avg_log = weighted_log_sum / total_weight
        result = math.exp(avg_log)

        # Cap wildly extrapolated estimates — with few samples the
        # log-space interpolation can produce absurd values.
        max_observed = max((s[1] for s in samples), default=100.0)
        result = min(result, max_observed * 2.0, 500.0)

        # Round to reasonable precision
        if result >= 10:
            return round(result, 0)
        elif result >= 1:
            return round(result, 1)
        else:
            return round(result, 2)
