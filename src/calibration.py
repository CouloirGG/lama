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

# Sample tuple: (score, divine, grade_num, dps_factor, defense_factor,
#                top_tier_count, mod_count, timestamp, is_user, mod_groups, base_type)
# top_tier_count: number of T1/T2 mods with weight >= 1.0 (0-6)
# mod_count: total parsed mods on the item (0-12)
# timestamp: epoch seconds (0 for shard data)
# is_user: True for user's own data (recency-weighted), False for shard data
# mod_groups: sorted tuple of mod group name strings (empty tuple for legacy data)
# base_type: item base type string (empty for legacy data)
Sample = Tuple[float, float, int, float, float, int, int, int, bool, Tuple[str, ...], str]


class CalibrationEngine:
    MIN_CLASS_SAMPLES = 10
    MIN_GLOBAL_SAMPLES = 50
    _EPSILON = 0.02  # smoothing floor; exact match gets ~50x weight vs dist=1.0

    # Distance metric weights
    GRADE_PENALTY = 0.40       # per grade step (r=0.275, strongest predictor)
    TOP_TIER_WEIGHT = 0.35     # top-tier mod count difference weight (r=0.237)
    MOD_COUNT_WEIGHT = 0.15    # total mod count difference weight (r=0.114)
    DPS_WEIGHT = 0.10          # DPS factor difference weight (r=0.039)
    DEFENSE_WEIGHT = 0.10      # defense factor difference weight (r=0.028)
    MOD_IDENTITY_WEIGHT = 0.15 # weighted Jaccard distance on mod groups
    BASE_TYPE_WEIGHT = 0.35    # binary: same base=0, different=0.35

    # Group-prior blending: when k-NN neighbors are distant, blend toward
    # group median to prevent wild extrapolation
    BLEND_START_DIST = 0.5     # begin blending when mean neighbor dist > this
    BLEND_FULL_DIST = 1.5      # full blend at this distance

    # Sanity cap applied when loading data
    _MAX_PRICE_DIVINE = 1500.0      # above this = likely price-fixer

    # Recency bonus: user data from last 7 days gets 2x weight
    _RECENCY_WINDOW = 7 * 86400    # 7 days in seconds
    _RECENCY_MULTIPLIER = 2.0

    def __init__(self):
        # class -> [Sample] sorted by score
        self._by_class: Dict[str, List[Sample]] = {}
        # all samples sorted by score
        self._global: List[Sample] = []
        # Group median cache: (grade_num, item_class) -> median log-price
        self._group_medians: Dict[Tuple[int, str], float] = {}
        self._group_medians_dirty: bool = True
        # Mod importance weights: mod_group -> weight (from _WEIGHT_TABLE)
        self._mod_weights: Dict[str, float] = {}
        # Learned regression weights (loaded from shard v5+)
        self._learned_weights = None  # Optional[LearnedWeights]

    @property
    def _k(self) -> int:
        """Scale k with data size: min(20, len/5), floor 12."""
        n = len(self._global)
        if n < 60:
            return 5
        return min(20, max(12, n // 5))

    def set_mod_weights(self, weights: Dict[str, float]):
        """Set mod importance weights for weighted Jaccard distance."""
        self._mod_weights = dict(weights)

    def _auto_populate_mod_weights(self):
        """Build _mod_weights from sample data + mod_database lookup."""
        try:
            from mod_database import _get_weight_for_group
        except ImportError:
            return
        groups = set()
        for s in self._global:
            for g in s[9]:
                groups.add(g)
        for g in groups:
            if g not in self._mod_weights:
                w = _get_weight_for_group(g)
                self._mod_weights[g] = w if w is not None else 0.3

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
                    dedup_key = (round(score, 3), round(divine, 2), item_class, grade)
                    if dedup_key in seen:
                        skipped += 1
                        continue
                    seen.add(dedup_key)

                    grade_num = _GRADE_NUM.get(grade, 1)
                    ts = rec.get("ts", 0)
                    dps_factor = rec.get("dps_factor", 1.0)
                    defense_factor = rec.get("defense_factor", 1.0)
                    top_tier_count = rec.get("top_tier_count", 0)
                    mod_count = rec.get("mod_count", 4)
                    mod_groups = rec.get("mod_groups", [])
                    base_type = rec.get("base_type", "")
                    self._insert(float(score), float(divine),
                                 item_class, grade_num,
                                 dps_factor, defense_factor,
                                 top_tier_count=top_tier_count,
                                 mod_count=mod_count,
                                 ts=ts, is_user=True,
                                 mod_groups=mod_groups,
                                 base_type=base_type)
                    count += 1
        except Exception as e:
            logger.warning(f"Calibration: load error: {e}")

        if count:
            logger.info(f"Calibration: {count} user samples loaded "
                         f"({len(self._by_class)} item classes)"
                         + (f", {skipped} filtered" if skipped else ""))
        if not self._mod_weights:
            self._auto_populate_mod_weights()
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

            # Load mod group index (v3+ shards)
            mod_index = shard.get("mod_index", [])
            idx_to_group = {}
            for i, entry in enumerate(mod_index):
                if isinstance(entry, list) and len(entry) >= 2:
                    idx_to_group[i] = entry[0]
                    self._mod_weights[entry[0]] = entry[1]

            # Load base type index (v4+ shards)
            base_index = shard.get("base_index", [])

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
                top_tier_count = s.get("t", 0)
                mod_count = s.get("n", 4)
                mod_groups = [idx_to_group[idx] for idx in s.get("m", [])
                              if idx in idx_to_group]
                bt_idx = s.get("b")
                base_type = base_index[bt_idx] if (bt_idx is not None and bt_idx < len(base_index)) else ""

                self._insert(float(score), float(price),
                             item_class, grade_num,
                             dps_factor, defense_factor,
                             top_tier_count=top_tier_count,
                             mod_count=mod_count,
                             ts=0, is_user=False,
                             mod_groups=mod_groups,
                             base_type=base_type)
                count += 1

            # Load learned weights (v5+ shards)
            lw_data = shard.get("learned_weights")
            if lw_data:
                try:
                    from weight_learner import LearnedWeights
                    self._learned_weights = LearnedWeights.from_dict(lw_data)
                    logger.info(f"Loaded {len(self._learned_weights._models)} regression models")
                except Exception as e:
                    logger.warning(f"Failed to load learned weights: {e}")

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
                   item_class: str, grade: str = "",
                   top_tier_count: int = 0, mod_count: int = 4,
                   mod_groups: list = None, base_type: str = ""):
        """Live-add a calibration point (called after each deep query)."""
        if divine <= 0:
            return
        if divine > self._MAX_PRICE_DIVINE:
            return
        grade_num = _GRADE_NUM.get(grade, 1)
        self._insert(score, divine, item_class, grade_num,
                     top_tier_count=top_tier_count, mod_count=mod_count,
                     ts=int(time.time()), is_user=True,
                     mod_groups=mod_groups, base_type=base_type)

    def _regression_estimate(self, item_class: str, mod_groups: list,
                             base_type: str, grade_num: int,
                             top_tier_count: int, mod_count: int,
                             dps_factor: float, defense_factor: float,
                             somv_factor: float = 1.0) -> Optional[float]:
        """Try regression-based estimate. Returns None if unavailable."""
        if self._learned_weights is None:
            return None
        if not self._learned_weights.has_model(item_class):
            return None
        if not mod_groups:
            return None

        est = self._learned_weights.predict(
            item_class, mod_groups, base_type,
            grade_num=grade_num,
            top_tier_count=top_tier_count,
            mod_count=mod_count,
            dps_factor=dps_factor,
            defense_factor=defense_factor,
            somv_factor=somv_factor,
        )
        if est is None:
            return None

        # Apply same cap as k-NN
        class_samples = self._by_class.get(item_class, self._global)
        if class_samples:
            max_observed = max((s[1] for s in class_samples), default=100.0)
            est = min(est, max_observed * 2.0, 1500.0)

        # Round to reasonable precision
        if est >= 10:
            return round(est, 0)
        elif est >= 1:
            return round(est, 1)
        else:
            return round(est, 2)

    def estimate(self, score: float, item_class: str,
                 grade: str = "", dps_factor: float = 1.0,
                 defense_factor: float = 1.0,
                 top_tier_count: int = 0,
                 mod_count: int = 4,
                 mod_groups: list = None,
                 base_type: str = "",
                 somv_factor: float = 1.0) -> Optional[float]:
        """Return estimated divine value, or None if insufficient data.

        Tries regression first (if learned weights available),
        falls back to class-specific k-NN, then global k-NN.
        """
        grade_num = _GRADE_NUM.get(grade, 1)

        # Try regression first (if learned weights available)
        reg_est = self._regression_estimate(
            item_class, mod_groups or [], base_type,
            grade_num, top_tier_count, mod_count,
            dps_factor, defense_factor, somv_factor)
        if reg_est is not None:
            return reg_est

        # Fall back to k-NN
        mg_set = frozenset(mod_groups) if mod_groups else frozenset()

        # Try class-specific
        class_samples = self._by_class.get(item_class)
        if class_samples and len(class_samples) >= self.MIN_CLASS_SAMPLES:
            return self._interpolate(score, class_samples, grade_num,
                                     dps_factor, defense_factor,
                                     top_tier_count, mod_count,
                                     item_class, mod_groups=mg_set,
                                     base_type=base_type)

        # Fall back to global
        if len(self._global) >= self.MIN_GLOBAL_SAMPLES:
            return self._interpolate(score, self._global, grade_num,
                                     dps_factor, defense_factor,
                                     top_tier_count, mod_count,
                                     item_class, mod_groups=mg_set,
                                     base_type=base_type)

        return None

    def sample_count(self, item_class: str = "") -> int:
        """How many samples for a class (or global total if empty)."""
        if item_class:
            return len(self._by_class.get(item_class, []))
        return len(self._global)

    def _insert(self, score: float, divine: float,
                item_class: str, grade_num: int,
                dps_factor: float = 1.0, defense_factor: float = 1.0,
                top_tier_count: int = 0, mod_count: int = 4,
                ts: int = 0, is_user: bool = False,
                mod_groups: list = None, base_type: str = ""):
        """Insert a sample into class-specific and global lists (sorted)."""
        mg_tuple = tuple(sorted(set(mod_groups))) if mod_groups else ()
        entry: Sample = (score, divine, grade_num, dps_factor, defense_factor,
                         top_tier_count, mod_count, ts, is_user, mg_tuple, base_type)
        self._group_medians_dirty = True
        insort(self._global, entry)
        if item_class:
            if item_class not in self._by_class:
                self._by_class[item_class] = []
            insort(self._by_class[item_class], entry)

    def _recompute_group_medians(self):
        """Recompute median log-price per (grade_num, item_class) group."""
        self._group_medians.clear()

        # Collect log-prices per group from class-specific pools
        groups: Dict[Tuple[int, str], List[float]] = {}
        for item_class, samples in self._by_class.items():
            for s in samples:
                key = (s[2], item_class)  # (grade_num, item_class)
                if key not in groups:
                    groups[key] = []
                groups[key].append(math.log(s[1]))

        # Also build grade-only groups (empty item_class) from global
        grade_groups: Dict[int, List[float]] = {}
        for s in self._global:
            gn = s[2]
            if gn not in grade_groups:
                grade_groups[gn] = []
            grade_groups[gn].append(math.log(s[1]))

        for key, log_prices in groups.items():
            if len(log_prices) >= 3:
                log_prices.sort()
                self._group_medians[key] = log_prices[len(log_prices) // 2]

        # Grade-only fallbacks (item_class = "")
        for gn, log_prices in grade_groups.items():
            key = (gn, "")
            if key not in self._group_medians and len(log_prices) >= 3:
                log_prices.sort()
                self._group_medians[key] = log_prices[len(log_prices) // 2]

        self._group_medians_dirty = False

    def _weighted_jaccard_distance(self, set_a: frozenset, set_b: frozenset) -> float:
        """Weighted Jaccard distance in [0.0, MOD_IDENTITY_WEIGHT].
        Returns 0.0 when either set is empty (neutral for legacy data).
        """
        if not set_a or not set_b:
            return 0.0
        union = set_a | set_b
        intersection = set_a & set_b
        if not union:
            return 0.0
        w_union = sum(self._mod_weights.get(g, 0.3) for g in union)
        w_inter = sum(self._mod_weights.get(g, 0.3) for g in intersection)
        if w_union == 0:
            return 0.0
        return (1.0 - w_inter / w_union) * self.MOD_IDENTITY_WEIGHT

    def _interpolate(self, score: float, samples: List[Sample],
                     grade_num: int = 1, dps_factor: float = 1.0,
                     defense_factor: float = 1.0,
                     top_tier_count: int = 0, mod_count: int = 4,
                     item_class: str = "",
                     mod_groups: frozenset = None,
                     base_type: str = "") -> float:
        """k-NN inverse-distance-weighted interpolation in log-price space.

        Distance = |score_diff| + 0.40*|grade_diff| + 0.35*|ttc_diff|
                 + 0.15*|mc_diff| + 0.10*|dps_diff| + 0.10*|def_diff|
                 + weighted_jaccard(mod_groups) + 0.35*(base_type_mismatch)

        1. Compute multi-dimensional distance
        2. Sort by distance, take k nearest
        3. Weight by 1 / (distance + 0.02), with recency bonus for user data
        4. Weighted average of log(divine), then exp() back
        5. Blend toward group median when neighbors are distant
        """
        if mod_groups is None:
            mod_groups = frozenset()
        k = self._k
        now = time.time()

        def _dist(s: Sample) -> float:
            score_d = abs(s[0] - score)
            grade_d = abs(s[2] - grade_num) * self.GRADE_PENALTY
            dps_d = abs(s[3] - dps_factor) * self.DPS_WEIGHT
            def_d = abs(s[4] - defense_factor) * self.DEFENSE_WEIGHT
            ttc_d = abs(s[5] - top_tier_count) * self.TOP_TIER_WEIGHT
            mc_d = abs(s[6] - mod_count) * self.MOD_COUNT_WEIGHT
            s_mods = frozenset(s[9]) if s[9] else frozenset()
            mod_d = self._weighted_jaccard_distance(mod_groups, s_mods)
            bt_d = 0.0
            if base_type and s[10] and base_type != s[10]:
                bt_d = self.BASE_TYPE_WEIGHT
            return score_d + grade_d + ttc_d + mc_d + dps_d + def_d + mod_d + bt_d

        by_dist = sorted(samples, key=_dist)
        neighbors = by_dist[:k]

        total_weight = 0.0
        weighted_log_sum = 0.0
        dist_sum = 0.0

        for s in neighbors:
            (s_score, s_divine, s_grade, s_dps, s_def,
             s_ttc, s_mc, s_ts, s_user, s_mg, s_bt) = s
            dist = _dist(s)
            dist_sum += dist
            w = 1.0 / (dist + self._EPSILON)

            # Recency bonus: recent user data gets extra weight
            if s_user and s_ts > 0:
                age = now - s_ts
                if age < self._RECENCY_WINDOW:
                    w *= self._RECENCY_MULTIPLIER

            weighted_log_sum += w * math.log(s_divine)
            total_weight += w

        avg_log = weighted_log_sum / total_weight

        # Group-prior blending: when neighbors are distant, blend toward
        # group median to prevent wild extrapolation
        mean_dist = dist_sum / k if k > 0 else 0.0
        if mean_dist > self.BLEND_START_DIST:
            if self._group_medians_dirty:
                self._recompute_group_medians()
            group_median = self._group_medians.get((grade_num, item_class))
            if group_median is None:
                group_median = self._group_medians.get((grade_num, ""))
            if group_median is not None:
                alpha = min(1.0, (mean_dist - self.BLEND_START_DIST)
                            / (self.BLEND_FULL_DIST - self.BLEND_START_DIST))
                avg_log = (1.0 - alpha) * avg_log + alpha * group_median

        result = math.exp(avg_log)

        # Cap wildly extrapolated estimates — with few samples the
        # log-space interpolation can produce absurd values.
        max_observed = max((s[1] for s in samples), default=100.0)
        result = min(result, max_observed * 2.0, 1500.0)

        # Round to reasonable precision
        if result >= 10:
            return round(result, 0)
        elif result >= 1:
            return round(result, 1)
        else:
            return round(result, 2)
