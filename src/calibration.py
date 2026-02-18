"""
POE2 Price Overlay - Calibration Engine

Reads the calibration.jsonl log (written by deep queries) and produces
divine-value estimates from local scoring data alone.

Uses k-NN inverse-distance-weighted interpolation in log-price space:
items with similar normalized scores tend to have similar market prices.
Log-space linearizes the exponential score->price relationship.

Grade-aware distance: neighbors with a different grade incur a penalty
so that S-grade items aren't dragged down by nearby C-grade data points
(which may have the same score but wildly different market value).

Starts blank (grade-only display) and shows price estimates once enough
data accumulates (3 samples per item class, or 10 globally).
"""

import json
import math
import logging
from bisect import insort
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Numeric grade for distance calculation: higher = better
_GRADE_NUM = {"S": 4, "A": 3, "B": 2, "C": 1, "JUNK": 0}

# Sample tuple: (score, divine, grade_num)
Sample = Tuple[float, float, int]


class CalibrationEngine:
    MIN_CLASS_SAMPLES = 3
    MIN_GLOBAL_SAMPLES = 10
    K_NEIGHBORS = 5
    _EPSILON = 1e-6  # prevents division by zero in distance weighting
    # Each grade step apart adds this much to the kNN distance.
    # A 2-grade gap (e.g. S vs B) adds 0.3 — comparable to a 0.3
    # score difference, which is the span of an entire grade band.
    GRADE_PENALTY = 0.15

    # Sanity cap applied when loading data
    _MAX_PRICE_DIVINE = 300.0       # above this = likely price-fixer

    def __init__(self):
        # class -> [Sample] sorted by score
        self._by_class: Dict[str, List[Sample]] = {}
        # all samples sorted by score
        self._global: List[Sample] = []

    def load(self, log_file: Path) -> int:
        """Read calibration JSONL, return total sample count.

        Applies sanity filters:
        - Skips records where min_divine <= 0
        - Skips records with price > _MAX_PRICE_DIVINE (price-fixers)
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
                    self._insert(float(score), float(divine),
                                 item_class, grade_num)
                    count += 1
        except Exception as e:
            logger.warning(f"Calibration: load error: {e}")

        if count:
            logger.info(f"Calibration: {count} samples loaded "
                         f"({len(self._by_class)} item classes)"
                         + (f", {skipped} filtered" if skipped else ""))
        return count

    def add_sample(self, score: float, divine: float,
                   item_class: str, grade: str = ""):
        """Live-add a calibration point (called after each deep query)."""
        if divine <= 0:
            return
        if divine > self._MAX_PRICE_DIVINE:
            return
        grade_num = _GRADE_NUM.get(grade, 1)
        self._insert(score, divine, item_class, grade_num)

    def estimate(self, score: float, item_class: str,
                 grade: str = "") -> Optional[float]:
        """Return estimated divine value, or None if insufficient data.

        Tries class-specific data first, falls back to global.
        grade parameter enables grade-aware distance weighting.
        """
        grade_num = _GRADE_NUM.get(grade, 1)

        # Try class-specific
        class_samples = self._by_class.get(item_class)
        if class_samples and len(class_samples) >= self.MIN_CLASS_SAMPLES:
            return self._interpolate(score, class_samples, grade_num)

        # Fall back to global
        if len(self._global) >= self.MIN_GLOBAL_SAMPLES:
            return self._interpolate(score, self._global, grade_num)

        return None

    def sample_count(self, item_class: str = "") -> int:
        """How many samples for a class (or global total if empty)."""
        if item_class:
            return len(self._by_class.get(item_class, []))
        return len(self._global)

    def _insert(self, score: float, divine: float,
                item_class: str, grade_num: int):
        """Insert a sample into class-specific and global lists (sorted)."""
        entry = (score, divine, grade_num)
        insort(self._global, entry)
        if item_class:
            if item_class not in self._by_class:
                self._by_class[item_class] = []
            insort(self._by_class[item_class], entry)

    def _interpolate(self, score: float, samples: List[Sample],
                     grade_num: int = 1) -> float:
        """k-NN inverse-distance-weighted interpolation in log-price space.

        1. Compute distance = |score_diff| + GRADE_PENALTY * |grade_diff|
        2. Sort by distance, take k nearest
        3. Weight by 1 / (distance + epsilon)
        4. Weighted average of log(divine), then exp() back
        """
        def _dist(s: Sample) -> float:
            score_d = abs(s[0] - score)
            grade_d = abs(s[2] - grade_num) * self.GRADE_PENALTY
            return score_d + grade_d

        by_dist = sorted(samples, key=_dist)
        neighbors = by_dist[:self.K_NEIGHBORS]

        total_weight = 0.0
        weighted_log_sum = 0.0

        for s_score, s_divine, s_grade in neighbors:
            dist = _dist((s_score, s_divine, s_grade))
            w = 1.0 / (dist + self._EPSILON)
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
