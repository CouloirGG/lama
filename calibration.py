"""
POE2 Price Overlay - Calibration Engine

Reads the calibration.jsonl log (written by deep queries) and produces
divine-value estimates from local scoring data alone.

Uses k-NN inverse-distance-weighted interpolation in log-price space:
items with similar normalized scores tend to have similar market prices.
Log-space linearizes the exponential score→price relationship.

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


class CalibrationEngine:
    MIN_CLASS_SAMPLES = 3
    MIN_GLOBAL_SAMPLES = 10
    K_NEIGHBORS = 5
    _EPSILON = 1e-6  # prevents division by zero in distance weighting

    def __init__(self):
        # class → [(score, divine)] sorted by score
        self._by_class: Dict[str, List[Tuple[float, float]]] = {}
        # all samples sorted by score
        self._global: List[Tuple[float, float]] = []

    def load(self, log_file: Path) -> int:
        """Read calibration JSONL, return total sample count.

        Skips records where min_divine <= 0 (no listings / failed queries)
        and records missing required fields.
        """
        if not log_file.exists():
            logger.debug("Calibration: no log file yet")
            return 0

        count = 0
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

                    if score is None or divine is None:
                        continue
                    if divine <= 0:
                        continue

                    self._insert(float(score), float(divine), item_class)
                    count += 1
        except Exception as e:
            logger.warning(f"Calibration: load error: {e}")

        if count:
            logger.info(f"Calibration: {count} samples loaded "
                         f"({len(self._by_class)} item classes)")
        return count

    def add_sample(self, score: float, divine: float, item_class: str):
        """Live-add a calibration point (called after each deep query)."""
        if divine <= 0:
            return
        self._insert(score, divine, item_class)

    def estimate(self, score: float, item_class: str) -> Optional[float]:
        """Return estimated divine value, or None if insufficient data.

        Tries class-specific data first, falls back to global.
        """
        # Try class-specific
        class_samples = self._by_class.get(item_class)
        if class_samples and len(class_samples) >= self.MIN_CLASS_SAMPLES:
            return self._interpolate(score, class_samples)

        # Fall back to global
        if len(self._global) >= self.MIN_GLOBAL_SAMPLES:
            return self._interpolate(score, self._global)

        return None

    def sample_count(self, item_class: str = "") -> int:
        """How many samples for a class (or global total if empty)."""
        if item_class:
            return len(self._by_class.get(item_class, []))
        return len(self._global)

    def _insert(self, score: float, divine: float, item_class: str):
        """Insert a sample into class-specific and global lists (sorted)."""
        entry = (score, divine)
        insort(self._global, entry)
        if item_class:
            if item_class not in self._by_class:
                self._by_class[item_class] = []
            insort(self._by_class[item_class], entry)

    def _interpolate(self, score: float,
                     samples: List[Tuple[float, float]]) -> float:
        """k-NN inverse-distance-weighted interpolation in log-price space.

        1. Sort samples by distance to target score
        2. Take k nearest neighbors
        3. Weight by 1 / (distance + epsilon)
        4. Weighted average of log(divine), then exp() back
        """
        # Sort by distance to target score
        by_dist = sorted(samples, key=lambda s: abs(s[0] - score))
        neighbors = by_dist[:self.K_NEIGHBORS]

        total_weight = 0.0
        weighted_log_sum = 0.0

        for s_score, s_divine in neighbors:
            dist = abs(s_score - score)
            w = 1.0 / (dist + self._EPSILON)
            weighted_log_sum += w * math.log(s_divine)
            total_weight += w

        avg_log = weighted_log_sum / total_weight
        result = math.exp(avg_log)

        # Cap wildly extrapolated estimates — with few samples the
        # log-space interpolation can produce absurd values (8000+ divine).
        # Cap at 500 divine until we have enough data to trust higher.
        max_observed = max((s[1] for s in samples), default=100.0)
        result = min(result, max_observed * 2.0, 500.0)

        # Round to reasonable precision
        if result >= 10:
            return round(result, 0)
        elif result >= 1:
            return round(result, 1)
        else:
            return round(result, 2)
