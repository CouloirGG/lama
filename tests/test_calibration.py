"""Regression test for calibration accuracy.

80/20 holdout validation against the bundled shard.
Asserts >= 70% of estimates are within 2x of actual price.
"""

import gzip
import json
import math
import random
from pathlib import Path

import pytest

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from calibration import CalibrationEngine

_GRADE_FROM_NUM = {4: "S", 3: "A", 2: "B", 1: "C", 0: "JUNK"}

SHARD_PATH = PROJECT_ROOT / "resources" / "calibration_shard.json.gz"


def _load_shard():
    if not SHARD_PATH.exists():
        pytest.skip("Bundled shard not found")
    with gzip.open(SHARD_PATH, "rt", encoding="utf-8") as f:
        shard = json.load(f)
    samples = shard.get("samples", [])
    if len(samples) < 100:
        pytest.skip(f"Shard too small ({len(samples)} samples)")
    return samples


def test_calibration_accuracy_within_2x():
    """80/20 holdout: >= 70% of estimates within 2x of actual."""
    samples = _load_shard()

    rng = random.Random(42)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    split = int(len(indices) * 0.8)
    train_idx = indices[:split]
    test_idx = indices[split:]

    engine = CalibrationEngine()
    for i in train_idx:
        s = samples[i]
        engine._insert(
            score=s["s"],
            divine=s["p"],
            item_class=s.get("c", ""),
            grade_num=s.get("g", 1),
            dps_factor=s.get("d", 1.0),
            defense_factor=s.get("f", 1.0),
        )

    within_2x = 0
    total_tested = 0

    for i in test_idx:
        s = samples[i]
        actual = s["p"]
        grade = _GRADE_FROM_NUM.get(s.get("g", 1), "C")

        est = engine.estimate(
            s["s"], s.get("c", ""),
            grade=grade,
            dps_factor=s.get("d", 1.0),
            defense_factor=s.get("f", 1.0),
        )
        if est is None:
            continue

        total_tested += 1
        ratio = max(est / actual, actual / est) if actual > 0 else float("inf")
        if ratio <= 2.0:
            within_2x += 1

    assert total_tested > 0, "No estimates produced"
    pct = within_2x / total_tested * 100
    assert pct >= 70.0, (
        f"Calibration accuracy {pct:.1f}% is below 70% target "
        f"({within_2x}/{total_tested} within 2x)"
    )
