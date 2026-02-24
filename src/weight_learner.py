"""
LAMA - Learned Mod Weights via Ridge Regression

Trains per-item-class Ridge regression models from harvester data:
    log(price) ~ intercept + sum(coeff_i * has_mod_group_i) + ...

Training uses numpy (offline in shard_generator).
Inference uses pure Python dot product (no numpy at runtime).
Coefficients stored in shard JSON alongside k-NN samples.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Synergy pairs: known-valuable mod combinations ──────────

SYNERGY_PAIRS: List[Tuple[str, str]] = [
    ("CriticalStrikeChance", "CriticalStrikeMultiplier"),
    ("SpellDamage", "CastSpeed"),
    ("PhysicalDamage", "AttackSpeed"),
    ("AddedSkillLevels", "SpellDamage"),
    ("AddedSkillLevels", "CriticalStrikeMultiplier"),
    ("MovementVelocity", "IncreasedLife"),
    ("IncreasedLife", "EnergyShield"),
    ("AttackSpeed", "CriticalStrikeChance"),
    ("CriticalStrikeMultiplier", "SpellDamage"),
    ("AreaDamage", "CriticalStrikeMultiplier"),
]

# ── Numeric normalization constants (fixed, not learned) ────

NUMERIC_NORMS: Dict[str, Tuple[float, float]] = {
    "grade_num":       (2.0, 2.0),
    "top_tier_count":  (1.0, 2.0),
    "mod_count":       (4.0, 3.0),
    "dps_factor":      (1.0, 0.4),
    "defense_factor":  (1.0, 0.2),
    "somv_factor":     (1.0, 0.1),
}

# Price clamp range
MIN_PRICE = 0.01
MAX_PRICE = 1500.0


class LearnedWeights:
    """Learned Ridge regression models for per-class price estimation.

    Each model stores coefficients for mod groups, base types, synergy pairs,
    and numeric features. Prediction is a pure Python dot product.
    """

    def __init__(self):
        # item_class -> model dict
        self._models: Dict[str, dict] = {}

    def has_model(self, item_class: str) -> bool:
        return item_class in self._models

    def predict(self, item_class: str, mod_groups: List[str],
                base_type: str, grade_num: int, top_tier_count: int,
                mod_count: int, dps_factor: float,
                defense_factor: float,
                somv_factor: float = 1.0) -> Optional[float]:
        """Pure Python dot product prediction. Returns divine price or None."""
        model = self._models.get(item_class)
        if model is None:
            return None

        log_price = model["intercept"]

        # Mod group features (binary)
        mod_coeffs = model.get("mod_coeffs", {})
        mg_set = set(mod_groups) if mod_groups else set()
        for group, coeff in mod_coeffs.items():
            if group in mg_set:
                log_price += coeff

        # Base type features (one-hot)
        base_coeffs = model.get("base_coeffs", {})
        if base_type and base_type in base_coeffs:
            log_price += base_coeffs[base_type]

        # Synergy pair features (binary)
        synergy_coeffs = model.get("synergy_coeffs", {})
        for pair_key, coeff in synergy_coeffs.items():
            parts = pair_key.split("|")
            if len(parts) == 2 and parts[0] in mg_set and parts[1] in mg_set:
                log_price += coeff

        # Numeric features (normalized)
        numeric_coeffs = model.get("numeric_coeffs", {})
        numerics = {
            "grade_num": grade_num,
            "top_tier_count": top_tier_count,
            "mod_count": mod_count,
            "dps_factor": dps_factor,
            "defense_factor": defense_factor,
            "somv_factor": somv_factor,
        }
        for feat, coeff in numeric_coeffs.items():
            val = numerics.get(feat, 0)
            center, scale = NUMERIC_NORMS.get(feat, (0.0, 1.0))
            log_price += coeff * ((val - center) / scale)

        # Convert from log-price and clamp
        price = math.exp(log_price)
        price = max(MIN_PRICE, min(MAX_PRICE, price))
        return price

    def to_dict(self) -> dict:
        """Serialize all models to JSON-compatible dict."""
        return {"models": dict(self._models)}

    @classmethod
    def from_dict(cls, data: dict) -> "LearnedWeights":
        """Deserialize from shard JSON."""
        lw = cls()
        lw._models = dict(data.get("models", {}))
        return lw

    def summary(self) -> str:
        """Diagnostic output for shard generation."""
        lines = [f"Learned weights: {len(self._models)} class models"]
        for item_class, model in sorted(self._models.items()):
            n_mods = len(model.get("mod_coeffs", {}))
            n_bases = len(model.get("base_coeffs", {}))
            n_syn = len(model.get("synergy_coeffs", {}))
            r2 = model.get("r2_cv", 0.0)
            n_train = model.get("n_train", 0)
            lines.append(
                f"    {item_class:20s}: n={n_train:5d}, "
                f"R2_cv={r2:.3f}, "
                f"{n_mods} mods, {n_bases} bases, {n_syn} synergies"
            )
            # Top 5 positive mod coefficients
            mod_coeffs = model.get("mod_coeffs", {})
            if mod_coeffs:
                top_pos = sorted(mod_coeffs.items(), key=lambda x: -x[1])[:5]
                top_neg = sorted(mod_coeffs.items(), key=lambda x: x[1])[:3]
                if top_pos and top_pos[0][1] > 0:
                    pos_str = ", ".join(f"{g}={c:+.3f}" for g, c in top_pos if c > 0)
                    if pos_str:
                        lines.append(f"      top+: {pos_str}")
                if top_neg and top_neg[0][1] < 0:
                    neg_str = ", ".join(f"{g}={c:+.3f}" for g, c in top_neg if c < 0)
                    if neg_str:
                        lines.append(f"      top-: {neg_str}")
        return "\n".join(lines)


# ── Training (requires numpy) ───────────────────────────────

def train_weights(records: List[dict],
                  min_class_samples: int = 50,
                  min_mod_frequency: int = 10,
                  max_base_types: int = 30,
                  alpha: float = 1.0) -> LearnedWeights:
    """Train per-item-class Ridge regression models from harvester records.

    Records should have keys: c (item_class), g (grade_num), p (min_divine),
    t (top_tier_count), n (mod_count), d (dps_factor), f (defense_factor),
    v (somv_factor), mod_groups_resolved (list of str), base_type_resolved (str).

    Requires numpy (training only). Inference uses pure Python.
    """
    import numpy as np

    # Group records by item class
    by_class: Dict[str, List[dict]] = {}
    for rec in records:
        item_class = rec.get("c", "")
        if not item_class:
            continue
        price = rec.get("p", 0)
        if price <= 0:
            continue
        if item_class not in by_class:
            by_class[item_class] = []
        by_class[item_class].append(rec)

    lw = LearnedWeights()

    for item_class, class_records in by_class.items():
        if len(class_records) < min_class_samples:
            continue
        model = _train_class_model(
            item_class, class_records, np,
            min_mod_frequency=min_mod_frequency,
            max_base_types=max_base_types,
            alpha=alpha,
        )
        if model is not None:
            lw._models[item_class] = model

    return lw


def _train_class_model(item_class: str, records: List[dict], np,
                       min_mod_frequency: int = 10,
                       max_base_types: int = 30,
                       alpha: float = 1.0) -> Optional[dict]:
    """Train a single Ridge regression model for one item class."""
    n = len(records)

    # ── Feature discovery ──────────────────────────────────

    # Count mod group frequencies
    mod_freq: Dict[str, int] = {}
    for rec in records:
        for g in rec.get("mod_groups_resolved", []):
            mod_freq[g] = mod_freq.get(g, 0) + 1

    # Keep mods above frequency threshold
    valid_mods = sorted(g for g, c in mod_freq.items() if c >= min_mod_frequency)
    valid_mod_set = set(valid_mods)

    # Count base type frequencies
    base_freq: Dict[str, int] = {}
    for rec in records:
        bt = rec.get("base_type_resolved", "")
        if bt:
            base_freq[bt] = base_freq.get(bt, 0) + 1

    # Keep top N base types (with >= 3 occurrences)
    valid_bases = sorted(
        (bt for bt, c in base_freq.items() if c >= 3),
        key=lambda bt: -base_freq[bt]
    )[:max_base_types]
    valid_base_set = set(valid_bases)

    # Synergy pairs: only include if both mods meet frequency threshold
    valid_synergies = []
    for a, b in SYNERGY_PAIRS:
        if a in valid_mod_set and b in valid_mod_set:
            valid_synergies.append((a, b))

    # ── Build feature matrix ───────────────────────────────

    n_mod_feats = len(valid_mods)
    n_base_feats = len(valid_bases)
    n_syn_feats = len(valid_synergies)
    n_numeric = len(NUMERIC_NORMS)
    n_features = n_mod_feats + n_base_feats + n_syn_feats + n_numeric

    if n_features == 0:
        return None

    mod_idx = {g: i for i, g in enumerate(valid_mods)}
    base_idx = {bt: n_mod_feats + i for i, bt in enumerate(valid_bases)}
    syn_offset = n_mod_feats + n_base_feats
    numeric_offset = syn_offset + n_syn_feats
    numeric_keys = list(NUMERIC_NORMS.keys())

    X = np.zeros((n, n_features), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)

    for row, rec in enumerate(records):
        # Target: log(price)
        y[row] = math.log(rec["p"])

        # Mod group features (binary)
        for g in rec.get("mod_groups_resolved", []):
            if g in mod_idx:
                X[row, mod_idx[g]] = 1.0

        # Base type features (one-hot)
        bt = rec.get("base_type_resolved", "")
        if bt in base_idx:
            X[row, base_idx[bt]] = 1.0

        # Synergy pair features (binary)
        mg_set = set(rec.get("mod_groups_resolved", []))
        for si, (a, b) in enumerate(valid_synergies):
            if a in mg_set and b in mg_set:
                X[row, syn_offset + si] = 1.0

        # Numeric features (normalized)
        raw_nums = {
            "grade_num": rec.get("g", 2),
            "top_tier_count": rec.get("t", 0),
            "mod_count": rec.get("n", 4),
            "dps_factor": rec.get("d", 1.0),
            "defense_factor": rec.get("f", 1.0),
            "somv_factor": rec.get("v", 1.0),
        }
        for ki, feat in enumerate(numeric_keys):
            center, scale = NUMERIC_NORMS[feat]
            X[row, numeric_offset + ki] = (raw_nums.get(feat, center) - center) / scale

    # ── Ridge regression (closed form) ─────────────────────
    # Center y for intercept, then: beta = (X^T X + alpha*I)^{-1} X^T y_centered

    y_mean = np.mean(y)
    y_centered = y - y_mean

    XtX = X.T @ X
    XtX += alpha * np.eye(n_features)
    Xty = X.T @ y_centered

    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        logger.warning(f"Ridge regression failed for {item_class} (singular matrix)")
        return None

    intercept = float(y_mean)

    # ── Cross-validation R2 ────────────────────────────────
    r2_cv = _cross_validate_r2(X, y, alpha, np, n_folds=3)

    # ── Pack coefficients ──────────────────────────────────
    COEFF_THRESHOLD = 1e-6

    mod_coeffs = {}
    for i, g in enumerate(valid_mods):
        c = float(beta[i])
        if abs(c) > COEFF_THRESHOLD:
            mod_coeffs[g] = round(c, 6)

    base_coeffs = {}
    for i, bt in enumerate(valid_bases):
        c = float(beta[n_mod_feats + i])
        if abs(c) > COEFF_THRESHOLD:
            base_coeffs[bt] = round(c, 6)

    synergy_coeffs = {}
    for i, (a, b) in enumerate(valid_synergies):
        c = float(beta[syn_offset + i])
        if abs(c) > COEFF_THRESHOLD:
            synergy_coeffs[f"{a}|{b}"] = round(c, 6)

    numeric_coeffs = {}
    for i, feat in enumerate(numeric_keys):
        c = float(beta[numeric_offset + i])
        if abs(c) > COEFF_THRESHOLD:
            numeric_coeffs[feat] = round(c, 6)

    return {
        "intercept": round(intercept, 6),
        "mod_coeffs": mod_coeffs,
        "base_coeffs": base_coeffs,
        "synergy_coeffs": synergy_coeffs,
        "numeric_coeffs": numeric_coeffs,
        "n_train": n,
        "r2_cv": round(r2_cv, 4),
    }


def _cross_validate_r2(X, y, alpha: float, np, n_folds: int = 3) -> float:
    """3-fold cross-validation R2 for diagnostics."""
    n = len(y)
    if n < n_folds * 2:
        return 0.0

    indices = np.arange(n)
    fold_size = n // n_folds
    ss_res_total = 0.0
    ss_tot_total = 0.0

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else n

        test_mask = np.zeros(n, dtype=bool)
        test_mask[start:end] = True
        train_mask = ~test_mask

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        if len(y_train) == 0 or len(y_test) == 0:
            continue

        y_mean_train = np.mean(y_train)
        y_centered = y_train - y_mean_train

        n_feat = X_train.shape[1]
        XtX = X_train.T @ X_train + alpha * np.eye(n_feat)
        Xty = X_train.T @ y_centered

        try:
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            continue

        y_pred = X_test @ beta + y_mean_train
        ss_res_total += np.sum((y_test - y_pred) ** 2)
        ss_tot_total += np.sum((y_test - np.mean(y_test)) ** 2)

    if ss_tot_total == 0:
        return 0.0
    return max(0.0, 1.0 - ss_res_total / ss_tot_total)
