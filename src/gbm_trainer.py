"""
LAMA - GBM Trainer Module

Trains per-item-class Gradient Boosted Tree models from harvester data.
Serializes sklearn trees into JSON-compatible parallel arrays for
pure-Python inference at runtime (no sklearn dependency needed).

Training uses sklearn + numpy (offline during shard generation).
Inference is a simple tree traversal loop in calibration.py.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── GBM hyperparameters ──────────────────────────────────

GBM_N_ESTIMATORS = 100
GBM_MAX_DEPTH = 4
GBM_LEARNING_RATE = 0.1
GBM_SUBSAMPLE = 1.0
GBM_MIN_SAMPLES_LEAF = 15
GBM_LOSS = "huber"

# Feature selection thresholds — lowered to include rarer but important mods
MIN_MOD_FREQUENCY = 20
MAX_BASE_FEATURES = 10
MIN_CLASS_SAMPLES_DEFAULT = 50

# Quality gate: models with CV R2 below this are discarded
MIN_R2_CV = 0.02

# sklearn TREE_UNDEFINED sentinel (leaf node marker)
_TREE_UNDEFINED = -2


def train_gbm_models(records: List[dict],
                     min_class_samples: int = MIN_CLASS_SAMPLES_DEFAULT
                     ) -> Dict[str, dict]:
    """Train per-class GBM models. Returns {class_name: serialized_model}.

    Records should have keys:
        item_class, grade_num, score, top_tier_count, mod_count,
        dps_factor, defense_factor, somv_factor, tier_score, best_tier,
        avg_tier, coc_score, es_score, mana_score,
        item_level, armour, evasion, energy_shield, total_dps, total_defense,
        quality, sockets, corrupted, open_prefixes, open_suffixes,
        mod_groups (list of str), base_type (str), mod_tiers (dict)
    """
    try:
        import numpy as np
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        logger.warning("sklearn/numpy not available, skipping GBM training")
        return {}

    # Group records by item class
    by_class: Dict[str, List[dict]] = {}
    for rec in records:
        item_class = rec.get("item_class", "")
        if not item_class:
            continue
        price = rec.get("min_divine", 0)
        if price <= 0:
            continue
        if item_class not in by_class:
            by_class[item_class] = []
        by_class[item_class].append(rec)

    models = {}
    skipped = []
    for item_class, class_records in by_class.items():
        if len(class_records) < min_class_samples:
            continue
        model = _train_class_gbm(item_class, class_records, np)
        if model is None:
            continue
        # Quality gate: discard models that can't beat the mean
        if model["r2_cv"] < MIN_R2_CV:
            skipped.append((item_class, model["r2_cv"]))
            continue
        models[item_class] = model
        logger.info(f"GBM {item_class}: n={model['n_train']}, "
                    f"R2_cv={model['r2_cv']:.3f}, "
                    f"{len(model['feature_names'])} features")
    if skipped:
        logger.info(f"GBM skipped {len(skipped)} classes (R2_cv < {MIN_R2_CV}): "
                    f"{', '.join(f'{c}={r:.3f}' for c, r in skipped)}")

    return models


def _train_class_gbm(item_class: str, records: List[dict], np,
                     ) -> Optional[dict]:
    """Train one GradientBoostingRegressor, return serialized model.

    Scales tree complexity with available data:
    - Large classes (1500+): deeper trees, more estimators
    - Medium classes (500+): balanced defaults
    - Small classes (<500): shallower trees to prevent overfitting
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    n = len(records)

    # Adaptive hyperparameters based on class size
    if n >= 1500:
        n_est, depth, leaf = 150, 5, 20
    elif n >= 500:
        n_est, depth, leaf = 100, 4, 15
    else:
        n_est, depth, leaf = 60, 3, 10

    # ── Feature discovery ──────────────────────────────

    # Count mod group frequencies
    mod_freq: Dict[str, int] = {}
    for rec in records:
        for g in rec.get("mod_groups", []):
            if g:
                mod_freq[g] = mod_freq.get(g, 0) + 1

    valid_mods = sorted(g for g, c in mod_freq.items()
                        if c >= MIN_MOD_FREQUENCY)

    # Count base type frequencies
    base_freq: Dict[str, int] = {}
    for rec in records:
        bt = rec.get("base_type", "")
        if bt:
            base_freq[bt] = base_freq.get(bt, 0) + 1

    valid_bases = sorted(
        (bt for bt, c in base_freq.items() if c >= 3),
        key=lambda bt: -base_freq[bt]
    )[:MAX_BASE_FEATURES]

    # Count stat_id frequencies (from mod_stats — 100% mod coverage)
    stat_freq: Dict[str, int] = {}
    for rec in records:
        for sid in rec.get("mod_stats", {}):
            stat_freq[sid] = stat_freq.get(sid, 0) + 1

    # Keep stats appearing in ≥1% of records to limit feature count
    min_stat_freq = max(1, n // 100)
    valid_stats = sorted(sid for sid, c in stat_freq.items()
                         if c >= min_stat_freq)

    # ── Build feature names ────────────────────────────

    numeric_names = [
        "grade_num", "score", "top_tier_count", "mod_count",
        "dps_factor", "defense_factor", "somv_factor",
        "tier_score", "best_tier", "avg_tier",
        "arch_coc_spell", "arch_ci_es", "arch_mom_mana",
        "pdps", "edps", "demand_score",
        "item_level", "armour", "evasion", "energy_shield",
        "total_dps", "total_defense",
        "quality", "sockets", "corrupted",
        "open_prefixes", "open_suffixes",
    ]
    mod_feature_names = [f"mod:{g}" for g in valid_mods]
    base_feature_names = [f"base:{bt}" for bt in valid_bases]
    stat_feature_names = [f"stat:{sid}" for sid in valid_stats]
    feature_names = numeric_names + mod_feature_names + base_feature_names + stat_feature_names

    n_features = len(feature_names)
    if n_features == 0:
        return None

    # ── Build feature matrix ───────────────────────────

    mod_set_lookup = set(valid_mods)
    base_set_lookup = set(valid_bases)
    mod_idx = {g: len(numeric_names) + i for i, g in enumerate(valid_mods)}
    base_idx = {bt: len(numeric_names) + len(valid_mods) + i
                for i, bt in enumerate(valid_bases)}
    stat_idx = {sid: len(numeric_names) + len(valid_mods) + len(valid_bases) + i
                for i, sid in enumerate(valid_stats)}

    X = np.zeros((n, n_features), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)

    for row, rec in enumerate(records):
        y[row] = math.log(max(rec["min_divine"], 0.01))

        # Numeric features
        X[row, 0] = rec.get("grade_num", 2)
        X[row, 1] = rec.get("score", 0)
        X[row, 2] = rec.get("top_tier_count", 0)
        X[row, 3] = rec.get("mod_count", 4)
        X[row, 4] = rec.get("dps_factor", 1.0)
        X[row, 5] = rec.get("defense_factor", 1.0)
        X[row, 6] = rec.get("somv_factor", 1.0)
        X[row, 7] = rec.get("tier_score", 0.0)
        X[row, 8] = rec.get("best_tier", 0)
        X[row, 9] = rec.get("avg_tier", 0.0)
        X[row, 10] = rec.get("coc_score", 0.0)
        X[row, 11] = rec.get("es_score", 0.0)
        X[row, 12] = rec.get("mana_score", 0.0)
        X[row, 13] = rec.get("pdps", 0.0)
        X[row, 14] = rec.get("edps", 0.0)
        X[row, 15] = rec.get("demand_score", 0.0)
        X[row, 16] = rec.get("item_level", 0)
        X[row, 17] = rec.get("armour", 0)
        X[row, 18] = rec.get("evasion", 0)
        X[row, 19] = rec.get("energy_shield", 0)
        X[row, 20] = rec.get("total_dps", 0.0)
        X[row, 21] = rec.get("total_defense", 0)
        X[row, 22] = rec.get("quality", 0)
        X[row, 23] = rec.get("sockets", 0)
        X[row, 24] = rec.get("corrupted", 0)
        X[row, 25] = rec.get("open_prefixes", 0)
        X[row, 26] = rec.get("open_suffixes", 0)

        # Mod features: roll-quality-weighted tier encoding
        mod_tiers = rec.get("mod_tiers", {})
        mod_rolls = rec.get("mod_rolls", {})
        for g in rec.get("mod_groups", []):
            if g in mod_idx:
                tier = mod_tiers.get(g, 0)
                rq = mod_rolls.get(g, -1)
                if rq >= 0 and tier > 0:
                    X[row, mod_idx[g]] = rq * (1.0 / tier)
                elif tier > 0:
                    X[row, mod_idx[g]] = 0.5 * (1.0 / tier)
                else:
                    X[row, mod_idx[g]] = 0.25

        # Base type features: one-hot
        bt = rec.get("base_type", "")
        if bt in base_idx:
            X[row, base_idx[bt]] = 1.0

        # Stat features: raw stat_id values (from mod_stats)
        for sid, val in rec.get("mod_stats", {}).items():
            if sid in stat_idx:
                X[row, stat_idx[sid]] = val

    # ── Train GBM ──────────────────────────────────────

    # Build sample weights from sale_confidence (confirmed sales get more weight)
    weights = np.ones(n, dtype=np.float64)
    for row, rec in enumerate(records):
        sc = rec.get("sale_confidence", 1.0)
        if sc and sc != 1.0:
            weights[row] = sc

    gbm = GradientBoostingRegressor(
        n_estimators=n_est,
        max_depth=depth,
        learning_rate=GBM_LEARNING_RATE,
        subsample=GBM_SUBSAMPLE,
        min_samples_leaf=leaf,
        loss=GBM_LOSS,
        random_state=42,
    )
    gbm.fit(X, y, sample_weight=weights)

    # ── Cross-validation R2 ────────────────────────────

    try:
        cv_scores = cross_val_score(
            GradientBoostingRegressor(
                n_estimators=n_est,
                max_depth=depth,
                learning_rate=GBM_LEARNING_RATE,
                subsample=GBM_SUBSAMPLE,
                min_samples_leaf=leaf,
                loss=GBM_LOSS,
                random_state=42,
            ),
            X, y, cv=3, scoring="r2",
        )
        r2_cv = max(0.0, float(np.mean(cv_scores)))
    except Exception:
        r2_cv = 0.0

    # ── Serialize trees ────────────────────────────────

    trees = _serialize_trees(gbm, feature_names)

    # Base prediction (init estimator — mean of y for huber)
    init_pred = gbm._raw_predict_init(X[:1])
    base_prediction = float(init_pred.flat[0])

    return {
        "learning_rate": GBM_LEARNING_RATE,
        "base_prediction": round(base_prediction, 6),
        "trees": trees,
        "feature_names": feature_names,
        "mod_features": valid_mods,
        "base_features": valid_bases,
        "stat_features": valid_stats,
        "n_train": n,
        "r2_cv": round(r2_cv, 4),
    }


def _serialize_trees(gbm, feature_names: List[str]) -> List[dict]:
    """Extract sklearn tree structure into JSON-serializable parallel arrays.

    Each tree dict has:
        feature: list of int (feature index, or -2 for leaf)
        threshold: list of float
        left: list of int (left child index, or -1 for leaf)
        right: list of int (right child index, or -1 for leaf)
        value: list of float (leaf value or 0.0 for internal)
    """
    trees = []
    for estimator_arr in gbm.estimators_:
        tree = estimator_arr[0].tree_
        n_nodes = tree.node_count

        feature = []
        threshold = []
        left = []
        right = []
        value = []

        for i in range(n_nodes):
            feat_idx = int(tree.feature[i])
            if feat_idx == _TREE_UNDEFINED:
                # Leaf node
                feature.append(_TREE_UNDEFINED)
                threshold.append(0.0)
                left.append(-1)
                right.append(-1)
                value.append(round(float(tree.value[i, 0, 0]), 6))
            else:
                # Internal node
                feature.append(feat_idx)
                threshold.append(round(float(tree.threshold[i]), 6))
                left.append(int(tree.children_left[i]))
                right.append(int(tree.children_right[i]))
                value.append(0.0)

        trees.append({
            "feature": feature,
            "threshold": threshold,
            "left": left,
            "right": right,
            "value": value,
        })

    return trees
