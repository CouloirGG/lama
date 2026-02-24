"""Quick sweep of estimation approaches to find what actually moves accuracy."""

import sys, json, math, random, glob, re
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import numpy as np
from shard_generator import load_raw_records, quality_filter, remove_outliers, dedup_records, _GRADE_NUM

# ── Load and split data ──────────────────────────────────────

files = glob.glob("C:/Users/Stuar/.poe2-price-overlay/cache/calibration_shard_fate_of_the_vaal*.jsonl")
records = load_raw_records(files)
filtered, _ = quality_filter(records)
cleaned, _ = remove_outliers(filtered)
deduped, _ = dedup_records(cleaned)

rng = random.Random(42)
indices = list(range(len(deduped)))
rng.shuffle(indices)
split = int(len(indices) * 0.8)
train = [deduped[i] for i in indices[:split]]
test = [deduped[i] for i in indices[split:]]

print(f"Train: {len(train)}, Test: {len(test)}\n")


def eval_approach(name, predict_fn):
    within_2x = 0
    within_3x = 0
    total = 0
    for rec in test:
        actual = rec["min_divine"]
        if actual <= 0:
            continue
        est = predict_fn(rec)
        if est is None:
            continue
        total += 1
        ratio = max(est / actual, actual / est)
        if ratio <= 2.0:
            within_2x += 1
        if ratio <= 3.0:
            within_3x += 1
    pct2 = within_2x / total * 100 if total else 0
    pct3 = within_3x / total * 100 if total else 0
    print(f"  {name:55s}: {pct2:5.1f}% 2x, {pct3:5.1f}% 3x  ({within_2x}/{total})")
    return pct2


# ── Helpers ──────────────────────────────────────────────────

def parse_top_mods(tm_str):
    """Parse 'T1 CastSpd, T4 Mana' -> [(1, 'CastSpd'), (4, 'Mana')]"""
    if not tm_str:
        return []
    pairs = []
    for part in tm_str.split(","):
        part = part.strip()
        m = re.match(r"T(\d+)\s+(.+)", part)
        if m:
            pairs.append((int(m.group(1)), m.group(2).strip()))
    return pairs


def get_mod_tier_dict(rec):
    """Return {short_mod_name: best_tier_number}."""
    result = {}
    for tier, name in parse_top_mods(rec.get("top_mods", "")):
        if name not in result or tier < result[name]:
            result[name] = tier
    return result


def tier_value(tier_num):
    """Convert tier number to value: T1=1.0, T2=0.7, T3=0.5, ..., T10=0.05"""
    return 1.0 / (1.0 + (tier_num - 1) * 0.5)


# ── Build class+grade medians (used as fallback everywhere) ──

medians_cg = {}
for rec in train:
    key = (rec.get("item_class", ""), _GRADE_NUM.get(rec.get("grade", "C"), 1))
    medians_cg.setdefault(key, []).append(math.log(rec["min_divine"]))
for k, v in medians_cg.items():
    v.sort()
    medians_cg[k] = v[len(v) // 2]


def cg_median(rec):
    key = (rec.get("item_class", ""), _GRADE_NUM.get(rec.get("grade", "C"), 1))
    m = medians_cg.get(key)
    return math.exp(m) if m is not None else None


# ── Approach A: Class+Grade median ───────────────────────────
print("=== Simple stratification approaches ===")
eval_approach("A: Class+Grade median", cg_median)

# ── Approach B: Class+Grade+TopTierCount ─────────────────────
medians_cgt = {}
for rec in train:
    ttc = min(rec.get("top_tier_count", 0), 3)
    key = (rec.get("item_class", ""), _GRADE_NUM.get(rec.get("grade", "C"), 1), ttc)
    medians_cgt.setdefault(key, []).append(math.log(rec["min_divine"]))
for k, v in medians_cgt.items():
    v.sort()
    medians_cgt[k] = v[len(v) // 2]


def predict_cgt(rec):
    ttc = min(rec.get("top_tier_count", 0), 3)
    key = (rec.get("item_class", ""), _GRADE_NUM.get(rec.get("grade", "C"), 1), ttc)
    m = medians_cgt.get(key)
    return math.exp(m) if m is not None else cg_median(rec)


eval_approach("B: Class+Grade+TopTierCount", predict_cgt)

# ── Approach C: Class+Grade+ScoreBucket ──────────────────────
medians_cgs = {}
for rec in train:
    sb = round(rec.get("score", 0) * 5) / 5
    key = (rec.get("item_class", ""), _GRADE_NUM.get(rec.get("grade", "C"), 1), sb)
    medians_cgs.setdefault(key, []).append(math.log(rec["min_divine"]))
for k, v in medians_cgs.items():
    v.sort()
    medians_cgs[k] = v[len(v) // 2]


def predict_cgs(rec):
    sb = round(rec.get("score", 0) * 5) / 5
    key = (rec.get("item_class", ""), _GRADE_NUM.get(rec.get("grade", "C"), 1), sb)
    m = medians_cgs.get(key)
    return math.exp(m) if m is not None else cg_median(rec)


eval_approach("C: Class+Grade+ScoreBucket(0.2)", predict_cgs)

# ── Approach D: Class+Grade + mod_count bucket ───────────────
medians_cgm = {}
for rec in train:
    mc = min(rec.get("mod_count", 4), 6)
    key = (rec.get("item_class", ""), _GRADE_NUM.get(rec.get("grade", "C"), 1), mc)
    medians_cgm.setdefault(key, []).append(math.log(rec["min_divine"]))
for k, v in medians_cgm.items():
    v.sort()
    medians_cgm[k] = v[len(v) // 2]


def predict_cgm(rec):
    mc = min(rec.get("mod_count", 4), 6)
    key = (rec.get("item_class", ""), _GRADE_NUM.get(rec.get("grade", "C"), 1), mc)
    m = medians_cgm.get(key)
    return math.exp(m) if m is not None else cg_median(rec)


eval_approach("D: Class+Grade+ModCount", predict_cgm)


# ── Tier-aware approaches ────────────────────────────────────
print("\n=== Tier-aware approaches (using top_mods) ===")


# ── Approach E: tier_score = sum(1/tier) as continuous feature ─
def tier_score(rec):
    pairs = parse_top_mods(rec.get("top_mods", ""))
    if not pairs:
        return 0.0
    return sum(1.0 / t for t, _ in pairs)


# Per (class, grade) linear regression on tier_score
cg_linreg = {}
for key in medians_cg:
    cls, gn = key
    group = [r for r in train
             if r.get("item_class", "") == cls
             and _GRADE_NUM.get(r.get("grade", "C"), 1) == gn]
    if len(group) < 20:
        continue
    xs = np.array([tier_score(r) for r in group])
    ys = np.array([math.log(r["min_divine"]) for r in group])
    if np.std(xs) < 1e-6:
        continue
    x_mean, y_mean = np.mean(xs), np.mean(ys)
    slope = np.sum((xs - x_mean) * (ys - y_mean)) / np.sum((xs - x_mean) ** 2)
    cg_linreg[key] = (slope, y_mean - slope * x_mean)


def predict_tier_linreg(rec):
    key = (rec.get("item_class", ""), _GRADE_NUM.get(rec.get("grade", "C"), 1))
    model = cg_linreg.get(key)
    if model is None:
        return cg_median(rec)
    slope, intercept = model
    ts = tier_score(rec)
    return max(0.01, min(1500, math.exp(slope * ts + intercept)))


eval_approach("E: Class+Grade + tier_score (linear)", predict_tier_linreg)


# ── Approach F: Per-class Ridge with tier-weighted mod features ──
class_ridge = {}
for cls in set(r.get("item_class", "") for r in train):
    cls_recs = [r for r in train if r.get("item_class", "") == cls]
    if len(cls_recs) < 50:
        continue

    # Count mod short names
    mod_freq = defaultdict(int)
    for r in cls_recs:
        for _, name in parse_top_mods(r.get("top_mods", "")):
            mod_freq[name] += 1
    valid_mods = sorted(m for m, c in mod_freq.items() if c >= 10)

    n = len(cls_recs)
    # Features: grade, score, top_tier_count, mod_count, + tier value per mod
    n_feat = 4 + len(valid_mods)
    X = np.zeros((n, n_feat))
    y = np.zeros(n)
    for i, r in enumerate(cls_recs):
        y[i] = math.log(r["min_divine"])
        X[i, 0] = (_GRADE_NUM.get(r.get("grade", "C"), 1) - 2) / 2.0
        X[i, 1] = r.get("score", 0)
        X[i, 2] = (r.get("top_tier_count", 0) - 1) / 2.0
        X[i, 3] = (r.get("mod_count", 4) - 4) / 3.0
        tiers = get_mod_tier_dict(r)
        for j, mod in enumerate(valid_mods):
            if mod in tiers:
                X[i, 4 + j] = tier_value(tiers[mod])

    y_mean = np.mean(y)
    yc = y - y_mean
    alpha = 1.0
    try:
        beta = np.linalg.solve(X.T @ X + alpha * np.eye(n_feat), X.T @ yc)
        class_ridge[cls] = (y_mean, beta, valid_mods)
    except np.linalg.LinAlgError:
        pass


def predict_tier_ridge(rec):
    cls = rec.get("item_class", "")
    model = class_ridge.get(cls)
    if model is None:
        return cg_median(rec)
    y_mean, beta, valid_mods = model
    n_feat = 4 + len(valid_mods)
    x = np.zeros(n_feat)
    x[0] = (_GRADE_NUM.get(rec.get("grade", "C"), 1) - 2) / 2.0
    x[1] = rec.get("score", 0)
    x[2] = (rec.get("top_tier_count", 0) - 1) / 2.0
    x[3] = (rec.get("mod_count", 4) - 4) / 3.0
    tiers = get_mod_tier_dict(rec)
    for j, mod in enumerate(valid_mods):
        if mod in tiers:
            x[4 + j] = tier_value(tiers[mod])
    log_est = x @ beta + y_mean
    return max(0.01, min(1500, math.exp(log_est)))


eval_approach("F: Per-class Ridge w/ tier-weighted mods", predict_tier_ridge)


# ── Approach G: same but with synergy interaction terms ──────
class_ridge_syn = {}
# Top synergy pairs (short names from top_mods)
SYNERGY_SHORT = [
    ("CritChance", "CritMulti"),
    ("SpellDmg", "CastSpd"),
    ("PhysDmg", "AtkSpd"),
    ("AtkSpd", "CritChance"),
    ("CritMulti", "SpellDmg"),
    ("MoveSpd", "Life"),
    ("Life", "ES"),
]

for cls in set(r.get("item_class", "") for r in train):
    cls_recs = [r for r in train if r.get("item_class", "") == cls]
    if len(cls_recs) < 50:
        continue

    mod_freq = defaultdict(int)
    for r in cls_recs:
        for _, name in parse_top_mods(r.get("top_mods", "")):
            mod_freq[name] += 1
    valid_mods = sorted(m for m, c in mod_freq.items() if c >= 10)
    valid_set = set(valid_mods)

    # Find applicable synergy pairs
    valid_syns = [(a, b) for a, b in SYNERGY_SHORT if a in valid_set and b in valid_set]

    n = len(cls_recs)
    n_feat = 4 + len(valid_mods) + len(valid_syns)
    X = np.zeros((n, n_feat))
    y = np.zeros(n)
    for i, r in enumerate(cls_recs):
        y[i] = math.log(r["min_divine"])
        X[i, 0] = (_GRADE_NUM.get(r.get("grade", "C"), 1) - 2) / 2.0
        X[i, 1] = r.get("score", 0)
        X[i, 2] = (r.get("top_tier_count", 0) - 1) / 2.0
        X[i, 3] = (r.get("mod_count", 4) - 4) / 3.0
        tiers = get_mod_tier_dict(r)
        for j, mod in enumerate(valid_mods):
            if mod in tiers:
                X[i, 4 + j] = tier_value(tiers[mod])
        # Synergy features: product of both tier values (0 if either missing)
        for k, (a, b) in enumerate(valid_syns):
            if a in tiers and b in tiers:
                X[i, 4 + len(valid_mods) + k] = tier_value(tiers[a]) * tier_value(tiers[b])

    y_mean = np.mean(y)
    yc = y - y_mean
    alpha = 1.0
    try:
        beta = np.linalg.solve(X.T @ X + alpha * np.eye(n_feat), X.T @ yc)
        class_ridge_syn[cls] = (y_mean, beta, valid_mods, valid_syns)
    except np.linalg.LinAlgError:
        pass


def predict_tier_ridge_syn(rec):
    cls = rec.get("item_class", "")
    model = class_ridge_syn.get(cls)
    if model is None:
        return cg_median(rec)
    y_mean, beta, valid_mods, valid_syns = model
    n_feat = 4 + len(valid_mods) + len(valid_syns)
    x = np.zeros(n_feat)
    x[0] = (_GRADE_NUM.get(rec.get("grade", "C"), 1) - 2) / 2.0
    x[1] = rec.get("score", 0)
    x[2] = (rec.get("top_tier_count", 0) - 1) / 2.0
    x[3] = (rec.get("mod_count", 4) - 4) / 3.0
    tiers = get_mod_tier_dict(rec)
    for j, mod in enumerate(valid_mods):
        if mod in tiers:
            x[4 + j] = tier_value(tiers[mod])
    for k, (a, b) in enumerate(valid_syns):
        if a in tiers and b in tiers:
            x[4 + len(valid_mods) + k] = tier_value(tiers[a]) * tier_value(tiers[b])
    log_est = x @ beta + y_mean
    return max(0.01, min(1500, math.exp(log_est)))


eval_approach("G: Tier Ridge + synergy interactions", predict_tier_ridge_syn)


# ── Approach H: Ensemble — tier ridge when top_mods populated, cg median otherwise ──
def predict_ensemble(rec):
    if rec.get("top_mods"):
        return predict_tier_ridge(rec)
    return cg_median(rec)


eval_approach("H: Ensemble (tier ridge if top_mods, else cg median)", predict_ensemble)


# ── Data coverage check ──────────────────────────────────────
print("\n=== Data coverage ===")
has_top_mods = sum(1 for r in deduped if r.get("top_mods"))
has_mod_groups = sum(1 for r in deduped if r.get("mod_groups"))
has_base = sum(1 for r in deduped if r.get("base_type"))
print(f"  Records with top_mods:   {has_top_mods}/{len(deduped)} ({has_top_mods/len(deduped)*100:.1f}%)")
print(f"  Records with mod_groups: {has_mod_groups}/{len(deduped)} ({has_mod_groups/len(deduped)*100:.1f}%)")
print(f"  Records with base_type:  {has_base}/{len(deduped)} ({has_base/len(deduped)*100:.1f}%)")

# Check: how does tier ridge do ONLY on items that have top_mods?
print("\n=== Accuracy split by data availability ===")
for label, filter_fn in [
    ("Has top_mods", lambda r: bool(r.get("top_mods"))),
    ("No top_mods", lambda r: not r.get("top_mods")),
    ("Has mod_groups", lambda r: bool(r.get("mod_groups"))),
]:
    subset = [r for r in test if filter_fn(r)]
    within_2x = 0
    total = 0
    for rec in subset:
        actual = rec["min_divine"]
        if actual <= 0: continue
        est = predict_tier_ridge(rec)
        if est is None: continue
        total += 1
        ratio = max(est / actual, actual / est)
        if ratio <= 2.0:
            within_2x += 1
    pct = within_2x / total * 100 if total else 0
    print(f"  {label:25s}: {pct:5.1f}% within 2x ({within_2x}/{total})")
