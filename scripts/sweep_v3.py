"""Final sweep — learned composite score + tier-aware approaches."""

import sys, json, math, random, glob, re
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import numpy as np
from shard_generator import load_raw_records, quality_filter, remove_outliers, dedup_records, _GRADE_NUM

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


def parse_top_mods(tm_str):
    if not tm_str:
        return []
    pairs = []
    for part in tm_str.split(","):
        part = part.strip()
        m = re.match(r"T(\d+)\s+(.+)", part)
        if m:
            pairs.append((int(m.group(1)), m.group(2).strip()))
    return pairs


def eval_full(name, predict_fn):
    buckets = {"<1": [0, 0], "1-5": [0, 0], "5-20": [0, 0], "20-100": [0, 0], "100+": [0, 0]}
    total_2x = 0
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
        hit = ratio <= 2.0
        if hit:
            total_2x += 1
        if actual < 1: b = "<1"
        elif actual < 5: b = "1-5"
        elif actual < 20: b = "5-20"
        elif actual < 100: b = "20-100"
        else: b = "100+"
        buckets[b][0] += int(hit)
        buckets[b][1] += 1
    pct = total_2x / total * 100 if total else 0
    detail = "  ".join(f"{b}:{buckets[b][0]}/{buckets[b][1]}({buckets[b][0]/buckets[b][1]*100:.0f}%)" if buckets[b][1] else f"{b}:n/a" for b in ["<1", "1-5", "5-20", "20-100", "100+"])
    print(f"  {name:45s}: {pct:5.1f}%  {detail}")
    return pct


# ── Per-(class, grade) learned composite score ───────────────
# Within each (class, grade), learn a linear combo of features that best
# predicts log(price), then use that as a 1D index into the price distribution.

cg_models = {}
for cls in set(r.get("item_class", "") for r in train):
    for gn in range(5):  # 0-4
        group = [r for r in train if r.get("item_class") == cls
                 and _GRADE_NUM.get(r.get("grade", "C"), 1) == gn]
        if len(group) < 20:
            continue
        # Features: score, top_tier_count, mod_count, somv_factor
        X = np.zeros((len(group), 4))
        y = np.zeros(len(group))
        for i, r in enumerate(group):
            X[i, 0] = r.get("score", 0)
            X[i, 1] = r.get("top_tier_count", 0)
            X[i, 2] = r.get("mod_count", 4)
            X[i, 3] = r.get("somv_factor", 1.0)
            y[i] = math.log(r["min_divine"])
        y_mean = np.mean(y)
        yc = y - y_mean
        try:
            beta = np.linalg.solve(X.T @ X + 0.1 * np.eye(4), X.T @ yc)
        except np.linalg.LinAlgError:
            continue
        # Store: (y_mean, beta, sorted pairs of (composite_score, log_price))
        composites = X @ beta
        pairs = sorted(zip(composites.tolist(), y.tolist()))
        cg_models[(cls, gn)] = (y_mean, beta.tolist(), pairs)


def predict_composite_score(rec):
    cls = rec.get("item_class", "")
    gn = _GRADE_NUM.get(rec.get("grade", "C"), 1)
    model = cg_models.get((cls, gn))
    if model is None:
        return None
    y_mean, beta, pairs = model
    x = np.array([rec.get("score", 0), rec.get("top_tier_count", 0),
                   rec.get("mod_count", 4), rec.get("somv_factor", 1.0)])
    comp = float(x @ np.array(beta))
    # Find nearest neighbors by composite score in training data
    n = len(pairs)
    # Binary search for position
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if pairs[mid][0] < comp:
            lo = mid + 1
        else:
            hi = mid
    # Take k=7 nearest
    k = min(7, n)
    start = max(0, lo - k // 2)
    end = min(n, start + k)
    start = max(0, end - k)
    neighbors = pairs[start:end]
    if not neighbors:
        return math.exp(y_mean)
    # IDW in log space
    total_w = 0
    weighted_sum = 0
    for cs, lp in neighbors:
        d = abs(cs - comp)
        w = 1.0 / (d + 0.05)
        weighted_sum += w * lp
        total_w += w
    return max(0.01, min(1500, math.exp(weighted_sum / total_w)))


# ── Approach: composite score -> decile -> decile median ─────
cg_deciles = {}
for key, (y_mean, beta, pairs) in cg_models.items():
    n = len(pairs)
    if n < 20:
        continue
    decile_size = n // 10
    deciles = []
    for d in range(10):
        start = d * decile_size
        end = (d + 1) * decile_size if d < 9 else n
        slice_prices = sorted(p[1] for p in pairs[start:end])
        median_lp = slice_prices[len(slice_prices) // 2]
        # Composite score range for this decile
        c_min = pairs[start][0]
        c_max = pairs[end - 1][0]
        deciles.append((c_min, c_max, median_lp))
    cg_deciles[key] = (beta, deciles)


def predict_composite_decile(rec):
    cls = rec.get("item_class", "")
    gn = _GRADE_NUM.get(rec.get("grade", "C"), 1)
    model = cg_deciles.get((cls, gn))
    if model is None:
        return None
    beta, deciles = model
    x = np.array([rec.get("score", 0), rec.get("top_tier_count", 0),
                   rec.get("mod_count", 4), rec.get("somv_factor", 1.0)])
    comp = float(x @ np.array(beta))
    # Find decile
    for c_min, c_max, median_lp in deciles:
        if comp <= c_max:
            return max(0.01, min(1500, math.exp(median_lp)))
    # Above all deciles — use last
    return max(0.01, min(1500, math.exp(deciles[-1][2])))


# ── Approach: tier_score (sum of 1/tier) as feature added to composite ──
cg_models_tier = {}
for cls in set(r.get("item_class", "") for r in train):
    for gn in range(5):
        group = [r for r in train if r.get("item_class") == cls
                 and _GRADE_NUM.get(r.get("grade", "C"), 1) == gn]
        if len(group) < 20:
            continue
        X = np.zeros((len(group), 5))
        y = np.zeros(len(group))
        for i, r in enumerate(group):
            X[i, 0] = r.get("score", 0)
            X[i, 1] = r.get("top_tier_count", 0)
            X[i, 2] = r.get("mod_count", 4)
            X[i, 3] = r.get("somv_factor", 1.0)
            mods = parse_top_mods(r.get("top_mods", ""))
            X[i, 4] = sum(1.0 / t for t, _ in mods) if mods else 0
            y[i] = math.log(r["min_divine"])
        y_mean = np.mean(y)
        yc = y - y_mean
        try:
            beta = np.linalg.solve(X.T @ X + 0.1 * np.eye(5), X.T @ yc)
        except np.linalg.LinAlgError:
            continue
        composites = X @ beta
        pairs = sorted(zip(composites.tolist(), y.tolist()))
        cg_models_tier[(cls, gn)] = (y_mean, beta.tolist(), pairs)


def predict_composite_tier(rec):
    cls = rec.get("item_class", "")
    gn = _GRADE_NUM.get(rec.get("grade", "C"), 1)
    model = cg_models_tier.get((cls, gn))
    if model is None:
        return predict_composite_score(rec)
    y_mean, beta, pairs = model
    mods = parse_top_mods(rec.get("top_mods", ""))
    x = np.array([rec.get("score", 0), rec.get("top_tier_count", 0),
                   rec.get("mod_count", 4), rec.get("somv_factor", 1.0),
                   sum(1.0 / t for t, _ in mods) if mods else 0])
    comp = float(x @ np.array(beta))
    n = len(pairs)
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if pairs[mid][0] < comp:
            lo = mid + 1
        else:
            hi = mid
    k = min(7, n)
    start = max(0, lo - k // 2)
    end = min(n, start + k)
    start = max(0, end - k)
    neighbors = pairs[start:end]
    if not neighbors:
        return math.exp(y_mean)
    total_w = 0
    weighted_sum = 0
    for cs, lp in neighbors:
        d = abs(cs - comp)
        w = 1.0 / (d + 0.05)
        weighted_sum += w * lp
        total_w += w
    return max(0.01, min(1500, math.exp(weighted_sum / total_w)))


# ── Approach: class+grade median as a baseline ───────────────
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


# score bucket median
medians_sb = {}
for rec in train:
    sb = round(rec.get("score", 0) * 10) / 10
    key = (rec.get("item_class", ""), _GRADE_NUM.get(rec.get("grade", "C"), 1), sb)
    medians_sb.setdefault(key, []).append(math.log(rec["min_divine"]))
for k, v in medians_sb.items():
    v.sort()
    medians_sb[k] = v[len(v) // 2]


def predict_sb(rec):
    gn = _GRADE_NUM.get(rec.get("grade", "C"), 1)
    cls = rec.get("item_class", "")
    sb = round(rec.get("score", 0) * 10) / 10
    m = medians_sb.get((cls, gn, sb))
    if m is not None:
        return math.exp(m)
    for d in [0.1, -0.1]:
        m = medians_sb.get((cls, gn, round((sb + d) * 10) / 10))
        if m is not None:
            return math.exp(m)
    return cg_median(rec)


# ── RUN ──────────────────────────────────────────────────────
print("=== Baselines ===")
eval_full("Class+Grade median", cg_median)
eval_full("Class+Grade+Score(0.1) median", predict_sb)

print("\n=== Learned composite score ===")
eval_full("Composite k-NN (score,ttc,mc,somv)", predict_composite_score)
eval_full("Composite decile", predict_composite_decile)
eval_full("Composite+tier k-NN", predict_composite_tier)


# ── Best hybrid: try composite, fall back to score bucket ────
def predict_hybrid(rec):
    est = predict_composite_tier(rec)
    return est if est is not None else predict_sb(rec)


eval_full("Hybrid: composite_tier -> score_bucket", predict_hybrid)


# ── Check: what if we knew the correct price bracket? ────────
# Simulate perfect bracket classification + bracket median
bracket_medians = defaultdict(list)
for rec in train:
    cls = rec.get("item_class", "")
    gn = _GRADE_NUM.get(rec.get("grade", "C"), 1)
    p = rec["min_divine"]
    if p < 1: b = 0
    elif p < 5: b = 1
    elif p < 20: b = 2
    elif p < 100: b = 3
    else: b = 4
    bracket_medians[(cls, gn, b)].append(math.log(p))

for k, v in bracket_medians.items():
    v.sort()
    bracket_medians[k] = v[len(v) // 2]


def predict_oracle_bracket(rec):
    """Cheating: use actual price to pick bracket, then median within bracket."""
    cls = rec.get("item_class", "")
    gn = _GRADE_NUM.get(rec.get("grade", "C"), 1)
    p = rec["min_divine"]
    if p < 1: b = 0
    elif p < 5: b = 1
    elif p < 20: b = 2
    elif p < 100: b = 3
    else: b = 4
    m = bracket_medians.get((cls, gn, b))
    return math.exp(m) if m is not None else cg_median(rec)


print("\n=== Oracle (ceiling analysis) ===")
eval_full("Oracle: perfect bracket + bracket median", predict_oracle_bracket)
print("  ^ This shows the ceiling if we could perfectly classify price brackets")
