"""Extended sweep — finer stratification, price-range analysis, hybrid approaches."""

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

print(f"Train: {len(train)}, Test: {len(test)}\n")


def eval_approach(name, predict_fn, test_data=None, detail=False):
    if test_data is None:
        test_data = test
    within_2x = 0
    total = 0
    by_bucket = defaultdict(lambda: [0, 0])  # bucket -> [within_2x, total]
    for rec in test_data:
        actual = rec["min_divine"]
        if actual <= 0: continue
        est = predict_fn(rec)
        if est is None: continue
        total += 1
        ratio = max(est / actual, actual / est)
        hit = ratio <= 2.0
        if hit:
            within_2x += 1
        # Price buckets
        if actual < 1:
            b = "<1"
        elif actual < 5:
            b = "1-5"
        elif actual < 20:
            b = "5-20"
        elif actual < 100:
            b = "20-100"
        else:
            b = "100+"
        by_bucket[b][0] += int(hit)
        by_bucket[b][1] += 1
    pct = within_2x / total * 100 if total else 0
    print(f"  {name:55s}: {pct:5.1f}% ({within_2x}/{total})")
    if detail:
        for b in ["<1", "1-5", "5-20", "20-100", "100+"]:
            h, t = by_bucket.get(b, [0, 0])
            p = h / t * 100 if t else 0
            print(f"    {b:8s}: {p:5.1f}% ({h}/{t})")
    return pct


# ── Helpers ──────────────────────────────────────────────────

def parse_top_mods(tm_str):
    if not tm_str: return []
    pairs = []
    for part in tm_str.split(","):
        part = part.strip()
        m = re.match(r"T(\d+)\s+(.+)", part)
        if m:
            pairs.append((int(m.group(1)), m.group(2).strip()))
    return pairs


# ── Build lookup tables ──────────────────────────────────────

# Class+Grade median
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


# Class+Grade+Score(0.1) median with fallback
medians_cgs01 = {}
for rec in train:
    sb = round(rec.get("score", 0) * 10) / 10  # 0.1 buckets
    key = (rec.get("item_class", ""), _GRADE_NUM.get(rec.get("grade", "C"), 1), sb)
    medians_cgs01.setdefault(key, []).append(math.log(rec["min_divine"]))
for k, v in medians_cgs01.items():
    v.sort()
    medians_cgs01[k] = v[len(v) // 2]


def predict_cgs01(rec):
    gn = _GRADE_NUM.get(rec.get("grade", "C"), 1)
    cls = rec.get("item_class", "")
    sb = round(rec.get("score", 0) * 10) / 10
    key = (cls, gn, sb)
    m = medians_cgs01.get(key)
    if m is not None:
        return math.exp(m)
    # Try adjacent score buckets
    for delta in [0.1, -0.1, 0.2, -0.2]:
        nearby = round((sb + delta) * 10) / 10
        m = medians_cgs01.get((cls, gn, nearby))
        if m is not None:
            return math.exp(m)
    return cg_median(rec)


# Class+Grade+Score(0.05) finer
medians_cgs005 = {}
for rec in train:
    sb = round(rec.get("score", 0) * 20) / 20  # 0.05 buckets
    key = (rec.get("item_class", ""), _GRADE_NUM.get(rec.get("grade", "C"), 1), sb)
    medians_cgs005.setdefault(key, []).append(math.log(rec["min_divine"]))
for k, v in medians_cgs005.items():
    v.sort()
    medians_cgs005[k] = v[len(v) // 2]


def predict_cgs005(rec):
    gn = _GRADE_NUM.get(rec.get("grade", "C"), 1)
    cls = rec.get("item_class", "")
    sb = round(rec.get("score", 0) * 20) / 20
    key = (cls, gn, sb)
    m = medians_cgs005.get(key)
    if m is not None:
        return math.exp(m)
    for delta in [0.05, -0.05, 0.1, -0.1]:
        nearby = round((sb + delta) * 20) / 20
        m = medians_cgs005.get((cls, gn, nearby))
        if m is not None:
            return math.exp(m)
    return cg_median(rec)


# ── Combined: Class+Grade+Score+ModCount ─────────────────────
medians_cgsm = {}
for rec in train:
    sb = round(rec.get("score", 0) * 5) / 5
    mc = min(rec.get("mod_count", 4), 6)
    key = (rec.get("item_class", ""), _GRADE_NUM.get(rec.get("grade", "C"), 1), sb, mc)
    medians_cgsm.setdefault(key, []).append(math.log(rec["min_divine"]))
for k, v in medians_cgsm.items():
    v.sort()
    medians_cgsm[k] = v[len(v) // 2]


def predict_cgsm(rec):
    gn = _GRADE_NUM.get(rec.get("grade", "C"), 1)
    cls = rec.get("item_class", "")
    sb = round(rec.get("score", 0) * 5) / 5
    mc = min(rec.get("mod_count", 4), 6)
    key = (cls, gn, sb, mc)
    m = medians_cgsm.get(key)
    if m is not None:
        return math.exp(m)
    # Fallback: drop mod_count
    key2 = (cls, gn, sb)
    medians_cgs02 = {}
    for r in train:
        s = round(r.get("score", 0) * 5) / 5
        k2 = (r.get("item_class", ""), _GRADE_NUM.get(r.get("grade", "C"), 1), s)
        medians_cgs02.setdefault(k2, []).append(math.log(r["min_divine"]))
    for k2, v in medians_cgs02.items():
        v.sort()
        medians_cgs02[k2] = v[len(v) // 2]
    m = medians_cgs02.get(key2)
    return math.exp(m) if m is not None else cg_median(rec)


# ── Percentile rank approach ─────────────────────────────────
# Within (class, grade), use score percentile to pick from the price distribution
cg_price_dists = {}
for rec in train:
    key = (rec.get("item_class", ""), _GRADE_NUM.get(rec.get("grade", "C"), 1))
    cg_price_dists.setdefault(key, {"scores": [], "prices": []})
    cg_price_dists[key]["scores"].append(rec.get("score", 0))
    cg_price_dists[key]["prices"].append(rec["min_divine"])

# Sort prices for percentile lookup
cg_sorted = {}
for key, data in cg_price_dists.items():
    # Sort by score, get corresponding price sorted
    pairs = list(zip(data["scores"], data["prices"]))
    pairs.sort(key=lambda x: x[0])
    cg_sorted[key] = {
        "scores": [p[0] for p in pairs],
        "prices": [p[1] for p in pairs],
        "sorted_prices": sorted(data["prices"]),
    }


def predict_percentile(rec):
    key = (rec.get("item_class", ""), _GRADE_NUM.get(rec.get("grade", "C"), 1))
    data = cg_sorted.get(key)
    if data is None or len(data["scores"]) < 10:
        return cg_median(rec)

    score = rec.get("score", 0)
    scores = data["scores"]
    n = len(scores)

    # Find score percentile
    rank = sum(1 for s in scores if s <= score)
    pctile = rank / n

    # Pick that percentile from the price distribution
    sorted_prices = data["sorted_prices"]
    idx = min(int(pctile * len(sorted_prices)), len(sorted_prices) - 1)
    return sorted_prices[idx]


# ── top_mods signature matching ──────────────────────────────
# Hash top_mods (ignoring tiers) as an item "signature" for exact matching
from collections import Counter

sig_prices = defaultdict(list)  # (class, grade, mod_sig) -> [prices]
for rec in train:
    mods = parse_top_mods(rec.get("top_mods", ""))
    if not mods:
        continue
    # Signature = frozenset of mod SHORT names (ignore tier)
    sig = frozenset(name for _, name in mods)
    key = (rec.get("item_class", ""), _GRADE_NUM.get(rec.get("grade", "C"), 1), sig)
    sig_prices[key].append(math.log(rec["min_divine"]))

sig_medians = {}
for k, v in sig_prices.items():
    if len(v) >= 3:
        v.sort()
        sig_medians[k] = v[len(v) // 2]


def predict_mod_signature(rec):
    mods = parse_top_mods(rec.get("top_mods", ""))
    if mods:
        sig = frozenset(name for _, name in mods)
        key = (rec.get("item_class", ""), _GRADE_NUM.get(rec.get("grade", "C"), 1), sig)
        m = sig_medians.get(key)
        if m is not None:
            return math.exp(m)
    return cg_median(rec)


# ── top_mods signature + tier quality ────────────────────────
sig_tier_prices = defaultdict(list)
for rec in train:
    mods = parse_top_mods(rec.get("top_mods", ""))
    if not mods:
        continue
    sig = frozenset(name for _, name in mods)
    # Tier quality bucket: avg tier -> bucket (1-3=high, 4-6=mid, 7+=low)
    avg_tier = sum(t for t, _ in mods) / len(mods)
    tier_bucket = "high" if avg_tier <= 3 else ("mid" if avg_tier <= 6 else "low")
    key = (rec.get("item_class", ""), _GRADE_NUM.get(rec.get("grade", "C"), 1), sig, tier_bucket)
    sig_tier_prices[key].append(math.log(rec["min_divine"]))

sig_tier_medians = {}
for k, v in sig_tier_prices.items():
    if len(v) >= 3:
        v.sort()
        sig_tier_medians[k] = v[len(v) // 2]


def predict_sig_tier(rec):
    mods = parse_top_mods(rec.get("top_mods", ""))
    if mods:
        sig = frozenset(name for _, name in mods)
        avg_tier = sum(t for t, _ in mods) / len(mods)
        tier_bucket = "high" if avg_tier <= 3 else ("mid" if avg_tier <= 6 else "low")
        key = (rec.get("item_class", ""), _GRADE_NUM.get(rec.get("grade", "C"), 1), sig, tier_bucket)
        m = sig_tier_medians.get(key)
        if m is not None:
            return math.exp(m)
        # Fallback to signature without tier
        return predict_mod_signature(rec)
    return cg_median(rec)


# ── Approach: signature matching with score interpolation ────
sig_score_data = defaultdict(lambda: {"scores": [], "log_prices": []})
for rec in train:
    mods = parse_top_mods(rec.get("top_mods", ""))
    if not mods:
        continue
    sig = frozenset(name for _, name in mods)
    key = (rec.get("item_class", ""), sig)
    sig_score_data[key]["scores"].append(rec.get("score", 0))
    sig_score_data[key]["log_prices"].append(math.log(rec["min_divine"]))


def predict_sig_interpolate(rec):
    mods = parse_top_mods(rec.get("top_mods", ""))
    if mods:
        sig = frozenset(name for _, name in mods)
        key = (rec.get("item_class", ""), sig)
        data = sig_score_data.get(key)
        if data and len(data["scores"]) >= 5:
            # Within this signature group, interpolate by score
            score = rec.get("score", 0)
            pairs = list(zip(data["scores"], data["log_prices"]))
            pairs.sort(key=lambda x: x[0])
            # Find nearest neighbors by score
            dists = [(abs(s - score), lp) for s, lp in pairs]
            dists.sort()
            k = min(5, len(dists))
            neighbors = dists[:k]
            total_w = 0
            weighted_sum = 0
            for d, lp in neighbors:
                w = 1.0 / (d + 0.02)
                weighted_sum += w * lp
                total_w += w
            return math.exp(weighted_sum / total_w)
    return predict_cgs01(rec)


# ── RUN ALL APPROACHES ──────────────────────────────────────
print("=== Stratification approaches ===")
eval_approach("Class+Grade median (baseline)", cg_median, detail=True)
print()
eval_approach("Class+Grade+Score(0.1)", predict_cgs01)
eval_approach("Class+Grade+Score(0.05)", predict_cgs005)
eval_approach("Score percentile within class+grade", predict_percentile)

print("\n=== Mod signature approaches (using top_mods) ===")
eval_approach("Mod signature (ignore tier)", predict_mod_signature, detail=True)
print()
eval_approach("Mod signature + tier quality bucket", predict_sig_tier, detail=True)
print()
eval_approach("Mod signature + score interpolation", predict_sig_interpolate, detail=True)

# Check how many test items get signature matches
matched = sum(1 for r in test if parse_top_mods(r.get("top_mods", ""))
              and (r.get("item_class", ""), _GRADE_NUM.get(r.get("grade", "C"), 1),
                   frozenset(n for _, n in parse_top_mods(r.get("top_mods", ""))))
              in sig_medians)
print(f"\n  Signature match rate: {matched}/{len(test)} ({matched/len(test)*100:.1f}%)")

# Check signature counts
sig_counts = defaultdict(int)
for key, v in sig_prices.items():
    sig_counts[len(v)] += 1
print(f"  Signature group sizes: " +
      ", ".join(f"{k}={v}" for k, v in sorted(sig_counts.items())[:10]))

print(f"\n  Unique signatures: {len(sig_prices)}")
print(f"  Signatures with >=3 samples: {len(sig_medians)}")
print(f"  Signatures with >=10 samples: {sum(1 for v in sig_prices.values() if len(v) >= 10)}")
