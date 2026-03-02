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

# Sample tuple positions:
#  [0]  score              float   normalized item score
#  [1]  divine             float   price in divine orbs
#  [2]  grade_num          int     0=JUNK..4=S
#  [3]  dps_factor         float   combat DPS multiplier
#  [4]  defense_factor     float   defense multiplier
#  [5]  top_tier_count     int     T1/T2 mods with weight >= 1.0 (0-6)
#  [6]  mod_count          int     total parsed mods (0-12)
#  [7]  ts                 int     timestamp (epoch seconds, 0 for shard)
#  [8]  is_user            bool    True for user data (recency-weighted)
#  [9]  mod_groups         Tuple[str, ...]   sorted mod group names
#  [10] base_type          str     item base type
#  [11] tier_score         float   sum(1/tier) for all mods
#  [12] best_tier          int     lowest (best) tier number
#  [13] mod_tiers_tuple    Tuple[Tuple[str, int], ...]  (group, tier) pairs
#  [14] coc_score          float   CoC archetype alignment [0.0, 1.0]
#  [15] es_score           float   CI/ES archetype alignment
#  [16] mana_score         float   MoM/mana archetype alignment
#  [17] somv_factor        float   average roll quality (from score_result)
#  [18] mod_rolls_tuple    Tuple[Tuple[str, float], ...]  (group, roll_quality) pairs
#  [19] pdps               float   physical DPS
#  [20] edps               float   elemental DPS
#  [21] sale_confidence    float   disappearance-based confidence (3.0=sold, 1.0=unknown, 0.3=stale)
#  [22] mod_stats_tuple    Tuple[Tuple[str, float], ...]  sorted (stat_id, value) pairs
#  [23] quality            int     item quality (0-20)
#  [24] sockets            int     socket count
#  [25] corrupted          int     1=corrupted, 0=not
#  [26] open_prefixes      int     3 - prefix_count
#  [27] open_suffixes      int     3 - suffix_count
Sample = Tuple[float, float, int, float, float, int, int, int, bool,
               Tuple[str, ...], str, float, int, Tuple[Tuple[str, int], ...],
               float, float, float, float, Tuple[Tuple[str, float], ...],
               float, float, float, Tuple[Tuple[str, float], ...],
               int, int, int, int, int]


class CalibrationEngine:
    MIN_CLASS_SAMPLES = 10
    MIN_GLOBAL_SAMPLES = 50
    _EPSILON = 0.02  # smoothing floor; exact match gets ~50x weight vs dist=1.0

    # Distance metric weights — tuned for actual data coverage
    SCORE_WEIGHT = 0.05        # low: score is redundant with other features
    GRADE_PENALTY = 0.30       # per grade step
    TOP_TIER_WEIGHT = 0.25     # top-tier mod count difference
    MOD_COUNT_WEIGHT = 0.10    # total mod count difference
    DPS_WEIGHT = 0.25          # DPS factor difference — captures roll quality for weapons
    DEFENSE_WEIGHT = 0.25      # defense factor difference — captures roll quality for armor
    MOD_IDENTITY_WEIGHT = 1.2  # weighted Jaccard distance — THE key price signal (boosted)
    MOD_TIER_WEIGHT = 0.25     # per-shared-mod tier mismatch penalty
    BASE_TYPE_WEIGHT = 0.20    # binary: same base=0, different=0.20
    TIER_SCORE_WEIGHT = 0.15   # tier quality (sum(1/tier)) aggregate difference
    ARCHETYPE_WEIGHT = 0.15    # per-archetype score difference
    SOMV_WEIGHT = 0.25         # average roll quality — reduced, somv ~1.0 for most data
    ROLL_QUALITY_WEIGHT = 0.20 # per-mod roll quality difference
    DPS_TYPE_WEIGHT = 0.08     # phys vs elemental DPS — reduced, <1% coverage
    DEMAND_WEIGHT = 0.15       # meta demand score difference
    QUALITY_WEIGHT = 0.10      # quality % difference / 20
    SOCKET_WEIGHT = 0.15       # binary penalty for different socket counts
    OPEN_AFFIX_WEIGHT = 0.10   # prefix/suffix slot difference

    # Core-mod pricing threshold: mods with log-price uplift above this are "core"
    CORE_MOD_THRESHOLD = 0.3  # ~1.35x price increase = core mod

    # Group-prior blending: when k-NN neighbors are distant, blend toward
    # group median to prevent wild extrapolation.
    # Diagnostic showed grade median (34.3%) beats k-NN (32.8%) for distant
    # neighbors, so blend toward median early.
    BLEND_START_DIST = 1.5     # lowered: blend sooner when neighbors are distant
    BLEND_FULL_DIST = 3.5      # full blend at this distance

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
        # Price tables: "class|grade_num" -> {weights, deciles, y_mean}
        self._price_tables: Dict[str, dict] = {}
        # GBM models: item_class -> serialized model dict (shard v6+)
        self._gbm_models: Dict[str, dict] = {}
        # Last estimate confidence (0.0-1.0, higher = better)
        self.last_confidence: float = 0.0
        # Price range: 25th/75th percentile from neighbors
        self.last_estimate_low: float = 0.0
        self.last_estimate_high: float = 0.0
        # Confidence tier: "HIGH", "MEDIUM", "LOW"
        self.last_confidence_tier: str = ""
        # Value tier: "HIGH", "MID", "LOW" (from core-mod market values)
        self.last_value_tier: str = ""
        # Demand index: per-(item_class, mod_group) demand scores
        self._demand_index = None  # Optional[DemandIndex]
        # Mod-set lookup: (item_class, frozenset(mod_groups)) -> [log_price, ...]
        self._modset_lookup: Dict[Tuple[str, frozenset], List[float]] = {}
        self._modset_lookup_dirty: bool = True
        # Core-mod pricing: learned mod market values and core mod sets
        self._mod_market_values: Dict[str, Dict[str, float]] = {}  # class -> {mod_group: uplift}
        self._core_mods: Dict[str, frozenset] = {}  # class -> frozenset of core mod groups
        self._core_modset_lookup: Dict[Tuple[str, frozenset], List[Tuple[float, float, float]]] = {}
        self._last_modset_match_type: str = ""  # track which tier matched

    @property
    def _k(self) -> int:
        """Scale k with data size: min(10, len/5), floor 5."""
        n = len(self._global)
        if n < 30:
            return 3
        return min(10, max(5, n // 5))

    @staticmethod
    def _k_for_pool(pool_size: int) -> int:
        """Pool-aware k: sqrt-scaled with bounds [3, 15].

        Small pools (< 10 samples) get k=3 to avoid averaging noise.
        Large pools scale with sqrt for stable estimates.
        """
        if pool_size < 10:
            return 3
        return min(15, max(3, int(math.sqrt(pool_size))))

    @staticmethod
    def _weighted_percentile(wt_pairs: list, quantile: float) -> float:
        """Return the log-price at a weighted quantile from sorted (log_price, weight) pairs.

        wt_pairs must be sorted by log_price ascending.
        quantile is 0.0-1.0 (e.g. 0.25 for 25th percentile).
        """
        if not wt_pairs:
            return 0.0
        total_weight = sum(w for _, w in wt_pairs)
        if total_weight <= 0:
            return wt_pairs[0][0]
        target = total_weight * quantile
        cumulative = 0.0
        for lp, w in wt_pairs:
            cumulative += w
            if cumulative >= target:
                return lp
        return wt_pairs[-1][0]

    def set_mod_weights(self, weights: Dict[str, float]):
        """Set mod importance weights for weighted Jaccard distance."""
        self._mod_weights = dict(weights)

    def set_demand_index(self, demand_index):
        """Set demand index for meta-aware pricing."""
        self._demand_index = demand_index

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
                    # Include mod composition so distinct items with same score/price are kept
                    dedup_key = (round(score, 3), round(divine, 2), item_class, grade,
                                 tuple(sorted(rec.get("mod_groups", []))))
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
                    mod_tiers = rec.get("mod_tiers", {})
                    mod_rolls = rec.get("mod_rolls", {})
                    rec_somv = rec.get("somv_factor", 1.0)
                    rec_pdps = rec.get("pdps", 0.0)
                    rec_edps = rec.get("edps", 0.0)
                    rec_sc = rec.get("sale_confidence", 1.0)
                    rec_mod_stats = rec.get("mod_stats", {})
                    self._insert(float(score), float(divine),
                                 item_class, grade_num,
                                 dps_factor, defense_factor,
                                 top_tier_count=top_tier_count,
                                 mod_count=mod_count,
                                 ts=ts, is_user=True,
                                 mod_groups=mod_groups,
                                 base_type=base_type,
                                 mod_tiers=mod_tiers,
                                 somv_factor=rec_somv,
                                 mod_rolls=mod_rolls,
                                 pdps=rec_pdps, edps=rec_edps,
                                 sale_confidence=rec_sc,
                                 mod_stats=rec_mod_stats,
                                 quality=rec.get("quality", 0),
                                 sockets=rec.get("sockets", 0),
                                 corrupted=1 if rec.get("corrupted", False) else 0,
                                 open_prefixes=rec.get("open_prefixes", 0),
                                 open_suffixes=rec.get("open_suffixes", 0))
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

            # Load stat index (v8+ shards)
            stat_index = shard.get("stat_index", [])

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
                m_indices = s.get("m", [])
                mod_groups = [idx_to_group[idx] for idx in m_indices
                              if idx in idx_to_group]
                bt_idx = s.get("b")
                base_type = base_index[bt_idx] if (bt_idx is not None and bt_idx < len(base_index)) else ""

                # Reconstruct mod_tiers from parallel arrays
                mt_arr = s.get("mt", [])
                mod_tiers_dict = {}
                if mt_arr and len(mt_arr) == len(m_indices):
                    for j, idx in enumerate(m_indices):
                        if idx in idx_to_group and mt_arr[j] > 0:
                            mod_tiers_dict[idx_to_group[idx]] = mt_arr[j]

                # Reconstruct mod_rolls from parallel arrays (v7+)
                mr_arr = s.get("mr", [])
                mod_rolls_dict = {}
                if mr_arr and len(mr_arr) == len(m_indices):
                    for j, idx in enumerate(m_indices):
                        if idx in idx_to_group and mr_arr[j] >= 0:
                            mod_rolls_dict[idx_to_group[idx]] = mr_arr[j]

                # Reconstruct mod_stats from stat index arrays (v8+)
                ms_arr = s.get("ms", [])
                mv_arr = s.get("mv", [])
                mod_stats_dict = {}
                if ms_arr and mv_arr and len(ms_arr) == len(mv_arr) and stat_index:
                    for j, sidx in enumerate(ms_arr):
                        if sidx < len(stat_index):
                            mod_stats_dict[stat_index[sidx]] = mv_arr[j]

                somv_factor = s.get("v", 1.0)
                pdps = s.get("pd", 0.0)
                edps = s.get("ed", 0.0)
                sale_confidence = s.get("sc", 1.0)

                self._insert(float(score), float(price),
                             item_class, grade_num,
                             dps_factor, defense_factor,
                             top_tier_count=top_tier_count,
                             mod_count=mod_count,
                             ts=0, is_user=False,
                             mod_groups=mod_groups,
                             base_type=base_type,
                             mod_tiers=mod_tiers_dict,
                             somv_factor=somv_factor,
                             mod_rolls=mod_rolls_dict,
                             pdps=pdps, edps=edps,
                             sale_confidence=sale_confidence,
                             mod_stats=mod_stats_dict,
                             quality=s.get("q", 0),
                             sockets=s.get("sk", 0),
                             corrupted=s.get("cr", 0),
                             open_prefixes=s.get("op", 0),
                             open_suffixes=s.get("os", 0))
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

            # Load price tables (v5+ shards)
            pt_data = shard.get("price_tables")
            if pt_data:
                self._price_tables = dict(pt_data)
                logger.info(f"Loaded {len(self._price_tables)} price tables")

            # Load GBM models (v6+ shards)
            gbm_data = shard.get("gbm_models")
            if gbm_data:
                self._gbm_models = dict(gbm_data)
                logger.info(f"Loaded {len(self._gbm_models)} GBM models")

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
                   mod_groups: list = None, base_type: str = "",
                   mod_tiers: dict = None, somv_factor: float = 1.0,
                   mod_rolls: dict = None, pdps: float = 0.0,
                   edps: float = 0.0, mod_stats: dict = None,
                   quality: int = 0, sockets: int = 0,
                   corrupted: int = 0, open_prefixes: int = 0,
                   open_suffixes: int = 0):
        """Live-add a calibration point (called after each deep query)."""
        if divine <= 0:
            return
        if divine > self._MAX_PRICE_DIVINE:
            return
        grade_num = _GRADE_NUM.get(grade, 1)
        self._insert(score, divine, item_class, grade_num,
                     top_tier_count=top_tier_count, mod_count=mod_count,
                     ts=int(time.time()), is_user=True,
                     mod_groups=mod_groups, base_type=base_type,
                     mod_tiers=mod_tiers, somv_factor=somv_factor,
                     mod_rolls=mod_rolls, pdps=pdps, edps=edps,
                     mod_stats=mod_stats,
                     quality=quality, sockets=sockets,
                     corrupted=corrupted, open_prefixes=open_prefixes,
                     open_suffixes=open_suffixes)

    def _table_estimate(self, item_class: str, grade_num: int,
                        score: float, top_tier_count: int,
                        mod_count: int, somv_factor: float,
                        tier_score: float = 0.0,
                        best_tier: int = 0,
                        avg_tier: float = 0.0,
                        coc_score: float = 0.0,
                        es_score: float = 0.0,
                        mana_score: float = 0.0) -> Optional[float]:
        """Price table estimate via composite score deciles.

        Computes a composite score from learned weights, finds the matching
        decile, and returns the decile median price. Returns None if no
        table exists for this (class, grade).
        """
        key = f"{item_class}|{grade_num}"
        table = self._price_tables.get(key)
        if table is None:
            return None

        weights = table.get("weights")
        deciles = table.get("deciles")
        if not weights or not deciles:
            return None

        # Compute composite score: dot product of features with learned weights
        # Support old (4-weight), tier (7-weight), and archetype (10-weight) tables
        if len(weights) == 10:
            features = [score, top_tier_count, mod_count, somv_factor,
                        tier_score, best_tier, avg_tier,
                        coc_score, es_score, mana_score]
        elif len(weights) == 7:
            features = [score, top_tier_count, mod_count, somv_factor,
                        tier_score, best_tier, avg_tier]
        else:
            features = [score, top_tier_count, mod_count, somv_factor]
        composite = sum(w * f for w, f in zip(weights, features))

        # Find the decile this composite falls into
        median_lp = deciles[-1][1]  # default to last decile
        for upper, med_lp in deciles:
            if composite <= upper:
                median_lp = med_lp
                break

        est = math.exp(median_lp)

        # Cap to observed range
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

    def _regression_estimate(self, item_class: str, mod_groups: list,
                             base_type: str, grade_num: int,
                             top_tier_count: int, mod_count: int,
                             dps_factor: float, defense_factor: float,
                             somv_factor: float = 1.0,
                             mod_tiers: dict = None) -> Optional[float]:
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
            mod_tiers=mod_tiers,
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

    def _gbm_estimate(self, item_class: str, grade_num: int, score: float,
                      top_tier_count: int, mod_count: int,
                      dps_factor: float, defense_factor: float,
                      somv_factor: float, tier_score: float, best_tier: int,
                      avg_tier: float, coc_score: float, es_score: float,
                      mana_score: float, mod_groups: list = None,
                      base_type: str = "", mod_tiers: dict = None,
                      mod_rolls: dict = None, pdps: float = 0.0,
                      edps: float = 0.0,
                      demand_score: float = 0.0,
                      mod_stats: dict = None,
                      item_level: int = 0,
                      armour: int = 0, evasion: int = 0,
                      energy_shield: int = 0,
                      quality: int = 0, sockets: int = 0,
                      corrupted: int = 0,
                      open_prefixes: int = 0,
                      open_suffixes: int = 0) -> Optional[float]:
        """GBM estimate via pure-Python tree traversal. No sklearn at runtime.

        Returns estimated divine value, or None if no model available.
        """
        model = self._gbm_models.get(item_class)
        if model is None:
            return None

        feature_names = model.get("feature_names", [])
        if not feature_names:
            return None

        # Build feature dict matching training feature order
        features = {
            "grade_num": grade_num,
            "score": score,
            "top_tier_count": top_tier_count,
            "mod_count": mod_count,
            "dps_factor": dps_factor,
            "defense_factor": defense_factor,
            "somv_factor": somv_factor,
            "tier_score": tier_score,
            "best_tier": best_tier,
            "avg_tier": avg_tier,
            "arch_coc_spell": coc_score,
            "arch_ci_es": es_score,
            "arch_mom_mana": mana_score,
            "pdps": pdps,
            "edps": edps,
            "demand_score": demand_score,
            "item_level": item_level,
            "armour": armour,
            "evasion": evasion,
            "energy_shield": energy_shield,
            "total_dps": pdps + edps,
            "total_defense": armour + evasion + energy_shield,
            "quality": quality,
            "sockets": sockets,
            "corrupted": corrupted,
            "open_prefixes": open_prefixes,
            "open_suffixes": open_suffixes,
        }

        # Mod features: roll-quality-weighted tier encoding
        mt = mod_tiers or {}
        mr = mod_rolls or {}
        mg_set = set(mod_groups) if mod_groups else set()
        for g in mg_set:
            tier = mt.get(g, 0)
            rq = mr.get(g, -1)
            if rq >= 0 and tier > 0:
                features[f"mod:{g}"] = rq * (1.0 / tier)
            elif tier > 0:
                features[f"mod:{g}"] = 0.5 * (1.0 / tier)
            else:
                features[f"mod:{g}"] = 0.25

        # Base type features: one-hot
        if base_type:
            features[f"base:{base_type}"] = 1.0

        # Stat features: raw stat_id values (from mod_stats)
        if mod_stats:
            for sid, val in mod_stats.items():
                features[f"stat:{sid}"] = val

        # Build feature array in training order
        feat_array = [features.get(fn, 0.0) for fn in feature_names]

        # Traverse all trees
        pred = model["base_prediction"]
        lr = model["learning_rate"]
        for tree in model["trees"]:
            t_feat = tree["feature"]
            t_thresh = tree["threshold"]
            t_left = tree["left"]
            t_right = tree["right"]
            t_val = tree["value"]
            node = 0
            while t_feat[node] >= 0:
                if feat_array[t_feat[node]] <= t_thresh[node]:
                    node = t_left[node]
                else:
                    node = t_right[node]
            pred += lr * t_val[node]

        est = math.exp(pred)

        # Cap to observed range
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

    def _grade_median_estimate(self, item_class: str,
                               grade_num: int) -> Optional[float]:
        """Last-resort estimate using class+grade median price."""
        if self._group_medians_dirty:
            self._recompute_group_medians()
        median = self._group_medians.get((grade_num, item_class))
        if median is None:
            median = self._group_medians.get((grade_num, ""))
        if median is None:
            return None
        est = math.exp(median)
        if est >= 10:
            return round(est, 0)
        elif est >= 1:
            return round(est, 1)
        else:
            return round(est, 2)

    def _assign_confidence_tier(self):
        """Assign HIGH/MEDIUM/LOW confidence tier based on neighbor price range.

        Range ratio (75th/25th percentile) is the only reliable accuracy signal.
        Confidence score clusters too narrowly to discriminate.
        Thresholds tuned from accuracy_lab diagnostic:
          ratio < 5x  → 38% within-2x (25% of items)
          ratio < 50x → 33% within-2x (66% of items)
          ratio >= 50x → 27% within-2x (9% of items)
        """
        low = self.last_estimate_low
        high = self.last_estimate_high
        ratio = high / low if low > 0 else 999.0
        if ratio < 5.0:
            self.last_confidence_tier = "HIGH"
        elif ratio < 50.0:
            self.last_confidence_tier = "MEDIUM"
        else:
            self.last_confidence_tier = "LOW"

    def _compute_value_tier(self, item_class: str, mod_groups: list):
        """Assign HIGH/MID/LOW value tier from learned core-mod market values."""
        market_vals = self._mod_market_values.get(item_class)
        if not market_vals or not mod_groups:
            self.last_value_tier = "LOW"
            return
        total_uplift = sum(market_vals.get(g, 0.0) for g in mod_groups)
        if total_uplift > 1.5:
            self.last_value_tier = "HIGH"
        elif total_uplift > 0.3:
            self.last_value_tier = "MID"
        else:
            self.last_value_tier = "LOW"

    def estimate(self, score: float, item_class: str,
                 grade: str = "", dps_factor: float = 1.0,
                 defense_factor: float = 1.0,
                 top_tier_count: int = 0,
                 mod_count: int = 4,
                 mod_groups: list = None,
                 base_type: str = "",
                 somv_factor: float = 1.0,
                 mod_tiers: dict = None,
                 mod_rolls: dict = None,
                 pdps: float = 0.0,
                 edps: float = 0.0,
                 mod_stats: dict = None,
                 item_level: int = 0,
                 armour: int = 0,
                 evasion: int = 0,
                 energy_shield: int = 0,
                 quality: int = 0,
                 sockets: int = 0,
                 corrupted: int = 0,
                 open_prefixes: int = 0,
                 open_suffixes: int = 0) -> Optional[float]:
        """Return estimated divine value, or None if insufficient data.

        Priority: modset (exact→n-1→core-mod) > class k-NN > GBM > global k-NN > grade median.

        Also stores the last confidence value in self.last_confidence
        (0.0-1.0 scale, higher = more confident estimate).
        """
        grade_num = _GRADE_NUM.get(grade, 1)

        # Reset range/tier fields for this estimate
        self.last_estimate_low = 0.0
        self.last_estimate_high = 0.0
        self.last_confidence_tier = ""
        self.last_value_tier = ""

        # Compute tier aggregates from mod_tiers
        mt = mod_tiers or {}
        tiers = [t for t in mt.values() if t > 0]
        tier_score = round(sum(1.0 / t for t in tiers), 3) if tiers else 0.0
        best_tier = min(tiers) if tiers else 0
        avg_tier = round(sum(tiers) / len(tiers), 2) if tiers else 0.0

        # Compute archetype scores
        from weight_learner import compute_archetype_scores
        arch = compute_archetype_scores(mod_groups or [])
        coc_score = arch.get("coc_spell", 0.0)
        es_score = arch.get("ci_es", 0.0)
        mana_score = arch.get("mom_mana", 0.0)

        # Compute demand score if demand index is available
        query_demand = 0.0
        if self._demand_index and mod_groups:
            query_demand = self._demand_index.get_demand_score(
                item_class, mod_groups)

        # Build mod_stats tuple for k-NN stat distance
        ms_tuple = tuple(sorted(mod_stats.items())) if mod_stats else ()
        mg_set = frozenset(mod_groups) if mod_groups else frozenset()

        # Cascade with selective blending: first good match wins.
        # When modset fires, check if k-NN agrees — if so, average them.
        # This avoids the pitfall of full blending with miscalibrated confidence.

        # k-NN args shared across class/global calls
        _knn_kwargs = dict(
            mod_groups=mg_set, base_type=base_type,
            tier_score=tier_score,
            coc_score=coc_score, es_score=es_score,
            mana_score=mana_score,
            mod_tiers=mod_tiers, somv_factor=somv_factor,
            mod_rolls=mod_rolls, pdps=pdps, edps=edps,
            demand_score=query_demand,
            mod_stats_tuple=ms_tuple,
            quality=quality, sockets=sockets,
            corrupted=corrupted,
            open_prefixes=open_prefixes,
            open_suffixes=open_suffixes)

        # Modset match-type -> confidence mapping
        _MODSET_CONFIDENCE = {
            "exact_grade": 0.75, "exact": 0.73,
            "n_minus_1": 0.70, "core_mod": 0.65,
        }

        # 1. Try mod-set lookup (exact → n-1 → core-mod)
        if mod_groups:
            modset_est = self._modset_estimate(
                item_class, frozenset(mod_groups), grade_num, tier_score,
                somv_factor=somv_factor)
            if modset_est is not None:
                self.last_confidence = _MODSET_CONFIDENCE.get(
                    self._last_modset_match_type, 0.70)
                self._assign_confidence_tier()
                self._compute_value_tier(item_class, mod_groups)
                return modset_est

        # 2. Class k-NN (with value-weighted distance) — before GBM since
        #    learned mod weights give k-NN better mod-identity awareness
        class_samples = self._by_class.get(item_class)
        if class_samples and len(class_samples) >= self.MIN_CLASS_SAMPLES:
            result, confidence = self._interpolate(
                score, class_samples, grade_num,
                dps_factor, defense_factor,
                top_tier_count, mod_count,
                item_class, **_knn_kwargs)
            self.last_confidence = confidence
            self._assign_confidence_tier()
            self._compute_value_tier(item_class, mod_groups)
            return result

        # 3. GBM fallback (for classes with trained models but few class samples)
        if mod_groups and self._gbm_models:
            gbm_est = self._gbm_estimate(
                item_class, grade_num, score,
                top_tier_count, mod_count, dps_factor, defense_factor,
                somv_factor, tier_score, best_tier, avg_tier,
                coc_score, es_score, mana_score,
                mod_groups=mod_groups, base_type=base_type,
                mod_tiers=mod_tiers, mod_rolls=mod_rolls,
                pdps=pdps, edps=edps,
                demand_score=query_demand,
                mod_stats=mod_stats,
                item_level=item_level,
                armour=armour, evasion=evasion,
                energy_shield=energy_shield,
                quality=quality, sockets=sockets,
                corrupted=corrupted,
                open_prefixes=open_prefixes,
                open_suffixes=open_suffixes)
            if gbm_est is not None:
                self.last_confidence = 0.6
                # GBM doesn't produce range — use point estimate as range
                self.last_estimate_low = gbm_est * 0.5
                self.last_estimate_high = gbm_est * 2.0
                self._assign_confidence_tier()
                self._compute_value_tier(item_class, mod_groups)
                return gbm_est

        # 4. Global k-NN
        if len(self._global) >= self.MIN_GLOBAL_SAMPLES:
            result, confidence = self._interpolate(
                score, self._global, grade_num,
                dps_factor, defense_factor,
                top_tier_count, mod_count,
                item_class, **_knn_kwargs)
            self.last_confidence = confidence * 0.7  # global pool is less specific
            self._assign_confidence_tier()
            self._compute_value_tier(item_class, mod_groups)
            return result

        # 5. Last resort: class+grade median
        self.last_confidence = 0.1  # very low confidence
        self._assign_confidence_tier()
        self._compute_value_tier(item_class, mod_groups)
        return self._grade_median_estimate(item_class, grade_num)

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
                mod_groups: list = None, base_type: str = "",
                mod_tiers: dict = None, somv_factor: float = 1.0,
                mod_rolls: dict = None, pdps: float = 0.0,
                edps: float = 0.0, sale_confidence: float = 1.0,
                mod_stats: dict = None,
                quality: int = 0, sockets: int = 0,
                corrupted: int = 0, open_prefixes: int = 0,
                open_suffixes: int = 0):
        """Insert a sample into class-specific and global lists (sorted)."""
        mg_tuple = tuple(sorted(set(mod_groups))) if mod_groups else ()
        # Compute tier aggregates
        mt = mod_tiers or {}
        tiers = [t for t in mt.values() if t > 0]
        tier_score = round(sum(1.0 / t for t in tiers), 3) if tiers else 0.0
        best_tier = min(tiers) if tiers else 0
        mt_tuple = tuple(sorted(mt.items())) if mt else ()
        # Build mod_rolls tuple
        mr = mod_rolls or {}
        mr_tuple = tuple(sorted(mr.items())) if mr else ()
        # Build mod_stats tuple (raw stat_id -> value pairs)
        ms = mod_stats or {}
        ms_tuple = tuple(sorted(ms.items())) if ms else ()
        # Compute archetype scores
        from weight_learner import compute_archetype_scores
        arch = compute_archetype_scores(list(mg_tuple))
        entry: Sample = (score, divine, grade_num, dps_factor, defense_factor,
                         top_tier_count, mod_count, ts, is_user, mg_tuple, base_type,
                         tier_score, best_tier, mt_tuple,
                         arch.get("coc_spell", 0.0),
                         arch.get("ci_es", 0.0),
                         arch.get("mom_mana", 0.0),
                         somv_factor, mr_tuple, pdps, edps, sale_confidence,
                         ms_tuple,
                         int(quality), int(sockets), int(corrupted),
                         int(open_prefixes), int(open_suffixes))
        self._group_medians_dirty = True
        self._modset_lookup_dirty = True
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

    def _build_modset_lookup(self):
        """Build mod-set lookup tables:

        Exact: (item_class, full_mod_set) -> [(log_price, tier_score, somv_factor)]
        Also grade-stratified: (item_class, mod_set, grade_num) -> [...]
        """
        self._modset_lookup.clear()
        self._modset_grade_lookup: Dict[Tuple[str, frozenset, int], List[Tuple[float, float, float]]] = {}
        for item_class, samples in self._by_class.items():
            for s in samples:
                mg = frozenset(s[9]) if s[9] else frozenset()
                if not mg:
                    continue
                lp = math.log(s[1])
                ts = s[11]
                somv = s[17]
                entry = (lp, ts, somv)
                key = (item_class, mg)
                if key not in self._modset_lookup:
                    self._modset_lookup[key] = []
                self._modset_lookup[key].append(entry)
                gkey = (item_class, mg, s[2])
                if gkey not in self._modset_grade_lookup:
                    self._modset_grade_lookup[gkey] = []
                self._modset_grade_lookup[gkey].append(entry)
        self._modset_lookup_dirty = False
        # Learn mod values and build core-mod lookup after modset lookup
        self._learn_mod_market_values()

    def _learn_mod_market_values(self):
        """Learn per-mod market value uplift from data.

        For each (item_class, mod_group), computes:
            median(log_price WITH mod) - median(log_price WITHOUT mod)
        Positive = price-driving mod, negative = cheap-item indicator.
        """
        self._mod_market_values.clear()
        MIN_SIDE = 5   # min samples on each side (with/without)
        MIN_CLASS = 30  # min samples in class

        for item_class, samples in self._by_class.items():
            if len(samples) < MIN_CLASS:
                continue

            # Collect all mod groups and log-prices in this class
            all_lps = []
            mod_to_lps = {}  # mod_group -> [log_prices of items WITH this mod]
            all_mods = set()

            for s in samples:
                lp = math.log(s[1])
                all_lps.append(lp)
                for g in s[9]:
                    all_mods.add(g)
                    if g not in mod_to_lps:
                        mod_to_lps[g] = []
                    mod_to_lps[g].append(lp)

            if not all_mods:
                continue

            all_lps.sort()
            n_total = len(all_lps)
            class_values = {}

            for mod, with_lps in mod_to_lps.items():
                n_with = len(with_lps)
                n_without = n_total - n_with
                if n_with < MIN_SIDE or n_without < MIN_SIDE:
                    continue

                with_lps_sorted = sorted(with_lps)
                median_with = with_lps_sorted[n_with // 2]

                # Compute median of items WITHOUT this mod
                without_lps = [math.log(s[1]) for s in samples
                               if mod not in s[9]]
                without_lps.sort()
                if len(without_lps) < MIN_SIDE:
                    continue
                median_without = without_lps[len(without_lps) // 2]
                class_values[mod] = round(median_with - median_without, 4)

            if class_values:
                self._mod_market_values[item_class] = class_values

        self._classify_core_mods()

    def _classify_core_mods(self):
        """Classify mods as core (price-driving) or filler per item class.

        Core mod = market_value uplift > CORE_MOD_THRESHOLD.
        """
        self._core_mods.clear()
        for item_class, values in self._mod_market_values.items():
            cores = frozenset(
                mod for mod, uplift in values.items()
                if uplift > self.CORE_MOD_THRESHOLD
            )
            if cores:
                self._core_mods[item_class] = cores

        # Build core-mod lookup table
        self._build_core_modset_lookup()

    def _build_core_modset_lookup(self):
        """Build core-mod lookup: (item_class, frozenset(core_mods+synergy)) -> entries.

        Uses _extract_core_key() to include synergy pair tokens, so build
        and query keys are consistent.
        """
        self._core_modset_lookup.clear()
        for item_class, samples in self._by_class.items():
            class_cores = self._core_mods.get(item_class)
            if not class_cores:
                continue
            for s in samples:
                mg = frozenset(s[9]) if s[9] else frozenset()
                if not mg:
                    continue
                core_key = self._extract_core_key(item_class, mg)
                if not core_key:
                    continue
                lp = math.log(s[1])
                ts = s[11]
                somv = s[17]
                entry = (lp, ts, somv)
                key = (item_class, core_key)
                if key not in self._core_modset_lookup:
                    self._core_modset_lookup[key] = []
                self._core_modset_lookup[key].append(entry)

    _MODSET_MIN_SAMPLES = 3  # need at least this many to trust the lookup

    @staticmethod
    def _modset_median(entries, tier_score: float = None,
                       tier_window: float = 1.0, min_samples: int = 3,
                       max_log_spread: float = 2.2,
                       somv_factor: float = None,
                       somv_window: float = 0.3):
        """Compute median log-price from [(log_price, tier_score, somv)] entries.

        When tier_score is provided, filters to entries within tier_window
        of the query tier_score. This differentiates T1 vs T7 rolls of the
        same mod set without discarding the entire pool.

        When somv_factor is provided, further filters to entries within
        somv_window of the query somv_factor (only if enough samples remain).

        After filtering, if the price spread is still > max_log_spread (~7.4x),
        falls through to k-NN which can use roll quality for finer pricing.
        """
        if not entries:
            return None
        if tier_score is not None:
            nearby = [e for e in entries
                      if abs(e[1] - tier_score) <= tier_window]
            if len(nearby) >= min_samples:
                entries = nearby
        # SOMV filtering: separate high-roll from low-roll items
        if somv_factor is not None:
            somv_nearby = [e for e in entries
                           if len(e) > 2 and abs(e[2] - somv_factor) <= somv_window]
            if len(somv_nearby) >= min_samples:
                entries = somv_nearby
        lps = sorted(e[0] for e in entries)
        # After filtering, check if price spread is still too wide.
        # Wide spread means factors beyond mod identity + tier (like roll
        # quality or somv) drive the price — let k-NN handle these.
        if lps[-1] - lps[0] > max_log_spread:
            return None
        est = math.exp(lps[len(lps) // 2])
        # Compute 25th/75th percentile range
        n = len(lps)
        low = math.exp(lps[max(0, n // 4)])
        high = math.exp(lps[min(n - 1, (3 * n) // 4)])
        def _rnd(v):
            if v >= 10:
                return round(v, 0)
            elif v >= 1:
                return round(v, 1)
            else:
                return round(v, 2)
        return (_rnd(est), _rnd(low), _rnd(high))

    def _extract_core_key(self, item_class: str,
                          mod_groups: frozenset) -> Optional[frozenset]:
        """Extract core-mod key for an item, with synergy pair tokens.

        Returns frozenset of core mod names + "SYN:ModA|ModB" tokens for
        synergy pairs where both mods are present. Returns None if no
        core mods match.
        """
        class_cores = self._core_mods.get(item_class)
        if not class_cores:
            return None
        item_cores = mod_groups & class_cores
        if not item_cores:
            return None

        # Add synergy tokens when both mods of a known pair are present
        from weight_learner import SYNERGY_PAIRS
        tokens = set(item_cores)
        for a, b in SYNERGY_PAIRS:
            if a in mod_groups and b in mod_groups:
                tokens.add(f"SYN:{a}|{b}")
        return frozenset(tokens)

    def _modset_estimate(self, item_class: str, mod_groups: frozenset,
                         grade_num: int = 1, tier_score: float = None,
                         somv_factor: float = None
                         ) -> Optional[float]:
        """Lookup estimate: median price of items with same or similar mod set.

        Priority:
        1. Grade-stratified exact match (≥2 samples)
        2. Unstratified exact match (≥3 samples)
        3. Fuzzy match: n-1 mod overlap (≥4 mods, ≥3 samples)
        4. Core-mod match: items sharing the same core (price-driving) mods (≥5 samples)
        Returns None if not enough data.

        When tier_score is provided, filters entries by tier proximity so
        T1-rolled items get different prices than T7-rolled items.
        When somv_factor is provided, filters by roll quality proximity.
        """
        if not mod_groups:
            return None
        if self._modset_lookup_dirty:
            self._build_modset_lookup()

        self._last_modset_match_type = ""

        # 1. Grade-stratified exact match (relaxed: 2 samples enough since
        #    grade filtering already narrows the pool)
        gkey = (item_class, mod_groups, grade_num)
        entries = self._modset_grade_lookup.get(gkey)
        if entries and len(entries) >= 2:
            result = self._modset_median(entries, tier_score=tier_score,
                                         somv_factor=somv_factor,
                                         min_samples=2)
            if result is not None:
                est, low, high = result
                self.last_estimate_low = low
                self.last_estimate_high = high
                self._last_modset_match_type = "exact_grade"
                return est

        # 2. Unstratified exact match
        key = (item_class, mod_groups)
        entries = self._modset_lookup.get(key)
        if entries and len(entries) >= self._MODSET_MIN_SAMPLES:
            result = self._modset_median(entries, tier_score=tier_score,
                                         somv_factor=somv_factor)
            if result is not None:
                est, low, high = result
                self.last_estimate_low = low
                self.last_estimate_high = high
                self._last_modset_match_type = "exact"
                return est

        # 3. Fuzzy match: try removing one mod at a time (n-1 overlap)
        if len(mod_groups) >= 4:
            best_entries = None
            best_count = 0
            for mod in mod_groups:
                subset = mod_groups - {mod}
                # Check existing keys that match this subset
                sub_key = (item_class, subset)
                sub_entries = self._modset_lookup.get(sub_key)
                if sub_entries and len(sub_entries) >= self._MODSET_MIN_SAMPLES:
                    if len(sub_entries) > best_count:
                        best_entries = sub_entries
                        best_count = len(sub_entries)
            if best_entries:
                result = self._modset_median(best_entries, tier_score=tier_score,
                                             somv_factor=somv_factor)
                if result is not None:
                    est, low, high = result
                    self.last_estimate_low = low
                    self.last_estimate_high = high
                    self._last_modset_match_type = "n_minus_1"
                    return est

        # 4. Core-mod match: match on price-driving mods only (filler stripped)
        class_cores = self._core_mods.get(item_class)
        if class_cores:
            core_key = self._extract_core_key(item_class, mod_groups)
            if core_key:
                # Try synergy-enhanced key first, then plain core key
                core_entries = self._core_modset_lookup.get(
                    (item_class, core_key))
                if core_entries and len(core_entries) >= 5:
                    result = self._modset_median(
                        core_entries, tier_score=tier_score,
                        somv_factor=somv_factor,
                        min_samples=5, max_log_spread=1.8)
                    if result is not None:
                        est, low, high = result
                        self.last_estimate_low = low
                        self.last_estimate_high = high
                        self._last_modset_match_type = "core_mod"
                        return est

        return None

    def _weighted_jaccard_distance(self, set_a: frozenset, set_b: frozenset,
                                    item_class: str = "") -> float:
        """Weighted Jaccard distance in [0.0, MOD_IDENTITY_WEIGHT].
        Returns 0.0 when either set is empty (neutral for legacy data).

        When learned mod market values are available for the item class,
        uses abs(market_value) as weight so price-driving mods contribute
        more to the distance metric.
        """
        if not set_a or not set_b:
            return 0.0
        union = set_a | set_b
        intersection = set_a & set_b
        if not union:
            return 0.0
        # Use learned market values when available, fall back to static weights
        market_vals = self._mod_market_values.get(item_class) if item_class else None
        if market_vals:
            def _weight(g):
                mv = market_vals.get(g)
                if mv is not None:
                    return max(abs(mv), 0.1)  # floor at 0.1 so filler mods aren't zero
                return self._mod_weights.get(g, 0.3)
        else:
            def _weight(g):
                return self._mod_weights.get(g, 0.3)
        w_union = sum(_weight(g) for g in union)
        w_inter = sum(_weight(g) for g in intersection)
        if w_union == 0:
            return 0.0
        return (1.0 - w_inter / w_union) * self.MOD_IDENTITY_WEIGHT

    STAT_WEIGHT = 1.5  # primary price signal for stat-based distance

    def _stat_distance(self, stats_a: tuple, stats_b: tuple) -> float:
        """Stat-id Jaccard + value distance. Returns [0, STAT_WEIGHT].

        Uses raw stat_ids from ModParser for 100% mod coverage.
        Falls back gracefully: returns 0.0 when either side has no stats
        (neutral for legacy data without mod_stats).
        """
        if not stats_a or not stats_b:
            return 0.0  # neutral for legacy data
        a_dict = dict(stats_a)
        b_dict = dict(stats_b)
        all_ids = set(a_dict) | set(b_dict)
        shared = set(a_dict) & set(b_dict)
        if not all_ids:
            return 0.0
        jaccard = 1.0 - len(shared) / len(all_ids)
        val_dist = 0.0
        if shared:
            for sid in shared:
                va, vb = abs(a_dict[sid]), abs(b_dict[sid])
                max_v = max(va, vb, 1.0)
                val_dist += abs(va - vb) / max_v
            val_dist /= len(shared)
        return (jaccard * 0.7 + val_dist * 0.3) * self.STAT_WEIGHT

    def _interpolate(self, score: float, samples: List[Sample],
                     grade_num: int = 1, dps_factor: float = 1.0,
                     defense_factor: float = 1.0,
                     top_tier_count: int = 0, mod_count: int = 4,
                     item_class: str = "",
                     mod_groups: frozenset = None,
                     base_type: str = "",
                     tier_score: float = 0.0,
                     coc_score: float = 0.0,
                     es_score: float = 0.0,
                     mana_score: float = 0.0,
                     mod_tiers: dict = None,
                     somv_factor: float = 1.0,
                     mod_rolls: dict = None,
                     pdps: float = 0.0,
                     edps: float = 0.0,
                     demand_score: float = 0.0,
                     mod_stats_tuple: tuple = None,
                     quality: int = 0,
                     sockets: int = 0,
                     corrupted: int = 0,
                     open_prefixes: int = 0,
                     open_suffixes: int = 0) -> Tuple[float, float]:
        """k-NN inverse-distance-weighted interpolation in log-price space.

        Distance includes mod composition (Jaccard), per-mod tier matching,
        roll quality, grade, base type, combat factors, DPS type, and archetype.

        1. Compute multi-dimensional distance
        2. Sort by distance, take k nearest
        3. Weight by 1 / (distance + 0.02), with recency bonus for user data
        4. Weighted average of log(divine), then exp() back
        5. Blend toward group median when neighbors are distant

        Returns (estimate, confidence) where confidence is 0.0-1.0.
        """
        if mod_groups is None:
            mod_groups = frozenset()
        k_base = self._k_for_pool(len(samples))
        now = time.time()
        query_tiers = mod_tiers or {}
        query_rolls = mod_rolls or {}

        # Pre-filter: prefer candidates sharing mods with query.
        # Graduated overlap: more mods → require more shared mods.
        # Falls back to full pool if too few matches.
        MIN_FILTERED = k_base * 3
        n_mods = len(mod_groups) if mod_groups else 0
        if n_mods >= 6:
            min_overlap = 3
        elif n_mods >= 4:
            min_overlap = 2
        else:
            min_overlap = 1  # 2-3 mod items: any shared mod is meaningful

        if mod_groups and n_mods >= 2:
            filtered = [s for s in samples
                        if len(mod_groups & frozenset(s[9])) >= min_overlap]
            candidates = filtered if len(filtered) >= MIN_FILTERED else samples
        else:
            candidates = samples

        def _dist(s: Sample) -> float:
            score_d = abs(s[0] - score) * self.SCORE_WEIGHT
            grade_d = abs(s[2] - grade_num) * self.GRADE_PENALTY
            dps_d = abs(s[3] - dps_factor) * self.DPS_WEIGHT
            def_d = abs(s[4] - defense_factor) * self.DEFENSE_WEIGHT
            ttc_d = abs(s[5] - top_tier_count) * self.TOP_TIER_WEIGHT
            mc_d = abs(s[6] - mod_count) * self.MOD_COUNT_WEIGHT
            s_mods = frozenset(s[9]) if s[9] else frozenset()
            # Use stat-based distance when both items have mod_stats (100% coverage)
            # Fall back to weighted Jaccard for legacy data without mod_stats
            s_stats = s[22] if len(s) > 22 else ()
            if mod_stats_tuple and s_stats:
                mod_d = self._stat_distance(mod_stats_tuple, s_stats)
            else:
                mod_d = self._weighted_jaccard_distance(mod_groups, s_mods, item_class)
            bt_d = 0.0
            if base_type and s[10] and base_type != s[10]:
                bt_d = self.BASE_TYPE_WEIGHT
            ts_d = abs(s[11] - tier_score) * self.TIER_SCORE_WEIGHT
            # Per-mod tier matching: penalise tier differences on shared mods
            tier_d = 0.0
            if query_tiers and s[13]:
                s_tiers = dict(s[13])  # (mod_group, tier) pairs
                shared = mod_groups & s_mods
                if shared:
                    tier_diff_sum = 0.0
                    n_shared = 0
                    for mod in shared:
                        qt = query_tiers.get(mod, 0)
                        st = s_tiers.get(mod, 0)
                        if qt > 0 and st > 0:
                            # Use 1/tier difference: T1(1.0) vs T7(0.14) = 0.86
                            tier_diff_sum += abs(1.0 / qt - 1.0 / st)
                            n_shared += 1
                    if n_shared > 0:
                        tier_d = (tier_diff_sum / n_shared) * self.MOD_TIER_WEIGHT
            # Archetype distances (positions 14, 15, 16)
            arch_d = (abs(s[14] - coc_score) + abs(s[15] - es_score)
                      + abs(s[16] - mana_score)) * self.ARCHETYPE_WEIGHT

            # somv_factor distance (avg roll quality): items with higher avg rolls cost more
            somv_d = abs(s[17] - somv_factor) * self.SOMV_WEIGHT

            # Per-mod roll quality distance: penalise roll quality differences on shared mods
            rq_d = 0.0
            if query_rolls and s[18]:
                s_rolls = dict(s[18])
                shared_rq = mod_groups & s_mods
                if shared_rq:
                    rq_diff_sum = 0.0
                    n_rq = 0
                    for mod in shared_rq:
                        qr = query_rolls.get(mod, -1)
                        sr = s_rolls.get(mod, -1)
                        if qr >= 0 and sr >= 0:
                            rq_diff_sum += abs(qr - sr)
                            n_rq += 1
                    if n_rq > 0:
                        rq_d = (rq_diff_sum / n_rq) * self.ROLL_QUALITY_WEIGHT

            # DPS type distance: phys-heavy vs ele-heavy weapons
            dps_type_d = 0.0
            if (pdps > 0 or edps > 0) and (s[19] > 0 or s[20] > 0):
                q_total = pdps + edps + 0.01
                s_total = s[19] + s[20] + 0.01
                q_phys_ratio = pdps / q_total
                s_phys_ratio = s[19] / s_total
                dps_type_d = abs(q_phys_ratio - s_phys_ratio) * self.DPS_TYPE_WEIGHT

            # Demand distance: meta-relevant items should match similar demand
            demand_d = 0.0
            if demand_score > 0 and self._demand_index and item_class:
                s_demand = self._demand_index.get_demand_score(
                    item_class, list(s_mods))
                demand_d = abs(demand_score - s_demand) * self.DEMAND_WEIGHT

            # Quality distance
            quality_d = 0.0
            if len(s) > 23 and (quality > 0 or s[23] > 0):
                quality_d = abs(quality - s[23]) / 20.0 * self.QUALITY_WEIGHT

            # Socket distance
            socket_d = 0.0
            if len(s) > 24 and (sockets > 0 or s[24] > 0):
                if sockets != s[24]:
                    socket_d = self.SOCKET_WEIGHT

            # Open affix distance
            affix_d = 0.0
            if len(s) > 27:
                s_op = s[26]
                s_os = s[27]
                if open_prefixes > 0 or s_op > 0 or open_suffixes > 0 or s_os > 0:
                    affix_d = (abs(open_prefixes - s_op) + abs(open_suffixes - s_os)
                               ) / 6.0 * self.OPEN_AFFIX_WEIGHT

            return (score_d + grade_d + ttc_d + mc_d + dps_d + def_d + mod_d
                    + bt_d + ts_d + tier_d + arch_d + somv_d + rq_d
                    + dps_type_d + demand_d + quality_d + socket_d + affix_d)

        by_dist = sorted(candidates, key=_dist)

        # Per-query adaptive k: tight neighborhoods use fewer neighbors (more
        # precise), sparse neighborhoods use more (more smoothing).
        # Check distance to the base-k-th neighbor to decide.
        if len(by_dist) > k_base:
            kth_dist = _dist(by_dist[k_base - 1])
            if kth_dist < 0.5:
                # Very tight neighborhood — use fewer neighbors for precision
                k = max(3, k_base - 2)
            elif kth_dist > 2.0:
                # Sparse neighborhood — use more neighbors, blend toward median
                k = min(len(by_dist), k_base + 5)
            else:
                k = k_base
        else:
            k = min(k_base, len(by_dist))

        neighbors = by_dist[:k]

        dist_sum = 0.0
        wt_pairs = []  # [(log_price, weight), ...]

        for s in neighbors:
            s_divine = s[1]
            s_ts = s[7]
            s_user = s[8]
            s_sale_conf = s[21]  # sale_confidence: 3.0=sold, 1.0=unknown, 0.3=stale
            dist = _dist(s)
            dist_sum += dist
            w = 1.0 / (dist + self._EPSILON)

            # Sale confidence: confirmed sales get 3x weight, stale listings get 0.3x
            w *= s_sale_conf

            # Recency bonus: recent user data gets extra weight
            if s_user and s_ts > 0:
                age = now - s_ts
                if age < self._RECENCY_WINDOW:
                    w *= self._RECENCY_MULTIPLIER

            # Temporal decay: all samples with timestamps get age-based decay
            # Half-life of 14 days, floor at 25% weight
            if s_ts > 0:
                age = now - s_ts
                decay = 0.5 ** (age / (14 * 86400))
                w *= max(decay, 0.25)

            wt_pairs.append((math.log(s_divine), w))

        # Weighted quantile in log-price space: more robust to distribution
        # skew than weighted mean (resists pull toward 1-5d density).
        # TSM insight: listings overstate true sale prices. Use slightly lower
        # quantile to counteract this bias. Keep S-grade at 0.50.
        _GRADE_QUANTILE = {4: 0.50, 3: 0.48, 2: 0.47, 1: 0.45, 0: 0.40}
        quantile = _GRADE_QUANTILE.get(grade_num, 0.47)

        wt_pairs.sort(key=lambda x: x[0])
        total_weight = sum(w for _, w in wt_pairs)
        target = total_weight * quantile
        cumulative = 0.0
        avg_log = wt_pairs[-1][0]  # fallback
        for lp, w in wt_pairs:
            cumulative += w
            if cumulative >= target:
                avg_log = lp
                break

        # Compute 25th/75th percentile range from neighbors
        log_low = self._weighted_percentile(wt_pairs, 0.25)
        log_high = self._weighted_percentile(wt_pairs, 0.75)
        self.last_estimate_low = math.exp(log_low)
        self.last_estimate_high = math.exp(log_high)

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

        # Compute confidence: inverse of mean k-NN distance, 0-1 scale
        confidence = 1.0 / (1.0 + mean_dist)

        # Cap wildly extrapolated estimates — with few samples the
        # log-space interpolation can produce absurd values.
        max_observed = max((s[1] for s in samples), default=100.0)
        result = min(result, max_observed * 2.0, 1500.0)

        # Round to reasonable precision
        if result >= 10:
            return round(result, 0), confidence
        elif result >= 1:
            return round(result, 1), confidence
        else:
            return round(result, 2), confidence
