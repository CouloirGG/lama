"""
LAMA - Trade API Client
Queries the POE2 trade API for rare item pricing based on mod filters.

Pipeline:
1. Build query from base_type + matched mods (min = 80% of actual value)
2. POST /api/trade2/search/poe2/{league} → query ID + result IDs
3. GET /api/trade2/fetch/{ids}?query={query_id} → listings with prices
4. Normalize to divine, return price range from lowest N listings

Rate limiting: max 2 req/sec with thread-safe lock.
Caching: in-memory cache keyed by (base_type + sorted stat_ids + rounded values), 5min TTL.
"""

import time
import logging
import threading
import hashlib
from dataclasses import dataclass
from typing import List, Optional

import requests

from config import (
    TRADE_API_BASE,
    TRADE_MAX_REQUESTS_PER_SECOND,
    TRADE_RESULT_COUNT,
    TRADE_CACHE_TTL,
    TRADE_MOD_MIN_MULTIPLIER,
    DEFAULT_LEAGUE,
    DPS_ITEM_CLASSES,
    DEFENSE_ITEM_CLASSES,
    TRADE_DPS_FILTER_MULT,
    TRADE_DEFENSE_FILTER_MULT,
)
from mod_parser import ParsedMod

logger = logging.getLogger(__name__)


@dataclass
class RarePriceResult:
    min_price: float     # lowest listing (divine)
    max_price: float     # Nth lowest listing (divine)
    num_results: int     # total results found
    display: str         # "~5-8 Divine"
    tier: str            # "high"/"good"/"decent"/"low"
    estimate: bool = False  # True when no exact match — conservative lower bound


class TradeClient:
    """
    Queries the POE2 trade API to price rare items based on their mods.

    Usage:
        tc = TradeClient(league="Fate of the Vaal",
                         divine_to_chaos_fn=..., divine_to_exalted_fn=...)
        result = tc.price_rare_item(item, parsed_mods)
    """

    def __init__(self, league: str = DEFAULT_LEAGUE,
                 divine_to_chaos_fn=None, divine_to_exalted_fn=None):
        self.league = league
        self._divine_to_chaos_fn = divine_to_chaos_fn or (lambda: 68.0)
        self._divine_to_exalted_fn = divine_to_exalted_fn or (lambda: 300.0)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "LAMA/1.0"})

        # Rate limiting
        self._last_request_time = 0.0
        self._rate_lock = threading.Lock()
        self._min_interval = 1.0 / TRADE_MAX_REQUESTS_PER_SECOND
        # When the trade API returns 429, don't retry until this time
        self._rate_limited_until = 0.0

        # Adaptive rate limiting — learned from API response headers
        self._rl_rules = []          # [(max_hits, window_secs, penalty_secs), ...]
        self._rl_state = []          # [(current_hits, window_secs, penalty_remaining), ...]
        self._rl_rules_logged = False

        # In-memory cache: fingerprint → (result, timestamp)
        self._cache: dict = {}
        self._cache_lock = threading.Lock()

    def price_rare_item(self, item, mods: List[ParsedMod],
                        is_stale=None) -> Optional[RarePriceResult]:
        """
        Price a rare item by querying the trade API.

        Args:
            item: ParsedItem with base_type and quality
            mods: List of ParsedMod from ModParser
            is_stale: Optional callback that returns True if this query has been
                superseded by a newer detection (abort early to save API calls)

        Returns:
            RarePriceResult or None if pricing fails
        """
        if not mods:
            logger.debug("TradeClient: no mods to query")
            return None

        base_type = item.base_type
        if not base_type:
            logger.debug("TradeClient: no base_type")
            return None

        quality = getattr(item, "quality", 0) or 0
        sockets = getattr(item, "sockets", 0) or 0

        # Items with fractured/desecrated mods or 3+ sockets are likely
        # valuable but may be too niche for base-type-specific results.
        has_value_signals = (
            sockets >= 3
            or any(m.mod_type in ("fractured", "desecrated") for m in mods)
        )

        # Check cache (includes quality, sockets, DPS, defense in fingerprint)
        total_dps = getattr(item, 'total_dps', 0.0) or 0.0
        total_defense = getattr(item, 'total_defense', 0) or 0
        fingerprint = self._make_fingerprint(
            base_type, mods, quality, sockets,
            dps=total_dps, defense=total_defense)
        cached = self._check_cache(fingerprint)
        if cached:
            logger.debug(f"TradeClient: cache hit for {base_type}")
            return cached

        try:
            # Bail early if we're in a rate-limit cooldown
            if self._is_rate_limited():
                wait = int(self._rate_limited_until - time.time())
                logger.info(f"TradeClient: rate limited, {wait}s remaining")
                return RarePriceResult(
                    min_price=0, max_price=0, num_results=0,
                    display=f"Rate limited ({wait}s)",
                    tier="low",
                )

            # Build stat filters once, then try progressively looser queries
            # Include explicit, implicit, fractured, and desecrated mods.
            # Skip rune (socketed), enchant (changeable), and crafted (bench).
            _PRICEABLE_TYPES = ("explicit", "implicit", "fractured", "desecrated")
            priceable = [m for m in mods if m.mod_type in _PRICEABLE_TYPES]
            stat_filters = self._build_stat_filters(priceable)

            # Log what we're querying for diagnostics
            skipped = [m for m in mods if m.mod_type not in _PRICEABLE_TYPES]
            if skipped:
                logger.info(
                    f"TradeClient: skipped {len(skipped)} non-priceable: "
                    + ", ".join(f"[{m.mod_type}] {m.raw_text}" for m in skipped))
            for m in priceable:
                logger.info(
                    f"TradeClient: mod [{m.mod_type}] {m.raw_text} "
                    f"→ {m.stat_id} (val={m.value})")

            # Compute DPS/defense filters for trade API
            item_class = getattr(item, 'item_class', '') or ''
            dps_min = 0
            defense_mins = {}
            if item_class in DPS_ITEM_CLASSES:
                tdps = getattr(item, 'total_dps', 0.0) or 0.0
                if tdps > 0:
                    dps_min = tdps * TRADE_DPS_FILTER_MULT
            if item_class in DEFENSE_ITEM_CLASSES:
                for attr, key in [('armour', 'ar'), ('evasion', 'ev'), ('energy_shield', 'es')]:
                    val = getattr(item, attr, 0) or 0
                    if val > 0:
                        defense_mins[key] = int(val * TRADE_DEFENSE_FILTER_MULT)

            if has_value_signals:
                # Niche items (fractured/desecrated/3S): quick probe with
                # specific base type (1 call), then broad count(n-1) without
                # base type (up to 3 calls).  Total max 4 — stays well under
                # the API's rate limit window.
                search_result, exact_match, mods_dropped = self._search_progressive(
                    base_type, stat_filters, priceable, quality=quality,
                    sockets=sockets, is_stale=is_stale, max_calls=1,
                    dps_min=dps_min, defense_mins=defense_mins,
                )

                if not search_result:
                    if is_stale and is_stale():
                        return None
                    # Broad search: count queries without base type.
                    # Skip count(n/n) and hybrid — too tight for niche items.
                    # Interleave count levels: dropping a mod is more
                    # productive than loosening values for the same count.
                    n = len(stat_filters)
                    floor = max(2, n - 2)
                    broad_specs = [
                        (n - 1, 0.85),   # drop 1 mod, medium values
                        (floor, 0.85),   # drop 2 mods, medium values
                        (floor, 0.80),   # drop 2 mods, loose values
                    ]
                    logger.info(
                        f"TradeClient: broad search (no base type) "
                        f"for {base_type}")
                    for min_count, mult in broad_specs:
                        if (is_stale and is_stale()) or self._is_rate_limited():
                            break
                        filters_at = self._build_stat_filters_custom(
                            priceable, mult)
                        query = self._build_query(
                            None, filters_at, "count", min_count,
                            quality=quality, sockets=sockets,
                            dps_min=dps_min, defense_mins=defense_mins)
                        result = self._do_search(query)
                        pct = int(mult * 100)
                        if result and result.get("result"):
                            total = result.get("total", 0)
                            logger.info(
                                f"TradeClient: broad count({min_count}/{n}) "
                                f"@{pct}% = {total} results")
                            search_result = result
                            exact_match = False
                            mods_dropped = n - min_count
                            break
                        logger.info(
                            f"TradeClient: broad count({min_count}/{n}) "
                            f"@{pct}% = 0 results")
            else:
                # Normal items: full progressive search + socket fallback.
                search_result, exact_match, mods_dropped = self._search_progressive(
                    base_type, stat_filters, priceable, quality=quality,
                    sockets=sockets, is_stale=is_stale,
                    dps_min=dps_min, defense_mins=defense_mins,
                )

                if not search_result and sockets > 0:
                    if is_stale and is_stale():
                        return None
                    if self._is_rate_limited():
                        logger.info(
                            "TradeClient: skipping socket retry — rate limited")
                        return None
                    logger.info(
                        f"TradeClient: no results with {sockets} sockets, "
                        f"retrying without socket filter")
                    search_result, exact_match, mods_dropped = self._search_progressive(
                        base_type, stat_filters, priceable, quality=quality,
                        sockets=0, is_stale=is_stale, max_calls=3,
                        dps_min=dps_min, defense_mins=defense_mins,
                    )

            if not search_result:
                if has_value_signals:
                    logger.info(
                        f"TradeClient: no results for {base_type}, "
                        f"flagging as probably valuable "
                        f"(fractured/desecrated/3S)")
                    return RarePriceResult(
                        min_price=0, max_price=0, num_results=0,
                        display="+", tier="good", estimate=True,
                    )
                return None

            # Abort if a newer item detection superseded us
            if is_stale and is_stale():
                logger.debug(f"TradeClient: query stale after search, aborting")
                return None

            query_id = search_result.get("id")
            result_ids = search_result.get("result", [])
            total = search_result.get("total", 0)

            if not query_id or not result_ids:
                logger.debug(f"TradeClient: no results for {base_type}")
                return None

            # Fetch the first N listings
            fetch_ids = result_ids[:TRADE_RESULT_COUNT]
            listings = self._do_fetch(query_id, fetch_ids)
            if not listings:
                return None

            # Extract prices and build result.
            # When 1 mod is dropped, comparables are slightly worse than
            # the actual item — use upper prices as a floor estimate.
            # When 2+ mods are dropped, comparables may be much BETTER
            # (they have their own amazing mods in the dropped slots),
            # so don't inflate — use conservative regular pricing.
            is_estimate = not exact_match
            if mods_dropped >= 2:
                # Many mods dropped — comparables aren't representative.
                # With few results, they're extreme outliers (god-tier items
                # that happen to share some mods). Suppress entirely.
                if total < 8:
                    logger.info(
                        f"TradeClient: suppressing estimate — {mods_dropped} "
                        f"mods dropped, only {total} results (not representative)")
                    return None
                # With enough results, use regular pricing (no inflation)
                logger.info(
                    f"TradeClient: {mods_dropped} mods dropped — using "
                    f"conservative pricing (no lower_bound inflation)")
                result = self._build_result(
                    listings, total, lower_bound=False)
            else:
                # Low result count filter: estimates from 1-2 listings
                # are almost always price-fixers (Bramble Spiral at 900 div
                # from 1 listing, etc.).  Suppress these entirely.
                if is_estimate and total <= 2:
                    logger.info(
                        f"TradeClient: suppressing low-count estimate "
                        f"({total} result(s), {mods_dropped} mod(s) dropped)"
                        f" -- too few listings for reliable pricing")
                    return None
                # Exact matches with 1-2 results: force estimate mode —
                # a single listing could still be a price-fixer
                if not is_estimate and total <= 2:
                    logger.info(
                        f"TradeClient: only {total} exact match(es) "
                        f"-- downgrading to estimate")
                    is_estimate = True
                result = self._build_result(
                    listings, total, lower_bound=is_estimate)
            if is_estimate and result:
                logger.info(
                    f"TradeClient: estimate from similar items "
                    f"(dropped {mods_dropped} mod(s))")
            if result:
                self._put_cache(fingerprint, result)
                logger.info(
                    f"RARE PRICE {base_type}: {result.display} "
                    f"({result.num_results} results)"
                )

            return result

        except Exception as e:
            logger.warning(f"TradeClient: pricing failed for {base_type}: {e}")
            return RarePriceResult(
                min_price=0, max_price=0, num_results=0,
                display="?", tier="low",
            )

    def price_base_item(self, item, is_stale=None) -> Optional[RarePriceResult]:
        """
        Price a normal/magic item by base type + sockets via the trade API.
        Used for items where the value comes from the base itself (e.g., 3-socket bases).
        """
        base_type = item.base_type or item.name
        if not base_type:
            return None

        sockets = getattr(item, "sockets", 0) or 0
        item_level = getattr(item, "item_level", 0) or 0

        # Cache key: base + sockets + ilvl
        cache_key = f"base:{base_type}:{sockets}:{item_level}"
        cached = self._check_cache(cache_key)
        if cached:
            logger.debug(f"TradeClient: cache hit for base {base_type}")
            return cached

        try:
            if self._is_rate_limited():
                wait = int(self._rate_limited_until - time.time())
                return RarePriceResult(
                    min_price=0, max_price=0, num_results=0,
                    display=f"Rate limited ({wait}s)", tier="low",
                )

            logger.info(
                f"TradeClient: pricing base {base_type} "
                f"(sockets={sockets}, ilvl={item_level})"
            )

            # Build a simple query: base type + socket count + ilvl
            query = self._build_base_query(base_type, sockets, item_level)

            if is_stale and is_stale():
                return None

            search_result = self._do_search(query)
            if not search_result:
                return None

            query_id = search_result.get("id")
            result_ids = search_result.get("result", [])
            total = search_result.get("total", 0)

            if not query_id or not result_ids:
                return None

            if is_stale and is_stale():
                return None

            fetch_ids = result_ids[:TRADE_RESULT_COUNT]
            listings = self._do_fetch(query_id, fetch_ids)
            if not listings:
                return None

            result = self._build_result(listings, total)
            if result:
                self._put_cache(cache_key, result)
                logger.info(
                    f"BASE PRICE {base_type}: {result.display} "
                    f"({result.num_results} results)"
                )
            return result

        except Exception as e:
            logger.warning(f"TradeClient: base pricing failed for {base_type}: {e}")
            return RarePriceResult(
                min_price=0, max_price=0, num_results=0,
                display="?", tier="low",
            )

    def price_unique_item(self, item, is_stale=None) -> Optional[RarePriceResult]:
        """Price a corrupted unique item by name + sockets via the trade API.

        Used for corrupted uniques where Vaal outcomes (rerolled mods, added
        sockets) significantly affect value compared to the static base price.
        """
        name = item.name
        base_type = item.base_type or ""
        sockets = getattr(item, "sockets", 0) or 0

        if not name:
            return None

        cache_key = f"unique:{name}:{sockets}"
        cached = self._check_cache(cache_key)
        if cached:
            logger.debug(f"TradeClient: cache hit for unique {name}")
            return cached

        try:
            if self._is_rate_limited():
                wait = int(self._rate_limited_until - time.time())
                return RarePriceResult(
                    min_price=0, max_price=0, num_results=0,
                    display=f"Rate limited ({wait}s)", tier="low",
                )

            logger.info(
                f"TradeClient: pricing unique {name} "
                f"(corrupted, sockets={sockets})"
            )

            query = self._build_unique_query(name, base_type, sockets)

            if is_stale and is_stale():
                return None

            search_result = self._do_search(query)
            if not search_result:
                return None

            query_id = search_result.get("id")
            result_ids = search_result.get("result", [])
            total = search_result.get("total", 0)

            if not query_id or not result_ids:
                return None

            if is_stale and is_stale():
                return None

            fetch_ids = result_ids[:TRADE_RESULT_COUNT]
            listings = self._do_fetch(query_id, fetch_ids)
            if not listings:
                return None

            result = self._build_result(listings, total)
            if result:
                self._put_cache(cache_key, result)
                logger.info(
                    f"UNIQUE PRICE {name} ({sockets}S corrupted): "
                    f"{result.display} ({result.num_results} results)"
                )
            return result

        except Exception as e:
            logger.warning(f"TradeClient: unique pricing failed for {name}: {e}")
            return RarePriceResult(
                min_price=0, max_price=0, num_results=0,
                display="?", tier="low",
            )

    def _build_unique_query(self, name: str, base_type: str,
                            sockets: int = 0) -> dict:
        """Build a trade API query for a corrupted unique by name + sockets."""
        filters = {
            "type_filters": {
                "filters": {
                    "rarity": {"option": "unique"},
                }
            },
            "misc_filters": {
                "filters": {
                    "corrupted": {"option": "true"},
                },
            },
        }

        if sockets > 0:
            filters["equipment_filters"] = {
                "filters": {
                    "rune_sockets": {"min": sockets},
                },
            }

        query_inner = {
            "status": {"option": "any"},
            "name": name,
            "stats": [{"type": "and", "filters": []}],
            "filters": filters,
        }
        if base_type:
            query_inner["type"] = base_type

        return {"query": query_inner, "sort": {"price": "asc"}}

    def _build_base_query(self, base_type: str, sockets: int = 0,
                          item_level: int = 0) -> dict:
        """Build a trade API query for a base item (no mods, just type + sockets + ilvl)."""
        misc = {
            "mirrored": {"option": "false"},
        }
        if item_level > 0:
            misc["ilvl"] = {"min": item_level}

        filters = {
            "type_filters": {
                "filters": {
                    "rarity": {"option": "nonunique"},
                }
            },
            "misc_filters": {
                "filters": misc,
            },
        }

        if sockets > 0:
            # POE2 uses "rune_sockets" (Augmentable Sockets), not "sockets"
            filters["equipment_filters"] = {
                "filters": {
                    "rune_sockets": {"min": sockets},
                },
            }

        return {
            "query": {
                "status": {"option": "any"},
                "type": base_type,
                "stats": [{"type": "and", "filters": []}],
                "filters": filters,
            },
            "sort": {"price": "asc"},
        }

    # ─── Query Building ───────────────────────────

    def _build_stat_filters(self, priceable: List[ParsedMod]) -> list:
        """Build stat filter dicts from priceable mods."""
        stat_filters = []
        for mod in priceable:
            if mod.value >= 0:
                min_val = self._compute_min_value(mod.value)
                stat_filters.append({"id": mod.stat_id, "value": {"min": min_val}})
            else:
                max_val = self._compute_min_value(mod.value)
                stat_filters.append({"id": mod.stat_id, "value": {"max": max_val}})
        return stat_filters

    @staticmethod
    def _build_stat_filters_custom(mods: List[ParsedMod], multiplier: float) -> list:
        """Build stat filters with a custom value multiplier."""
        stat_filters = []
        for mod in mods:
            val = mod.value * multiplier
            rounded = int(val) if val == int(val) else round(val, 1)
            if mod.value >= 0:
                stat_filters.append({"id": mod.stat_id, "value": {"min": rounded}})
            else:
                stat_filters.append({"id": mod.stat_id, "value": {"max": rounded}})
        return stat_filters

    def _build_stat_filters_relaxed(self, priceable: List[ParsedMod]) -> list:
        """Build stat filters with much looser value minimums (60%).

        Used when normal minimums return 0 results with all mods required.
        Keeps ALL mods in the query (preserving item identity) while accepting
        items with lower rolls — better than dropping mods entirely.
        """
        _RELAXED_MULTIPLIER = 0.6
        stat_filters = []
        for mod in priceable:
            val = mod.value * _RELAXED_MULTIPLIER
            rounded = int(val) if val == int(val) else round(val, 1)
            if mod.value >= 0:
                stat_filters.append({"id": mod.stat_id, "value": {"min": rounded}})
            else:
                stat_filters.append({"id": mod.stat_id, "value": {"max": rounded}})
        return stat_filters

    @staticmethod
    def _compute_min_value(value: float) -> float:
        """Compute minimum filter value with value-dependent tightness.

        Low values (1-10) use tight matching because they represent discrete
        tiers where each point matters enormously (e.g., +6 vs +7 skills).
        Higher values use progressively looser matching.

        For negative values (e.g. "reduced" mods), uses the absolute value
        for tier selection, then applies to the original sign.
        """
        abs_val = abs(value)
        if abs_val <= 10:
            multiplier = 0.95
        elif abs_val <= 50:
            multiplier = 0.90
        else:
            multiplier = TRADE_MOD_MIN_MULTIPLIER  # 0.8
        result = value * multiplier
        return int(result) if result == int(result) else round(result, 1)

    def _build_query(self, base_type: str, stat_filters: list,
                     match_mode: str = "and", min_count: int = 0,
                     quality: int = 0, sockets: int = 0,
                     dps_min: float = 0, defense_mins: dict = None) -> dict:
        """Build a trade API search query.

        Args:
            match_mode: "and" (all must match) or "count" (at least min_count)
            quality: item quality — included as a min filter when > 0
            sockets: minimum rune sockets — included when > 0
            dps_min: minimum total DPS — included when > 0
            defense_mins: dict of defense minimums (ar/ev/es) — included when > 0
        """
        if match_mode == "count" and min_count > 0:
            stat_group = {"type": "count", "value": {"min": min_count},
                          "filters": stat_filters}
        else:
            stat_group = {"type": "and", "filters": stat_filters}

        misc_filters = {
            "mirrored": {"option": "false"},
        }
        if quality > 0:
            misc_filters["quality"] = {"min": quality}

        filters = {
            "type_filters": {
                "filters": {
                    "rarity": {"option": "nonunique"},
                }
            },
            "misc_filters": {
                "filters": misc_filters,
            },
        }

        # Equipment filters: sockets + DPS + defense
        equip_filters = {}
        if sockets > 0:
            equip_filters["rune_sockets"] = {"min": sockets}
        if dps_min > 0:
            equip_filters["dps"] = {"min": int(dps_min)}
        if defense_mins:
            for key in ("ar", "ev", "es"):
                if defense_mins.get(key, 0) > 0:
                    equip_filters[key] = {"min": defense_mins[key]}
        if equip_filters:
            filters["equipment_filters"] = {"filters": equip_filters}

        query_inner = {
            "status": {"option": "any"},
            "stats": [stat_group],
            "filters": filters,
        }
        if base_type:
            query_inner["type"] = base_type
        return {"query": query_inner, "sort": {"price": "asc"}}

    def _build_hybrid_query(self, base_type, key_filters: list,
                            common_filters: list, min_common: int,
                            quality: int = 0, sockets: int = 0,
                            dps_min: float = 0, defense_mins: dict = None) -> dict:
        """Build a query with key mods as "and" + common mods as "count".

        This ensures price-driving mods always match while allowing
        some common mods to be absent.
        """
        stat_groups = []

        if key_filters:
            stat_groups.append({"type": "and", "filters": key_filters})

        if common_filters and min_common > 0:
            stat_groups.append({
                "type": "count",
                "value": {"min": min_common},
                "filters": common_filters,
            })

        misc_filters = {
            "mirrored": {"option": "false"},
        }
        if quality > 0:
            misc_filters["quality"] = {"min": quality}

        filters = {
            "type_filters": {
                "filters": {
                    "rarity": {"option": "nonunique"},
                }
            },
            "misc_filters": {
                "filters": misc_filters,
            },
        }

        # Equipment filters: sockets + DPS + defense
        equip_filters = {}
        if sockets > 0:
            equip_filters["rune_sockets"] = {"min": sockets}
        if dps_min > 0:
            equip_filters["dps"] = {"min": int(dps_min)}
        if defense_mins:
            for key in ("ar", "ev", "es"):
                if defense_mins.get(key, 0) > 0:
                    equip_filters[key] = {"min": defense_mins[key]}
        if equip_filters:
            filters["equipment_filters"] = {"filters": equip_filters}

        query_inner = {
            "status": {"option": "any"},
            "stats": stat_groups,
            "filters": filters,
        }
        if base_type:
            query_inner["type"] = base_type
        return {"query": query_inner, "sort": {"price": "asc"}}

    # Patterns for common "filler" mods that rarely drive item price.
    # Mods NOT matching any of these patterns are considered "key" mods.
    _COMMON_MOD_PATTERNS = (
        # Defenses / mana (life and all-res removed — they drive value on jewelry)
        "maximum mana", "maximum energy shield",
        "mana regeneration", "life regeneration", "energy shield recharge",
        "to armour", "to evasion", "to energy shield",
        "increased armour", "increased evasion", "increased energy shield",
        "increased armour and evasion", "increased armour and energy shield",
        "increased evasion and energy shield",
        # Resistances / attributes
        "to fire resistance", "to cold resistance", "to lightning resistance",
        "to chaos resistance",
        "to strength", "to dexterity", "to intelligence", "to all attributes",
        # Flask mods (generally low value)
        "flask charges", "flask effect", "flask duration",
        "reduced flask", "increased flask",
        # Thorns / reflect (generally worthless)
        "thorns damage", "damage taken on block",
        # Accuracy / leech
        "to accuracy", "accuracy rating",
        "leeches", "leech",
        # Flat added damage (filler at low rolls; high rolls come with crit/% damage)
        "damage to attacks", "damage to spells",
        # Ailment / status duration
        "freeze duration", "chill effect", "ignite duration",
        "shock effect", "poison duration", "bleed duration",
        "curse effect", "ailment",
        # Misc filler
        "item rarity", "rarity of items", "light radius", "stun ", "knockback",
        "mana on kill", "life on kill", "mana cost",
        "reduced attribute requirements",
        "reduced projectile range",
        "effect of socketed",
    )

    def _classify_filters(self, priceable: List[ParsedMod], stat_filters: list):
        """Split stat filters into key (price-driving) and common (filler).

        Returns:
            (key_mods, common_mods, key_filters, common_filters)
            key_mods/common_mods: ParsedMod lists (for rebuilding with different multipliers)
            key_filters/common_filters: pre-built stat filter dicts (standard multiplier)
        """
        key_mods = []
        common_mods = []
        key_filters = []
        common_filters = []
        for mod, sf in zip(priceable, stat_filters):
            text_lower = mod.raw_text.lower()
            # Implicit mods are inherent to the base type — they don't
            # differentiate value and should never be key mods
            is_common = (
                mod.mod_type == "implicit"
                or any(pat in text_lower for pat in self._COMMON_MOD_PATTERNS)
            )
            if is_common:
                common_mods.append(mod)
                common_filters.append(sf)
            else:
                key_mods.append(mod)
                key_filters.append(sf)
        return key_mods, common_mods, key_filters, common_filters

    # Queries returning more results than this are considered too loose —
    # the price will reflect generic items rather than this specific item.
    _TOO_MANY_RESULTS = 50

    def _search_progressive(self, base_type: str, stat_filters: list,
                            priceable: List[ParsedMod] = None,
                            quality: int = 0, sockets: int = 0,
                            is_stale=None, max_calls: int = 6,
                            dps_min: float = 0, defense_mins: dict = None):
        """Find the best price query using minimal API calls.

        Args:
            max_calls: Maximum number of _do_search invocations allowed.
                Prevents niche items from burning through the rate limit.
                Default 6 covers the most productive search steps.
            dps_min: minimum total DPS filter
            defense_mins: dict of defense minimums (ar/ev/es)

        Returns:
            (search_result, exact_match, mods_dropped) tuple.
            exact_match is True when ALL mods matched (count n/n or "and"),
            False when mods were dropped (comparables are worse than actual item).
            mods_dropped is the number of mods removed from the query.
            Returns (None, False, 0) when no results found.
        """
        q = quality
        s = sockets
        dm = dps_min
        df = defense_mins
        calls_remaining = max_calls

        def _budget_search(query):
            """Call _do_search if budget remains, decrement counter."""
            nonlocal calls_remaining
            if calls_remaining <= 0:
                logger.info("TradeClient: search budget exhausted")
                return None
            calls_remaining -= 1
            return self._do_search(query)

        def _stale():
            if is_stale and is_stale():
                logger.debug("TradeClient: search aborted (stale)")
                return True
            return False

        n = len(stat_filters)

        # Small mod count — just try exact "and" match
        if n <= 4:
            # For 1-2 mod items (typically magic), rolls drive value —
            # try tight match first so we compare against similar quality.
            if n <= 2 and priceable:
                tight = self._build_stat_filters_custom(priceable, 0.98)
                query = self._build_query(base_type, tight, quality=q, sockets=s,
                                          dps_min=dm, defense_mins=df)
                result = _budget_search(query)
                if result and result.get("result"):
                    logger.info(
                        f"TradeClient: tight match @98% "
                        f"({result.get('total', 0)} results)")
                    return result, True, 0
                if _stale():
                    return None, False, 0

            query = self._build_query(base_type, stat_filters, quality=q, sockets=s,
                                      dps_min=dm, defense_mins=df)
            result = _budget_search(query)
            if result and result.get("result"):
                logger.info(
                    f"TradeClient: exact match ({result.get('total', 0)} results)")
                return result, True, 0
            if _stale():
                return None, False, 0

            # For socketed items: try key mods + sockets (drop common mods).
            # E.g., 2-socket boots with 35% MS — the sockets + key mod drive value,
            # common mods like fire res are just bonus.
            if s > 0 and priceable and len(priceable) == n and n >= 2:
                key_m, common_m, key_f, common_f = self._classify_filters(
                    priceable, stat_filters)
                if key_f and common_f:
                    query = self._build_query(
                        base_type, key_f, quality=q, sockets=s,
                        dps_min=dm, defense_mins=df)
                    result = _budget_search(query)
                    if result and result.get("result"):
                        logger.info(
                            f"TradeClient: key mods + {s}S "
                            f"({result.get('total', 0)} results)")
                        return result, False, len(common_f)
                    if _stale():
                        return None, False, 0

            # Fall through to count for small sets too
            if n >= 3:
                query = self._build_query(
                    base_type, stat_filters, "count", n - 1, quality=q, sockets=s,
                    dps_min=dm, defense_mins=df)
                result = _budget_search(query)
                if result and result.get("result"):
                    logger.info(
                        f"TradeClient: count({n-1}/{n}) = "
                        f"{result.get('total', 0)} results")
                    return result, False, 1
            return None, False, 0

        # For 5+ mods: use a multi-group strategy that preserves key mods.

        # Classify into key vs common
        if priceable and len(priceable) == len(stat_filters):
            key_mods, common_mods, key_filters, common_filters = \
                self._classify_filters(priceable, stat_filters)
        else:
            key_mods, common_mods = priceable or [], []
            key_filters, common_filters = stat_filters, []

        n_key = len(key_filters)
        n_common = len(common_filters)

        # Low-key-confidence: items with <= 1 key mod among 4+ total mods
        # are likely overpriced because the API returns items with the same
        # key mod but much better overall stats. Never claim exact_match.
        low_key_confidence = n_key <= 1 and n >= 4
        if low_key_confidence:
            logger.warning(
                f"TradeClient: low key confidence — only {n_key} key mod(s) "
                f"among {n} total. Prices will show as estimates."
            )

        logger.info(
            f"TradeClient: classified {n_key} key + {n_common} common mods "
            f"(of {n} total)"
        )

        # Step 1: Try count(n/n) at tight minimums first — if an exact
        # match exists at 90%, that's the best price.
        tight_filters = self._build_stat_filters_custom(priceable, 0.90)
        query = self._build_query(
            base_type, tight_filters, "count", n, quality=q, sockets=s,
            dps_min=dm, defense_mins=df)
        result = _budget_search(query)
        if result and result.get("result"):
            total = result.get("total", 0)
            logger.info(f"TradeClient: count({n}/{n}) @90% = {total} results")
            if total <= self._TOO_MANY_RESULTS:
                return result, not low_key_confidence, 0

        if _stale():
            return None, False, 0

        # Step 2 (NEW): Hybrid query — key mods as "and" (always required)
        # + common mods as "count" (allow some to be absent).
        # Only useful when we have >=2 key mods and >=1 common mod.
        if n_key >= 2 and n_common >= 1:
            # Try all common mods first
            query = self._build_hybrid_query(
                base_type, key_filters, common_filters,
                min_common=n_common, quality=q, sockets=s,
                dps_min=dm, defense_mins=df)
            result = _budget_search(query)
            if result and result.get("result"):
                total = result.get("total", 0)
                logger.info(
                    f"TradeClient: hybrid key={n_key} AND + "
                    f"common>={n_common} = {total} results"
                )
                if total <= self._TOO_MANY_RESULTS:
                    return result, not low_key_confidence, 0

            if _stale():
                return None, False, 0

            # Drop 1 common mod
            if n_common >= 2:
                query = self._build_hybrid_query(
                    base_type, key_filters, common_filters,
                    min_common=n_common - 1, quality=q, sockets=s,
                    dps_min=dm, defense_mins=df)
                result = _budget_search(query)
                if result and result.get("result"):
                    total = result.get("total", 0)
                    logger.info(
                        f"TradeClient: hybrid key={n_key} AND + "
                        f"common>={n_common - 1} = {total} results"
                    )
                    if total <= self._TOO_MANY_RESULTS:
                        return result, False, 1

                if _stale():
                    return None, False, 0

        # Step 3: count(n-1) at progressively looser minimums.
        # Results are LOWER BOUNDS — items are missing one mod, so the
        # actual item is strictly better. Flag as not exact.
        best = None
        for mult in (0.90, 0.85, 0.80):
            if _stale() or calls_remaining <= 0:
                break
            filters_at = self._build_stat_filters_custom(priceable, mult)
            query = self._build_query(
                base_type, filters_at, "count", n - 1, quality=q, sockets=s,
                dps_min=dm, defense_mins=df)
            result = _budget_search(query)
            if result and result.get("result"):
                total = result.get("total", 0)
                pct = int(mult * 100)
                logger.info(
                    f"TradeClient: count({n-1}/{n}) @{pct}% = {total} results")
                if total <= self._TOO_MANY_RESULTS:
                    return result, False, 1
                # Too many — use as best so far, stop loosening
                if best is None:
                    best = result
                break

        if best:
            return best, False, 1

        # Step 4: count(n-2), count(n-3), ... with standard minimums
        # These drop 2+ mods — comparables may be much better than actual
        # item, so mods_dropped is tracked for conservative pricing.
        floor = max(2, n // 2)
        best_dropped = 0
        for min_count in range(n - 2, floor - 1, -1):
            if _stale() or calls_remaining <= 0:
                break
            dropped = n - min_count
            query = self._build_query(
                base_type, stat_filters, "count", min_count, quality=q, sockets=s,
                dps_min=dm, defense_mins=df)
            result = _budget_search(query)
            if not result or not result.get("result"):
                continue

            total = result.get("total", 0)
            logger.info(f"TradeClient: count({min_count}/{n}) = {total} results")

            if total <= self._TOO_MANY_RESULTS:
                return result, False, dropped

            if best is None:
                best = result
                best_dropped = dropped
            break

        if best:
            return best, False, best_dropped

        logger.info(f"TradeClient: all queries returned 0 for {base_type}")
        return None, False, 0

    # ─── API Calls ────────────────────────────────

    # Don't wait longer than this for a rate-limit retry (seconds).
    # Avoids "Checking..." hanging for 60s — user can re-hover instead.
    _MAX_RETRY_WAIT = 5

    def _is_rate_limited(self) -> bool:
        """Check if we're in a rate-limit cooldown period."""
        if time.time() < self._rate_limited_until:
            return True
        return False

    def _set_rate_limited(self, retry_after: int):
        """Record a rate-limit cooldown so all calls bail immediately."""
        self._rate_limited_until = time.time() + retry_after
        logger.warning(
            f"TradeClient: rate limited for {retry_after}s, "
            f"pausing all requests"
        )

    def _do_search(self, query: dict) -> Optional[dict]:
        """POST search query to trade API. Returns search result or None.
        Retries once on connection errors / timeouts."""
        if self._is_rate_limited():
            return None

        url = f"{TRADE_API_BASE}/search/poe2/{self.league}"

        for attempt in range(2):
            self._rate_limit()
            try:
                resp = self._session.post(url, json=query, timeout=5)
                self._parse_rate_limit_headers(resp)

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 2))
                    if retry_after > self._MAX_RETRY_WAIT:
                        self._set_rate_limited(retry_after)
                        return None
                    logger.warning(f"TradeClient: rate limited, waiting {retry_after}s")
                    time.sleep(retry_after)
                    self._rate_limit()
                    resp = self._session.post(url, json=query, timeout=5)
                    self._parse_rate_limit_headers(resp)

                if resp.status_code != 200:
                    logger.warning(f"TradeClient: search returned HTTP {resp.status_code}")
                    return None

                return resp.json()

            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt == 0:
                    logger.warning(f"TradeClient: search {type(e).__name__}, retrying in 1s")
                    time.sleep(1)
                    continue
                logger.warning(f"TradeClient: search failed after retry: {e}")
                return None
            except Exception as e:
                logger.warning(f"TradeClient: search failed: {e}")
                return None

    def _do_fetch(self, query_id: str, result_ids: List[str]) -> Optional[list]:
        """GET listing details for the given result IDs.
        Retries once on connection errors / timeouts."""
        if not result_ids:
            return None
        if self._is_rate_limited():
            return None

        ids_str = ",".join(result_ids)
        url = f"{TRADE_API_BASE}/fetch/{ids_str}"

        for attempt in range(2):
            self._rate_limit()
            try:
                resp = self._session.get(url, params={"query": query_id}, timeout=5)
                self._parse_rate_limit_headers(resp)

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 2))
                    if retry_after > self._MAX_RETRY_WAIT:
                        self._set_rate_limited(retry_after)
                        return None
                    logger.warning(f"TradeClient: fetch rate limited, waiting {retry_after}s")
                    time.sleep(retry_after)
                    self._rate_limit()
                    resp = self._session.get(url, params={"query": query_id}, timeout=5)
                    self._parse_rate_limit_headers(resp)

                if resp.status_code != 200:
                    logger.warning(f"TradeClient: fetch returned HTTP {resp.status_code}")
                    return None

                data = resp.json()
                return data.get("result", [])

            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt == 0:
                    logger.warning(f"TradeClient: fetch {type(e).__name__}, retrying in 1s")
                    time.sleep(1)
                    continue
                logger.warning(f"TradeClient: fetch failed after retry: {e}")
                return None
            except Exception as e:
                logger.warning(f"TradeClient: fetch failed: {e}")
            return None

    # ─── Price Extraction ─────────────────────────

    def _build_result(self, listings: list, total: int,
                      lower_bound: bool = False) -> Optional[RarePriceResult]:
        """Extract prices from listings and build a RarePriceResult.

        Args:
            lower_bound: When True, comparables are worse than the actual item
                (mods were dropped to find results). Uses upper prices and
                shows "X+" instead of "X-Y" to indicate a floor price.
        """
        divine_to_chaos = self._divine_to_chaos_fn()
        # Collect (divine_value, original_amount, original_currency) for display
        prices = []

        for listing in listings:
            price_info = listing.get("listing", {}).get("price", {})
            amount = price_info.get("amount", 0)
            currency = price_info.get("currency", "")

            logger.debug(f"TradeClient: listing price: {amount} {currency}")

            if amount <= 0:
                continue

            divine_val = self._normalize_to_divine(amount, currency, divine_to_chaos)
            if divine_val is not None:
                prices.append((divine_val, amount, currency))
            else:
                logger.warning(f"TradeClient: unknown currency '{currency}' (amount={amount})")

        if not prices:
            return None

        prices.sort(key=lambda p: p[0])

        if lower_bound:
            # Comparables are missing mod(s) — the actual item is likely
            # worth more. Use the max price as the floor estimate.
            prices = prices[-1:]
        elif len(prices) >= 5:
            # Adaptive outlier trimming: when our sample is a tiny slice of
            # total listings, the cheapest ones are extreme low outliers.
            fetched = len(prices)
            sample_pct = fetched / total if total > 0 else 1.0
            if sample_pct < 0.05:
                # Bottom ~4% of market — skip bottom 75%, keep top quarter
                skip = fetched * 3 // 4
            elif sample_pct < 0.10:
                # Bottom 5-10% — skip bottom 50%
                skip = fetched // 2
            elif sample_pct < 0.25:
                # Bottom 10-25% — skip bottom 33%
                skip = fetched // 3
            else:
                # Small result set — skip bottom 25% (original behavior)
                skip = fetched // 4
            prices = prices[skip:]
        elif len(prices) >= 2:
            # Very few results — bottom listing(s) likely mispriced.
            # Use upper half as more representative.
            prices = prices[len(prices) // 2:]

        min_divine, min_amount, min_currency = prices[0]
        max_divine, max_amount, max_currency = prices[-1]

        # Collapse wide ranges to median — "~5 Divine" is more useful than "~1-20 Divine"
        if min_divine > 0 and max_divine > min_divine * 3:
            mid = prices[len(prices) // 2]
            min_divine, min_amount, min_currency = mid
            max_divine, max_amount, max_currency = mid

        # Items below ~10 exalted are effectively worthless — return None
        # so callers show "Low value" instead of a misleading price.
        divine_to_exalted = self._divine_to_exalted_fn()
        low_value_threshold = 10.0 / divine_to_exalted if divine_to_exalted > 0 else 0.03
        if min_divine < low_value_threshold:
            logger.info(
                f"TradeClient: below low-value threshold "
                f"({min_divine:.4f} div < {low_value_threshold:.4f} div = 10 exalted)")
            return None

        # Build display string
        if lower_bound:
            display = self._format_display_lower_bound(
                min_divine, min_amount, min_currency)
        else:
            display = self._format_display(
                min_divine, max_divine,
                min_amount, max_amount, min_currency, max_currency,
            )
        tier = self._determine_tier(min_divine)

        return RarePriceResult(
            min_price=round(min_divine, 2),
            max_price=round(max_divine, 2),
            num_results=total,
            display=display,
            tier=tier,
            estimate=lower_bound,
        )

    def _normalize_to_divine(self, amount: float, currency: str, divine_to_chaos: float) -> Optional[float]:
        """Convert a price to divine orb equivalent."""
        c = currency.lower()

        if c == "divine":
            return amount
        elif c == "chaos":
            return amount / divine_to_chaos if divine_to_chaos > 0 else None
        elif c == "exalted":
            divine_to_exalted = self._divine_to_exalted_fn()
            return amount / divine_to_exalted if divine_to_exalted > 0 else None
        elif c == "mirror":
            # Mirror of Kalandra — extremely high value
            return amount * 9000.0
        elif c in ("alchemy", "chance", "alteration", "transmute",
                    "augmentation", "jeweller", "fusing", "chromatic",
                    "scouring", "regret", "vaal", "regal"):
            # Low-value currencies — treat as near-zero
            return 0.001 * amount
        else:
            # Unknown currency — treat as chaos equivalent
            return amount / divine_to_chaos if divine_to_chaos > 0 else None

    # Currency display names
    _CURRENCY_NAMES = {
        "divine": "Divine", "chaos": "Chaos", "exalted": "Exalted",
        "mirror": "Mirror",
        "alchemy": "Alch", "vaal": "Vaal", "chance": "Chance",
    }

    def _format_display(self, min_divine: float, max_divine: float,
                        min_amount: float, max_amount: float,
                        min_currency: str, max_currency: str) -> str:
        """Format a price range for display, using the most readable denomination."""
        # If priced in divine, show divine
        if min_currency == "divine":
            if min_amount == max_amount or abs(max_amount - min_amount) < 1:
                return f"~{min_amount:.0f} Divine" if min_amount >= 10 else f"~{min_amount:.1f} Divine"
            lo = f"{min_amount:.0f}" if min_amount >= 10 else f"{min_amount:.1f}"
            hi = f"{max_amount:.0f}" if max_amount >= 10 else f"{max_amount:.1f}"
            return f"~{lo}-{hi} Divine"

        # If priced in same currency, show that currency directly
        if min_currency == max_currency:
            cname = self._CURRENCY_NAMES.get(min_currency, min_currency.title())
            if abs(max_amount - min_amount) < 2:
                return f"~{min_amount:.0f} {cname}"
            return f"~{min_amount:.0f}-{max_amount:.0f} {cname}"

        # Mixed currencies — normalize to divine or chaos
        if min_divine >= 1.0:
            if abs(max_divine - min_divine) < 0.5:
                return f"~{min_divine:.1f} Divine"
            return f"~{min_divine:.1f}-{max_divine:.1f} Divine"

        # Show in chaos
        divine_to_chaos = self._divine_to_chaos_fn()
        chaos_min = min_divine * divine_to_chaos
        chaos_max = max_divine * divine_to_chaos
        if abs(chaos_max - chaos_min) < 3:
            return f"~{chaos_min:.0f} Chaos"
        return f"~{chaos_min:.0f}-{chaos_max:.0f} Chaos"

    @staticmethod
    def _fmt_price(val: float) -> str:
        """Format a price value, dropping unnecessary '.0'."""
        if val == int(val):
            return str(int(val))
        return f"{val:.1f}"

    def _format_display_lower_bound(self, divine_val: float,
                                    amount: float, currency: str) -> str:
        """Format a lower-bound price (item is better than comparables).

        Shows 'X+ Divine (est.)' to indicate the price is a conservative
        estimate — no exact comparables exist on the trade site.
        """
        if currency == "divine":
            return f"{self._fmt_price(amount)}+ Divine (est.)"

        # Normalize to divine for display
        if divine_val >= 1.0:
            return f"{self._fmt_price(divine_val)}+ Divine (est.)"

        # Show in chaos
        divine_to_chaos = self._divine_to_chaos_fn()
        chaos_val = divine_val * divine_to_chaos
        return f"{int(chaos_val)}+ Chaos (est.)"

    def _determine_tier(self, min_price: float) -> str:
        """Determine price tier from divine value."""
        if min_price >= 5.0:
            return "high"
        elif min_price >= 1.0:
            return "good"
        elif min_price >= 0.1:
            return "decent"
        else:
            return "low"

    # ─── Rate Limiting ────────────────────────────

    def _rate_limit(self):
        """Enforce rate limit between API requests.

        Uses adaptive interval from API headers when available,
        falling back to _min_interval (1.0s) when no rules are known.
        """
        with self._rate_lock:
            interval = self._compute_adaptive_interval()
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < interval:
                time.sleep(interval - elapsed)
            self._last_request_time = time.time()

    def _parse_rate_limit_headers(self, resp):
        """Parse rate limit rules and state from API response headers.

        Headers format (comma-separated triplets):
            X-Rate-Limit-Ip: max_hits:window_secs:penalty_secs,...
            X-Rate-Limit-Ip-State: current_hits:window_secs:penalty_remaining,...
        """
        rules_raw = resp.headers.get("X-Rate-Limit-Ip")
        state_raw = resp.headers.get("X-Rate-Limit-Ip-State")
        if not rules_raw or not state_raw:
            return

        try:
            rules = []
            for part in rules_raw.split(","):
                max_hits, window, penalty = part.strip().split(":")
                rules.append((int(max_hits), int(window), int(penalty)))

            state = []
            for part in state_raw.split(","):
                hits, window, penalty_rem = part.strip().split(":")
                state.append((int(hits), int(window), int(penalty_rem)))

            with self._rate_lock:
                self._rl_rules = rules
                self._rl_state = state

                # Set rate limited if any window has an active penalty
                for hits, window, penalty_rem in state:
                    if penalty_rem > 0:
                        until = time.time() + penalty_rem
                        if until > self._rate_limited_until:
                            self._rate_limited_until = until
                            logger.warning(
                                f"TradeClient: API penalty active — "
                                f"{penalty_rem}s remaining")

                if not self._rl_rules_logged:
                    self._rl_rules_logged = True
                    rule_strs = [
                        f"{m}/{w}s (penalty {p}s)" for m, w, p in rules]
                    logger.info(
                        f"TradeClient: rate limit rules discovered: "
                        + ", ".join(rule_strs))
        except (ValueError, AttributeError):
            pass  # Malformed headers — ignore silently

    def _compute_adaptive_interval(self):
        """Compute the safest request interval based on current API usage.

        Called inside _rate_lock. For each window, if usage exceeds 50%
        of the limit, switches to the safe rate (window_secs / max_hits).
        Returns the most conservative (largest) interval across all windows.
        Falls back to _min_interval when no rules are known.
        """
        if not self._rl_rules:
            return self._min_interval

        best_interval = self._min_interval

        for i, (max_hits, window, penalty) in enumerate(self._rl_rules):
            safe_rate = window / max_hits
            if i < len(self._rl_state):
                current_hits, _, _ = self._rl_state[i]
                usage_ratio = current_hits / max_hits if max_hits > 0 else 1.0
                if usage_ratio >= 0.5 and safe_rate > best_interval:
                    if safe_rate > self._min_interval:
                        logger.info(
                            f"TradeClient: adaptive backoff — "
                            f"{current_hits}/{max_hits} hits in {window}s "
                            f"window, slowing to {safe_rate:.1f}s/req")
                    best_interval = safe_rate

        return best_interval

    # ─── Caching ──────────────────────────────────

    def _make_fingerprint(self, base_type: str, mods: List[ParsedMod],
                          quality: int = 0, sockets: int = 0,
                          dps: float = 0, defense: int = 0) -> str:
        """Create a cache key from base type, mods, quality, sockets, and combat stats."""
        parts = [base_type.lower()]
        if quality > 0:
            parts.append(f"q{quality}")
        if sockets > 0:
            parts.append(f"s{sockets}")
        if dps > 0:
            parts.append(f"dps{int(dps)}")
        if defense > 0:
            parts.append(f"def{defense}")
        for mod in sorted(mods, key=lambda m: m.stat_id):
            # Round values to reduce cache misses for similar items
            rounded = round(mod.value / 5) * 5
            parts.append(f"{mod.stat_id}:{rounded}")
        key = "|".join(parts)
        return hashlib.md5(key.encode()).hexdigest()

    def _check_cache(self, fingerprint: str) -> Optional[RarePriceResult]:
        """Check in-memory cache. Returns result or None."""
        with self._cache_lock:
            entry = self._cache.get(fingerprint)
            if entry:
                result, timestamp = entry
                if time.time() - timestamp < TRADE_CACHE_TTL:
                    return result
                else:
                    del self._cache[fingerprint]
        return None

    def _put_cache(self, fingerprint: str, result: RarePriceResult):
        """Store result in cache."""
        with self._cache_lock:
            self._cache[fingerprint] = (result, time.time())

            # Evict old entries if cache gets large
            if len(self._cache) > 200:
                now = time.time()
                expired = [k for k, (_, ts) in self._cache.items() if now - ts > TRADE_CACHE_TTL]
                for k in expired:
                    del self._cache[k]
