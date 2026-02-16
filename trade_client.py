"""
POE2 Price Overlay - Trade API Client
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
        self._session.headers.update({"User-Agent": "POE2PriceOverlay/1.0"})

        # Rate limiting
        self._last_request_time = 0.0
        self._rate_lock = threading.Lock()
        self._min_interval = 1.0 / TRADE_MAX_REQUESTS_PER_SECOND

        # In-memory cache: fingerprint → (result, timestamp)
        self._cache: dict = {}
        self._cache_lock = threading.Lock()

    def price_rare_item(self, item, mods: List[ParsedMod]) -> Optional[RarePriceResult]:
        """
        Price a rare item by querying the trade API.

        Args:
            item: ParsedItem with base_type
            mods: List of ParsedMod from ModParser

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

        # Check cache
        fingerprint = self._make_fingerprint(base_type, mods)
        cached = self._check_cache(fingerprint)
        if cached:
            logger.debug(f"TradeClient: cache hit for {base_type}")
            return cached

        try:
            # Build stat filters once, then try progressively looser queries
            # Include explicit, implicit, fractured, and desecrated mods.
            # Skip rune (socketed), enchant (changeable), and crafted (bench).
            _PRICEABLE_TYPES = ("explicit", "implicit", "fractured", "desecrated")
            priceable = [m for m in mods if m.mod_type in _PRICEABLE_TYPES]
            stat_filters = self._build_stat_filters(priceable)

            search_result = self._search_progressive(base_type, stat_filters, priceable)
            if not search_result:
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

            # Extract prices and build result
            result = self._build_result(listings, total)
            if result:
                self._put_cache(fingerprint, result)
                logger.info(
                    f"RARE PRICE {base_type}: {result.display} "
                    f"({result.num_results} results)"
                )

            return result

        except Exception as e:
            logger.warning(f"TradeClient: pricing failed for {base_type}: {e}")
            return None

    # ─── Query Building ───────────────────────────

    def _build_stat_filters(self, priceable: List[ParsedMod]) -> list:
        """Build stat filter dicts from priceable mods."""
        stat_filters = []
        for mod in priceable:
            min_val = self._compute_min_value(mod.value)
            stat_filters.append({"id": mod.stat_id, "value": {"min": min_val}})
        return stat_filters

    @staticmethod
    def _compute_min_value(value: float) -> float:
        """Compute minimum filter value with value-dependent tightness.

        Low values (1-10) use tight matching because they represent discrete
        tiers where each point matters enormously (e.g., +6 vs +7 skills).
        Higher values use progressively looser matching.
        """
        if value <= 10:
            multiplier = 0.95
        elif value <= 50:
            multiplier = 0.90
        else:
            multiplier = TRADE_MOD_MIN_MULTIPLIER  # 0.8
        min_val = value * multiplier
        return int(min_val) if min_val == int(min_val) else round(min_val, 1)

    def _build_query(self, base_type: str, stat_filters: list,
                     match_mode: str = "and", min_count: int = 0) -> dict:
        """Build a trade API search query.

        Args:
            match_mode: "and" (all must match) or "count" (at least min_count)
        """
        if match_mode == "count" and min_count > 0:
            stat_group = {"type": "count", "value": {"min": min_count},
                          "filters": stat_filters}
        else:
            stat_group = {"type": "and", "filters": stat_filters}

        return {
            "query": {
                "status": {"option": "any"},
                "type": base_type,
                "stats": [stat_group],
                "filters": {
                    "type_filters": {
                        "filters": {
                            "rarity": {"option": "nonunique"},
                        }
                    },
                    "misc_filters": {
                        "filters": {
                            "mirrored": {"option": "false"},
                        }
                    },
                },
            },
            "sort": {"price": "asc"},
        }

    # Patterns for common "filler" mods that rarely drive item price.
    # Mods NOT matching any of these patterns are considered "key" mods.
    _COMMON_MOD_PATTERNS = (
        "maximum mana", "maximum life", "maximum energy shield",
        "mana regeneration", "life regeneration", "energy shield recharge",
        "to fire resistance", "to cold resistance", "to lightning resistance",
        "to chaos resistance", "to all elemental resistances",
        "to strength", "to dexterity", "to intelligence", "to all attributes",
        "item rarity", "light radius", "stun ", "knockback",
        "mana on kill", "life on kill",
    )

    def _classify_filters(self, priceable: List[ParsedMod], stat_filters: list):
        """Split stat filters into key (price-driving) and common (filler)."""
        key = []
        common = []
        for mod, sf in zip(priceable, stat_filters):
            text_lower = mod.raw_text.lower()
            is_common = any(pat in text_lower for pat in self._COMMON_MOD_PATTERNS)
            if is_common:
                common.append(sf)
            else:
                key.append(sf)
        return key, common

    def _search_progressive(self, base_type: str, stat_filters: list,
                            priceable: List[ParsedMod] = None) -> Optional[dict]:
        """Find the best price query by progressively relaxing filters.

        Strategy:
        1. Try "and" with ALL mods (exact match, best accuracy)
        2. Try "and" with only KEY mods (drop common filler mods)
        3. Progressively remove key mods until results are found
        """
        n = len(stat_filters)
        if n <= 3:
            return self._do_search(self._build_query(base_type, stat_filters))

        # Step 1: Try all mods
        query = self._build_query(base_type, stat_filters, "and")
        result = self._do_search(query)
        if result and result.get("result"):
            logger.debug(f"TradeClient: exact match ({result.get('total', 0)} results)")
            return result

        # Step 2: Classify into key vs common, try key mods only
        if priceable and len(priceable) == len(stat_filters):
            key_filters, common_filters = self._classify_filters(priceable, stat_filters)
        else:
            key_filters, common_filters = stat_filters, []

        if key_filters and len(key_filters) < n:
            logger.debug(
                f"TradeClient: trying {len(key_filters)} key mods "
                f"(dropped {len(common_filters)} common)"
            )
            query = self._build_query(base_type, key_filters, "and")
            result = self._do_search(query)
            if result and result.get("result"):
                logger.debug(
                    f"TradeClient: key-mods hit ({result.get('total', 0)} results)"
                )
                return result

            # Step 3: Still 0 — try dropping key mods one at a time from the end
            # (keep the first/most distinctive mods)
            for drop in range(1, len(key_filters) - 1):
                subset = key_filters[:len(key_filters) - drop]
                if len(subset) < 2:
                    break
                query = self._build_query(base_type, subset, "and")
                result = self._do_search(query)
                if result and result.get("result"):
                    logger.debug(
                        f"TradeClient: {len(subset)}-key-mods hit "
                        f"({result.get('total', 0)} results)"
                    )
                    return result

        # Step 4: Last resort — count-based with all filters
        min_count = max(2, n - 2)
        query = self._build_query(base_type, stat_filters, "count", min_count)
        result = self._do_search(query)
        if result and result.get("result"):
            logger.debug(
                f"TradeClient: count({min_count}/{n}) fallback "
                f"({result.get('total', 0)} results)"
            )
            return result

        logger.debug(f"TradeClient: all queries returned 0 for {base_type}")
        return None

    # ─── API Calls ────────────────────────────────

    def _do_search(self, query: dict) -> Optional[dict]:
        """POST search query to trade API. Returns search result or None."""
        url = f"{TRADE_API_BASE}/search/poe2/{self.league}"
        self._rate_limit()

        try:
            resp = self._session.post(url, json=query, timeout=5)

            if resp.status_code == 429:
                # Rate limited — try once more after Retry-After
                retry_after = int(resp.headers.get("Retry-After", 2))
                logger.warning(f"TradeClient: rate limited, waiting {retry_after}s")
                time.sleep(retry_after)
                self._rate_limit()
                resp = self._session.post(url, json=query, timeout=5)

            if resp.status_code != 200:
                logger.warning(f"TradeClient: search returned HTTP {resp.status_code}")
                return None

            return resp.json()

        except requests.Timeout:
            logger.warning("TradeClient: search timed out")
            return None
        except Exception as e:
            logger.warning(f"TradeClient: search failed: {e}")
            return None

    def _do_fetch(self, query_id: str, result_ids: List[str]) -> Optional[list]:
        """GET listing details for the given result IDs."""
        if not result_ids:
            return None

        ids_str = ",".join(result_ids)
        url = f"{TRADE_API_BASE}/fetch/{ids_str}"
        self._rate_limit()

        try:
            resp = self._session.get(url, params={"query": query_id}, timeout=5)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 2))
                logger.warning(f"TradeClient: fetch rate limited, waiting {retry_after}s")
                time.sleep(retry_after)
                self._rate_limit()
                resp = self._session.get(url, params={"query": query_id}, timeout=5)

            if resp.status_code != 200:
                logger.warning(f"TradeClient: fetch returned HTTP {resp.status_code}")
                return None

            data = resp.json()
            return data.get("result", [])

        except requests.Timeout:
            logger.warning("TradeClient: fetch timed out")
            return None
        except Exception as e:
            logger.warning(f"TradeClient: fetch failed: {e}")
            return None

    # ─── Price Extraction ─────────────────────────

    def _build_result(self, listings: list, total: int) -> Optional[RarePriceResult]:
        """Extract prices from listings and build a RarePriceResult."""
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

        # Skip cheapest ~25% as outliers (mispriced or lower-quality matches)
        if len(prices) >= 5:
            skip = len(prices) // 4
            prices = prices[skip:]

        min_divine, min_amount, min_currency = prices[0]
        max_divine, max_amount, max_currency = prices[-1]

        # Build display string using original currency for readability
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
        elif c in ("alchemy", "chance", "alteration", "transmute",
                    "augmentation", "jeweller", "fusing", "chromatic",
                    "scouring", "regret", "vaal"):
            # Low-value currencies — treat as near-zero
            return 0.001 * amount
        else:
            # Unknown currency — treat as chaos equivalent
            return amount / divine_to_chaos if divine_to_chaos > 0 else None

    # Currency display names
    _CURRENCY_NAMES = {
        "divine": "Divine", "chaos": "Chaos", "exalted": "Exalted",
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
        """Enforce rate limit between API requests."""
        with self._rate_lock:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_request_time = time.time()

    # ─── Caching ──────────────────────────────────

    def _make_fingerprint(self, base_type: str, mods: List[ParsedMod]) -> str:
        """Create a cache key from base type and mods."""
        parts = [base_type.lower()]
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
