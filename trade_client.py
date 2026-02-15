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
        tc = TradeClient(league="Fate of the Vaal", divine_to_chaos_fn=lambda: 68.0)
        result = tc.price_rare_item(item, parsed_mods)
    """

    def __init__(self, league: str = DEFAULT_LEAGUE, divine_to_chaos_fn=None):
        self.league = league
        self._divine_to_chaos_fn = divine_to_chaos_fn or (lambda: 68.0)
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
            # Build and execute search query
            query = self._build_query(base_type, mods)
            search_result = self._do_search(query)
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

    def _build_query(self, base_type: str, mods: List[ParsedMod]) -> dict:
        """Build a trade API search query from base type and mods."""
        stat_filters = []
        for mod in mods:
            min_val = mod.value * TRADE_MOD_MIN_MULTIPLIER
            # Round down to integer for cleaner filtering
            min_val = int(min_val) if min_val == int(min_val) else round(min_val, 1)

            stat_filter = {"id": mod.stat_id, "value": {"min": min_val}}
            stat_filters.append(stat_filter)

        query = {
            "query": {
                "status": {"option": "online"},
                "type": base_type,
                "stats": [{"type": "and", "filters": stat_filters}],
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

        return query

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
        prices_divine = []

        for listing in listings:
            price_info = listing.get("listing", {}).get("price", {})
            amount = price_info.get("amount", 0)
            currency = price_info.get("currency", "")

            if amount <= 0:
                continue

            # Normalize to divine
            divine_val = self._normalize_to_divine(amount, currency, divine_to_chaos)
            if divine_val is not None:
                prices_divine.append(divine_val)

        if not prices_divine:
            return None

        prices_divine.sort()
        min_price = prices_divine[0]
        max_price = prices_divine[-1]  # highest of the fetched listings

        # Build display string
        display = self._format_display(min_price, max_price)
        tier = self._determine_tier(min_price)

        return RarePriceResult(
            min_price=round(min_price, 2),
            max_price=round(max_price, 2),
            num_results=total,
            display=display,
            tier=tier,
        )

    def _normalize_to_divine(self, amount: float, currency: str, divine_to_chaos: float) -> Optional[float]:
        """Convert a price to divine orb equivalent."""
        currency_lower = currency.lower()

        if currency_lower in ("divine", "divine_orb", "divine-orb"):
            return amount
        elif currency_lower in ("chaos", "chaos_orb", "chaos-orb"):
            if divine_to_chaos > 0:
                return amount / divine_to_chaos
            return None
        elif currency_lower in ("exalted", "exalted_orb", "exalted-orb"):
            # Exalted is typically much less than divine
            # Approximate: 1 exalted ≈ 1/387 divine (varies by league)
            return amount / 387.0
        else:
            # Unknown currency — try treating as chaos
            if divine_to_chaos > 0:
                return amount / divine_to_chaos
            return None

    def _format_display(self, min_price: float, max_price: float) -> str:
        """Format a price range for display."""
        if min_price >= 1.0:
            if min_price == max_price or abs(max_price - min_price) < 0.5:
                return f"~{min_price:.0f} Divine" if min_price >= 10 else f"~{min_price:.1f} Divine"
            else:
                lo = f"{min_price:.0f}" if min_price >= 10 else f"{min_price:.1f}"
                hi = f"{max_price:.0f}" if max_price >= 10 else f"{max_price:.1f}"
                return f"~{lo}-{hi} Divine"
        else:
            # Show in chaos
            divine_to_chaos = self._divine_to_chaos_fn()
            chaos_min = min_price * divine_to_chaos
            chaos_max = max_price * divine_to_chaos
            if abs(chaos_max - chaos_min) < 3:
                return f"~{chaos_min:.0f} Chaos"
            else:
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
