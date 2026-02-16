"""
POE2 Price Overlay - Price Cache
Fetches item prices from poe2scout.com (primary) and poe.ninja (secondary).

poe2scout provides pre-aggregated prices for 900+ items across all categories
(uniques, currencies, gems, maps, etc.). Prices are converted to divine values
using the league's divinePrice from the /leagues endpoint.

poe.ninja exchange endpoint provides conversion rates (divine→chaos, divine→exalted)
and serves as a fallback data source for currency-type items.
"""

import json
import time
import logging
import threading
from pathlib import Path
from typing import Optional

import requests

from config import (
    PRICE_REFRESH_INTERVAL,
    CACHE_DIR,
    DEFAULT_LEAGUE,
    POE2SCOUT_BASE_URL,
)

logger = logging.getLogger(__name__)

# ─── poe.ninja POE2 exchange API ─────────────
POE2_BASE = "https://poe.ninja/poe2/api/economy"
POE2_EXCHANGE_URL = f"{POE2_BASE}/exchange/current/overview"

EXCHANGE_CATEGORIES = [
    "Currency",
    "Fragments",
    "Essences",
    "Runes",
    "Expedition",
    "SoulCores",
    "Idols",
    "UncutGems",
    "LineageSupportGems",
    "Ultimatum",
    "Breach",
    "Delirium",
    "Ritual",
    "Abyss",
]

# ─── poe2scout categories ────────────────────
POE2SCOUT_UNIQUE_CATEGORIES = [
    "accessory", "armour", "flask", "jewel", "map", "weapon", "sanctum",
]

POE2SCOUT_CURRENCY_CATEGORIES = [
    "currency", "fragments", "runes", "talismans", "essences", "ultimatum",
    "expedition", "ritual", "vaultkeys", "breach", "abyss", "uncutgems",
    "lineagesupportgems", "delirium", "incursion", "idol",
]

REQUEST_DELAY = 0.3  # seconds between API calls


class PriceCache:
    def __init__(self, league: str = DEFAULT_LEAGUE):
        self.league = league
        self.prices: dict = {}           # name.lower() -> {divine_value, chaos_value, name, category}
        self.divine_to_chaos: float = 68.0   # 1 divine = X chaos (from poe.ninja rates)
        self.divine_to_exalted: float = 387.0  # 1 divine = X exalted (from poe.ninja rates)
        self._poe2scout_divine_price: float = 0  # chaos per divine from poe2scout leagues
        self.last_refresh: float = 0
        self._lock = threading.Lock()
        self._running = False
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def start(self):
        self._running = True
        self._load_from_disk()
        t = threading.Thread(target=self._refresh_loop, daemon=True)
        t.start()

    def stop(self):
        self._running = False

    def lookup(self, item_name: str, base_type: str = "", item_level: int = 0) -> Optional[dict]:
        """Look up price. Returns enriched dict or None."""
        with self._lock:
            key = item_name.strip().lower()

            # Direct match
            if key in self.prices:
                return self._enrich(self.prices[key])

            # Base type match
            if base_type:
                bk = base_type.strip().lower()
                if bk in self.prices:
                    r = self.prices[bk].copy()
                    if item_level > 0:
                        r = self._adjust_ilvl(r, item_level)
                    return self._enrich(r)

            # Fuzzy match (OCR errors)
            for ck, cd in self.prices.items():
                if self._fuzzy(key, ck):
                    return self._enrich(cd)

            return None

    def lookup_unidentified(self, base_type: str) -> Optional[dict]:
        """
        Look up all uniques sharing a base type and return a price range.
        Used for unidentified unique items where only the base type is known.
        """
        if not base_type:
            return None

        bt_lower = base_type.strip().lower()
        matches = []

        with self._lock:
            for data in self.prices.values():
                if data.get("base_type", "").lower() == bt_lower:
                    matches.append(data)

        if not matches:
            return None

        # Sort by divine value
        matches.sort(key=lambda m: m.get("divine_value", 0))
        low = matches[0]
        high = matches[-1]

        low_dv = low["divine_value"]
        high_dv = high["divine_value"]

        # Build display string
        if len(matches) == 1:
            display = self._enrich(low)["display"]
            name = low["name"]
        else:
            low_str = self._format_value(low_dv)
            high_str = self._format_value(high_dv)
            display = f"{low_str}-{high_str}"
            name = f"{len(matches)} possible uniques"

        # Tier based on highest possible value
        if high_dv >= 5:
            tier = "high"
        elif high_dv >= 1:
            tier = "good"
        elif high_dv * self.divine_to_exalted >= 1:
            tier = "decent"
        else:
            tier = "low"

        return {
            "display": display,
            "tier": tier,
            "name": name,
            "divine_value": high_dv,
            "unidentified": True,
        }

    def _format_value(self, divine_value: float) -> str:
        """Format a divine value to a readable string."""
        if divine_value >= 0.99:
            if divine_value >= 10:
                return f"{divine_value:.0f} Divine"
            return f"{divine_value:.1f} Divine"
        ev = divine_value * self.divine_to_exalted
        if ev >= 5:
            return f"{ev:.0f} Exalted"
        if ev >= 1:
            return f"{ev:.1f} Exalted"
        chaos = divine_value * self.divine_to_chaos
        if chaos >= 3:
            return f"{chaos:.0f} Chaos"
        return "< 3 Chaos"

    def lookup_from_text(self, ocr_text: str) -> Optional[dict]:
        """
        Try to find a priced item name anywhere in the OCR text.
        Returns enriched price dict or None.
        """
        if not ocr_text:
            return None

        text_lower = ocr_text.lower()

        with self._lock:
            best_match = None
            best_len = 0

            for cache_key, data in self.prices.items():
                if len(cache_key) < 8:
                    continue
                if cache_key in text_lower and len(cache_key) > best_len:
                    best_match = data
                    best_len = len(cache_key)

            if best_match:
                return self._enrich(best_match)

            return None

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "total_items": len(self.prices),
                "divine_to_chaos": self.divine_to_chaos,
                "divine_to_exalted": self.divine_to_exalted,
                "last_refresh": time.strftime("%H:%M:%S", time.localtime(self.last_refresh)) if self.last_refresh else "Never",
                "league": self.league,
            }

    # ─── Fetch ───────────────────────────────────────

    def _refresh_loop(self):
        while self._running:
            try:
                self._fetch_all()
                self._save_to_disk()
                self.last_refresh = time.time()
                logger.info(f"Cache refreshed: {len(self.prices)} items "
                            f"(1 div = {self.divine_to_chaos:.0f}c / {self.divine_to_exalted:.0f}ex)")
            except Exception as e:
                logger.error(f"Refresh failed: {e}")
            for _ in range(int(PRICE_REFRESH_INTERVAL)):
                if not self._running:
                    return
                time.sleep(1)

    def _fetch_all(self):
        new_prices = {}
        headers = {"User-Agent": "POE2PriceOverlay/1.0"}

        # 1. Fetch poe.ninja first — we need conversion rates before poe2scout
        self._fetch_poe_ninja(headers, new_prices)

        # 2. Fetch poe2scout league data (divine price for unit conversion)
        self._fetch_poe2scout_leagues(headers)

        # 3. Fetch all poe2scout categories (primary source, 900+ items)
        #    poe2scout items overwrite poe.ninja items (more comprehensive)
        self._fetch_poe2scout(headers, new_prices)

        with self._lock:
            if new_prices:
                self.prices = new_prices

    # ─── poe2scout ────────────────────────────────────

    def _fetch_poe2scout_leagues(self, headers: dict):
        """Fetch league data to get divine orb price (for unit conversion)."""
        try:
            url = f"{POE2SCOUT_BASE_URL}/leagues"
            resp = requests.get(url, timeout=15, headers=headers)
            if resp.status_code != 200:
                logger.warning(f"poe2scout leagues: HTTP {resp.status_code}")
                return

            leagues = resp.json()
            for league in leagues:
                if league.get("value") == self.league:
                    dp = league.get("divinePrice", 0)
                    if dp > 0:
                        self._poe2scout_divine_price = dp
                        logger.info(f"poe2scout: divinePrice={dp:.1f} ({self.league})")
                    return

            logger.warning(f"poe2scout: league '{self.league}' not found in API")
        except Exception as e:
            logger.warning(f"poe2scout leagues failed: {e}")

    def _fetch_poe2scout(self, headers: dict, prices: dict):
        """Fetch all poe2scout categories (uniques + currencies)."""
        if self._poe2scout_divine_price <= 0:
            logger.warning("poe2scout: no divine price — skipping")
            return

        total = 0

        # Unique categories
        for cat in POE2SCOUT_UNIQUE_CATEGORIES:
            try:
                count = self._fetch_poe2scout_paginated(
                    f"{POE2SCOUT_BASE_URL}/items/unique/{cat}",
                    headers, prices, cat, is_unique=True,
                )
                total += count
                time.sleep(REQUEST_DELAY)
            except Exception as e:
                logger.warning(f"poe2scout unique/{cat} failed: {e}")

        # Currency categories
        for cat in POE2SCOUT_CURRENCY_CATEGORIES:
            try:
                count = self._fetch_poe2scout_paginated(
                    f"{POE2SCOUT_BASE_URL}/items/currency/{cat}",
                    headers, prices, cat, is_unique=False,
                )
                total += count
                time.sleep(REQUEST_DELAY)
            except Exception as e:
                logger.warning(f"poe2scout currency/{cat} failed: {e}")

        logger.info(f"poe2scout total: {total} items")

    def _fetch_poe2scout_paginated(self, base_url: str, headers: dict,
                                    prices: dict, category: str,
                                    is_unique: bool) -> int:
        """Fetch all pages from a poe2scout endpoint. Returns item count."""
        count = 0
        page = 1

        while True:
            resp = requests.get(
                base_url,
                params={"league": self.league, "page": page},
                timeout=15, headers=headers,
            )
            if resp.status_code != 200:
                logger.warning(f"poe2scout {category} p{page}: HTTP {resp.status_code}")
                break

            data = resp.json()

            # Handle both paginated {items:[]} and flat array responses
            if isinstance(data, list):
                items = data
                total_pages = 1
            else:
                items = data.get("items", [])
                total_pages = data.get("pages", 1)

            for item in items:
                if is_unique:
                    count += self._parse_poe2scout_unique(item, category, prices)
                else:
                    count += self._parse_poe2scout_currency(item, category, prices)

            if page >= total_pages:
                break
            page += 1
            time.sleep(REQUEST_DELAY)

        logger.debug(f"  poe2scout {category}: {count} items")
        return count

    def _parse_poe2scout_unique(self, item: dict, category: str, prices: dict) -> int:
        """Parse a unique item from poe2scout. Returns 1 if added, 0 if skipped."""
        name = item.get("name", "")
        base_type = item.get("type", "")
        raw_price = item.get("currentPrice", 0)

        if not name or not raw_price:
            return 0

        divine_value = raw_price / self._poe2scout_divine_price
        chaos_value = divine_value * self.divine_to_chaos

        entry = {
            "divine_value": divine_value,
            "chaos_value": chaos_value,
            "name": name,
            "base_type": base_type,
            "category": f"unique/{category}",
            "source": "poe2scout",
        }

        # Store by unique name only — not by base_type, which would cause
        # magic/normal/rare items with the same base to match this unique's price
        key = name.lower()
        prices[key] = entry

        return 1

    def _parse_poe2scout_currency(self, item: dict, category: str, prices: dict) -> int:
        """Parse a currency item from poe2scout. Returns 1 if added, 0 if skipped."""
        name = item.get("text", "")
        raw_price = item.get("currentPrice", 0)

        if not name or not raw_price:
            return 0

        divine_value = raw_price / self._poe2scout_divine_price
        chaos_value = divine_value * self.divine_to_chaos

        key = name.lower()
        prices[key] = {
            "divine_value": divine_value,
            "chaos_value": chaos_value,
            "name": name,
            "category": category,
            "source": "poe2scout",
        }
        return 1

    # ─── poe.ninja (secondary) ────────────────────────

    def _fetch_poe_ninja(self, headers: dict, prices: dict):
        """Fetch poe.ninja exchange data — provides conversion rates and currency prices."""
        ninja_count = 0

        for cat in EXCHANGE_CATEGORIES:
            try:
                resp = requests.get(POE2_EXCHANGE_URL,
                    params={"league": self.league, "type": cat},
                    timeout=15, headers=headers)

                if resp.status_code == 200 and resp.content:
                    data = resp.json()
                    ninja_count += self._parse_exchange(data, cat, prices)
                else:
                    logger.debug(f"poe.ninja {cat}: HTTP {resp.status_code}")

                time.sleep(REQUEST_DELAY)
            except Exception as e:
                logger.debug(f"poe.ninja {cat} failed: {e}")

        logger.info(f"poe.ninja: {ninja_count} items loaded")

    def _parse_exchange(self, data: dict, category: str, prices: dict) -> int:
        """
        Parse the poe.ninja exchange endpoint. Adds currency items and updates
        conversion rates. poe2scout will overwrite these if it has the same items.
        """
        core = data.get("core", {})
        lines = data.get("lines", [])
        rates = core.get("rates", {})

        items = data.get("items", []) or core.get("items", [])

        # Update global conversion rates (poe.ninja is authoritative for these)
        if rates.get("chaos"):
            self.divine_to_chaos = rates["chaos"]
        if rates.get("exalted"):
            self.divine_to_exalted = rates["exalted"]

        # Build id → name map
        id_map = {}
        for item in items:
            iid = item.get("id", "")
            name = item.get("name", "")
            if iid and name:
                id_map[iid] = name

        count = 0
        for line in lines:
            lid = line.get("id", "")
            divine_value = line.get("primaryValue", 0)
            name = id_map.get(lid, lid)

            if not name or divine_value == 0:
                continue

            key = name.lower()
            prices[key] = {
                "divine_value": divine_value,
                "chaos_value": divine_value * self.divine_to_chaos,
                "name": name,
                "category": category,
                "source": "poe.ninja",
            }
            count += 1

        return count

    # ─── Display ─────────────────────────────────────

    def _enrich(self, data: dict) -> dict:
        """Add display string and tier."""
        result = data.copy()
        dv = result.get("divine_value", 0)
        chaos = result.get("chaos_value", 0)

        # Calculate exalted value
        ex_rate = self.divine_to_exalted
        ev = dv * ex_rate if ex_rate > 0 else 0
        result["exalted_value"] = round(ev, 1)

        # Display string - use the most readable denomination
        # Use 0.99 threshold for divine to handle API rounding (e.g. Divine Orb = 0.993 div)
        if dv >= 0.99:
            result["display"] = f"{dv:.1f} Divine" if dv < 10 else f"{dv:.0f} Divine"
            result["tier"] = "high" if dv >= 5 else "good"
        elif ev >= 5:
            result["display"] = f"{ev:.0f} Exalted"
            result["tier"] = "good"
        elif ev >= 1:
            result["display"] = f"{ev:.1f} Exalted"
            result["tier"] = "decent"
        elif chaos >= 3:
            result["display"] = f"{chaos:.0f} Chaos"
            result["tier"] = "low"
        else:
            result["display"] = "< 3 Chaos"
            result["tier"] = "low"

        return result

    def _adjust_ilvl(self, data, ilvl):
        r = data.copy()
        if ilvl >= 86: m = 1.5
        elif ilvl >= 83: m = 1.2
        elif ilvl >= 80: m = 1.0
        elif ilvl >= 75: m = 0.5
        else: m = 0.2
        r["divine_value"] *= m
        r["chaos_value"] *= m
        return r

    def _fuzzy(self, a, b, t=0.85):
        if not a or not b:
            return False
        if abs(len(a) - len(b)) > max(len(a), len(b)) * 0.2:
            return False
        ac, bc = set(a), set(b)
        o = len(ac & bc)
        u = len(ac | bc)
        return (o / u) >= t if u > 0 else False

    # ─── Disk Cache ──────────────────────────────────

    def _save_to_disk(self):
        try:
            f = CACHE_DIR / f"prices_{self.league.lower().replace(' ', '_')}.json"
            with self._lock:
                d = {
                    "prices": self.prices,
                    "divine_to_chaos": self.divine_to_chaos,
                    "divine_to_exalted": self.divine_to_exalted,
                    "poe2scout_divine_price": self._poe2scout_divine_price,
                    "timestamp": time.time(),
                    "league": self.league,
                }
            with open(f, "w") as fh:
                json.dump(d, fh, indent=2)
        except Exception as e:
            logger.warning(f"Save failed: {e}")

    def _load_from_disk(self):
        try:
            f = CACHE_DIR / f"prices_{self.league.lower().replace(' ', '_')}.json"
            if not f.exists():
                return
            with open(f) as fh:
                d = json.load(fh)
            age = time.time() - d.get("timestamp", 0)
            if age > PRICE_REFRESH_INTERVAL * 4:
                return
            with self._lock:
                self.prices = d.get("prices", {})
                self.divine_to_chaos = d.get("divine_to_chaos", 68.0)
                self.divine_to_exalted = d.get("divine_to_exalted", 387.0)
                self._poe2scout_divine_price = d.get("poe2scout_divine_price", 0)
                self.last_refresh = d.get("timestamp", 0)
            logger.info(f"Loaded {len(self.prices)} from disk ({age:.0f}s old)")
        except Exception as e:
            logger.warning(f"Load failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    cache = PriceCache(league="Fate of the Vaal")
    print("Fetching prices (poe2scout + poe.ninja)...")
    cache._fetch_all()
    stats = cache.get_stats()
    print(f"\nStats: {stats}")

    # Test lookups
    print(f"\nSample prices:")
    test_items = [
        "Divine Orb", "Exalted Orb", "Chaos Orb", "Fracturing Orb",
        "Orb of Annulment", "Orb of Chance", "Vaal Orb",
        "Temporalis", "Headhunter", "The Adorned",
        "Uncut Skill Gem (Level 19)",
    ]
    for name in test_items:
        r = cache.lookup(name)
        if r:
            src = r.get("source", "?")
            print(f"  {name}: {r['display']}  ({r['name']}, {src})")
        else:
            print(f"  {name}: NOT FOUND")

    # Count by source
    sources = {}
    for data in cache.prices.values():
        src = data.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    print(f"\nItems by source: {sources}")
