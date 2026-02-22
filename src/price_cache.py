"""
LAMA - Price Cache
Fetches item prices from poe2scout.com (primary) and poe.ninja (secondary).

poe2scout provides pre-aggregated prices for 900+ items across all categories
(uniques, currencies, gems, maps, etc.). Prices are converted to divine values
using the league's divinePrice from the /leagues endpoint.

poe.ninja exchange endpoint provides conversion rates (divine→chaos, divine→exalted)
and serves as a fallback data source for currency-type items.
"""

import json
import shutil
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
    RATE_HISTORY_FILE,
    RATE_HISTORY_BACKUP,
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
    "SoulCores",      # includes Soul Cores + Theses (34 items)
    "Idols",
    "UncutGems",
    "LineageSupportGems",
    # "Ultimatum" intentionally omitted — duplicates SoulCores with stale pricing
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

# ─── Per-category source preference ──────────
# poe2scout categories where poe.ninja is more accurate and should NOT be
# overwritten.  For unlisted categories both sources agree well enough that
# poe2scout (fetched second) can overwrite poe.ninja.
NINJA_PREFERRED = {
    "currency",           # more items (45 vs 37), includes incursion currency
    "fragments",          # more items (20 vs 13), includes reliquary keys
    "runes",              # poe2scout has wild outliers (Body Rune 298ex vs 1.4ex)
    "ritual",             # poe.ninja more stable on mid-tier omens
    "idol",               # large mid-tier disagreements on poe2scout
    "lineagesupportgems", # poe.ninja more stable on mid/high-tier
    "ultimatum",          # soul cores — poe.ninja SoulCores type is authoritative
}

# ─── Name aliases: in-game clipboard name → API name ─────
# Some items have different names in-game vs on poe.ninja/poe2scout.
# Keys must be lowercase.
NAME_ALIASES = {
    # Delirium: in-game uses "Distilled X", APIs use "Liquid/Diluted Liquid X"
    "distilled ire": "diluted liquid ire",
    "distilled paranoia": "liquid paranoia",
    "distilled despair": "liquid despair",
    "distilled guilt": "diluted liquid guilt",
    "distilled greed": "diluted liquid greed",
    "distilled disgust": "liquid disgust",
    "distilled isolation": "concentrated liquid isolation",
    "distilled suffering": "concentrated liquid suffering",
    "distilled fear": "concentrated liquid fear",
    "distilled envy": "liquid envy",
}


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

            # Resolve name alias (e.g. in-game "Distilled Ire" → API "Diluted Liquid Ire")
            key = NAME_ALIASES.get(key, key)

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
            display = self._format_range(low_dv, high_dv)
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

    def _format_range(self, low_dv: float, high_dv: float) -> str:
        """Format a divine-value range with a single currency suffix.

        Uses the high value to pick denomination, then expresses both
        values in that unit so the overlay currency parser splits cleanly.
        """
        if high_dv >= 0.85:
            lo = f"{low_dv:.0f}" if low_dv >= 10 else f"{low_dv:.1f}"
            hi = f"{high_dv:.0f}" if high_dv >= 10 else f"{high_dv:.1f}"
            return f"~{lo}-{hi}d"
        ex_rate = self.divine_to_exalted
        if ex_rate > 0:
            low_ex = low_dv * ex_rate
            high_ex = high_dv * ex_rate
            if high_ex >= 1:
                lo = f"{low_ex:.0f}" if low_ex >= 10 else f"{low_ex:.1f}"
                hi = f"{high_ex:.0f}" if high_ex >= 10 else f"{high_ex:.1f}"
                return f"~{lo}-{hi}ex"
        low_c = low_dv * self.divine_to_chaos
        high_c = high_dv * self.divine_to_chaos
        lo = f"{low_c:.0f}" if low_c >= 10 else f"{low_c:.1f}"
        hi = f"{high_c:.0f}" if high_c >= 10 else f"{high_c:.1f}"
        return f"~{lo}-{hi}c"

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
            mirror = self.prices.get("mirror of kalandra", {})
            return {
                "total_items": len(self.prices),
                "divine_to_chaos": self.divine_to_chaos,
                "divine_to_exalted": self.divine_to_exalted,
                "mirror_to_divine": mirror.get("divine_value", 0),
                "last_refresh": time.strftime("%H:%M:%S", time.localtime(self.last_refresh)) if self.last_refresh else "Never",
                "league": self.league,
            }

    # ─── Market Data (Markets tab) ──────────────────────

    def get_market_data(self) -> dict:
        """Return all poe.ninja currencies with sparkline data for the Markets tab."""
        with self._lock:
            currencies = []
            for key, data in self.prices.items():
                # Include items that have sparkline data (from poe.ninja, possibly overwritten by poe2scout)
                if not data.get("sparkline_data"):
                    continue
                currencies.append({
                    "name": data.get("name", key),
                    "divine_value": data.get("divine_value", 0),
                    "chaos_value": data.get("chaos_value", 0),
                    "category": data.get("category", ""),
                    "sparkline_data": data.get("sparkline_data", []),
                    "sparkline_change": data.get("sparkline_change", 0),
                    "volume": data.get("volume", 0),
                    "image_url": data.get("image_url", ""),
                })

            # Sort expensive → cheap
            currencies.sort(key=lambda c: -c["divine_value"])

            rates = {
                "divine_to_chaos": self.divine_to_chaos,
                "divine_to_exalted": self.divine_to_exalted,
            }

            last_refresh = (
                time.strftime("%H:%M:%S", time.localtime(self.last_refresh))
                if self.last_refresh else "Never"
            )

        # Load rate history from disk (outside lock)
        history = self._load_rate_history()

        # Oldest timestamp so frontend can grey out 14d/30d when insufficient data
        oldest_history_ts = min((h["ts"] for h in history), default=0) if history else 0

        return {
            "currencies": currencies,
            "rates": rates,
            "history": history,
            "oldest_history_ts": oldest_history_ts,
            "last_refresh": last_refresh,
            "league": self.league,
        }

    def _track_rate_history(self):
        """Append a rate snapshot to the history file, pruning entries > 30 days old."""
        try:
            now = time.time()
            cutoff = now - 30 * 86400  # 30 days

            # Build snapshot of top currencies by volume
            with self._lock:
                top_currencies = {}
                items = sorted(
                    self.prices.values(),
                    key=lambda x: x.get("volume", 0),
                    reverse=True,
                )
                for item in items[:30]:
                    name = item.get("name", "")
                    if name:
                        top_currencies[name] = round(item.get("divine_value", 0), 6)

                entry = {
                    "ts": int(now),
                    "divine_to_chaos": round(self.divine_to_chaos, 2),
                    "divine_to_exalted": round(self.divine_to_exalted, 2),
                    "currencies": top_currencies,
                }

            # Read existing entries, prune old ones, append new
            existing = []
            if RATE_HISTORY_FILE.exists():
                try:
                    with open(RATE_HISTORY_FILE, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                                if rec.get("ts", 0) > cutoff:
                                    existing.append(line)
                            except json.JSONDecodeError:
                                continue
                except Exception:
                    pass

            existing.append(json.dumps(entry))

            with open(RATE_HISTORY_FILE, "w") as f:
                f.write("\n".join(existing) + "\n")

            # Backup to OneDrive
            try:
                RATE_HISTORY_BACKUP.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(RATE_HISTORY_FILE, RATE_HISTORY_BACKUP)
            except Exception as be:
                logger.debug(f"Rate history backup failed: {be}")

        except Exception as e:
            logger.debug(f"Rate history tracking failed: {e}")

    def _load_rate_history(self) -> list:
        """Load rate history entries from disk. Restores from OneDrive backup if primary missing."""
        history = []
        # Restore from backup if primary is missing
        if not RATE_HISTORY_FILE.exists() and RATE_HISTORY_BACKUP.exists():
            try:
                RATE_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(RATE_HISTORY_BACKUP, RATE_HISTORY_FILE)
                logger.info("Restored rate history from OneDrive backup")
            except Exception as e:
                logger.debug(f"Rate history restore failed: {e}")
        if not RATE_HISTORY_FILE.exists():
            return history
        try:
            cutoff = time.time() - 30 * 86400
            with open(RATE_HISTORY_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("ts", 0) > cutoff:
                            history.append(rec)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.debug(f"Failed to load rate history: {e}")
        return history

    # ─── Fetch ───────────────────────────────────────

    def _refresh_loop(self):
        while self._running:
            try:
                self._fetch_all()
                self._save_to_disk()
                self._track_rate_history()
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
        headers = {"User-Agent": "LAMA/1.0"}

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
        entry = {
            "divine_value": divine_value,
            "chaos_value": chaos_value,
            "name": name,
            "category": category,
            "source": "poe2scout",
        }
        # Check if poe.ninja already has this item and is the preferred source
        existing = prices.get(key)
        if existing and existing.get("source") == "poe.ninja":
            if category in NINJA_PREFERRED:
                # poe.ninja is authoritative for this category — keep it,
                # but still fill in items poe.ninja didn't have
                return 0
            # poe2scout overwrites price but preserves sparkline/image data
            for field in ("sparkline_data", "sparkline_change", "volume", "image_url"):
                if field in existing:
                    entry[field] = existing[field]
        prices[key] = entry
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
        Also preserves sparkline, volume, and image data for the Markets tab.
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

        # Build id → name map and id → image map
        id_map = {}
        image_map = {}
        for item in items:
            iid = item.get("id", "")
            name = item.get("name", "")
            if iid and name:
                id_map[iid] = name
            img = item.get("icon") or item.get("image")
            if iid and img:
                if img.startswith("/"):
                    img = "https://web.poecdn.com" + img
                image_map[iid] = img

        count = 0
        for line in lines:
            lid = line.get("id", "")
            divine_value = line.get("primaryValue", 0)
            name = id_map.get(lid, lid)

            if not name or divine_value == 0:
                continue

            # Extract sparkline data (7-day cumulative % changes)
            sparkline = line.get("sparkline", {}) or {}
            sparkline_data = sparkline.get("data") or []
            sparkline_change = sparkline.get("totalChange", 0)

            key = name.lower()
            prices[key] = {
                "divine_value": divine_value,
                "chaos_value": divine_value * self.divine_to_chaos,
                "name": name,
                "category": category,
                "source": "poe.ninja",
                "sparkline_data": sparkline_data,
                "sparkline_change": sparkline_change,
                "volume": line.get("volumePrimaryValue", 0),
                "image_url": image_map.get(lid, ""),
            }
            count += 1

        return count

    # ─── Display ─────────────────────────────────────

    def _enrich(self, data: dict) -> dict:
        """Add display string and tier.

        Tier is based on chaos value so it stays consistent with the loot
        filter updater regardless of how cheap exalted orbs are.
        """
        result = data.copy()
        dv = result.get("divine_value", 0)
        chaos = dv * self.divine_to_chaos

        # Calculate exalted value (for display only, NOT for tier)
        ex_rate = self.divine_to_exalted
        ev = dv * ex_rate if ex_rate > 0 else 0
        result["exalted_value"] = round(ev, 1)

        is_currency = "currency" in data.get("category", "").lower()

        # Tier based on chaos value (aligned with filter updater thresholds)
        #   high  = filter S  (>= 25c, divine+)
        #   good  = filter A  (>= 5c)
        #   decent = filter B/C (>= 1c)
        #   low   = filter D/E (< 1c)
        if chaos >= 25:
            result["tier"] = "high" if dv >= 5 else "good"
        elif chaos >= 5:
            result["tier"] = "good"
        elif chaos >= 1:
            result["tier"] = "decent"
        else:
            result["tier"] = "low"

        # Display string - shorthand format consistent with local scoring
        # POE2 economy: Divine > Exalted (base trade currency) > Chaos
        if dv >= 0.85:
            result["display"] = f"~{dv:.0f}d" if dv >= 10 else f"~{dv:.1f}d"
        elif ev >= 1:
            result["display"] = f"~{ev:.0f}ex" if ev >= 10 else f"~{ev:.1f}ex"
        elif chaos >= 1:
            result["display"] = f"~{chaos:.0f}c"
        else:
            result["display"] = "< 1c"

        # Currency-specific display cleanup
        if is_currency:
            name_lower = data.get("name", "").lower()
            # Avoid self-referential display (Divine Orb = "1.0d", Exalted = "1.0ex")
            if ("divine" in name_lower and result["display"].endswith("d")) or \
               ("exalted" in name_lower and result["display"].endswith("ex")):
                # Force chaos denomination
                if chaos >= 1:
                    result["display"] = f"{chaos:.0f}c"
                else:
                    result["display"] = "< 1c"
            else:
                # Strip ~ prefix for all currency
                result["display"] = result["display"].lstrip("~").lstrip(" ")

        return result

    def _adjust_ilvl(self, data, ilvl):
        r = data.copy()
        if ilvl >= 86:   m = 1.5
        elif ilvl >= 83: m = 1.3
        elif ilvl >= 80: m = 1.1
        elif ilvl >= 75: m = 0.9
        elif ilvl >= 68: m = 0.7
        elif ilvl >= 45: m = 0.4
        else:            m = 0.2
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
