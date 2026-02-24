"""
stash_client.py — Reads stash tabs via GGG's OAuth API.

Fetches stash tab metadata and item contents, converts API item JSON
to ParsedItem objects compatible with LAMA's scoring pipeline.

Rate limiting follows the same adaptive pattern as trade_client.py,
parsing X-Rate-Limit-* headers from responses.
"""

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable

import requests

from item_parser import ParsedItem

logger = logging.getLogger(__name__)

STASH_API_BASE = "https://api.pathofexile.com"

# frameType → rarity mapping for the stash API
FRAME_TYPE_MAP = {
    0: "normal",
    1: "magic",
    2: "rare",
    3: "unique",
    4: "gem",
    5: "currency",
    6: "currency",   # divination card
    7: "currency",   # quest item
    8: "currency",   # prophecy
    9: "currency",   # relic
}


@dataclass
class StashTab:
    """Metadata for a single stash tab."""
    id: str = ""
    name: str = ""
    type: str = ""
    colour: Optional[dict] = None  # {"r": int, "g": int, "b": int}
    index: int = 0


@dataclass
class StashItem:
    """A scored stash item with display metadata."""
    parsed: ParsedItem = field(default_factory=ParsedItem)
    icon_url: str = ""
    stack_size: int = 1
    note: str = ""            # player-set price note (e.g. "~price 5 divine")
    listed_price: float = 0   # extracted divine value from note
    tab_name: str = ""
    tab_id: str = ""


class StashClient:
    """Fetches stash tabs and items via OAuth-authenticated API calls."""

    def __init__(self, oauth_manager):
        self._oauth = oauth_manager
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "LAMA/1.0"})

        # Rate limiting (same pattern as TradeClient)
        self._last_request_time = 0.0
        self._rate_lock = threading.Lock()
        self._min_interval = 1.0  # ~1 req/s for stash endpoints
        self._rate_limited_until = 0.0

    def _rate_limit(self):
        """Enforce minimum interval between requests."""
        with self._rate_lock:
            now = time.time()

            # Respect 429 cooldown
            if now < self._rate_limited_until:
                wait = self._rate_limited_until - now
                time.sleep(wait)

            # Enforce minimum interval
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

            self._last_request_time = time.time()

    def _parse_rate_headers(self, resp: requests.Response):
        """Parse X-Rate-Limit-* headers and adjust pacing."""
        # X-Rate-Limit-Ip: 45:60:60,240:240:900
        # X-Rate-Limit-Ip-State: 1:60:0,1:240:0
        state = resp.headers.get("X-Rate-Limit-Ip-State", "")
        rules = resp.headers.get("X-Rate-Limit-Ip", "")
        if not state or not rules:
            return

        try:
            state_parts = state.split(",")
            rule_parts = rules.split(",")
            for sp, rp in zip(state_parts, rule_parts):
                s_fields = sp.split(":")
                r_fields = rp.split(":")
                current_hits = int(s_fields[0])
                max_hits = int(r_fields[0])
                # If we're at 80%+ of the limit, slow down
                if current_hits > max_hits * 0.8:
                    self._min_interval = max(self._min_interval, 2.0)
                    logger.debug(f"Stash rate limit pressure: {current_hits}/{max_hits}, slowing to {self._min_interval}s")
        except (ValueError, IndexError):
            pass

    def _get(self, url: str) -> Optional[requests.Response]:
        """Make an authenticated GET request with rate limiting."""
        headers = self._oauth.get_headers()
        if not headers:
            logger.warning("StashClient: not authenticated")
            return None

        self._rate_limit()

        try:
            resp = self._session.get(url, headers=headers, timeout=30)

            self._parse_rate_headers(resp)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                self._rate_limited_until = time.time() + retry_after
                logger.warning(f"Stash API rate limited, waiting {retry_after}s")
                return None

            if resp.status_code == 401:
                logger.warning("Stash API: unauthorized (token may be expired)")
                return None

            if resp.status_code != 200:
                logger.warning(f"Stash API error: HTTP {resp.status_code}")
                return None

            return resp

        except requests.RequestException as e:
            logger.error(f"Stash API request failed: {e}")
            return None

    def list_tabs(self, league: str) -> List[StashTab]:
        """Fetch all stash tab metadata for a league."""
        url = f"{STASH_API_BASE}/stash/poe2/{league}"
        resp = self._get(url)
        if not resp:
            return []

        try:
            data = resp.json()
            tabs = []
            for i, tab_data in enumerate(data.get("stashes", [])):
                tabs.append(StashTab(
                    id=tab_data.get("id", ""),
                    name=tab_data.get("name", f"Tab {i+1}"),
                    type=tab_data.get("type", ""),
                    colour=tab_data.get("colour"),
                    index=i,
                ))
            logger.info(f"Fetched {len(tabs)} stash tabs for {league}")
            return tabs
        except Exception as e:
            logger.error(f"Failed to parse stash tab list: {e}")
            return []

    def get_tab_items(self, league: str, stash_id: str) -> List[dict]:
        """Fetch items from a specific stash tab. Returns raw API JSON items."""
        url = f"{STASH_API_BASE}/stash/poe2/{league}/{stash_id}"
        resp = self._get(url)
        if not resp:
            return []

        try:
            data = resp.json()
            stash = data.get("stash", {})
            items = stash.get("items", [])
            # Handle sub-stashes (e.g., quad tabs with children)
            children = stash.get("children", [])
            for child in children:
                items.extend(child.get("items", []))
            return items
        except Exception as e:
            logger.error(f"Failed to parse stash items: {e}")
            return []

    def fetch_all_tabs(self, league: str,
                       progress_cb: Optional[Callable] = None) -> List[tuple]:
        """Fetch all tabs and their items.

        Args:
            league: League name
            progress_cb: Optional callback(tab_name, done, total)

        Returns:
            List of (StashTab, [StashItem, ...]) tuples
        """
        tabs = self.list_tabs(league)
        if not tabs:
            return []

        results = []
        for i, tab in enumerate(tabs):
            if progress_cb:
                progress_cb(tab.name, i, len(tabs))

            raw_items = self.get_tab_items(league, tab.id)
            stash_items = []
            for item_json in raw_items:
                parsed = self.api_item_to_parsed(item_json)
                if parsed:
                    si = StashItem(
                        parsed=parsed,
                        icon_url=item_json.get("icon", ""),
                        stack_size=item_json.get("stackSize", 1),
                        note=item_json.get("note", ""),
                        listed_price=self._extract_price_from_note(item_json.get("note", "")),
                        tab_name=tab.name,
                        tab_id=tab.id,
                    )
                    stash_items.append(si)

            results.append((tab, stash_items))
            logger.debug(f"Tab '{tab.name}': {len(stash_items)} items")

        if progress_cb:
            progress_cb("Done", len(tabs), len(tabs))

        return results

    @staticmethod
    def api_item_to_parsed(item_json: dict) -> Optional[ParsedItem]:
        """Convert a stash API item JSON object to a ParsedItem.

        Maps API fields directly to ParsedItem fields (Option B from plan).
        """
        if not item_json:
            return None

        frame_type = item_json.get("frameType", 0)
        rarity = FRAME_TYPE_MAP.get(frame_type, "normal")

        name = item_json.get("name", "")
        type_line = item_json.get("typeLine", "")
        base_type = item_json.get("baseType", type_line)

        # For unique items, name is the unique name, base_type is the base
        # For rare items, name is the random name, type_line/base_type is the base
        if rarity in ("currency", "normal", "gem") and not name:
            name = type_line

        item = ParsedItem(
            name=name or type_line,
            base_type=base_type or type_line,
            rarity=rarity,
            item_level=item_json.get("ilvl", 0),
            quality=0,
            sockets=0,
            stack_size=item_json.get("stackSize", 1),
            unidentified=not item_json.get("identified", True),
            corrupted=item_json.get("corrupted", False),
        )

        # Item class from extended info
        extended = item_json.get("extended", {})
        if extended.get("category"):
            item.item_class = extended["category"]

        # Parse properties for quality, sockets, weapon/defense stats
        for prop in item_json.get("properties", []):
            prop_name = prop.get("name", "")
            values = prop.get("values", [])

            if prop_name == "Quality" and values:
                q_str = values[0][0] if values[0] else ""
                q_match = re.search(r"(\d+)", q_str)
                if q_match:
                    item.quality = int(q_match.group(1))

            elif prop_name == "Physical Damage" and values:
                dmg_str = values[0][0] if values[0] else ""
                m = re.match(r"(\d+)-(\d+)", dmg_str)
                if m:
                    item.physical_damage = (int(m.group(1)), int(m.group(2)))

            elif prop_name in ("Elemental Damage", "Fire Damage", "Cold Damage",
                               "Lightning Damage", "Chaos Damage") and values:
                dmg_str = values[0][0] if values[0] else ""
                m = re.match(r"(\d+)-(\d+)", dmg_str)
                if m:
                    item.elemental_damages.append((int(m.group(1)), int(m.group(2))))

            elif prop_name == "Attacks per Second" and values:
                try:
                    item.attacks_per_second = float(values[0][0])
                except (ValueError, IndexError):
                    pass

            elif prop_name == "Armour" and values:
                try:
                    item.armour = int(values[0][0])
                except (ValueError, IndexError):
                    pass

            elif prop_name == "Evasion Rating" and values:
                try:
                    item.evasion = int(values[0][0])
                except (ValueError, IndexError):
                    pass

            elif prop_name == "Energy Shield" and values:
                try:
                    item.energy_shield = int(values[0][0])
                except (ValueError, IndexError):
                    pass

        # Sockets (POE2 format)
        sockets = item_json.get("sockets", [])
        if sockets:
            item.sockets = len(sockets)

        # Compute DPS
        if item.physical_damage and item.attacks_per_second > 0:
            avg = (item.physical_damage[0] + item.physical_damage[1]) / 2
            item.physical_dps = avg * item.attacks_per_second
        if item.elemental_damages and item.attacks_per_second > 0:
            for lo, hi in item.elemental_damages:
                item.elemental_dps += ((lo + hi) / 2) * item.attacks_per_second
        item.total_dps = item.physical_dps + item.elemental_dps
        item.total_defense = item.armour + item.evasion + item.energy_shield

        # Parse mods — combine implicit + explicit
        mods = []
        for mod_text in item_json.get("implicitMods", []):
            mods.append(("implicit", mod_text))
        for mod_text in item_json.get("explicitMods", []):
            mods.append(("explicit", mod_text))
        for mod_text in item_json.get("enchantMods", []):
            mods.append(("enchant", mod_text))
        for mod_text in item_json.get("craftedMods", []):
            mods.append(("crafted", mod_text))
        for mod_text in item_json.get("fracturedMods", []):
            mods.append(("fractured", mod_text))
        item.mods = mods

        return item

    @staticmethod
    def _extract_price_from_note(note: str) -> float:
        """Extract listed price in divine from a stash note.

        Formats: ~price 5 divine, ~b/o 3 divine, ~price 100 chaos
        """
        if not note:
            return 0.0
        m = re.match(r"~(?:price|b/o)\s+([\d.]+)\s+(\w+)", note)
        if not m:
            return 0.0
        amount = float(m.group(1))
        currency = m.group(2).lower()
        if currency in ("divine", "divines"):
            return amount
        # Other currencies would need conversion — return 0 for now
        return 0.0
