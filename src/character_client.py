"""
character_client.py — GGG Character API client for OAuth-authenticated users.

Fetches character lists and full character profiles from the official POE2
API using OAuth tokens. Converts GGG character JSON into CharacterData
objects compatible with the existing builds_client infrastructure.

Endpoints used:
  GET /character/poe2                → list all characters
  GET /character/poe2/{name}         → full character profile (equipment, skills)
"""

import logging
import threading
import time
from typing import List, Optional

import requests

from builds_client import (
    ASCENDANCY_MAP, CharacterData, CharacterItem, SkillGroup,
)
from oauth import GGG_USER_AGENT
from stash_client import FRAME_TYPE_MAP

logger = logging.getLogger(__name__)

CHARACTER_API_BASE = "https://api.pathofexile.com"

# Cache TTLs (seconds)
TTL_LIST = 60       # character list
TTL_CHARACTER = 300  # individual character


class CharacterClient:
    """Fetches characters and equipment via GGG's OAuth-authenticated API."""

    def __init__(self, oauth_manager):
        self._oauth = oauth_manager
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": GGG_USER_AGENT})

        # Cache
        self._cache: dict = {}  # key → (data, timestamp)
        self._cache_lock = threading.Lock()

        # Rate limiting
        self._last_request_time = 0.0
        self._rate_lock = threading.Lock()
        self._min_interval = 1.0
        self._rate_limited_until = 0.0

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _get_cached(self, key: str, ttl: int):
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry and (time.time() - entry[1]) < ttl:
                return entry[0]
            return None

    def _set_cache(self, key: str, data):
        with self._cache_lock:
            self._cache[key] = (data, time.time())

    # ------------------------------------------------------------------
    # HTTP helpers (mirrors stash_client.py patterns)
    # ------------------------------------------------------------------
    def _rate_limit(self):
        with self._rate_lock:
            now = time.time()
            if now < self._rate_limited_until:
                time.sleep(self._rate_limited_until - now)
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_request_time = time.time()

    def _parse_rate_headers(self, resp: requests.Response):
        state = resp.headers.get("X-Rate-Limit-Ip-State", "")
        rules = resp.headers.get("X-Rate-Limit-Ip", "")
        if not state or not rules:
            return
        try:
            for sp, rp in zip(state.split(","), rules.split(",")):
                current = int(sp.split(":")[0])
                maximum = int(rp.split(":")[0])
                if current > maximum * 0.8:
                    self._min_interval = max(self._min_interval, 2.0)
        except (ValueError, IndexError):
            pass

    def _get(self, url: str) -> Optional[requests.Response]:
        headers = self._oauth.get_headers()
        if not headers:
            logger.warning("CharacterClient: not authenticated")
            return None

        self._rate_limit()

        try:
            resp = self._session.get(url, headers=headers, timeout=30)
            self._parse_rate_headers(resp)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                self._rate_limited_until = time.time() + retry_after
                logger.warning(f"Character API rate limited, waiting {retry_after}s")
                return None
            if resp.status_code == 401:
                logger.warning("Character API: unauthorized (token may be expired)")
                return None
            if resp.status_code != 200:
                logger.warning(f"Character API error: HTTP {resp.status_code}")
                return None
            return resp

        except requests.RequestException as e:
            logger.error(f"Character API request failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def list_characters(self) -> List[dict]:
        """List all characters for the authenticated account.

        Returns list of {name, class, league, level}.
        """
        cached = self._get_cached("char-list", TTL_LIST)
        if cached is not None:
            return cached

        resp = self._get(f"{CHARACTER_API_BASE}/character/poe2")
        if not resp:
            return []

        try:
            data = resp.json()
            characters = data.get("characters", data) if isinstance(data, dict) else data
            if not isinstance(characters, list):
                return []

            result = []
            for ch in characters:
                asc = ch.get("class", "")
                base_class = ASCENDANCY_MAP.get(asc, asc)
                result.append({
                    "name": ch.get("name", ""),
                    "class": asc,
                    "baseClass": base_class,
                    "league": ch.get("league", ""),
                    "level": ch.get("level", 0),
                })

            self._set_cache("char-list", result)
            logger.info(f"Fetched {len(result)} characters from GGG API")
            return result

        except Exception as e:
            logger.error(f"Failed to parse character list: {e}")
            return []

    def get_character(self, name: str) -> Optional[CharacterData]:
        """Fetch a character by name and convert to CharacterData.

        Returns CharacterData or None on failure.
        """
        cache_key = f"char-{name}"
        cached = self._get_cached(cache_key, TTL_CHARACTER)
        if cached is not None:
            return cached

        resp = self._get(f"{CHARACTER_API_BASE}/character/poe2/{name}")
        if not resp:
            return None

        try:
            data = resp.json()
            char = data.get("character", data) if isinstance(data, dict) else data
            result = self._parse_character(char)
            if result:
                self._set_cache(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Failed to parse character {name}: {e}")
            return None

    def get_character_raw(self, name: str) -> Optional[dict]:
        """Fetch a character by name and return raw GGG JSON.

        Needed for Price My Gear which processes items via api_item_to_parsed.
        """
        cache_key = f"char-raw-{name}"
        cached = self._get_cached(cache_key, TTL_CHARACTER)
        if cached is not None:
            return cached

        resp = self._get(f"{CHARACTER_API_BASE}/character/poe2/{name}")
        if not resp:
            return None

        try:
            data = resp.json()
            self._set_cache(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Failed to fetch raw character {name}: {e}")
            return None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _parse_character(self, data: dict) -> Optional[CharacterData]:
        """Convert GGG character API response into CharacterData."""
        try:
            asc_name = data.get("class", "")
            base_class = ASCENDANCY_MAP.get(asc_name, asc_name)

            # Equipment
            equipment = []
            for item in data.get("equipment", []):
                eq = self._parse_equipment_item(item)
                if eq:
                    equipment.append(eq)

            # Also check "items" key (GGG API may use either)
            for item in data.get("items", []):
                eq = self._parse_equipment_item(item)
                if eq:
                    equipment.append(eq)

            # Skills — GGG API returns skill gem info but no DPS
            skill_groups = []
            for sg in data.get("skills", []):
                gems = []
                for gem in sg.get("gems", []):
                    gem_name = gem.get("name", "") or gem.get("baseType", "")
                    if gem_name:
                        gems.append(gem_name)
                if gems:
                    skill_groups.append(SkillGroup(gems=gems, dps=[]))

            # Keystones — GGG returns hash IDs, not names (no passive tree
            # lookup available), so we leave empty
            keystones = []

            return CharacterData(
                account=data.get("account", {}).get("name", "") if isinstance(data.get("account"), dict) else data.get("account", ""),
                name=data.get("name", ""),
                char_class=base_class,
                ascendancy=asc_name,
                level=data.get("level", 0),
                equipment=equipment,
                skill_groups=skill_groups,
                keystones=keystones,
            )

        except Exception as e:
            logger.error(f"Failed to parse character data: {e}")
            return None

    @staticmethod
    def _parse_equipment_item(item: dict) -> Optional[CharacterItem]:
        """Convert a single GGG API equipment item to CharacterItem."""
        if not item:
            return None

        frame_type = item.get("frameType", 0)
        rarity_map = {0: "normal", 1: "magic", 2: "rare", 3: "unique"}
        rarity = rarity_map.get(frame_type, FRAME_TYPE_MAP.get(frame_type, "normal"))

        slot = item.get("inventoryId", "")
        if not slot:
            return None

        return CharacterItem(
            name=item.get("name", ""),
            type_line=item.get("typeLine", ""),
            slot=slot,
            rarity=rarity,
            sockets=item.get("sockets", []) or [],
            implicit_mods=item.get("implicitMods", []) or [],
            explicit_mods=item.get("explicitMods", []) or [],
            crafted_mods=item.get("craftedMods", []) or [],
            enchant_mods=item.get("enchantMods", []) or [],
            fractured_mods=item.get("fracturedMods", []) or [],
            desecrated_mods=item.get("desecratedMods", []) or [],
            rune_mods=item.get("runeMods", []) or [],
        )
