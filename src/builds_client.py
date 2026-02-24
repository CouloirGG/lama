"""
builds_client.py — poe.ninja Builds API client for character lookup.

Fetches character profiles (equipment, skills, keystones) from poe.ninja's
POE2 builds API. Also fetches popular items per slot via the protobuf-based
search/dictionary endpoints. Ported from lama-mobile.

Endpoints used:
  GET /poe2/api/data/index-state          → snapshot version info
  GET /poe2/api/builds/{ver}/character    → full character profile
  GET /poe2/api/builds/{ver}/search       → popular items search (protobuf)
  GET /poe2/api/builds/dictionary/{hash}  → item name dictionary (protobuf)
"""

import logging
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://poe.ninja/poe2/api"
HEADERS = {"User-Agent": "LAMA/1.0"}

# Cache TTLs (seconds)
TTL_SNAPSHOT = 3600     # 1 hour
TTL_CHARACTER = 300     # 5 minutes
TTL_SEARCH = 600        # 10 minutes
TTL_DICT = 600          # 10 minutes
TTL_PRICES = 900        # 15 minutes

POE2SCOUT_API = "https://poe2scout.com/api"

# Ascendancy → base class mapping
ASCENDANCY_MAP = {
    "Blood Mage": "Witch",
    "Oracle": "Witch",
    "Pathfinder": "Ranger",
    "Deadeye": "Ranger",
    "Titan": "Warrior",
    "Warbringer": "Warrior",
    "Stormweaver": "Sorceress",
    "Invoker": "Sorceress",
    "Amazon": "Huntress",
    "Ritualist": "Huntress",
    "Witchhunter": "Mercenary",
    "Gemling Legionnaire": "Mercenary",
    "Disciple of Varashta": "Monk",
    "Acolyte of Chayula": "Monk",
    "Lich": "Druid",
    "Shaman": "Druid",
}

# Equipment slot display names (inventoryId → readable)
SLOT_DISPLAY = {
    "Helm": "Helmet",
    "BodyArmour": "Body Armour",
    "Gloves": "Gloves",
    "Boots": "Boots",
    "Weapon": "Weapon",
    "Weapon2": "Weapon (Swap)",
    "Offhand": "Offhand",
    "Offhand2": "Offhand (Swap)",
    "Belt": "Belt",
    "Amulet": "Amulet",
    "Ring": "Ring",
    "Ring2": "Ring 2",
    "Flask": "Flask",
}

# Mod type display order and colors (for dashboard rendering)
MOD_TYPES = ["enchantMods", "implicitMods", "explicitMods", "fracturedMods",
             "craftedMods", "desecratedMods", "runeMods"]

# Slot → dictionary "type" values for filtering popular items
SLOT_TO_DICT_TYPE = {
    "Weapon": ["Weapon", "Staff", "Wand", "Sceptre", "Bow", "Crossbow"],
    "Weapon2": ["Weapon", "Staff", "Wand", "Sceptre"],
    "Offhand": ["Shield", "Quiver", "Focus"],
    "Helm": ["Helmet"],
    "BodyArmour": ["Body Armour"],
    "Body Armour": ["Body Armour"],
    "Gloves": ["Gloves"],
    "Boots": ["Boots"],
    "Belt": ["Belt"],
    "Amulet": ["Amulet"],
    "Ring": ["Ring"],
    "Ring2": ["Ring"],
    "Shield": ["Shield"],
    "Flask": ["Flask"],
}

# Slot → poe2scout unique item slug
SLOT_TO_UNIQUE_SLUG = {
    "Helm": "armour", "Helmet": "armour",
    "BodyArmour": "armour", "Body Armour": "armour",
    "Gloves": "armour", "Boots": "armour",
    "Shield": "armour", "Offhand": "armour",
    "Belt": "accessory", "Amulet": "accessory",
    "Ring": "accessory", "Ring2": "accessory",
    "Weapon": "weapon", "Weapon2": "weapon",
    "Flask": "flask",
}


# ---------------------------------------------------------------------------
# Minimal protobuf decoder (wire types 0=varint, 2=length-delimited)
# ---------------------------------------------------------------------------
def _decode_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    """Decode a varint from buf at pos. Returns (value, new_pos)."""
    result = 0
    shift = 0
    while pos < len(buf):
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return result, pos
        shift += 7
    return result, pos


def _decode_fields(buf: bytes) -> List[dict]:
    """Decode top-level protobuf fields from a buffer."""
    fields = []
    pos = 0
    while pos < len(buf):
        tag, pos = _decode_varint(buf, pos)
        wire_type = tag & 0x07
        field_number = tag >> 3

        if wire_type == 0:  # varint
            value, pos = _decode_varint(buf, pos)
            fields.append({"fieldNumber": field_number, "wireType": 0, "value": value})
        elif wire_type == 2:  # length-delimited
            length, pos = _decode_varint(buf, pos)
            data = buf[pos:pos + length]
            pos += length
            fields.append({"fieldNumber": field_number, "wireType": 2, "data": data})
        elif wire_type == 5:  # 32-bit fixed
            pos += 4
        elif wire_type == 1:  # 64-bit fixed
            pos += 8
        else:
            break  # unknown wire type
    return fields


def _field_as_string(f: dict) -> str:
    """Interpret a length-delimited field as UTF-8 string."""
    return f.get("data", b"").decode("utf-8", errors="replace")


def _field_as_message(f: dict) -> List[dict]:
    """Interpret a length-delimited field as a nested message."""
    return _decode_fields(f.get("data", b""))


@dataclass
class CharacterItem:
    """An equipped item with full mod data."""
    name: str = ""
    type_line: str = ""
    slot: str = ""
    rarity: str = ""
    sockets: list = field(default_factory=list)
    implicit_mods: list = field(default_factory=list)
    explicit_mods: list = field(default_factory=list)
    crafted_mods: list = field(default_factory=list)
    enchant_mods: list = field(default_factory=list)
    fractured_mods: list = field(default_factory=list)
    desecrated_mods: list = field(default_factory=list)
    rune_mods: list = field(default_factory=list)


@dataclass
class SkillGroupDps:
    """DPS info for a single skill."""
    name: str = ""
    dps: float = 0
    dot_dps: float = 0
    damage: float = 0  # per-hit damage


@dataclass
class SkillGroup:
    """A group of linked gems with DPS data."""
    gems: list = field(default_factory=list)       # gem names
    dps: list = field(default_factory=list)         # List[SkillGroupDps]


@dataclass
class CharacterData:
    """Full character profile from poe.ninja."""
    account: str = ""
    name: str = ""
    char_class: str = ""        # base class (Witch, Warrior, etc.)
    ascendancy: str = ""        # ascendancy name (Blood Mage, Titan, etc.)
    level: int = 0
    equipment: list = field(default_factory=list)   # List[CharacterItem]
    skill_groups: list = field(default_factory=list) # List[SkillGroup]
    keystones: list = field(default_factory=list)    # List[str]
    pob_code: str = ""


@dataclass
class PopularItem:
    """A popular item entry from poe.ninja builds search."""
    name: str = ""
    count: int = 0
    percentage: float = 0.0
    rarity: str = ""        # "unique", "rare", "magic", "normal"
    price_text: str = ""    # e.g. "~2.5 div", "~150c"


class BuildsClient:
    """Fetches character data from poe.ninja Builds API."""

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(HEADERS)
        self._cache: Dict[str, tuple] = {}  # key → (data, timestamp)
        self._lock = threading.Lock()
        self._snapshot_version: Optional[str] = None
        self._snapshot_name: Optional[str] = None

    def _get_cached(self, key: str, ttl: int) -> Optional[Any]:
        """Check cache. Returns data or None if expired/missing."""
        with self._lock:
            entry = self._cache.get(key)
            if entry and (time.time() - entry[1]) < ttl:
                return entry[0]
            return None

    def _set_cache(self, key: str, data: Any):
        with self._lock:
            self._cache[key] = (data, time.time())

    def _fetch_snapshot_info(self) -> bool:
        """Fetch current snapshot version + name. Returns True on success."""
        if self._snapshot_version and self._snapshot_name:
            cached = self._get_cached("snapshot", TTL_SNAPSHOT)
            if cached:
                return True

        try:
            resp = self._session.get(f"{BASE_URL}/data/index-state", timeout=10)
            if resp.status_code != 200:
                logger.warning(f"poe.ninja index-state: HTTP {resp.status_code}")
                return False

            data = resp.json()
            snapshots = data.get("snapshotVersions", [])
            economy_leagues = data.get("economyLeagues", [])

            primary_url = economy_leagues[0]["url"] if economy_leagues else None
            snapshot = None
            if primary_url:
                snapshot = next((s for s in snapshots if s.get("url") == primary_url), None)
            if not snapshot and snapshots:
                snapshot = snapshots[0]

            if not snapshot:
                logger.warning("poe.ninja: no snapshot found")
                return False

            self._snapshot_version = snapshot["version"]
            self._snapshot_name = snapshot["snapshotName"]
            self._set_cache("snapshot", True)
            logger.debug(f"poe.ninja snapshot: v={self._snapshot_version}, name={self._snapshot_name}")
            return True

        except Exception as e:
            logger.warning(f"poe.ninja snapshot fetch failed: {e}")
            return False

    def lookup_character(self, account: str, character: str) -> Optional[CharacterData]:
        """Look up a character by account + name.

        Returns CharacterData or None on failure.
        """
        if not account or not character:
            return None

        # Ensure we have snapshot info
        if not self._fetch_snapshot_info():
            return None

        # poe.ninja uses "-" instead of "#" for discriminators
        normalized_account = account.replace("#", "-")
        cache_key = f"char-{normalized_account}-{character}"
        cached = self._get_cached(cache_key, TTL_CHARACTER)
        if cached:
            return cached

        try:
            url = (
                f"{BASE_URL}/builds/{self._snapshot_version}/character"
                f"?account={normalized_account}"
                f"&name={character}"
                f"&overview={self._snapshot_name}"
            )
            resp = self._session.get(url, timeout=15)

            if resp.status_code == 404:
                logger.info(f"Character not found: {account}/{character}")
                return None
            if resp.status_code != 200:
                logger.warning(f"poe.ninja character: HTTP {resp.status_code}")
                return None

            data = resp.json()
            result = self._parse_character(data, normalized_account, character)
            if result:
                self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"Character lookup failed: {e}")
            return None

    def _parse_character(self, data: dict, account: str, char_name: str) -> Optional[CharacterData]:
        """Parse poe.ninja character response into CharacterData."""
        try:
            # Equipment
            equipment = []
            for item in data.get("items", []):
                idata = item.get("itemData", item)
                eq = CharacterItem(
                    name=idata.get("name", ""),
                    type_line=idata.get("typeLine", ""),
                    slot=idata.get("inventoryId", "") or idata.get("slot", ""),
                    rarity=idata.get("rarity", ""),
                    sockets=idata.get("sockets", []) or [],
                    implicit_mods=self._to_str_list(idata.get("implicitMods")),
                    explicit_mods=self._to_str_list(idata.get("explicitMods")),
                    crafted_mods=self._to_str_list(idata.get("craftedMods")),
                    enchant_mods=self._to_str_list(idata.get("enchantMods")),
                    fractured_mods=self._to_str_list(idata.get("fracturedMods")),
                    desecrated_mods=self._to_str_list(idata.get("desecratedMods")),
                    rune_mods=self._to_str_list(idata.get("runeMods")),
                )
                equipment.append(eq)

            # Skill groups
            skill_groups = []
            for sg in data.get("skills", []):
                all_gems = sg.get("allGems", [])
                dps_arr = sg.get("dps", [])
                group = SkillGroup(
                    gems=[g.get("name", "") for g in all_gems if g.get("name")],
                    dps=[
                        SkillGroupDps(
                            name=d.get("name", ""),
                            dps=d.get("dps", 0) or 0,
                            dot_dps=d.get("dotDps", 0) or 0,
                            damage=(d.get("damage", [0]) or [0])[0] if isinstance(d.get("damage"), list) else d.get("dps", 0),
                        )
                        for d in dps_arr
                    ],
                )
                skill_groups.append(group)

            # Keystones
            keystones = []
            for k in data.get("keystones", []):
                if isinstance(k, str):
                    keystones.append(k)
                elif isinstance(k, dict) and "name" in k:
                    keystones.append(k["name"])

            # Class/ascendancy
            asc_name = data.get("class", "")
            base_class = ASCENDANCY_MAP.get(asc_name, asc_name)

            return CharacterData(
                account=data.get("account", account),
                name=data.get("name", char_name),
                char_class=base_class,
                ascendancy=asc_name,
                level=data.get("level", 0),
                equipment=equipment,
                skill_groups=skill_groups,
                keystones=keystones,
                pob_code=data.get("pathOfBuildingExport", "") or "",
            )

        except Exception as e:
            logger.error(f"Failed to parse character data: {e}")
            return None

    @staticmethod
    def _to_str_list(val) -> list:
        """Safely convert a value to a list of strings."""
        if not isinstance(val, list):
            return []
        return [s for s in val if isinstance(s, str)]

    def serialize_character(self, char: CharacterData) -> dict:
        """Convert CharacterData to JSON-serializable dict for the API."""
        return {
            "account": char.account,
            "name": char.name,
            "class": char.char_class,
            "ascendancy": char.ascendancy,
            "level": char.level,
            "equipment": [
                {
                    "name": eq.name,
                    "typeLine": eq.type_line,
                    "slot": eq.slot,
                    "slotDisplay": SLOT_DISPLAY.get(eq.slot, eq.slot),
                    "rarity": eq.rarity,
                    "sockets": eq.sockets,
                    "implicitMods": eq.implicit_mods,
                    "explicitMods": eq.explicit_mods,
                    "craftedMods": eq.crafted_mods,
                    "enchantMods": eq.enchant_mods,
                    "fracturedMods": eq.fractured_mods,
                    "desecratedMods": eq.desecrated_mods,
                    "runeMods": eq.rune_mods,
                }
                for eq in char.equipment
            ],
            "skillGroups": [
                {
                    "gems": sg.gems,
                    "dps": [
                        {"name": d.name, "dps": d.dps, "dotDps": d.dot_dps, "damage": d.damage}
                        for d in sg.dps
                    ],
                }
                for sg in char.skill_groups
            ],
            "keystones": char.keystones,
            "pobCode": char.pob_code,
        }

    # -------------------------------------------------------------------
    # Popular items search (protobuf-based search + dictionary protocol)
    # -------------------------------------------------------------------

    def fetch_popular_items(self, char_class: str, skill: str,
                            slot: str) -> List[PopularItem]:
        """Fetch popular items for a slot from poe.ninja builds search.

        Uses the protobuf search API filtered by class + skill, then resolves
        item names via the dictionary endpoint.

        Returns list of PopularItem sorted by count descending (top 20).
        """
        if not self._fetch_snapshot_info():
            return []

        # Check cache
        cache_key = f"popular-{self._snapshot_version}-{char_class}-{skill}-{slot}"
        cached = self._get_cached(cache_key, TTL_SEARCH)
        if cached is not None:
            return cached

        try:
            search = self._fetch_search(char_class, skill)
            if not search:
                return []

            # Find the "items" dimension
            items_dim = None
            for dim in search["dimensions"]:
                if dim["name"] == "items":
                    items_dim = dim
                    break
            if not items_dim:
                return []

            # Get dictionary hash for "item" type
            dict_hash = (search["dictHashes"].get("item") or
                         search["dictHashes"].get(items_dim["displayName"]))
            if not dict_hash:
                return []

            # Fetch dictionary
            dictionary = self._fetch_dictionary(dict_hash)
            if not dictionary:
                return []

            # Get type + color metadata columns
            type_col = dictionary["metadata"].get("type", [])
            color_col = dictionary["metadata"].get("color", [])

            # Filter by slot type
            valid_types = SLOT_TO_DICT_TYPE.get(slot)
            if not valid_types:
                logger.warning(f"No type mapping for slot '{slot}'")
                return []

            valid_indices = set()
            for i, name in enumerate(dictionary["names"]):
                item_type = type_col[i] if i < len(type_col) else ""
                if item_type in valid_types:
                    valid_indices.add(i)

            # Build popular items list
            total_count = search["totalCount"] or 1
            items = []
            for entry in items_dim["entries"]:
                key = entry["key"]
                if key not in valid_indices:
                    continue
                if key >= len(dictionary["names"]):
                    continue
                name = dictionary["names"][key]
                if not name:
                    continue

                rarity = ""
                if key < len(color_col):
                    rarity = self._parse_rarity(color_col[key])

                items.append(PopularItem(
                    name=name,
                    count=entry["count"],
                    percentage=(entry["count"] / total_count) * 100,
                    rarity=rarity,
                ))

            # Sort by count descending, take top 20
            items.sort(key=lambda x: x.count, reverse=True)
            items = items[:20]

            self._set_cache(cache_key, items)
            return items

        except Exception as e:
            logger.warning(f"Popular items fetch failed: {e}")
            return []

    def _fetch_search(self, char_class: str, skill: str) -> Optional[dict]:
        """Fetch search results (protobuf) from poe.ninja."""
        cache_key = f"search-{self._snapshot_version}-{char_class}-{skill}"
        cached = self._get_cached(cache_key, TTL_SEARCH)
        if cached is not None:
            return cached

        try:
            url = (
                f"{BASE_URL}/builds/{quote(self._snapshot_version)}/search"
                f"?overview={quote(self._snapshot_name)}"
                f"&class={quote(char_class)}"
                f"&skills={quote(skill)}"
            )
            resp = self._session.get(url, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"poe.ninja search: HTTP {resp.status_code}")
                return None

            buf = resp.content
            top_fields = _decode_fields(buf)

            # Field 1 is the outer wrapper
            wrapper = None
            for f in top_fields:
                if f["fieldNumber"] == 1 and f["wireType"] == 2:
                    wrapper = f
                    break
            if not wrapper:
                return None

            inner_fields = _field_as_message(wrapper)

            total_count = 0
            dimensions = []
            dict_hashes = {}

            for f in inner_fields:
                if f["fieldNumber"] == 1 and f["wireType"] == 0:
                    total_count = f["value"]
                elif f["fieldNumber"] == 2 and f["wireType"] == 2:
                    # Dimension message
                    dim_fields = _field_as_message(f)
                    name = ""
                    display_name = ""
                    entries = []

                    for df in dim_fields:
                        if df["fieldNumber"] == 1 and df["wireType"] == 2:
                            name = _field_as_string(df)
                        elif df["fieldNumber"] == 2 and df["wireType"] == 2:
                            display_name = _field_as_string(df)
                        elif df["fieldNumber"] == 3 and df["wireType"] == 2:
                            entry_fields = _field_as_message(df)
                            key = -1
                            count = 0
                            for ef in entry_fields:
                                if ef["fieldNumber"] == 1 and ef["wireType"] == 0:
                                    key = ef["value"]
                                elif ef["fieldNumber"] == 2 and ef["wireType"] == 0:
                                    count = ef["value"]
                            if key >= 0:
                                entries.append({"key": key, "count": count})

                    if name:
                        dimensions.append({
                            "name": name,
                            "displayName": display_name,
                            "entries": entries,
                        })
                elif f["fieldNumber"] == 6 and f["wireType"] == 2:
                    # Dictionary hash message
                    hash_fields = _field_as_message(f)
                    type_name = ""
                    hash_val = ""
                    for hf in hash_fields:
                        if hf["fieldNumber"] == 1 and hf["wireType"] == 2:
                            type_name = _field_as_string(hf)
                        elif hf["fieldNumber"] == 2 and hf["wireType"] == 2:
                            hash_val = _field_as_string(hf)
                    if type_name and hash_val:
                        dict_hashes[type_name] = hash_val

            result = {
                "totalCount": total_count,
                "dimensions": dimensions,
                "dictHashes": dict_hashes,
            }
            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"poe.ninja search fetch failed: {e}")
            return None

    def _fetch_dictionary(self, hash_val: str) -> Optional[dict]:
        """Fetch a dictionary (protobuf) from poe.ninja."""
        cache_key = f"dict-{hash_val}"
        cached = self._get_cached(cache_key, TTL_DICT)
        if cached is not None:
            return cached

        try:
            url = f"{BASE_URL}/builds/dictionary/{quote(hash_val)}"
            resp = self._session.get(url, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"poe.ninja dictionary: HTTP {resp.status_code}")
                return None

            buf = resp.content
            top_fields = _decode_fields(buf)

            names = []
            metadata = {}

            for f in top_fields:
                if f["fieldNumber"] == 2 and f["wireType"] == 2:
                    names.append(_field_as_string(f))
                elif f["fieldNumber"] == 3 and f["wireType"] == 2:
                    col_fields = _field_as_message(f)
                    col_name = ""
                    col_values = []
                    for cf in col_fields:
                        if cf["fieldNumber"] == 1 and cf["wireType"] == 2:
                            col_name = _field_as_string(cf)
                        elif cf["fieldNumber"] == 2 and cf["wireType"] == 2:
                            col_values.append(_field_as_string(cf))
                    if col_name:
                        metadata[col_name] = col_values

            result = {"names": names, "metadata": metadata}
            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"poe.ninja dictionary fetch failed: {e}")
            return None

    @staticmethod
    def _parse_rarity(color_value: str) -> str:
        """Parse rarity from CSS color variable name."""
        lc = color_value.lower()
        if "unique" in lc:
            return "unique"
        if "rare" in lc:
            return "rare"
        if "magic" in lc:
            return "magic"
        if "normal" in lc:
            return "normal"
        return ""

    # -------------------------------------------------------------------
    # Unique item prices from poe2scout
    # -------------------------------------------------------------------

    def fetch_unique_prices(self, slot: str) -> Dict[str, str]:
        """Fetch unique item prices for a slot from poe2scout.

        Returns dict of item_name → price_text (e.g. "~2.5 div", "~150c").
        """
        slug = SLOT_TO_UNIQUE_SLUG.get(slot)
        if not slug:
            return {}

        cache_key = f"unique-prices-{slug}"
        cached = self._get_cached(cache_key, TTL_PRICES)
        if cached is not None:
            return cached

        prices = {}
        try:
            # Get divine price from leagues endpoint
            divine_price = self._get_divine_price()
            if divine_price <= 0:
                return prices

            resp = self._session.get(
                f"{POE2SCOUT_API}/items/unique/{slug}",
                timeout=15,
            )
            if resp.status_code != 200:
                return prices

            data = resp.json()
            items = data if isinstance(data, list) else data.get("items", [])

            for item in items:
                name = item.get("name", "")
                raw_price = item.get("currentPrice", 0) or 0
                if not name or not raw_price:
                    continue

                divine_value = raw_price / divine_price
                if divine_value >= 0.85:
                    display = f"~{divine_value:.0f} div" if divine_value >= 10 else f"~{divine_value:.1f} div"
                elif raw_price >= 1:
                    display = f"~{round(raw_price)}c"
                else:
                    display = "< 1c"
                prices[name] = display

            self._set_cache(cache_key, prices)

        except Exception as e:
            logger.warning(f"poe2scout unique prices ({slug}) failed: {e}")

        return prices

    def _get_divine_price(self) -> float:
        """Get chaos-per-divine from poe2scout leagues endpoint."""
        cached = self._get_cached("divine_price", TTL_PRICES)
        if cached is not None:
            return cached

        try:
            resp = self._session.get(f"{POE2SCOUT_API}/leagues", timeout=10)
            if resp.status_code != 200:
                return 0
            leagues = resp.json()
            if leagues and isinstance(leagues, list):
                price = leagues[0].get("divinePrice", 0) or 0
                if price > 0:
                    self._set_cache("divine_price", price)
                    return price
        except Exception as e:
            logger.debug(f"poe2scout divine price fetch failed: {e}")
        return 0

    def get_popular_items_for_slot(self, char: CharacterData,
                                    slot: str) -> dict:
        """Get popular items for a slot with prices merged in.

        Returns a JSON-serializable dict for the API response.
        """
        # Determine class + main skill for search filter
        char_class = char.ascendancy or char.char_class
        main_skill = ""
        for sg in char.skill_groups:
            for d in sg.dps:
                if d.damage > 0:
                    main_skill = d.name
                    break
            if main_skill:
                break
        if not main_skill and char.skill_groups:
            main_skill = char.skill_groups[0].gems[0] if char.skill_groups[0].gems else ""

        # Fetch popular items
        items = self.fetch_popular_items(char_class, main_skill, slot)

        # Merge unique prices
        slug = SLOT_TO_UNIQUE_SLUG.get(slot)
        if slug:
            try:
                prices = self.fetch_unique_prices(slot)
                for item in items:
                    if item.rarity == "unique" and item.name in prices:
                        item.price_text = prices[item.name]
            except Exception:
                pass

        # Find the current item for this slot
        current_item = None
        for eq in char.equipment:
            if eq.slot == slot:
                current_item = eq
                break

        return {
            "slot": slot,
            "slotDisplay": SLOT_DISPLAY.get(slot, slot),
            "items": [
                {
                    "name": pi.name,
                    "count": pi.count,
                    "percentage": round(pi.percentage, 2),
                    "rarity": pi.rarity,
                    "priceText": pi.price_text,
                }
                for pi in items
            ],
            "currentItem": {
                "name": current_item.name,
                "typeLine": current_item.type_line,
                "slot": current_item.slot,
                "slotDisplay": SLOT_DISPLAY.get(current_item.slot, current_item.slot),
                "rarity": current_item.rarity,
                "sockets": current_item.sockets,
                "implicitMods": current_item.implicit_mods,
                "explicitMods": current_item.explicit_mods,
                "craftedMods": current_item.crafted_mods,
                "enchantMods": current_item.enchant_mods,
                "fracturedMods": current_item.fractured_mods,
                "desecratedMods": current_item.desecrated_mods,
                "runeMods": current_item.rune_mods,
            } if current_item else None,
        }
