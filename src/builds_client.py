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
import re
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
    "Chronomancer": "Sorceress",
    "Disciple of Varashta": "Sorceress",
    "Amazon": "Huntress",
    "Ritualist": "Huntress",
    "Witchhunter": "Mercenary",
    "Gemling Legionnaire": "Mercenary",
    "Invoker": "Monk",
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

# Slots to skip in insights/enrichment (trinkets, incursion items, etc.)
_SKIP_SLOTS = frozenset([
    "Trinket", "IncursionLegLeft", "IncursionLegRight",
    "IncursionArmLeft", "IncursionArmRight",
])

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
    defensive_stats: Optional[dict] = None           # poe.ninja pre-calculated defenses


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

            # Defensive stats (pre-calculated by poe.ninja)
            raw_ds = data.get("defensiveStats")
            defensive_stats = None
            if isinstance(raw_ds, dict):
                _n = lambda v: v if isinstance(v, (int, float)) else 0
                defensive_stats = {
                    "life": _n(raw_ds.get("life")),
                    "energyShield": _n(raw_ds.get("energyShield")),
                    "mana": _n(raw_ds.get("mana")),
                    "spirit": _n(raw_ds.get("spirit")),
                    "armour": _n(raw_ds.get("armour")),
                    "evasionRating": _n(raw_ds.get("evasionRating")),
                    "movementSpeed": _n(raw_ds.get("movementSpeed")),
                    "fireResistance": _n(raw_ds.get("fireResistance")),
                    "fireResistanceOverCap": _n(raw_ds.get("fireResistanceOverCap")),
                    "coldResistance": _n(raw_ds.get("coldResistance")),
                    "coldResistanceOverCap": _n(raw_ds.get("coldResistanceOverCap")),
                    "lightningResistance": _n(raw_ds.get("lightningResistance")),
                    "lightningResistanceOverCap": _n(raw_ds.get("lightningResistanceOverCap")),
                    "chaosResistance": _n(raw_ds.get("chaosResistance")),
                    "chaosResistanceOverCap": _n(raw_ds.get("chaosResistanceOverCap")),
                    "effectiveHealthPool": _n(raw_ds.get("effectiveHealthPool")),
                    "physicalMaximumHitTaken": _n(raw_ds.get("physicalMaximumHitTaken")),
                    "fireMaximumHitTaken": _n(raw_ds.get("fireMaximumHitTaken")),
                    "coldMaximumHitTaken": _n(raw_ds.get("coldMaximumHitTaken")),
                    "lightningMaximumHitTaken": _n(raw_ds.get("lightningMaximumHitTaken")),
                    "chaosMaximumHitTaken": _n(raw_ds.get("chaosMaximumHitTaken")),
                    "lowestMaximumHitTaken": _n(raw_ds.get("lowestMaximumHitTaken")),
                    "blockChance": _n(raw_ds.get("blockChance")),
                    "spellBlockChance": _n(raw_ds.get("spellBlockChance")),
                    "spellSuppressionChance": _n(raw_ds.get("spellSuppressionChance")),
                    "enduranceCharges": _n(raw_ds.get("enduranceCharges")),
                    "frenzyCharges": _n(raw_ds.get("frenzyCharges")),
                    "powerCharges": _n(raw_ds.get("powerCharges")),
                    "strength": _n(raw_ds.get("strength")),
                    "dexterity": _n(raw_ds.get("dexterity")),
                    "intelligence": _n(raw_ds.get("intelligence")),
                }

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
                defensive_stats=defensive_stats,
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
            "defensiveStats": char.defensive_stats,
        }

    # -------------------------------------------------------------------
    # Meta overview — class stats + popular skills
    # -------------------------------------------------------------------

    def fetch_build_summary(self) -> Optional[dict]:
        """Fetch league build summary (class distribution) from poe.ninja.

        Returns dict with leagueName, totalCharacters, classes[].
        """
        cached = self._get_cached("build-summary", TTL_SEARCH)
        if cached is not None:
            return cached

        try:
            resp = self._session.get(f"{BASE_URL}/data/build-index-state", timeout=10)
            if resp.status_code != 200:
                logger.warning(f"poe.ninja build-index-state: HTTP {resp.status_code}")
                return None

            data = resp.json()
            leagues = data.get("leagues", [])
            if not leagues:
                return None

            league = leagues[0]
            total = league.get("totalCount", 0)
            classes = []
            for stat in league.get("statistics", []):
                name = stat.get("name", "")
                pct = stat.get("percentage", 0)
                is_asc = name in ASCENDANCY_MAP
                classes.append({
                    "name": name,
                    "percentage": pct,
                    "count": round((pct / 100) * total),
                    "isAscendancy": is_asc,
                    "baseClass": ASCENDANCY_MAP.get(name) if is_asc else None,
                })

            result = {
                "leagueName": league.get("name", ""),
                "totalCharacters": total,
                "classes": classes,
            }
            self._set_cache("build-summary", result)
            return result

        except Exception as e:
            logger.warning(f"fetch_build_summary failed: {e}")
            return None

    def fetch_popular_skills_list(self) -> list:
        """Fetch popular skills for the current league.

        Returns list of {name, count, percentage} sorted by count desc.
        """
        if not self._fetch_snapshot_info():
            return []

        cache_key = f"popular-skills-{self._snapshot_version}"
        cached = self._get_cached(cache_key, TTL_SEARCH)
        if cached is not None:
            return cached

        try:
            url = (
                f"{BASE_URL}/builds/{quote(self._snapshot_version)}/popular-skills"
                f"?overview={quote(self._snapshot_name)}"
            )
            resp = self._session.get(url, timeout=10)
            if resp.status_code != 200:
                logger.warning(f"poe.ninja popular-skills: HTTP {resp.status_code}")
                return []

            data = resp.json()
            skills = data.get("skills", data) if isinstance(data, dict) else data
            if not isinstance(skills, list):
                return []

            result = sorted(
                [
                    {"name": s.get("name", ""), "count": s.get("count", 0), "percentage": s.get("percentage", 0)}
                    for s in skills if s.get("name")
                ],
                key=lambda x: x["count"],
                reverse=True,
            )[:20]

            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"fetch_popular_skills_list failed: {e}")
            return []

    def fetch_popular_anoints(self, char_class: str, skill: str) -> list:
        """Fetch popular anoints for a class+skill combo.

        Returns list of {name, percentage}.
        """
        if not self._fetch_snapshot_info():
            return []

        cache_key = f"anoints-{self._snapshot_version}-{char_class}-{skill}"
        cached = self._get_cached(cache_key, TTL_SEARCH)
        if cached is not None:
            return cached

        try:
            url = (
                f"{BASE_URL}/builds/{quote(self._snapshot_version)}/popular-anoints"
                f"?overview={quote(self._snapshot_name)}"
                f"&characterClass={quote(char_class)}"
                f"&skill={quote(skill)}"
            )
            resp = self._session.get(url, timeout=10)
            if resp.status_code != 200:
                return []

            data = resp.json()
            anoints = data.get("anoints", data) if isinstance(data, dict) else data
            if not isinstance(anoints, list):
                return []

            result = [
                {"name": a.get("name", ""), "percentage": a.get("percentage", 0)}
                for a in anoints if a.get("name")
            ]
            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"fetch_popular_anoints failed: {e}")
            return []

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

    # -------------------------------------------------------------------
    # Popular keystones search (protobuf)
    # -------------------------------------------------------------------

    def fetch_popular_keystones(self, char_class: str,
                                 skill: str) -> List[Dict[str, Any]]:
        """Fetch popular keystones for a class+skill from poe.ninja.

        Returns list of {name, count, percentage} sorted by count desc.
        """
        if not self._fetch_snapshot_info():
            return []

        cache_key = f"keystones-{self._snapshot_version}-{char_class}-{skill}"
        cached = self._get_cached(cache_key, TTL_SEARCH)
        if cached is not None:
            return cached

        try:
            search = self._fetch_search(char_class, skill)
            if not search:
                return []

            ks_dim = None
            for dim in search["dimensions"]:
                if dim["name"] == "keystones":
                    ks_dim = dim
                    break
            if not ks_dim:
                return []

            dict_hash = (search["dictHashes"].get("keystone") or
                         search["dictHashes"].get(ks_dim.get("displayName", "")))
            if not dict_hash:
                return []

            dictionary = self._fetch_dictionary(dict_hash)
            if not dictionary:
                return []

            total = search["totalCount"] or 1
            result = []
            for entry in ks_dim["entries"]:
                key = entry["key"]
                if key >= len(dictionary["names"]):
                    continue
                name = dictionary["names"][key]
                if not name:
                    continue
                result.append({
                    "name": name,
                    "count": entry["count"],
                    "percentage": round((entry["count"] / total) * 100, 1),
                })

            result.sort(key=lambda x: x["count"], reverse=True)
            result = result[:20]
            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"Popular keystones fetch failed: {e}")
            return []


# ---------------------------------------------------------------------------
# Bracket-stripping utility (poe.ninja mod text: [tag|display] → display)
# ---------------------------------------------------------------------------
_NINJA_BRACKET_RE = re.compile(r"\[([^|\]]*\|)?([^\]]*)\]")


def strip_ninja_brackets(text: str) -> str:
    """Strip poe.ninja [tag|display] → display from mod text."""
    return _NINJA_BRACKET_RE.sub(r"\2", text)


# ---------------------------------------------------------------------------
# Slot → RePoE item class mapping
# ---------------------------------------------------------------------------
SLOT_TO_ITEM_CLASS = {
    "Helm": "Helmet",
    "BodyArmour": "Body Armour",
    "Gloves": "Gloves",
    "Boots": "Boots",
    "Belt": "Belt",
    "Amulet": "Amulet",
    "Ring": "Ring",
    "Ring2": "Ring",
    "Shield": "Shield",
    "Offhand": "Shield",    # default; weapons resolved by type_line
    "Flask": "Flask",
}

# type_line keywords → item class for weapons
_WEAPON_TYPE_HINTS = {
    "Bow": "Bows", "Crossbow": "Crossbows", "Wand": "Wands",
    "Staff": "Staves", "Warstaff": "Warstaves", "Sceptre": "Sceptres",
    "Dagger": "Daggers", "Claw": "Claws",
    "Sword": "One Hand Swords", "Axe": "One Hand Axes",
    "Mace": "One Hand Maces", "Flail": "Flails", "Spear": "Spears",
    "Quiver": "Quivers", "Focus": "Foci", "Buckler": "Bucklers",
}


def _resolve_item_class(slot: str, type_line: str = "") -> str:
    """Resolve poe.ninja inventoryId + typeLine to a RePoE item class."""
    if slot in ("Weapon", "Weapon2", "Offhand", "Offhand2"):
        tl = type_line.lower()
        for hint, cls in _WEAPON_TYPE_HINTS.items():
            if hint.lower() in tl:
                return cls
        # "Two Hand" prefix
        if "two hand" in tl:
            if "sword" in tl:
                return "Two Hand Swords"
            if "axe" in tl:
                return "Two Hand Axes"
            if "mace" in tl:
                return "Two Hand Maces"
    return SLOT_TO_ITEM_CLASS.get(slot, slot)


# ---------------------------------------------------------------------------
# Mod enrichment — adds tier data to character equipment mods
# ---------------------------------------------------------------------------

# Mod types and their corresponding mod_parser type labels
_MOD_TYPE_MAP = {
    "implicit_mods": "implicit",
    "explicit_mods": "explicit",
    "crafted_mods": "explicit",
    "fractured_mods": "explicit",
    "desecrated_mods": "explicit",
    "enchant_mods": "enchant",
    "rune_mods": "explicit",
}


def enrich_item_mods(item: 'CharacterItem', mod_parser, mod_database) -> Dict[str, list]:
    """Enrich a CharacterItem's mods with tier data.

    Returns dict keyed by mod type ("implicitMods", "explicitMods", etc.)
    with parallel arrays of tier info dicts (or None for unmatched mods).
    """
    item_class = _resolve_item_class(item.slot, item.type_line)
    result = {}

    mod_type_pairs = [
        ("implicitMods", item.implicit_mods, "implicit"),
        ("explicitMods", item.explicit_mods, "explicit"),
        ("craftedMods", item.crafted_mods, "explicit"),
        ("fracturedMods", item.fractured_mods, "explicit"),
        ("desecratedMods", item.desecrated_mods, "explicit"),
        ("enchantMods", item.enchant_mods, "enchant"),
        ("runeMods", item.rune_mods, "explicit"),
    ]

    for key, mods, parse_type in mod_type_pairs:
        if not mods:
            continue
        tiers = []
        for mod_text in mods:
            tier_data = _enrich_single_mod(mod_text, parse_type, item_class,
                                            mod_parser, mod_database)
            tiers.append(tier_data)
        result[key] = tiers

    return result


def _enrich_single_mod(mod_text: str, mod_type: str, item_class: str,
                        mod_parser, mod_database) -> Optional[dict]:
    """Try to enrich a single mod line with tier data."""
    try:
        clean = strip_ninja_brackets(mod_text)
        parsed = mod_parser._match_mod(clean, mod_type)
        if not parsed or not parsed.stat_id:
            return None
        return mod_database.get_full_tier_data(parsed.stat_id, parsed.value,
                                                item_class)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Build classification (ported from lama-mobile BuildsScreen.tsx)
# ---------------------------------------------------------------------------

ATTACK_SKILLS = frozenset([
    # Melee
    "Power Siphon", "Boneshatter", "Earthquake", "Ground Slam", "Sunder",
    "Heavy Strike", "Glacial Hammer", "Lightning Strike", "Molten Strike",
    "Viper Strike", "Double Strike", "Dual Strike", "Cleave", "Lacerate",
    "Cyclone", "Flicker Strike", "Whirling Slash", "Shield Charge",
    "Leap Slam", "Consecrated Path", "Tectonic Slam", "Perforate",
    "Bladestorm", "Chain Hook", "Static Strike", "Smite",
    "Splitting Steel", "Shattering Steel", "Lancing Steel",
    "Mace Bash", "Spinning Assault", "Pounce",
    "Ice Strike", "Quarterstaff Strike", "Shred",
    "Rampage", "Hammer of the Gods", "Furious Slam", "Maul",
    "Shield Wall", "Gathering Storm",
    "Whirling Assault", "Devour", "Seismic Cry", "Primal Strikes",
    "Fangs of Frost", "Storm Wave", "Falling Thunder",
    # Ranged - Bow
    "Split Arrow", "Lightning Arrow", "Ice Shot", "Burning Arrow",
    "Tornado Shot", "Rain of Arrows", "Barrage", "Caustic Arrow",
    "Scourge Arrow", "Galvanic Arrow", "Artillery Ballista",
    "Shrapnel Ballista", "Siege Ballista", "Explosive Arrow",
    "Power Shot", "Gas Arrow", "Rend", "Bow Shot", "Oil Barrage",
    "Rapid Shot", "Focused Shot", "Snipe",
    "Poisonburst Arrow", "Vine Arrow",
    # Crossbow
    "Bolt Burst", "Crossbow Shot", "Armour Piercing Rounds",
    "Plasma Blast", "Explosive Grenade", "Oil Grenade",
    "Galvanic Shards", "Stormblast Bolts",
])

SPELL_SKILLS = frozenset([
    # Cold
    "Comet", "Ice Nova", "Frost Bolt", "Frostbolt", "Glacial Cascade",
    "Arctic Breath", "Freezing Pulse", "Cold Snap", "Vortex", "Winter Orb",
    "Frost Wall", "Frozen Orb", "Ice Spear",
    "Frost Bomb", "Snap", "Freezing Shards",
    # Fire
    "Fireball", "Fire Ball", "Incinerate", "Flame Wall", "Fire Trap",
    "Flammability", "Flame Surge", "Flame Bolt", "Living Bomb",
    "Infernal Cry",
    # Lightning
    "Arc", "Ball Lightning", "Storm Call", "Lightning Tendrils",
    "Spark", "Shock Nova", "Lightning Conduit", "Galvanic Field",
    "Conductivity", "Orb of Storms", "Storm Bolt", "Lightning Spear",
    "Lightning Rod", "Thunderstorm", "Lightning Warp",
    # Chaos
    "Blight", "Essence Drain", "Contagion", "Soulrend", "Bane",
    "Dark Pact", "Forbidden Rite", "Chaos Bolt", "Hexblast",
    "Entangle", "Requiem", "Toxic Growth", "Thrashing Vines",
    # Nature / Druid
    "Twister",
    # Physical / generic
    "Blade Vortex", "Ethereal Knives", "Bladefall", "Blade Blast",
    "Reap", "Exsanguinate", "Rolling Magma", "Magma Orb",
    "Bone Offering", "Spirit Offering", "Flesh Offering",
    "Raise Zombie", "Summon Skeletons", "Summon Raging Spirit",
    "Summon Phantasm", "Raise Spectre",
    "Unearth", "Desecrate", "Spirit Nova",
])

MINION_SKILLS = frozenset([
    "Raise Zombie", "Summon Skeletons", "Summon Raging Spirit",
    "Summon Phantasm", "Raise Spectre", "Animate Weapon",
    "Dominate", "Summon Reaper", "Summon Volatile Dead",
])

MELEE_SKILLS = frozenset([
    "Sunder", "Heavy Strike", "Glacial Hammer", "Molten Strike", "Cyclone",
    "Flicker Strike", "Whirling Slash", "Shield Charge", "Leap Slam",
    "Tectonic Slam", "Perforate", "Bladestorm", "Static Strike", "Smite",
    "Mace Bash", "Spinning Assault", "Pounce", "Ice Strike",
    "Quarterstaff Strike", "Shred", "Rampage", "Hammer of the Gods",
    "Furious Slam", "Maul", "Shield Wall", "Gathering Storm",
    "Boneshatter", "Earthquake", "Ground Slam", "Lacerate", "Cleave",
    "Whirling Assault", "Devour", "Seismic Cry", "Primal Strikes",
    "Fangs of Frost", "Storm Wave", "Falling Thunder",
])

CRIT_KEYSTONES = frozenset([
    "Inevitable Judgement", "Elemental Overload", "Precision",
    "Deadly Precision", "Assassin's Mark", "Nightblade",
])

ES_KEYSTONES = frozenset([
    "Chaos Inoculation", "Ghost Reaver", "Wicked Ward",
    "Arcane Surge", "Pain Attunement", "Energy Blade",
])

_ELEMENT_KEYWORDS = [
    ("fire", [re.compile(p, re.I) for p in [
        r"\bfire\b", r"\bburn", r"\bignite", r"\bincinerate", r"\bflame",
        r"\binfernal", r"\bliving bomb", r"\boil barrage"]]),
    ("cold", [re.compile(p, re.I) for p in [
        r"\bcold\b", r"\bfreez", r"\bfrost", r"\bice\b", r"\bglacial",
        r"\bwinter", r"\bcomet\b", r"\bsnap\b", r"\bfangs of frost"]]),
    ("lightning", [re.compile(p, re.I) for p in [
        r"\blightning\b", r"\bshock", r"\barc\b", r"\bspark\b",
        r"\bgalvanic", r"\bstorm", r"\bconducti", r"\bthunder"]]),
    ("chaos", [re.compile(p, re.I) for p in [
        r"\bchaos\b", r"\bpoison", r"\bviper", r"\bblight",
        r"\bwither", r"\bhexblast", r"\bentangle", r"\brequiem",
        r"\btoxic", r"\bthrashing vines"]]),
    ("physical", [re.compile(p, re.I) for p in [
        r"\bphysical\b", r"\bbleed", r"\bimpale", r"\bsteel\b",
        r"\bbone\b", r"\brampage\b", r"\bsunder\b", r"\bhammer\b",
        r"\bmaul\b", r"\bshred\b", r"\btwister\b", r"\bdevour\b",
        r"\bseismic"]]),
]

# Dead mod patterns for classification
_DEAD_MOD_ATTACK = [
    (re.compile(r"increased Spell Damage", re.I), "spell damage doesn't help attack builds"),
    (re.compile(r"increased Cast Speed", re.I), "cast speed doesn't help attack builds"),
    (re.compile(r"adds \d+ to \d+ .* Damage to Spells", re.I), "flat spell damage doesn't help attack builds"),
]
_DEAD_MOD_SPELL = [
    (re.compile(r"increased Attack Speed", re.I), "attack speed doesn't scale spell damage"),
]
_DEAD_MOD_UNIVERSAL = [
    (re.compile(r"Allies in your Presence", re.I), "party/mount mod — only active with allies nearby"),
]


@dataclass
class BuildArchetype:
    """Concise classification of a build."""
    tags: List[str] = field(default_factory=list)
    damage_type: str = "unknown"   # attack / spell / mixed / unknown
    defense_type: str = "life"     # life / es / hybrid / mom
    main_skill: str = ""
    is_crit: bool = False
    is_coc: bool = False
    elements: List[str] = field(default_factory=list)
    dead_mods: List[Dict[str, str]] = field(default_factory=list)


def classify_build(char: CharacterData) -> BuildArchetype:
    """Classify a build's archetype from character data.

    Determines damage type, defense strategy, elements, crit, CoC,
    and identifies dead mod patterns.
    """
    # Find main skill by highest damage
    main_skill = ""
    main_dps = 0
    for sg in char.skill_groups:
        for d in sg.dps:
            effective = d.dps if d.dps > 0 else d.damage
            if effective > main_dps:
                main_dps = effective
                main_skill = d.name

    # Fallback: first gem in first skill group
    if not main_skill and char.skill_groups:
        gems = char.skill_groups[0].gems
        if gems:
            main_skill = gems[0]

    # Detect Cast on Crit
    is_coc = False
    for sg in char.skill_groups:
        has_coc = any("Cast on Crit" in g or "Cast when Crit" in g for g in sg.gems)
        has_attack = any(g in ATTACK_SKILLS for g in sg.gems)
        has_spell = any(g in SPELL_SKILLS for g in sg.gems)
        if has_coc and has_attack and has_spell:
            is_coc = True
            break
    # Heuristic: spell DPS + attack gem in same group
    if not is_coc and main_skill in SPELL_SKILLS:
        for sg in char.skill_groups:
            has_dps = any(d.name == main_skill for d in sg.dps)
            has_attack = any(g in ATTACK_SKILLS for g in sg.gems)
            if has_dps and has_attack:
                is_coc = True
                break

    # Damage type
    if main_skill in MINION_SKILLS:
        damage_type = "spell"
    elif main_skill in SPELL_SKILLS:
        damage_type = "spell"
    elif main_skill in ATTACK_SKILLS:
        damage_type = "attack"
    else:
        # Heuristic: count attack vs spell mods on gear
        atk_count = 0
        spell_count = 0
        for eq in char.equipment:
            mods = " ".join(
                strip_ninja_brackets(m)
                for m_list in [eq.explicit_mods, eq.implicit_mods, eq.crafted_mods]
                for m in (m_list or [])
            ).lower()
            if "attack speed" in mods:
                atk_count += 1
            if "spell" in mods or "cast speed" in mods:
                spell_count += 1
        if atk_count > spell_count:
            damage_type = "attack"
        elif spell_count > atk_count:
            damage_type = "spell"
        elif atk_count > 0 and spell_count > 0:
            damage_type = "mixed"
        else:
            damage_type = "unknown"

    # Elements
    elements = set()
    for elem, patterns in _ELEMENT_KEYWORDS:
        for pat in patterns:
            if pat.search(main_skill):
                elements.add(elem)
    if damage_type == "attack" and not elements:
        elements.add("physical")
    if not elements and damage_type == "spell":
        # Check all skill gem names
        for sg in char.skill_groups:
            for gem in sg.gems:
                for elem, patterns in _ELEMENT_KEYWORDS:
                    for pat in patterns:
                        if pat.search(gem):
                            elements.add(elem)
    if not elements:
        elements.add("physical")

    # Crit detection
    is_crit = is_coc  # CoC is always crit
    if not is_crit:
        is_crit = any(k in CRIT_KEYSTONES for k in char.keystones)
    if not is_crit:
        for eq in char.equipment:
            mods = " ".join(
                strip_ninja_brackets(m)
                for m_list in [eq.explicit_mods, eq.implicit_mods,
                               eq.crafted_mods, eq.fractured_mods, eq.rune_mods]
                for m in (m_list or [])
            )
            if re.search(r"Critical Hit Chance|Critical Damage Bonus", mods, re.I):
                is_crit = True
                break

    # Defense type
    has_ci = "Chaos Inoculation" in char.keystones
    has_mom = "Mind Over Matter" in char.keystones
    has_eb = "Eldritch Battery" in char.keystones
    if has_ci:
        defense_type = "es"
    elif has_mom and has_eb:
        defense_type = "mom"
    elif any(k in ES_KEYSTONES for k in char.keystones):
        defense_type = "hybrid"
    else:
        defense_type = "life"

    # Dead mods
    dead_mods = []
    if damage_type == "spell" and not is_coc:
        dead_mods.extend(_DEAD_MOD_SPELL)
    if damage_type == "attack" and not is_coc:
        dead_mods.extend(_DEAD_MOD_ATTACK)

    # Scan equipment for dead mods
    found_dead = []
    for eq in char.equipment:
        # Skip weapons — too complex
        if eq.slot in ("Weapon", "Weapon2"):
            continue
        all_mods = []
        for m_list in [eq.explicit_mods, eq.crafted_mods, eq.enchant_mods]:
            for m in (m_list or []):
                all_mods.append(strip_ninja_brackets(m))
        for mod_text in all_mods:
            for pat, reason in dead_mods + _DEAD_MOD_UNIVERSAL:
                if pat.search(mod_text):
                    found_dead.append({
                        "slot": eq.slot,
                        "mod": mod_text,
                        "reason": reason,
                    })
                    break

    # Build tags
    tags = []
    if damage_type != "unknown":
        tags.append(damage_type)
    for elem in sorted(elements):
        tags.append(elem)
    if is_coc:
        tags.append("CoC")
    elif is_crit:
        tags.append("crit")
    tags.append(defense_type)

    return BuildArchetype(
        tags=tags,
        damage_type=damage_type,
        defense_type=defense_type,
        main_skill=main_skill,
        is_crit=is_crit,
        is_coc=is_coc,
        elements=sorted(elements),
        dead_mods=found_dead,
    )


# ---------------------------------------------------------------------------
# Build Efficiency — upgrade priority, anoint optimizer, cost tiers, lineage ROI
# ---------------------------------------------------------------------------

INVESTMENT_TIERS = [
    {"label": "Starter", "min": 0, "max": 5, "desc": "Build-defining cheap uniques"},
    {"label": "Core", "min": 5, "max": 15, "desc": "Gem levels + core unique upgrades"},
    {"label": "Lineage", "min": 15, "max": 50, "desc": "Lineage support gems (best DPS/div)", "isBestRoi": True},
    {"label": "Variant", "min": 50, "max": 100, "desc": "Build variant switch / premium gear"},
    {"label": "Endgame", "min": 100, "max": 500, "desc": "Self-craft triple T1 / mirror-tier"},
]

_ALLOCATES_RE = re.compile(r"Allocates\s+(.+)", re.I)


def _parse_price_text(text: str) -> float:
    """Parse price text like '~2.5 div', '~150c', '< 1c' to float divine value.

    Returns value in divines (assumes ~150c per divine for chaos conversion).
    """
    if not text:
        return 0.0
    text = text.strip().lstrip("~").strip()
    if text.startswith("<"):
        return 0.01  # "< 1c" → negligible
    m = re.match(r"([\d.]+)\s*(div|d|c)", text, re.I)
    if not m:
        return 0.0
    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit in ("div", "d"):
        return val
    # chaos → divine (rough 150c/div)
    return val / 150.0


def detect_current_anoint(char: CharacterData) -> Optional[str]:
    """Detect the current anoint on a character's amulet.

    Scans amulet enchant_mods for 'Allocates <Notable>' pattern.
    Returns the notable name or None.
    """
    for eq in char.equipment:
        if eq.slot == "Amulet":
            for mod in eq.enchant_mods:
                m = _ALLOCATES_RE.search(strip_ninja_brackets(mod))
                if m:
                    return m.group(1).strip()
            break
    return None


# Notable anoint descriptions for top ~50 commonly anointed passives
ANOINT_DESCRIPTIONS = {
    "constitution": "+10% maximum Life",
    "heart of the warrior": "+20 Strength, +10% maximum Life",
    "discipline and training": "+12% maximum Life",
    "profane chemistry": "+20% Flask Charges gained, +12% maximum Life",
    "crystal skin": "+1% all maximum Resistances",
    "diamond skin": "+15% all Elemental Resistances",
    "whispers of doom": "Can apply an additional Curse",
    "corruption": "+1 Curse limit, +10% Chaos Damage",
    "charisma": "20% reduced Mana Reservation of Skills",
    "champion of the cause": "10% reduced Mana Reservation of Skills",
    "sovereignty": "12% reduced Mana Reservation of Skills",
    "influence": "8% increased Area of Effect, 8% reduced Mana Reservation",
    "golem's blood": "1.6% Life Regeneration, +10% maximum Life",
    "herbalism": "+12% maximum Life, 20% increased Flask Charges gained",
    "overcharge": "+40% increased Spell Damage per Power Charge",
    "potency of will": "+20% Skill Effect Duration",
    "instability": "+25% Elemental Damage, 10% chance to Shock",
    "breath of flames": "+25% Fire Damage, +20% Burning Damage",
    "breath of rime": "+25% Cold Damage, +15% Freeze Duration",
    "breath of lightning": "+25% Lightning Damage, +10% Shock Effect",
    "assassination": "+30% Critical Strike Multiplier",
    "doom cast": "+40% Spell Critical Strike Chance",
    "deadly precision": "+30% Critical Strike Chance for Attacks",
    "command of steel": "Fortify effect is doubled, +20% Armour",
    "iron reflexes": "Converts all Evasion Rating to Armour",
    "phase acrobatics": "+30% Spell Dodge Chance",
    "spell suppression": "+8% Spell Suppression Chance",
    "devotion": "+6% maximum Life, +20% Armour",
    "tireless": "+12% maximum Life, 10% reduced Cost of Skills",
    "mind over matter": "40% of Damage taken from Mana before Life",
    "pain attunement": "30% more Spell Damage when on Low Life",
    "clever thief": "Leech 0.6% Attack Damage as Life and Mana",
    "swift killer": "+3% Attack/Cast Speed per Frenzy Charge",
    "aspect of the eagle": "+10% Accuracy, +20% Evasion Rating",
    "fangs of the viper": "+5% Chaos Damage over Time Multiplier",
    "graceful assault": "+20% Attack Speed while Moving",
    "lava lash": "+30% Fire Damage with Attack Skills",
    "snowforged": "+15% Cold Damage, +20% Fire Damage",
    "heart of ice": "+20% Cold Damage, +4% Cold DoT Multiplier",
    "arcane potency": "+25% Spell Critical Strike Chance, +25% Crit Multi",
    "force shaper": "+15% Spell Damage, +10% Area of Effect",
    "successive detonations": "+2 to max number of Mines, +10% Mine Damage",
    "explosive impact": "+20% Area of Effect, +15% Area Damage",
    "thick skin": "+15% maximum Life, +10% Life Recovery Rate",
    "sanctity": "+15% Energy Shield, +12% Life Regeneration",
    "essence surge": "+30% faster start of ES Recharge",
    "utmost might": "+40 Strength",
    "utmost swiftness": "+40 Dexterity",
    "utmost intellect": "+40 Intelligence",
}

def get_anoint_description(name: str) -> Optional[str]:
    """Get a short description for a notable passive by name."""
    if not name:
        return None
    return ANOINT_DESCRIPTIONS.get(name.lower())


def compute_upgrade_priority(char: CharacterData, slot_summary: list,
                             builds_client_inst: 'BuildsClient',
                             price_cache_dict: dict) -> list:
    """Rank equipment slots by improvement gap vs top builds.

    Args:
        char: Character data
        slot_summary: Per-slot tier summary from build-insights enrichment
        builds_client_inst: BuildsClient instance for fetching popular items
        price_cache_dict: Dict of item_name → price_text from unique prices

    Returns list of {slot, slotDisplay, currentTier, targetTier, gap,
                     efficiency, topItem, topItemPrice} sorted by gap desc.
    """
    if not slot_summary:
        return []

    char_class = char.ascendancy or char.char_class
    main_skill = ""
    for sg in char.skill_groups:
        for d in sg.dps:
            if d.damage > 0:
                main_skill = d.name
                break
        if main_skill:
            break
    if not main_skill and char.skill_groups and char.skill_groups[0].gems:
        main_skill = char.skill_groups[0].gems[0]

    results = []
    for ss in slot_summary:
        if ss.get("enrichedCount", 0) == 0:
            continue
        slot = ss["slot"]
        current_tier = ss.get("avgTier", 0)
        if current_tier == 0:
            continue

        # Fetch popular items for this slot to estimate target tier
        try:
            popular = builds_client_inst.fetch_popular_items(
                char_class, main_skill, slot)
        except Exception:
            popular = []

        # Estimate target tier from popular item distribution
        # Unique items with >30% adoption suggest T1.0 target, else T1.8
        top_item_name = ""
        top_item_price = ""
        target_tier = 1.8  # default
        if popular:
            top = popular[0]
            top_item_name = top.name
            if top.rarity == "unique" and top.percentage > 30:
                target_tier = 1.0
            # Get price from price_cache or popular item's own price_text
            top_item_price = price_cache_dict.get(top.name, "") or top.price_text

        gap = current_tier - target_tier
        if gap <= 0:
            continue

        price_val = _parse_price_text(top_item_price)
        efficiency = gap / price_val if price_val > 0 else gap

        results.append({
            "slot": slot,
            "slotDisplay": SLOT_DISPLAY.get(slot, slot),
            "currentTier": current_tier,
            "targetTier": round(target_tier, 1),
            "gap": round(gap, 1),
            "efficiency": round(efficiency, 2),
            "topItem": top_item_name,
            "topItemPrice": top_item_price,
        })

    results.sort(key=lambda x: x["gap"], reverse=True)
    return results[:5]


def compute_cost_tiers(archetype: BuildArchetype,
                       popular_by_slot: dict,
                       price_cache_dict: dict,
                       lineage_gems: list) -> list:
    """Build 5-tier investment breakdown with specific recommendations.

    Args:
        archetype: Build classification
        popular_by_slot: Dict of slot → list of PopularItem
        price_cache_dict: Dict of item_name → price_text
        lineage_gems: List of lineage gem dicts from find_lineage_upgrades()

    Returns list of tier dicts with recommendations.
    """
    tiers = []
    for tier in INVESTMENT_TIERS:
        recs = []
        lo, hi = tier["min"], tier["max"]

        # Check popular items across slots
        for slot, items in popular_by_slot.items():
            for item in items[:5]:
                price_text = price_cache_dict.get(item.name, "") or item.price_text
                price_val = _parse_price_text(price_text)
                if lo <= price_val <= hi and item.name:
                    recs.append({
                        "name": item.name,
                        "price": price_text,
                        "slot": SLOT_DISPLAY.get(slot, slot),
                        "type": "gear",
                    })

        # Check lineage gems in this tier
        for gem in lineage_gems:
            price_val = gem.get("priceDiv", 0)
            if lo <= price_val <= hi:
                recs.append({
                    "name": gem["name"],
                    "price": gem.get("priceText", ""),
                    "slot": "Gem",
                    "type": "lineage",
                })

        # Deduplicate by name, take top 3
        seen = set()
        unique_recs = []
        for r in recs:
            if r["name"] not in seen:
                seen.add(r["name"])
                unique_recs.append(r)
        unique_recs = unique_recs[:3]

        tiers.append({
            "label": tier["label"],
            "min": lo,
            "max": hi,
            "desc": tier["desc"],
            "isBestRoi": tier.get("isBestRoi", False),
            "recommendations": unique_recs,
        })

    return tiers


def find_lineage_upgrades(skill_groups: list,
                          price_cache_dict: dict) -> list:
    """Find lineage support gem upgrades for character's active gems.

    Args:
        skill_groups: Character's skill groups (list of SkillGroup)
        price_cache_dict: Dict of item_name → price_text (includes lineage gems)

    Returns list of {name, gem, priceText, priceDiv} sorted by price asc.
    """
    # Collect active gem names
    active_gems = set()
    for sg in skill_groups:
        for gem in sg.gems:
            if gem:
                active_gems.add(gem.lower())

    # Search price cache for lineage support gems
    lineage_results = []
    for item_name, price_text in price_cache_dict.items():
        name_lower = item_name.lower()
        # Lineage gems have names like "Uhtred's Augury", "Garukhan's Resolve"
        # They're categorized as support gems with possessive names
        if "'s " not in name_lower:
            continue
        # Check if it's priced (indicating it's a real tradeable item)
        price_val = _parse_price_text(price_text)
        if price_val <= 0:
            continue

        lineage_results.append({
            "name": item_name,
            "gem": "",  # We don't know which gem it supports from name alone
            "priceText": price_text,
            "priceDiv": round(price_val, 1),
        })

    lineage_results.sort(key=lambda x: x["priceDiv"])
    return lineage_results


def compute_improvement_package(
    char: CharacterData,
    archetype: 'BuildArchetype',
    slot_summary: list,
    popular_anoints: list,
    lineage_gems: list,
    upgrade_priority: list,
) -> dict:
    """Compute a prioritized improvement package in 3 sections.

    Returns dict with:
      free: [{action, detail, impact}]       — no-cost changes
      spend: [{action, detail, cost, impact}] — gear upgrades
      alternatives: [{action, detail, pros, cons}] — build variants
    """
    free = []
    spend = []
    alternatives = []

    # Free: Anoint
    current_anoint = detect_current_anoint(char)
    if popular_anoints:
        top_anoint = popular_anoints[0]
        if not current_anoint:
            free.append({
                "action": "Add anoint",
                "detail": f"Anoint amulet with {top_anoint['name']} ({top_anoint['percentage']}% of top builds)",
                "impact": "Free damage/defense boost",
            })
        elif top_anoint["name"].lower() != (current_anoint or "").lower():
            free.append({
                "action": "Change anoint",
                "detail": f"Switch from {current_anoint} to {top_anoint['name']} ({top_anoint['percentage']}% of top builds)",
                "impact": "Better alignment with meta",
            })

    # Free: Gem links (check for missing support gems)
    if lineage_gems:
        for lg in lineage_gems[:2]:
            if lg.get("priceDiv", 0) < 1:
                free.append({
                    "action": "Lineage gem",
                    "detail": f"Use {lg['name']} (cheap at ~{lg.get('priceDiv', 0):.1f}d)",
                    "impact": "Multiplicative 'More' damage",
                })

    # Spend: From upgrade priority
    for u in (upgrade_priority or [])[:3]:
        if u.get("topItem") and u.get("topItemPrice"):
            spend.append({
                "action": f"Upgrade {u['slotDisplay']}",
                "detail": f"Replace with {u['topItem']}",
                "cost": u["topItemPrice"],
                "impact": f"T{u['currentTier']} → better tier (gap +{u['gap']})",
            })

    # Spend: Lineage gems (those that cost something)
    for lg in (lineage_gems or []):
        if lg.get("priceDiv", 0) >= 1:
            spend.append({
                "action": f"Buy {lg['name']}",
                "detail": "Lineage support gem — multiplicative damage",
                "cost": f"~{lg['priceDiv']:.1f}d",
                "impact": "More DPS multiplier",
            })
            if len(spend) >= 5:
                break

    # Alternatives: Build variant ideas
    if archetype.is_crit and not archetype.is_coc:
        alternatives.append({
            "action": "Consider CoC variant",
            "detail": f"Convert to Cast on Critical Strike with {archetype.main_skill or 'primary skill'}",
            "pros": "Higher clear speed, automated casting",
            "cons": "Requires specific weapon, higher investment",
        })
    if archetype.defense_type == "life" and any(
        k in (char.keystones or []) for k in ["Pain Attunement", "Low Life"]
    ):
        alternatives.append({
            "action": "Switch to ES/Low Life",
            "detail": "Respec to Energy Shield with Pain Attunement",
            "pros": "30% more spell damage (PA always active), high EHP",
            "cons": "Expensive gear, chaos vulnerability",
        })
    if archetype.defense_type == "life":
        alternatives.append({
            "action": "Add block layer",
            "detail": "Shield + block nodes for damage mitigation",
            "pros": "50%+ chance to avoid all damage from hits",
            "cons": "Lose dual-wield/2H DPS, passive point investment",
        })

    return {
        "free": free,
        "spend": spend[:5],
        "alternatives": alternatives[:3],
    }


def compute_build_comparison(
    char: CharacterData,
    popular_keystones: list,
    popular_anoints: list,
    popular_by_slot: Dict[str, List[PopularItem]],
    slot_summary: list,
) -> dict:
    """Compare user's build against aggregate top-build data.

    Returns dict with:
      keystoneDiffs: [{name, yourHasIt, topPct}]
      gearMatches: [{slot, slotDisplay, yourItem, topItem, topPct, matches}]
      anointMatch: {yourAnoint, topAnoint, topPct, matches}
      overallScore: 0-100 how closely user matches meta
    """
    user_ks = set(char.keystones or [])
    ks_diffs = []
    for pk in (popular_keystones or []):
        ks_diffs.append({
            "name": pk["name"],
            "yourHasIt": pk["name"] in user_ks,
            "topPct": pk.get("percentage", 0),
        })

    # Anoint comparison
    current_anoint = detect_current_anoint(char)
    top_anoint = (popular_anoints or [{}])[0] if popular_anoints else {}
    anoint_match = {
        "yourAnoint": current_anoint,
        "topAnoint": top_anoint.get("name"),
        "topPct": top_anoint.get("percentage", 0),
        "matches": current_anoint and top_anoint.get("name", "").lower() == (current_anoint or "").lower(),
    }

    # Per-slot gear match
    gear_matches = []
    match_count = 0
    total_slots = 0
    for eq in char.equipment:
        if eq.slot in _SKIP_SLOTS:
            continue
        items = popular_by_slot.get(eq.slot, [])
        if not items:
            continue
        total_slots += 1
        top_item = items[0]
        user_name = (eq.name or eq.type_line or "").lower()
        top_name = (top_item.name or "").lower()
        matches = user_name == top_name or (top_name in user_name) or (user_name in top_name)
        if matches:
            match_count += 1
        sd = SLOT_DISPLAY.get(eq.slot, eq.slot)
        gear_matches.append({
            "slot": eq.slot,
            "slotDisplay": sd,
            "yourItem": eq.name or eq.type_line,
            "topItem": top_item.name,
            "topPct": round(top_item.percentage, 1),
            "matches": matches,
        })

    # Overall meta-alignment score
    ks_score = sum(1 for kd in ks_diffs if kd["yourHasIt"] and kd["topPct"] >= 20) / max(
        sum(1 for kd in ks_diffs if kd["topPct"] >= 20), 1)
    gear_score = match_count / max(total_slots, 1)
    anoint_score = 1 if anoint_match["matches"] else 0
    overall = round((ks_score * 30 + gear_score * 50 + anoint_score * 20), 0)

    return {
        "keystoneDiffs": ks_diffs,
        "gearMatches": gear_matches,
        "anointMatch": anoint_match,
        "overallScore": int(overall),
    }
