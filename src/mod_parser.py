"""
LAMA - Mod Parser
Matches item mod text to trade API stat IDs for rare item pricing.

On startup, fetches stat filter definitions from the POE2 trade API
(GET /api/trade2/data/stats), caches them to disk for 24h, and compiles
each stat text template into a regex for matching against clipboard mod lines.
"""

import re
import json
import time
import logging
from dataclasses import dataclass
from typing import List, Optional

import requests

from config import (
    TRADE_STATS_URL,
    TRADE_STATS_CACHE_FILE,
    TRADE_ITEMS_URL,
    TRADE_ITEMS_CACHE_FILE,
)

logger = logging.getLogger(__name__)

STATS_CACHE_MAX_AGE = 86400  # 24 hours


@dataclass
class StatDefinition:
    id: str            # "explicit.stat_3299347043"
    text: str          # "+# to maximum Life"
    type: str          # "explicit"
    regex: re.Pattern  # compiled from text template


@dataclass
class ParsedMod:
    raw_text: str      # "+42 to maximum Life"
    stat_id: str       # "explicit.stat_3299347043"
    value: float       # 42.0
    mod_type: str      # "explicit" or "implicit"


# Word pairs where the game uses one for positive and the other for negative
# values of the same stat. "increased" in the template also matches "reduced", etc.
_OPPOSITE_WORDS = {
    "increased": "reduced",
    "reduced": "increased",
    "more": "less",
    "less": "more",
}


def _template_to_regex(text: str) -> Optional[re.Pattern]:
    """
    Convert a trade API stat text template to a regex.

    Example: "+# to maximum Life" → r"^\+?(\d+(?:\.\d+)?)\s+to\s+maximum\s+Life$"

    The '#' placeholder is replaced with a number capture group.
    Special regex chars are escaped, whitespace is made flexible.
    "increased"/"reduced" and "more"/"less" are made interchangeable
    so that the same stat matches both directions.
    """
    if not text or "#" not in text:
        return None

    # Split on '#' placeholders, escape each part, rejoin with capture group
    parts = text.split("#")
    escaped = []
    for i, part in enumerate(parts):
        # Escape regex special chars
        esc = re.escape(part)
        # Make whitespace flexible (collapse multiple spaces, allow any whitespace)
        esc = re.sub(r"\\ ", r"\\s+", esc)
        escaped.append(esc)

    # Join parts with a number capture group for each '#'
    number_group = r"(\d+(?:\.\d+)?)"
    pattern_str = number_group.join(escaped)

    # Allow "increased"↔"reduced" and "more"↔"less" to be interchangeable
    for word, opposite in _OPPOSITE_WORDS.items():
        escaped_word = re.escape(word)
        if re.search(escaped_word, pattern_str, re.IGNORECASE):
            pattern_str = re.sub(
                escaped_word,
                "(?:%s|%s)" % (re.escape(word), re.escape(opposite)),
                pattern_str,
                flags=re.IGNORECASE,
            )

    # Optional leading +/- sign (many mods start with +N or -N)
    pattern_str = r"^[\+\-]?" + pattern_str + r"$"

    try:
        return re.compile(pattern_str, re.IGNORECASE)
    except re.error:
        return None


class ModParser:
    """
    Matches raw mod text lines against trade API stat definitions.

    Usage:
        mp = ModParser()
        mp.load_stats()  # fetches/caches stat definitions
        mods = mp.parse_mods(item)  # matches mod lines to stat IDs
    """

    def __init__(self):
        self._stats: List[StatDefinition] = []
        self._loaded = False
        # Base types for resolving magic item names → base_type
        # Sorted longest-first so "Stellar Amulet" matches before "Amulet"
        self._base_types: List[str] = []

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load_stats(self):
        """Load stat definitions and base types from disk cache or trade API."""
        # Try disk cache first
        if self._load_from_disk():
            self._loaded = True
        elif self._fetch_from_api():
            self._loaded = True
        else:
            logger.warning("ModParser: no stat definitions available — rare pricing disabled")

        # Load base types (for magic item base_type resolution)
        self._load_base_types()

    def parse_mods(self, item) -> List[ParsedMod]:
        """
        Match item mod lines against known stat definitions.

        Args:
            item: ParsedItem with .mods list of (mod_type, text) tuples

        Returns:
            List of ParsedMod with stat_id and extracted numeric value
        """
        if not self._loaded or not item.mods:
            return []

        results = []
        for mod_type, raw_text in item.mods:
            matched = self._match_mod(raw_text, mod_type)
            if matched:
                results.append(matched)

        if results:
            logger.debug(f"Matched {len(results)}/{len(item.mods)} mods")
        return results

    def _match_mod(self, raw_text: str, mod_type: str) -> Optional[ParsedMod]:
        """Try to match a single mod line against all stat definitions."""
        text = raw_text.strip()
        if not text:
            return None

        # Filter stats by mod type for efficiency
        type_filter = mod_type if mod_type in ("explicit", "implicit") else None

        text_lower = text.lower()

        for stat in self._stats:
            if type_filter and stat.type != type_filter:
                continue
            if not stat.regex:
                continue

            m = stat.regex.match(text)
            if m:
                # Extract first captured number as the value
                value = 0.0
                for group in m.groups():
                    if group is not None:
                        try:
                            value = float(group)
                            break
                        except ValueError:
                            continue

                # Negate value when the mod uses the opposite word from the
                # template (e.g. template says "increased", mod says "reduced")
                stat_lower = stat.text.lower()
                for positive, negative in (("increased", "reduced"), ("more", "less")):
                    if positive in stat_lower and negative in text_lower:
                        value = -value
                        break
                    if negative in stat_lower and positive in text_lower:
                        value = -value
                        break

                return ParsedMod(
                    raw_text=raw_text,
                    stat_id=stat.id,
                    value=value,
                    mod_type=mod_type,
                )

        return None

    # ─── Base Type Resolution ─────────────────────

    def resolve_base_type(self, magic_name: str) -> Optional[str]:
        """
        Extract the base type from a magic item name.

        Magic items combine prefix + base type + suffix into one name, e.g.
        "Mystic Stellar Amulet of the Fox". This finds the longest known
        base type that appears as a substring.

        Returns the base type string or None.
        """
        if not self._base_types or not magic_name:
            return None

        name_lower = magic_name.lower()

        # _base_types is sorted longest-first, so the first match
        # is the most specific (e.g. "Stellar Amulet" before "Amulet")
        for bt in self._base_types:
            if bt.lower() in name_lower:
                logger.debug(f"Resolved base type: '{magic_name}' → '{bt}'")
                return bt

        logger.debug(f"Could not resolve base type from '{magic_name}'")
        return None

    def _load_base_types(self):
        """Load base type list from disk cache or trade API items endpoint."""
        if self._load_base_types_from_disk():
            return
        self._fetch_base_types_from_api()

    def _load_base_types_from_disk(self) -> bool:
        """Load cached base types from disk."""
        try:
            if not TRADE_ITEMS_CACHE_FILE.exists():
                return False

            age = time.time() - TRADE_ITEMS_CACHE_FILE.stat().st_mtime
            if age > STATS_CACHE_MAX_AGE:
                return False

            with open(TRADE_ITEMS_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._build_base_types(data)
            logger.info(f"ModParser: loaded {len(self._base_types)} base types from disk cache")
            return True
        except Exception as e:
            logger.warning(f"ModParser: base types disk cache load failed: {e}")
            return False

    def _fetch_base_types_from_api(self) -> bool:
        """Fetch item base types from the trade API items endpoint."""
        try:
            logger.info("ModParser: fetching base types from trade API...")
            resp = requests.get(
                TRADE_ITEMS_URL,
                timeout=15,
                headers={"User-Agent": "LAMA/1.0"},
            )
            if resp.status_code != 200:
                logger.warning(f"ModParser: items API returned HTTP {resp.status_code}")
                return False

            data = resp.json()

            TRADE_ITEMS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(TRADE_ITEMS_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f)

            self._build_base_types(data)
            logger.info(f"ModParser: fetched {len(self._base_types)} base types from API")
            return True
        except Exception as e:
            logger.warning(f"ModParser: items API fetch failed: {e}")
            return False

    def _build_base_types(self, data: dict):
        """Extract unique base type names from the items API response.

        The API returns categories, each with entries that have a "type" field
        (the base type name). We collect all unique non-empty type values and
        sort them longest-first so substring matching prefers specific types.
        """
        types = set()
        for group in data.get("result", []):
            for entry in group.get("entries", []):
                base = entry.get("type", "")
                if base:
                    types.add(base)
        # Sort longest-first for greedy substring matching
        self._base_types = sorted(types, key=len, reverse=True)

    # ─── Data Loading ─────────────────────────────

    def _load_from_disk(self) -> bool:
        """Load cached stat definitions from disk."""
        try:
            if not TRADE_STATS_CACHE_FILE.exists():
                return False

            age = time.time() - TRADE_STATS_CACHE_FILE.stat().st_mtime
            if age > STATS_CACHE_MAX_AGE:
                logger.debug("Stats cache expired, will re-fetch")
                return False

            with open(TRADE_STATS_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._build_stats(data)
            logger.info(f"ModParser: loaded {len(self._stats)} stats from disk cache")
            return True
        except Exception as e:
            logger.warning(f"ModParser: disk cache load failed: {e}")
            return False

    def _fetch_from_api(self) -> bool:
        """Fetch stat definitions from the trade API."""
        try:
            logger.info("ModParser: fetching stat definitions from trade API...")
            resp = requests.get(
                TRADE_STATS_URL,
                timeout=15,
                headers={"User-Agent": "LAMA/1.0"},
            )
            if resp.status_code != 200:
                logger.warning(f"ModParser: trade stats API returned HTTP {resp.status_code}")
                return False

            data = resp.json()

            # Save to disk cache
            TRADE_STATS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(TRADE_STATS_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f)

            self._build_stats(data)
            logger.info(f"ModParser: fetched {len(self._stats)} stats from API")
            return True
        except Exception as e:
            logger.warning(f"ModParser: API fetch failed: {e}")
            return False

    def _build_stats(self, data: dict):
        """Build StatDefinition list from API response data."""
        stats = []
        result = data.get("result", [])

        for group in result:
            group_label = group.get("label", "")
            entries = group.get("entries", [])

            for entry in entries:
                stat_id = entry.get("id", "")
                stat_text = entry.get("text", "")

                if not stat_id or not stat_text:
                    continue

                # Determine mod type from the stat ID prefix
                stat_type = stat_id.split(".")[0] if "." in stat_id else group_label.lower()

                regex = _template_to_regex(stat_text)

                stats.append(StatDefinition(
                    id=stat_id,
                    text=stat_text,
                    type=stat_type,
                    regex=regex,
                ))

        self._stats = stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    mp = ModParser()
    mp.load_stats()
    print(f"Loaded {len(mp._stats)} stat definitions")

    # Show some examples
    with_regex = [s for s in mp._stats if s.regex]
    print(f"Stats with regex: {len(with_regex)}")
    for s in with_regex[:5]:
        print(f"  {s.id}: {s.text!r} → {s.regex.pattern}")

    # Test matching
    test_mods = [
        ("explicit", "+42 to maximum Life"),
        ("explicit", "+35 to Strength"),
        ("explicit", "Adds 10 to 20 Fire Damage"),
        ("explicit", "+15% to Fire Resistance"),
    ]
    print("\nTest matching:")
    for mod_type, text in test_mods:
        result = mp._match_mod(text, mod_type)
        if result:
            print(f"  '{text}' → {result.stat_id} (value={result.value})")
        else:
            print(f"  '{text}' → NO MATCH")
