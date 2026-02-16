"""
POE2 Price Overlay - Item Parser
Takes raw OCR text from tooltips/nameplates and extracts structured item data.

POE2 Tooltip Structure (when hovering):
    Line 1: Item Name (e.g., "Kaom's Heart" or "Stellar Amulet")
    Line 2: Base Type (for named items) or item class
    Line 3+: Item Level, Requirements, Mods, etc.

Ground Nameplate Structure (expanded):
    Line 1: Item Name
    Line 2: (with Show Full Descriptions) Item Level: XX
"""

import re
import logging
from typing import Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ParsedItem:
    """Structured item data extracted from OCR text."""
    name: str = ""
    base_type: str = ""
    item_class: str = ""  # "Amulets", "Body Armours", etc. from clipboard header
    rarity: str = "unknown"  # normal, magic, rare, unique, currency, gem
    item_level: int = 0
    gem_level: int = 0
    quality: int = 0
    sockets: int = 0
    stack_size: int = 1
    raw_text: str = ""
    confidence: float = 0.0
    mods: list = field(default_factory=list)
    unidentified: bool = False

    @property
    def lookup_key(self) -> str:
        """Primary key for price lookup."""
        if self.rarity == "currency":
            return self.name
        if self.rarity == "unique":
            return self.name
        if self.rarity == "gem":
            return self.name
        # For non-unique items, base type is what matters
        return self.base_type or self.name


# ─── Known Item Patterns ─────────────────────────────

# Common currency items (for quick matching without full OCR)
CURRENCY_KEYWORDS = {
    "divine orb", "exalted orb", "chaos orb", "mirror of kalandra",
    "orb of alchemy", "orb of alteration", "orb of chance", "orb of fusing",
    "orb of regret", "orb of scouring", "blessed orb", "regal orb",
    "vaal orb", "jeweller's orb", "chromatic orb", "glassblower's bauble",
    "gemcutter's prism", "cartographer's chisel", "orb of annulment",
    "orb of binding", "engineer's orb", "harbinger's orb",
    "ancient orb", "orb of horizons", "orb of unmaking",
    "scroll of wisdom", "portal scroll", "armourer's scrap",
    "blacksmith's whetstone", "silver coin",
}

# Common valuable base types in POE2
VALUABLE_BASES = {
    "stellar amulet", "astral plate", "vaal regalia", "hubris circlet",
    "sorcerer boots", "sorcerer gloves", "titanium spirit shield",
    "imbued wand", "profane wand", "opal ring", "steel ring",
    "crystal belt", "stygian vise", "two-toned boots",
    "fingerless silk gloves", "gripped gloves", "bone helmet",
    "spiked gloves", "marble amulet", "agate amulet",
}

# Rarity keywords that appear in tooltips
RARITY_PATTERNS = {
    "unique": r"\b(unique)\b",
    "rare": r"\b(rare)\b",
    "magic": r"\b(magic)\b",
    "normal": r"\b(normal)\b",
    "currency": r"\b(currency|orb|scroll|shard|splinter|fragment)\b",
    "gem": r"\b(gem|skill gem|support gem)\b",
}

# Regex patterns for extracting item properties
ITEM_LEVEL_PATTERN = re.compile(r"item\s*level[:\s]*(\d+)", re.IGNORECASE)
GEM_LEVEL_PATTERN = re.compile(r"(?:level|lvl)[:\s]*(\d+)", re.IGNORECASE)
QUALITY_PATTERN = re.compile(r"quality[:\s]*\+?(\d+)%", re.IGNORECASE)
# POE2 clipboard shows sockets as "Sockets: S S S" (one S per socket)
SOCKET_PATTERN = re.compile(r"sockets?[:\s]*((?:S\s*)+)", re.IGNORECASE)
STACK_PATTERN = re.compile(r"stack\s*size[:\s]*(\d+)", re.IGNORECASE)

# Waystone/Map patterns
WAYSTONE_PATTERN = re.compile(r"(waystone|map)\s*(?:tier)?\s*(\d+)", re.IGNORECASE)


class ItemParser:
    """
    Parses item text into structured item data.
    Supports both clipboard-format text (structured) and legacy OCR text.
    """

    def parse_clipboard(self, text: str) -> Optional[ParsedItem]:
        """
        Parse POE2 clipboard-format item text.

        POE2 copies items in a structured format:
            Item Class: Quivers
            Rarity: Unique
            Murkshaft
            Toxic Quiver
            --------
            Quiver
            Requires Level 84
            --------
            ...mods...

        Sections are separated by "--------".
        """
        if not text or len(text.strip()) < 10:
            return None

        text = text.strip()
        sections = text.split("--------")

        if not sections:
            return None

        item = ParsedItem(raw_text=text)

        # ─── Parse header section ─────────────────────
        header = sections[0].strip()
        header_lines = [l.strip() for l in header.split("\n") if l.strip()]

        for line in header_lines:
            if line.startswith("Item Class:"):
                item_class = line.split(":", 1)[1].strip()
                item.item_class = item_class
                # Use item class to help classify
                if item_class.lower() in ("currency", "stackable currency"):
                    item.rarity = "currency"
                elif "gem" in item_class.lower():
                    item.rarity = "gem"
            elif line.startswith("Rarity:"):
                rarity_str = line.split(":", 1)[1].strip().lower()
                if rarity_str in ("unique", "rare", "magic", "normal", "currency", "gem"):
                    item.rarity = rarity_str

        # Item name is the line after "Rarity:" (or last header lines)
        rarity_idx = None
        for i, line in enumerate(header_lines):
            if line.startswith("Rarity:"):
                rarity_idx = i
                break

        if rarity_idx is not None and rarity_idx + 1 < len(header_lines):
            item.name = header_lines[rarity_idx + 1]
            # Base type is the next line (for unique/rare items)
            if rarity_idx + 2 < len(header_lines):
                item.base_type = header_lines[rarity_idx + 2]

        # For currency/normal items, name IS the base type
        if item.rarity in ("currency", "normal") and not item.base_type:
            item.base_type = item.name

        # Strip quality prefixes from base type (trade API uses raw base names)
        # e.g., "Exceptional Gemini Crossbow" → "Gemini Crossbow"
        if item.base_type:
            item.base_type = self._strip_quality_prefix(item.base_type)

        # ─── Parse remaining sections for properties ──
        full_text = text
        item.item_level = self._extract_item_level(full_text)
        item.quality = self._extract_quality(full_text)
        item.gem_level = self._extract_gem_level(full_text)

        # Sockets: POE2 shows "Sockets: S S S" — count the S characters
        socket_match = SOCKET_PATTERN.search(full_text)
        if socket_match:
            item.sockets = socket_match.group(1).upper().count("S")

        # Stack size for currency
        match = STACK_PATTERN.search(full_text)
        if match:
            item.stack_size = int(match.group(1))

        # Waystone check
        waystone = self._try_waystone_match(full_text)
        if waystone:
            item.name = waystone
            item.rarity = "currency"

        # Check for unidentified items
        if "\nUnidentified" in text or text.endswith("Unidentified"):
            item.unidentified = True

        # Extract mod lines for non-unique equippable items
        if item.rarity in ("rare", "magic"):
            item.mods = self._extract_mod_lines(sections)

        if not item.name:
            return None

        logger.debug(
            f"Clipboard parsed: name='{item.name}', base='{item.base_type}', "
            f"rarity={item.rarity}, ilvl={item.item_level}, mods={len(item.mods)}"
        )
        return item

    def parse(self, ocr_text: str, detected_rarity: str = "unknown") -> Optional[ParsedItem]:
        """
        Parse OCR text into a ParsedItem.
        
        Args:
            ocr_text: Raw text from OCR engine
            detected_rarity: Rarity inferred from text color (optional)
            
        Returns:
            ParsedItem or None if text is unparseable
        """
        if not ocr_text or len(ocr_text.strip()) < 3:
            return None

        text = ocr_text.strip()
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        if not lines:
            return None

        item = ParsedItem(raw_text=text)

        # Step 1: Try currency match first (fastest path)
        currency = self._try_currency_match(lines)
        if currency:
            return currency

        # Step 2: Extract structured properties
        item.item_level = self._extract_item_level(text)
        item.quality = self._extract_quality(text)
        item.gem_level = self._extract_gem_level(text)

        # Step 3: Determine rarity
        item.rarity = self._determine_rarity(text, detected_rarity)

        # Step 4: Extract name and base type
        self._extract_name_and_base(lines, item)

        # Step 5: Check for waystone/map
        waystone = self._try_waystone_match(text)
        if waystone:
            item.name = waystone
            item.rarity = "currency"  # Waystones are priced like currency

        # Validate
        if not item.name and not item.base_type:
            return None

        logger.debug(
            f"Parsed: name='{item.name}', base='{item.base_type}', "
            f"rarity={item.rarity}, ilvl={item.item_level}"
        )
        return item

    def parse_ground_nameplate(self, ocr_text: str) -> Optional[ParsedItem]:
        """
        Parse a ground item nameplate (less info than full tooltip).
        Ground nameplates typically show just the item name.
        With "Show Full Descriptions" enabled, also shows item level.
        """
        if not ocr_text or len(ocr_text.strip()) < 3:
            return None

        lines = [l.strip() for l in ocr_text.split("\n") if l.strip()]
        if not lines:
            return None

        item = ParsedItem(raw_text=ocr_text)

        # First line is always the item name
        item.name = lines[0]

        # Check if it's currency
        if item.name.lower() in CURRENCY_KEYWORDS:
            item.rarity = "currency"
            return item

        # Check for item level in subsequent lines
        full_text = " ".join(lines)
        item.item_level = self._extract_item_level(full_text)

        # Check if the name matches a known valuable base
        if item.name.lower() in VALUABLE_BASES:
            item.base_type = item.name
            item.rarity = "normal"  # Bases are typically normal/magic

        return item

    # ─── Internal Methods ────────────────────────────

    # Quality prefixes that POE2 prepends to base type names
    _QUALITY_PREFIXES = (
        "Superior ", "Exceptional ", "Masterful ",
    )

    @staticmethod
    def _strip_quality_prefix(base_type: str) -> str:
        """Strip quality prefix from a base type name.
        e.g., 'Exceptional Gemini Crossbow' → 'Gemini Crossbow'
        """
        for prefix in ItemParser._QUALITY_PREFIXES:
            if base_type.startswith(prefix):
                return base_type[len(prefix):]
        return base_type

    # Markers that indicate non-mod sections in clipboard text
    _NON_MOD_MARKERS = frozenset({
        "corrupted", "mirrored", "split", "unmodifiable",
    })

    # POE2 appends mod type annotations like (implicit), (enchant), (rune), etc.
    _MOD_ANNOTATION_RE = re.compile(r"\s*\((implicit|enchant|rune|mutated|desecrated|fractured|crafted|augmented)\)\s*$", re.IGNORECASE)

    # Lines starting with "Grants Skill:" are skill grants, not tradeable mods
    _SKIP_LINE_RE = re.compile(
        r"^(Item Level|Level|Quality|Sockets|Requires|Rarity|Item Class|Grants Skill)",
        re.IGNORECASE,
    )

    # Lines that look like flavour text (quoted strings)
    _FLAVOUR_RE = re.compile(r'^".*"$|^- .+$')

    def _extract_mod_lines(self, sections: list) -> list:
        """
        Extract mod lines from clipboard sections for rare items.

        Returns a list of (mod_type, text) tuples where mod_type is
        "explicit", "implicit", "enchant", "rune", etc.

        POE2 clipboard appends annotations like (implicit), (enchant), (rune)
        to mod lines. We use these to classify mods and strip them from the text.
        Unannotated mods in the final section are assumed to be explicit.
        """
        mods = []

        # Find the section index containing "Item Level:"
        ilvl_idx = None
        for i, section in enumerate(sections):
            if "Item Level:" in section:
                ilvl_idx = i
                break

        if ilvl_idx is None:
            return mods

        # Collect all lines from sections after Item Level
        mod_sections = sections[ilvl_idx + 1:]

        for section in mod_sections:
            section_text = section.strip()
            if not section_text:
                continue

            lines = [l.strip() for l in section_text.split("\n") if l.strip()]

            # Skip terminal markers
            if len(lines) == 1 and lines[0].lower() in self._NON_MOD_MARKERS:
                continue

            # Skip "Note:" sections (player-added notes)
            if any(l.startswith("Note:") for l in lines):
                continue

            for line in lines:
                # Skip property/requirement lines
                if self._SKIP_LINE_RE.match(line):
                    continue
                # Skip flavour text
                if self._FLAVOUR_RE.match(line):
                    continue

                # Check for annotation suffix: (implicit), (enchant), (rune), etc.
                ann_match = self._MOD_ANNOTATION_RE.search(line)
                if ann_match:
                    mod_type = ann_match.group(1).lower()
                    clean_text = line[:ann_match.start()].strip()
                else:
                    mod_type = "explicit"
                    clean_text = line

                if clean_text:
                    mods.append((mod_type, clean_text))

        return mods

    def _try_currency_match(self, lines: List[str]) -> Optional[ParsedItem]:
        """Quick check if this is a currency item."""
        for line in lines[:3]:  # Currency name is usually in first few lines
            clean = line.lower().strip()
            if clean in CURRENCY_KEYWORDS:
                item = ParsedItem(
                    name=line.strip(),
                    rarity="currency",
                    raw_text="\n".join(lines),
                )
                # Try to get stack size
                full = " ".join(lines)
                match = STACK_PATTERN.search(full)
                if match:
                    item.stack_size = int(match.group(1))
                return item

            # Partial match (OCR might miss a character)
            for currency in CURRENCY_KEYWORDS:
                if self._similar(clean, currency):
                    item = ParsedItem(
                        name=currency.title(),
                        rarity="currency",
                        raw_text="\n".join(lines),
                    )
                    return item
        return None

    def _determine_rarity(self, text: str, color_rarity: str) -> str:
        """Determine item rarity from text content and color hint."""
        text_lower = text.lower()

        # Color-based rarity is more reliable if available
        if color_rarity and color_rarity != "unknown":
            return color_rarity

        # Check text for rarity indicators
        if any(c in text_lower for c in ["orb", "scroll", "shard", "fragment"]):
            return "currency"
        if "unique" in text_lower:
            return "unique"
        if "rare" in text_lower:
            return "rare"
        if "magic" in text_lower:
            return "magic"
        if "skill gem" in text_lower or "support gem" in text_lower:
            return "gem"

        return "unknown"

    def _extract_name_and_base(self, lines: list, item: ParsedItem):
        """
        Extract item name and base type from tooltip lines.
        
        For unique items: Line 1 = unique name, Line 2 = base type
        For rare items: Line 1 = random name, Line 2 = base type
        For normal/magic: Line 1 = base type (possibly with prefix/suffix)
        """
        if not lines:
            return

        # Filter out property lines (contain numbers, colons, %)
        name_lines = []
        for line in lines:
            # Skip lines that look like properties
            if re.match(r"^(item level|level|quality|sockets|requires|rarity)", line, re.IGNORECASE):
                continue
            if re.match(r"^\d+[\s-]", line):  # Lines starting with numbers
                continue
            if ":" in line and any(c.isdigit() for c in line):  # "Stat: 123"
                continue
            name_lines.append(line)

        if not name_lines:
            item.name = lines[0]
            return

        item.name = name_lines[0]

        # Second name line is typically the base type
        if len(name_lines) >= 2 and item.rarity in ("unique", "rare"):
            item.base_type = name_lines[1]

    def _extract_item_level(self, text: str) -> int:
        """Extract item level from text."""
        match = ITEM_LEVEL_PATTERN.search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return 0

    def _extract_gem_level(self, text: str) -> int:
        """Extract gem level from text."""
        match = GEM_LEVEL_PATTERN.search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return 0

    def _extract_quality(self, text: str) -> int:
        """Extract quality percentage from text."""
        match = QUALITY_PATTERN.search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return 0

    def _try_waystone_match(self, text: str) -> Optional[str]:
        """Check if this is a waystone/map item."""
        match = WAYSTONE_PATTERN.search(text)
        if match:
            return f"Waystone (Tier {match.group(2)})"
        return None

    def _similar(self, a: str, b: str, threshold: float = 0.85) -> bool:
        """Simple string similarity check for OCR error tolerance."""
        if not a or not b:
            return False
        if abs(len(a) - len(b)) > 3:
            return False

        # Character-level matching
        matches = sum(1 for ca, cb in zip(a, b) if ca == cb)
        max_len = max(len(a), len(b))
        return (matches / max_len) >= threshold if max_len > 0 else False


# ─── Quick Test ──────────────────────────────────────

if __name__ == "__main__":
    parser = ItemParser()

    # Test clipboard-format texts (POE2 Ctrl+C output)
    clipboard_tests = [
        # Unique item
        "Item Class: Body Armours\nRarity: Unique\nKaom's Heart\nGlorious Plate\n--------\nArmour: 553\n--------\nItem Level: 84",
        # Currency
        "Item Class: Stackable Currency\nRarity: Currency\nDivine Orb\n--------\nStack Size: 3/20",
        # Gem
        "Item Class: Skill Gems\nRarity: Gem\nIce Nova\n--------\nLevel: 20\nQuality: +20%",
        # Rare
        "Item Class: Shields\nRarity: Rare\nApocalypse Hold\nTitanium Spirit Shield\n--------\nQuality: +20%\n--------\nItem Level: 86",
    ]

    print("=== Clipboard format tests ===")
    for text in clipboard_tests:
        result = parser.parse_clipboard(text)
        if result:
            print(f"Input: {text.split(chr(10))[2]}")  # Name is 3rd line
            print(f"  > name='{result.name}', base='{result.base_type}', "
                  f"rarity={result.rarity}, ilvl={result.item_level}, "
                  f"lookup_key='{result.lookup_key}'")
            print()

    # Legacy OCR format tests
    print("=== Legacy OCR format tests ===")
    ocr_tests = [
        "Divine Orb\nStack Size: 3",
        "Kaom's Heart\nGlorious Plate\nItem Level: 84",
        "Waystone Tier 16",
    ]

    for text in ocr_tests:
        result = parser.parse(text)
        if result:
            print(f"Input: {text.split(chr(10))[0]}")
            print(f"  > name='{result.name}', base='{result.base_type}', "
                  f"rarity={result.rarity}, ilvl={result.item_level}, "
                  f"lookup_key='{result.lookup_key}'")
            print()
