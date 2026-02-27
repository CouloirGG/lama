"""
Build Guide Companion — scrape, parse, store, and compare build guides.

Supports Maxroll.gg and Mobalytics guide formats with a generic fallback.
"""

import hashlib
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger("guide_scraper")

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
GUIDES_DIR = Path.home() / ".poe2-price-overlay" / "guides"


def _ensure_dir():
    GUIDES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------
class GuideStage(str, Enum):
    ACT_1 = "act1"
    ACT_2 = "act2"
    ACT_3 = "act3"
    ACT_4 = "act4"
    INTERLUDE = "interlude"
    ENDGAME = "endgame"


STAGE_RANGES: List[Tuple[GuideStage, int, int]] = [
    (GuideStage.ACT_1, 1, 12),
    (GuideStage.ACT_2, 13, 24),
    (GuideStage.ACT_3, 25, 36),
    (GuideStage.ACT_4, 37, 48),
    (GuideStage.INTERLUDE, 49, 59),
    (GuideStage.ENDGAME, 60, 100),
]

STAGE_LABELS = {
    GuideStage.ACT_1: "Act I",
    GuideStage.ACT_2: "Act II",
    GuideStage.ACT_3: "Act III",
    GuideStage.ACT_4: "Act IV",
    GuideStage.INTERLUDE: "Interlude",
    GuideStage.ENDGAME: "Endgame",
}


def stage_for_level(level: int) -> GuideStage:
    for stage, lo, hi in STAGE_RANGES:
        if lo <= level <= hi:
            return stage
    return GuideStage.ENDGAME


@dataclass
class GuideGearItem:
    slot: str                       # weapon, helmet, body, gloves, boots, belt, ring, ring2, amulet, shield
    name: str                       # item name or base type
    is_unique: bool = False
    key_mods: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    priority: str = "recommended"   # required | recommended | optional


@dataclass
class GuideSkillSetup:
    name: str
    supports: List[str] = field(default_factory=list)
    is_main: bool = False
    notes: str = ""


@dataclass
class GuidePassives:
    keystones: List[str] = field(default_factory=list)
    notable_priorities: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class GuideStageData:
    stage: str                      # GuideStage value
    level_range: Tuple[int, int] = (1, 100)
    gear: List[GuideGearItem] = field(default_factory=list)
    skills: List[GuideSkillSetup] = field(default_factory=list)
    passives: GuidePassives = field(default_factory=GuidePassives)
    notes: str = ""


@dataclass
class ParsedGuide:
    id: str = ""
    title: str = ""
    url: str = ""
    source: str = "unknown"         # maxroll | mobalytics | unknown
    char_class: str = ""
    ascendancy: str = ""
    main_skill: str = ""
    stages: List[GuideStageData] = field(default_factory=list)
    imported_at: str = ""
    raw_html_hash: str = ""


# ---------------------------------------------------------------------------
# Comparison output
# ---------------------------------------------------------------------------
@dataclass
class SlotComparison:
    slot: str = ""
    guide_item_name: str = ""
    guide_item_is_unique: bool = False
    guide_key_mods: List[str] = field(default_factory=list)
    current_item: str = ""
    status: str = "missing"         # match | partial | missing | upgrade_needed
    explanation: str = ""
    price_display: str = ""         # ~5.2d format


@dataclass
class SkillComparison:
    skill_name: str = ""
    status: str = "missing"         # match | missing | wrong_supports
    missing_supports: List[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class GuideComparison:
    current_stage: str = ""
    player_level: int = 0
    gear_matches: List[SlotComparison] = field(default_factory=list)
    skill_matches: List[SkillComparison] = field(default_factory=list)
    passive_matches: Dict[str, Any] = field(default_factory=dict)
    overall_score: float = 0.0
    next_upgrades: List[SlotComparison] = field(default_factory=list)
    stage_notes: str = ""
    building_right: str = "on track"


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------
def _guide_to_dict(g: ParsedGuide) -> dict:
    d = asdict(g)
    return d


def _dict_to_guide(d: dict) -> ParsedGuide:
    stages = []
    for sd in d.get("stages", []):
        passives_d = sd.get("passives", {})
        passives = GuidePassives(
            keystones=passives_d.get("keystones", []),
            notable_priorities=passives_d.get("notable_priorities", []),
            notes=passives_d.get("notes", ""),
        )
        gear = [GuideGearItem(**gi) for gi in sd.get("gear", [])]
        skills = [GuideSkillSetup(**si) for si in sd.get("skills", [])]
        lr = sd.get("level_range", [1, 100])
        stages.append(GuideStageData(
            stage=sd.get("stage", "endgame"),
            level_range=tuple(lr) if isinstance(lr, list) else lr,
            gear=gear,
            skills=skills,
            passives=passives,
            notes=sd.get("notes", ""),
        ))
    return ParsedGuide(
        id=d.get("id", ""),
        title=d.get("title", ""),
        url=d.get("url", ""),
        source=d.get("source", "unknown"),
        char_class=d.get("char_class", ""),
        ascendancy=d.get("ascendancy", ""),
        main_skill=d.get("main_skill", ""),
        stages=stages,
        imported_at=d.get("imported_at", ""),
        raw_html_hash=d.get("raw_html_hash", ""),
    )


# ---------------------------------------------------------------------------
# Storage (CRUD)
# ---------------------------------------------------------------------------
def save_guide(guide: ParsedGuide) -> str:
    _ensure_dir()
    path = GUIDES_DIR / f"{guide.id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_guide_to_dict(guide), f, indent=2, ensure_ascii=False)
    return guide.id


def load_guide(guide_id: str) -> Optional[ParsedGuide]:
    path = GUIDES_DIR / f"{guide_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return _dict_to_guide(json.load(f))


def list_guides() -> List[dict]:
    _ensure_dir()
    results = []
    for p in sorted(GUIDES_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            results.append({
                "id": d.get("id", p.stem),
                "title": d.get("title", "Untitled"),
                "source": d.get("source", "unknown"),
                "char_class": d.get("char_class", ""),
                "ascendancy": d.get("ascendancy", ""),
                "main_skill": d.get("main_skill", ""),
                "imported_at": d.get("imported_at", ""),
            })
        except Exception:
            continue
    return results


def delete_guide(guide_id: str) -> bool:
    path = GUIDES_DIR / f"{guide_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False


# ---------------------------------------------------------------------------
# Source Detection
# ---------------------------------------------------------------------------
def detect_source(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if "maxroll" in host:
        return "maxroll"
    if "mobalytics" in host:
        return "mobalytics"
    return "unknown"


# ---------------------------------------------------------------------------
# HTML fetching
# ---------------------------------------------------------------------------
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "LAMA/0.2 (Build Guide Companion)",
    "Accept": "text/html,application/xhtml+xml",
})


def fetch_html(url: str) -> str:
    resp = _SESSION.get(url, timeout=20)
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# Parser Helpers
# ---------------------------------------------------------------------------
_SLOT_ALIASES = {
    "helm": "helmet", "head": "helmet", "hat": "helmet",
    "chest": "body", "body armour": "body", "body armor": "body",
    "glove": "gloves", "gauntlets": "gloves",
    "boot": "boots", "shoes": "boots",
    "ring 1": "ring", "ring 2": "ring2", "left ring": "ring", "right ring": "ring2",
    "amulet": "amulet", "neck": "amulet", "necklace": "amulet",
    "offhand": "shield", "off-hand": "shield", "off hand": "shield",
    "main hand": "weapon", "mainhand": "weapon",
    "two-hand": "weapon", "2h weapon": "weapon",
    "flask": "flask",
    "jewel": "jewel",
}

POE2_CLASSES = [
    "Witch", "Warrior", "Ranger", "Mercenary", "Monk", "Sorceress",
]
POE2_ASCENDANCIES = [
    "Blood Mage", "Infernalist",
    "Titan", "Warbringer",
    "Deadeye", "Pathfinder",
    "Witchhunter", "Gemling Legionnaire",
    "Acolyte of Chayula", "Invoker",
    "Chronomancer", "Stormweaver",
]


def _normalize_slot(raw: str) -> str:
    low = raw.strip().lower()
    return _SLOT_ALIASES.get(low, low)


def _clean_text(el) -> str:
    if el is None:
        return ""
    return re.sub(r"\s+", " ", el.get_text(strip=True))


def _hash_html(html: str) -> str:
    return hashlib.sha256(html.encode("utf-8")).hexdigest()[:16]


def _detect_class_ascendancy(text: str) -> Tuple[str, str]:
    """Try to detect class/ascendancy from page text."""
    text_lower = text.lower()
    asc = ""
    cls = ""
    for a in POE2_ASCENDANCIES:
        if a.lower() in text_lower:
            asc = a
            break
    for c in POE2_CLASSES:
        if c.lower() in text_lower:
            cls = c
            break
    return cls, asc


# ---------------------------------------------------------------------------
# Maxroll Parser
# ---------------------------------------------------------------------------
class MaxrollParser:
    """Parse Maxroll.gg POE2 build guides."""

    # Maxroll uses tabbed stage sections. The exact structure may vary
    # but typically has sections like "Leveling", "Early Endgame", "Endgame".

    STAGE_KEYWORDS = {
        GuideStage.ACT_1: ["act 1", "act i", "level 1", "start"],
        GuideStage.ACT_2: ["act 2", "act ii", "level 13"],
        GuideStage.ACT_3: ["act 3", "act iii", "level 25"],
        GuideStage.ACT_4: ["act 4", "act iv", "level 37"],
        GuideStage.INTERLUDE: ["interlude", "level 49", "cruel"],
        GuideStage.ENDGAME: ["endgame", "end game", "end-game", "maps", "level 60"],
    }

    # Broader stage grouping keywords for when guides use fewer stages
    BROAD_STAGE_MAP = {
        "leveling": [GuideStage.ACT_1, GuideStage.ACT_2, GuideStage.ACT_3, GuideStage.ACT_4],
        "early maps": [GuideStage.INTERLUDE],
        "early endgame": [GuideStage.INTERLUDE],
        "mid endgame": [GuideStage.ENDGAME],
        "late endgame": [GuideStage.ENDGAME],
        "endgame": [GuideStage.ENDGAME],
    }

    @classmethod
    def parse(cls, html: str, url: str) -> ParsedGuide:
        soup = BeautifulSoup(html, "html.parser")
        guide = ParsedGuide(
            id=str(uuid.uuid4()),
            url=url,
            source="maxroll",
            imported_at=datetime.now(timezone.utc).isoformat(),
            raw_html_hash=_hash_html(html),
        )

        # Title
        title_el = soup.find("h1")
        guide.title = _clean_text(title_el) if title_el else "Maxroll Guide"

        # Class/Ascendancy detection
        page_text = soup.get_text(" ", strip=True)
        guide.char_class, guide.ascendancy = _detect_class_ascendancy(page_text)

        # Try to detect main skill from title
        guide.main_skill = cls._detect_main_skill(guide.title, page_text)

        # Parse stages
        guide.stages = cls._parse_stages(soup, page_text)

        # If no stages found, create a single endgame stage from the whole page
        if not guide.stages:
            stage = cls._parse_whole_page(soup)
            if stage:
                guide.stages = [stage]

        return guide

    @classmethod
    def _detect_main_skill(cls, title: str, page_text: str) -> str:
        """Extract main skill name from guide title."""
        # Maxroll titles are often like "Ice Nova Sorceress Build Guide"
        # or "Lightning Arrow Deadeye Guide"
        title_clean = re.sub(r"\b(build|guide|poe2?|path of exile 2?)\b", "", title, flags=re.I).strip()
        # Remove class/ascendancy names
        for name in POE2_CLASSES + POE2_ASCENDANCIES:
            title_clean = re.sub(re.escape(name), "", title_clean, flags=re.I)
        title_clean = re.sub(r"\s+", " ", title_clean).strip(" -–—|")
        return title_clean if title_clean else ""

    @classmethod
    def _parse_stages(cls, soup: BeautifulSoup, page_text: str) -> List[GuideStageData]:
        """Try to find stage-based sections in the guide."""
        stages = []

        # Look for tab buttons or section headers that denote stages
        # Maxroll often uses div.tab-content or similar
        tab_sections = soup.find_all(["section", "div"], class_=re.compile(
            r"tab[-_]?(content|panel|pane)|stage|section[-_]?content", re.I
        ))

        if tab_sections:
            for section in tab_sections:
                stage = cls._classify_and_parse_section(section)
                if stage:
                    stages.append(stage)

        # Also look for h2/h3 headers that denote stages
        if not stages:
            for header in soup.find_all(["h2", "h3"]):
                header_text = _clean_text(header).lower()
                stage_type = cls._classify_stage_text(header_text)
                if stage_type:
                    # Get content between this header and the next same-level header
                    content_els = []
                    for sib in header.find_next_siblings():
                        if sib.name in ["h2", "h3"]:
                            break
                        content_els.append(sib)
                    if content_els:
                        fake_div = soup.new_tag("div")
                        for el in content_els:
                            fake_div.append(el.__copy__() if hasattr(el, '__copy__') else el)
                        stage = cls._parse_section_content(stage_type, fake_div)
                        if stage:
                            stages.append(stage)

        return stages

    @classmethod
    def _classify_stage_text(cls, text: str) -> Optional[GuideStage]:
        text_lower = text.lower()
        for stage, keywords in cls.STAGE_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return stage
        # Broader grouping
        for keyword, stage_list in cls.BROAD_STAGE_MAP.items():
            if keyword in text_lower:
                return stage_list[-1]  # Use the last (most advanced) stage
        return None

    @classmethod
    def _classify_and_parse_section(cls, section: Tag) -> Optional[GuideStageData]:
        # Check section text or data attributes for stage identification
        section_text = ""
        header = section.find(["h1", "h2", "h3", "h4"])
        if header:
            section_text = _clean_text(header).lower()
        if not section_text:
            # Check for data attributes or class names
            section_text = " ".join([
                section.get("data-tab", ""),
                section.get("data-label", ""),
                section.get("id", ""),
                " ".join(section.get("class", [])),
            ]).lower()

        stage_type = cls._classify_stage_text(section_text)
        if not stage_type:
            return None
        return cls._parse_section_content(stage_type, section)

    @classmethod
    def _parse_section_content(cls, stage_type: GuideStage, section: Tag) -> GuideStageData:
        for s, lo, hi in STAGE_RANGES:
            if s == stage_type:
                level_range = (lo, hi)
                break
        else:
            level_range = (1, 100)

        gear = cls._extract_gear(section)
        skills = cls._extract_skills(section)
        passives = cls._extract_passives(section)
        notes = ""
        # Look for general notes/tips paragraphs
        for p in section.find_all("p"):
            txt = _clean_text(p)
            if len(txt) > 40:
                notes = txt[:500]
                break

        return GuideStageData(
            stage=stage_type.value,
            level_range=level_range,
            gear=gear,
            skills=skills,
            passives=passives,
            notes=notes,
        )

    @classmethod
    def _extract_gear(cls, section: Tag) -> List[GuideGearItem]:
        """Extract gear recommendations from a section."""
        gear = []
        seen_slots = set()

        # Look for gear tables
        for table in section.find_all("table"):
            for row in table.find_all("tr"):
                cells = row.find_all(["td", "th"])
                if len(cells) >= 2:
                    slot_text = _clean_text(cells[0])
                    item_text = _clean_text(cells[1])
                    slot = _normalize_slot(slot_text)
                    if slot and item_text and slot not in seen_slots:
                        mods = []
                        if len(cells) > 2:
                            mods = [_clean_text(cells[2])]
                        gear.append(GuideGearItem(
                            slot=slot,
                            name=item_text,
                            is_unique=cls._looks_unique(item_text),
                            key_mods=mods,
                        ))
                        seen_slots.add(slot)

        # Look for gear in list items with slot labels
        if not gear:
            for li in section.find_all("li"):
                text = _clean_text(li)
                match = re.match(r"([\w\s-]+?):\s*(.+)", text)
                if match:
                    slot = _normalize_slot(match.group(1))
                    if slot in ("weapon", "helmet", "body", "gloves", "boots",
                                "belt", "ring", "ring2", "amulet", "shield",
                                "flask", "jewel"):
                        name = match.group(2).strip()
                        if slot not in seen_slots:
                            gear.append(GuideGearItem(
                                slot=slot,
                                name=name,
                                is_unique=cls._looks_unique(name),
                            ))
                            seen_slots.add(slot)

        # Look for gear in divs/spans with slot-like labels (Maxroll card-style)
        if not gear:
            for el in section.find_all(["div", "span"], class_=re.compile(r"item|gear|slot|equip", re.I)):
                text = _clean_text(el)
                if len(text) > 3 and len(text) < 200:
                    # Try to parse "Slot: Item" pattern
                    match = re.match(r"([\w\s-]+?):\s*(.+)", text)
                    if match:
                        slot = _normalize_slot(match.group(1))
                        if slot and slot not in seen_slots:
                            gear.append(GuideGearItem(
                                slot=slot,
                                name=match.group(2).strip(),
                                is_unique=cls._looks_unique(match.group(2)),
                            ))
                            seen_slots.add(slot)

        return gear

    @classmethod
    def _extract_skills(cls, section: Tag) -> List[GuideSkillSetup]:
        """Extract skill gem setups from a section."""
        skills = []
        seen_skills = set()

        # Look for skill sections - often in grouped containers
        # Pattern: main skill name + list of support gems
        for el in section.find_all(["div", "ul", "ol", "table"], class_=re.compile(
            r"skill|gem|link|setup|socket", re.I
        )):
            items = el.find_all("li")
            if items:
                # First item is often the main skill
                main = _clean_text(items[0])
                supports = [_clean_text(i) for i in items[1:] if _clean_text(i)]
                if main and main not in seen_skills:
                    skills.append(GuideSkillSetup(
                        name=main,
                        supports=supports,
                        is_main=len(skills) == 0,
                    ))
                    seen_skills.add(main)

        # Fallback: look for "Skill Gems" or similar headers followed by lists
        if not skills:
            for header in section.find_all(["h3", "h4", "h5", "strong"]):
                header_text = _clean_text(header).lower()
                if any(kw in header_text for kw in ("skill", "gem", "setup", "link")):
                    # Get next list
                    next_list = header.find_next(["ul", "ol"])
                    if next_list:
                        items = next_list.find_all("li")
                        group_name = ""
                        supports = []
                        for item in items:
                            txt = _clean_text(item)
                            if not txt:
                                continue
                            if not group_name:
                                group_name = txt
                            else:
                                supports.append(txt)
                        if group_name and group_name not in seen_skills:
                            skills.append(GuideSkillSetup(
                                name=group_name,
                                supports=supports,
                                is_main=len(skills) == 0,
                            ))
                            seen_skills.add(group_name)

        return skills

    @classmethod
    def _extract_passives(cls, section: Tag) -> GuidePassives:
        """Extract passive/keystone recommendations."""
        keystones = []
        notables = []
        notes = ""

        for el in section.find_all(["div", "section", "ul", "ol"], class_=re.compile(
            r"passive|keystone|tree|notable|ascend", re.I
        )):
            for li in el.find_all("li"):
                txt = _clean_text(li)
                if txt:
                    # Heuristic: keystones are often capitalized multi-word names
                    if len(txt.split()) <= 4 and txt[0].isupper():
                        keystones.append(txt)
                    else:
                        notables.append(txt)

        # Fallback: look for keystone mentions in headers
        if not keystones:
            for header in section.find_all(["h3", "h4", "h5", "strong"]):
                txt = _clean_text(header).lower()
                if "keystone" in txt or "passive" in txt or "ascendancy" in txt:
                    next_el = header.find_next(["ul", "ol", "p"])
                    if next_el:
                        items = next_el.find_all("li") if next_el.name in ["ul", "ol"] else [next_el]
                        for item in items:
                            t = _clean_text(item)
                            if t and len(t) < 60:
                                keystones.append(t)
                        if next_el.name == "p":
                            notes = _clean_text(next_el)[:300]

        return GuidePassives(keystones=keystones, notable_priorities=notables, notes=notes)

    @classmethod
    def _parse_whole_page(cls, soup: BeautifulSoup) -> Optional[GuideStageData]:
        """Fallback: parse the whole page as a single endgame stage."""
        body = soup.find("body") or soup
        gear = cls._extract_gear(body)
        skills = cls._extract_skills(body)
        passives = cls._extract_passives(body)

        if not gear and not skills and not passives.keystones:
            return None

        return GuideStageData(
            stage=GuideStage.ENDGAME.value,
            level_range=(1, 100),
            gear=gear,
            skills=skills,
            passives=passives,
            notes="Parsed from full page (no stage sections detected)",
        )

    @classmethod
    def _looks_unique(cls, name: str) -> bool:
        """Heuristic: unique items are typically proper names."""
        if not name:
            return False
        # If it contains "any" or generic words, probably not unique
        if re.search(r"\b(any|rare|magic|normal|base|with)\b", name, re.I):
            return False
        # If it looks like a specific named item (2+ capitalized words, no mod-like text)
        words = name.split()
        if len(words) >= 1 and all(w[0].isupper() for w in words if len(w) > 2):
            return True
        return False


# ---------------------------------------------------------------------------
# Mobalytics Parser
# ---------------------------------------------------------------------------
class MobalyticsParser:
    """Parse Mobalytics POE2 build guides."""

    @classmethod
    def parse(cls, html: str, url: str) -> ParsedGuide:
        soup = BeautifulSoup(html, "html.parser")
        guide = ParsedGuide(
            id=str(uuid.uuid4()),
            url=url,
            source="mobalytics",
            imported_at=datetime.now(timezone.utc).isoformat(),
            raw_html_hash=_hash_html(html),
        )

        title_el = soup.find("h1")
        guide.title = _clean_text(title_el) if title_el else "Mobalytics Guide"

        page_text = soup.get_text(" ", strip=True)
        guide.char_class, guide.ascendancy = _detect_class_ascendancy(page_text)
        guide.main_skill = MaxrollParser._detect_main_skill(guide.title, page_text)

        # Mobalytics uses card-based layouts — try same extraction logic
        guide.stages = MaxrollParser._parse_stages(soup, page_text)

        if not guide.stages:
            stage = MaxrollParser._parse_whole_page(soup)
            if stage:
                guide.stages = [stage]

        return guide


# ---------------------------------------------------------------------------
# Generic Parser (fallback)
# ---------------------------------------------------------------------------
class GenericParser:
    """Best-effort parser for any build guide page."""

    @classmethod
    def parse(cls, html: str, url: str) -> ParsedGuide:
        soup = BeautifulSoup(html, "html.parser")
        guide = ParsedGuide(
            id=str(uuid.uuid4()),
            url=url,
            source="unknown",
            imported_at=datetime.now(timezone.utc).isoformat(),
            raw_html_hash=_hash_html(html),
        )

        title_el = soup.find("h1") or soup.find("title")
        guide.title = _clean_text(title_el) if title_el else "Build Guide"

        page_text = soup.get_text(" ", strip=True)
        guide.char_class, guide.ascendancy = _detect_class_ascendancy(page_text)
        guide.main_skill = MaxrollParser._detect_main_skill(guide.title, page_text)

        guide.stages = MaxrollParser._parse_stages(soup, page_text)

        if not guide.stages:
            stage = MaxrollParser._parse_whole_page(soup)
            if stage:
                guide.stages = [stage]

        return guide


# ---------------------------------------------------------------------------
# Import entry point
# ---------------------------------------------------------------------------
PARSERS = {
    "maxroll": MaxrollParser,
    "mobalytics": MobalyticsParser,
    "unknown": GenericParser,
}


def import_guide(url: str) -> ParsedGuide:
    """Fetch a guide URL, parse it, save to disk, and return the result."""
    source = detect_source(url)
    html = fetch_html(url)
    parser = PARSERS.get(source, GenericParser)
    guide = parser.parse(html, url)

    if not guide.stages:
        raise ValueError("Could not extract any guide content from this page. "
                         "The page structure may not be supported yet.")

    save_guide(guide)
    logger.info("Imported guide %s (%s) with %d stages", guide.id, guide.title, len(guide.stages))
    return guide


# ---------------------------------------------------------------------------
# Comparison Engine
# ---------------------------------------------------------------------------
def compare_character_to_guide(
    char_data: dict,
    guide: ParsedGuide,
    price_cache=None,
) -> dict:
    """
    Compare a character (serialized dict from builds_client) against a guide.
    Returns a GuideComparison as dict.

    char_data keys: name, char_class, ascendancy, level, equipment (list of item dicts),
                    skill_groups (list of {gems, dps}), keystones (list of str).
    """
    level = char_data.get("level", 60)
    current_stage = stage_for_level(level)

    # Find the matching stage data in the guide
    stage_data = None
    for s in guide.stages:
        if s.stage == current_stage.value:
            stage_data = s
            break
    # If exact stage not found, use the closest available
    if not stage_data and guide.stages:
        stage_data = guide.stages[-1]  # Default to last (most advanced)
        # Try to find the closest earlier stage
        stage_order = [s.value for s in GuideStage]
        current_idx = stage_order.index(current_stage.value)
        for s in reversed(guide.stages):
            s_idx = stage_order.index(s.stage)
            if s_idx <= current_idx:
                stage_data = s
                break

    if not stage_data:
        return asdict(GuideComparison(
            current_stage=current_stage.value,
            player_level=level,
            building_right="No guide data for this stage",
        ))

    # --- Gear comparison ---
    equipment = char_data.get("equipment", [])
    equip_by_slot = {}
    for item in equipment:
        slot = item.get("slot", "").lower()
        if slot:
            equip_by_slot[slot] = item

    gear_matches = []
    for guide_item in stage_data.gear:
        slot = guide_item.slot.lower()
        player_item = equip_by_slot.get(slot)

        sc = SlotComparison(
            slot=guide_item.slot,
            guide_item_name=guide_item.name,
            guide_item_is_unique=guide_item.is_unique,
            guide_key_mods=guide_item.key_mods,
        )

        if not player_item:
            sc.status = "missing"
            sc.explanation = f"No item equipped in {guide_item.slot}"
        else:
            player_name = player_item.get("name", "") or player_item.get("typeLine", "")
            sc.current_item = player_name

            if guide_item.is_unique:
                # Check if player has the specific unique
                if _fuzzy_name_match(player_name, guide_item.name):
                    sc.status = "match"
                    sc.explanation = "Correct unique equipped"
                else:
                    sc.status = "upgrade_needed"
                    sc.explanation = f"Guide recommends {guide_item.name}"
            else:
                # Rare gear — check if key mods are present
                player_mods = (
                    player_item.get("explicitMods", []) +
                    player_item.get("explicit_mods", []) +
                    player_item.get("implicitMods", []) +
                    player_item.get("implicit_mods", [])
                )
                if guide_item.key_mods:
                    matched_mods = sum(
                        1 for m in guide_item.key_mods
                        if any(_fuzzy_name_match(pm, m) for pm in player_mods)
                    )
                    if matched_mods == len(guide_item.key_mods):
                        sc.status = "match"
                        sc.explanation = "All key mods present"
                    elif matched_mods > 0:
                        sc.status = "partial"
                        sc.explanation = f"{matched_mods}/{len(guide_item.key_mods)} key mods"
                    else:
                        sc.status = "upgrade_needed"
                        sc.explanation = f"Missing key mods: {', '.join(guide_item.key_mods)}"
                else:
                    # No specific mods to check — having something in the slot counts
                    sc.status = "match"
                    sc.explanation = "Slot filled"

        # Price lookup for guide item
        if price_cache and guide_item.is_unique:
            price = price_cache.lookup(guide_item.name)
            if price:
                sc.price_display = price.get("display", "")

        gear_matches.append(sc)

    # --- Skills comparison ---
    player_skills = set()
    player_supports_by_skill = {}
    for sg in char_data.get("skill_groups", []):
        gems = sg.get("gems", [])
        if gems:
            main = gems[0]
            player_skills.add(main.lower())
            player_supports_by_skill[main.lower()] = [g.lower() for g in gems[1:]]

    skill_matches = []
    for guide_skill in stage_data.skills:
        sk = SkillComparison(skill_name=guide_skill.name)
        skill_lower = guide_skill.name.lower()

        if skill_lower in player_skills:
            # Check supports
            player_sups = player_supports_by_skill.get(skill_lower, [])
            missing = [s for s in guide_skill.supports
                       if s.lower() not in player_sups]
            if not missing:
                sk.status = "match"
                sk.explanation = "Skill and all supports present"
            else:
                sk.status = "wrong_supports"
                sk.missing_supports = missing
                sk.explanation = f"Missing supports: {', '.join(missing)}"
        else:
            sk.status = "missing"
            sk.explanation = f"{guide_skill.name} not found in skill setup"

        skill_matches.append(sk)

    # --- Passives comparison ---
    player_keystones = set(k.lower() for k in char_data.get("keystones", []))
    guide_keystones = stage_data.passives.keystones
    ks_matched = [k for k in guide_keystones if k.lower() in player_keystones]
    ks_missing = [k for k in guide_keystones if k.lower() not in player_keystones]
    passive_matches = {
        "keystones_matched": ks_matched,
        "keystones_missing": ks_missing,
        "notable_priorities": stage_data.passives.notable_priorities,
        "notes": stage_data.passives.notes,
    }

    # --- Scoring ---
    total_checks = 0
    matched_checks = 0

    for gm in gear_matches:
        total_checks += 1
        if gm.status == "match":
            matched_checks += 1
        elif gm.status == "partial":
            matched_checks += 0.5

    for sm in skill_matches:
        total_checks += 1
        if sm.status == "match":
            matched_checks += 1
        elif sm.status == "wrong_supports":
            matched_checks += 0.5

    if guide_keystones:
        total_checks += len(guide_keystones)
        matched_checks += len(ks_matched)

    overall_score = (matched_checks / total_checks * 100) if total_checks > 0 else 0

    # --- Next upgrades (priority items not yet matched) ---
    next_upgrades = [
        gm for gm in gear_matches
        if gm.status in ("missing", "upgrade_needed")
    ]

    # --- Building right signal ---
    if overall_score >= 80:
        building_right = "on track"
    elif overall_score >= 50:
        building_right = "minor divergence"
    else:
        building_right = "off path"

    # Class mismatch warning
    if guide.char_class and char_data.get("char_class"):
        if guide.char_class.lower() != char_data["char_class"].lower():
            building_right = f"Class mismatch: guide is for {guide.char_class}, you are {char_data['char_class']}"

    comp = GuideComparison(
        current_stage=current_stage.value,
        player_level=level,
        gear_matches=[asdict(gm) for gm in gear_matches],
        skill_matches=[asdict(sm) for sm in skill_matches],
        passive_matches=passive_matches,
        overall_score=round(overall_score, 1),
        next_upgrades=[asdict(u) for u in next_upgrades],
        stage_notes=stage_data.notes,
        building_right=building_right,
    )
    return asdict(comp)


def get_stage_prices(guide: ParsedGuide, stage_value: str, price_cache=None) -> List[dict]:
    """Get price estimates for all gear items in a guide stage."""
    stage_data = None
    for s in guide.stages:
        if s.stage == stage_value:
            stage_data = s
            break

    if not stage_data:
        return []

    results = []
    for item in stage_data.gear:
        entry = {
            "slot": item.slot,
            "name": item.name,
            "is_unique": item.is_unique,
            "price_display": "",
            "divine_value": None,
        }
        if price_cache and item.is_unique:
            price = price_cache.lookup(item.name)
            if price:
                entry["price_display"] = price.get("display", "")
                entry["divine_value"] = price.get("divine_value")
        results.append(entry)
    return results


def _fuzzy_name_match(a: str, b: str) -> bool:
    """Loose name matching — case-insensitive substring."""
    if not a or not b:
        return False
    a_low = a.strip().lower()
    b_low = b.strip().lower()
    return a_low == b_low or a_low in b_low or b_low in a_low
