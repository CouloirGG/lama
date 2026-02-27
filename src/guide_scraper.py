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
# Maxroll Planner API
# ---------------------------------------------------------------------------
MAXROLL_PLANNER_API = "https://planners.maxroll.gg/profiles/poe2"

# Maxroll internal class codes → display names
_MAXROLL_CLASS_MAP = {
    "IntFour": ("Witch", "Infernalist"),
    "IntThree": ("Witch", "Blood Mage"),
    "StrFour": ("Warrior", "Warbringer"),
    "StrThree": ("Warrior", "Titan"),
    "DexFour": ("Ranger", "Pathfinder"),
    "DexThree": ("Ranger", "Deadeye"),
    "StrDexFour": ("Mercenary", "Gemling Legionnaire"),
    "StrDexThree": ("Mercenary", "Witchhunter"),
    "DexIntFour": ("Monk", "Invoker"),
    "DexIntThree": ("Monk", "Acolyte of Chayula"),
    "StrIntFour": ("Sorceress", "Stormweaver"),
    "StrIntThree": ("Sorceress", "Chronomancer"),
}

# Maxroll equipment slot keys → normalized slot names
_MAXROLL_SLOT_MAP = {
    "Helm": "helmet",
    "BodyArmour": "body",
    "Gloves": "gloves",
    "Boots": "boots",
    "Belt": "belt",
    "Amulet": "amulet",
    "Ring": "ring",
    "Ring2": "ring2",
    "Weapon": "weapon",
    "Weapon2": "weapon2",
    "Offhand": "shield",
    "Offhand2": "shield2",
    "Flask1": "flask1",
    "Flask2": "flask2",
    "Charm1": "charm1",
    "Charm2": "charm2",
    "Charm3": "charm3",
    "Jewel1": "jewel1",
    "Jewel2": "jewel2",
}

# Gearing variant names → GuideStage mapping
_VARIANT_STAGE_MAP = {
    "campaign": GuideStage.ACT_1,
    "leveling": GuideStage.ACT_1,
    "early": GuideStage.INTERLUDE,
    "early maps": GuideStage.INTERLUDE,
    "mid": GuideStage.ENDGAME,
    "late": GuideStage.ENDGAME,
    "min-max": GuideStage.ENDGAME,
    "endgame": GuideStage.ENDGAME,
}

# Skill step names → GuideStage mapping
_STEP_STAGE_KEYWORDS = {
    GuideStage.ACT_1: ["act 1", "step 1", "step 2", "step 3"],
    GuideStage.ACT_2: ["act 2"],
    GuideStage.ACT_3: ["act 3"],
    GuideStage.ACT_4: ["act 4"],
    GuideStage.INTERLUDE: ["interlude", "early"],
    GuideStage.ENDGAME: ["mid", "late", "min-max", "endgame"],
}


def _gem_id_to_name(gem_id: str) -> str:
    """Convert Metadata gem IDs to display names.

    'Metadata/Items/Gems/SkillGemVolcano' → 'Volcano'
    'Metadata/Items/Gems/SupportGemIgnition' → 'Ignition'
    """
    name = gem_id.rsplit("/", 1)[-1]
    name = re.sub(r"^SkillGem", "", name)
    name = re.sub(r"^SupportGem", "", name)
    # Insert spaces before caps: "FireInfusion" → "Fire Infusion"
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    return name


def _base_to_display(base_path: str) -> str:
    """Convert Metadata base paths to readable names.

    'Metadata/Items/Armours/Helmets/FourHelmetInt8Endgame' → 'Int Helmet (Endgame)'
    """
    name = base_path.rsplit("/", 1)[-1]
    # Clean up the Four prefix
    name = re.sub(r"^Four", "", name)
    # Insert spaces before caps
    name = re.sub(r"(?<=[a-z])(?=[A-Z0-9])", " ", name)
    return name


def _mod_key_to_display(mod_key: str) -> str:
    """Convert internal mod stat_ids to readable labels.

    'minion_skill_gem_level_+' → '+Minion Skill Gem Level'
    'base_maximum_life' → 'Maximum Life'
    """
    s = mod_key.replace("_", " ").strip()
    s = re.sub(r"\s*\+\s*$", "", s)  # Remove trailing +
    s = re.sub(r"\s*%\s*$", "", s)   # Remove trailing %
    s = re.sub(r"^base ", "", s)
    s = re.sub(r"^local ", "", s)
    # Capitalize words
    return " ".join(w.capitalize() for w in s.split())


def _classify_step_stage(step_name: str) -> GuideStage:
    """Map a Maxroll skill step name to a GuideStage."""
    low = step_name.lower()
    for stage, keywords in _STEP_STAGE_KEYWORDS.items():
        for kw in keywords:
            if kw in low:
                return stage
    return GuideStage.ENDGAME


def _classify_variant_stage(variant_name: str) -> GuideStage:
    """Map a Maxroll gearing variant name to a GuideStage."""
    low = variant_name.strip().lower()
    return _VARIANT_STAGE_MAP.get(low, GuideStage.ENDGAME)


# ---------------------------------------------------------------------------
# Maxroll Parser (uses Planner API for structured data + HTML for text)
# ---------------------------------------------------------------------------
class MaxrollParser:
    """Parse Maxroll.gg POE2 build guides via their Planner API."""

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

        # Title from HTML
        title_el = soup.find("h1")
        guide.title = _clean_text(title_el) if title_el else "Maxroll Guide"

        # Class/Ascendancy from HTML text
        page_text = soup.get_text(" ", strip=True)
        guide.char_class, guide.ascendancy = _detect_class_ascendancy(page_text)
        guide.main_skill = cls._detect_main_skill(guide.title, page_text)

        # Extract planner profile ID from HTML embeds
        profile_id = cls._extract_profile_id(soup)
        if profile_id:
            try:
                planner_data = cls._fetch_planner(profile_id)
                if planner_data:
                    cls._parse_planner_data(guide, planner_data)
            except Exception as e:
                logger.warning("Maxroll planner API failed for %s: %s", profile_id, e)

        # Extract guide notes from HTML tab content
        cls._extract_html_notes(soup, guide)

        return guide

    @classmethod
    def _detect_main_skill(cls, title: str, page_text: str) -> str:
        title_clean = re.sub(r"\b(build|guide|poe2?|path of exile 2?)\b", "", title, flags=re.I).strip()
        for name in POE2_CLASSES + POE2_ASCENDANCIES:
            title_clean = re.sub(re.escape(name), "", title_clean, flags=re.I)
        title_clean = re.sub(r"\s+", " ", title_clean).strip(" -\u2013\u2014|")
        return title_clean if title_clean else ""

    @classmethod
    def _extract_profile_id(cls, soup: BeautifulSoup) -> Optional[str]:
        """Find the poe2-embed planner profile ID in the page."""
        for el in soup.find_all(class_="poe2-embed"):
            pid = el.get("data-poe2-profile")
            if pid:
                return pid
        return None

    @classmethod
    def _fetch_planner(cls, profile_id: str) -> Optional[dict]:
        """Fetch structured planner data from Maxroll API."""
        resp = _SESSION.get(f"{MAXROLL_PLANNER_API}/{profile_id}", timeout=15)
        if resp.status_code != 200:
            return None
        raw = resp.json()
        data = raw.get("data", {})
        if isinstance(data, str):
            data = json.loads(data)
        return data

    @classmethod
    def _parse_planner_data(cls, guide: ParsedGuide, data: dict):
        """Parse the Maxroll planner JSON into guide stages."""
        planner = data.get("planner", {})
        items_db = data.get("items", {})

        # Class/Ascendancy from planner (more reliable than HTML)
        class_code = planner.get("class", "")
        if class_code in _MAXROLL_CLASS_MAP:
            guide.char_class, guide.ascendancy = _MAXROLL_CLASS_MAP[class_code]

        # Build stages from gearing variants (most reliable stage source)
        equipment = planner.get("equipment", {})
        variants = equipment.get("variants", [])

        # Also get skills per step
        skills_data = planner.get("skills", {})
        skill_steps = skills_data.get("steps", [])

        # Group skill steps by stage
        skills_by_stage: Dict[str, List[GuideSkillSetup]] = {}
        for step in skill_steps:
            stage = _classify_step_stage(step.get("name", ""))
            stage_key = stage.value
            if stage_key not in skills_by_stage:
                skills_by_stage[stage_key] = []
            for sg in step.get("skills", []):
                gems = sg.get("gems", [])
                if not gems:
                    continue
                main_gem = _gem_id_to_name(gems[0].get("id", ""))
                supports = [_gem_id_to_name(g.get("id", "")) for g in gems[1:]]
                # Avoid duplicates within same stage
                existing_names = {s.name for s in skills_by_stage[stage_key]}
                if main_gem and main_gem not in existing_names:
                    skills_by_stage[stage_key].append(GuideSkillSetup(
                        name=main_gem,
                        supports=supports,
                        is_main=not skills_by_stage[stage_key],  # First is main
                    ))

        # Build stages from gearing variants
        stages_dict: Dict[str, GuideStageData] = {}
        for variant in variants:
            variant_name = variant.get("name", "")
            stage = _classify_variant_stage(variant_name)
            stage_key = stage.value

            # Parse gear items for this variant
            gear_items = []
            for slot_key, item_id in variant.get("items", {}).items():
                slot = _MAXROLL_SLOT_MAP.get(slot_key, slot_key.lower())
                item = items_db.get(str(item_id), {})
                if not item:
                    continue

                rarity = item.get("rarity", "rare")
                is_unique = rarity == "unique"
                base = _base_to_display(item.get("base", ""))
                name = item.get("name", "") if item.get("name", "") and item["name"] != variant_name else ""

                # Extract key mods
                stats = item.get("stats", {})
                explicit = stats.get("explicit", {})
                key_mods = [_mod_key_to_display(k) for k in list(explicit.keys())[:4]]

                display_name = name if name else (f"Rare {base}" if not is_unique else base)

                gear_items.append(GuideGearItem(
                    slot=slot,
                    name=display_name,
                    is_unique=is_unique,
                    key_mods=key_mods,
                ))

            # Get level range for stage
            for s, lo, hi in STAGE_RANGES:
                if s == stage:
                    level_range = (lo, hi)
                    break
            else:
                level_range = (1, 100)

            # Merge with existing stage or create new
            if stage_key in stages_dict:
                # Merge — later variants override gear
                stages_dict[stage_key].gear = gear_items
                stages_dict[stage_key].notes += f"\n{variant_name} gearing variant."
            else:
                stages_dict[stage_key] = GuideStageData(
                    stage=stage_key,
                    level_range=level_range,
                    gear=gear_items,
                    skills=skills_by_stage.get(stage_key, []),
                    notes=f"{variant_name} gearing variant.",
                )

        # Add skills-only stages that don't have gearing variants
        for stage_key, skills_list in skills_by_stage.items():
            if stage_key not in stages_dict and skills_list:
                for s, lo, hi in STAGE_RANGES:
                    if s.value == stage_key:
                        level_range = (lo, hi)
                        break
                else:
                    level_range = (1, 100)
                stages_dict[stage_key] = GuideStageData(
                    stage=stage_key,
                    level_range=level_range,
                    skills=skills_list,
                )

        # Sort stages by level range
        guide.stages = sorted(stages_dict.values(), key=lambda s: s.level_range[0])

    @classmethod
    def _extract_html_notes(cls, soup: BeautifulSoup, guide: ParsedGuide):
        """Extract guide prose/notes from HTML tab content to enrich stages."""
        try:
            tab_containers = soup.find_all(class_=re.compile(r'_tabsV2'))
            if not tab_containers:
                return
            # First tab container is usually Skills description
            first_tabs = tab_containers[0]
            container = first_tabs.find(class_=re.compile(r'_container'))
            if not container:
                return
            tab_divs = container.find_all(class_=re.compile(r'^_tab_'), recursive=False)
            # Get the visible (first) tab's text as general notes
            for td in tab_divs:
                text = td.get_text(" ", strip=True)[:500]
                if text and guide.stages:
                    # Add to the first stage's notes
                    guide.stages[0].notes = text
                    break
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Generic HTML Parser (fallback for Mobalytics and other sites)
# ---------------------------------------------------------------------------
class GenericHTMLParser:
    """Best-effort HTML parser for build guides without a structured API."""

    @classmethod
    def parse(cls, html: str, url: str, source: str = "unknown") -> ParsedGuide:
        soup = BeautifulSoup(html, "html.parser")
        guide = ParsedGuide(
            id=str(uuid.uuid4()),
            url=url,
            source=source,
            imported_at=datetime.now(timezone.utc).isoformat(),
            raw_html_hash=_hash_html(html),
        )

        title_el = soup.find("h1") or soup.find("title")
        guide.title = _clean_text(title_el) if title_el else "Build Guide"

        page_text = soup.get_text(" ", strip=True)
        guide.char_class, guide.ascendancy = _detect_class_ascendancy(page_text)
        guide.main_skill = MaxrollParser._detect_main_skill(guide.title, page_text)

        # Try to extract gear from tables or lists
        body = soup.find("body") or soup
        gear = cls._extract_gear_from_html(body)
        skills = cls._extract_skills_from_html(body)

        if gear or skills:
            guide.stages = [GuideStageData(
                stage=GuideStage.ENDGAME.value,
                level_range=(1, 100),
                gear=gear,
                skills=skills,
                notes="Parsed from page content (single-stage extraction).",
            )]

        return guide

    @classmethod
    def _extract_gear_from_html(cls, root: Tag) -> List[GuideGearItem]:
        gear = []
        seen = set()
        for table in root.find_all("table"):
            for row in table.find_all("tr"):
                cells = row.find_all(["td", "th"])
                if len(cells) >= 2:
                    slot = _normalize_slot(_clean_text(cells[0]))
                    name = _clean_text(cells[1])
                    if slot and name and slot not in seen:
                        mods = [_clean_text(cells[2])] if len(cells) > 2 else []
                        gear.append(GuideGearItem(slot=slot, name=name, key_mods=mods))
                        seen.add(slot)
        if not gear:
            for li in root.find_all("li"):
                text = _clean_text(li)
                m = re.match(r"([\w\s-]+?):\s*(.+)", text)
                if m:
                    slot = _normalize_slot(m.group(1))
                    if slot in ("weapon", "helmet", "body", "gloves", "boots", "belt",
                                "ring", "ring2", "amulet", "shield") and slot not in seen:
                        gear.append(GuideGearItem(slot=slot, name=m.group(2).strip()))
                        seen.add(slot)
        return gear

    @classmethod
    def _extract_skills_from_html(cls, root: Tag) -> List[GuideSkillSetup]:
        skills = []
        seen = set()
        for header in root.find_all(["h2", "h3", "h4", "strong"]):
            if any(kw in _clean_text(header).lower() for kw in ("skill", "gem", "setup")):
                next_list = header.find_next(["ul", "ol"])
                if next_list:
                    items = [_clean_text(li) for li in next_list.find_all("li") if _clean_text(li)]
                    if items and items[0] not in seen:
                        skills.append(GuideSkillSetup(
                            name=items[0], supports=items[1:], is_main=not skills,
                        ))
                        seen.add(items[0])
        return skills


# ---------------------------------------------------------------------------
# Import entry point
# ---------------------------------------------------------------------------
def import_guide(url: str) -> ParsedGuide:
    """Fetch a guide URL, parse it, save to disk, and return the result."""
    source = detect_source(url)

    if source == "maxroll":
        html = fetch_html(url)
        guide = MaxrollParser.parse(html, url)
    else:
        html = fetch_html(url)
        guide = GenericHTMLParser.parse(html, url, source=source)

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
