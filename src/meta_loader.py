"""
meta_loader.py — Runtime loader for meta shard data.

Loads the auto-generated meta_shard.json.gz at startup, providing
an API for the why-engine to access season-current meta data.
Falls back to game_knowledge.py defaults when no shard exists.

Usage:
    from meta_loader import MetaData
    meta = MetaData.load()
    scaling = meta.class_scaling("Witch")
    ceiling = meta.dps_ceiling("Blood Mage")
"""

import gzip
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger("meta_loader")

# Default shard location
SETTINGS_DIR = Path(os.path.expanduser("~")) / ".poe2-price-overlay"
SHARD_PATH = SETTINGS_DIR / "meta_shard.json.gz"
SHARD_MAX_AGE = 24 * 60 * 60  # 24 hours

# GitHub release URL for latest shard
SHARD_RELEASE_URL = "https://github.com/CouloirGG/lama/releases/download/meta-latest/meta_shard.json.gz"


@dataclass
class ClassScalingData:
    """Per-class DPS scaling data from meta shard."""
    sample_size: int = 0
    primary_factor: str = ""
    weights: Dict[str, float] = field(default_factory=dict)
    defense_meta: str = "life"
    defense_distribution: Dict[str, int] = field(default_factory=dict)
    top_skills: List[Dict] = field(default_factory=list)
    dps_range: Dict[str, float] = field(default_factory=dict)


class MetaData:
    """Runtime meta data provider. Loads from shard, falls back to defaults."""

    _instance: Optional["MetaData"] = None

    def __init__(self, data: dict):
        self._data = data
        self._league = data.get("league", "")
        self._generated = data.get("generated_utc", "")
        self._version = data.get("version", 0)
        self._source = data.get("_source", "unknown")

    @classmethod
    def load(cls, force_refresh: bool = False) -> "MetaData":
        """Load meta data. Returns cached instance unless force_refresh."""
        if cls._instance and not force_refresh:
            return cls._instance

        # Try loading from shard file
        shard_data = cls._load_shard()

        if shard_data:
            logger.info(
                f"Meta shard loaded: league={shard_data.get('league')}, "
                f"generated={shard_data.get('generated_utc')}, "
                f"version={shard_data.get('version')}"
            )
            shard_data["_source"] = "shard"
            cls._instance = cls(shard_data)
            return cls._instance

        # Try downloading latest shard
        if not shard_data:
            downloaded = cls._download_shard()
            if downloaded:
                shard_data = cls._load_shard()
                if shard_data:
                    shard_data["_source"] = "downloaded"
                    cls._instance = cls(shard_data)
                    return cls._instance

        # Fall back to game_knowledge.py defaults
        logger.info("No meta shard available — using game_knowledge.py defaults")
        defaults = cls._defaults_from_game_knowledge()
        defaults["_source"] = "defaults"
        cls._instance = cls(defaults)
        return cls._instance

    @classmethod
    def _load_shard(cls) -> Optional[dict]:
        """Load shard from disk if it exists and isn't too old."""
        if not SHARD_PATH.exists():
            return None

        # Check age
        age = time.time() - SHARD_PATH.stat().st_mtime
        if age > SHARD_MAX_AGE * 7:  # Allow up to 7 days before forcing refresh
            logger.info(f"Meta shard is {age/3600:.0f}h old (>7 days) — will try to refresh")
            # Still load it as fallback, but flag as stale
            pass

        try:
            with gzip.open(SHARD_PATH, "rt", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version", 0) < 2:
                logger.warning(f"Meta shard version {data.get('version')} is outdated")
                return None
            return data
        except Exception as e:
            logger.warning(f"Failed to load meta shard: {e}")
            return None

    @classmethod
    def _download_shard(cls) -> bool:
        """Try to download the latest shard from GitHub releases."""
        try:
            import requests
            logger.info(f"Downloading latest meta shard from GitHub...")
            resp = requests.get(SHARD_RELEASE_URL, timeout=10,
                                headers={"User-Agent": "LAMA/1.0"})
            if resp.status_code != 200:
                logger.warning(f"Shard download failed: HTTP {resp.status_code}")
                return False

            SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
            with open(SHARD_PATH, "wb") as f:
                f.write(resp.content)
            logger.info(f"Meta shard downloaded ({len(resp.content)} bytes)")
            return True
        except Exception as e:
            logger.warning(f"Shard download failed: {e}")
            return False

    @classmethod
    def _defaults_from_game_knowledge(cls) -> dict:
        """Build a shard-compatible dict from game_knowledge.py defaults."""
        try:
            from game_knowledge import (
                CLASS_SCALING as GK_CLASS_SCALING,
                TOP_SUPPORT_GEMS as GK_SUPPORT_GEMS,
                POPULAR_UNIQUES as GK_UNIQUES,
                KEYSTONE_COMBOS as GK_COMBOS,
            )
        except ImportError:
            return {"version": 2, "league": "unknown"}

        data = {
            "version": 2,
            "league": "defaults",
            "generated_utc": "hardcoded",
            "class_scaling": {},
            "top_support_gems": {},
            "popular_uniques": {},
            "keystone_combos": [],
            "mod_weights": {"global": {}, "per_class": {}},
            "dps_ceilings": {},
        }

        # Convert CLASS_SCALING
        for cls_name, cs in GK_CLASS_SCALING.items():
            data["class_scaling"][cls_name] = {
                "sample_size": 0,
                "primary_factor": cs.primary_factor,
                "weights": cs.weights,
                "defense_meta": cs.defense_meta,
                "top_skills": [],
                "dps_range": {},
            }

        # Convert TOP_SUPPORT_GEMS
        for gem_name, gi in GK_SUPPORT_GEMS.items():
            data["top_support_gems"][gem_name] = {
                "top_count": 0,
                "bottom_count": 0,
                "ratio": 99,
                "best_classes": gi.best_for,
            }

        # Convert POPULAR_UNIQUES
        for uname, ui in GK_UNIQUES.items():
            data["popular_uniques"][uname] = {
                "slot": ui.slot,
                "global_adoption_pct": ui.global_adoption,
                "class_adoption": ui.class_adoption,
            }

        # Convert KEYSTONE_COMBOS
        for combo in GK_COMBOS:
            data["keystone_combos"].append({
                "keystones": combo.keystones,
                "adoption_pct": combo.adoption_pct,
                "best_classes": combo.best_classes,
            })

        return data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def league(self) -> str:
        return self._league

    @property
    def source(self) -> str:
        return self._source

    @property
    def generated(self) -> str:
        return self._generated

    def class_scaling(self, base_class: str) -> Optional[ClassScalingData]:
        """Get scaling data for a base class."""
        cs = self._data.get("class_scaling", {}).get(base_class)
        if not cs:
            return None
        return ClassScalingData(
            sample_size=cs.get("sample_size", 0),
            primary_factor=cs.get("primary_factor", ""),
            weights=cs.get("weights", {}),
            defense_meta=cs.get("defense_meta", "life"),
            defense_distribution=cs.get("defense_distribution", {}),
            top_skills=cs.get("top_skills", []),
            dps_range=cs.get("dps_range", {}),
        )

    def top_support_gems(self) -> Dict[str, dict]:
        """Get top support gems that separate top from bottom builds."""
        return self._data.get("top_support_gems", {})

    def popular_uniques(self, base_class: str = "") -> Dict[str, dict]:
        """Get popular unique items, optionally filtered by class."""
        uniques = self._data.get("popular_uniques", {})
        if not base_class:
            return uniques
        # Filter to items with significant adoption for this class
        return {
            name: info for name, info in uniques.items()
            if info.get("class_adoption", {}).get(base_class, 0) >= 15
        }

    def keystone_combos(self, base_class: str = "") -> List[dict]:
        """Get keystone combos, optionally filtered by class."""
        combos = self._data.get("keystone_combos", [])
        if not base_class:
            return combos
        return [c for c in combos
                if not c.get("best_classes") or base_class in c["best_classes"]]

    def mod_weight(self, mod_type: str, base_class: str = "") -> float:
        """Get the scaling weight for a mod type, with class-specific override."""
        weights = self._data.get("mod_weights", {})
        # Try class-specific first
        if base_class:
            per_class = weights.get("per_class", {}).get(base_class, {})
            if mod_type in per_class:
                return per_class[mod_type]
        # Fall back to global
        return weights.get("global", {}).get(mod_type, 1.0)

    def dps_ceiling(self, ascendancy_or_class: str) -> dict:
        """Get DPS ceiling data for an ascendancy or class."""
        ceilings = self._data.get("dps_ceilings", {})
        return ceilings.get(ascendancy_or_class, {})

    def all_data(self) -> dict:
        """Return raw shard data for inspection."""
        return self._data
