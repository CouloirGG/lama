# Game-Agnostic Architecture — Developer Guide

> **Last updated:** 2026-02-19
> **Audience:** Co-developers adding new game support or maintaining the config layer

## A. Overview

LAMA's pricing pipeline was originally hardcoded for Path of Exile 2. Phases 1-2 extracted every game-specific constant into a configuration layer so the core pipeline can support **any game title** without modifying existing modules.

Three files make this work:

| File | Role |
|------|------|
| `src/core/game_config.py` | `GameConfig` dataclass — single source of truth for all game-specific values (44 fields) |
| `src/core/pricing_engine.py` | `PricingEngine` facade — takes a `GameConfig`, injects values into existing modules, exposes public API |
| `src/games/poe2.py` | `create_poe2_config()` factory — builds a `GameConfig` populated with all POE2 constants |

**Usage is three lines:**

```python
from core import PricingEngine
from games.poe2 import create_poe2_config

engine = PricingEngine(create_poe2_config())
engine.initialize()
result = engine.lookup(clipboard_text)
```

To add a new game, you create a new factory (e.g. `games/last_epoch.py`) that returns a `GameConfig` with that game's constants. No core files change.

---

## B. Architecture

### Data Flow

```
┌─────────────────────┐
│  games/poe2.py      │   Game factory — knows all POE2 constants
│  create_poe2_config()│
└──────────┬──────────┘
           │ returns
           ▼
┌─────────────────────┐
│  core/game_config.py│   GameConfig dataclass — game-agnostic container
│  GameConfig(...)    │   ~35 typed fields with defaults
└──────────┬──────────┘
           │ passed to
           ▼
┌─────────────────────┐
│  core/pricing_engine│   PricingEngine — facade & injector
│  .initialize()      │──► Overrides module-level constants
│  .lookup()          │──► Wraps all pipeline modules
│  .parse_item()      │
│  .score_item()      │
│  .deep_price()      │
└──────────┬──────────┘
           │ injects into
           ▼
┌──────────────────────────────────────────────────┐
│  Existing modules (unchanged)                     │
│  item_parser ─► mod_parser ─► mod_database       │
│  calibration    price_cache    trade_client       │
└──────────────────────────────────────────────────┘
```

### The Module Override Pattern

Each existing module (e.g. `mod_parser.py`) defines module-level constants like `TRADE_STATS_URL`. The original code imported these from `config.py`. PricingEngine replaces that pattern by:

1. **Importing the module as an alias** (not any class from it)
2. **Setting attributes** on the module object to override constants
3. **Then importing the class** and instantiating it

```python
def _init_mod_parser(self):
    import mod_parser as mp_module            # Step 1: import module

    mp_module.TRADE_STATS_URL = self.config.trade_stats_url   # Step 2: override
    mp_module.TRADE_STATS_CACHE_FILE = self.config.trade_stats_cache_file

    from mod_parser import ModParser          # Step 3: import class
    self._mod_parser = ModParser()            # Step 4: instantiate
    self._mod_parser.load_stats()
```

This is the **lowest-friction approach** — existing module files are never modified. The module-level constants they read at class/method time already have the right values.

---

## C. GameConfig Field Reference

All fields from `src/core/game_config.py`, organized by section.

### Identity

| Field | Type | Default | Configured Module |
|-------|------|---------|-------------------|
| `game_id` | `str` | *(required)* | PricingEngine logging |
| `default_league` | `str` | *(required)* | price_cache, trade_client |
| `cache_dir` | `Path` | *(required)* | price_cache |

### Trade API

| Field | Type | Default | Configured Module |
|-------|------|---------|-------------------|
| `trade_api_base` | `str` | *(required)* | trade_client |
| `trade_stats_url` | `str` | *(required)* | mod_parser |
| `trade_items_url` | `str` | *(required)* | mod_parser |
| `trade_stats_cache_file` | `Path` | *(required)* | mod_parser |
| `trade_items_cache_file` | `Path` | *(required)* | mod_parser |
| `trade_max_requests_per_second` | `int` | `1` | trade_client |
| `trade_result_count` | `int` | `8` | trade_client |
| `trade_cache_ttl` | `int` | `300` (5 min) | trade_client |
| `trade_mod_min_multiplier` | `float` | `0.8` | trade_client |
| `trade_dps_filter_mult` | `float` | `0.75` | trade_client |
| `trade_defense_filter_mult` | `float` | `0.70` | trade_client |

### Mod Database (RePoE)

| Field | Type | Default | Configured Module |
|-------|------|---------|-------------------|
| `repoe_base_url` | `str` | `""` | mod_database |
| `repoe_cache_dir` | `Optional[Path]` | `None` | mod_database |
| `repoe_cache_ttl` | `int` | `604800` (7 days) | mod_database |
| `dps_item_classes` | `FrozenSet[str]` | `frozenset()` | mod_database, trade_client |
| `two_hand_classes` | `FrozenSet[str]` | `frozenset()` | mod_database |
| `defense_item_classes` | `FrozenSet[str]` | `frozenset()` | mod_database, trade_client |
| `dps_brackets_2h` | `Dict[int, Tuple[int, ...]]` | `{}` | mod_database |
| `dps_brackets_1h` | `Dict[int, Tuple[int, ...]]` | `{}` | mod_database |
| `defense_thresholds` | `Dict[str, Tuple[int, ...]]` | `{}` | mod_database |

### Price Sources

| Field | Type | Default | Configured Module |
|-------|------|---------|-------------------|
| `price_source_url` | `str` | `""` | price_cache |
| `price_refresh_interval` | `int` | `900` (15 min) | price_cache |
| `rate_history_file` | `Optional[Path]` | `None` | price_cache |
| `rate_history_backup` | `Optional[Path]` | `None` | price_cache |

### Calibration

| Field | Type | Default | Configured Module |
|-------|------|---------|-------------------|
| `calibration_log_file` | `Optional[Path]` | `None` | calibration |
| `calibration_max_price_divine` | `float` | `300.0` | calibration |
| `calibration_min_results` | `int` | `3` | calibration |
| `shard_dir` | `Optional[Path]` | `None` | calibration |
| `shard_refresh_interval` | `int` | `86400` (24h) | calibration |
| `shard_github_repo` | `str` | `""` | calibration |

### Mod Database Scoring

| Field | Type | Default | Configured Module |
|-------|------|---------|-------------------|
| `weight_table` | `List[Tuple[float, List[str]]]` | `[]` | mod_database (`_WEIGHT_TABLE`) |
| `defence_group_markers` | `Tuple[str, ...]` | `()` | mod_database (`_DEFENCE_GROUP_MARKERS`) |
| `display_names` | `List[Tuple[str, str]]` | `[]` | mod_database (`_DISPLAY_NAMES`) |

### Item Parser Classification

| Field | Type | Default | Configured Module |
|-------|------|---------|-------------------|
| `currency_keywords` | `FrozenSet[str]` | `frozenset()` | item_parser (`CURRENCY_KEYWORDS`) |
| `valuable_bases` | `FrozenSet[str]` | `frozenset()` | item_parser (`VALUABLE_BASES`) |

### Price Cache Endpoints & Categories

| Field | Type | Default | Configured Module |
|-------|------|---------|-------------------|
| `poe_ninja_exchange_url` | `str` | `""` | price_cache (`POE2_EXCHANGE_URL`) |
| `exchange_categories` | `List[str]` | `[]` | price_cache (`EXCHANGE_CATEGORIES`) |
| `poe2scout_unique_categories` | `List[str]` | `[]` | price_cache (`POE2SCOUT_UNIQUE_CATEGORIES`) |
| `poe2scout_currency_categories` | `List[str]` | `[]` | price_cache (`POE2SCOUT_CURRENCY_CATEGORIES`) |
| `price_request_delay` | `float` | `0.0` | price_cache (`REQUEST_DELAY`) |

### Grade Display Mapping

| Field | Type | Default | Configured Module |
|-------|------|---------|-------------------|
| `grade_tier_map` | `Dict[str, str]` | `{}` | overlay (grade → color tier) |

---

## D. How PricingEngine Injects Config

`PricingEngine.initialize()` calls six private methods in a fixed order. Each method imports a module, overrides its constants from `GameConfig`, then instantiates the class.

### Initialization Order & Dependencies

| # | Method | Module | Constants Overridden | Depends On |
|---|--------|--------|---------------------|------------|
| 1 | `_init_item_parser()` | `item_parser` | `CURRENCY_KEYWORDS`, `VALUABLE_BASES` | — |
| 2 | `_init_mod_parser()` | `mod_parser` | `TRADE_STATS_URL`, `TRADE_STATS_CACHE_FILE`, `TRADE_ITEMS_URL`, `TRADE_ITEMS_CACHE_FILE` | — |
| 3 | `_init_mod_database()` | `mod_database` | `REPOE_BASE_URL`, `REPOE_CACHE_DIR`, `REPOE_CACHE_TTL`, `DPS_ITEM_CLASSES`, `TWO_HAND_CLASSES`, `DEFENSE_ITEM_CLASSES`, `DPS_BRACKETS_2H`, `DPS_BRACKETS_1H`, `DEFENSE_THRESHOLDS`, `_WEIGHT_TABLE`, `_DEFENCE_GROUP_MARKERS`, `_DISPLAY_NAMES` | ModParser must be loaded first |
| 4 | `_init_calibration()` | `calibration` | *(passes config values directly, no module overrides)* | — |
| 5 | `_init_price_cache()` | `price_cache` | `PRICE_REFRESH_INTERVAL`, `CACHE_DIR`, `DEFAULT_LEAGUE`, `POE2SCOUT_BASE_URL`, `RATE_HISTORY_FILE`, `RATE_HISTORY_BACKUP`, `POE2_EXCHANGE_URL`, `EXCHANGE_CATEGORIES`, `POE2SCOUT_UNIQUE_CATEGORIES`, `POE2SCOUT_CURRENCY_CATEGORIES`, `REQUEST_DELAY` | — |
| 6 | `_init_trade_client()` | `trade_client` | `TRADE_API_BASE`, `TRADE_MAX_REQUESTS_PER_SECOND`, `TRADE_RESULT_COUNT`, `TRADE_CACHE_TTL`, `TRADE_MOD_MIN_MULTIPLIER`, `DPS_ITEM_CLASSES`, `DEFENSE_ITEM_CLASSES`, `TRADE_DPS_FILTER_MULT`, `TRADE_DEFENSE_FILTER_MULT`, `DEFAULT_LEAGUE` | PriceCache (for exchange rates) |

### Critical Dependency

`_init_mod_database()` **skips entirely** if `_mod_parser` is `None` or `_mod_parser.loaded` is `False`. This is because `ModDatabase.load()` requires a loaded `ModParser` to build its stat-ID-to-RePoE bridge. Order matters.

### Readiness Check

After all six methods run, `PricingEngine.ready` is `True` only if both `_mod_parser.loaded` and `_mod_database.loaded` are `True`. These are the two critical modules for the scoring pipeline. `_price_cache` and `_trade_client` are optional — they may fail silently if the network is unavailable.

---

## E. The POE2 Implementation

`src/games/poe2.py` serves as the **reference implementation** for any new game factory.

### Structure

```python
def create_poe2_config(
    league: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> GameConfig:
```

The factory:
1. Imports constants from `src/config.py` (the legacy central config)
2. Allows `league` and `cache_dir` overrides via parameters
3. Returns a fully populated `GameConfig`

### The Transition Bridge Pattern

During the Phase 1→2 transition, POE2 values come from **two sources**:

| Source | Fields | Reason |
|--------|--------|--------|
| `config.py` imports | Trade API URLs, cache paths, RePoE URLs, item classes, DPS/defense brackets, calibration paths, grade map | These were already defined in `config.py` and used by the live app. Importing them keeps the factory in sync with the running system. |
| Hardcoded in factory | `weight_table`, `defence_group_markers`, `display_names`, `currency_keywords`, `valuable_bases`, `exchange_categories`, `poe2scout_*_categories`, `poe_ninja_exchange_url`, `price_request_delay` | These were previously defined inside individual modules (e.g. `_WEIGHT_TABLE` inside `mod_database.py`). Phase 2 extracted them into the factory. |

The bridge exists so that existing `config.py` remains the single source of truth for values shared by both the legacy pipeline (`main.py → ItemLookup`) and the new engine pipeline. Once all consumers migrate to `PricingEngine`, the imports from `config.py` can be replaced with inline values.

### What `game_id="poe2"` Means

The `game_id` is used only for logging and cache directory naming. It doesn't trigger any conditional logic in the core modules.

---

## F. Adding a New Game: Step-by-Step

### 1. Create the Factory

Create `src/games/new_game.py`:

```python
"""NewGame configuration factory."""

from pathlib import Path
from typing import Optional

from core.game_config import GameConfig


def create_newgame_config(
    league: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> GameConfig:
    _cache = cache_dir or Path.home() / ".newgame-overlay" / "cache"

    return GameConfig(
        # ── Identity ──
        game_id="newgame",
        default_league=league or "Standard",
        cache_dir=_cache,

        # ── Trade API ──
        trade_api_base="https://newgame.com/api/trade",
        trade_stats_url="https://newgame.com/api/trade/stats",
        trade_items_url="https://newgame.com/api/trade/items",
        trade_stats_cache_file=_cache / "trade_stats.json",
        trade_items_cache_file=_cache / "trade_items.json",

        # ── Mod Database ──
        repoe_base_url="https://newgame-data.example.com",
        repoe_cache_dir=_cache / "moddata",

        # ... fill remaining fields ...
    )
```

### 2. Register the Factory

Add the import to `src/games/__init__.py`:

```python
"""Game-specific configurations for LAMA pricing engine."""
from games.newgame import create_newgame_config
```

### 3. Required vs Optional Fields

**Required (no defaults — must be provided):**

| Field | Why |
|-------|-----|
| `game_id` | Identifies the game in logs and cache paths |
| `default_league` | Every trade API is scoped to a league/season/realm |
| `cache_dir` | Base directory for all disk caches |
| `trade_api_base` | Root URL for the game's trade API |
| `trade_stats_url` | Endpoint for stat/mod definitions |
| `trade_items_url` | Endpoint for item base type definitions |
| `trade_stats_cache_file` | Disk cache path for stat definitions |
| `trade_items_cache_file` | Disk cache path for item definitions |

**Optional (have defaults, but you should set them):**

Everything else. Empty defaults (`""`, `frozenset()`, `{}`, `[]`) mean the feature is disabled or uses fallback behavior. For a fully functional pipeline, you should populate at least:

- `repoe_base_url` — without this, ModDatabase can't load tier data → no scoring
- `dps_item_classes`, `defense_item_classes`, `dps_brackets_*`, `defense_thresholds` — without these, DPS/defense factors are disabled
- `weight_table` — without this, all mods score equally
- `currency_keywords` — without this, currency items aren't classified correctly
- `price_source_url` — without this, PriceCache has no data source

### 4. Where to Find Equivalent Data for a New Game

| POE2 Concept | What to look for | POE2 Source |
|--------------|-----------------|-------------|
| Trade API | Public trade site with search/stats endpoints | pathofexile.com/api/trade2 |
| Stat definitions | List of all mod/stat IDs with text templates | Trade API `/data/stats` |
| Mod tier data | Per-mod min/max values by item class | RePoE (community data extraction) |
| Price aggregation | Pre-computed price database for uniques/currency | poe2scout.com, poe.ninja |
| DPS/defense brackets | Community consensus on "good" DPS for each weapon type | Manual research / game wikis |
| Weight table | Which mod keywords are premium/key/filler | Game knowledge + community tier lists |
| Currency keywords | All currency item names (lowercase) | Game wiki / item database |

### 5. When the New Game Has Different Concepts

The current architecture assumes games have:
- **Items** with a name, base type, rarity, item level
- **Mods** (affixes) with stat IDs and numeric values
- **A trade API** that searches by stat filters
- **DPS** (for weapons) and **defense** (for armor) as summary stats

If the new game doesn't have mods (e.g. it uses a skill tree instead), you can still use the framework:
- Leave `weight_table`, `display_names`, `defence_group_markers` empty
- `ModParser` and `ModDatabase` will initialize with no data — `score_item()` returns `None`
- `PriceCache` and `TradeClient` still work independently for price lookups
- You may need to extend `ItemParser` with a game-specific subclass

If the new game has a fundamentally different item format (not clipboard text), you'll need to:
1. Write a custom parser (can reuse or extend `ItemParser`)
2. Use `PricingEngine.parse_item()` as the entry point — it delegates to whatever `ItemParser` is initialized

### 6. Testing the New Config

```python
# tests/test_newgame_config.py
from games.newgame import create_newgame_config

def test_create_default():
    cfg = create_newgame_config()
    assert cfg.game_id == "newgame"
    assert cfg.default_league
    assert cfg.trade_api_base

def test_all_required_fields():
    cfg = create_newgame_config()
    assert cfg.game_id
    assert cfg.default_league
    assert cfg.trade_api_base
    assert cfg.trade_stats_url
    assert cfg.trade_items_url

def test_engine_constructs():
    from core import PricingEngine
    cfg = create_newgame_config()
    engine = PricingEngine(cfg)
    assert not engine.ready  # not initialized yet
    assert engine.config.game_id == "newgame"
```

See `tests/test_game_config.py` (23 tests) and `tests/test_pricing_engine.py` (24 tests) for the full POE2 test patterns.

---

## G. What's NOT Configurable Yet (Phase 3+)

The current system extracts **module-level constants** but does not yet handle:

### Class-Level Attributes

Some constants live inside class bodies rather than at module level. These can't be overridden by the current `import module; module.CONST = value` pattern without modifying the class or using constructor injection.

### Inline Logic

Game-specific logic that's embedded in methods rather than driven by data:

- **`item_parser.py`** — Clipboard format parsing assumes POE2's `--------` separator, `Item Class:` header format, mod annotation patterns like `(implicit)`, `(rune)`, etc.
- **`mod_parser.py`** — Regex compilation from `#+` placeholders, stat text template matching
- **`trade_client.py`** — Common mod classification heuristics, equipment filter field names (`dps`, `ar`, `ev`, `es`)
- **`overlay.py`** — Display formatting, color schemes

### Roadmap for Full Parity

1. **Phase 3:** Extract remaining class-level constants into `GameConfig` fields
2. **Phase 4:** Make `ItemParser` pluggable (register parser per game for different clipboard/item formats)
3. **Phase 5:** Make `TradeClient` query builder pluggable (different trade APIs have different filter schemas)

---

## H. Test Coverage

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Config + engine tests only
python -m pytest tests/test_game_config.py tests/test_pricing_engine.py -v

# Single test class
python -m pytest tests/test_game_config.py::TestPoe2Config -v
```

### What the Tests Verify

**`tests/test_game_config.py`** — 23 tests across 2 classes:

| Class | Tests | Verifies |
|-------|-------|----------|
| `TestGameConfig` | 2 | Dataclass creation with minimal required fields; all defaults have expected values |
| `TestPoe2Config` | 21 | Factory produces correct `game_id`; league/cache overrides work; all trade API, RePoE, item class, DPS bracket, defense threshold, price source, calibration, grade map, weight table, display names, currency keywords, valuable bases, exchange categories, poe.ninja URL, request delay, and all-required-fields-populated check |

**`tests/test_pricing_engine.py`** — 24 tests across 7 classes:

| Class | Tests | Verifies |
|-------|-------|----------|
| `TestPricingEngineInit` | 5 | Construction with POE2 config and minimal config; methods return `None` before `initialize()` |
| `TestParseItem` | 4 | Parses rare, currency, unique fixtures; rejects invalid text |
| `TestScoreItem` | 3 | Scores rare items with grades and normalized scores; currency returns None |
| `TestFullPipeline` | 5 | End-to-end `lookup()` with dict structure; smoke test across all fixtures |
| `TestEstimatePrice` | 2 | Calibration returns float or None; None score → None estimate |
| `TestParseMods` | 2 | Returns ParsedMod list with expected attributes; currency has no mods |
| `TestModuleOverrides` | 3 | Verifies `_init_*()` methods actually set module-level constants (`CURRENCY_KEYWORDS`, `_WEIGHT_TABLE`, `POE2_EXCHANGE_URL`, etc.) |

---

## Quick Reference

```
src/
├── core/
│   ├── __init__.py          # Exports PricingEngine, GameConfig
│   ├── game_config.py       # GameConfig dataclass (44 fields)
│   └── pricing_engine.py    # PricingEngine facade (6 init methods, 10 public methods)
├── games/
│   ├── __init__.py          # Game registry
│   └── poe2.py              # create_poe2_config() — reference implementation
├── config.py                # Legacy constants (POE2-specific, bridged by poe2.py)
├── item_parser.py           # Clipboard → ParsedItem
├── mod_parser.py            # Mods → stat IDs (trade API matching)
├── mod_database.py          # Local scoring engine (RePoE tiers, DPS/defense)
├── calibration.py           # Score → price estimation (k-NN)
├── price_cache.py           # Static prices (uniques, currency)
└── trade_client.py          # Live trade API queries
```
