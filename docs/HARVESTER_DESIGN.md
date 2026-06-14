# LAMA Meta Harvester System — Design Document

> **Last updated:** 2026-03-31
> **Status:** Design phase — replaces original #6/#7/#8 issues

## Core Principle

POE2 is a live game. Every league season changes the meta — skills get buffed/nerfed, new uniques drop, ascendancy balance shifts, keystone combos rise and fall. **Nothing we learn about the meta should be hardcoded.** Everything must be data-driven and auto-refreshed.

What's currently hardcoded that must become data-driven:
- `CLASS_SCALING` — per-class DPS factor weights
- `TOP_SUPPORT_GEMS` — support gems that separate top from bottom
- `POPULAR_UNIQUES` — unique item adoption rates per class
- `KEYSTONE_COMBOS` — keystone synergy patterns
- Mod contribution weights in `_analyze_mod_contribution()`
- DPS ceilings and population stat ranges

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Meta Harvester                      │
│  (runs daily via GitHub Action or local cron)        │
│                                                      │
│  1. Discover current league + snapshot version        │
│  2. For each ascendancy (21+):                       │
│     a. Fetch search protobuf (stat ranges, featured) │
│     b. Sample 5-10 top chars (profile API)           │
│     c. Extract: gear mods, jewels, keystones,        │
│        support gems, DPS, defenses                   │
│  3. Run scaling analysis (top vs bottom per class)   │
│  4. Generate meta shard (compressed JSON)            │
│  5. Upload as GitHub release asset or commit         │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                   Meta Shard                         │
│  (~/.poe2-price-overlay/meta_shard.json.gz)         │
│                                                      │
│  {                                                   │
│    "version": 2,                                     │
│    "league": "Fate of the Vaal",                     │
│    "generated": "2026-03-31T06:00:00Z",             │
│    "class_scaling": { ... },                         │
│    "top_support_gems": { ... },                      │
│    "popular_uniques": { ... },                       │
│    "keystone_combos": [ ... ],                       │
│    "mod_weights": { ... },                           │
│    "dps_ceilings": { ... },                          │
│    "stat_ranges": { ... },                           │
│  }                                                   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                 LAMA App (Runtime)                    │
│                                                      │
│  On startup:                                         │
│    1. Check meta shard age                           │
│    2. If stale (>24h), download latest from GitHub   │
│    3. Load shard → overrides game_knowledge defaults │
│    4. Why-engine uses shard data for all analysis    │
│                                                      │
│  game_knowledge.py becomes DEFAULTS only:            │
│    - Used when no shard exists (first run)           │
│    - Overridden by shard data when available         │
│    - Still useful as documentation of known mechs    │
└─────────────────────────────────────────────────────┘
```

## Meta Shard Schema (v2)

```json
{
  "version": 2,
  "league": "Fate of the Vaal",
  "league_url": "vaal",
  "generated_utc": "2026-03-31T06:00:00Z",
  "snapshot_version": "2227-20260331-12992",
  "total_builds_sampled": 410,
  "total_characters_on_ladder": 124284,

  "class_scaling": {
    "Warrior": {
      "sample_size": 59,
      "primary_factor": "extra_as",
      "weights": {
        "extra_as": 3.3,
        "atk_speed": 3.2,
        "crit_chance": 2.6,
        "crit_multi": 2.3,
        "cast_speed": 4.1,
        "gem_levels": 1.0,
        "jewel_count": 1.3
      },
      "defense_meta": "life",
      "defense_distribution": {"life": 95, "es": 3, "mom": 2},
      "top_skills": [
        {"name": "Furious Slam", "count": 6, "avg_dps": 813423},
        {"name": "Hammer of the Gods", "count": 3, "avg_dps": 389238}
      ],
      "dps_range": {"min": 1412, "max": 813423, "spread": 576}
    }
  },

  "top_support_gems": {
    "Cast on Critical": {
      "top_count": 57,
      "bottom_count": 13,
      "ratio": 4.4,
      "best_classes": ["Witch", "Sorceress", "Mercenary", "Huntress"]
    },
    "Rakiata's Flow": {
      "top_count": 54,
      "bottom_count": 0,
      "ratio": "inf",
      "best_classes": ["all"]
    }
  },

  "popular_uniques": {
    "Headhunter": {
      "slot": "Belt",
      "global_adoption_pct": 67.0,
      "class_adoption": {
        "Ranger": 92, "Huntress": 76, "Monk": 69,
        "Druid": 67, "Mercenary": 58, "Warrior": 57, "Witch": 40
      }
    }
  },

  "keystone_combos": [
    {
      "keystones": ["Eldritch Battery", "Mind Over Matter"],
      "adoption_pct": 25.0,
      "best_classes": ["Witch", "Mercenary", "Huntress", "Druid"],
      "top_pct": 32,
      "bottom_pct": 9
    }
  ],

  "mod_weights": {
    "global": {
      "crit_multi": 4.9,
      "crit_chance": 3.2,
      "extra_as": 2.2,
      "cast_speed": 2.1,
      "spell_dmg": 1.6,
      "gem_levels": 1.1
    },
    "per_class": {
      "Witch": {"crit_multi": 11.7, "crit_chance": 3.5},
      "Druid": {"crit_multi": 10.7, "cast_speed": 2.8}
    }
  },

  "dps_ceilings": {
    "Blood Mage": {"max": 300542969, "p90": 1500000, "median": 280000},
    "Pathfinder": {"max": 27828388, "p90": 6000000, "median": 400000}
  }
}
```

## Harvester Components

### 1. `meta_harvester.py` — Data Collection CLI

```
python scripts/meta_harvester.py [--league "Fate of the Vaal"] [--output meta_shard.json.gz]
```

**What it does:**
1. Fetches current league and snapshot from poe.ninja index-state
2. For each of the 21 ascendancies:
   - Fetches search protobuf (stat ranges, featured characters, dimensions)
   - Samples 5-10 top characters via profile API (full gear, jewels, mods)
   - Extracts all mod totals, keystone lists, support gems, unique items
3. Runs the scaling analysis (same logic as `scripts/class_deep_dive.py`)
4. Generates the meta shard
5. Compresses and saves

**Rate limiting:** 1 req/sec to poe.ninja. ~400 total requests for full harvest. Takes ~10 minutes.

**Data flow:**
```
poe.ninja index-state → snapshot version
    ↓
For each ascendancy:
    search endpoint → stat ranges, featured chars, dimensions
    profile API × 5-10 → full character profiles
        ↓
    Extract per-character:
        - All gear mods (normalized)
        - Jewel names + mods
        - Keystones
        - Support gem links
        - DPS per skill
        - Defensive stats (life, ES, EHP, resists)
    ↓
Aggregate:
    - Top 1/3 vs bottom 1/3 correlation analysis
    - Per-class scaling weights
    - Support gem frequency analysis
    - Unique item adoption rates
    - Keystone combo detection
    - DPS ceilings and distributions
    ↓
Output: meta_shard.json.gz
```

### 2. `meta_loader.py` — Shard Loading at Runtime

**On app startup:**
1. Check for local shard at `~/.poe2-price-overlay/meta_shard.json.gz`
2. If missing or stale (>24h), try to download latest from GitHub releases
3. Decompress and validate schema
4. Override `game_knowledge.py` defaults with shard data
5. Log what changed since last shard

**API for why-engine:**
```python
from meta_loader import MetaData

meta = MetaData.load()  # loads shard or falls back to defaults
meta.class_scaling("Witch")  # returns ClassScaling for Witch
meta.top_support_gems()  # returns current top support gems
meta.popular_uniques("Ranger")  # returns popular uniques for Ranger
meta.mod_weight("crit_multi", "Witch")  # returns 11.7
meta.dps_ceiling("Blood Mage")  # returns {"max": 300M, "p90": 1.5M}
```

### 3. GitHub Action — `meta-harvest.yml`

```yaml
name: Meta Harvest
on:
  schedule:
    - cron: '0 6 * * *'  # Daily at 06:00 UTC
  workflow_dispatch: {}  # Manual trigger

jobs:
  harvest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.13' }
      - run: pip install requests
      - run: python scripts/meta_harvester.py --output meta_shard.json.gz
      - name: Upload shard
        env: { GH_TOKEN: ${{ github.token }} }
        run: |
          gh release delete meta-latest --yes 2>/dev/null || true
          git push origin :refs/tags/meta-latest 2>/dev/null || true
          gh release create meta-latest meta_shard.json.gz \
            --title "Meta Shard ($(date +%Y-%m-%d))" \
            --notes "Auto-generated meta analysis shard" \
            --prerelease
```

### 4. App Changes — `game_knowledge.py` as Defaults

Current `game_knowledge.py` becomes the **fallback defaults** — used when no shard exists. At runtime, the meta shard overrides everything:

```python
# game_knowledge.py stays as-is for documentation + first-run defaults

# meta_loader.py provides the runtime API:
class MetaData:
    _instance = None

    @classmethod
    def load(cls):
        if cls._instance:
            return cls._instance
        # Try shard first
        shard = cls._load_shard()
        if shard:
            cls._instance = cls(shard)
        else:
            # Fall back to game_knowledge.py defaults
            cls._instance = cls(cls._defaults_from_game_knowledge())
        return cls._instance
```

### 5. Why-Engine Integration

The why-engine currently imports directly from `game_knowledge`:
```python
from game_knowledge import CLASS_SCALING, TOP_SUPPORT_GEMS, POPULAR_UNIQUES
```

This changes to:
```python
from meta_loader import MetaData
meta = MetaData.load()
# Use meta.class_scaling("Witch") instead of CLASS_SCALING["Witch"]
```

This means:
- First run (no shard): uses hardcoded defaults from game_knowledge.py
- After first harvest: uses live data from the shard
- Every 24 hours: shard refreshes with latest meta
- New season: harvester runs on new league, discovers new meta automatically

## Season Transition Handling

When a new season starts:
1. poe.ninja's index-state returns a new league name/URL
2. The harvester detects this automatically (league name changed)
3. Old shard is invalidated (league mismatch)
4. First harvest on new league: small sample (few characters on ladder)
5. As the league matures: sample grows, weights stabilize
6. game_knowledge.py defaults serve as bootstrap until enough data exists

**What changes between seasons:**
- CLASS_SCALING weights (new skills, balance changes)
- TOP_SUPPORT_GEMS (new gems, nerfs)
- POPULAR_UNIQUES (new uniques, meta shifts)
- KEYSTONE_COMBOS (balance changes, new keystones)
- DPS ceilings (power creep or nerfs)
- Defense meta (new mechanics, balance)

**What stays stable:**
- KEYSTONES descriptions (game mechanics don't change often)
- DEFENSE_MECHANICS explanations
- STAT_THRESHOLDS (general ballpark stays similar)
- The harvester logic itself (data collection method is stable)

## Implementation Priority

1. **`meta_harvester.py`** — the collection script (most of the logic exists in `scripts/class_deep_dive.py`, needs to output shard format)
2. **`meta_loader.py`** — shard loading + runtime API
3. **Why-engine migration** — switch from hardcoded imports to meta_loader
4. **GitHub Action** — automated daily runs
5. **App startup integration** — check shard age, download if stale

## Migration Path

Phase 1: Build harvester + shard format (no app changes yet)
Phase 2: Build meta_loader with fallback to game_knowledge defaults
Phase 3: Update why-engine to use meta_loader
Phase 4: Deploy GitHub Action for daily harvests
Phase 5: Add shard download to app startup

The app works at every phase — game_knowledge.py serves as the safety net until the full pipeline is running.
