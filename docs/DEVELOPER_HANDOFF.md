# LAMA — Developer Reference

> **Last updated:** 2026-03-29
> **Status:** Build analysis tool — character lookup, build classification, build-aware item scoring, dashboard with Pulse/Gear/Market/Guide tabs. Pricing system removed.

## What This Is

A build analysis companion for Path of Exile 2. Look up any character via poe.ninja, get plain-language explanations of what's working, what needs upgrades, how their build compares to meta, and why specific gear/keystones matter.

**Not a pricing tool anymore.** The pricing/harvester/calibration system was killed in March 2026 due to unsolvable data quality problems. All trade API code, POESESSID usage, and calibration pipelines have been removed.

## Architecture

```
LAMA.bat / LAMA-debug.bat
  ↓
app.py::main()
  ├─ FastAPI server on 127.0.0.1:8450 (daemon thread)
  │   └─ server.py — REST API, WebSocket, dashboard serving
  ├─ System tray icon via pystray (daemon thread)
  └─ pywebview frameless window → /dashboard
      └─ dashboard.html — React18 + Tailwind (in-browser Babel)
          └─ WebSocket to /ws for real-time log streaming
```

**Overlay subprocess** (optional): `main.py` runs as a separate process when the user clicks Start in the dashboard. It monitors the game window, reads item clipboard data, and shows build-aware scores in an overlay. Launched via `subprocess.Popen` from `server.py`'s `OverlayManager`.

### Data Flow: Character Analysis

```
User enters query (name, URL, account/character)
  → POST /api/character/smart-lookup
    → _parse_ninja_url() or fuzzy-match saved chars
    → builds_client.lookup_character()
      → poe.ninja builds/ladder API (ladder-ranked chars)
      → poe.ninja profile API fallback (any public char)
    → CharacterData returned
  → Dashboard receives character data
  → Parallel analysis requests:
    → POST /api/character/build-insights → classify_build(), popular keystones, per-slot scoring, gap analysis
    → POST /api/character/improvement-package → upgrade priorities
    → POST /api/character/build-compare → meta comparison
    → POST /api/character/build-efficiency → cost/reward analysis
```

### Data Sources

| Source | Usage | Status |
|--------|-------|--------|
| poe.ninja builds API | Character lookup (ladder) | Active |
| poe.ninja profile API | Character lookup (any public char) | Active — requires league in URL path |
| poe.ninja popular skills/keystones | Meta comparison | Active |
| poe2scout | League list | Active |
| RePoE (repoe-fork.github.io/poe2/) | Mod tier data, base items | Active — cached 7 days |

**Forbidden:** POESESSID, trade2 API, any reverse-engineered GGG endpoints. OAuth app was rejected by GGG.

## File Inventory

### Source (`src/`)

| File | Purpose |
|------|---------|
| `app.py` | Desktop shell — frameless pywebview window, WindowApi (min/max/close/quit), system tray, debug mode |
| `server.py` | FastAPI backend — character lookup (smart + direct), build insights, settings, overlay process mgmt, WebSocket |
| `main.py` | Overlay entry point — item detection, clipboard parsing, build-aware scoring, overlay display |
| `builds_client.py` | poe.ninja API client — character lookup, build classification (BuildArchetype), anti-synergy detection, popular items/skills/keystones |
| `mod_database.py` | Item scoring engine — RePoE tier data, weight table, build-aware multipliers, progression multipliers, dual grading |
| `mod_parser.py` | Mod text → stat_id matching via regex (trade API stat definitions) |
| `item_parser.py` | Clipboard text → ParsedItem (name, rarity, mods, DPS, defense) |
| `item_lookup.py` | Dashboard item lookup facade — paste-and-score |
| `item_detection.py` | Cursor tracking, Ctrl+C sending, POE2 window detection |
| `overlay.py` | Transparent tkinter overlay + ConsoleOverlay fallback |
| `config.py` | Constants: API URLs, display settings, rate limits |
| `filter_updater.py` | Loot filter economy re-tiering |
| `price_cache.py` | Currency exchange rates from poe2scout (divine:chaos, divine:exalted) |
| `clipboard_reader.py` | Windows clipboard reading via ctypes |
| `screen_capture.py` | POE2 window detection via Win32 API |
| `tray.py` | System tray icon integration |
| `bundle_paths.py` | IS_FROZEN/APP_DIR/get_resource() for PyInstaller builds |
| `oauth.py` | OAuth2 flow (stash tab access — functional but GGG rejected the app) |

### Resources

| File | Purpose |
|------|---------|
| `resources/dashboard.html` | Single-file React UI — Pulse/Gear/Market/Guide tabs, smart character search, build analysis display |
| `resources/VERSION` | App version string |
| `resources/img/` | Icons and images |

### Key Endpoints (server.py)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/dashboard` | Serves dashboard HTML |
| POST | `/api/character/smart-lookup` | Flexible lookup: URLs, names, partial matches |
| POST | `/api/character/lookup` | Direct lookup by account + character |
| GET | `/api/character/saved` | Saved characters list |
| POST | `/api/character/build-insights` | Build classification, scoring, gap analysis |
| POST | `/api/character/improvement-package` | Upgrade priorities |
| POST | `/api/character/build-compare` | Meta comparison |
| POST | `/api/character/build-efficiency` | Cost/reward analysis |
| GET | `/api/status` | Overlay process status |
| POST | `/api/start` | Launch overlay subprocess |
| POST | `/api/stop` | Stop overlay |
| GET | `/api/settings` | Read settings |
| POST | `/api/settings` | Write settings |
| GET | `/api/leagues` | Available leagues from poe2scout |
| WS | `/ws` | Real-time log + status streaming |

### Build Classification (builds_client.py)

`classify_build(CharacterData) → BuildArchetype` analyzes:
- **Damage type**: spell vs attack (from skill gems + keystones)
- **Defense type**: life, ES, MoM, hybrid
- **Crit**: is_crit, is_coc (Cast on Crit)
- **Elements**: fire, cold, lightning (from skill tags + gear)
- **Dead mods**: mods on gear that don't benefit the build
- **Anti-synergy rules**: detects conflicts (e.g. crit mods + Elemental Overload)

### Build-Aware Scoring (mod_database.py)

When a `BuildArchetype` is provided, scoring applies multipliers:
- **Damage type**: spell builds penalize attack mods and vice versa (0.05x)
- **Crit**: non-crit builds penalize crit mods (0.1x)
- **Defense type**: ES builds boost ES mods (2.0x), penalize life (0.3x)
- **Progression**: leveling chars boost resists (2.5x), penalize crit multi (0.3x)

Dual scoring: every item gets both a universal grade (build-agnostic) and a build-aware grade when archetype is available.

## Settings

Stored in `~/.poe2-price-overlay/dashboard_settings.json`. Key fields:
- `league` — current league name
- `saved_characters` — array of `{accountName, characters: [{name, class, level, lastLookup}]}`
- `build_archetype` — persisted BuildArchetype for overlay scoring
- `overlay_mode`, `scan_fps`, `detection_cooldown` — overlay behavior
- `window_width`, `window_height`, `window_maximized` — window geometry

## Testing

```bash
python -m pytest tests/ -v
```

Test fixtures in `tests/fixtures/` — real clipboard captures from the game.

## Build & Distribution

```bash
pyinstaller scripts/build.spec --noconfirm --clean
```

Output: `dist/LAMA/LAMA.exe` — bundles Python runtime, all source, resources, and data files.

## Debug Mode

`LAMA-debug.bat` or `python src/app.py --debug`:
- Console window stays open with full log output
- WebView2 DevTools enabled (right-click → Inspect)
- Overlay subprocess: `python src/main.py --debug --league "..."` for file-level DEBUG logging
