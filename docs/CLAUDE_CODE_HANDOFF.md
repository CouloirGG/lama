# POE2 Price Overlay — Claude Code Handoff

> **Last updated:** 2026-02-17
> **Status:** Working prototype — local mod scoring + DPS/defense integration + trade API pricing + 106-test regression suite

## What This Is

A real-time price overlay for Path of Exile 2. Hover over any item in-game, it sends Ctrl+C to copy item data, parses the clipboard text, looks up the price, and shows an overlay near your cursor.

**Three pricing pipelines:**
1. **Static cache** (uniques, currency, gems) — prices from poe2scout.com, refreshed every 15 min
2. **Local mod scoring** (rare/magic items) — instant S/A/B/C/JUNK grades using RePoE tier data + DPS/defense factors, zero API calls
3. **Trade API** (deep query via Ctrl+Shift+C) — queries pathofexile.com/trade with the item's actual mods + DPS/defense filters

## Current Architecture

**Two interfaces:**
1. **Desktop Dashboard (primary)** — `python app.py` or `POE2 Dashboard.bat`
   - PyWebView native window → FastAPI server (`server.py`) → React UI (`dashboard.html`)
   - 3 tabs: Overlay (controls + live log), Loot Filter (strictness/styles), Watchlist (trade queries)
   - Manages overlay subprocess (main.py) with start/stop/restart
   - Real-time log streaming via WebSocket
   - Settings persistence to `~/.poe2-price-overlay/dashboard_settings.json`
   - Bug report submission (Discord webhook)
   - Loot filter update trigger
2. **CLI/Overlay mode** — `python main.py` or `START.bat`
   - Direct overlay without dashboard (legacy mode)

**Pricing pipeline (runs inside main.py):**
```
Cursor stops over POE2 window (8 fps polling)
  → Sends Ctrl+C via keybd_event (Windows API)
  → Reads clipboard text (POE2 item format)
  → ItemParser.parse_clipboard() → ParsedItem
  → If rare/magic with mods:
      → ModParser.parse_mods() → List[ParsedMod]
      → ModDatabase.score_item() → ItemScore with grade (S/A/B/C/JUNK)
        → DPS/defense factors applied (attack weapons penalized for low DPS,
          armor for low defense; staves/wands excluded from DPS scoring)
      → Shows grade instantly, Ctrl+Shift+C triggers trade API deep query
      → TradeClient.price_rare_item() → RarePriceResult (includes DPS/defense filters)
  → If unique/currency/gem:
      → PriceCache.lookup() → dict with display/tier
  → Overlay shows color-coded price near cursor
```

## File Inventory (root level, NOT in src/)

| File | Purpose |
|------|---------|
| `main.py` | Entry point & orchestrator. Pipeline, threading, startup |
| `config.py` | All constants: API URLs, rate limits, display settings |
| `item_detection.py` | Cursor tracking, Ctrl+C sending, POE2 window detection |
| `item_parser.py` | Clipboard text → ParsedItem (name, base_type, rarity, mods, DPS, defense) |
| `mod_parser.py` | Matches mod text to trade API stat IDs via regex |
| `mod_database.py` | Local mod scoring engine — RePoE tier data, DPS/defense factors, S/A/B/C/JUNK grades |
| `calibration.py` | Score-to-price calibration engine (learns from deep query results) |
| `trade_client.py` | Queries POE2 trade API for rare item pricing (includes DPS/defense equipment_filters) |
| `filter_updater.py` | Loot filter economy re-tiering (NeverSink .filter + poe.ninja prices) |
| `price_cache.py` | Fetches/caches prices from poe2scout.com (uniques, currency, gems) |
| `overlay.py` | Transparent tkinter overlay + ConsoleOverlay fallback |
| `clipboard_reader.py` | Windows clipboard reading via ctypes |
| `screen_capture.py` | POE2 window detection via Win32 API |
| `app.py` | **Desktop dashboard shell** — pywebview native window hosting FastAPI server |
| `server.py` | **FastAPI backend** — overlay process mgmt, WS log streaming, settings, watchlist, bug reports, filter updates, league API |
| `dashboard.html` | **Single-file React UI** — POE2 dark theme, 3 tabs (Overlay, Loot Filter, Watchlist), KPI cards, real-time log console |
| `watchlist.py` | Trade API polling worker for the Watchlist tab |
| `launcher.py` | Legacy CLI launcher (bat files call this for non-GUI mode) |
| `START.bat` | Main launcher (CLI mode) |
| `POE2 Dashboard.bat` | **Launches the GUI dashboard** (`python app.py`) |
| `SYNC.bat` | **Multi-machine sync** — pulls latest from GitHub, installs deps, verifies setup |
| `DEBUG.bat` | Launches with `--debug` flag for verbose logging |
| `REPORT_BUG.bat` | Zips logs to Desktop, opens GitHub issue page |
| `LICENSE` | GPLv3 |
| `.gitignore` | Standard Python gitignore |

## Key Technical Details

### Clipboard Item Format (POE2)
```
Item Class: Bows
Rarity: Rare
Armageddon Thirst
Recurve Bow
--------
Physical Damage: 120-250
Elemental Damage: 30-60
Attacks per Second: 1.50
Quality: +20% (augmented)
--------
Requires: Level 78, 100 Dex
--------
Item Level: 82
--------
170% increased Physical Damage
Adds 30 to 50 Physical Damage
35% increased Critical Hit Chance
24% increased Attack Speed
```

```
Item Class: Body Armours
Rarity: Rare
Fortress Plate
Full Plate
--------
Armour: 850
Energy Shield: 120
Quality: +20% (augmented)
--------
Item Level: 85
--------
+100 to maximum Life
+200 to Armour
+28% to Fire Resistance
```

Sections separated by `--------`. Mod annotations in parentheses: `(implicit)`, `(rune)`, `(enchant)`, `(fractured)`, `(desecrated)`, `(mutated)`, `(crafted)`, `(augmented)`.

**Weapon stats** (Physical Damage, Elemental Damage, Attacks per Second) and **defense stats** (Armour, Evasion Rating, Energy Shield) appear in the property section. The game shows final computed values including quality and mod bonuses. `item_parser.py` extracts these to compute total DPS and total defense.

### Mod Parser (`mod_parser.py`)
- Fetches ~6114 stat definitions from `GET /api/trade2/data/stats`
- Caches to disk for 24h at `~/.poe2-price-overlay/cache/trade_stats.json`
- Compiles each stat text template (`"+# to maximum Life"`) into a regex
- ~3259 stats have working regexes (others have no `#` placeholder)
- `_match_mod()` preserves the original clipboard mod_type (rune stays rune, fractured stays fractured)

### Mod Database (`mod_database.py`)
- **Local scoring engine** — evaluates rare/magic items instantly using static RePoE data (zero API calls)
- Data: `mods.min.json`, `mods_by_base.min.json`, `base_items.min.json` from repoe-fork.github.io/poe2/
- Cached in `~/.poe2-price-overlay/cache/repoe/`, 7-day TTL, stale-cache fallback
- **Stat ID bridge:** maps 942 trade API stat_ids to RePoE mod groups via normalized text matching
- **Tier ladders:** 2023 ladders keyed by (mod_group, item_class) — identifies T1/T2/T3 etc.
- **Scoring:** per-mod percentile × weight, grades S/A/B/C/JUNK based on normalized score + key mod count
- **Weight table:** Premium(3.0), Key(2.0), Standard(1.0), Filler(0.3), Near-zero(0.1)
- **DPS factor** (`_dps_factor()`): multiplicative penalty for low-DPS attack weapons
  - Staves/wands excluded (caster weapons where DPS is irrelevant)
  - DPS_ITEM_CLASSES: bows, crossbows, swords, axes, maces, daggers, claws, flails, spears
  - Curve: below terrible → 0.15, terrible→low → 0.5, low→decent → 0.85, decent→good → 1.0, above good → 1.15 (cap)
  - ilvl-aware brackets: endgame (ilvl 68+) vs leveling, 1H vs 2H
- **Defense factor** (`_defense_factor()`): softer penalty for low-defense armor pieces
  - Per-slot thresholds: body armours, shields, helmets, gloves, boots, bucklers, foci
  - Curve: below terrible → 0.6, up to 1.05 (narrower range than DPS)
- **Overlay display:** shows "280dps" or "850def" when factor penalizes; star count when neutral
- Legacy test harness: `python mod_database.py` runs inline test cases (superseded by pytest suite)

### Trade Client (`trade_client.py`)
- **Progressive search strategy:**
  1. Try "and" with ALL mods (exact match)
  2. If 0 results, classify mods as "key" vs "common" (filler) using domain knowledge patterns
  3. Try "and" with only key mods (drops +mana, +regen, +resistance, etc.)
  4. Progressively remove key mods until results found
  5. Last resort: "count" based matching

- **Common mod patterns** (deprioritized): maximum mana/life/ES, regeneration, resistances, attributes, item rarity

- **Value-dependent minimums:**
  - Values 1-10: 95% multiplier (tier-based mods like +7 skills, each point matters hugely)
  - Values 11-50: 90% multiplier
  - Values 51+: 80% multiplier (default)

- **Outlier trimming:** Fetches 8 results, skips cheapest 25% before computing price range

- **Priceable mod types:** explicit, implicit, fractured, desecrated (NOT rune, enchant, crafted)

- **Rate limiting:** 2 req/sec with thread-safe lock, retries on HTTP 429

- **DPS/defense filters:** For attack weapons, includes `dps` min (75% of item's total DPS) in `equipment_filters`. For armor pieces, includes `ar`/`ev`/`es` mins (70% of item's values). Ensures trade comparables have similar combat stats.

- **Caching:** In-memory, keyed by base_type + sorted stat_ids + rounded values + DPS + defense, 5min TTL

### Price Cache (`price_cache.py`)
- Sources: poe2scout.com API (uniques, currency, gems, fragments, etc.)
- Uniques stored by name only (NOT by base_type) to avoid collisions with rare items
- `divine_to_chaos` and `divine_to_exalted` rates exposed for trade_client normalization
- 15-minute refresh interval

### Item Detection (`item_detection.py`)
- Polls cursor position at 8 fps
- Triggers Ctrl+C when cursor is still for 3 frames within 20px radius
- Content-based dedup: same clipboard text within 5s = skip
- `SetConsoleCtrlHandler(None, True)` prevents the programmatic Ctrl+C from killing Python

### Overlay (`overlay.py`)
- tkinter-based, transparent, click-through, always-on-top
- Thread-safe updates via `root.after()`
- Color-coded by tier: high (orange), good (gold), decent (teal), low (grey)
- 4-second display duration

## Test Suite

106 pytest tests across 4 modules in `tests/`:

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_item_parser.py` | 21 | Clipboard parsing, combat stats, implicit separation, sockets, quality, currency/gem/unique detection |
| `test_mod_parser.py` | 15 | `_template_to_regex`, opposite word matching, real fixture mod parsing, `resolve_base_type`, value extraction |
| `test_mod_database.py` | 55 | 40 migrated from `__main__` (S/A/B/C/JUNK grading) + 15 new (SOMV, `_assign_grade` boundaries, DPS/defense factors) |
| `test_trade_client.py` | 15 | Common mod patterns, stat filter building, value-dependent minimums, fractured mods, price tiers, filter classification |

**Fixtures:** `tests/conftest.py` provides session-scoped `mod_parser` and `mod_database` (load once per run), plus `stat_ids` resolver, `make_item`/`make_mod` helpers, and `load_fixture` for reading clipboard captures from `tests/fixtures/`.

**Running:**
- `RUN_TESTS.bat` — spawns a PowerShell window per module (visual monitoring)
- `python -m pytest tests/ -v` — all tests in one terminal
- `python run_tests.py --module mod_database` — single module in PowerShell window
- `python -m pytest tests/test_mod_database.py -v` — single module inline

## User's Setup
- **OS:** Windows 11 Home (10.0.26200)
- **Monitor:** 3440x1440 ultrawide
- **POE2:** Windowed Fullscreen
- **Python:** 3.11
- **Project path:** `C:\Users\Stu\GitHub\POE2_OCR`
- **League:** Fate of the Vaal
- **Debug files:** `~/.poe2-price-overlay/debug/clipboard_*.txt`
- **Log file:** `~/.poe2-price-overlay/overlay.log`

## Git Status
- **Remote:** `https://github.com/CarbonSMASH/POE2_OCR.git` (private repo)
- **Branch:** `main`
- All changes committed and pushed
- GitHub CLI (`gh`) installed and authenticated as CarbonSMASH
- Issue template at `.github/ISSUE_TEMPLATE/bug_report.md`
- Bug reporting script: `REPORT_BUG.bat` (zips logs, opens GitHub issue page)

## Known Working
- Unique items price correctly from poe2scout cache
- Currency/gems/fragments price correctly
- Rare/magic items scored locally via ModDatabase (instant S/A/B/C/JUNK grades)
- DPS factor correctly penalizes low-DPS attack weapons; excludes caster weapons (staves/wands)
- Defense factor correctly penalizes low-defense armor; per-slot thresholds
- Deep query (Ctrl+Shift+C) prices via trade API with DPS/defense equipment_filters
- Loot filter auto-updates every 24h based on live economy data
- Overlay shows grade instantly, "Checking..." on deep query, then updates with price
- Content-based dedup prevents re-triggering on dead stash space
- Calibration engine learns score→price mapping from deep query results

## Known Issues / Technical Debt
1. ~~**Rate limiting under burst**~~ — Fixed: adaptive rate limiting parses API response headers, proactive backoff at 50% usage, API call budget per item
2. **"Grants Skill:" lines** are skipped via `_SKIP_LINE_RE` but unusual skill formats might slip through
3. ~~**Magic items without base_type**~~ — Fixed: `ModParser.resolve_base_type()` fetches all base types from the trade API items endpoint
4. **Common mod classification** is heuristic-based — may occasionally misclassify an unusual valuable mod as "common"
5. ~~**README.md is outdated**~~ — Fixed: rewritten for clipboard-based architecture

## Bugs Fixed This Session (2026-02-15)
1. Mod annotations (`(implicit)`, `(rune)`, etc.) not stripped before regex matching
2. Trade query too restrictive (online-only, too many mods)
3. Pipeline order: rare items hitting static cache before trade API
4. `keybd_event` Ctrl+C killing Python process (fixed with SetConsoleCtrlHandler)
5. Unique base_type collisions (Dream Fragments matching any Sapphire Ring)
6. `lookup_from_text()` false positives for magic tablets
7. Dead stash space re-triggering (content-based dedup)
8. Magic items not routed to trade API
9. Unknown currencies showing "~0 Chaos" (exalted, transmute, etc.)
10. Mod selection by numeric value excluding low-number high-value mods (+7 skills)
11. Rune mods leaking into trade query as "explicit"
12. All-mods "and" query returning 0 results (progressive search with mod classification)
13. Loose minimum values for tier-based mods (value-dependent multipliers)
14. Cheap outlier listings skewing price (outlier trimming)
15. Fractured/desecrated mods excluded from trade query

---

## Future Ideas (No Priority Yet)

### 1. Loot Filter Integration
Connect this tool with loot filters (like NeverSink-style). Since prices fluctuate constantly, having loot filter rules update automatically based on real-time prices would be a game changer. **No one is currently connecting these two.**

### 2. High-Value Visual Celebration
Stronger visual feedback when something is high value — animations, glow effects, sound alerts. Would drive engagement with the tool.

### 3. Currency vs Gear Education
Help newer players visualize the difference between currency items (Orbs, Catalysts, Runes) that have fixed exchange rates, and gear items where you need to figure out your own posting price.

### 4. Game-Native Visual Style
Make the price overlay look like it's actually part of the game UI (matching fonts, styling, borders) even though it isn't. Would drive engagement and feel more polished.

### 5. Cross-Platform / Resolution Testing
Test and ensure the tool works across different platforms, resolutions, and monitor setups (not just 3440x1440 ultrawide).

### 6. Desktop App Polish
Desktop dashboard exists (app.py + server.py + dashboard.html with pywebview). Next steps: installer/setup wizard, system tray icon, auto-start with Windows.

### 7. Auto-Launch with POE2
Detect when POE2 launches and automatically start the overlay. Could use Windows Task Scheduler or a lightweight tray service.

### 8. Multi-Game Support
Investigate using this pricing overlay approach for other games beyond POE2 (any game with tradeable items and public trade APIs).

### 9. GitHub Hosting
Set up remote repository on GitHub, proper releases, CI/CD, community contributions.

### 10. Buy Me a Coffee
Add a subtle "Buy me a coffee" button/link that connects to the user's accounts. Should be unobtrusive.

---

## How To Resume Development

**First time on a new machine:**
1. Clone: `git clone https://github.com/CarbonSMASH/POE2_OCR.git`
2. `cd POE2_OCR`
3. `pip install -r requirements.txt`
4. Run `claude` (Claude Code CLI)
5. Say: "Read CLAUDE_CODE_HANDOFF.md and continue development"

**Switching between machines:**
1. `cd POE2_OCR` → double-click `SYNC.bat` (or run `git pull && pip install -r requirements.txt`)
2. This pulls all latest code from GitHub and installs any new deps

**Running the tool:**
- **Dashboard GUI:** `python app.py` or double-click `POE2 Dashboard.bat`
- **CLI mode:** `python main.py --debug` or double-click `DEBUG.bat`
- **Tests:** `RUN_TESTS.bat` or `python -m pytest tests/ -v`

**Debug files:**
- Clipboard captures: `~/.poe2-price-overlay/debug/`
- Log file: `~/.poe2-price-overlay/overlay.log`
- Settings: `~/.poe2-price-overlay/dashboard_settings.json`
