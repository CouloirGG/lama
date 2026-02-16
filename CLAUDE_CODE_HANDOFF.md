# POE2 Price Overlay — Claude Code Handoff

> **Last updated:** 2026-02-15
> **Status:** Working prototype — rare item pricing via Trade API is functional

## What This Is

A real-time price overlay for Path of Exile 2. Hover over any item in-game, it sends Ctrl+C to copy item data, parses the clipboard text, looks up the price, and shows an overlay near your cursor.

**Two pricing pipelines:**
1. **Static cache** (uniques, currency, gems) — prices from poe2scout.com, refreshed every 15 min
2. **Trade API** (rare/magic items) — queries pathofexile.com/trade with the item's actual mods

## Current Architecture

```
Cursor stops over POE2 window (8 fps polling)
  → Sends Ctrl+C via keybd_event (Windows API)
  → Reads clipboard text (POE2 item format)
  → ItemParser.parse_clipboard() → ParsedItem
  → If rare/magic with mods:
      → ModParser.parse_mods() → List[ParsedMod]
      → TradeClient.price_rare_item() → RarePriceResult
      → Shows "Checking..." then updates with price
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
| `item_parser.py` | Clipboard text → ParsedItem (name, base_type, rarity, mods) |
| `mod_parser.py` | **NEW** — Matches mod text to trade API stat IDs via regex |
| `trade_client.py` | **NEW** — Queries POE2 trade API for rare item pricing |
| `price_cache.py` | Fetches/caches prices from poe2scout.com (uniques, currency, gems) |
| `overlay.py` | Transparent tkinter overlay + ConsoleOverlay fallback |
| `clipboard_reader.py` | Windows clipboard reading via ctypes |
| `screen_capture.py` | POE2 window detection via Win32 API |
| `launcher.py` | GUI launcher (bat files call this) |
| `START.bat` | Main launcher |
| `DEBUG.bat` | Launches with `--debug` flag for verbose logging |
| `REPORT_BUG.bat` | Zips logs to Desktop, opens GitHub issue page |
| `LICENSE` | GPLv3 |
| `.gitignore` | Standard Python gitignore |

## Key Technical Details

### Clipboard Item Format (POE2)
```
Item Class: Staves
Rarity: Rare
Horror Weaver
Sanctified Staff
--------
Quality: +20% (augmented)
--------
Requires: Level 78, 75 Str, 75 Int
--------
Item Level: 82
--------
30% increased Spell Damage (rune)
+1 to Level of all Spell Skills (rune)
--------
Grants Skill: Level 18 Consecrate
--------
Gain 51% of Damage as Extra Lightning Damage
167% increased Spell Damage
+212 to maximum Mana
+7 to Level of all Cold Spell Skills
95% increased Critical Hit Chance for Spells
102% increased Mana Regeneration Rate
```

Sections separated by `--------`. Mod annotations in parentheses: `(implicit)`, `(rune)`, `(enchant)`, `(fractured)`, `(desecrated)`, `(mutated)`, `(crafted)`, `(augmented)`.

### Mod Parser (`mod_parser.py`)
- Fetches ~6114 stat definitions from `GET /api/trade2/data/stats`
- Caches to disk for 24h at `~/.poe2-price-overlay/cache/trade_stats.json`
- Compiles each stat text template (`"+# to maximum Life"`) into a regex
- ~3259 stats have working regexes (others have no `#` placeholder)
- `_match_mod()` preserves the original clipboard mod_type (rune stays rune, fractured stays fractured)

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

- **Caching:** In-memory, keyed by base_type + sorted stat_ids + rounded values, 5min TTL

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
- Rare items with mods price via trade API (tested: Horror Weaver staff ~100-500 Divine, Ghoul Noose amulet ~250-388 Divine, Pain Emblem focus ~50-130 Divine)
- Magic items route to trade API (not the static cache)
- Overlay shows "Checking..." immediately, then updates with price
- Content-based dedup prevents re-triggering on dead stash space

## Known Issues / Technical Debt
1. **Rate limiting under burst:** The progressive search can make 2-3 API calls per item. Testing multiple items rapidly can trigger 60s rate limit from trade API. Not an issue in normal single-item usage.
2. **"Grants Skill:" lines** are skipped via `_SKIP_LINE_RE` but unusual skill formats might slip through
3. ~~**Magic items without base_type**~~ — Fixed: `ModParser.resolve_base_type()` fetches all base types from the trade API items endpoint, caches to disk, and extracts the base type from a magic item name by longest substring match (e.g., "Mystic Stellar Amulet of the Fox" → "Stellar Amulet")
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

### 6. True Desktop App
Build it into a proper application (installer, system tray, settings GUI) instead of just PowerShell .bat launchers.

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

1. Clone: `git clone https://github.com/CarbonSMASH/POE2_OCR.git`
2. `cd POE2_OCR`
3. Run `claude` (Claude Code CLI)
4. Say: "Read CLAUDE_CODE_HANDOFF.md and continue development"
5. To test: `python main.py --debug` or double-click `DEBUG.bat`
6. Debug clipboard captures: `~/.poe2-price-overlay/debug/`
7. Log file: `~/.poe2-price-overlay/overlay.log`
