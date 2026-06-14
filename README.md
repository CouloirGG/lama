# LAMA — Live Auction Market Assessor

**Real-time item pricing for Path of Exile 2 — clipboard-based, zero third-party installs.**

Copy any item with Ctrl+C and instantly see its market value in an overlay. Just clipboard monitoring and live market lookups — zero third-party installs.

---

## Quick Start

### 1. Install Python

Download Python 3.10+ from [python.org](https://www.python.org/downloads/).

**Important:** Check the box that says **"Add Python to PATH"** during install.

### 2. Install Git (if you don't have it)

Download from [git-scm.com](https://git-scm.com/downloads/win) and install with defaults.

### 3. Download the overlay

Open **File Explorer** and navigate to where you want the folder (Desktop, Documents, wherever).

Press **Ctrl+L** to select the address bar, type `powershell`, and hit **Enter** — this opens PowerShell in that folder.

Copy-paste this line into PowerShell and hit Enter:

```
git clone https://github.com/CouloirGG/lama.git
```

### 4. Run it

Open the `lama` folder and **double-click `START.bat`** — that's it.

On first run it will:
1. Install missing packages (`requests`)
2. Ask which league you're playing
3. Launch the overlay

Subsequent launches skip setup and go straight to the overlay.

### Requirements

- **Python 3.10+** (with "Add to PATH" checked)
- **Windows 10/11** (required for overlay and cursor tracking)

---

## How It Works

```
┌────────────────────────────────────────────────────────────┐
│                    CLIPBOARD PIPELINE                       │
│                                                            │
│   ┌───────────┐    ┌───────────┐    ┌──────────────────┐  │
│   │ Clipboard  │───>│   Item    │───>│   Price Cache    │  │
│   │ Monitor    │    │   Parser  │    │   (poe.ninja)    │  │
│   └───────────┘    └───────────┘    └────────┬─────────┘  │
│                                               │            │
│   Watches for        Parses item text    Looks up price    │
│   Ctrl+C in POE2     into structured     from local cache  │
│                      item data                             │
│                                          ┌────v─────────┐  │
│                                          │   Overlay     │  │
│                                          │   Window      │  │
│                                          └──────────────┘  │
│                                          Shows price near   │
│                                          cursor             │
└────────────────────────────────────────────────────────────┘
```

1. **Clipboard Monitoring** — Detects when you Ctrl+C an item in POE2. Parses the item text from the clipboard.
2. **Item Detection** — Identifies the item type (unique, currency, gem, rare, etc.) and extracts key properties including weapon DPS and armor defense values.
3. **Local Mod Scoring** — Rare/magic items are graded instantly (S/A/B/C/JUNK) using RePoE mod tier data. Attack weapons are penalized for low DPS; armor pieces for low defense. Zero API calls needed.
4. **Local Price Cache** — Downloads all price data from poe.ninja periodically and caches it locally. Price lookups are instant.
5. **Trade API** — Deep query (Ctrl+Shift+C) queries the POE2 trade API with the item's actual mods, DPS, and defense to find comparable listings.
6. **Transparent Overlay** — Click-through window that shows a color-coded price tag near your cursor.

---

## Hotkeys

| Hotkey | Context | What it does |
|--------|---------|-------------|
| **Ctrl+C** | In POE2 | Copy item — overlay shows price |
| **Ctrl+Shift+C** | In POE2 | Deep query — trade API lookup on last scored item |
| **Ctrl+Shift+B** | Anywhere | Open bug report dialog |

---

## Scripts

| File                   | Purpose                                        |
|------------------------|------------------------------------------------|
| `LAMA.bat`            | **Dashboard GUI** — the main way to run        |
| `START.bat`            | CLI launcher (no dashboard)                    |
| `DEBUG.bat`            | Launch with verbose logging to console         |
| `SETTINGS.bat`         | Change league, view logs, run tests            |
| `RUN_TESTS.bat`        | Run pytest suite — spawns a window per module  |
| `REPORT_BUG.bat`       | Zip logs and open a GitHub issue               |
| `scripts/SYNC.bat`     | **Multi-machine sync** — pull + install deps   |
| `scripts/BUILD.bat`    | Build distributable exe via PyInstaller        |

---

## Bug Reporting

Press **Ctrl+Shift+B** from anywhere (in-game or alt-tabbed). A dialog pops up — type what happened and hit Send (or Ctrl+Enter). It automatically attaches your logs, recent clipboard captures, and system info to Discord. No GitHub account needed, no zipping files.

Alternatively, **double-click `REPORT_BUG.bat`** to zip logs and open a GitHub issue.

---

## File Structure

```
lama/
├── LAMA.bat               # Dashboard GUI launcher (primary)
├── START.bat              # CLI launcher (no dashboard)
├── SETUP.bat              # One-click setup & install
├── requirements.txt
├── README.md
├── LICENSE
├── src/                   # All Python source files (43 modules)
│   ├── app.py             # Desktop dashboard shell (pywebview)
│   ├── server.py          # FastAPI backend (overlay mgmt, WS, settings)
│   ├── main.py            # Overlay engine & pricing pipeline
│   ├── config.py          # All tunable constants
│   ├── core/              # Game-agnostic pricing facade
│   │   ├── game_config.py # GameConfig dataclass (44 typed fields)
│   │   └── pricing_engine.py # PricingEngine — parse/score/price pipeline
│   ├── games/
│   │   └── poe2.py        # POE2 config factory (reference implementation)
│   ├── item_detection.py  # Cursor tracking, Ctrl+C, POE2 window detection
│   ├── item_parser.py     # Clipboard text → ParsedItem
│   ├── clipboard_reader.py # Windows clipboard I/O via ctypes
│   ├── mod_parser.py      # Mod text → trade API stat IDs
│   ├── mod_database.py    # Local mod scoring (RePoE tier data)
│   ├── calibration.py     # Score-to-price calibration (k-NN + GBM cascade)
│   ├── gbm_trainer.py     # Gradient Boosting Machine model training
│   ├── weight_learner.py  # Ridge regression mod weight learning
│   ├── price_cache.py     # poe2scout/poe.ninja data fetcher & local cache
│   ├── trade_client.py    # POE2 trade API client for rares
│   ├── builds_client.py   # poe.ninja Builds API (character lookup, meta data)
│   ├── stash_client.py    # OAuth2 stash tab API client
│   ├── stash_scorer.py    # Stash item scoring & quick-flip detection
│   ├── overlay.py         # Transparent overlay window (POE2 gothic theme)
│   ├── filter_updater.py  # Loot filter economy re-tiering
│   ├── watchlist.py       # Trade watchlist polling engine
│   ├── trade_actions.py   # Trade flow (whisper, invite, hideout, trade)
│   ├── game_commands.py   # Chat command engine via keystroke simulation
│   ├── demand_index.py    # Item demand trend tracking
│   ├── disappearance_tracker.py # Listing disappearance monitoring
│   ├── calibration_harvester.py # Bulk calibration data collection
│   ├── elite_harvester.py # High-value item harvester
│   ├── shard_generator.py # Calibration shard generation & compaction
│   ├── bug_reporter.py    # Discord webhook bug reporting
│   ├── flag_reporter.py   # One-click price inaccuracy flagging
│   ├── telemetry.py       # Opt-in session telemetry
│   ├── oauth.py           # OAuth2 flow for stash access
│   ├── bundle_paths.py    # Frozen-mode path resolution
│   ├── screen_capture.py  # POE2 window detection via Win32 API
│   ├── tray.py            # System tray icon integration
│   └── ...
├── resources/             # Bundled resource files
│   ├── dashboard.html     # Single-file React UI (frameless, POE2 theme)
│   ├── VERSION            # App version string
│   ├── NewBooBoo.filter   # Loot filter template
│   └── calibration_shard.json.gz  # Pre-built calibration data
├── scripts/               # Build & maintenance scripts
│   ├── BUILD.bat          # PyInstaller exe build
│   ├── SYNC.bat           # Multi-machine sync (git pull + pip install)
│   ├── build.spec         # PyInstaller spec file
│   └── installer.iss      # Inno Setup installer script
├── tests/                 # 255 tests across 10 modules
│   ├── conftest.py        # Shared pytest fixtures & factory helpers
│   ├── test_item_parser.py
│   ├── test_mod_parser.py
│   ├── test_mod_database.py
│   ├── test_trade_client.py
│   ├── test_calibration.py
│   ├── test_pricing_engine.py
│   ├── test_game_config.py
│   ├── test_shard_generator.py
│   ├── test_gbm_trainer.py
│   ├── test_weight_learner.py
│   └── fixtures/          # 24 real clipboard captures for tests
└── docs/                  # Developer documentation
    ├── DEVELOPER_HANDOFF.md
    ├── GAME_ABSTRACTION.md
    ├── PRICING_DATA_DEEP_DIVE.md
    ├── QA_TEST_PLAN.md
    └── TODO.md
```

---

## Testing

255 tests across 10 modules covering the full pricing pipeline:

| Module | Tests | What it covers |
|--------|-------|----------------|
| `test_mod_database` | 58 | Grade scoring (S/A/B/C/JUNK), SOMV, DPS/defense factors, skill level factor |
| `test_shard_generator` | 42 | Outlier removal, mod groups/tiers in shards, compact records, enrichment |
| `test_pricing_engine` | 24 | End-to-end integration, fixture parsing, scoring, module overrides |
| `test_calibration` | 24 | Holdout accuracy, mod identity, GBM/k-NN cascade, backward compat |
| `test_game_config` | 23 | GameConfig dataclass, POE2 factory, all 44 fields validated |
| `test_item_parser` | 21 | Clipboard parsing, combat stats, implicit separation, sockets |
| `test_weight_learner` | 19 | Ridge regression, tier-weighted features, archetype scores |
| `test_mod_parser` | 18 | Template-to-regex, opposite word matching, base type resolution |
| `test_trade_client` | 17 | Stat filters, common mod classification, price tiers |
| `test_gbm_trainer` | 9 | GBM training, pure-Python inference, serialization |

**Run all tests:**
```
RUN_TESTS.bat              # Spawns a PowerShell window per module
python -m pytest tests/ -v # All tests in one terminal
```

**Run a single module:**
```
python src/run_tests.py --module mod_database
python -m pytest tests/test_mod_database.py -v
```

---

## TOS Compliance

This tool is designed to be fully compliant with GGG's third-party tool policy:

- Does NOT inject into the game client
- Does NOT read game memory
- Does NOT modify any game files
- Does NOT automate any game actions
- Does NOT send any keypresses to the game
- ONLY reads text from the clipboard (Ctrl+C is a manual player action)
- ONLY displays information in a separate overlay window
- Same approach used by Awakened PoE Trade, Exiled Exchange, etc.

---

## Credits

Made by **Couloir** (cal schuss).

- **Email:** hello@couloir.gg
- **Web:** [couloir.gg](https://couloir.gg)

---

## License

See [LICENSE](LICENSE) for details.
