# LAMA — Live Auction Market Assessor

**Real-time item pricing for Path of Exile 2 — clipboard-based, zero third-party installs.**

Copy any item with Ctrl+C and instantly see its market value in an overlay. No OCR, no Tesseract, no screen capture. Just clipboard monitoring and poe.ninja lookups.

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

Open the `POE2_OCR` folder and **double-click `START.bat`** — that's it.

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
POE2_OCR/
├── LAMA.bat               # Dashboard GUI launcher (primary)
├── START.bat              # CLI launcher (no dashboard)
├── SETUP.bat              # One-click setup & install
├── requirements.txt
├── README.md
├── LICENSE
├── src/                   # All Python source files
│   ├── app.py             # Desktop dashboard shell (pywebview)
│   ├── server.py          # FastAPI backend (overlay mgmt, WS, settings)
│   ├── main.py            # Overlay engine & pricing pipeline
│   ├── config.py          # All tunable constants
│   ├── bundle_paths.py    # Frozen-mode path resolution
│   ├── item_detection.py  # Cursor tracking, Ctrl+C, POE2 window detection
│   ├── item_parser.py     # Clipboard text → ParsedItem
│   ├── mod_parser.py      # Mod text → trade API stat IDs
│   ├── mod_database.py    # Local mod scoring (RePoE tier data)
│   ├── calibration.py     # Score-to-price calibration engine
│   ├── price_cache.py     # poe2scout data fetcher & local cache
│   ├── trade_client.py    # POE2 trade API client for rares
│   ├── filter_updater.py  # Loot filter economy re-tiering
│   ├── overlay.py         # Transparent overlay window
│   ├── watchlist.py       # Trade watchlist polling engine
│   ├── bug_reporter.py    # Discord webhook bug reporting
│   └── ...
├── resources/             # Bundled resource files
│   ├── dashboard.html     # Single-file React UI (frameless, 3 tabs, POE2 theme)
│   ├── VERSION            # App version string
│   ├── NewBooBoo.filter   # Loot filter template
│   └── calibration_shard.json.gz  # Pre-built calibration data
├── scripts/               # Build & maintenance scripts
│   ├── BUILD.bat          # PyInstaller exe build
│   ├── SYNC.bat           # Multi-machine sync (git pull + pip install)
│   └── build.spec         # PyInstaller spec file
├── tests/
│   ├── conftest.py        # Shared pytest fixtures
│   ├── test_item_parser.py
│   ├── test_mod_parser.py
│   ├── test_mod_database.py
│   ├── test_trade_client.py
│   └── fixtures/          # Real clipboard captures for tests
└── docs/                  # Developer documentation
    ├── CLAUDE_CODE_HANDOFF.md
    └── TODO.md
```

---

## Testing

106 tests across 4 modules covering the full pricing pipeline:

| Module | Tests | What it covers |
|--------|-------|----------------|
| `test_item_parser` | 21 | Clipboard parsing, combat stats, implicit separation |
| `test_mod_parser` | 15 | Template-to-regex, mod matching, base type resolution |
| `test_mod_database` | 55 | Grade scoring (S/A/B/C/JUNK), SOMV, DPS/defense factors |
| `test_trade_client` | 15 | Stat filters, common mod classification, price tiers |

**Run all tests:**
```
RUN_TESTS.bat              # Spawns a PowerShell window per module
python -m pytest tests/ -v # All tests in one terminal
```

**Run a single module:**
```
python run_tests.py --module mod_database
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

- **Jira:** [couloirgg.atlassian.net/...PT/boards/36](https://couloirgg.atlassian.net/jira/software/projects/PT/boards/36)
- **Email:** hello@couloir.gg
- **Web:** [couloir.gg](https://couloir.gg)

---

## License

See [LICENSE](LICENSE) for details.
