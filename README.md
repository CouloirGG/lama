# POE2 Price Overlay

**Real-time item pricing for Path of Exile 2 — clipboard-based, zero third-party installs.**

Copy any item with Ctrl+C and instantly see its market value in an overlay. No OCR, no Tesseract, no screen capture. Just clipboard monitoring and poe.ninja lookups.

---

## Quick Start

```
git clone https://github.com/CarbonSMASH/POE2_OCR.git
cd POE2_OCR
```

**Double-click `START.bat`** — that's it.

On first run it will:
1. Check Python and install missing packages (`requests`)
2. Ask which league you're playing
3. Launch the overlay

Subsequent launches skip setup and go straight to the overlay.

### Prerequisites

- **Python 3.10+** — [Download](https://www.python.org/downloads/) (check "Add Python to PATH" during install)
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

| File             | Purpose                                        |
|------------------|------------------------------------------------|
| `START.bat`      | Main launcher — double-click to run            |
| `DEBUG.bat`      | Launch with verbose logging to console         |
| `SETTINGS.bat`   | Change league, view logs, run tests            |
| `REPORT_BUG.bat` | Zip logs and open a GitHub issue               |

---

## Bug Reporting

Press **Ctrl+Shift+B** from anywhere (in-game or alt-tabbed). A dialog pops up — type what happened and hit Send (or Ctrl+Enter). It automatically attaches your logs, recent clipboard captures, and system info to Discord. No GitHub account needed, no zipping files.

Alternatively, **double-click `REPORT_BUG.bat`** to zip logs and open a GitHub issue.

---

## File Structure

```
POE2_OCR/
├── START.bat              # Main launcher
├── DEBUG.bat              # Debug mode launcher
├── SETTINGS.bat           # Settings menu
├── REPORT_BUG.bat         # Bug report helper
├── launcher.py            # Python launcher with first-run wizard
├── main.py                # Entry point & orchestrator
├── config.py              # All tunable constants
├── clipboard_reader.py    # Clipboard monitoring
├── item_detection.py      # Item type detection from clipboard text
├── item_parser.py         # Parse item text into structured data
├── mod_parser.py          # Mod parsing for rare item pricing
├── mod_database.py        # Local mod scoring engine (RePoE tier data)
├── calibration.py         # Score-to-price calibration engine
├── price_cache.py         # poe.ninja data fetcher & local cache
├── trade_client.py        # POE2 trade API client for rare items
├── bug_reporter.py        # Discord webhook bug reporting
├── filter_updater.py      # Loot filter economy re-tiering
├── overlay.py             # Transparent overlay window
├── screen_capture.py      # Screen region capture utilities
├── test_pipeline.py       # Pipeline validation tests
├── diagnose.py            # Diagnostic tool
├── requirements.txt
└── README.md
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

## License

See [LICENSE](LICENSE) for details.
