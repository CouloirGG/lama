# POE2 Price Overlay

**Real-time item pricing for Path of Exile 2 — zero setup, zero hotkeys.**

Hover over any item and instantly see its market value. No Ctrl+C, no Ctrl+D, no alt-tabbing. Just play.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        DETECTION PIPELINE                       │
│                                                                 │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐ │
│   │  Screen   │───▶│   OCR    │───▶│  Item    │───▶│  Price  │ │
│   │  Capture  │    │  Engine  │    │  Parser  │    │  Cache  │ │
│   └──────────┘    └──────────┘    └──────────┘    └────┬────┘ │
│        │                                                │      │
│   Watches 600x400           Extracts text         Looks up     │
│   region around cursor      from tooltips        poe.ninja     │
│   at 10 fps                                      local cache   │
│                                                        │       │
│                                                   ┌────▼────┐  │
│                                                   │ Overlay  │  │
│                                                   │ Window   │  │
│                                                   └─────────┘  │
│                                                   Shows price   │
│                                                   near cursor   │
└─────────────────────────────────────────────────────────────────┘
```

### Core Concepts

1. **Cursor Region Monitoring** — Only captures a small area around your cursor (not the full screen). This keeps CPU usage minimal (~3-5%).

2. **Visual Change Detection** — Compares frames to detect when a tooltip appears or a nameplate expands. Only triggers OCR when something actually changes.

3. **Local Price Cache** — Downloads all price data from poe.ninja every 15 minutes and stores it locally. Price lookups are instant (no API calls during gameplay).

4. **Transparent Overlay** — Click-through window that shows a color-coded price tag next to the item. Disappears after 4 seconds.

---

## Quick Start (Desktop App)

### Prerequisites

- **Python 3.10+** — [Download](https://www.python.org/downloads/)
  - ⚠️ Check **"Add Python to PATH"** during install
- **Tesseract OCR** — Install via one of:
  - PowerShell: `winget install UB-Mannheim.TesseractOCR`
  - Or download from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- **Windows 10/11** (required for overlay and cursor tracking)

### Launch

**Double-click `START.bat`** — that's it.

On first run it will:
1. Check Python and install missing packages
2. Verify Tesseract is installed
3. Ask which league you're playing
4. Launch the overlay

Subsequent launches skip setup and go straight to the overlay.

### Other Scripts

| File             | Purpose                                       |
|------------------|-----------------------------------------------|
| `START.bat`      | Main launcher (double-click this)             |
| `SETTINGS.bat`   | Change league, view logs, run tests           |
| `BUILD.bat`      | Build standalone .exe (optional, advanced)    |
| `launcher.py`    | Python launcher with first-run wizard         |

### Manual Launch (Advanced)

```bash
# Install dependencies once
pip install -r requirements.txt

# Run with specific league
python src/main.py --league "Dawn"

# Debug mode (verbose logging, console output)
python src/main.py --console --debug
```

---

## POE2 Game Settings (Recommended)

For best results, enable these in POE2 settings:

1. **Options → UI → Show Full Descriptions**: `ON`
   - Shows item level on ground nameplates
   - Enables accurate base type pricing

2. **Display Mode**: `Windowed Fullscreen` (borderless)
   - Required for overlay to appear on top of game

---

## Price Display

Prices are color-coded by value:

| Color  | Meaning              | Threshold    |
|--------|----------------------|--------------|
| 🟠 Orange | Very valuable      | ≥ 50 Exalted |
| 🟡 Gold   | Worth picking up   | ≥ 5 Exalted  |
| 🔵 Teal   | Decent value       | ≥ 1 Exalted  |
| ⚪ Grey   | Low value          | < 1 Exalted  |

---

## What Gets Priced

| Item Type       | Ground Nameplate | Hover Tooltip | Inventory |
|-----------------|:----------------:|:-------------:|:---------:|
| Currency        | ✅               | ✅            | ✅        |
| Unique Items    | ✅               | ✅            | ✅        |
| Skill Gems      | ✅               | ✅            | ✅        |
| Waystones/Maps  | ✅               | ✅            | ✅        |
| Valuable Bases  | ✅ (with ilvl)   | ✅            | ✅        |
| Rare Items      | Base value only  | ✅ (mods)     | ✅        |

---

## TOS Compliance

This tool is designed to be fully compliant with GGG's third-party tool policy:

- ❌ Does NOT inject into the game client
- ❌ Does NOT read game memory
- ❌ Does NOT modify any game files
- ❌ Does NOT automate any game actions
- ❌ Does NOT send any keypresses to the game
- ✅ ONLY reads pixels from the screen (passive observation)
- ✅ ONLY displays information in a separate overlay window
- ✅ Same approach used by Awakened PoE Trade, Exiled Exchange, etc.

---

## Architecture

```
poe2-price-overlay/
├── START.bat              # ← Double-click to launch
├── SETTINGS.bat           # Change league, view logs, run tests
├── BUILD.bat              # Build standalone .exe (optional)
├── launcher.py            # Python launcher with first-run wizard
├── build.spec             # PyInstaller config for .exe build
├── requirements.txt
├── README.md
├── src/
│   ├── main.py            # Entry point & orchestrator
│   ├── config.py           # All tunable constants
│   ├── screen_capture.py   # Cursor tracking & change detection
│   ├── ocr_engine.py       # Text extraction from screenshots
│   ├── item_parser.py      # Parse OCR text → structured item data
│   ├── price_cache.py      # poe.ninja data fetcher & local cache
│   ├── overlay.py          # Transparent overlay window
│   └── test_pipeline.py    # Pipeline validation tests
├── data/                   # Cached price data (auto-generated)
└── assets/                 # Icons, fonts (future)
```

---

## Configuration

All settings are in `src/config.py`. Key tunables:

| Setting                  | Default | Description                        |
|--------------------------|---------|------------------------------------|
| `SCAN_FPS`              | 10      | Capture checks per second          |
| `CHANGE_THRESHOLD`      | 25      | Pixel change sensitivity           |
| `DETECTION_COOLDOWN`    | 0.5s    | Minimum time between triggers      |
| `PRICE_REFRESH_INTERVAL`| 900s    | poe.ninja refresh interval         |
| `OVERLAY_DISPLAY_DURATION`| 4.0s  | How long price tag stays visible   |

---

## Development Roadmap

### Phase 1 — Python Prototype (Current)
- [x] Screen capture around cursor
- [x] Visual change detection
- [x] OCR text extraction
- [x] Item name/type parsing
- [x] poe.ninja price cache
- [x] Transparent overlay window
- [ ] Real-world accuracy testing with POE2
- [ ] Performance benchmarking

### Phase 2 — Polish & Optimize
- [ ] Windows OCR API integration (faster than Tesseract)
- [ ] Loot filter parsing (fast-path detection)
- [ ] Settings GUI (system tray)
- [ ] Auto-detect active league
- [ ] Overlay customization (size, position, opacity)

### Phase 3 — Steam Release
- [ ] Electron wrapper for Steam distribution
- [ ] Steam SDK integration
- [ ] Auto-update system
- [ ] Store page & marketing
- [ ] Community beta testing

---

## License

TBD — This is a prototype. Do not distribute without permission.
