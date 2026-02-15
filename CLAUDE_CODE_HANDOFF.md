# POE2 Price Overlay — Project Handoff

## What This Is
A real-time price overlay for Path of Exile 2 that shows item market values when you hover over items in-game. Think "Poe Overlay" but lightweight — hover over an item, see its price from poe.ninja instantly.

**Target audience:** Casual players (95% of playerbase) who unknowingly skip valuable items.

## Tech Stack
- **Python 3.11** on Windows
- **mss** — fast screen capture
- **OpenCV + Pillow** — image processing
- **Tesseract OCR** — text extraction from screenshots
- **requests** — poe.ninja API
- **tkinter** — transparent overlay window (always-on-top)
- No game memory reading, no injection — 100% screen-based, TOS-safe

## Architecture (7 modules in `src/`)

```
Screen Capture → Change Detection → OCR → Item Parser → Price Lookup → Overlay
     ↓                                                        ↑
  mss grabs area          Tesseract reads        poe.ninja cache
  around cursor           tooltip text           (45+ currencies)
```

| Module | File | Purpose |
|--------|------|---------|
| Config | `src/config.py` | All settings, thresholds, paths |
| Screen Capture | `src/screen_capture.py` | Captures region around cursor, detects visual changes with cursor-stillness filter |
| OCR Engine | `src/ocr_engine.py` | Tesseract wrapper, preprocesses dark POE2 tooltips |
| Item Parser | `src/item_parser.py` | Extracts item name, base type, rarity, mods from OCR text |
| Price Cache | `src/price_cache.py` | Fetches/caches prices from poe.ninja POE2 API |
| Overlay | `src/overlay.py` | Transparent tkinter window, shows price tags |
| Orchestrator | `src/main.py` | Ties everything together, main loop |
| Tests | `src/test_pipeline.py` | Unit tests for each module |
| Diagnostics | `diagnose.py` | Step-by-step pipeline tester |

## Current Status: ALL DIAGNOSTICS PASS ✓

User ran `DIAGNOSE.bat` — all 8 steps green:
1. ✓ Python environment + all dependencies
2. ✓ Tesseract OCR installed and working
3. ✓ POE2 window detection (3440×1440 ultrawide)
4. ✓ Screen capture working
5. ✓ Cursor tracking working
6. ✓ **Price data from poe.ninja** — 45 currencies loaded
7. ✓ Overlay window renders (transparent, topmost)
8. ✓ Change detection fires

**But: overlay shows nothing in actual gameplay.**

## The Confirmed Working API Endpoint

```
GET https://poe.ninja/poe2/api/economy/exchange/current/overview
    ?league=Fate+of+the+Vaal
    &type=Currency
```

Response structure:
```json
{
  "core": {
    "items": [{"id": "divine", "name": "Divine Orb", ...}],
    "rates": {"exalted": 308.6, "chaos": 28.0},
    "primary": "divine"
  },
  "lines": [
    {"id": "divine", "primaryValue": 1.0, "volumePrimaryValue": 28386},
    {"id": "fracturing-orb", "primaryValue": 13.04, ...}
  ]
}
```

- `primaryValue` is in Divines (the primary currency)
- `core.rates` gives conversion: 1 Divine = X Chaos, Y Exalted
- This was discovered by user intercepting via Chrome DevTools (old endpoints were 404ing)

## Known Issues To Fix (Priority Order)

### 1. CRITICAL: Capture Region Too Small / Misaligned
**The user confirmed debug screenshots show tooltips partially cut off** — affixes visible but item name/header missing. The item name is at the TOP of the tooltip, so if the capture box starts at the cursor, it misses the header entirely.

**Why this matters:** Without the item name, OCR can't identify what the item is, and price lookup fails.

**Fix needed:** 
- Capture region needs to extend ABOVE the cursor, not just below
- POE2 tooltips appear above-right or above-left of cursor
- Current capture is 600×400 centered (or below) cursor — needs to be ~800×600 biased upward
- May need to detect tooltip boundaries dynamically (dark rectangle on screen)

### 2. HIGH: Change Detection Still Too Sensitive
Diagnostic showed triggers at 65%, 79%, 87% — these should be blocked by the 60% max filter. Either:
- The max filter isn't being applied in the live detection code
- Or the cursor-stillness check isn't working correctly

13 triggers in ~15 seconds is better than the original 54, but many are still camera pans (high % changes with moving cursor).

### 3. MEDIUM: OCR May Struggle With Real POE2 Tooltips
- POE2 tooltips have dark backgrounds, stylized fonts, colored text (magic/rare item names)
- The OCR preprocessing (contrast boost, thresholding) was tuned on clean test images
- May need custom preprocessing: invert dark backgrounds, isolate the tooltip rectangle, handle colored text
- Consider: only OCR the top portion of tooltip (where item name lives) for speed

### 4. MEDIUM: Item Parser May Not Handle POE2 Format
- POE2 tooltip format differs from POE1
- Rarity lines, base types, implicit/explicit mod formatting
- The parser was written speculatively — needs validation against real OCR output

### 5. LOW: Only Currency Prices Currently
- Only fetches from `exchange/current/overview` (currency items)
- Item endpoint (`item/current/overview`) may work with same pattern for uniques, gems, etc.
- Categories to add: UniqueWeapon, UniqueArmour, UniqueAccessory, SkillGem, etc.

## User's Setup
- **OS:** Windows, Python 3.11.9
- **Monitor:** 3440×1440 ultrawide (Monitor 3, primary)
- **POE2:** Windowed Fullscreen
- **Tesseract:** C:\Program Files\Tesseract-OCR\tesseract.exe
- **Project path:** C:\Users\Stu\GitHub\POE2_OCR
- **League:** Fate of the Vaal (current temp league, Patch 0.4.0)

## File Inventory

```
poe2-price-overlay/
├── START.bat              # Main launcher (league selection → run)
├── SETTINGS.bat           # Change league setting
├── DIAGNOSE.bat           # Run diagnostic tool
├── DISCOVER_API.bat       # API endpoint discovery
├── BUILD.bat              # PyInstaller build script
├── build.spec             # PyInstaller spec
├── requirements.txt       # pip dependencies
├── README.md              # User documentation
├── diagnose.py            # Diagnostic tool (standalone)
├── discover_api.py        # API endpoint tester
├── launcher.py            # GUI launcher
├── src/
│   ├── __init__.py
│   ├── config.py          # Settings & constants
│   ├── screen_capture.py  # Screen grab + change detection
│   ├── ocr_engine.py      # Tesseract OCR wrapper
│   ├── item_parser.py     # Text → item data
│   ├── price_cache.py     # poe.ninja API client
│   ├── overlay.py         # Transparent price tag window
│   ├── main.py            # Main orchestrator
│   └── test_pipeline.py   # Unit tests
├── assets/                # (empty, for icons later)
└── data/                  # (empty, for static data later)
```

## How To Continue

1. **Clone/open the project folder** in Claude Code
2. **Fix the capture region** — this is the #1 blocker. The tooltip header (item name) is being cut off
3. **Run with debug output:** `python src/main.py --debug --console` and hover items to see what OCR reads
4. **Check debug captures:** `~/.poe2-price-overlay/debug/trigger_*.png` to see what's being captured
5. **Test OCR on real captures** once capture region is fixed

## Stretch Goals (After Core Works)
- Steam Workshop / background service distribution (like Wallpaper Engine)
- Unique item prices (not just currency)
- Gem quality/level pricing
- Mini price history sparkline
- Configurable hotkey to toggle overlay
- Sound alert for high-value items
- Auto-league detection
