# LAMA (Live Auction Market Assessor) — Bug & Work Tracker

**Project Tracker:** Internal

## Bugs
- [x] **KPI card currency icon swap** — Fixed: chaos_orb.png and exalted_orb.png filenames were swapped
- [x] **Mirror KPI missing icon** — Fixed: added mirror_of_kalandra.png and wired into Mirror KPI card
- [x] **Stale clipboard spam** — Fixed: distance-guarded reshow only allows re-fire when cursor returns near the original item position; stale cached data at new positions is suppressed. Also added clipboard retry (30ms) for slow game responses.
- [x] **"Terminate batch job?" on START.bat** — Fixed: Ctrl+C is now gated on `is_poe2_foreground()` check — keystrokes only sent when POE2 has focus, preventing console from receiving them. Removed trailing echo/pause from START.bat for clean exit.

## Backlog
- [x] **Overlay UI overhaul** — Full redesign of the in-game overlay text UI. Current overlay looks dated (90s aesthetic); needs a clean, modern, unobtrusive design that matches LAMA's dashboard styling. Consider: semi-transparent glass/frosted panels, refined typography, subtle animations, tier-based color accents without visual clutter, compact information density. Should feel native to POE2's aesthetic.
- [x] **Opt-in session telemetry** — Users currently have no way to share diagnostic data. Add an opt-in system where users can upload session data (scored items, estimates vs actuals, errors, performance) for investigation. Needs: consent toggle in dashboard settings, lightweight upload endpoint, privacy-respecting data collection (no PII), periodic batch upload rather than real-time.
- [x] **Flag inaccurate result** — Quick one-click way for users to flag a price estimate as wrong directly from the overlay. Could be a small button/hotkey on the overlay itself. Flagged items should include the item data, our estimate, and optionally the user's correction. Feed into calibration improvement pipeline.
- [x] **Harvester speed improvements** — (1) Fetch 50 items per query instead of 20 (2.5x samples/query), (2) burst-then-pause rate strategy (4 calls + 8s pause), (3) auto-wait through penalties instead of exiting, (4) skip dead category/bracket combos with `--reset-dead` CLI flag. Target ~20-25 min/pass.
- [x] **Process name in Task Manager** — `pythonw.exe` is hard to find; set a custom process/app name so the app shows properly in Task Manager's Apps list (e.g., "POE2 Price Overlay")
- [x] **ilvl breakpoint tables** — Scoring engine is now ilvl-aware: `identify_tier()` filters out unrollable mod tiers, percentile computed against rollable range only, DPS brackets expanded to 4 ilvl tiers (82/68/45/0), defense thresholds keyed by ilvl per slot, `price_cache._adjust_ilvl()` smoothed to 7 brackets. Diagnostic logging when ilvl caps a tier assignment.
- [ ] **Common mod classification** — Heuristic-based; may occasionally misclassify an unusual valuable mod as "common". Mitigated by hybrid queries that always require key mods.
- [x] **"Grants Skill:" edge cases** — Unusual skill grant formats might not be stripped by `_SKIP_LINE_RE`, leaking into trade queries.
- [x] **Rate limiting under burst** — Fixed: adaptive rate limiting parses `X-Rate-Limit-Ip` headers from API responses, backs off proactively at 50% usage per window. Socket retry path also guarded against wasted calls when rate limited.
- [ ] **Fancier ✗ dismiss indicator** — Current ✗ is plain Unicode in grey. Explore nicer options (custom icon, styled background, animation, etc.).
- [x] **Single key mod overpricing** — Items with ≤1 key mod among 4+ total now flagged as low confidence; results show as estimates with "(est.)" suffix and pulsing gold border.
- [ ] **User-configurable mod classification UI** — When we build an app interface, expose the common/key mod lists as toggleable options (radio buttons or checkboxes). Lets users override our defaults, adapt to meta shifts, and adjust per-league without code changes. Also addresses the "we can't be right for everyone" problem.
- [ ] **Scrap indicator for worthless items with quality/sockets** — Items dismissed as ✗ that have quality % or sockets should show a scrap icon (hammer 🔨) instead, reminding players to break them down. Scrapping quality/socketed items yields etchers, armour scraps, whetstones, baubles, gemcutters — all valuable for upgrades and worth trading on the currency exchange.
- [x] **SOMV (Sum of Mod Values)** — Implemented as a universal roll quality factor (0.90-1.10) applied to all mods with tier data. Perfect rolls get ~10% boost, bottom rolls get ~10% penalty. Chaos resistance bumped to weight 0.5 (from 0.3) reflecting rarity and ES-bypass value. 6 new test cases, 40/40 pass.
- [x] **Automated regression test suite** — 106 pytest tests across 4 modules (item_parser, mod_parser, mod_database, trade_client). Fixtures from real clipboard captures. `RUN_TESTS.bat` spawns a PowerShell window per module for visual monitoring. Also runnable via `python -m pytest tests/ -v`.
- [ ] **Currency icons in overlay** — Show small currency images (Divine, Exalted, Chaos, etc.) next to the price text in the overlay instead of just the name string. Makes prices instantly recognizable at a glance.
- [ ] **Chanceable base icons** — Show a Chance Orb icon and the target unique's icon (e.g., Headhunter) in the overlay for chanceable normal bases. Visual support alongside the text.
- [x] **Pre-built calibration data shard** — Ship a curated `calibration.jsonl` with the repo so new users get reasonable price estimates from day one instead of starting from scratch. Update periodically as more data is collected. Consider league-aware shards (calibration data from one league may not apply to another). Harvester (`calibration_harvester.py`) can now generate these shards automatically.

## Completed

### Session 25 (2026-02-20)
- [x] **Trade action buttons** — Full trade flow in Watchlist tab: Whisper, Invite, Hideout, Trade, Kick buttons per listing. Chat command engine (`game_commands.py`) sends commands via keystroke simulation + clipboard paste. Authenticated mode (`trade_actions.py`) uses POESESSID + whisper_token/hideout_token for API-based actions without chat. POESESSID input with password masking + clear button in Watchlist settings. Token indicators show API vs chat mode per listing.

### Session 24 (2026-02-20)
- [x] **ilvl breakpoint tables (PT-12)** — `identify_tier()` now skips tiers whose `required_level > item_level`, percentile denominator scoped to rollable tiers only, DPS brackets expanded from 2 to 4 ilvl tiers, defense thresholds changed from flat tuples to ilvl-keyed dicts, `_adjust_ilvl()` smoothed from 5 to 7 brackets removing cliff-edge at ilvl 80→75. All 40 tests pass.

### Session 23 (2026-02-19)
- [x] **POE2 gothic overlay theme** — Full theme system with grunge effects (blood splatters, scratch marks, vignette, corner diamonds), sheen sweep animation, serif font fallback chain (`Palatino Linotype → Book Antiqua → Georgia → Segoe UI`), and tier-based border styling. Theme (`poe2`/`classic`) and pulse style (`sheen`/`border`/`both`/`none`) exposed as dashboard settings.
- [x] **Dashboard layout reorganization** — Detection settings moved from right column into expanded OverlayPreview section (2×2 grid). Right column now shows InventoryPreview with real item art.
- [x] **Inventory preview with POE2 item art** — 8×4 grid using real item images from `web.poecdn.com` CDN (sword, body armor, helmet, ring, boots, gloves). SVG silhouette filler icons for non-art cells. Overlay bars use actual `OverlaySampleChip` component for 1:1 match with in-game overlay.
- [x] **Calibration shard regenerated** — From harvester passes 6-15: 15,302 raw → 3,849 samples (25.3 KB). Validation: 75.7% within 2x accuracy (PASS, target ≥70%).
- [x] **Game abstraction docs** — `GAME_ABSTRACTION.md` covering `GameConfig` dataclass and `PricingEngine` facade; `DEVELOPER_HANDOFF.md` updated with `core/` and `games/` modules, test count updated to 153.

### Session 22 (2026-02-19)
- [x] **Overlay tier customization** — Per-tier color pickers (text, border, background) in dashboard with live preview; custom styles persist in `overlay_tier_styles` setting and apply in overlay subprocess
- [x] **In-game inventory mockup** — Expanded overlay preview shows 12x5 inventory grid with filler items, highlighted item under cursor, POE2 tooltip, and LAMA overlay tag; click any tier to preview
- [x] **Collapsible overlay preview** — Collapsed state shows 3 representative chips; expanded shows full preview + tier editor + "Reset to defaults"
- [x] **Sound toggle UI** — Per-tier sound toggle rendered (UI only, not wired to audio)
- [x] **Compact scanner card** — Stopped state reduced to single inline row (red dot + text + Start button)
- [x] **Compact detection panel** — 4-column single-row layout with tighter padding
- [x] **Overlay display presets with toggles** — format_overlay_text accepts show_grade/show_price/show_stars/show_mods/show_dps flags from dashboard settings
- [x] **Item lookup endpoint** — `/api/item-lookup` POST endpoint + ItemLookup class for paste-and-score in dashboard
- [x] **Shortcut AppUserModelID** — create_shortcut.py stamps System.AppUserModel.ID so pinned taskbar icon matches running app

### Session 21 (2026-02-18)
- [x] **Enhanced Markets chart selectors** — Currency multi-select (click/Ctrl+Click), denomination picker (Chaos/Divine/Exalted), time range pills (7d/14d/30d), Top 5 button, smooth curved lines, currency icon markers at peaks/lulls
- [x] **Multi-series normalization** — Multi-select uses % change so currencies with different values are comparable; Y-axis and crosshair switch to % mode
- [x] **Rate history extended to 30 days** — History retention `7d → 30d`, `oldest_history_ts` exposed so frontend can grey out unavailable time ranges
- [x] **OneDrive backup for rate history** — `rate_history.jsonl` auto-copies to `~/OneDrive/POE2PriceOverlay/` on every write; restores from backup on load if primary missing
- [x] **KPI cards updated** — Hinekora's Lock→Divine, Fracturing→Divine, Omen of Light→Divine, Essence of the Abyss→Exalted; currency icons from poe.ninja
- [x] **Category filter fix** — Case-insensitive comparison (poe2scout lowercase vs PascalCase pills)
- [x] **Chart edge padding** — Lines extend to chart edges with `fixLeftEdge`/`fixRightEdge`
- [x] **Number formatting** — Large values abbreviated (54.4k), decimals dropped on whole numbers, theme-matched chart colors
- [x] **TradingView watermark hidden** — CSS rule hides attribution link

### Session 20 (2026-02-18)
- [x] **Fix Restart App** — server broadcasts `app_restart` via WebSocket so dashboard calls `pywebview.api.close()` for clean WebView2 shutdown; `TerminateProcess` fallback
- [x] **Mirror KPI** — added Mirror-to-Divine exchange rate card to dashboard; pipeline through price_cache → main.py status line → server.py → dashboard (PT-41)
- [x] **Item names in filter tiers** — new `/api/filter-items` endpoint reads price cache, computes tier assignments per economy section; SectionRow shows item name badges with chaos values under each tier (PT-38)
- [x] **No-scroll dashboard layout** — fixed header/KPIs/tabs/footer with scrollable tab content (PT-40)
- [x] **Title bar overlay controls** — Start/Stop/Restart buttons + connection/state badges in title bar (PT-32)
- [x] **De-emphasized Bug/Restart** — moved to footer as text links (PT-33)
- [x] **League in footer** — dropdown moved from prominent header to footer (PT-37)
- [x] **Settings overhaul** — grouped Display/Detection panels, auto-start toggle (PT-35)
- [x] **Disclaimers** — accuracy warnings on status panel, filter tab, watchlist tab (PT-36)
- [x] **Credits flyout** — "Thanks To" section: GGG, poe.ninja, NeverSink, RePoE, community (PT-42)
- [x] **Feedback/Feature Request** — modal dialog + Discord webhook endpoint, footer buttons (PT-34)
- [x] **Footer readability** — bumped button/select colors from `textMuted` to `textSecond`

### Session 19 (2026-02-18)
- [x] Tooltip z-index fix — rewrote Tooltip component to use `position: fixed` with `getBoundingClientRect()`, preventing clipping inside scrollable containers (PT-27, PT-31)
- [x] KPI card reorder — Divine→Exalted first, then Divine→Chaos, then Exalted→Chaos (PT-28)
- [x] Window resize reflow — added `scrollbarGutter: stable` to prevent content shift when scrollbar appears/disappears (PT-29)
- [x] Whisper button tooltip — added "Copy whisper message to clipboard" tooltip to Whisper button in watchlist (PT-30)

### Session 18 (2026-02-18)
- [x] Frameless window — removed OS title bar, added custom title bar with drag region and min/max/close buttons (pywebview `frameless=True` + `WindowApi` via Win32 ctypes)
- [x] Rounded app frame — `border-radius: 10px` with layered box-shadow replacing corner diamond pseudo-elements
- [x] Scrollable content area — title bar stays pinned, content scrolls beneath via flex layout
- [x] Hide console on dashboard launch — `POE2 Dashboard.bat` uses `pythonw` + `start ""` to detach
- [x] About modal — Couloir branding, version display, GitHub link; opens from title bar click
- [x] Calibration shards — pre-built compressed JSON for instant accuracy from first launch; shard loading in CalibrationEngine
- [x] Calibration write-time quality filters — skip estimates, price-fixers, thin results
- [x] Harvester multi-pass + shard output — generates distributable `.json.gz` shard files
- [x] Status line extended — divine-to-chaos, divine-to-exalted, calibration sample count parsed by server
- [x] Build spec updated — bundles calibration shards (`.json.gz`) and image assets (`resources/img/`)
- [x] Version exposed in `/api/status` response

### Session 17 (2026-02-18)
- [x] Inno Setup installer — `scripts/installer.iss` reads VERSION, bundles PyInstaller output, installs to `%LOCALAPPDATA%\POE2PriceOverlay`, creates desktop + Start Menu shortcuts
- [x] BUILD.bat step 3 — auto-detects Inno Setup, builds `dist/POE2PriceOverlay-Setup-{version}.exe`
- [x] One-click auto-update — `/api/apply-update` endpoint downloads Setup exe from GitHub releases with WebSocket progress streaming, launches `/SILENT` install
- [x] Dashboard update banner — "Install Update" button with download progress bar and installing state

### Session 16 (2026-02-17)
- [x] File cleanup — deleted 8 superseded files (launcher.py, test_pipeline.py, SETUP_GUIDE.md, 5 .bat wrappers), moved 4 dev docs to `docs/`
- [x] PyInstaller distributable exe — `bundle_paths.py` for frozen-mode paths, `--overlay-worker` flag for single-exe subprocess spawning, rewritten `build.spec` (app.py entry, proper hiddenimports, console=False)
- [x] Auto-update check — `VERSION` file (1.0.0), server checks GitHub releases API on startup, dashboard shows dismissible gold update banner
- [x] `BUILD.bat` fixed — `python -m pip`, `python -m PyInstaller` (works when scripts dir not on PATH)
- [x] Verified: exe builds, dashboard launches, overlay subprocess starts from frozen exe

### Session 15 (2026-02-17)
- [x] Automated regression test suite — 106 pytest tests across 4 modules: `test_item_parser` (21 tests), `test_mod_parser` (15 tests), `test_mod_database` (55 tests migrated from `__main__` + new), `test_trade_client` (15 tests)
- [x] Test fixtures — 13 curated real clipboard captures (rares, uniques, currency, gems, magic) in `tests/fixtures/`
- [x] `conftest.py` — session-scoped ModParser/ModDatabase fixtures, stat ID resolver, make_item/make_mod helpers
- [x] `run_tests.py` + `RUN_TESTS.bat` — PowerShell window spawner per test module with visual pass/fail
- [x] pytest added to requirements.txt

### Session 14 (2026-02-17)
- [x] SOMV (Sum of Mod Values) roll quality factor — `ModScore.roll_quality` (0.0-1.0 within tier), `ItemScore.somv_factor` (0.90-1.10 multiplier based on avg roll quality across all mods with tier data)
- [x] Perfect rolls boost score ~10%, bottom rolls penalize ~10% — 20% total spread between perfect and bottom rolls of identical mods
- [x] Chaos resistance weight bumped from 0.3 to 0.5 — rarer than elemental res, lower cap (27% vs 45%), bypasses energy shield
- [x] SOMV factor logged in overlay output and recorded in calibration JSONL (main.py + calibration_harvester.py)
- [x] 6 new SOMV test cases (V1-V6) + SOMV diagnostics section in test harness — 40/40 tests pass

### Session 13 (2026-02-17)
- [x] Calibration harvester (`calibration_harvester.py`) — standalone CLI that queries trade API for rare items across 24 equipment categories x 4 price brackets (96 queries), scores them locally via ModDatabase, and writes (score, price) calibration pairs to `calibration.jsonl`
- [x] Resumability — state file tracks completed queries with date-based seed; re-running same day skips finished queries
- [x] Fake listing detection — sanity filter rejects JUNK items at 5+ divine, C items at 50+ divine, and low-score items at 20+ divine (caught 157 price-fixers in first run)
- [x] First harvest: 486 raw samples collected, cleaned to 392 across 19 item classes (10x increase from 37 manual samples)
- [x] `HARVESTER_STATE_FILE` added to config.py

### Session 12 (2026-02-17)
- [x] New 0.5 weight tier for armour/evasion — secondary defense mods no longer count as "key mods", fixing pure-evasion items (like Gloom Veil) getting false A grades
- [x] Spirit added to weight table at 1.0 (Standard) — was accidentally getting 2.0 via unknown mod fallback
- [x] Charm duration/effect added to Filler (0.3) — belt charm mods were inflating grades (Ambush Tether A→C)
- [x] Unknown non-common mod fallback reduced from 2.0 to 1.0 — prevents niche mods from getting Key weight
- [x] Fixed "Rarity of Items found" not matching common pattern "item rarity" — word order mismatch was causing rarity to be classified as key mod, inflating grades on multiple items (Wrath Salvation A→JUNK, Corruption Coil A→JUNK)
- [x] Fixed "effect of Socketed Items" not in common patterns — niche mod was getting key weight
- [x] Fixed armour display name — LocalPhysicalDamageReductionRating now shows "Armour" instead of "PhysDmg"
- [x] Fixed orange C display bug — calibration estimate no longer overrides text color for C/JUNK grades

### Session 11 (2026-02-16)
- [x] Auto-calibration queue — A/S items always queued for background trade API lookup, B sampled 1-in-3, C/JUNK sampled 1-in-10. Results logged to calibration.jsonl silently.
- [x] Fix Ctrl key desync — synthetic Ctrl+C now skips Ctrl down/up when user is already holding Ctrl physically, preventing game from seeing bare 'C' keypresses during rapid stash scanning
- [x] Fix Mana weight inflation — moved maximummana/increasedmana from Standard (1.0) to Filler (0.3); was driving false A/S grades
- [x] Fix T40 phantom tier scoring — percentile floor (0.30) now only applies to T1-T10; deep tiers in oversized ladders get natural low percentile
- [x] Fix Life/AllRes as common in trade — removed "maximum life" and "to all elemental resistances" from common patterns; they drive real value on jewelry
- [x] Cap wild calibration estimates — capped at min(2x max observed sample, 500 divine) to prevent 8000+ divine extrapolations with few samples
- [x] Fix rate-limited auto-cal items lost — queue waits out rate limits and re-queues items instead of discarding
- [x] Tighten S/A grade requirements — S needs 3+ key mods + 4+ total mods; A needs 3+ total mods
- [x] Added display names for "damage to" mod groups (AddPhys, AddFire, etc.)

### Session 10 (2026-02-16)
- [x] Discord webhook bug reporting — Ctrl+Shift+B opens dark-themed dialog, collects logs + clipboard captures + system info, uploads to Discord channel
- [x] Bug report dialog — standalone Toplevel (not transient), forced above fullscreen via Win32 SetWindowPos, timestamp header, title + description fields, Ctrl+Enter to send
- [x] Local bug report database — each report appended to `bug_reports.jsonl` for cross-session analysis
- [x] Debug prompt in Discord — each bug report includes copyable `Look at bug: <title>` for quick triage

### Session 9 (2026-02-16)
- [x] DPS & defense stat integration — weapons scored by total DPS output, armor by total defense; low values crush grade
- [x] Combat stat parsing — extract physical/elemental damage, APS, armour, evasion, ES from clipboard text
- [x] DPS scoring factor — multiplicative penalty for low-DPS attack weapons (0.15 for terrible, up to 1.15 for exceptional); staves/wands excluded as caster weapons
- [x] Defense scoring factor — softer penalty for low-defense armor pieces (0.6 for terrible, up to 1.05); per-slot thresholds (body armour through foci)
- [x] Trade API DPS/defense filters — equipment_filters include `dps`, `ar`, `ev`, `es` minimums (75%/70% of item's values) for comparable pricing
- [x] Overlay combat tags — shows "280dps" or "850def" when factor penalizes (replaces star count)
- [x] Calibration JSONL extended — records include total_dps, total_defense, dps_factor, defense_factor
- [x] 5 new test cases + DPS/defense factor diagnostic curves (34/34 pass)

### Session 8 (2026-02-16)
- [x] Adaptive rate limiting — parse `X-Rate-Limit-Ip` and `X-Rate-Limit-Ip-State` headers from every API response to learn actual rate windows (e.g., 5/10s, 15/60s, 30/300s)
- [x] Proactive backoff — at 50% usage in any window, switch to safe rate (`window_secs / max_hits`); most conservative interval wins
- [x] Rate limit rule discovery logging — first API call logs all discovered windows and penalties
- [x] Socket retry rate-limit guard — skip `_search_progressive` retry when already rate limited, with clear log message

### Session 7 (2026-02-16)
- [x] Fix stale clipboard spam — distance-guarded reshow via `_reshow_origin_pos`; suppress stale cached data at new positions
- [x] Clipboard retry — 30ms retry on empty clipboard read after Ctrl+C
- [x] Fix "Terminate batch job?" — gate Ctrl+C on `is_poe2_foreground()` so keystrokes only go to POE2
- [x] Add `GameWindowDetector.is_poe2_foreground()` — checks `GetForegroundWindow()` title
- [x] Clean START.bat exit — remove trailing echo/pause

### Session 6 (2026-02-16)
- [x] API call budget — `_search_progressive` now caps `_do_search` invocations via `max_calls` param (default 6), prevents single niche item from burning through rate limit
- [x] Broad search for niche items — items with fractured/desecrated mods or 3+ sockets get a two-phase strategy: quick base-type probe (1 call) then broad count(n-1)/count(n-2) queries without base type (up to 3 calls)
- [x] "+" indicator for unpriced valuable items — pulsing green "+" with gold border when all searches fail but item has value signals (fractured/desecrated/3S)
- [x] Optional base_type in query builders — `_build_query` and `_build_hybrid_query` omit `type` field when base_type is None, enabling category-agnostic searches

### Session 5 (2026-02-16)
- [x] Deduplicate common mod patterns — single canonical list in TradeClient, main.py references it
- [x] Wire hybrid queries into progressive search — key mods as "and" + common as "count" for 5+ mod items
- [x] Low-key-confidence flag — items with ≤1 key mod among 4+ total show as estimates (prevents overpricing)

### Session 4 (2026-02-16)
- [x] ilvl-aware base pricing — trade API queries now include ilvl min filter, separate cache entries per ilvl, overlay shows ilvl in display text
- [x] Terminal window sizing — resize console to ~420x400 via Win32 `GetConsoleWindow` + `MoveWindow`

### Session 3 (2026-02-15)
- [x] Fix clipboard reader same-item bug — clear verification instead of comparing with original
- [x] Unidentified item handling — uniques show price range by base type, rares/magic skip
- [x] Base item pricing via trade API for normal/magic items with 2+ sockets
- [x] Fix socket parsing — POE2 uses "S S S" format, count S characters
- [x] Strip quality prefixes from base types (Exceptional/Superior/Masterful)
- [x] Fix socket filter — POE2 uses `rune_sockets` in `equipment_filters`
- [x] Skip trade API for magic items with only common mods (instant "Low value")
- [x] Show "Low value" for magic items with no trade results
- [x] Store `base_type` in price cache entries for unidentified unique lookups
- [x] Reduce overlay display duration from 4s to 2s
- [x] Simplify START.bat banner text
- [x] Add TODO.md for tracking

### Session 2 (2026-02-15)
- [x] Overlay hides when cursor moves away (matches game tooltip)
- [x] Lower-bound pricing for rare items with "(est.)" indicator
- [x] Pulsing gold border for estimate prices
- [x] Value-tiered border effects (25/50/100/250/500/1000+ divine)
- [x] Trade query cancellation via generation counter
- [x] Stale clipboard detection (`new_text == original` check)
- [x] Content-based dedup preserves across cursor moves (30s TTL)
- [x] Move `SetConsoleCtrlHandler` to start of `main()` before threads

### Session 1
- [x] Magic item base_type resolution via trade API items endpoint
- [x] Mod annotation stripping, rune/fractured mod exclusion
- [x] Progressive search with mod classification (key vs common)
- [x] Outlier price trimming
- [x] Content-based dedup for dead stash space
