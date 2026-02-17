# POE2 Price Overlay — Bug & Work Tracker

## Bugs
- [x] **Stale clipboard spam** — Fixed: distance-guarded reshow only allows re-fire when cursor returns near the original item position; stale cached data at new positions is suppressed. Also added clipboard retry (30ms) for slow game responses.
- [x] **"Terminate batch job?" on START.bat** — Fixed: Ctrl+C is now gated on `is_poe2_foreground()` check — keystrokes only sent when POE2 has focus, preventing console from receiving them. Removed trailing echo/pause from START.bat for clean exit.

## Backlog
- [ ] **ilvl breakpoint tables** — ilvl is now included in trade API queries for base items, but different slots have different ilvl breakpoints (bows need 82 for top phys%, wands only need 81). Could build per-slot breakpoint tables and consider ilvl in loot filter tiering for exceptional bases.
- [ ] **Common mod classification** — Heuristic-based; may occasionally misclassify an unusual valuable mod as "common". Mitigated by hybrid queries that always require key mods.
- [ ] **"Grants Skill:" edge cases** — Unusual skill grant formats might not be stripped by `_SKIP_LINE_RE`, leaking into trade queries.
- [x] **Rate limiting under burst** — Fixed: adaptive rate limiting parses `X-Rate-Limit-Ip` headers from API responses, backs off proactively at 50% usage per window. Socket retry path also guarded against wasted calls when rate limited.
- [ ] **Fancier ✗ dismiss indicator** — Current ✗ is plain Unicode in grey. Explore nicer options (custom icon, styled background, animation, etc.).
- [x] **Single key mod overpricing** — Items with ≤1 key mod among 4+ total now flagged as low confidence; results show as estimates with "(est.)" suffix and pulsing gold border.
- [ ] **User-configurable mod classification UI** — When we build an app interface, expose the common/key mod lists as toggleable options (radio buttons or checkboxes). Lets users override our defaults, adapt to meta shifts, and adjust per-league without code changes. Also addresses the "we can't be right for everyone" problem.
- [ ] **Scrap indicator for worthless items with quality/sockets** — Items dismissed as ✗ that have quality % or sockets should show a scrap icon (hammer 🔨) instead, reminding players to break them down. Scrapping quality/socketed items yields etchers, armour scraps, whetstones, baubles, gemcutters — all valuable for upgrades and worth trading on the currency exchange.
- [ ] **Resistance SOMV (Sum of Mod Value)** — High resist rolls add real value beyond what the trade query captures. If a resist mod is above 40% (fire/cold/lightning) or 20% (chaos), calculate a sum-of-mod-value bonus and factor it into the price. Could be a multiplier or flat addition to the estimate, helping differentiate a ring with 43% fire res from one with 20%.
- [ ] **Automated regression test suite** — `python mod_database.py` runs 29 mock items covering S/A/B/C/JUNK grades, edge cases, and tier comparisons across item classes. Should be extended into a proper test framework (`pytest`) that runs against all major CLs: mod_database scoring, mod_parser stat matching, item_parser clipboard parsing, trade_client query building. CI integration to run on every commit.
- [ ] **Currency icons in overlay** — Show small currency images (Divine, Exalted, Chaos, etc.) next to the price text in the overlay instead of just the name string. Makes prices instantly recognizable at a glance.
- [ ] **Chanceable base icons** — Show a Chance Orb icon and the target unique's icon (e.g., Headhunter) in the overlay for chanceable normal bases. Visual support alongside the text.
- [ ] **Pre-built calibration data shard** — Ship a curated `calibration.jsonl` with the repo so new users get reasonable price estimates from day one instead of starting from scratch. Update periodically as more data is collected. Consider league-aware shards (calibration data from one league may not apply to another). Harvester (`calibration_harvester.py`) can now generate these shards automatically.

## Completed

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
- [x] Claude prompt in Discord — each message includes copyable `claude "Look at bug: <title>"` for quick session start

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
