# POE2 Price Overlay — Bug & Work Tracker

## Bugs
- [ ] **Stale clipboard spam** — Pain Emblem (or last-copied item) sometimes re-triggers at new cursor positions. Improved with dedup TTL (30s) and clipboard clear verification, but not fully resolved. Root cause: game may return cached item data on Ctrl+C even when cursor isn't over an item.
- [ ] **"Terminate batch job?" on START.bat** — Moved `SetConsoleCtrlHandler` earlier but prompt may still appear occasionally. Needs further investigation.

## Backlog
- [ ] **ilvl breakpoint tables** — ilvl is now included in trade API queries for base items, but different slots have different ilvl breakpoints (bows need 82 for top phys%, wands only need 81). Could build per-slot breakpoint tables and consider ilvl in loot filter tiering for exceptional bases.
- [ ] **Common mod classification** — Heuristic-based; may occasionally misclassify an unusual valuable mod as "common". Mitigated by hybrid queries that always require key mods.
- [ ] **"Grants Skill:" edge cases** — Unusual skill grant formats might not be stripped by `_SKIP_LINE_RE`, leaking into trade queries.
- [ ] **Rate limiting under burst** — Progressive search makes 2-3 API calls per item. Rapid scanning can trigger 60s trade API ban. Not an issue in normal usage but could be improved.
- [ ] **Fancier ✗ dismiss indicator** — Current ✗ is plain Unicode in grey. Explore nicer options (custom icon, styled background, animation, etc.).
- [x] **Single key mod overpricing** — Items with ≤1 key mod among 4+ total now flagged as low confidence; results show as estimates with "(est.)" suffix and pulsing gold border.
- [ ] **User-configurable mod classification UI** — When we build an app interface, expose the common/key mod lists as toggleable options (radio buttons or checkboxes). Lets users override our defaults, adapt to meta shifts, and adjust per-league without code changes. Also addresses the "we can't be right for everyone" problem.
- [ ] **Scrap indicator for worthless items with quality/sockets** — Items dismissed as ✗ that have quality % or sockets should show a scrap icon (hammer 🔨) instead, reminding players to break them down. Scrapping quality/socketed items yields etchers, armour scraps, whetstones, baubles, gemcutters — all valuable for upgrades and worth trading on the currency exchange.
- [ ] **Resistance SOMV (Sum of Mod Value)** — High resist rolls add real value beyond what the trade query captures. If a resist mod is above 40% (fire/cold/lightning) or 20% (chaos), calculate a sum-of-mod-value bonus and factor it into the price. Could be a multiplier or flat addition to the estimate, helping differentiate a ring with 43% fire res from one with 20%.
- [ ] **Currency icons in overlay** — Show small currency images (Divine, Exalted, Chaos, etc.) next to the price text in the overlay instead of just the name string. Makes prices instantly recognizable at a glance.
- [ ] **Chanceable base icons** — Show a Chance Orb icon and the target unique's icon (e.g., Headhunter) in the overlay for chanceable normal bases. Visual support alongside the text.

## Completed

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
