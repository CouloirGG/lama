# POE2 Price Overlay — Bug & Work Tracker

## Bugs
- [ ] **Stale clipboard spam** — Pain Emblem (or last-copied item) sometimes re-triggers at new cursor positions. Improved with dedup TTL (30s) and clipboard clear verification, but not fully resolved. Root cause: game may return cached item data on Ctrl+C even when cursor isn't over an item.
- [ ] **"Terminate batch job?" on START.bat** — Moved `SetConsoleCtrlHandler` earlier but prompt may still appear occasionally. Needs further investigation.

## Backlog
- [ ] **Terminal window sizing** — Make the console window smaller (~400x400). `mode con` and PowerShell `MoveWindow` didn't take effect. Try alternative approaches.
- [ ] **Common mod classification** — Heuristic-based; may occasionally misclassify an unusual valuable mod as "common", affecting rare item pricing.
- [ ] **"Grants Skill:" edge cases** — Unusual skill grant formats might not be stripped by `_SKIP_LINE_RE`, leaking into trade queries.
- [ ] **Rate limiting under burst** — Progressive search makes 2-3 API calls per item. Rapid scanning can trigger 60s trade API ban. Not an issue in normal usage but could be improved.
- [ ] **Fancier ✗ dismiss indicator** — Current ✗ is plain Unicode in grey. Explore nicer options (custom icon, styled background, animation, etc.).
- [ ] **Single key mod overpricing** — When an item has only 1 key mod (e.g., +36 spirit) among all-common filler, the trade API returns inflated prices from items with the same mod but much better overall stats. Need a smarter approach: e.g., discount result when key-to-common ratio is low, require minimum key mod count, or apply value-based thresholds per mod type.

## Completed

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
