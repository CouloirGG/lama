# POE2 Price Overlay — Bug & Work Tracker

## Bugs
- [ ] **Stale clipboard spam** — Pain Emblem (or last-copied item) sometimes re-triggers at new cursor positions. Dedup TTL (30s) helps but doesn't fully prevent it. Root cause: game may return cached item data on Ctrl+C even when cursor isn't over an item.
- [ ] **"Terminate batch job?" on START.bat** — Moved `SetConsoleCtrlHandler` earlier but prompt may still appear occasionally. Needs further investigation.

## Backlog
- [ ] **Terminal window sizing** — Make the console window smaller (~400x400). `mode con` and PowerShell `MoveWindow` didn't take effect. Try alternative approaches.
- [ ] **Common mod classification** — Heuristic-based; may occasionally misclassify an unusual valuable mod as "common", affecting rare item pricing.
- [ ] **"Grants Skill:" edge cases** — Unusual skill grant formats might not be stripped by `_SKIP_LINE_RE`, leaking into trade queries.
- [ ] **Rate limiting under burst** — Progressive search makes 2-3 API calls per item. Rapid scanning can trigger 60s trade API ban. Not an issue in normal usage but could be improved.

## Completed
- [x] Overlay hides when cursor moves away (matches game tooltip)
- [x] Lower-bound pricing for rare items with "(est.)" indicator
- [x] Pulsing gold border for estimate prices
- [x] Value-tiered border effects (25/50/100/250/500/1000+ divine)
- [x] Trade query cancellation via generation counter
- [x] Stale clipboard detection (`new_text == original` check)
- [x] Magic item base_type resolution via trade API items endpoint
- [x] Mod annotation stripping, rune/fractured mod exclusion
- [x] Progressive search with mod classification (key vs common)
- [x] Outlier price trimming
- [x] Content-based dedup for dead stash space
