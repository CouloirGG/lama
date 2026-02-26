## LAMA v0.2.6

### New Features
- Add roll quality, DPS/defense enrichment to calibration pipeline (shard v7)
- Add gothic-themed mini frame for dismiss and scrap overlay indicators
- Add character viewer, popular items, stash viewer scaffold, and saved characters
- Add learned mod weights via Ridge regression for price estimation
- Add mod-identity features to calibration k-NN
- Add LAMA splash screen with crossfade launch sequence
- Add overlay config export/import with native file dialogs
- Add top-tier and mod-count features to calibration k-NN
- Add upload feedback and logging to telemetry
- Add 20% drop-rate guard and tests for shard outlier removal
- Add files via upload
- Add full QA test plan for alpha testers
- Add composable Discord release message with editor flow
- Add CI-friendly progress logging with percentage and ETA to harvesters
- Add harvest summary dashboard to GitHub Actions job summary
- Add daily calibration harvest GitHub Actions workflow
- Add multi-pass support to elite harvester
- Add elite harvester for high-value rare calibration data

### Bug Fixes
- Fix deep query race condition and multi-monitor overlay positioning
- Fix mod_count to use total parsed mods and update shard (23,548 samples)
- Fix harvester process hang after passes complete
- Fix STARTUPINFO crash on Linux (GitHub Actions harvester)

### Improvements
- Update bundled calibration shard (3,810 -> 19,549 samples)

### Other
- Tune MOD_IDENTITY_WEIGHT to 0.15 and regenerate shard (27,564 samples)
- Strip trade API markup tags in harvester mod text
- Suppress low-value overlay by default and add toggle
- Tighten dashboard UI density without reducing font sizes
- Eliminate startup pop-in with opacity gate and ready signal
- Replace shard generator outlier removal with IQR in log-price space
- Bump elite harvester timeout to 180 min
- Merge pull request #3 from CouloirGG/dev
- Remove 7 POE1-only weapon categories from harvester
