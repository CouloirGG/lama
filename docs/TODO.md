# LAMA — Active Backlog & Roadmap

**Last updated:** 2026-03-29
**Focus:** Build analysis sherpa — explain the WHY behind every recommendation

---

## Current Bugs

- [ ] **Gear tab: card disappears on click** — Clicking equipment on the left panel makes the detail card on the right disappear and drop to the bottom of the stack
- [ ] **CDN warnings on launch** — Tracking Prevention blocks Babel CDN storage access (cosmetic, does not affect functionality)

## Priority 1: "Why" Engine (Foundation)

The core gap: every tab shows data without explanation. This module generates plain-language context for any build data point.

- [ ] **Build context explanation module** — Backend service that takes (BuildArchetype, CharacterData) and produces human-readable explanations for:
  - Why a keystone matters for this specific build
  - What a mod tier means practically ("your fire res is 45%, T5 — endgame expects 75%+")
  - Why a popular item pairing works (mod synergy analysis)
  - What anti-synergies mean and how to fix them
- [ ] **Keystone interaction explanations** — For each keystone, explain how it plays off gear and other keystones in meta combinations
- [ ] **Mod replacement recommendations** — When a mod doesn't benefit the build, explain what specific prefix/suffix to replace it with and in what range

## Priority 2: Pulse Tab Rework

- [ ] **Build health KPIs with visual explanation** — Each KPI (survivability, DPS source, meta match) needs a visual breakdown of why/what it means
- [ ] **Survivability explanation** — Not just "Life" — explain effective HP, resist gaps, defense layers
- [ ] **DPS source breakdown** — Don't just say "Comet is king DPS" — explain what supports it, what scales it, what's bottlenecking it
- [ ] **Meta match % expansion** — Show what contributes to or detracts from the meta match score. What specific choices differ from meta and why those matter
- [ ] **Critical actions with reasoning** — "Consider Sanguimancy — 100% of players use this" must explain WHY they use it, what it enables, what changes in the build
- [ ] **Upgrade recommendations in plain language** — Replace "Avg Tier: t5.5" with human explanations of what to look for
- [ ] **Keystones vs meta with context** — If player has no keystones, explain what stats/power they're missing. For each popular keystone, explain how it interacts with the player's specific gear and skills

## Priority 3: Gear Tab Rework

- [ ] **Equipment layout matching community tools** — Grid layout matching existing POE2 tools (helm top center, body center, weapon left, etc.)
- [ ] **Item drill-down with tier explanations** — T0/T1/T2 labels must explain: what the tier means, what range it covers, what the next tier up would give, why it matters for this build
- [ ] **Mod benefit analysis** — For each mod on an item, show whether it benefits the build, is neutral, or is wasted. For wasted mods, recommend specific replacements with target ranges
- [ ] **Fix card disappearing bug** — Detail card on right must stay visible when clicking equipment items
- [ ] **Popular items with context** — For rares: explain what mod pairings people are stacking and what total value those give. For uniques: explain what roll values people target, whether people are corrupting for specific implicits (e.g., Soul Tether, Vertex), not just running base
- [ ] **Corruption awareness** — For popular uniques, show what corruption outcomes players target and why

## Priority 4: Market Tab Rework

- [ ] **Contextual item recommendations** — Replace raw listings like "Body Armour/Rare/T5 ChaosRes" with explanations of why that item matters for the loaded build
- [ ] **Synergy mapping (miro-style)** — Visual map of how items, mods, perks, and abilities play off each other. Explain the complexity of meta builds and why specific pairings set them apart. In-depth.
- [ ] **Budget planner → cost estimator** — Replace the empty currency box with actual cost estimates for recommended upgrades. Based on recommendations already made in Pulse/Gear tabs
- [ ] **Proactive build path recommendations** — Pull top-tier build paths for the current player's class/ascendancy. Sort by recommended priority (cost vs reward). Include per-recommendation context explaining what each upgrade path gives them

## Completed (This Session — 2026-03-29)

- [x] **Debug mode** — `LAMA-debug.bat` + `--debug` flag enables WebView2 DevTools and console output
- [x] **Fix accountName vs account field mismatch** — Saved character clicks were passing `undefined` as account, causing 422 errors and blank screens
- [x] **Fix league dropdown crash** — `/api/leagues` returns `{value, label}` objects but dropdown rendered them as strings → React error #31
- [x] **Fix poe.ninja profile API** — Added league segment to profile URL path (was returning 404 for all profile lookups)
- [x] **Smart character search** — Single input field that accepts poe.ninja URLs, account/character pairs, or partial names with fuzzy matching against saved characters
- [x] **URL parser for both poe.ninja formats** — Handles URLs with and without league segment in path
- [x] **Window controls** — Added minimize, maximize, close buttons to frameless title bar with drag region
- [x] **Disabled calibration harvest workflow** — GitHub Actions cron job was failing daily since pricing system removal
- [x] **Fix API error handling** — FastAPI 422 validation errors now extract human-readable messages instead of crashing React
- [x] **Validation error logging** — RequestValidationError handler logs full request body for debugging

---

## Archived (Pre-Pivot — Pricing Era)

The following features were part of the pricing system that was killed in March 2026. Keeping for historical reference only — none of these are on the roadmap.

<details>
<summary>Archived pricing-era backlog</summary>

- Calibration pipeline improvements (temporal decay, adaptive k, fuzzy modset)
- Trade API deep query with DPS/defense filters
- Disappearance tracker for listing sell-through rates
- Cloud push notifications for trade watchlist
- Harvest scheduler for continuous data collection
- GBM/Ridge regression price prediction models
- Calibration shard generation and distribution
- Trade action buttons (whisper, invite, hideout, trade, kick)
- Currency icons in overlay
- Chanceable base detection

</details>
