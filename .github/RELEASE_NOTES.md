## LAMA v0.3.1 — Pricing fixed + Budget Planner

### Fixes
- **Item pricing restored.** poe2scout changed its API, which had silently broken all unique pricing — the compare shopping list and the "what top players use" prices were showing blank. Prices are live again (uniques, currency, gems).
- **Stale-league auto-heal.** If a league setting was carried over from a past season, LAMA now resets it to the current league so it doesn't load old-league prices.

### New
- **Budget Planner.** On a loaded character, hit **Budget** to see the unique upgrades top players run for your build — priced live, ranked by how common they are, with a budget cap and a running total of what fits. It's honest about rare-gear slots (it points you to trade rather than inventing a price).

### Also (since v0.3.0)
- DPS-percentile card and a passive-tree mini-map on the character view.
- 0.5.0 (Runes of Aldur) scoring/keystone tuning and the two new ascendancies.
