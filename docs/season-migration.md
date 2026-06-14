# Season Migration Runbook

How to migrate LAMA when Path of Exile 2 rolls over to a new challenge league
(e.g. patch `0.4.0` → `0.5.0`). Follow top to bottom. Most of it is one config
block + a re-harvest; the rest is game-knowledge upkeep.

> **Design principle:** the league string lives in **one place** — `src/config.py`
> (`DEFAULT_LEAGUE` / `DEFAULT_LEAGUE_HC` / `LEAGUE_OPTIONS`). Everything else
> (server fallbacks, price cache, diagnostics, the selector) reads from those
> constants. `builds_client` *auto-detects* the live league from poe.ninja, so
> character lookups keep working even before you change config — but the
> fallbacks, harvested data, and any new ascendancies still need this runbook.

---

## 0. Identify the new league

1. **Confirm it's live and get the exact name.** Run the auto-detect probe:
   ```bash
   python -c "import sys; sys.path.insert(0,'src'); from builds_client import BuildsClient; c=BuildsClient(); print('ok:', c._fetch_snapshot_info()); print('league:', c._snapshot_name)"
   ```
   `_snapshot_name` is poe.ninja's slug form (e.g. `runes-of-aldur`). The
   snapshot version string is dated (`0429-20260614-…`) — check the date is recent.
2. **Get the human/API name** (with spaces/caps) from poe.ninja's economy/builds
   dropdown URLs and poe2scout. Convention in this repo:
   - SC value: `"<League Name>"` (e.g. `Runes of Aldur`)
   - HC value: `"Hardcore <League Name>"` (e.g. `Hardcore Runes of Aldur`)
3. **Sanity-check against an official source** (pathofexile.com news / forum) so you
   don't confuse the *content patch* name with the *challenge league* name — they
   differ (e.g. patch "Return of the Ancients" 0.5.0 vs league "Runes of Aldur").

## 1. Update the single source of truth — `src/config.py`

Edit the **Current League** block only:
```python
DEFAULT_LEAGUE    = "Runes of Aldur"
DEFAULT_LEAGUE_HC = "Hardcore Runes of Aldur"
# LEAGUE_OPTIONS derives from the two above — no edit needed
```
This automatically flows to: `server.py` (DEFAULT_SETTINGS + every
`settings.get("league", DEFAULT_LEAGUE)` fallback + `/api/leagues` fallback list),
`price_cache.py`, `diagnose.py`, `games/poe2.py`. Nothing else in Python should
contain a literal league name — verify:
```bash
grep -rn "Fate of the Vaal\|<old league name>" src/ scripts/   # expect: none
```

## 2. Update the overlay launcher — `START.bat`

`START.bat` writes `~/.poe2-price-overlay/league.txt` from a hardcoded menu (it's a
`.bat`, can't read Python). Update the 4 menu lines + the 3 `echo …>league.txt`
defaults to the new league name. (Used only by the CLI overlay path.)

## 3. Re-harvest the meta shard

The shard (`meta_shard.json.gz`) holds league-current scaling / popular uniques /
support gems. The committed one is stamped with the league it was built for.

- **Preferred (once `dev`→`main` is released so the workflow is on the default
  branch):** trigger the GitHub Action — it auto-detects the league, no `--league`:
  ```bash
  gh workflow run meta-harvest.yml -R CouloirGG/lama
  ```
  It validates (`version>=2`, `>=6` classes) and republishes the `meta-latest`
  release that clients pull via `meta_loader`.
- **Local (current default — the Action isn't on `main` yet):**
  ```bash
  python scripts/meta_harvester.py --output meta_shard.new.json.gz
  # verify league + freshness, then replace the committed shard:
  python -c "import gzip,json; d=json.load(gzip.open('meta_shard.new.json.gz')); print(d['league'], d['generated_utc'], len(d.get('class_scaling',{})),'classes')"
  mv meta_shard.new.json.gz meta_shard.json.gz
  ```
  Commit the refreshed `meta_shard.json.gz` on `dev`.
- **Rate limiting (429):** poe.ninja throttles aggressively, especially early in a
  league and if you've run the harvest several times in a short window (your IP
  gets throttled). `builds_client._get_with_retry()` backs off and retries on 429/503
  (honoring `Retry-After`), so a single run completes even while throttled — it just
  runs slower. Symptom of a bad run: a class harvests to **0 builds** (it lost the
  429 race) and drops out of `class_scaling`, leaving 7/8 classes. Validate the
  class list after every harvest; re-run if a class you expect is missing. The
  GitHub Action runs from GitHub's IP at off-peak (06:00 UTC) and is less throttled
  than repeated local runs.
- **Early-league caveat:** ladders are also genuinely sparse in the first days. If
  validation fails or classes < 6, wait a day or two and re-run.

  Always confirm the class list, e.g.:
  ```bash
  python -c "import gzip,json; d=json.load(gzip.open('meta_shard.new.json.gz')); print(sorted(d['class_scaling']))"
  ```

## 4. New / reworked ascendancies (game-structural — only when GGG adds them)

A new league often adds an ascendancy per a couple of base classes. These are
**hardcoded** and must be added by hand or lookups for those characters break.

- **`src/builds_client.py` → `ASCENDANCY_MAP`** (authoritative; keyed by the name
  poe.ninja returns). Add `"<Ascendancy>": "<BaseClass>"` and bump the `(n/3)` comment.
- **`src/guide_scraper.py` → `_MAXROLL_ASC_MAP`** (keyed by Maxroll's `"{Class}{N}"`
  codes). Add the new ones; **verify the numbering** against a real Maxroll planner
  URL for that ascendancy — Maxroll's index order isn't guaranteed.
- **`scripts/meta_harvester.py` → `CLASS_SKILLS`** add a few probe `(ascendancy, skill)`
  pairs so the harvester samples the new ascendancy. Probe skills are best-effort
  until the meta is observable; wrong pairs just yield empty samples (harmless).

## 5. Game-knowledge drift (review after the meta settles, ~1–2 weeks in)

These are meta-dependent and drift every league. The runtime shard overrides some
at load, but the static defaults in **`src/game_knowledge.py`** are the fallback:

- `CLASS_SCALING`, `POPULAR_UNIQUES`, `TOP_SUPPORT_GEMS` — refreshed by the shard.
- `KEYSTONE_COMBOS` / `anti_synergies` — re-validate against patch notes. Big
  mechanic changes (leech reworks, defense-layer changes, keystone reworks) can
  make old scoring/anti-synergy rules stale. Check the patch notes for: leech,
  energy-shield/recharge, new defensive layers, armour/evasion math, weapon ranges.

## 6. Clear stale per-user caches (on the dev machine / document for users)

These live in `~/.poe2-price-overlay/` and are keyed to the old league:
```bash
# Windows: %USERPROFILE%\.poe2-price-overlay\
rm -f league.txt                              # or overwrite with new league
rm -f meta_shard.json.gz                      # re-pulled from meta-latest
rm -f cache/prices_<old_league_slug>.json     # new file auto-created
rm -f cache/calibration_shard_<old>*.jsonl    # dead pricing system; safe to delete
```
New-league cache files are created automatically on next run; the persisted
`settings["league"]` overrides `DEFAULT_LEAGUE`, so an existing user may need to
re-pick the league in the selector (or delete `league.txt`).

## 7. Verify

```bash
python -m py_compile src/config.py src/server.py src/builds_client.py \
  src/guide_scraper.py src/diagnose.py src/price_cache.py scripts/meta_harvester.py
python src/diagnose.py                         # confirms league fetch end-to-end
```
Then launch `LAMA-debug.bat`, look up a real current-league character (ideally one
on a **new** ascendancy), and confirm the dashboard populates and the log shows the
new league. Watch the tree-analysis path specifically (historical blank-screen source).

---

## Migration log

| Date | From → To | Patch | New ascendancies | Notes |
|------|-----------|-------|------------------|-------|
| 2026-06-14 | Fate of the Vaal → **Runes of Aldur** | 0.4.0 → 0.5.0 (Return of the Ancients) | Martial Artist (Monk), Spirit Walker (Huntress) | First migration using this runbook. Centralized league string into `config.py`. Fixed pre-existing `guide_scraper` bug (Huntress2 was "Beastmaster" → Ritualist). Auto-detect resolved `runes-of-aldur` live. Meta-relevant 0.5.0 changes to review in game_knowledge: leech overhaul (no instant leech, single-instance, 40k cap), ES-recharge nerfs, new Runic Ward defensive layer, armour/evasion/deflection reformulas. |
