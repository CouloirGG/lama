# LAMA (Live Auction Market Assessor) — Project Instructions

## What LAMA Is
LAMA is a **build analysis sherpa** for Path of Exile 2. It looks up a player's character via poe.ninja, analyzes their gear, skills, and keystones, and explains — in plain language — what's working, what's not, and what to do next. It runs as a frameless desktop app (pywebview + FastAPI + React).

**Core principle:** Every data point must explain **WHY**. Don't just show "T5 Chaos Res" — explain what that means for the player's build, what range they should target, and how it interacts with their other gear. LAMA is a guide, not a spreadsheet.

## Git Identity & Workflow
- **Git author**: All commits MUST use `calschuss <couloirgg@gmail.com>`. If git config doesn't match, run: `git config --local user.name "calschuss" && git config --local user.email "couloirgg@gmail.com"`
- **Never add `Co-Authored-By` trailers to commits.** No AI attribution in commit messages.
- **Always work on the `dev` branch.** All commits go to `dev`.
- Never commit directly to `main`. `main` is the stable release branch for players.
- When ready to release, merge `dev → main` via PR and tag with a version.

## Architecture Overview
```
app.py (entry point)
  ├─ FastAPI server on port 8450 (daemon thread)
  │   └─ server.py — REST API + WebSocket + serves dashboard
  ├─ System tray icon (daemon thread)
  └─ pywebview frameless window → http://127.0.0.1:8450/dashboard
      └─ resources/dashboard.html (React18 + Tailwind, in-browser Babel)
```

**Key modules:**
- `builds_client.py` — poe.ninja API client: character lookup, build classification, popular items/skills/keystones, anti-synergy detection
- `mod_database.py` — Item scoring engine: RePoE tier data, build-aware multipliers, progression-aware multipliers, dual grading (universal + build-specific)
- `server.py` — Backend: character lookup (smart + direct), build insights, gap analysis, settings persistence
- `main.py` — Overlay process: item detection, clipboard parsing, score display

**Data sources (safe under GGG TOS):**
- poe.ninja (builds, profiles, economy) — primary
- poe2scout (leagues, economy) — supplementary
- RePoE (mod tier data, base items) — static game data

**Forbidden:** No POESESSID, no reverse-engineered GGG APIs, no trade2 API. OAuth app was rejected.

## Current State (March 2026)
- Pivoted from pricing tool to build analysis tool (pricing killed due to unsolvable data quality)
- Dashboard has 4 tabs: Pulse (build health), Gear (equipment analysis), Market (economy), Guide (build guides)
- Character lookup via poe.ninja (ladder + profile APIs), smart search supports URLs/names/partial matches
- Build classification: damage type, defense type, crit/CoC detection, anti-synergy rules
- Build-aware item scoring with progression multipliers (leveling/cruel/endgame)
- Frameless window with custom title bar, system tray, debug mode (`--debug` flag)

## Testing
```bash
python -m pytest tests/ -v
```
Test fixtures use real clipboard captures in `tests/fixtures/`.

## Launching
- **Normal:** `LAMA.bat` (uses exe if built, falls back to pythonw)
- **Debug:** `LAMA-debug.bat` (console output + WebView2 DevTools via right-click → Inspect)
- **CLI overlay only:** `python src/main.py --league "Runes of Aldur" --debug`
