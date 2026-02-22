# LAMA (Live Auction Market Assessor) — Project Instructions

## Git Identity & Workflow
- **Git author**: All commits MUST use `calschuss <stuart.white28@protonmail.com>`. Never use CarbonSMASH, Couloir, or any other identity. If git config doesn't match, run: `git config --local user.name "calschuss" && git config --local user.email "stuart.white28@protonmail.com"`
- **Never add `Co-Authored-By` trailers to commits.** No AI attribution in commit messages.
- **Always work on the `dev` branch.** All commits go to `dev`.
- Never commit directly to `main`. `main` is the stable release branch for players.
- When ready to release, merge `dev → main` via PR and tag with a version.
