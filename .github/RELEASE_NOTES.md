## LAMA v0.3.2 — Character lookup fixed

### Fixes
- **Looking up your own (non-ladder) character works again.** poe.ninja changed its profile API response, which silently broke every non-ladder lookup — they returned "Character not found" even for valid public characters. Fixed; off-ladder characters resolve again.

This is a quick follow-up to v0.3.1 (which restored item pricing and added the Budget Planner).
