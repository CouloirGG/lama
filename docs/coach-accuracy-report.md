# LAMA AI Coach — Accuracy Report

**Test:** Top-5-DPS builds for each of the 8 base classes = **40 real ladder builds** (poe.ninja, Runes of Aldur).
**Coach model:** `phi4:latest` (local, via Ollama) — grounded mode (LAMA ranks priorities + supplies facts; the LLM only narrates them in order).
**Independent judge:** `qwen2.5:14b` (local), plus a deterministic programmatic scan.
**Result files:** `coach_results.json` (full transcripts + facts + scores), `builds40.json` (the test set).

---

## Headline

| Metric | Score | Notes |
|---|---|---|
| Builds coached without error | **40 / 40** | 0 crashes, 0 timeouts, 0 lookup failures |
| Led with LAMA's #1 priority | **37 / 37** | every build that *had* a priority; remaining 3 were fully healthy (nothing to prioritize → correctly pivoted to DPS) |
| Grounded (no invented items/mechanics) | **39 / 40** | the one miss is a typo, not a fabrication (see below) |
| Actionable | **40 / 40** | judge agreed unanimously |

**The grounded-coach design held.** Across 40 builds, phi4 did not invent a single item, unique, gem, keystone, or PoE1 mechanic. Everything it named traces back to the facts LAMA computed and handed it.

---

## Per-class breakdown

| Class | n | Led with #1 / had-priority | Healthy (no priority) | Fabrications |
|---|---|---|---|---|
| Warrior | 5 | 4/4 | 1 | 0 |
| Witch | 5 | 5/5 | 0 | 0 |
| Ranger | 5 | 3/3 | 2 | 1 (typo) |
| Mercenary | 5 | 5/5 | 0 | 0 |
| Monk | 5 | 5/5 | 0 | 0 |
| Huntress | 5 | 5/5 | 0 | 0 |
| Sorceress | 5 | 5/5 | 0 | 0 |
| Druid | 5 | 5/5 | 0 | 0 |

Priority #1 distribution: **resists 32, survival 5, healthy/no-priority 3.** (Chaos res uncapped dominates the top of the ladder — consistent with the guild findings.)

---

## The one real defect

**Ranger / stillAengus** — the coach wrote *"Kalandri's Touch"* instead of **"Kalandra's Touch"**. The item is real and *is* in the facts (spelled correctly); phi4 transcribed it with a one-letter error. Low severity, but it's the kind of thing that breaks a copy-paste search. Worth a post-process pass that snaps any named item back to the exact spelling from the facts.

## The judge disagreed — and the judge was wrong

The qwen2.5:14b judge scored grounding at only **27/40**, flagging 13 builds. I pulled every flagged transcript and checked each named item against the actual facts string LAMA fed the model:

- **12 of 13 were false flags.** The judge doesn't recognize PoE2's support-gem names (Cast on Critical, Rakiata's Flow, Boundless Energy II, Uhtred's Augury, Garukhan's Resolve, Eldritch Battery…) and assumed they were invented or Path of Exile 1 leftovers. Every one of them is present in LAMA's facts — verified by string match. A few other flags were the judge conflating "didn't state the exact chaos-cap number" with "ungrounded," which is a detail complaint, not a grounding failure.
- **1 of 13 was the real typo** above.

Takeaway: **qwen2.5:14b is not a usable grader for PoE2 content** — its game knowledge is PoE1-era, so it penalizes correct PoE2 terminology. The deterministic facts-membership scan (1/40 flagged) is the trustworthy signal here.

---

## The finding that matters more than the coach

The coach is grounded — but it's only as good as what LAMA feeds it, and LAMA is occasionally feeding it junk. Recurring across the facts:

```
MISSING FOR DPS: Support: Cast on Critical (+8% impact, used by 0% of top builds);
                 Support: Rakiata's Flow (+8% impact, used by 0% of top builds);
                 Support: Boundless Energy II (+8% impact, used by 0% of top builds)
```

**A "missing" DPS upgrade that 0% of top builds use is noise.** The coach faithfully relays it ("although it's not commonly seen in top builds, it could give you an edge") — which is honest, but it's recommending nothing. This is an upstream synergy-map issue (builds_client / mod_database), **not** a coach issue.

**Recommended fix:** filter synergy recommendations by adoption — drop anything under ~10–15% adoptionPct unless it's the only suggestion for that category. That alone would sharply raise the signal of every coach response.

Secondary observation (scoring philosophy, not accuracy): LAMA flags `Chaos 60%` as "not capped → priority #1" even on a 1.5M-DPS, level-98, 51k-EHP build with all elemental res capped. Chaos res is hard to cap in PoE2 and 60% is genuinely good; treating chaos < 75% as a critical one-shot priority may be too aggressive. Worth revisiting the chaos-res threshold separately.

---

## Verdict

The grounded mini-coach works and is safe to ship: faithful prioritization (40/40), zero fabrications (1 typo), 100% completion on real ladder data with a local 7B-class model. The architecture — **LAMA decides, the LLM narrates** — is the reason it doesn't hallucinate, and it's directly portable to the other tools.

Before shipping to players: (1) snap named items to exact facts spelling, (2) suppress 0%-adoption synergy recommendations upstream, (3) streaming responses for latency. None block the coach itself.
