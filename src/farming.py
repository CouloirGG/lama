"""Grounded PoE2 currency-farming guidance, personalized to the player's stage.

The strategy CONTENT here is curated from the current community meta (0.5.0
"Runes of Aldur" / Return of the Ancients — Maxroll + community farming guides,
researched 2026-06) so it is grounded, not LLM-invented. Only the SELECTION
(which strategies fit this character) is data-driven.

REFRESH each patch: the staged structure is evergreen, but specific numbers,
node names, and which league mechanics are strongest drift between leagues.
Numbers are community estimates — present them as guidance, not guarantees.
"""

# Progression stages, easiest -> hardest to reach.
STAGES = ("budget", "mid", "geared")

# Each strategy is tagged with the stages it suits.
STRATEGIES = [
    {
        "id": "expedition", "name": "Expedition", "stages": ["budget", "mid"],
        "tag": "Best budget starter",
        "barrier": "No gear or atlas investment — runs on white maps.",
        "income": "~15 div from the quest line, then ~40 div/day from regular mapping (community estimate).",
        "how": "Scout each area before committing; only detonate Grand Expeditions with worthwhile Remnant "
               "rewards (Divine, Runes, Uniques), and chase 7-8 rune-slot encounters. Pick Jado as your Atlas "
               "Master and grab the Verisium reroll node — it compounds all league.",
    },
    {
        "id": "belt_craft", "name": "Triple-resistance belt crafting", "stages": ["budget", "mid"],
        "tag": "No mapping, no luck needed",
        "barrier": "Off-atlas — pure desecration crafting at the Well of Souls. Zero mapping.",
        "income": "Triple-res belts sell for ~15-20 div each; ~1-3 div per craft attempt. Demand is highest "
                  "early-league while everyone is gearing.",
        "how": "Deterministic-ish crafting you control, so it's the cleanest answer to 'I never get good drops'. "
               "Roll resistance belts and list them; reinvest the profit.",
    },
    {
        "id": "strongbox", "name": "Strongboxes", "stages": ["budget", "mid"],
        "tag": "Passive while you map",
        "barrier": "Minimal atlas investment; Research strongboxes spawn 1-2 times per map naturally.",
        "income": "Drop Exalted Orbs and the occasional Divine on open — steady currency on top of whatever else you run.",
        "how": "Take the Strongbox / Research nodes on the atlas tree and just open them as you clear.",
    },
    {
        "id": "heist", "name": "Heists", "stages": ["budget"],
        "tag": "Quick & reliable",
        "barrier": "Off the main map grind — short, self-contained runs.",
        "income": "A reliable trickle of currency with fast completion; good when you want low-commitment sessions.",
        "how": "Run heists between mapping sessions for consistent, low-risk income.",
    },
    {
        "id": "breach", "name": "Breach", "stages": ["mid", "geared"],
        "tag": "Scales with clear speed",
        "barrier": "Wants decent AoE + clear speed to pop the hands fast before they close.",
        "income": "Wombgifts sell directly (no crafting), and splinters bundle into Breachstones (sell in sets of 100).",
        "how": "Spec Breach on the atlas, trigger the hands every map, and push through the rifts for the density.",
    },
    {
        "id": "ritual", "name": "Ritual", "stages": ["mid", "geared"],
        "tag": "Consistent, low-dilution",
        "barrier": "Needs to survive the ritual waves; reward pool is uniques + Omens (doesn't dilute itself).",
        "income": "Steady tribute -> uniques and Omens; profitable even without a full atlas setup.",
        "how": "Grab the Ritual cluster atlas nodes (tribute generation + reward quality) and run rituals on every map.",
    },
    {
        "id": "abyss", "name": "Abyss", "stages": ["geared"],
        "tag": "Top endgame engine",
        "barrier": "Needs to tank six-mod tier-15 maps comfortably.",
        "income": "A geared character can clear ~400-500 div over a couple of days (community estimate) — best consistent engine.",
        "how": "Spec all tablet-effect + rare-monster / pack-size atlas nodes. Rogue Exiles -> Heart of the Well "
               "jewels are the main income; pull Lichborn enemies into the open before killing for Omen drops.",
    },
]

UNIVERSAL_TIPS = [
    "Push map tier as high as you can survive — higher area level means more and better loot (aim for T15+ once tanky).",
    "Run ~100% Item Rarity on gear: it makes drops more valuable, but there are steep diminishing returns past ~100%.",
    "Spec the atlas toward ONE mechanic and run a tight 2-3 map rotation — focus beats a random spread.",
    "Forest biome gives the most rare monsters (up to +65%) — pick it when farming rare-monster density.",
    "Always check poe.ninja / poe2scout before you sell — league prices swing constantly.",
]

EARLY_TIP = ("Just leveling? Pick up every Transmutation / Augmentation / Alteration orb (free crafting fuel), "
             "alch your Waystones, and hoard Exalted Orbs — the real currency hunt starts once you're mapping.")


def _to_int(v):
    try:
        return int(str(v).replace(",", "").split()[0])
    except Exception:
        return None


def classify_stage(scorecard: dict, level) -> dict:
    """Classify the player's progression stage from survivability + level + DPS.

    budget  = just hitting maps / can't yet tank juiced T15
    mid     = stable mapper climbing tiers
    geared  = tanks six-mod T15, ready for the top engines (Abyss)
    """
    sc = scorecard or {}
    ehp = _to_int(sc.get("ehp"))
    pct = _to_int(sc.get("dpsPercentile")) or 0
    lvl = _to_int(level) or 0
    status = sc.get("ehpStatus", "")

    if status == "critical" or (ehp is not None and ehp < 15000) or lvl < 80:
        stage = "budget"
        label = "Just hitting maps / budget"
    elif lvl >= 90 and ehp is not None and ehp >= 30000 and pct >= 55 and status != "critical":
        stage = "geared"
        label = "Geared mapper"
    else:
        stage = "mid"
        label = "Climbing the tiers"

    bits = [f"Level {lvl}" if lvl else None,
            f"{ehp:,} EHP" if ehp is not None else None,
            f"{pct}th-pct DPS" if pct else None]
    why = ", ".join(b for b in bits if b)
    return {"stage": stage, "label": label, "why": why}


def funding_plan(scorecard: dict, level, chase_items=None) -> dict:
    """Personalized, grounded 'how to fund it' plan for this character."""
    st = classify_stage(scorecard, level)
    stage = st["stage"]
    strategies = [s for s in STRATEGIES if stage in s["stages"]]
    # Budget starters first within the stage.
    order = {"budget": 0, "mid": 1, "geared": 2}
    strategies = sorted(strategies, key=lambda s: order.get(s["stages"][0], 9))

    chase = chase_items or []
    total = round(sum((c.get("div") or 0) for c in chase), 1)
    gap = None
    if chase:
        gap = {
            "items": [{"name": c.get("name"), "div": c.get("div"), "cost": c.get("cost")} for c in chase],
            "totalDiv": total if total > 0 else None,
        }
    return {
        "stage": stage,
        "stageLabel": st["label"],
        "stageWhy": st["why"],
        "strategies": strategies,
        "universal": UNIVERSAL_TIPS,
        "earlyTip": EARLY_TIP if stage == "budget" else None,
        "gap": gap,
    }
