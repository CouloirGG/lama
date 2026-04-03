# LAMA UI Mockups (Stitch)

5 mockup HTML files from Stitch for the full LAMA UI redesign.

## Design Language
- Dark gothic POE2 aesthetic (#0e0e0e to #2a2a2a surfaces)
- Amber primary (#FFB77B), gold secondary (#E9C176), ice tertiary (#B5EAFF)
- Sharp corners (border-radius: 0)
- Noto Serif for headlines/numbers, Space Grotesk for body/labels
- Material Symbols Outlined icons
- Uppercase tracking-widest monospace labels
- Border-l-2/4 severity accents (error red, secondary gold, outline grey)

## Views
1. `prompt1-overview.html` — Build Overview (main dashboard with 3-col layout)
2. `prompt2-gear-detail.html` — Gear Inspection slide-in panel
3. `prompt3-perk-swaps.html` — Passive tree swap recommendations
4. `prompt4-market.html` — Upgrade Priority / Market tab
5. `prompt5-comparison.html` — Build Comparison diff view

## Implementation Priority
1. Adopt the color palette and Tailwind config from prompt1 (replaces current theme)
2. Implement the 3-column overview layout with sidebar navigation
3. Implement gear detail slide-in panel (prompt2)
4. Implement perk swap cards (prompt3)
5. Implement market upgrade priority list (prompt4)
6. Implement build comparison diff (prompt5)

## Key Components to Extract
- Stat cards with progress bars and population context
- Synergy map with SVG connection lines
- Gear efficiency list with green/grey/red dot indicators
- Perk swap cards with gain/lose boxes
- Upgrade priority cards with severity borders and adoption rates
- Build diff cards with category icons and stat deltas
- Sidebar navigation with active state
