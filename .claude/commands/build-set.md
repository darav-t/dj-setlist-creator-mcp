---
description: Build a DJ set from a natural language prompt using your Rekordbox library
argument-hint: "e.g. 60min sunset progressive house, afters techno 2hrs, festival main stage with vocals"
---

Build a DJ set from this prompt: **$ARGUMENTS**

Follow these steps exactly:

## Step 1 — Parse and build the set

Call `mcp__mcp-dj__build_set_from_prompt` with:
- `prompt`: the full user prompt (everything in $ARGUMENTS)
- `duration_minutes`: extract from the prompt if mentioned (e.g. "2 hour" → 120, "90 min" → 90), otherwise use 60

## Step 2 — Present the setlist

Format the result as a clean DJ-readable setlist. Use this exact layout:

---

### 🎧 [set name from result]

**Intent detected**
- Tags: [my_tags_detected joined with " · ", or "none — genre/BPM fallback used" if empty]
- Genre: [genre or "mixed"] · BPM: [bpm_range] · Arc: [energy_profile] · Pool: [candidate_pool] tracks
- Reasoning: *[reasoning string]*

---

**Tracklist** ([track_count] tracks · [duration_minutes] min · avg [avg_bpm] BPM · harmonic score [harmonic_score]/10)

| # | Artist – Title | BPM | Key | Energy | Duration | Notes |
|---|---------------|-----|-----|--------|----------|-------|

Fill each row from the `tracks` array. For the Notes column use `key_relation` and `transition_score` if present, plus any Essentia mood/genre hints from `dominant_mood` / `top_genre_discogs`.

Below the table show:

**Energy arc:** [energy_arc — list of energy values across the set, formatted as a mini sparkline using ▁▂▃▄▅▆▇█ characters mapping energy 1-10 to the 8 bar heights]

**Genre breakdown:** [genre_distribution — top 3 genres as "Genre (N tracks)"]

---

## Step 3 — Offer export and follow-up actions

After presenting the set, ask:

> **What next?**
> - **Export to Rekordbox** — type `/export-set [playlist name]` or say "export as [name]"
> - **Swap a track** — say "replace track #N" or "swap [title] for something more [vibe]"
> - **Adjust** — say "make it darker", "add more vocals", "shift BPM up 5"
> - **Analyze energy flow** — say "check the energy flow" to run a harmonic analysis
> - **Another set** — say "build another one" for a fresh variation

If the user says anything matching export/save/rekordbox, immediately call `mcp__mcp-dj__export_setlist_to_rekordbox` with the `setlist_id` from the result and the playlist name they provided (default: the set name).

## Rules

- Always show the full tracklist table — never truncate it
- If `my_tags_detected` is empty, note that the set used genre/BPM filtering from the full library and suggest adding MyTags in Rekordbox for better curation
- If `harmonic_score` is below 6.0, add a note: "⚠ Harmonic score is low — consider using `/build-set` with a more specific genre or BPM range for tighter key compatibility"
- Format BPM values as integers (no decimals)
- Format energy as filled circles: ●●●●●○○○○○ (filled = energy level out of 10)
- The energy sparkline maps energy 1=▁ 2=▂ 3=▃ 4=▄ 5=▅ 6=▆ 7=▇ 8-10=█
