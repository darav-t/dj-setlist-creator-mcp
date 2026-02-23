# Building Sets: How the AI Creates Your Setlist

This document explains what happens when you ask for a DJ set — from your natural language prompt to a fully ordered tracklist. It's written for DJs, not engineers, so the focus is on *why* the system makes the choices it does and how you can guide it.

---

## The Shortest Possible Explanation

You describe what you want. The system:

1. Figures out which of your My Tags match the vibe you described
2. Finds all tracks with those tags
3. Orders them using harmonic mixing (Camelot wheel) and a planned energy arc
4. Returns a setlist with transition notes

The key insight: **the system is not guessing your taste from scratch — it's using the curatorial work you've already done in Rekordbox.** Your My Tags are the starting point for every set.

---

## How to Ask for a Set

The natural language interface is intentionally flexible. These are all valid prompts:

```
"60-minute festival set"
"dark underground techno, build to peak over 90 minutes"
"sunset rooftop, melodic and emotional, 2 hours"
"wedding set — happy famous tracks that everyone knows"
"afters vibe, hard and dark, no time limit"
"warm-up set, start at 120 BPM, housy, gradually build energy"
```

You can mix and match:
- **Vibe descriptors:** dark, happy, melodic, hard, underground, emotional
- **Situations:** festival, afters, wedding, sunset, warm-up, peak time, closing
- **Genre:** tech house, techno, melodic house, deep house, progressive house
- **Duration:** any number + minutes/hours ("90 min", "2 hours", "45 minutes")
- **BPM:** "starting at 124", "around 128 BPM", "120-130 BPM"
- **Specific starting track:** "starting with [track title]"

---

## Step 1: Intent Parsing — What Do You Actually Want?

When you type a prompt, the system reads it and extracts:

### My Tags to Target

The system compares your prompt against every tag that actually exists in your library. This is not a fixed keyword list — it reads your actual tags.

If you say "festival set," it finds the `Festival` tag in your library. If you say "something for afters," it finds `Afters`. If you say "high energy dance tracks," it matches `High Energy` and `Dance`.

The matching is fuzzy: it looks for the full tag name as a substring first, then tries matching all significant words. So "tech house banger" matches your `Tech House Banger` tag even though you didn't type the exact tag name.

Tags are ranked by how many tracks you have with them. The system picks the most populated matching tags — this prevents it from filtering to 3 tracks because it matched an obscure tag with very few entries.

### Energy Arc

The system maps your situation to one of five energy arc profiles:

| Profile | Shape | When It Applies |
|---------|-------|----------------|
| **journey** | Warm → Build → Peak → Cool-down | "festival set", "full set" |
| **build** | Continuous rise from low to high | "build", "warm-up into peak" |
| **peak** | High energy throughout | "peak time", "main stage", "bangers" |
| **chill** | Low energy, ambient | "lounge", "background", "chill" |
| **wave** | Multiple peaks with valleys | "wave", experimental, long sets |

Venue/time keywords also influence this:
- "sunrise" or "morning" → chill profile
- "closing" → wave or journey (peak then cool-down)
- "opening" or "warm-up" → build profile
- "wedding" → journey (but with happy/famous track filtering)

### BPM Range

If you specify a BPM or genre, the system sets a BPM window:

- Explicit: "124-128 BPM" → restricts candidates to that range
- Genre: if you say "tech house," it reads the actual BPM distribution of your tech house tracks (not a hardcoded number) and uses the 25th–75th percentile as the target range
- Vibe: "après ski" → 122-128 BPM (this one is hardcoded because it's a venue archetype, not your library data)

---

## Step 2: Candidate Selection — Which Tracks Are Eligible?

After parsing intent, the system builds a pool of candidate tracks by:

1. Filtering to tracks with matching My Tags (or all tracks if no tags matched)
2. Applying BPM range filter (if specified)
3. Applying genre filter (if specified)

**Why filter by My Tags first?**

Because you've already told the system what these tracks are for. A track tagged `Festival` has been curated by you for festival situations. A track tagged `Afters` is appropriate for afters. Using your tags as a primary filter means the system respects your judgment instead of trying to infer it from audio features alone.

If the tag filter returns fewer than 20 tracks, the system widens the search (relaxes BPM tolerance, adds neighboring tags) to ensure there's enough variety for a full set.

---

## Step 3: Scoring — Which Track Comes Next?

This is where the actual set-building happens. The system uses a **greedy algorithm**: it picks tracks one at a time, choosing the best available track for each position in the set.

For each candidate track, it calculates a combined score from six factors:

### Factor 1: Harmonic Score (Camelot Wheel)

The Camelot wheel is a circular diagram of musical keys organized by harmonic compatibility. Adjacent keys on the wheel blend naturally; distant keys clash.

```
        12A - 12B
      1A        1B
    2A            2B
   3A              3B
    4A            4B
      5A        5B
        6A - 6B
        7A - 7B
      8A        8B
    9A            9B
   10A             10B
    11A           11B
      12A      12B
```

Compatibility scores:

| Transition | Score | Sounds Like |
|-----------|-------|-------------|
| Same key (e.g., 8A → 8A) | 1.0 | Perfectly seamless |
| Adjacent key (8A → 7A or 9A) | 0.9 | Natural, almost unnoticeable |
| Ring switch (8A → 8B) | 0.85 | Subtle brightening/darkening |
| Energy boost (+7 positions: 8A → 3A) | 0.7 | Classic DJ trick, adds lift |
| Diagonal (+1 position and ring switch) | 0.6 | Works but slight tension |
| Incompatible | 0.1 | Potential clash |

The system prefers high-compatibility transitions. It won't refuse an incompatible transition — sometimes you want the drama — but it takes a strong advantage in other factors to override the harmonic score.

### Factor 2: Energy Score

The system follows the planned energy arc (see Step 1). At each position in the set, there's a **target energy level**. The score for each candidate track is higher if its energy level is close to the target.

There are two sub-penalties:
- **Rate-of-change penalty:** Penalizes large energy jumps. Going from energy 4 to energy 9 in one step sounds abrupt even if both tracks are harmonically compatible.
- **Plateau penalty:** Penalizes playing three or more tracks at the same energy level in a row. Even a perfect energy level gets boring if it never moves.

Together these create sets that flow naturally through the energy arc rather than jumping around or flatlining.

### Factor 3: BPM Score

Large BPM differences are penalized. The system prefers staying within ±3 BPM of the previous track. It allows larger jumps as a set progresses (the engine gradually eases the BPM constraint as the set advances past the midpoint).

### Factor 4: Genre Coherence

If the genre of a candidate track is significantly different from the last 3 tracks, it gets a small penalty. This prevents genre whiplash — dropping a full techno track in the middle of a deep house set because it happened to have the right key.

The penalty is intentionally mild. Genre coherence matters less than harmonic and energy fit. If a track has perfect harmonic compatibility and is exactly the right energy, the genre penalty alone won't stop it from being picked.

### Factor 5: Artist Diversity

The system penalizes playing the same artist twice within a short window (typically 5 tracks). Even if you have 10 tracks by the same artist in your library that all fit the set, playing five in a row is rarely the right call.

This penalty is proportional to recency: playing the same artist 2 tracks ago is penalized more than 5 tracks ago.

### Factor 6: Plateau Avoidance

Already mentioned above — penalizes repetitive energy levels. This keeps the energy arc dynamic even within sections of the set that have the same overall target.

### Final Score Formula

```
score = (harmonic × 0.35) + (energy × 0.30) + (bpm × 0.15) + (genre × 0.10) + (artist × 0.05) + (plateau × 0.05)
```

The weights reflect a real DJ's priorities: harmonic compatibility and energy flow are most important, BPM matching and genre coherence matter but are secondary, and artist diversity and plateau avoidance are light corrections on top.

---

## Step 4: Transition Notes

After the set is built, the system generates a short note for each transition explaining the logic:

- "8A → 8A (same key, energy 6→7, +1 BPM)"
- "5A → 12A (energy boost +7, energy 7→9, +2 BPM)"
- "3B → 3A (ring switch, brightening, energy 8→8)"

These notes tell you exactly what the system was thinking. Use them to:
- Spot transitions you might want to reorder
- Understand which transitions might need a longer blend vs. a quick cut
- Identify tracks where the harmonic compatibility is low (you might want to swap something)

---

## The Five Energy Arc Profiles in Detail

### Journey (Default for Most Sets)

```
Energy:  4   5   6   7   8   9   9   8   7   6
Track:   1   2   3   4   5   6   7   8   9   10
         └── warm ──┴── build ──┴─ peak ─┴─ cool ─┘
```

Classic structure: start accessible, build gradually, peak, come back down. Use this for full festival sets, club nights, birthday parties — any set with a proper arc.

### Build (Continuous Rise)

```
Energy:  3   4   5   5   6   7   8   9   10  10
```

Only goes up. Use this when you're opening for someone else and handing off a peaked crowd, or when the vibe is explicitly a warm-up that needs to end on a high.

### Peak (High Energy Throughout)

```
Energy:  7   8   9   8   9   9   8   9   8   9
```

Stays in the 7–9 range with natural oscillation. Use this for peak-time slots, main stage sets where the crowd is already there, or when you've been asked to "just keep the energy high."

### Chill (Low Energy)

```
Energy:  3   4   3   4   4   3   4   5   3   4
```

Never goes above 5. Use this for lounge sets, background music at dinner, warm-up at 11pm before the club fills, or if you're closing an after-hours event at sunrise when people are winding down.

### Wave (Multiple Peaks)

```
Energy:  5   7   9   7   5   7   9   6   4   3
```

Builds to a peak, drops back, builds again. More complex and less predictable. Use this for longer sets (2+ hours) where a single arc would be boring, or for experimental/festival contexts where you want to control the crowd's energy more actively.

---

## Improving Your Results

### Use Specific My Tags

The more specific your prompt matches your actual tag names, the better the filtering. If your tag is `Tech House Banger`, asking for "tech house bangers" will match it. Asking for "hard tech house" might not match directly, but the system will try both full-name and word-level matching.

### Tag More Tracks

Every untagged track in your library is invisible to the tag-based filtering. The system can still use untagged tracks in the fallback pool, but it prioritizes tagged tracks. More tagging = more control.

### Specify Duration

Duration matters more than you might think. A 60-minute set at 128 BPM averages about 15–18 tracks. A 90-minute set needs 20–25 tracks. If you ask for "festival set" without a duration, the system defaults to 60 minutes. If your Festival-tagged library is 50+ tracks, you'll get a different (less curated) subset each time.

### Specify Starting Track

If you know exactly what you want to open with:

```
"90-minute festival set starting with Drumcode - Adam Beyer - Teach Me"
```

The system anchors the set to that track's key and energy, then builds forward from there. Everything else flows from that starting point.

### Adjust After Generation

The first result is a starting point, not the final answer. Common adjustments:

- "Swap track 4 for something darker"
- "The energy jumps too much at track 7, can you suggest alternatives?"
- "I want more vocal tracks in the first half"
- "Move the peak earlier — track 6 instead of track 9"

The system can re-generate with additional constraints, recommend alternatives for specific positions, or analyze why a particular transition might feel off.

---

## Exporting to Rekordbox

Once you have a set you like, export it as a Rekordbox playlist:

```
/export-set My Festival Set 2025
```

This creates a playlist inside Rekordbox with all the tracks in the suggested order. From there you can load it onto your USB, adjust the order manually if needed, and play it like any other playlist.

The export is non-destructive — it only creates a new playlist. It doesn't move files, modify tags, or change anything else in your library.

---

## What the System Can't Do

**It doesn't know your audience.** It knows your music and it follows the energy arc. But it doesn't know if the crowd is drunk or sober, European or American, listening to a soundsystem or club speakers. Your read of the room still drives the set.

**It doesn't know your mixing style.** The Camelot-based harmonic scoring assumes you're blending tracks tonally. If you prefer hard cuts or effects-heavy transitions, the harmonic score matters less and you might want to override some suggestions.

**It can't hear bad recordings.** If a track is poorly mastered, has a weird intro, or has a badly analyzed BPM, the system will slot it in based on its metadata — it doesn't know the track sounds off until you hear it.

**It doesn't replace your ears.** This is a planning tool. The final test is always how the set sounds when you play it.
