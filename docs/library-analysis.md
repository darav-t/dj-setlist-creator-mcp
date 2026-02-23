# Library Analysis: What the System Reads and Why It Matters

Before the AI can build you a set, it needs to understand your music. This document explains exactly what data the system pulls from your library, where that data comes from, and — most importantly — why each piece of information makes your generated sets better.

---

## The Three Data Sources

The system pulls from three places and merges them into one complete picture of each track:

```
Rekordbox database  ──┐
                       ├──► Library Index (one record per track)
Essentia ML analysis ──┤
                       │
Mixed In Key (CSV) ────┘
```

Each source fills in gaps the others can't cover. Here's what comes from each.

---

## Source 1: Rekordbox

This is the foundation — everything you've already tagged in Rekordbox.

### What's Read

| Data | Example | Why It Matters |
|------|---------|----------------|
| Title, Artist, Album | "Shake It" / "Metro Area" | Track identity, artist diversity |
| BPM | 124.0 | Harmonic mixing, energy estimates |
| Key | 8A (Camelot), or Am/C | Camelot wheel compatibility |
| Genre | Tech House | Set coherence, genre filtering |
| Rating | 4 stars | Track quality hints |
| Play count | 23 times played | Track familiarity |
| My Tags | Festival, High Energy, Has Vocal | The most powerful data source |
| Playlists | "Ibiza 2024", "Warm-Up" | Context clues |
| Date Added | 2024-03-15 | Library freshness |
| Hot Cues | A, B, C... | Transition planning hints |
| Color labels | Red, Blue... | Your personal organization |

### My Tags: The Most Important Thing

Your My Tags are the most valuable data in the system — more valuable than Essentia ML analysis, more valuable than genre tags. Why? Because you manually tagged these tracks. You already made the curatorial decisions. The system uses your tags as the primary filter when building sets.

When you ask for "a festival set," the system doesn't just guess — it searches for tracks you've tagged as `Festival`. When you ask for "something dark and underground," it looks for tracks tagged `Dark` or `Afters`. Your tagging work directly controls what ends up in your sets.

This is why the system reads your entire My Tag hierarchy:

```
Has           → Has Vocal, My Comment
Vibes         → High Energy, Dark, Dance, Tribal, Hard, Chill, Housy...
Situation     → Festival, Sunset, Afters, Intro, Wedding, Generic Bar...
Bangers       → Tech House Banger, House Bangers, Techno Bangers...
```

If a tag doesn't exist in your library, it won't be suggested. The system only works with what you've actually tagged.

**Practical tip:** The more tags you apply, the more precise your generated sets. Even coarse tags like "Intro" or "Ending" are enough for the system to pick appropriate tracks for the start and end of a set.

### BPM and Key

Rekordbox stores the BPM and key as analyzed during import (or as corrected manually). These are the primary inputs for:

- **BPM matching:** The system avoids jumps larger than ~6 BPM between consecutive tracks (unless you're doing an intentional key change)
- **Camelot wheel:** The harmonic mixing system uses the key in Camelot notation to ensure tracks flow musically — more on this in the Building Sets doc

If your BPMs are wrong in Rekordbox (common with complex rhythms, live recordings, or improperly imported tracks), the system will try to mix those tracks in awkward positions. It's worth correcting your BPM grid analysis in Rekordbox for tracks you care about.

---

## Source 2: Essentia ML Analysis

This is the "listening to your music" part. Essentia is an open-source audio analysis library developed by the Music Technology Group at Universitat Pompeu Fabra. It actually processes the audio file and extracts musical features from the waveform.

### Why Essentia Exists Here

Rekordbox knows your BPM and key. But it doesn't know if a track sounds *dark* or *happy*. It doesn't know if the track is highly danceable or more of a tension builder. It doesn't know the acoustic genre (which may differ from how you tagged it). Essentia fills these gaps by listening.

### What Essentia Extracts

#### BPM (from audio)

Essentia recalculates BPM from the actual audio using a rhythm extractor. This serves as a cross-check against the Rekordbox value. If they disagree by more than a few BPM, something is off — usually a doubled or halved BPM reading.

The system uses the Rekordbox BPM as the primary value (since you may have manually corrected it), but the Essentia BPM is stored as a backup.

#### Key (from audio)

Key detection from audio is harder than BPM — pitch analysis is inherently ambiguous. Essentia provides a confidence score (0–1) alongside the key. A confidence of 0.9 means it's very sure; 0.4 means there's significant ambiguity (common with highly percussive or atonal music).

The system uses the Rekordbox key as primary (since Rekordbox uses Mixed In Key internally), but Essentia's key confidence score is used to flag tracks where key mixing might be unreliable.

#### Danceability

A score from 0 to 3+ (the system scales this to 1–10) that estimates how suitable a track is for dancing, based on rhythmic regularity, beat strength, and tempo stability.

- **High danceability (8–10):** Consistent kick, predictable structure, crowd-pleasing rhythm section
- **Low danceability (2–4):** Experimental, arrhythmic, or ambient tracks

Why this matters for set building: you probably don't want to drop a danceability-4 track at the peak of your festival set, even if it has the right BPM and key.

#### Loudness (EBU R128)

The industry standard for measuring perceived loudness. Essentia reports:

| Metric | What It Is | Why It Matters |
|--------|-----------|----------------|
| Integrated LUFS | Average loudness over the whole track | Gain matching between tracks |
| Loudness Range (LU) | Dynamic variation within the track | Compression/headroom indicator |
| RMS dBFS | Root mean square amplitude | Peak energy level |

Tracks with similar LUFS values blend better. A track that's 6 dB louder than the previous one will feel like a jarring jump even if the BPM and key are perfect. The system uses this to flag potentially problematic loudness jumps in the suggested setlist.

#### Mood (5-Class Probability)

Essentia runs a ML classifier trained on a large music database to estimate mood. Each track gets a probability for each of five moods:

| Mood | What It Sounds Like in Electronic Music |
|------|-----------------------------------------|
| Happy | Uplifting chords, major keys, bright synths |
| Sad | Minor keys, slower progressions, melancholic |
| Aggressive | Distorted kicks, heavy drops, driving energy |
| Relaxed | Atmospheric pads, sparse percussion, low energy |
| Party | Anthemic, big room, crowd-reactive elements |

The system uses the **dominant mood** (highest probability) when filtering and scoring. If you ask for "dark underground techno," the system preferentially selects tracks where `aggressive` is the dominant mood.

This mood data layers on top of your My Tags. If you've tagged something `Dark`, it's very likely the system will also find high aggressive and low happy scores on that track — and the combination of both signals makes the filtering more confident.

#### Genre (Discogs400)

Essentia classifies tracks against 400 genre/style labels from the Discogs database. The top-scoring genres are stored with their confidence values.

This is different from your Rekordbox genre tag. Your Rekordbox tag is what *you* called it; the Discogs genre is what the audio *sounds like* to a ML classifier.

These often agree. Sometimes they don't — which is interesting information. A track you tagged "Tech House" that the classifier thinks sounds like "Deep House" or "Techno" might be a style-boundary track that could work in either context.

The system uses this for genre coherence scoring in setlists — it avoids dropping a full techno-sounding track in the middle of a house set even if you tagged them both loosely as "House."

#### Music Tags (MagnaTagATune)

The MagnaTagATune model outputs a set of music descriptor tags — things like `electronic`, `drums`, `bass`, `vocal`, `ambient`, `fast`, `guitar`, etc. Tags above a 0.1 confidence threshold are stored.

These tags help the system understand track character beyond genre. A track tagged `vocal` by MagnaTagATune confirms your `Has Vocal` My Tag. A track tagged `ambient` alongside a high `relaxed` mood score strongly suggests it works as an intro or cooldown track.

---

## Source 3: Mixed In Key (Optional)

Mixed In Key is a paid desktop application that analyzes your music and writes energy levels (1–10) and keys into your Rekordbox tags. Many DJs already use it.

If you've exported your Mixed In Key library as a CSV, you can point the system at it via `MIK_CSV_PATH` in your `.env` file. The system will read the energy values MIK assigned to each track and use those as the primary energy source.

### Why MIK Beats Inferred Energy

MIK's energy levels are calculated from loudness, rhythmic complexity, and spectral energy. They're not perfect, but they're consistent and DJ-workflow-tested. Thousands of DJs have used them to plan sets.

When MIK data is available, the system prefers it over inferred energy values. Here's the full priority chain:

```
1. Mixed In Key energy value (if MIK_CSV_PATH set and track found)
   ↓
2. Rekordbox album tag in "1A 7" format (some DJs embed energy in tags)
   ↓
3. Estimated from BPM + genre via heuristic
   (e.g., 128 BPM Tech House → estimated energy 7–8)
```

---

## The Library Index: One Record Per Track

After reading all three sources, the system merges everything into a single JSONL file (`.data/library_index.jsonl`). Each line is one track, containing everything known about it:

```json
{
  "id": 12847,
  "title": "Shake It",
  "artist": "Metro Area",
  "bpm": 124.0,
  "camelot_key": "8A",
  "energy": 7,
  "energy_source": "mik",
  "my_tags": ["Festival", "Dance", "Tech House Banger"],
  "genre": "Tech House",
  "mood_dominant": "party",
  "mood_scores": {"happy": 0.42, "aggressive": 0.31, "party": 0.62},
  "danceability": 8.2,
  "lufs": -8.4,
  "essentia_genres": ["Tech House", "House"],
  "music_tags": ["electronic", "drums", "bass", "dance"],
  "_text": "my_tags:Festival my_tags:Dance energy:7 mood:party genre:Tech House ..."
}
```

The `_text` field is a grep-friendly version — prefixed tokens that make it easy to search the whole library fast.

### Library Attributes

On top of the individual track records, the system builds a **library attributes summary** that answers questions like:

- What My Tags exist in this library, and how many tracks have each?
- What's the BPM range for tracks tagged `Sunset`?
- What's the dominant mood for tracks tagged `Afters`?
- What genres appear in this library, and what's their typical energy range?

This summary is used by the intent parser — when you ask for "a dark sunset set," it reads the actual characteristics of your `Dark` and `Sunset` tagged tracks to understand what BPM range and energy arc that means *for your library specifically*, not some generic definition.

---

## When to Re-Run Analysis

| Situation | Action |
|-----------|--------|
| Added new tracks to Rekordbox | `./analyze-library.sh` (only analyzes new tracks) |
| Updated My Tags in Rekordbox | `rebuild_library_index()` (re-reads tags, no re-analysis needed) |
| Changed energy scores in MIK | Update `MIK_CSV_PATH`, then `rebuild_library_index()` |
| Corrected BPM/key in Rekordbox | `rebuild_library_index()` |
| Want to re-analyze a specific track | `analyze_track(file_path, force=True)` |

---

## What the System Does NOT Read

- **Your audio files directly in real-time** — analysis is done once and cached
- **Other DJ software libraries** (Traktor, Serato, VirtualDJ) — Rekordbox only
- **Cloud-hosted music** (Beatport streaming, Tidal DJ) — local files only
- **Track waveform data** — that's stored in Rekordbox's `.pdb` files and not used here
- **Comments or notes fields** from Rekordbox — only the `My Comment` My Tag is read

---

## Privacy and Data

All analysis runs 100% locally on your machine. Your music, your tags, and your library data never leave your computer. The only network call is the optional Anthropic API for AI chat features, and that only sends your text prompt — not your music or library data.
