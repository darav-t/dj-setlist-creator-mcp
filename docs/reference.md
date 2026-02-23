# Technical Reference

This document is the detailed technical reference for the mcp-dj system. It covers the full tool API, configuration options, architecture, file formats, and extension points. Intended for developers and technically curious DJs who want to understand or modify the system.

---

## Tool API Reference

All tools are available via the MCP server (Claude Desktop) and as REST endpoints (web UI).

### Set Building Tools

#### `build_set_from_prompt(prompt, duration_minutes)`

The primary set-building tool. Natural language → harmonic setlist using My Tag filtering.

**Parameters:**
- `prompt` *(string, required)* — Natural language description of the set
- `duration_minutes` *(number, default 60)* — Fallback duration; overridden if prompt contains explicit duration

**Process:**
1. Calls `_parse_set_intent(prompt, attrs)` to extract: My Tags, genre, BPM range, energy profile, starting track
2. Filters library index using matched My Tags
3. Runs `SetlistEngine.generate_setlist()` on filtered candidates
4. Returns setlist with track list, energy arc, harmonic score summary, and `intent` block

**Returns:**
```json
{
  "setlist_id": "sl_abc123",
  "tracks": [...],
  "duration_minutes": 62,
  "harmonic_score": 0.87,
  "energy_arc": [4, 5, 7, 8, 9, 9, 7, 5],
  "intent": {
    "detected_tags": ["Festival", "High Energy"],
    "candidates": 48,
    "energy_profile": "journey",
    "bpm_range": [124, 130]
  }
}
```

---

#### `generate_setlist(duration_minutes, genre, bpm_min, bpm_max, energy_profile, starting_track_title)`

Structured-parameter setlist generation. Use this when you know the exact technical constraints.

**Parameters:**
- `duration_minutes` *(number, default 60)*
- `genre` *(string, optional)* — Genre filter, e.g. `"tech house"`, `"techno"`
- `bpm_min` / `bpm_max` *(number, optional)* — BPM range
- `energy_profile` *(string)* — One of: `journey`, `build`, `peak`, `chill`, `wave`
- `starting_track_title` *(string, optional)* — Exact or partial track title

---

#### `plan_set(duration_minutes, vibe, situation, venue, crowd_energy, time_of_day, genre_preference)`

Vibe-based planning. Interprets contextual clues and translates to musical parameters.

**Parameters:**
- `vibe` — e.g. `"après ski"`, `"underground warehouse"`, `"rooftop sundowner"`
- `situation` — e.g. `"warm-up"`, `"peak time"`, `"closing"`, `"after party"`
- `venue` — e.g. `"ski chalet"`, `"nightclub"`, `"festival stage"`, `"beach bar"`
- `crowd_energy` — e.g. `"casual"`, `"hyped"`, `"just arriving"`, `"mixed"`
- `time_of_day` — e.g. `"afternoon"`, `"evening"`, `"late night"`, `"sunrise"`
- `genre_preference` — Optional override

**Returns:** Setlist + interpretation block showing parameter mapping.

---

### Analysis Tools

#### `analyze_track(file_path, force)`

Analyze a single audio file with Essentia ML.

**Parameters:**
- `file_path` *(string, required)* — Absolute path to audio file (`.mp3`, `.wav`, `.flac`, `.aiff`, `.m4a`)
- `force` *(boolean, default false)* — Re-analyze even if cached result exists

**Returns:**
```json
{
  "bpm": 126.0,
  "camelot_key": "8A",
  "key_confidence": 0.84,
  "danceability": 7.2,
  "lufs_integrated": -8.4,
  "rms_dbfs": -12.1,
  "mood_dominant": "party",
  "mood_scores": {
    "happy": 0.31,
    "sad": 0.08,
    "aggressive": 0.27,
    "relaxed": 0.11,
    "party": 0.58
  },
  "essentia_genres": [
    {"genre": "Tech House", "score": 0.72},
    {"genre": "House", "score": 0.51}
  ],
  "music_tags": ["electronic", "drums", "bass", "dance", "beat"]
}
```

**Cache:** Results saved to `.data/essentia_cache/<file_stem>.json`. Subsequent calls are instant.

---

#### `analyze_library_essentia(force, limit)`

Batch-analyze all tracks in the library.

**Parameters:**
- `force` *(boolean, default false)* — Re-analyze all tracks (ignores cache)
- `limit` *(integer, optional)* — Max tracks to analyze in this run (useful for incremental batches)

**Returns:**
```json
{
  "analyzed": 148,
  "cached": 2699,
  "skipped": 0,
  "errors": 3,
  "cache_dir": ".data/essentia_cache/"
}
```

**Performance:** CPU-intensive. Uses multiprocessing with one TensorFlow session per worker. Large libraries (2000+ tracks) take 1–3 hours on first run.

---

#### `analyze_energy_flow(track_titles)`

Analyze the energy flow of a sequence of tracks.

**Parameters:**
- `track_titles` *(array of strings)*

**Returns:** Energy arc analysis with per-transition commentary and recommendations.

---

#### `get_track_compatibility(track_a_title, track_b_title)`

Detailed harmonic + energy compatibility between two specific tracks.

**Returns:** Camelot transition type, harmonic score, energy delta, BPM delta, recommended blend technique.

---

#### `recommend_next_track(current_track_title, energy_direction, limit)`

What tracks play well after a specific track.

**Parameters:**
- `current_track_title` *(string, required)*
- `energy_direction` *(string)* — `"up"`, `"down"`, or `"maintain"`
- `limit` *(integer, default 5, max 20)*

**Returns:** Ranked list of recommendations with harmonic score and reasoning.

---

#### `get_compatible_tracks(key, bpm, bpm_tolerance, energy_min, energy_max, genre, limit)`

Find tracks harmonically compatible with a given key.

**Parameters:**
- `key` *(string, required)* — Camelot key, e.g. `"8A"`, `"12B"`
- `bpm` *(number, optional)* — Target BPM
- `bpm_tolerance` *(number, default 4)* — BPM tolerance percentage
- `energy_min` / `energy_max` *(number 1–10, optional)*
- `genre` *(string, optional)*
- `limit` *(integer, default 20)*

---

### Library Tools

#### `search_library(query, date_from, date_to, my_tag, limit)`

Search tracks with free-text and filters.

**Parameters:**
- `query` *(string, optional)* — Searches title, artist, album, genre
- `date_from` / `date_to` *(YYYY-MM-DD, optional)* — Filter by date added to Rekordbox
- `my_tag` *(string, optional)* — Filter by exact My Tag label
- `limit` *(integer, default 20)*

---

#### `get_library_summary()`

High-level library statistics: track count, BPM range, top genres, key distribution.

---

#### `get_library_attributes()`

Full dynamic attribute summary. The most comprehensive library overview tool.

**Returns:**
```json
{
  "my_tag_hierarchy": {
    "Vibes": ["High Energy", "Dark", "Dance", ...],
    "Situation": ["Festival", "Sunset", "Afters", ...],
    "Bangers": ["Tech House Banger", "House Bangers", ...]
  },
  "my_tag_details": {
    "Festival": {
      "count": 143,
      "bpm": {"p25": 124, "p75": 130},
      "energy": {"avg": 7.8},
      "mood_dominant": "party",
      "top_genres": ["Tech House", "Techno"],
      "co_tags": ["High Energy", "Dance"]
    }
  },
  "genres": {"Tech House": 412, "Techno": 287, ...},
  "bpm": {"min": 98, "max": 145, "avg": 126.4, "p25": 122, "p75": 130},
  "energy": {"distribution": {1: 12, 2: 34, 3: 89, ...}},
  "moods": {"party": 891, "aggressive": 445, ...},
  "top_artists": [["Adam Beyer", 28], ["Amelie Lens", 21], ...]
}
```

---

#### `get_track_full_info(title_or_id)`

Complete merged record for a track from the library index.

**Parameters:**
- `title_or_id` *(string)* — Track title (substring match) or Rekordbox track ID

---

#### `rebuild_library_index(force)`

Rebuild the JSONL index and dynamic attributes.

**Parameters:**
- `force` *(boolean, default false)* — Force rebuild even if index was built within the last hour

**When to use:** After adding tracks to Rekordbox, updating My Tags, or importing new MIK data.

---

### Export Tools

#### `export_setlist_to_rekordbox(setlist_name, track_titles, setlist_id)`

Create a Rekordbox playlist from a setlist.

**Parameters:**
- `setlist_name` *(string, required)* — Name for the new playlist
- `track_titles` *(array of strings, optional)* — Track titles in order (if `setlist_id` not provided)
- `setlist_id` *(string, optional)* — ID of a previously generated setlist

---

### Raw Database Query Tools

These tools expose the Rekordbox database tables directly. Useful for debugging or building custom integrations.

| Tool | Table | Returns |
|------|-------|---------|
| `get_db_artists(limit, offset)` | `djmdArtist` | ID, Name, SearchStr |
| `get_db_albums(limit, offset)` | `djmdAlbum` | ID, Name, AlbumArtistID, Compilation |
| `get_db_genres(limit, offset)` | `djmdGenre` | ID, Name |
| `get_db_labels(limit, offset)` | `djmdLabel` | ID, Name |
| `get_db_keys(limit, offset)` | `djmdKey` | ID, ScaleName, Seq |
| `get_db_colors(limit, offset)` | `djmdColor` | ID, ColorCode, SortKey |
| `get_db_playlists(limit, offset)` | `djmdPlaylist` | ID, Name, Seq, Attribute (0=playlist, 1=folder, 4=smart), ParentID |
| `get_db_playlist_songs(playlist_id, limit, offset)` | `djmdSongPlaylist` | ID, PlaylistID, ContentID, TrackNo |
| `get_db_history(limit, offset)` | `djmdHistory` | ID, Name, Seq, DateCreated |
| `get_db_history_songs(history_id, limit, offset)` | `djmdSongHistory` | ID, HistoryID, ContentID, TrackNo |
| `get_db_my_tags(limit, offset)` | `djmdMyTag` | ID, Name, Seq, Attribute, ParentID |
| `get_db_my_tag_songs(my_tag_id, limit, offset)` | `djmdSongMyTag` | ID, MyTagID, ContentID, TrackNo |
| `get_db_cues(content_id, limit, offset)` | `djmdCue` | ID, ContentID, InMsec, OutMsec, Kind, Color, Comment |
| `get_db_mixer_params(content_id, limit, offset)` | `djmdMixerParam` | ID, ContentID, GainHigh, GainLow, PeakHigh, PeakLow |
| `get_db_property()` | `djmdProperty` | DBID, DBVersion, BaseDBDrive, CurrentDBDrive |

---

## Configuration Reference

### Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(empty)* | Claude API key. Required for AI chat features; system runs in rule-based mode without it |
| `SETLIST_PORT` | `8888` | Port for the FastAPI web UI |
| `MIK_CSV_PATH` | *(empty)* | Path to Mixed In Key `Library.csv` export |
| `REKORDBOX_DB_PATH` | *(auto-detect)* | Override Rekordbox database path. Auto-detects at `~/Library/Pioneer/` (macOS) or `%APPDATA%\Pioneer\` (Windows) |

### Claude Desktop Config

File location:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "mcp-dj": {
      "command": "/bin/bash",
      "args": [
        "-c",
        "cd '/absolute/path/to/mcp_dj' && exec .venv/bin/python3 -m mcp_dj.mcp_server"
      ]
    }
  }
}
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Interfaces                              │
│  Claude Desktop (MCP/stdio)    FastAPI Web UI (:8888)           │
└──────────────────────┬──────────────────┬───────────────────────┘
                       │                  │
              ┌────────▼──────────────────▼────────┐
              │           mcp_server.py             │
              │   FastMCP tool decorators + intent  │
              │   parser (_parse_set_intent)         │
              └────────────────┬────────────────────┘
                               │
              ┌────────────────▼────────────────────┐
              │         SetlistEngine               │
              │   Greedy scoring: Camelot + energy  │
              │   + BPM + genre + artist + plateau  │
              └────┬─────────────┬──────────────────┘
                   │             │
     ┌─────────────▼──┐  ┌──────▼──────────┐
     │  LibraryIndex  │  │  EnergyPlanner  │
     │  JSONL search  │  │  5 arc profiles │
     └───────┬────────┘  └─────────────────┘
             │
    ┌────────▼─────────────────────────────┐
    │           Data Sources               │
    │                                      │
    │  RekordboxDatabase    EssentiaAnalyzer│
    │  (pyrekordbox,        (ML audio,     │
    │   read-only)          cached JSONL)  │
    │                                      │
    │  MixedInKeyLibrary                   │
    │  (CSV import,                        │
    │   energy scores)                     │
    └──────────────────────────────────────┘
```

---

## Key Source Files

| File | Lines | Responsibility |
|------|-------|----------------|
| `mcp_dj/mcp_server.py` | ~1200 | MCP tool definitions + `_parse_set_intent()` |
| `mcp_dj/setlist_engine.py` | ~600 | Core greedy algorithm, scoring weights |
| `mcp_dj/library_index.py` | ~500 | JSONL index build + `build_attributes()` |
| `mcp_dj/essentia_analyzer.py` | ~450 | ML analysis + multiprocessing |
| `mcp_dj/database.py` | ~400 | Rekordbox database layer |
| `mcp_dj/camelot.py` | ~200 | Camelot wheel scoring |
| `mcp_dj/energy_planner.py` | ~180 | Energy arc profiles |
| `mcp_dj/energy.py` | ~160 | Energy resolution (MIK/album tag/heuristic) |
| `mcp_dj/intent.py` | ~250 | Vibe/venue/time keyword tables |
| `mcp_dj/app.py` | ~400 | FastAPI endpoints |
| `mcp_dj/models.py` | ~150 | Pydantic data models |

---

## Data File Formats

### `.data/library_index.jsonl`

One JSON object per line. Fields:

```typescript
{
  id: number,                    // Rekordbox ContentID
  title: string,
  artist: string,
  album: string,
  genre: string,                 // Rekordbox genre tag
  bpm: number,                   // Rekordbox BPM
  camelot_key: string,           // "8A", "3B", etc.
  rating: number,                // 0-5 stars
  play_count: number,
  date_added: string,            // ISO date
  file_path: string,

  // My Tags
  my_tags: string[],
  my_tag_ids: number[],

  // Energy (from best available source)
  energy: number,                // 1-10
  energy_source: "mik" | "album_tag" | "inferred",

  // Essentia features (null if not analyzed)
  essentia_bpm: number | null,
  essentia_key: string | null,
  key_confidence: number | null,
  danceability: number | null,   // 1-10 scaled
  lufs_integrated: number | null,
  rms_dbfs: number | null,
  mood_dominant: string | null,
  mood_scores: { happy, sad, aggressive, relaxed, party } | null,
  essentia_genres: Array<{genre: string, score: number}> | null,
  music_tags: string[] | null,

  // Grep-optimized search field
  _text: string
}
```

### `.data/library_attributes.json`

Summarized library statistics used for intent parsing and UI display. Rebuilt by `build_attributes()` from the JSONL index.

### `.data/essentia_cache/<stem>.json`

Per-track analysis results keyed by audio file stem (filename without extension). Contains the raw Essentia output before merging into the library index.

### `.data/models/`

TensorFlow model files downloaded by `download_models.sh`:

| File | Model | Purpose |
|------|-------|---------|
| `mood_happy-classifier-musicnn-msd-2.pb` | MusicNN | Happy mood detection |
| `mood_sad-classifier-musicnn-msd-2.pb` | MusicNN | Sad mood detection |
| `mood_aggressive-classifier-musicnn-msd-2.pb` | MusicNN | Aggressive mood |
| `mood_relaxed-classifier-musicnn-msd-2.pb` | MusicNN | Relaxed mood |
| `mood_party-classifier-musicnn-msd-2.pb` | MusicNN | Party mood |
| `danceability-classifier-musicnn-msd-2.pb` | MusicNN | Danceability score |
| `genre_discogs400-discogs-effnet-1.pb` | EfficientNet | Genre classification |
| `mtg_jamendo_moodtheme-discogs-effnet-1.pb` | EfficientNet | Mood/theme tags |
| `mtt-msd-musicnn-1.pb` | MusicNN | MagnaTagATune tags |

---

## Energy Arc Implementation

Energy arcs are defined in `mcp_dj/energy_planner.py` as normalized curves (values 0–1) mapped to energy levels 1–10.

```python
ENERGY_PROFILES = {
    "journey": [0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.0, 0.85, 0.7, 0.5],
    "build":   [0.3, 0.4, 0.5, 0.55, 0.65, 0.75, 0.85, 0.9, 1.0, 1.0],
    "peak":    [0.7, 0.8, 0.9, 0.8, 0.9, 1.0, 0.9, 1.0, 0.9, 0.85],
    "chill":   [0.3, 0.35, 0.3, 0.4, 0.4, 0.35, 0.4, 0.45, 0.35, 0.4],
    "wave":    [0.5, 0.7, 0.9, 0.7, 0.5, 0.7, 0.9, 0.65, 0.4, 0.3],
}
```

`EnergyPlanner.get_target_energy(profile, position, total_tracks)` interpolates over this curve and returns a target energy (1–10) for each position in the set.

---

## Camelot Wheel Implementation

The Camelot wheel maps musical keys to compatible neighbors. Implemented in `mcp_dj/camelot.py`.

**Key properties:**
- 24 positions: 12 inner (A = minor) + 12 outer (B = major)
- Position numbers 1–12, each representing a pitch class
- Compatible neighbors: same position, ±1 position (same ring), same position opposite ring

**Transition scoring matrix:**
```python
TRANSITION_SCORES = {
    "same":      1.00,
    "adjacent":  0.90,  # +1 or -1 same ring
    "ring":      0.85,  # A↔B same number
    "boost":     0.70,  # +7 positions (energy boost technique)
    "diagonal":  0.60,  # +1 position + ring switch
    "other":     0.10,  # everything else
}
```

**Key parsing:** Accepts Camelot (`"8A"`), Open Key (`"6m"`), and standard notation (`"Am"`, `"C"`, `"F#m"`) — all internally normalized to Camelot.

---

## Setlist Engine Scoring

From `mcp_dj/setlist_engine.py`:

```python
SCORE_WEIGHTS = {
    "harmonic":  0.35,
    "energy":    0.30,
    "bpm":       0.15,
    "genre":     0.10,
    "artist":    0.05,
    "plateau":   0.05,
}
```

**BPM scoring function:**
```
bpm_score = max(0, 1 - (abs(bpm_delta) / 8))
```
Zero penalty within 0 BPM delta, zero score at 8+ BPM delta.

**Artist diversity window:** 5 tracks. Score penalty = `1 - (1 / (1 + tracks_since_last_play))`.

**Plateau detection:** 3+ consecutive tracks at same energy level triggers a 0.3 penalty multiplier on the plateau score.

---

## FastAPI REST Endpoints

Base URL: `http://localhost:8888`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/api/library/stats` | Library statistics |
| `GET` | `/api/library/attributes` | Full dynamic attributes |
| `GET` | `/api/library/tracks` | Search tracks (`?query=&my_tag=&date_from=&date_to=`) |
| `GET` | `/api/library/track/{id}` | Full merged track record |
| `POST` | `/api/library/rebuild-index` | Rebuild JSONL index (`?force=true`) |
| `POST` | `/api/setlist/generate` | Structured setlist generation |
| `POST` | `/api/setlist/plan` | Vibe-based planning |
| `POST` | `/api/setlist/build` | Prompt-driven setlist |
| `GET` | `/api/setlist/{id}` | Retrieve generated setlist |
| `POST` | `/api/setlist/recommend` | Next-track recommendations |
| `POST` | `/api/setlist/compatibility` | Track compatibility check |
| `POST` | `/api/setlist/energy-flow` | Energy arc analysis |
| `POST` | `/api/setlist/compatible-tracks` | Find harmonically compatible tracks |
| `POST` | `/api/essentia/analyze` | Single file analysis |
| `POST` | `/api/essentia/analyze-library` | Batch analysis |
| `POST` | `/api/chat` | Chat with AI |
| `POST` | `/api/chat/clear` | Clear conversation history |

---

## Claude Slash Commands

Located in `.claude/commands/`. These are shortcuts for common operations when using Claude Code.

| Command | File | Description |
|---------|------|-------------|
| `/build-set [prompt]` | `build-set.md` | Build a setlist from natural language |
| `/dj-library [query]` | `dj-library.md` | Browse library, search tracks, view stats |
| `/export-set [name]` | `export-set.md` | Export last set to Rekordbox |

---

## Dependency Summary

| Package | Version | Purpose |
|---------|---------|---------|
| `pyrekordbox` | 0.4.3+ | Rekordbox database access |
| `pydantic` | 2.0+ | Data models and validation |
| `fastapi` | 0.129+ | Web UI REST server |
| `fastmcp` | 2.11+ | MCP server framework |
| `anthropic` | 0.49+ | Claude API client |
| `essentia-tensorflow` | 2.1b6+ | ML audio analysis (optional) |
| `uvicorn` | latest | ASGI server for FastAPI |
| `python-dotenv` | latest | `.env` file loading |

Python 3.12+ required. Package management via `uv`.

---

## Extending the System

### Adding a New Energy Profile

In `mcp_dj/energy_planner.py`, add to `ENERGY_PROFILES`:

```python
"myprofile": [0.6, 0.6, 0.8, 0.8, 1.0, 1.0, 0.8, 0.6, 0.4, 0.3],
```

This immediately makes `energy_profile="myprofile"` available to all tools.

### Adding a New Vibe Keyword

In `mcp_dj/intent.py`, add to `VIBE_PROFILES`:

```python
"jungle rave": ("jungle", 160, 175),  # (genre_hint, bpm_min, bpm_max)
```

### Adding a New MCP Tool

In `mcp_dj/mcp_server.py`, use the `@mcp.tool()` decorator:

```python
@mcp.tool()
async def my_new_tool(param: str) -> dict:
    """Description shown to Claude."""
    await _ensure_initialized()
    # ... implementation
    return {"result": "..."}
```

### Rebuilding the Library Index Programmatically

```python
from mcp_dj.library_index import LibraryIndex
index = LibraryIndex()
index.build(force=True)
attrs = index.get_attributes()
```
