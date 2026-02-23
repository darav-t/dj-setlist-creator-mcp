# mcp-dj Documentation

This folder contains all documentation for the mcp-dj AI DJ setlist system.

---

## Where to Start

### I want to install this right now

Run the master installer — it handles everything from scratch, no Python or package manager needed:

```bash
# macOS / Linux
./install-master.sh --essentia

# Windows (PowerShell)
.\install-master.ps1 -Essentia
```

Then read **[Setup Guide](setup.md)** for what each step does, what to configure after, and troubleshooting.

### I'm a DJ and I want to understand the full setup

Read **[Setup Guide](setup.md)** — covers the master installer in detail, what each step does, how to configure your API key and Rekordbox path, and how to start the web UI or Claude Desktop integration.

### I want to understand what data the system reads from my library

Read **[Library Analysis](library-analysis.md)** — explains what gets read from Rekordbox, what Essentia ML analysis extracts from your audio files, and why each piece of data makes your generated sets better.

### I want to understand how sets are actually built

Read **[Building Sets](building-sets.md)** — covers how natural language prompts get interpreted, how the harmonic mixing algorithm works, what the energy arc profiles do, and how to get better results.

### I want the full technical reference

Read **[Technical Reference](reference.md)** — complete tool API, configuration options, data file formats, architecture diagrams, and extension points.

---

## Document Overview

| Document | Audience | What It Covers |
|----------|----------|----------------|
| [setup.md](setup.md) | Everyone | Master installer, manual install, Rekordbox detection, first run, troubleshooting |
| [library-analysis.md](library-analysis.md) | DJs | What data is read, what Essentia extracts, why it matters |
| [building-sets.md](building-sets.md) | DJs | How to prompt, how the algorithm works, energy profiles |
| [reference.md](reference.md) | Technical | Full API, config, file formats, architecture, extending |

---

## Quick Reference

### Common Commands

```bash
# ── First time ──────────────────────────────────────────────────────
# macOS/Linux — installs everything from scratch (no Python needed):
./install-master.sh --essentia

# Windows (PowerShell):
.\install-master.ps1 -Essentia

# ── Daily use ───────────────────────────────────────────────────────
# Start the web UI
./run-server.sh
# → open http://localhost:8888

# Analyze your library (run once, then incrementally for new tracks)
./analyze-library.sh

# Rebuild the track index (after updating Rekordbox tags)
# In Claude or web UI:
rebuild_library_index()

# Start MCP server for Claude Desktop
./run-mcp.sh
```

### Slash Commands (Claude Code)

```
/build-set 60-minute festival set starting mellow
/build-set dark afters techno, 2 hours
/dj-library search Festival
/export-set My Festival Set 2025
```

### What You Need to Run

| Required | Optional (but recommended) |
|----------|---------------------------|
| Rekordbox 6 with a library | Anthropic API key (for AI chat) |
| That's it — master installer handles the rest | Essentia ML models (~300 MB, for mood/genre analysis) |
| | Mixed In Key CSV export (better energy levels) |

> Python 3.12 and uv are installed automatically by `install-master.sh` / `install-master.ps1`.

---

## How the System Works (One Paragraph)

mcp-dj reads your Rekordbox library — tracks, BPMs, keys, My Tags, playlists — and optionally runs ML audio analysis (Essentia) over your audio files to extract mood, danceability, and genre from the actual sound. All of this is merged into a searchable index. When you ask for a set, the system uses your My Tags as the primary filter to narrow down candidate tracks, then orders them using the Camelot wheel for harmonic compatibility and a planned energy arc profile. The result is a fully ordered setlist with transition notes, exportable to Rekordbox as a playlist.

---

## Key Concepts

**My Tags** — The most important data in the system. Your existing Rekordbox My Tags (`Festival`, `Afters`, `High Energy`, etc.) are used as the primary filter when building sets. More tagging = more precise sets.

**Camelot Wheel** — A circular diagram of musical keys organized by harmonic compatibility. The system scores candidate tracks by how well they mix harmonically with the previous track.

**Energy Arc** — A planned curve of energy levels (1–10) across the set duration. Five profiles: `journey` (warm → build → peak → cool), `build` (continuous rise), `peak` (high throughout), `chill` (low throughout), `wave` (multiple peaks).

**Library Index** — A merged JSONL file (`.data/library_index.jsonl`) containing one record per track, combining Rekordbox metadata, Essentia ML features, and Mixed In Key energy data.

**Essentia** — Open-source ML audio analysis library. Runs locally on your machine. Extracts BPM, key, danceability, loudness, mood, and genre from audio files.
