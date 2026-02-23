# Setup Guide: From Zero to Your First AI-Generated Set

This guide walks you through getting mcp-dj running from a completely fresh machine. The fastest path is the **master installer** — one command handles everything.

---

## What You're Installing

mcp-dj connects to your existing Rekordbox library and adds an AI layer on top. The system reads your library — tracks, tags, BPMs, keys — and uses that to build harmonically correct, energy-planned DJ sets. You don't move your music anywhere. It just reads what's already there.

The software has two parts:

- **The MCP Server** — runs in the background, lets Claude talk directly to your library
- **The Web UI** — a browser-based chat interface you can use without Claude Desktop

---

## Option 1: Master Installer (Recommended)

One command. No Python, no package managers, nothing pre-installed. The master installer handles everything — including Python itself.

### macOS / Linux

Open Terminal, navigate to the project folder, and run:

```bash
chmod +x install-master.sh && ./install-master.sh
```

With Essentia ML audio analysis (strongly recommended — enables mood, genre, danceability):

```bash
./install-master.sh --essentia
```

**What it does, step by step:**

| Step | What Happens |
|------|-------------|
| 0 | Installs Xcode Command Line Tools (macOS only, if missing) |
| 1 | Ensures `curl` is available (installs via apt/dnf/pacman on Linux if needed) |
| 2 | Installs `uv` — the Python package manager |
| 3 | Installs Python 3.12 via uv (no system Python required) |
| 4 | Creates `.venv/` with all project dependencies |
| 5 | Hands off to `install.sh` for the rest (see below) |
| 6 | Optionally installs Essentia ML analysis (~135 MB) |
| 7 | Optionally downloads ML models (~300 MB) |
| 8 | Configures Rekordbox database access (pyrekordbox setup) |
| 9 | Creates your `.env` configuration file |
| 10 | Registers the MCP server in Claude Code |
| 11 | Registers the MCP server in Claude Desktop |
| 12 | Optionally analyzes and indexes your full library |

### Windows

Open **PowerShell** (no admin required), navigate to the project folder, and run:

```powershell
# Allow scripts to run (one-time, current user only)
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# Run the master installer
.\install-master.ps1
```

With Essentia ML analysis:

```powershell
.\install-master.ps1 -Essentia
```

**What it does:**

| Step | What Happens |
|------|-------------|
| 0 | Checks PowerShell version and execution policy |
| 1 | Installs Python 3.12 via `winget` (or downloads installer as fallback) |
| 2 | Installs `uv` via PowerShell |
| 3 | Installs core Python packages via `uv sync` |
| 4 | Optionally installs Essentia ML analysis |
| 5 | Optionally downloads ML models |
| 6 | Configures Rekordbox database access |
| 7 | Creates `.env` configuration file |
| 8 | Registers MCP server in Claude Desktop config |
| 9 | Optionally analyzes and indexes your library |

### Flags

| Flag | macOS/Linux | Windows | Effect |
|------|-------------|---------|--------|
| Include Essentia | `--essentia` | `-Essentia` | Install ML analysis + download models |
| Skip model download | `--skip-models` | `-SkipModels` | Install Essentia but skip the ~300 MB download |

---

## After the Installer Completes

The installer tells you exactly what to do next at the end, but the short version is:

### 1. Add your Anthropic API key

Open `.env` in any text editor and set:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Get a key at [console.anthropic.com](https://console.anthropic.com) → API Keys. This is only needed for the AI chat features. The set builder itself works without it.

### 2. Start the web UI

```bash
./run-server.sh               # macOS/Linux
```

Then open [http://localhost:8888](http://localhost:8888). You'll see a chat where you can type:

> "Build me a 60-minute festival set"

### 3. Connect Claude Desktop

The installer registers the MCP server automatically. Just **restart Claude Desktop** and the `mcp-dj` tools will appear in the tools panel.

---

## Option 2: Manual Installation

If you prefer to control each step yourself, or if the master installer fails for your specific setup.

### Prerequisites

| What | Why |
|------|-----|
| macOS 12+ or Windows 11 | Supported platforms |
| Rekordbox 6 with your library loaded | The source of all your track data |
| Python 3.12 or newer | The runtime the code runs on |
| uv (package manager) | Installs Python packages reliably |
| ~500 MB free disk space | For ML models (optional but recommended) |
| Anthropic API key | Optional — only needed for AI chat features |

### Step 1: Install Python 3.12+

**macOS** — via the official installer:
1. Go to [python.org/downloads](https://python.org/downloads)
2. Download **Python 3.12** (or newer)
3. Run the `.pkg` installer
4. Open Terminal: `python3 --version` should print `Python 3.12.x`

Alternatively, with Homebrew:
```bash
brew install python
```

**Windows:**
1. Download from [python.org/downloads](https://python.org/downloads)
2. Run — **check "Add Python to PATH"** before clicking Install
3. Verify: `python --version` in PowerShell

### Step 2: Install uv

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then restart your terminal (or `source ~/.zshrc`).

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify: `uv --version` should print `uv 0.5.x` or newer.

### Step 3: Clone or Download the Project

```bash
git clone <your-repo-url> ~/Music/mcp_dj
cd ~/Music/mcp_dj
```

Or download the zip and extract it. The location matters — you'll reference it in config files.

### Step 4: Run the Standard Installer

```bash
chmod +x install.sh
./install.sh
```

This handles everything from here: packages, Rekordbox setup, .env, Claude Desktop config, and optionally Essentia + library analysis.

```bash
./install.sh --essentia          # with Essentia ML
./install.sh --essentia --skip-models   # Essentia but skip model download
```

### Step 5: Configure .env

Open the `.env` file:

```bash
# Your Anthropic API key (get one at console.anthropic.com)
ANTHROPIC_API_KEY=

# Port for the web UI (default: 8888)
SETLIST_PORT=8888

# Optional: path to your Mixed In Key Library.csv export
MIK_CSV_PATH=

# Optional: override Rekordbox database location (auto-detected by default)
REKORDBOX_DB_PATH=
```

### Step 6: Verify Rekordbox is Detected

```bash
.venv/bin/python -c "from mcp_dj.database import RekordboxDatabase; db = RekordboxDatabase(); tracks = db.get_all_tracks(); print(f'Found {len(tracks)} tracks')"
```

Should print `Found XXXX tracks`.

### Step 7: Download ML Models (Optional but Recommended)

```bash
chmod +x download_models.sh
./download_models.sh
```

~300 MB, one-time download. Without models: the system works using BPM and your tags. With models: every track also gets mood, danceability, and genre from the audio itself.

### Step 8: Analyze Your Library

```bash
chmod +x analyze-library.sh
./analyze-library.sh
```

Reads all your audio files, caches results per track, builds the searchable index. Takes a while on first run; subsequent runs only process new tracks.

---

## Claude Desktop: Manual MCP Registration

If the installer couldn't register automatically, add this to your Claude Desktop config file:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

**macOS / Linux:**
```json
{
  "mcpServers": {
    "mcp-dj": {
      "command": "/bin/bash",
      "args": [
        "-c",
        "cd '/Users/you/Music/mcp_dj' && exec .venv/bin/python3 -m mcp_dj.mcp_server"
      ]
    }
  }
}
```

**Windows:**
```json
{
  "mcpServers": {
    "mcp-dj": {
      "command": "cmd",
      "args": [
        "/c",
        "cd /d \"C:\\Users\\you\\Music\\mcp_dj\" && .venv\\Scripts\\python.exe -m mcp_dj.mcp_server"
      ]
    }
  }
}
```

Restart Claude Desktop after saving.

---

## Quick-Start Commands Reference

```bash
# Start web UI
./run-server.sh

# Start MCP server (stdio, for Claude Desktop)
./run-mcp.sh

# Start MCP server (HTTP/SSE, for other MCP clients)
./run-mcp-http.sh

# Analyze a single track
.venv/bin/python -m mcp_dj.analyze_track /path/to/track.mp3

# Analyze entire library (incremental — skips already-cached tracks)
./analyze-library.sh

# Force re-analyze all tracks
./analyze-library.sh --force

# Rebuild the track index only (after updating Rekordbox tags, no re-analysis)
.venv/bin/python -c "from mcp_dj.library_index import LibraryIndex; LibraryIndex().build(force=True)"
```

---

## Troubleshooting

### "Rekordbox database not found"

The system auto-detects Rekordbox at standard paths:
- macOS: `~/Library/Pioneer`
- Windows: `%APPDATA%\Pioneer`

If your installation is non-standard, set `REKORDBOX_DB_PATH` in `.env`. Also make sure Rekordbox 6 has been opened at least once — it creates the database on first launch.

### "essentia not found" or "TensorFlow error"

Essentia is the optional ML component. If it fails to install:

```bash
uv add essentia-tensorflow
```

If that doesn't work (Essentia is picky about platform/architecture), the system automatically falls back to metadata-only mode. Everything still works — you just don't get mood/genre ML features.

### "Permission denied" on shell scripts

```bash
chmod +x *.sh
```

### Python version error

```bash
python3 --version           # check system python
.venv/bin/python --version  # check venv python
```

If your system Python is older than 3.12, the master installer handles this automatically via `uv python install 3.12`. For manual install: follow Step 1 above.

### Library takes too long to analyze

The ML analysis is CPU-intensive. For very large libraries (5000+ tracks), use the `--limit` flag to batch it:

```bash
./analyze-library.sh --limit 500
```

Run multiple times until complete. Already-analyzed tracks are skipped each run.

### install-master.sh fails on Linux: "sudo not found"

Some minimal Linux installs don't have sudo. If the script can't install curl:

```bash
# As root:
apt-get install -y curl sudo   # Debian/Ubuntu
dnf install -y curl sudo       # RHEL/Fedora
```

Then re-run `install-master.sh` as your normal user.

---

## What's Next

Once everything is running:

- Read [Library Analysis](library-analysis.md) to understand what data the system reads from your tracks and why it matters for set building
- Read [Building Sets](building-sets.md) to learn how to write prompts and how the algorithm makes its decisions
- Read [Technical Reference](reference.md) for the full tool API and architecture details
