# =============================================================================
# MCP DJ — Master Installer for Windows
# =============================================================================
# One command to go from a completely fresh Windows machine to a running
# mcp-dj setup. No Python, no uv, nothing required up-front.
#
# What this script does:
#   0. Check PowerShell version + execution policy
#   1. Install Python 3.12 via winget (built into Windows 10 1809+ / 11)
#   2. Install uv — Python package manager
#   3. Install core Python packages via uv sync
#   4. Optionally install Essentia ML audio analysis
#   5. Optionally download ML models (~300 MB)
#   6. Set up Rekordbox database key via pyrekordbox
#   7. Create .env configuration file
#   8. Register MCP server in Claude Desktop
#   9. Optionally analyze and index Rekordbox library
#
# Usage (run in PowerShell as your normal user — no Admin required):
#
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
#   .\install-master.ps1
#
#   With Essentia ML analysis enabled:
#   .\install-master.ps1 -Essentia
#
#   Essentia but skip model download:
#   .\install-master.ps1 -Essentia -SkipModels
# =============================================================================

param(
    [switch]$Essentia,
    [switch]$SkipModels,
    [switch]$Help
)

if ($Help) {
    Write-Host ""
    Write-Host "Usage: .\install-master.ps1 [-Essentia] [-SkipModels]"
    Write-Host ""
    Write-Host "  -Essentia     Install Essentia ML audio analysis + download ML models"
    Write-Host "                Enables: BPM, key, mood, genre, danceability analysis"
    Write-Host ""
    Write-Host "  -SkipModels   Install Essentia but skip model download (~300 MB)"
    Write-Host "                Download later with: .\download_models.sh (in WSL or Git Bash)"
    Write-Host ""
    exit 0
}

# ── Helpers ───────────────────────────────────────────────────────────────────
function Write-Step  { param($msg) Write-Host "`n$("="*64)`n  $msg`n$("="*64)" -ForegroundColor Cyan }
function Write-Ok    { param($msg) Write-Host "  [OK]  $msg" -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "  [!]   $msg" -ForegroundColor Yellow }
function Write-Fail  { param($msg) Write-Host "  [ERR] $msg" -ForegroundColor Red }
function Write-Info  { param($msg) Write-Host "        $msg" }
function Ask-YesNo   {
    param($prompt, $default = "N")
    $suffix = if ($default -eq "Y") { "[Y/n]" } else { "[y/N]" }
    $reply = Read-Host "  $prompt $suffix"
    if ([string]::IsNullOrWhiteSpace($reply)) { $reply = $default }
    return $reply -match "^[Yy]"
}

# ── Banner ────────────────────────────────────────────────────────────────────
Clear-Host
Write-Host ""
Write-Host "  ==============================================" -ForegroundColor Cyan
Write-Host "    MCP-DJ  Master Installer  (Windows)" -ForegroundColor Cyan
Write-Host "    AI-powered DJ Setlist Builder" -ForegroundColor Cyan
Write-Host "  ==============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  This script installs everything from scratch."
Write-Host "  No Python or package manager required to start."
Write-Host ""
Write-Host "  Press Ctrl+C at any time to cancel."
Write-Host ""

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# ── Step 0: PowerShell version + execution policy ─────────────────────────────
Write-Step "Step 0 - Checking PowerShell version and execution policy"

$PSMajor = $PSVersionTable.PSVersion.Major
Write-Ok "PowerShell $($PSVersionTable.PSVersion)"

if ($PSMajor -lt 5) {
    Write-Fail "PowerShell 5.0 or newer is required."
    Write-Fail "Please update Windows PowerShell or install PowerShell 7:"
    Write-Fail "  https://aka.ms/powershell"
    exit 1
}

# Check execution policy
$ExecPolicy = Get-ExecutionPolicy -Scope CurrentUser
if ($ExecPolicy -eq "Restricted") {
    Write-Warn "Execution policy is Restricted. Updating to RemoteSigned for current user..."
    try {
        Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
        Write-Ok "Execution policy set to RemoteSigned"
    } catch {
        Write-Fail "Could not update execution policy. Run this first:"
        Write-Fail "  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned"
        exit 1
    }
} else {
    Write-Ok "Execution policy: $ExecPolicy"
}

# ── Step 1: Python 3.12 ───────────────────────────────────────────────────────
Write-Step "Step 1 - Python 3.12"

function Test-PythonVersion {
    # Check for Python 3.12+ anywhere in PATH
    try {
        $ver = & python --version 2>&1
        if ($ver -match "Python 3\.1[2-9]") { return $true }
    } catch {}
    try {
        $ver = & python3 --version 2>&1
        if ($ver -match "Python 3\.1[2-9]") { return $true }
    } catch {}
    return $false
}

if (Test-PythonVersion) {
    Write-Ok "Python 3.12+ already installed"
} else {
    Write-Warn "Python 3.12+ not found"
    Write-Info ""

    # Try winget first (Windows 10 1809+ / Windows 11)
    $WingetAvailable = $false
    try {
        $null = & winget --version 2>&1
        $WingetAvailable = $true
    } catch {}

    if ($WingetAvailable) {
        Write-Info "Installing Python 3.12 via winget..."
        Write-Info "(Windows Package Manager — built into Windows 10/11)"
        Write-Info ""
        try {
            & winget install --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements --silent
            # Refresh PATH to include newly installed Python
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + `
                        [System.Environment]::GetEnvironmentVariable("Path", "User")
            Write-Ok "Python 3.12 installed via winget"
        } catch {
            Write-Warn "winget install failed: $_"
            Write-Warn "Falling back to direct download..."
            $WingetAvailable = $false
        }
    }

    if (-not $WingetAvailable) {
        # Direct download fallback
        Write-Info "Downloading Python 3.12 installer..."
        $PythonUrl = "https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe"
        $PythonInstaller = "$env:TEMP\python-3.12-installer.exe"
        Write-Info "  From: $PythonUrl"
        Write-Info ""

        try {
            Invoke-WebRequest -Uri $PythonUrl -OutFile $PythonInstaller -UseBasicParsing
            Write-Info "Running Python installer..."
            Write-Info "  Installing for current user (no admin required)"
            Write-Info ""
            Start-Process -FilePath $PythonInstaller `
                -ArgumentList "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_test=0" `
                -Wait -NoNewWindow
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + `
                        [System.Environment]::GetEnvironmentVariable("Path", "User")
            Remove-Item $PythonInstaller -Force -ErrorAction SilentlyContinue
            Write-Ok "Python 3.12 installed from python.org"
        } catch {
            Write-Fail "Python installation failed: $_"
            Write-Fail ""
            Write-Fail "Please install Python 3.12 manually:"
            Write-Fail "  https://www.python.org/downloads/release/python-3128/"
            Write-Fail "  Check 'Add Python to PATH' during installation."
            Write-Fail ""
            Write-Fail "Then re-run this script."
            exit 1
        }
    }

    # Final check
    if (-not (Test-PythonVersion)) {
        Write-Fail "Python 3.12 was installed but is not found in PATH."
        Write-Fail "Please restart PowerShell and re-run this script."
        exit 1
    }
    Write-Ok "Python 3.12 confirmed in PATH"
}

# ── Step 2: uv ────────────────────────────────────────────────────────────────
Write-Step "Step 2 - uv (Python package manager)"

# Add common uv locations to PATH
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:USERPROFILE\.local\bin;$env:Path"

$UvAvailable = $false
try {
    $UvVer = & uv --version 2>&1
    $UvAvailable = $true
    Write-Ok "uv already installed: $UvVer"
} catch {}

if (-not $UvAvailable) {
    Write-Warn "uv not found — installing..."
    Write-Info ""
    Write-Info "uv handles all Python package installation."
    Write-Info "Install location: $env:USERPROFILE\.local\bin\uv.exe"
    Write-Info ""
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + `
                    [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + `
                    "$env:USERPROFILE\.local\bin"
        $UvVer = & uv --version 2>&1
        Write-Ok "uv installed: $UvVer"
    } catch {
        Write-Fail "uv installation failed: $_"
        Write-Fail "Install manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    }
}

# ── Step 3: Core Python packages ──────────────────────────────────────────────
Write-Step "Step 3 - Core Python packages"

Write-Info "  pyrekordbox  — reads your Rekordbox library"
Write-Info "  pydantic     — data models"
Write-Info "  fastapi      — web server"
Write-Info "  uvicorn      — ASGI server"
Write-Info "  anthropic    — Claude API"
Write-Info "  fastmcp      — MCP server for Claude Desktop"
Write-Info ""

try {
    & uv sync
    Write-Ok "Core packages installed"
} catch {
    Write-Fail "uv sync failed: $_"
    Write-Fail "Check pyproject.toml and try: uv sync"
    exit 1
}

# Verify .venv was created
$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Fail ".venv\Scripts\python.exe not found after uv sync"
    exit 1
}
$VenvVer = & $VenvPython --version 2>&1
Write-Ok "Virtual environment ready: $VenvVer"

# ── Step 4: Essentia ML audio analysis ────────────────────────────────────────
Write-Step "Step 4 - Essentia ML audio analysis"

$InstallEssentia = $Essentia.IsPresent

if (-not $InstallEssentia) {
    Write-Info "Essentia analyzes your actual audio to extract:"
    Write-Info "  * Accurate BPM + beat confidence"
    Write-Info "  * Key detection (Camelot wheel)"
    Write-Info "  * Danceability scoring"
    Write-Info "  * EBU R128 loudness (LUFS + RMS)"
    Write-Info "  * Mood: happy / sad / aggressive / relaxed / party"
    Write-Info "  * Genre (Discogs400 — 400 music styles)"
    Write-Info "  * Music tags: techno, beat, electronic, etc."
    Write-Info ""
    Write-Info "  Package size: ~135 MB + ~300 MB ML models"
    Write-Info ""
    $InstallEssentia = Ask-YesNo "Install Essentia for song analysis?"
}

if ($InstallEssentia) {
    Write-Info ""
    Write-Info "Installing essentia-tensorflow (~135 MB)..."
    Write-Info ""
    try {
        & uv sync --extra essentia
        Write-Ok "essentia-tensorflow installed"
    } catch {
        Write-Warn "uv sync --extra failed, trying uv add..."
        try {
            & uv add essentia-tensorflow
            Write-Ok "essentia-tensorflow installed"
        } catch {
            Write-Warn "essentia-tensorflow installation failed: $_"
            Write-Warn "Try manually: uv add essentia-tensorflow"
            Write-Warn "Essentia is optional — continuing without it."
            $InstallEssentia = $false
        }
    }
} else {
    Write-Warn "Skipping Essentia — install later with: .\install.sh --essentia"
}

# ── Step 5: ML models ─────────────────────────────────────────────────────────
Write-Step "Step 5 - ML model files"

$ModelDir = Join-Path $ScriptDir ".data\models"
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null

if ($InstallEssentia) {
    if ($SkipModels.IsPresent) {
        Write-Warn "Skipping model download (-SkipModels)"
        Write-Info "Download later with: .\download_models.sh (in Git Bash / WSL)"
    } else {
        # Count existing .pb files
        $ExistingModels = (Get-ChildItem -Path $ModelDir -Filter "*.pb" -ErrorAction SilentlyContinue).Count

        if ($ExistingModels -ge 20) {
            Write-Ok "All ML models already present ($ExistingModels .pb files)"
        } else {
            Write-Info "Downloading ML models (~300 MB)..."
            Write-Info "  Models: VGGish, EffNet, mood classifiers, genre Discogs400, MagnaTagATune"
            Write-Info ""

            $BaseUrl = "https://essentia.upf.edu/models"
            $TotalFiles = 0
            $FailedFiles = 0

            function Download-Model {
                param($Url)
                $FileName = Split-Path $Url -Leaf
                $Dest = Join-Path $ModelDir $FileName
                if (Test-Path $Dest) {
                    Write-Ok "Already exists: $FileName"
                    return
                }
                Write-Warn "Downloading: $FileName"
                try {
                    Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
                    Write-Ok "Downloaded: $FileName"
                    $script:TotalFiles++
                } catch {
                    Write-Fail "Failed: $FileName — $_"
                    Remove-Item $Dest -Force -ErrorAction SilentlyContinue
                    $script:FailedFiles++
                }
            }

            # Embedding models
            Write-Info "Embedding models..."
            Download-Model "$BaseUrl/feature-extractors/vggish/audioset-vggish-3.pb"
            Download-Model "$BaseUrl/feature-extractors/vggish/audioset-vggish-3.json"
            Download-Model "$BaseUrl/feature-extractors/musicnn/msd-musicnn-1.pb"
            Download-Model "$BaseUrl/feature-extractors/musicnn/msd-musicnn-1.json"
            Download-Model "$BaseUrl/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb"
            Download-Model "$BaseUrl/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.json"

            # Mood classifiers
            Write-Info "Mood classifiers..."
            foreach ($Mood in @("mood_happy", "mood_sad", "mood_aggressive", "mood_relaxed", "mood_party")) {
                Download-Model "$BaseUrl/classification-heads/$Mood/$Mood-audioset-vggish-1.pb"
                Download-Model "$BaseUrl/classification-heads/$Mood/$Mood-audioset-vggish-1.json"
            }

            # MagnaTagATune
            Write-Info "Music autotagging..."
            Download-Model "$BaseUrl/classification-heads/mtt/mtt-discogs-effnet-1.pb"
            Download-Model "$BaseUrl/classification-heads/mtt/mtt-discogs-effnet-1.json"

            # Genre Discogs400
            Write-Info "Genre classification..."
            Download-Model "$BaseUrl/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb"
            Download-Model "$BaseUrl/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json"

            if ($FailedFiles -gt 0) {
                Write-Warn "$FailedFiles model(s) failed to download. Re-run later: .\download_models.sh"
            } else {
                Write-Ok "All ML models downloaded to .data\models\"
            }
        }
    }
} else {
    Write-Warn "Skipping ML models (Essentia not installed)"
}

# ── Step 6: Rekordbox database key ────────────────────────────────────────────
Write-Step "Step 6 - Rekordbox database access"

Write-Info "Checking pyrekordbox configuration..."

$PyrekordboxConfigured = $false
try {
    $ConfigOutput = & $VenvPython -c "import pyrekordbox; pyrekordbox.show_config()" 2>&1
    if ($ConfigOutput -match "app_password|db_key") {
        $PyrekordboxConfigured = $true
        Write-Ok "pyrekordbox already configured"
    }
} catch {}

if (-not $PyrekordboxConfigured) {
    Write-Info ""
    Write-Info "pyrekordbox needs the Rekordbox master.db decryption key."
    Write-Info "This is read automatically from your Rekordbox installation."
    Write-Info "Rekordbox 6 must be installed on this machine."
    Write-Info ""

    if (Ask-YesNo "Run pyrekordbox setup now?" "Y") {
        try {
            & $VenvPython -m pyrekordbox setup-db
            Write-Ok "Rekordbox database key configured"
        } catch {
            Write-Warn "Setup failed: $_"
            Write-Warn "Run manually once Rekordbox is installed:"
            Write-Warn "  .\.venv\Scripts\python.exe -m pyrekordbox setup-db"
        }
    } else {
        Write-Warn "Skipped — run before starting the server:"
        Write-Warn "  .\.venv\Scripts\python.exe -m pyrekordbox setup-db"
    }
}

# ── Step 7: .env file ─────────────────────────────────────────────────────────
Write-Step "Step 7 - Environment configuration (.env)"

$EnvFile = Join-Path $ScriptDir ".env"
$EnvExample = Join-Path $ScriptDir ".env.example"

if (Test-Path $EnvFile) {
    Write-Ok ".env already exists"
} else {
    if (Test-Path $EnvExample) {
        Copy-Item $EnvExample $EnvFile
        Write-Ok ".env created from .env.example"
    } else {
        @"
# Anthropic API key — required for AI chat features
# Get one at: https://console.anthropic.com/
ANTHROPIC_API_KEY=

# Web server port (default: 8888)
# SETLIST_PORT=8888

# Optional: path to your Mixed In Key Library.csv for energy ratings
# MIK_CSV_PATH=C:\Users\you\Documents\Mixed In Key Data\Library.csv

# Optional: override Rekordbox database path (auto-detected by default)
# REKORDBOX_DB_PATH=C:\Users\you\AppData\Roaming\Pioneer
"@ | Set-Content $EnvFile
        Write-Ok ".env created"
    }
    Write-Info ""
    Write-Warn "ACTION REQUIRED: Add your Anthropic API key to .env"
    Write-Info "  Open .env in any text editor and set:"
    Write-Info "  ANTHROPIC_API_KEY=sk-ant-..."
    Write-Info ""
    Write-Info "  Get a key at: https://console.anthropic.com/"
}

# Create required .data subdirectories
New-Item -ItemType Directory -Force -Path (Join-Path $ScriptDir ".data\essentia_cache") | Out-Null

# ── Step 8: Claude Desktop config ─────────────────────────────────────────────
Write-Step "Step 8 - Claude Desktop MCP server registration"

$ClaudeConfigPath = Join-Path $env:APPDATA "Claude\claude_desktop_config.json"
$ClaudeConfigDir  = Split-Path $ClaudeConfigPath -Parent

New-Item -ItemType Directory -Force -Path $ClaudeConfigDir | Out-Null

# Read existing config
$Config = @{}
if (Test-Path $ClaudeConfigPath) {
    try {
        $Config = Get-Content $ClaudeConfigPath -Raw | ConvertFrom-Json -AsHashtable
    } catch {
        Write-Warn "Could not parse existing claude_desktop_config.json — will create a new one"
        $Config = @{}
    }
}

# Check if already registered
$AlreadyRegistered = $false
if ($Config.ContainsKey("mcpServers") -and $Config["mcpServers"].ContainsKey("mcp-dj")) {
    $AlreadyRegistered = $true
}

if ($AlreadyRegistered) {
    Write-Ok "MCP server 'mcp-dj' already registered in Claude Desktop"
} else {
    # Build env block from .env
    $McpEnv = @{}
    if (Test-Path $EnvFile) {
        foreach ($Line in Get-Content $EnvFile) {
            $Line = $Line.Trim()
            if ($Line -match "^#" -or [string]::IsNullOrWhiteSpace($Line)) { continue }
            if ($Line -match "^([A-Z_][A-Z0-9_]*)=(.+)$") {
                $Key = $Matches[1]
                $Val = $Matches[2].Trim('"').Trim("'")
                if (-not [string]::IsNullOrWhiteSpace($Val)) {
                    $McpEnv[$Key] = $Val
                }
            }
        }
    }

    # Build the mcp-dj server entry
    $ProjectDirWin = $ScriptDir -replace "/", "\"
    $McpEntry = @{
        command = "cmd"
        args    = @("/c", "cd /d `"$ProjectDirWin`" && .venv\Scripts\python.exe -m mcp_dj.mcp_server")
    }
    if ($McpEnv.Count -gt 0) {
        $McpEntry["env"] = $McpEnv
    }

    if (-not $Config.ContainsKey("mcpServers")) {
        $Config["mcpServers"] = @{}
    }
    $Config["mcpServers"]["mcp-dj"] = $McpEntry

    try {
        $Config | ConvertTo-Json -Depth 10 | Set-Content $ClaudeConfigPath
        Write-Ok "MCP server 'mcp-dj' added to Claude Desktop config"
        Write-Info "  Config: $ClaudeConfigPath"
        Write-Info "  Restart Claude Desktop to load the new server."
    } catch {
        Write-Warn "Failed to update Claude Desktop config: $_"
        Write-Warn "Add manually to: $ClaudeConfigPath"
        Write-Info "  See: claude_desktop_config.json in the project folder for the entry."
    }
}

# ── Step 9: Analyze library ───────────────────────────────────────────────────
if ($InstallEssentia) {
    Write-Step "Step 9 - Analyze and index Rekordbox library"

    Write-Info "  This step analyzes your audio files and builds a searchable index."
    Write-Info "  It does two things:"
    Write-Info ""
    Write-Info "  1. Essentia analysis — extracts from every song:"
    Write-Info "       BPM + beat confidence, key (Camelot), danceability"
    Write-Info "       EBU R128 loudness, mood, genre (Discogs400), music tags"
    Write-Info ""
    Write-Info "  2. Library index build — merges all data sources into:"
    Write-Info "       .data\library_index.jsonl       (one record per track)"
    Write-Info "       .data\library_attributes.json   (dynamic tag/genre/BPM summary)"
    Write-Info "       .data\library_context.md        (LLM context file)"
    Write-Info ""
    Write-Info "  Analysis is CPU-intensive. Stop with Ctrl+C and resume later."
    Write-Info "  Already-analyzed tracks are skipped automatically."
    Write-Info ""

    if (Ask-YesNo "Analyze and index your Rekordbox library now?") {
        # Suggest worker count
        $CpuCount = [Environment]::ProcessorCount
        $DefaultWorkers = [Math]::Max(1, [Math]::Min(4, [Math]::Floor($CpuCount / 2)))

        Write-Info ""
        $WorkersInput = Read-Host "  Parallel workers? (default: $DefaultWorkers)"
        if ([string]::IsNullOrWhiteSpace($WorkersInput)) { $WorkersInput = $DefaultWorkers }

        Write-Info ""
        try {
            & $VenvPython -m mcp_dj.analyze_library --workers $WorkersInput
        } catch {
            Write-Warn "Library analysis failed: $_"
            Write-Warn "Run manually later: .\.venv\Scripts\python.exe -m mcp_dj.analyze_library"
        }
    } else {
        Write-Warn "Skipping — run later with:"
        Write-Warn "  .\.venv\Scripts\python.exe -m mcp_dj.analyze_library"
    }
}

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host ""
Write-Host "  $("="*62)" -ForegroundColor Green
Write-Host "    Installation complete!" -ForegroundColor Green
Write-Host "  $("="*62)" -ForegroundColor Green
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. Add your Anthropic API key to .env:"
Write-Host "       ANTHROPIC_API_KEY=sk-ant-..." -ForegroundColor Yellow
Write-Host "     Get one at: https://console.anthropic.com/"
Write-Host ""
Write-Host "  2. Start the web UI:"
Write-Host "       .\.venv\Scripts\python.exe -m uvicorn mcp_dj.app:app --port 8888"
Write-Host "       Then open: http://localhost:8888" -ForegroundColor Cyan
Write-Host ""
Write-Host "  3. Connect Claude Desktop:"
Write-Host "       Restart Claude Desktop — mcp-dj should appear in the tools panel"
Write-Host ""

if ($InstallEssentia) {
    Write-Host "  4. Analyze a single track:"
    Write-Host "       .\.venv\Scripts\python.exe -m mcp_dj.analyze_track C:\path\to\song.mp3"
    Write-Host ""
    Write-Host "  5. Re-analyze library after adding new tracks:"
    Write-Host "       .\.venv\Scripts\python.exe -m mcp_dj.analyze_library"
    Write-Host ""
} else {
    Write-Host "  4. Enable audio analysis (optional, recommended):"
    Write-Host "       .\install-master.ps1 -Essentia" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "  Documentation: docs\README.md" -ForegroundColor Cyan
Write-Host ""
