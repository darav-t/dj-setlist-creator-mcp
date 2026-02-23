#!/usr/bin/env bash
# =============================================================================
# MCP DJ — Master Installer
# =============================================================================
# One command to go from a completely fresh machine to a running mcp-dj setup.
# No Python, no uv, nothing required up-front.
#
# What this script does:
#   0. Check supported OS (macOS, Ubuntu/Debian, RHEL/Fedora/Arch)
#   1. Ensure curl is available (installs via package manager if missing)
#   2. Install uv — the Python package manager that bootstraps everything else
#   3. Install Python 3.12 via uv (no system Python required)
#   4. Run uv sync to create .venv with a working python3 binary
#   5. Hand off to install.sh for the rest:
#        - Essentia ML audio analysis (optional)
#        - ML model download (~300 MB, optional)
#        - Rekordbox database key setup
#        - .env configuration file
#        - Claude Code MCP server registration
#        - Claude Desktop MCP server registration
#        - Library analysis and index build (optional)
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/.../install-master.sh | bash
#   # or locally:
#   chmod +x install-master.sh && ./install-master.sh
#
#   Flags are passed through to install.sh:
#   ./install-master.sh --essentia          # include Essentia ML + models
#   ./install-master.sh --essentia --skip-models  # Essentia but skip model download
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${GREEN}[✓]${NC} $*"; }
warn()    { echo -e "${YELLOW}[!]${NC} $*"; }
error()   { echo -e "${RED}[✗]${NC} $*" >&2; }
step()    { echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${BOLD}$*${NC}"; echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

# ── banner ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}"
echo "  ███╗   ███╗ ██████╗██████╗       ██████╗      ██╗"
echo "  ████╗ ████║██╔════╝██╔══██╗      ██╔══██╗     ██║"
echo "  ██╔████╔██║██║     ██████╔╝█████╗██║  ██║     ██║"
echo "  ██║╚██╔╝██║██║     ██╔═══╝ ╚════╝██║  ██║██   ██║"
echo "  ██║ ╚═╝ ██║╚██████╗██║           ██████╔╝╚█████╔╝"
echo "  ╚═╝     ╚═╝ ╚═════╝╚═╝           ╚═════╝  ╚════╝ "
echo -e "${NC}"
echo -e "  ${BOLD}Master Installer${NC} — AI-powered DJ Setlist Builder"
echo ""
echo "  This script installs everything from scratch."
echo "  No Python or uv required to start."
echo ""
echo "  Press Ctrl+C at any time to cancel."
echo ""

# ── OS detection ─────────────────────────────────────────────────────────────
step "Detecting your operating system"

OS=""
PKG_MANAGER=""

if [[ "$OSTYPE" == "darwin"* ]]; then
  OS="macos"
  MACOS_VERSION=$(sw_vers -productVersion 2>/dev/null || echo "unknown")
  info "macOS $MACOS_VERSION"

elif [ -f /etc/os-release ]; then
  # shellcheck source=/dev/null
  source /etc/os-release
  OS_ID="${ID:-unknown}"
  OS_VERSION="${VERSION_ID:-}"

  case "$OS_ID" in
    ubuntu|debian|linuxmint|pop|elementary|zorin|kali)
      OS="debian"
      PKG_MANAGER="apt"
      info "Debian-based Linux: $PRETTY_NAME"
      ;;
    rhel|centos|rocky|almalinux|fedora|ol)
      OS="rhel"
      if command -v dnf &>/dev/null; then PKG_MANAGER="dnf"; else PKG_MANAGER="yum"; fi
      info "RHEL-based Linux: $PRETTY_NAME"
      ;;
    arch|manjaro|endeavouros|garuda)
      OS="arch"
      PKG_MANAGER="pacman"
      info "Arch-based Linux: $PRETTY_NAME"
      ;;
    *)
      OS="linux"
      warn "Unknown Linux distribution: $OS_ID — will try to proceed"
      ;;
  esac
else
  error "Unsupported operating system."
  error "Supported: macOS 12+, Ubuntu 20.04+, Debian 11+, RHEL/CentOS/Fedora/Rocky, Arch"
  error "For Windows, use: install-master.ps1"
  exit 1
fi

# Check for Apple Silicon vs Intel (macOS only)
if [ "$OS" = "macos" ]; then
  ARCH=$(uname -m)
  if [ "$ARCH" = "arm64" ]; then
    info "Apple Silicon (arm64) — native support"
  else
    info "Intel (x86_64)"
  fi
fi

# ── Step 0: Xcode Command Line Tools (macOS only) ────────────────────────────
if [ "$OS" = "macos" ]; then
  step "Step 0 — Xcode Command Line Tools (macOS)"

  if xcode-select -p &>/dev/null 2>&1; then
    info "Xcode Command Line Tools already installed: $(xcode-select -p)"
  else
    warn "Xcode Command Line Tools not found — installing..."
    echo ""
    echo "  A system dialog will appear asking you to install the Command Line Tools."
    echo "  Click 'Install' (not 'Get Xcode') and wait for it to finish."
    echo "  This provides git, curl, make and other essential tools."
    echo ""

    # Trigger the installer
    xcode-select --install 2>/dev/null || true

    echo "  Waiting for Xcode Command Line Tools installation to complete..."
    echo "  (This can take several minutes)"
    echo ""

    # Poll until installed
    MAX_WAIT=600  # 10 minutes
    WAITED=0
    while ! xcode-select -p &>/dev/null 2>&1; do
      sleep 5
      WAITED=$((WAITED + 5))
      printf "  ."
      if [ $WAITED -ge $MAX_WAIT ]; then
        echo ""
        error "Timed out waiting for Xcode Command Line Tools."
        error "Please install manually: xcode-select --install"
        error "Then re-run this script."
        exit 1
      fi
    done
    echo ""
    info "Xcode Command Line Tools installed"
  fi
fi

# ── Step 1: curl ──────────────────────────────────────────────────────────────
step "Step 1 — curl"

if command -v curl &>/dev/null; then
  info "curl already available: $(curl --version | head -1)"
else
  warn "curl not found — installing..."
  case "$PKG_MANAGER" in
    apt)
      sudo apt-get update -qq
      sudo apt-get install -y curl
      ;;
    dnf)  sudo dnf install -y curl ;;
    yum)  sudo yum install -y curl ;;
    pacman) sudo pacman -Sy --noconfirm curl ;;
    *)
      error "Cannot install curl automatically on this system."
      error "Please install curl manually and re-run this script."
      exit 1
      ;;
  esac
  info "curl installed"
fi

# ── Step 2: uv ───────────────────────────────────────────────────────────────
step "Step 2 — uv (Python package manager)"

# Ensure common uv install locations are in PATH
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

if command -v uv &>/dev/null; then
  UV_VERSION=$(uv --version 2>&1 | head -1)
  info "uv already installed: $UV_VERSION"
else
  warn "uv not found — installing..."
  echo ""
  echo "  uv is a fast Python package manager."
  echo "  It also installs Python itself — no separate Python download needed."
  echo "  Install location: ~/.local/bin/uv"
  echo ""

  if curl -LsSf https://astral.sh/uv/install.sh | sh; then
    # Re-source PATH additions
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

    if command -v uv &>/dev/null; then
      info "uv installed: $(uv --version 2>&1 | head -1)"
    else
      error "uv installed but not found in PATH."
      error "Add ~/.local/bin to your PATH and re-run this script."
      error ""
      error "  For bash:  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc && source ~/.bashrc"
      error "  For zsh:   echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc  && source ~/.zshrc"
      exit 1
    fi
  else
    error "uv installation failed."
    error "Try manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
  fi
fi

# ── Step 3: Python 3.12 ──────────────────────────────────────────────────────
step "Step 3 — Python 3.12"

echo "  uv manages Python versions independently of your system."
echo "  This doesn't modify any system Python installations."
echo ""

if uv python find 3.12 &>/dev/null 2>&1; then
  PYTHON_PATH=$(uv python find 3.12 2>/dev/null)
  info "Python 3.12 already available: $PYTHON_PATH"
else
  warn "Python 3.12 not found — installing via uv..."
  if uv python install 3.12; then
    info "Python 3.12 installed"
  else
    error "Python 3.12 installation failed."
    error "Try manually: uv python install 3.12"
    exit 1
  fi
fi

# Verify version
PYTHON_VERSION=$(uv run python3 --version 2>/dev/null || echo "unknown")
info "Python version: $PYTHON_VERSION"

# ── Step 4: Create virtual environment with working python3 ──────────────────
step "Step 4 — Creating Python virtual environment"

echo "  Creating .venv/ with Python 3.12..."
echo "  This is where all project packages will be installed."
echo ""

# Run uv sync to create .venv with core packages
# (install.sh will also run uv sync, but we need .venv/bin/python3
#  to exist before install.sh's inline python3 calls)
if uv sync --quiet 2>/dev/null || uv sync; then
  info ".venv created with core packages"
else
  error "uv sync failed — check pyproject.toml and try again"
  exit 1
fi

# Verify .venv/bin/python3 exists
if [ -f "$SCRIPT_DIR/.venv/bin/python3" ]; then
  VENV_PYTHON=$("$SCRIPT_DIR/.venv/bin/python3" --version 2>/dev/null)
  info "Virtual environment Python: $VENV_PYTHON"
else
  error ".venv/bin/python3 not found after uv sync"
  error "This is unexpected — please report this issue"
  exit 1
fi

# ── Inject .venv/bin into PATH so bare `python3` resolves correctly ───────────
# install.sh uses `python3` in several inline scripts. By putting .venv/bin
# first in PATH, those calls resolve to our managed Python instead of whatever
# (possibly absent) system python3.
export PATH="$SCRIPT_DIR/.venv/bin:$PATH"
info "PATH updated — python3 → $(command -v python3)"

# ── Step 5: Hand off to install.sh ───────────────────────────────────────────
step "Step 5 — Running main installer"

echo "  Python and uv are ready. Handing off to install.sh for:"
echo ""
echo "    • Essentia ML audio analysis (optional)"
echo "    • ML model download (~300 MB, optional)"
echo "    • Rekordbox database key configuration"
echo "    • .env file creation"
echo "    • Claude Code MCP server registration"
echo "    • Claude Desktop MCP server registration"
echo "    • Library analysis and index build (optional)"
echo ""

INSTALL_SCRIPT="$SCRIPT_DIR/install.sh"

if [ ! -f "$INSTALL_SCRIPT" ]; then
  error "install.sh not found at: $INSTALL_SCRIPT"
  error "Make sure you ran install-master.sh from the mcp-dj project directory."
  exit 1
fi

if [ ! -x "$INSTALL_SCRIPT" ]; then
  chmod +x "$INSTALL_SCRIPT"
fi

# Execute install.sh, passing through any flags (e.g. --essentia, --skip-models)
exec bash "$INSTALL_SCRIPT" "$@"
