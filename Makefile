SHELL := /usr/bin/env bash

REPO_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: help \
	install-master install install-essentia install-essentia-skip-models \
	run-server run-mcp run-mcp-venv run-mcp-http \
	analyze-library analyze-library-force analyze-library-dry-run \
	build-library-index rebuild-index \
	analyze-track \
	download-models \
	test

help:
	@echo ""
	@echo "MCP DJ — Makefile commands"
	@echo "========================================"
	@echo "Setup & installation:"
	@echo "  make install-master              # Run master installer (no Python needed)"
	@echo "  make install                     # Standard install (Python + uv already present)"
	@echo "  make install-essentia            # Install with Essentia audio analysis"
	@echo "  make install-essentia-skip-models# Essentia but skip model download"
	@echo ""
	@echo "Servers:"
	@echo "  make run-server                  # Start FastAPI web UI (http://localhost:\$$SETLIST_PORT)"
	@echo "  make run-mcp                     # Start MCP server via uv (for Claude Desktop/Code)"
	@echo "  make run-mcp-venv                # Start MCP server via .venv python directly"
	@echo "  make run-mcp-http                # Start MCP server over HTTP (SSE transport)"
	@echo ""
	@echo "Library analysis & indexing:"
	@echo "  make analyze-library             # Analyze new/uncached tracks + rebuild index"
	@echo "       args='--force --workers 4'  # Extra flags passed through to analyze-library.sh"
	@echo "  make analyze-library-force       # Force re-analyze all tracks + rebuild index"
	@echo "  make analyze-library-dry-run     # Preview what would be analyzed"
	@echo "  make build-library-index         # Rebuild index from existing Essentia cache"
	@echo "  make rebuild-index               # Alias for build-library-index"
	@echo ""
	@echo "Audio analysis (single track):"
	@echo "  make analyze-track file=/path/to/song.mp3"
	@echo ""
	@echo "Models & tests:"
	@echo "  make download-models             # Download Essentia ML models (~300 MB)"
	@echo "  make test                        # Run pytest suite via uv"
	@echo ""

install-master:
	@cd "$(REPO_ROOT)" && ./install-master.sh $(args)

install:
	@cd "$(REPO_ROOT)" && ./install.sh $(args)

install-essentia:
	@cd "$(REPO_ROOT)" && ./install.sh --essentia $(args)

install-essentia-skip-models:
	@cd "$(REPO_ROOT)" && ./install.sh --essentia --skip-models $(args)

run-server:
	@cd "$(REPO_ROOT)" && ./run-server.sh

run-mcp:
	@cd "$(REPO_ROOT)" && ./run-mcp.sh

run-mcp-venv:
	@cd "$(REPO_ROOT)" && ./run_mcp.sh

run-mcp-http:
	@cd "$(REPO_ROOT)" && ./run-mcp-http.sh

analyze-library:
	@cd "$(REPO_ROOT)" && ./analyze-library.sh $(args)

analyze-library-force:
	@cd "$(REPO_ROOT)" && ./analyze-library.sh --force $(args)

analyze-library-dry-run:
	@cd "$(REPO_ROOT)" && ./analyze-library.sh --dry-run $(args)

build-library-index:
	@cd "$(REPO_ROOT)" && ./build-library-index.sh $(args)

rebuild-index: build-library-index

analyze-track:
	@if [ -z "$$file" ]; then \
		echo "Usage: make analyze-track file=/path/to/song.mp3"; \
		exit 1; \
	fi
	@cd "$(REPO_ROOT)" && uv run analyze-track "$$file"

download-models:
	@cd "$(REPO_ROOT)" && ./download_models.sh

test:
	@cd "$(REPO_ROOT)" && uv run pytest $(args)

