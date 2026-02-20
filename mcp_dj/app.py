"""
FastAPI Web Application for MCP DJ

Endpoints:
  GET  /                          - Serve the chat UI
  GET  /api/library/stats         - Library statistics
  GET  /api/library/tracks        - Search/list tracks
  POST /api/chat                  - Chat with AI
  POST /api/setlist/generate      - Direct setlist generation
  POST /api/setlist/recommend     - Next-track recommendations
  POST /api/setlist/export        - Export setlist to Rekordbox playlist
  GET  /api/setlist/{id}          - Retrieve a generated setlist
"""

import os
import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from .database import RekordboxDatabase
from .energy import MixedInKeyLibrary, EnergyResolver
from .camelot import CamelotWheel
from .energy_planner import EnergyPlanner, ENERGY_PROFILES
from .setlist_engine import SetlistEngine
from .ai_integration import SetlistAI
from .models import SetlistRequest
from .library_index import LibraryIndex

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

db = RekordboxDatabase()
energy_resolver = EnergyResolver()
camelot = CamelotWheel()
planner = EnergyPlanner()
engine = SetlistEngine(camelot=camelot, energy_planner=planner)
ai: Optional[SetlistAI] = None
library_index: Optional[LibraryIndex] = None
_mik_library_app: Optional[MixedInKeyLibrary] = None

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global ai, energy_resolver, library_index, _mik_library_app

    # Startup
    await db.connect()

    # Load Mixed In Key energy data (optional — requires MIK_CSV_PATH env var)
    mik = MixedInKeyLibrary.from_env()
    if mik is not None:
        mik.load()
    _mik_library_app = mik
    energy_resolver = EnergyResolver(mik)

    # Load all tracks and resolve energy
    all_tracks = await db.get_all_tracks()
    energy_resolver.resolve_all(all_tracks)

    # Initialize engine
    engine.initialize(all_tracks)

    # Build the centralized library index (merged JSONL for LLM grep + vector search)
    essentia_store_app = None
    try:
        from .essentia_analyzer import EssentiaFeatureStore
        essentia_store_app = EssentiaFeatureStore(all_tracks)
    except ImportError:
        pass

    library_index = LibraryIndex()
    if library_index.is_fresh(max_age_seconds=3600):
        count = library_index.load_from_disk()
        logger.info(f"Library index loaded from disk: {count} records")
    else:
        stats = library_index.build(
            tracks=all_tracks,
            essentia_store=essentia_store_app,
            mik_library=mik,
        )
        logger.info(
            f"Library index built: {stats['total']} tracks "
            f"({stats['with_essentia']} with Essentia, {stats['with_mik']} with MIK)"
        )

    # Initialize AI
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    ai = SetlistAI(engine=engine, api_key=api_key)
    if api_key:
        logger.info("Claude API key found. AI chat enabled.")
    else:
        logger.warning("No ANTHROPIC_API_KEY set. Using fallback mode (no AI chat).")

    logger.info(f"Setlist Creator ready. {len(all_tracks)} tracks loaded.")

    yield

    # Shutdown
    await db.disconnect()


app = FastAPI(title="MCP DJ", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/library/stats")
async def library_stats():
    """Library summary stats."""
    summary = engine.get_library_summary()
    summary["energy_profiles"] = {
        k: v["description"] for k, v in ENERGY_PROFILES.items()
    }
    return JSONResponse(summary)


@app.get("/api/library/tracks")
async def library_tracks(search: Optional[str] = None, limit: int = 100):
    """Search/list tracks from the library."""
    tracks = await db.search_tracks(query=search or "", limit=min(limit, 500))
    return JSONResponse([
        {
            "id": t.id,
            "title": t.title,
            "artist": t.artist,
            "genre": t.genre,
            "bpm": t.bpm,
            "key": t.key,
            "energy": t.energy,
            "energy_source": t.energy_source,
            "rating": t.rating,
            "play_count": t.play_count,
            "length": t.length,
            "duration": t.duration_formatted(),
            "color": t.color,
        }
        for t in tracks
    ])


@app.post("/api/library/rebuild-index")
async def rebuild_library_index_endpoint(force: bool = False):
    """Rebuild the centralized JSONL library index on demand.

    Query params:
        force: If true, rebuild even if the index is fresh (< 1 hour old).
    """
    import asyncio as _asyncio

    if library_index is None:
        raise HTTPException(status_code=503, detail="Library index not initialized")

    if not force and library_index.is_fresh(max_age_seconds=3600):
        return JSONResponse({
            "skipped": True,
            "reason": "Index is fresh (< 1 hour old). Use ?force=true to rebuild.",
            "index_path": str(library_index._record_path),
        })

    essentia_store_rebuild = None
    try:
        from .essentia_analyzer import EssentiaFeatureStore
        essentia_store_rebuild = EssentiaFeatureStore(engine.tracks)
    except ImportError:
        pass

    stats = await _asyncio.get_event_loop().run_in_executor(
        None,
        lambda: library_index.build(
            tracks=engine.tracks,
            essentia_store=essentia_store_rebuild,
            mik_library=_mik_library_app,
        ),
    )
    return JSONResponse(stats)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def chat(body: ChatRequest):
    """Chat with the AI DJ assistant."""
    if not ai:
        raise HTTPException(status_code=503, detail="AI not initialized")

    response = await ai.chat(body.message)
    result = {
        "role": response.role,
        "content": response.content,
        "timestamp": response.timestamp,
    }

    if response.setlist:
        result["setlist"] = response.setlist.model_dump()
    if response.recommendations:
        result["recommendations"] = [r.model_dump() for r in response.recommendations]

    return JSONResponse(result)


@app.post("/api/chat/clear")
async def clear_chat():
    """Clear conversation history."""
    if ai:
        ai.clear_history()
    return {"success": True}


# ---------------------------------------------------------------------------
# Direct setlist generation
# ---------------------------------------------------------------------------

@app.post("/api/setlist/generate")
async def generate_setlist(request: SetlistRequest):
    """Generate a setlist directly (without chat)."""
    setlist = engine.generate_setlist(request)
    return JSONResponse(setlist.model_dump())


@app.get("/api/setlist/{setlist_id}")
async def get_setlist(setlist_id: str):
    """Retrieve a previously generated setlist."""
    setlist = engine.get_setlist(setlist_id)
    if not setlist:
        raise HTTPException(status_code=404, detail="Setlist not found")
    return JSONResponse(setlist.model_dump())


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    track_id: Optional[str] = None
    track_title: Optional[str] = None
    energy_direction: str = "maintain"
    limit: int = 10


@app.post("/api/setlist/recommend")
async def recommend_next(body: RecommendRequest):
    """Get next-track recommendations."""
    recs = engine.recommend_next(
        current_track_id=body.track_id,
        current_track_title=body.track_title,
        energy_direction=body.energy_direction,
        limit=body.limit,
    )
    return JSONResponse([r.model_dump() for r in recs])


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class ExportRequest(BaseModel):
    setlist_id: str
    playlist_name: str


@app.post("/api/setlist/export")
async def export_to_rekordbox(body: ExportRequest):
    """Export a setlist to a Rekordbox playlist."""
    setlist = engine.get_setlist(body.setlist_id)
    if not setlist:
        raise HTTPException(status_code=404, detail="Setlist not found")

    track_ids = [st.track.id for st in setlist.tracks]
    try:
        playlist_id = await db.create_playlist_with_tracks(
            name=body.playlist_name,
            track_ids=track_ids,
        )
        return {"success": True, "playlist_id": playlist_id, "track_count": len(track_ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    port = int(os.environ.get("SETLIST_PORT", "8888"))
    logger.info(f"Starting MCP DJ on port {port}")
    uvicorn.run(
        "mcp_dj.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
