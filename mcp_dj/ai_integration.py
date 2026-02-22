"""
Claude API Integration for Chat-Based Setlist Generation

Uses the Anthropic SDK with tool-calling to let Claude invoke the setlist
engine, search the library, recommend tracks, and build sets from natural
language prompts — including MyTag-aware candidate filtering.
"""

import json
import os
from typing import Any, Dict, List, Optional

from loguru import logger

from .models import (
    ChatMessage,
    Setlist,
    SetlistRequest,
    NextTrackRecommendation,
)
from .setlist_engine import SetlistEngine
from .energy_planner import EnergyPlanner, ENERGY_PROFILES
from .intent import parse_set_intent, interpret_vibe

MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Tool definitions for Claude
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "build_set_from_prompt",
        "description": (
            "Build a complete DJ set from a single natural language prompt. "
            "Automatically detects My Tags, genre, BPM range, and energy arc. "
            "Filters candidates using Rekordbox My Tags for best results. "
            "Use this as the primary tool for ANY setlist request."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Full natural language set description. Include context like "
                        "venue, situation, vibe, duration, genre, energy, and any "
                        "My Tag names you want included. Examples: "
                        "'60 minute tech house festival set', "
                        "'dark afters set building to peak', "
                        "'sunset melodic set with vocals, 90 minutes'."
                    ),
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "Set duration in minutes (default 60). Overridden if the prompt includes a duration.",
                    "default": 60,
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "plan_set",
        "description": (
            "Build a setlist from structured vibe/context parameters rather than a "
            "free-form prompt. Use when the user describes the situation in separate "
            "components (venue, crowd, time of day, etc.)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "duration_minutes": {
                    "type": "integer",
                    "description": "Set length in minutes (default 60)",
                    "default": 60,
                },
                "vibe": {
                    "type": "string",
                    "description": "Overall feel — e.g. 'après ski', 'underground warehouse', 'rooftop sundowner'",
                },
                "situation": {
                    "type": "string",
                    "description": "What's happening — e.g. 'warm-up', 'peak time', 'closing', 'after party'",
                },
                "venue": {
                    "type": "string",
                    "description": "Where you're playing — e.g. 'nightclub', 'festival stage', 'beach bar'",
                },
                "crowd_energy": {
                    "type": "string",
                    "description": "How the crowd feels — e.g. 'casual', 'hyped', 'just arriving'",
                },
                "time_of_day": {
                    "type": "string",
                    "description": "When the set is — e.g. 'afternoon', 'evening', 'late night', 'sunrise'",
                },
                "genre_preference": {
                    "type": "string",
                    "description": "Optional genre override — e.g. 'tech house', 'techno', 'melodic house'",
                },
            },
        },
    },
    {
        "name": "generate_setlist",
        "description": (
            "Generate a setlist with explicit technical parameters. "
            "Use when the user specifies exact BPM range, genre, energy profile, "
            "starting track, or My Tag filters directly. "
            "For natural language requests, prefer build_set_from_prompt instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "duration_minutes": {
                    "type": "integer",
                    "description": "Target set duration in minutes (default 60)",
                    "default": 60,
                },
                "genre": {
                    "type": "string",
                    "description": "Genre filter (e.g. 'tech house', 'techno'). Optional.",
                },
                "bpm_min": {
                    "type": "number",
                    "description": "Minimum BPM filter. Optional.",
                },
                "bpm_max": {
                    "type": "number",
                    "description": "Maximum BPM filter. Optional.",
                },
                "energy_profile": {
                    "type": "string",
                    "enum": ["journey", "build", "peak", "chill", "wave"],
                    "description": "Energy arc profile (default 'journey')",
                    "default": "journey",
                },
                "my_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Rekordbox My Tag names to filter candidates by "
                        "(e.g. ['Festival', 'High Energy']). Optional."
                    ),
                },
                "starting_track_title": {
                    "type": "string",
                    "description": "Title of the track to start with. Optional.",
                },
            },
        },
    },
    {
        "name": "recommend_next_track",
        "description": (
            "Recommend what to play next after a specific track based on harmonic "
            "compatibility, energy flow, and BPM proximity. "
            "Use when the user asks 'what should I play after X?'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "current_track_query": {
                    "type": "string",
                    "description": "Title or 'Artist - Title' of the currently playing track",
                },
                "energy_direction": {
                    "type": "string",
                    "enum": ["up", "down", "maintain"],
                    "description": "Whether to raise, lower, or maintain energy (default 'maintain')",
                    "default": "maintain",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of recommendations to return (default 5)",
                    "default": 5,
                },
            },
            "required": ["current_track_query"],
        },
    },
    {
        "name": "search_library",
        "description": (
            "Search the DJ's Rekordbox library by text, date added, or My Tag. "
            "Use to look up specific tracks, artists, genres, recently added tracks, "
            "or tracks tagged with a specific My Tag label."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text search (matches title, artist, album, genre). Optional.",
                },
                "date_from": {
                    "type": "string",
                    "description": "Filter tracks added on/after this date (YYYY-MM-DD). Optional.",
                },
                "date_to": {
                    "type": "string",
                    "description": "Filter tracks added on/before this date (YYYY-MM-DD). Optional.",
                },
                "my_tag": {
                    "type": "string",
                    "description": "Filter by Rekordbox My Tag label (e.g. 'High Energy', 'Festival'). Optional.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 20)",
                    "default": 20,
                },
            },
        },
    },
    {
        "name": "get_library_attributes",
        "description": (
            "Get detailed library statistics: My Tag hierarchy with per-tag BPM/energy/mood, "
            "genre breakdown with BPM ranges, key distribution, top artists, and more. "
            "Call this when you need to understand what's in the library before building a set."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_energy_advice",
        "description": (
            "Get advice on energy direction at a specific point in a set. "
            "Use when the user asks about energy flow or crowd management."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "current_energy": {
                    "type": "integer",
                    "description": "Current energy level (1-10)",
                },
                "position_pct": {
                    "type": "number",
                    "description": "How far through the set (0.0 = start, 1.0 = end)",
                },
                "profile": {
                    "type": "string",
                    "enum": ["journey", "build", "peak", "chill", "wave"],
                    "default": "journey",
                },
            },
            "required": ["current_energy", "position_pct"],
        },
    },
]


class SetlistAI:
    """
    Integrates Claude API for natural language setlist generation.

    Architecture:
    - User sends a chat message
    - We build a system prompt with rich library context (MyTags, genres, BPM stats)
    - Claude responds, optionally calling tools
    - We execute tool calls (including full build_set_from_prompt pipeline)
    - We return Claude's response + any generated setlist/recommendations
    """

    def __init__(
        self,
        engine: SetlistEngine,
        api_key: Optional[str] = None,
        library_index=None,
        library_attributes: Optional[Dict[str, Any]] = None,
        mik_library=None,
    ):
        self.engine = engine
        self.planner = EnergyPlanner()
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._library_index = library_index
        self._library_attributes = library_attributes
        self._mik_library = mik_library
        self._conversation_history: List[Dict[str, Any]] = []
        self._last_setlist: Optional[Setlist] = None
        self._last_recommendations: Optional[List[NextTrackRecommendation]] = None

    def update_library_context(
        self,
        library_index=None,
        library_attributes: Optional[Dict[str, Any]] = None,
        mik_library=None,
    ) -> None:
        """Refresh library references after an index rebuild or Essentia analysis."""
        if library_index is not None:
            self._library_index = library_index
        if library_attributes is not None:
            self._library_attributes = library_attributes
        if mik_library is not None:
            self._mik_library = mik_library

    def _get_client(self):
        """Lazy-init the Anthropic client."""
        import anthropic
        return anthropic.Anthropic(api_key=self._api_key)

    def _build_system_prompt(self) -> str:
        attrs = self._library_attributes
        stats = self.engine.get_library_summary()
        profiles = ", ".join(
            f"{k} ({v['description']})" for k, v in ENERGY_PROFILES.items()
        )

        # --- Library overview ---
        total = stats.get("total", 0)
        bpm_min = stats.get("bpm_min", 0)
        bpm_max = stats.get("bpm_max", 0)
        bpm_avg = stats.get("bpm_avg", 0)

        # Rich MyTag context from attributes (if available)
        my_tag_section = ""
        genre_section = ""
        if attrs:
            # MyTag hierarchy — list all tags grouped by parent
            hierarchy = attrs.get("my_tag_hierarchy", {})
            if hierarchy:
                lines = []
                for group, tags in hierarchy.items():
                    lines.append(f"  {group}: {', '.join(tags)}")
                my_tag_section = "MY TAG HIERARCHY (use these exact names in build_set_from_prompt):\n" + "\n".join(lines)

            # Genre stats — show genre + track count + BPM range
            genre_details = attrs.get("genre_details", {})
            if genre_details:
                top_genres = sorted(
                    genre_details.items(),
                    key=lambda kv: kv[1].get("count", 0),
                    reverse=True,
                )[:10]
                lines = []
                for gname, ginfo in top_genres:
                    bpm = ginfo.get("bpm", {})
                    bpm_str = (
                        f"{bpm.get('p25', bpm.get('min', '?')):.0f}–{bpm.get('p75', bpm.get('max', '?')):.0f} BPM"
                        if bpm else ""
                    )
                    lines.append(f"  {gname}: {ginfo.get('count', 0)} tracks {bpm_str}".rstrip())
                genre_section = "TOP GENRES:\n" + "\n".join(lines)

            # Top artists
            top_artists = attrs.get("top_artists", [])
            if top_artists:
                artist_names = [a["artist"] for a in top_artists[:15]]
                genre_section += f"\n\nTOP ARTISTS: {', '.join(artist_names)}"
        else:
            # Fallback to basic stats
            top_genres = stats.get("top_genres", [])
            genre_section = f"TOP GENRES: {', '.join(top_genres[:8])}"
            my_tag_section = f"MY TAGS: {', '.join(stats.get('top_my_tags', []))}"

        return f"""You are an expert DJ assistant helping create setlists from the user's Rekordbox library.
You understand harmonic mixing (Camelot wheel), energy flow, and genre coherence.

LIBRARY: {total} tracks | BPM {bpm_min:.0f}–{bpm_max:.0f} (avg {bpm_avg:.0f})

{my_tag_section}

{genre_section}

ENERGY PROFILES: {profiles}

TOOLS — choose the right one:
- build_set_from_prompt: PRIMARY tool for any setlist request. Paste the user's full context as the prompt. It auto-detects My Tags, genre, BPM range, and energy arc using the actual library data.
- plan_set: Use when context is naturally split into vibe / venue / situation / crowd / time.
- generate_setlist: Use only when the user explicitly specifies technical parameters (exact BPM, genre, energy profile).
- search_library: Look up tracks before or after building a set.
- get_library_attributes: Call this for the full detailed breakdown (per-tag BPM/mood/genre stats).
- recommend_next_track: What to play after a specific track.
- get_energy_advice: Energy flow advice for a point in a set.

HARMONIC MIXING (Camelot Wheel):
- Same key = perfect | ±1 position = smooth | A↔B = relative major/minor | +7 = energy boost

Always use tools to generate setlists — never fabricate track lists. Explain your reasoning concisely.
Format track names as "Artist - Title"."""

    # ------------------------------------------------------------------
    # Chat entry point
    # ------------------------------------------------------------------

    async def chat(self, user_message: str) -> ChatMessage:
        """Process a user message and return AI response."""
        self._last_setlist = None
        self._last_recommendations = None

        if not self._api_key:
            return await self._fallback_chat(user_message)

        self._conversation_history.append({"role": "user", "content": user_message})

        try:
            client = self._get_client()
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=self._build_system_prompt(),
                tools=TOOLS,
                messages=self._conversation_history,
            )
            return await self._process_response(response)
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return await self._fallback_chat(user_message)

    # ------------------------------------------------------------------
    # Response processing (handles tool-use loop)
    # ------------------------------------------------------------------

    async def _process_response(self, response) -> ChatMessage:
        """Process Claude's response, executing any tool calls."""
        text_parts = []
        tool_results = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                result = await self._execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })

        if tool_results:
            self._conversation_history.append({
                "role": "assistant",
                "content": response.content,
            })
            self._conversation_history.append({
                "role": "user",
                "content": tool_results,
            })

            try:
                client = self._get_client()
                final_response = client.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    system=self._build_system_prompt(),
                    tools=TOOLS,
                    messages=self._conversation_history,
                )

                final_text = ""
                for block in final_response.content:
                    if block.type == "text":
                        final_text += block.text
                    elif block.type == "tool_use":
                        await self._execute_tool(block.name, block.input)

                self._conversation_history.append({
                    "role": "assistant",
                    "content": final_text,
                })

                return ChatMessage(
                    role="assistant",
                    content=final_text,
                    setlist=self._last_setlist,
                    recommendations=self._last_recommendations,
                )
            except Exception as e:
                logger.error(f"Error getting final response: {e}")
                text = "\n".join(text_parts) or "Generated setlist successfully."
                return ChatMessage(
                    role="assistant",
                    content=text,
                    setlist=self._last_setlist,
                    recommendations=self._last_recommendations,
                )
        else:
            text = "\n".join(text_parts)
            self._conversation_history.append({"role": "assistant", "content": text})
            return ChatMessage(
                role="assistant",
                content=text,
                setlist=self._last_setlist,
                recommendations=self._last_recommendations,
            )

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    async def _execute_tool(self, tool_name: str, tool_input: Dict) -> Any:
        logger.info(f"Tool call: {tool_name} — {tool_input}")
        dispatch = {
            "build_set_from_prompt":   self._tool_build_set_from_prompt,
            "plan_set":                self._tool_plan_set,
            "generate_setlist":        self._tool_generate_setlist,
            "recommend_next_track":    self._tool_recommend_next,
            "search_library":          self._tool_search_library,
            "get_library_attributes":  self._tool_get_library_attributes,
            "get_energy_advice":       self._tool_energy_advice,
        }
        handler = dispatch.get(tool_name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}
        return await handler(tool_input)

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def _tool_build_set_from_prompt(self, inp: Dict) -> Dict:
        """Handle build_set_from_prompt tool call.

        Mirrors the /api/setlist/build endpoint logic:
        parse intent → MyTag candidate filtering → temp engine → setlist.
        """
        from .library_index import LibraryIndexFeatureStore
        from .setlist_engine import SetlistEngine as _SetlistEngine
        from .camelot import CamelotWheel as _CamelotWheel
        from .energy_planner import EnergyPlanner as _EnergyPlanner

        prompt = inp.get("prompt", "")
        duration_minutes = int(inp.get("duration_minutes", 60))

        # 1. Parse intent using dynamic library attributes
        intent = parse_set_intent(prompt, attrs=self._library_attributes)
        if intent["duration_minutes"]:
            duration_minutes = intent["duration_minutes"]
        duration_minutes = max(10, min(480, duration_minutes))

        # 2. MyTag candidate filtering via library index
        candidate_ids: set = set()
        tag_coverage: Dict[str, int] = {}

        if self._library_index is not None and intent["my_tags"]:
            for tag in intent["my_tags"]:
                matches = self._library_index.search(my_tag=tag, limit=500)
                tag_coverage[tag] = len(matches)
                for rec in matches:
                    candidate_ids.add(rec["id"])

        candidate_tracks = (
            [t for t in self.engine.tracks if str(t.id) in candidate_ids]
            if candidate_ids else []
        )

        used_fallback = len(candidate_tracks) < 10
        if used_fallback:
            candidate_tracks = self.engine.tracks

        # 3. Build a temp engine scoped to candidate pool
        ess_store = (
            LibraryIndexFeatureStore(self._library_index)
            if self._library_index is not None and len(self._library_index._by_id) > 0
            else self.engine.essentia_store
        )

        temp_engine = _SetlistEngine(
            tracks=candidate_tracks,
            camelot=_CamelotWheel(),
            energy_planner=_EnergyPlanner(),
            essentia_store=ess_store,
        )

        # 4. Generate setlist
        request = SetlistRequest(
            prompt=prompt,
            duration_minutes=duration_minutes,
            genre=intent["genre"],
            bpm_min=intent["bpm_min"],
            bpm_max=intent["bpm_max"],
            energy_profile=(
                intent["energy_profile"]
                if intent["energy_profile"] in ENERGY_PROFILES else "journey"
            ),
        )
        setlist = temp_engine.generate_setlist(request)
        self._last_setlist = setlist

        # Register in main engine so export works
        self.engine._setlists[setlist.id] = setlist

        return {
            "setlist_id": setlist.id,
            "prompt": prompt,
            "intent": {
                "my_tags_detected": intent["my_tags"],
                "tag_coverage": tag_coverage,
                "candidate_pool": (
                    len(candidate_tracks) if not used_fallback
                    else f"{len(candidate_tracks)} (fallback — no My Tag matches)"
                ),
                "genre": intent["genre"],
                "bpm_range": f"{intent['bpm_min']:.0f}–{intent['bpm_max']:.0f}",
                "energy_profile": intent["energy_profile"],
                "reasoning": intent["reasoning"],
            },
            "track_count": setlist.track_count,
            "duration_minutes": round(setlist.total_duration_seconds / 60),
            "avg_bpm": setlist.avg_bpm,
            "harmonic_score": setlist.harmonic_score,
            "energy_arc": setlist.energy_arc,
            "tracks": [
                {
                    "position": st.position,
                    "artist": st.track.artist,
                    "title": st.track.title,
                    "bpm": st.track.bpm,
                    "key": st.track.key,
                    "energy": st.track.energy,
                    "genre": st.track.genre,
                    "my_tags": st.track.my_tags,
                    "key_relation": st.key_relation,
                    "transition_score": st.transition_score,
                    "notes": st.notes,
                }
                for st in setlist.tracks
            ],
        }

    async def _tool_plan_set(self, inp: Dict) -> Dict:
        """Handle plan_set tool call — vibe context → setlist."""
        interpretation = interpret_vibe(
            vibe=inp.get("vibe", ""),
            situation=inp.get("situation", ""),
            venue=inp.get("venue", ""),
            crowd_energy=inp.get("crowd_energy", ""),
            time_of_day=inp.get("time_of_day", ""),
            genre_preference=inp.get("genre_preference") or "",
        )

        duration_minutes = max(10, min(480, int(inp.get("duration_minutes", 60))))
        context_parts = [p for p in [inp.get("vibe", ""), inp.get("situation", ""), inp.get("venue", "")] if p]
        prompt = " | ".join(context_parts) if context_parts else f"{duration_minutes}min set"

        request = SetlistRequest(
            prompt=prompt,
            duration_minutes=duration_minutes,
            genre=interpretation["genre"],
            bpm_min=interpretation["bpm_min"],
            bpm_max=interpretation["bpm_max"],
            energy_profile=interpretation["energy_profile"],
        )

        setlist = self.engine.generate_setlist(request)
        self._last_setlist = setlist

        return {
            "setlist_id": setlist.id,
            "interpretation": {
                "vibe_label": interpretation["vibe_label"],
                "genre": interpretation["genre"],
                "bpm_range": f"{interpretation['bpm_min']:.0f}–{interpretation['bpm_max']:.0f}",
                "energy_profile": interpretation["energy_profile"],
                "reasoning": interpretation["reasoning"],
            },
            "track_count": setlist.track_count,
            "duration_minutes": round(setlist.total_duration_seconds / 60),
            "avg_bpm": setlist.avg_bpm,
            "harmonic_score": setlist.harmonic_score,
            "energy_arc": setlist.energy_arc,
            "tracks": [
                {
                    "position": st.position,
                    "artist": st.track.artist,
                    "title": st.track.title,
                    "bpm": st.track.bpm,
                    "key": st.track.key,
                    "energy": st.track.energy,
                    "genre": st.track.genre,
                    "key_relation": st.key_relation,
                    "notes": st.notes,
                }
                for st in setlist.tracks
            ],
        }

    async def _tool_generate_setlist(self, inp: Dict) -> Dict:
        """Handle generate_setlist tool call with optional MyTag filtering."""
        # Find starting track by title if provided
        starting_id = None
        if inp.get("starting_track_title"):
            for t in self.engine.tracks:
                if inp["starting_track_title"].lower() in t.title.lower():
                    starting_id = t.id
                    break

        # Optional MyTag candidate filtering
        my_tags: List[str] = inp.get("my_tags") or []
        candidate_tracks = self.engine.tracks

        if my_tags and self._library_index is not None:
            candidate_ids: set = set()
            for tag in my_tags:
                for rec in self._library_index.search(my_tag=tag, limit=500):
                    candidate_ids.add(rec["id"])
            if len(candidate_ids) >= 10:
                candidate_tracks = [t for t in self.engine.tracks if str(t.id) in candidate_ids]

        # Build on a scoped or main engine
        if candidate_tracks is not self.engine.tracks:
            from .setlist_engine import SetlistEngine as _SE
            from .camelot import CamelotWheel as _CW
            from .energy_planner import EnergyPlanner as _EP
            gen_engine = _SE(
                tracks=candidate_tracks,
                camelot=_CW(),
                energy_planner=_EP(),
                essentia_store=self.engine.essentia_store,
            )
        else:
            gen_engine = self.engine

        request = SetlistRequest(
            prompt=inp.get("starting_track_title", "AI-generated set"),
            duration_minutes=max(10, min(480, int(inp.get("duration_minutes", 60)))),
            genre=inp.get("genre"),
            bpm_min=inp.get("bpm_min"),
            bpm_max=inp.get("bpm_max"),
            energy_profile=inp.get("energy_profile", "journey"),
            starting_track_id=starting_id,
        )

        setlist = gen_engine.generate_setlist(request)
        self._last_setlist = setlist
        self.engine._setlists[setlist.id] = setlist

        return {
            "setlist_id": setlist.id,
            "track_count": setlist.track_count,
            "duration_minutes": round(setlist.total_duration_seconds / 60),
            "avg_bpm": setlist.avg_bpm,
            "harmonic_score": setlist.harmonic_score,
            "energy_arc": setlist.energy_arc,
            "tracks": [
                {
                    "position": st.position,
                    "artist": st.track.artist,
                    "title": st.track.title,
                    "bpm": st.track.bpm,
                    "key": st.track.key,
                    "energy": st.track.energy,
                    "genre": st.track.genre,
                    "key_relation": st.key_relation,
                    "notes": st.notes,
                }
                for st in setlist.tracks
            ],
        }

    async def _tool_recommend_next(self, inp: Dict) -> Dict:
        """Handle recommend_next_track tool call."""
        recs = self.engine.recommend_next(
            current_track_title=inp["current_track_query"],
            energy_direction=inp.get("energy_direction", "maintain"),
            limit=inp.get("limit", 5),
        )
        self._last_recommendations = recs

        return {
            "recommendations": [
                {
                    "artist": r.track.artist,
                    "title": r.track.title,
                    "bpm": r.track.bpm,
                    "key": r.track.key,
                    "energy": r.track.energy,
                    "score": r.score,
                    "reason": r.reason,
                }
                for r in recs
            ]
        }

    async def _tool_search_library(self, inp: Dict) -> Dict:
        """Handle search_library tool call."""
        q = (inp.get("query") or "").strip().lower()
        date_from = inp.get("date_from", "")
        date_to = inp.get("date_to", "")
        my_tag = (inp.get("my_tag") or "").strip().lower()
        limit = int(inp.get("limit", 20))

        results = []
        for t in self.engine.tracks:
            if q and not (
                q in t.title.lower()
                or q in t.artist.lower()
                or q in (t.genre or "").lower()
                or q in (t.album or "").lower()
            ):
                continue
            d = t.date_added or ""
            if date_from and d < date_from:
                continue
            if date_to and d > date_to:
                continue
            if my_tag and not any(my_tag in tag.lower() for tag in t.my_tags):
                continue
            results.append({
                "id": t.id,
                "artist": t.artist,
                "title": t.title,
                "bpm": t.bpm,
                "key": t.key,
                "energy": t.energy,
                "genre": t.genre,
                "rating": t.rating,
                "date_added": t.date_added,
                "my_tags": t.my_tags,
            })
            if len(results) >= limit:
                break

        return {"results": results, "count": len(results)}

    async def _tool_get_library_attributes(self, _inp: Dict) -> Dict:
        """Return the full dynamic library attribute summary."""
        if self._library_attributes is None:
            return {"error": "Library attributes not available — rebuild the index first."}
        return self._library_attributes

    async def _tool_energy_advice(self, inp: Dict) -> Dict:
        """Handle get_energy_advice tool call."""
        return self.planner.recommend_energy_direction(
            current_position_pct=inp["position_pct"],
            current_energy=inp["current_energy"],
            profile=inp.get("profile", "journey"),
        )

    # ------------------------------------------------------------------
    # Fallback (no API key)
    # ------------------------------------------------------------------

    async def _fallback_chat(self, message: str) -> ChatMessage:
        """Direct engine usage when Claude API is unavailable."""
        msg_lower = message.lower()
        if any(w in msg_lower for w in ["next", "after", "recommend", "play after", "follow"]):
            return await self._fallback_recommend(message)
        elif any(w in msg_lower for w in ["search", "find", "look"]):
            return await self._fallback_search(message)
        else:
            return await self._fallback_generate(message)

    async def _fallback_generate(self, message: str) -> ChatMessage:
        """Parse message and generate setlist without Claude API."""
        # Use parse_set_intent for smarter parsing
        intent = parse_set_intent(message, attrs=self._library_attributes)

        duration_minutes = intent["duration_minutes"] or 60
        duration_minutes = max(10, min(480, duration_minutes))

        # MyTag candidate filtering
        candidate_tracks = self.engine.tracks
        if intent["my_tags"] and self._library_index is not None:
            candidate_ids: set = set()
            for tag in intent["my_tags"]:
                for rec in self._library_index.search(my_tag=tag, limit=500):
                    candidate_ids.add(rec["id"])
            if len(candidate_ids) >= 10:
                candidate_tracks = [t for t in self.engine.tracks if str(t.id) in candidate_ids]

        if candidate_tracks is not self.engine.tracks:
            from .setlist_engine import SetlistEngine as _SE
            from .camelot import CamelotWheel as _CW
            from .energy_planner import EnergyPlanner as _EP
            from .library_index import LibraryIndexFeatureStore
            ess = (
                LibraryIndexFeatureStore(self._library_index)
                if self._library_index and len(self._library_index._by_id) > 0
                else self.engine.essentia_store
            )
            gen_engine = _SE(tracks=candidate_tracks, camelot=_CW(), energy_planner=_EP(), essentia_store=ess)
        else:
            gen_engine = self.engine

        request = SetlistRequest(
            prompt=message,
            duration_minutes=duration_minutes,
            genre=intent["genre"],
            bpm_min=intent["bpm_min"],
            bpm_max=intent["bpm_max"],
            energy_profile=intent["energy_profile"],
        )

        setlist = gen_engine.generate_setlist(request)
        self._last_setlist = setlist
        self.engine._setlists[setlist.id] = setlist

        lines = [f"Generated a {setlist.track_count}-track setlist ({round(setlist.total_duration_seconds/60)} minutes):\n"]
        if intent["my_tags"]:
            lines.append(f"Using My Tags: {', '.join(intent['my_tags'])}\n")
        for st in setlist.tracks:
            energy_str = f"E{st.track.energy}" if st.track.energy else "E?"
            lines.append(
                f"  {st.position}. {st.track.artist} - {st.track.title} "
                f"[{st.track.bpm:.0f} BPM, {st.track.key or '?'}, {energy_str}]"
            )
            if st.position > 1 and st.notes:
                lines.append(f"     → {st.notes}")
        lines.append(f"\nHarmonic score: {setlist.harmonic_score:.1%}")
        lines.append(f"BPM range: {setlist.bpm_range[0]:.0f}–{setlist.bpm_range[1]:.0f}")

        return ChatMessage(role="assistant", content="\n".join(lines), setlist=setlist)

    async def _fallback_recommend(self, message: str) -> ChatMessage:
        """Parse message and recommend next tracks without Claude API."""
        import re
        cleaned = re.sub(
            r'\b(what|should|i|play|after|next|recommend|following|the|track|song)\b',
            '', message.lower()
        ).strip()

        track = None
        for t in self.engine.tracks:
            if cleaned and (cleaned in t.title.lower()
                           or cleaned in f"{t.artist} - {t.title}".lower()):
                track = t
                break

        if not track:
            return ChatMessage(
                role="assistant",
                content="I couldn't find that track in your library. Try a more specific title.",
            )

        energy_dir = "maintain"
        if any(w in message.lower() for w in ["higher", "up", "raise", "increase", "hype"]):
            energy_dir = "up"
        elif any(w in message.lower() for w in ["lower", "down", "cool", "decrease", "calm"]):
            energy_dir = "down"

        recs = self.engine.recommend_next(
            current_track_id=track.id, energy_direction=energy_dir, limit=5,
        )
        self._last_recommendations = recs

        lines = [f"After **{track.artist} - {track.title}** [{track.bpm:.0f} BPM, {track.key}, E{track.energy}]:\n"]
        for i, r in enumerate(recs, 1):
            lines.append(
                f"  {i}. **{r.track.artist} - {r.track.title}** "
                f"[{r.track.bpm:.0f} BPM, {r.track.key}, E{r.track.energy}] "
                f"(score: {r.score:.0%})"
            )
            lines.append(f"     {r.reason}")

        return ChatMessage(role="assistant", content="\n".join(lines), recommendations=recs)

    async def _fallback_search(self, message: str) -> ChatMessage:
        """Search library without Claude API."""
        import re
        cleaned = re.sub(
            r'\b(search|find|look|for|tracks?|songs?|in|library|my)\b',
            '', message.lower()
        ).strip()

        results = []
        for t in self.engine.tracks:
            if cleaned and (cleaned in t.title.lower()
                           or cleaned in t.artist.lower()
                           or cleaned in (t.genre or "").lower()):
                results.append(t)
                if len(results) >= 20:
                    break

        if not results:
            return ChatMessage(
                role="assistant",
                content=f"No tracks found matching '{cleaned}'.",
            )

        lines = [f"Found {len(results)} tracks matching '{cleaned}':\n"]
        for t in results:
            lines.append(
                f"  - {t.artist} - {t.title} [{t.bpm:.0f} BPM, {t.key}, "
                f"E{t.energy}, {t.genre}]"
            )
        return ChatMessage(role="assistant", content="\n".join(lines))

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history.clear()
        self._last_setlist = None
        self._last_recommendations = None
