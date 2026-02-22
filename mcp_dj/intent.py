"""
Shared intent-parsing utilities for building DJ setlists from natural language.

Used by both the HTTP app (app.py) and the MCP server (mcp_server.py) to avoid
duplicating the vibe-keyword tables and parsing logic.
"""

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Vibe / venue keyword tables  (music-theory constants, not user data)
# ---------------------------------------------------------------------------

# (keyword, genre, bpm_min, bpm_max, display_label)
# Checked in order — most specific entries first.
VIBE_PROFILES = [
    ("après ski",    "tech house",     122, 128, "Après Ski"),
    ("apres ski",    "tech house",     122, 128, "Après Ski"),
    ("après-ski",    "tech house",     122, 128, "Après Ski"),
    ("ski chalet",   "tech house",     122, 128, "Alpine Chalet"),
    ("chalet",       "tech house",     122, 128, "Alpine Chalet"),
    ("ski",          "tech house",     122, 128, "Après Ski"),
    ("after party",  "techno",         130, 140, "After Party"),
    ("afterparty",   "techno",         130, 140, "After Party"),
    ("warehouse",    "techno",         130, 138, "Warehouse Rave"),
    ("underground",  "techno",         128, 136, "Underground"),
    ("industrial",   "techno",         132, 140, "Industrial"),
    ("sunrise",      "melodic techno", 118, 124, "Sunrise"),
    ("rooftop",      "melodic house",  120, 126, "Rooftop"),
    ("sundowner",    "melodic house",  118, 125, "Sundowner"),
    ("sunset",       "melodic house",  118, 126, "Sunset"),
    ("pool party",   "house",          120, 128, "Pool Party"),
    ("pool",         "house",          120, 128, "Pool Party"),
    ("beach",        "house",          120, 128, "Beach"),
    ("main stage",   "tech house",     128, 135, "Main Stage"),
    ("mainstage",    "tech house",     128, 135, "Main Stage"),
    ("festival",     "tech house",     126, 132, "Festival"),
    ("nightclub",    "tech house",     126, 132, "Nightclub"),
    ("club",         "tech house",     126, 132, "Club"),
    ("lounge",       "deep house",     115, 122, "Lounge"),
    ("bar",          "house",          118, 126, "Bar"),
    ("deep house",   "deep house",     115, 122, "Deep House"),
    ("deep",         "deep house",     115, 122, "Deep House"),
    ("melodic",      "melodic techno", 120, 128, "Melodic"),
    ("techno",       "techno",         128, 138, "Techno"),
    ("tech house",   "tech house",     124, 132, "Tech House"),
    ("house",        "house",          120, 128, "House"),
]

# (keyword, energy_profile) — checked against situation + vibe + venue combined
SITUATION_TO_PROFILE = [
    ("warm-up",     "journey"),
    ("warm up",     "journey"),
    ("warmup",      "journey"),
    ("opening",     "build"),
    ("peak time",   "peak"),
    ("peak",        "peak"),
    ("headline",    "peak"),
    ("main slot",   "peak"),
    ("closing",     "wave"),
    ("close out",   "wave"),
    ("after party", "build"),
    ("afterparty",  "build"),
    ("sunrise",     "chill"),
    ("ambient",     "chill"),
    ("background",  "chill"),
    ("chill",       "chill"),
]

# (keyword, energy_profile) — checked against time_of_day
TIME_TO_PROFILE = [
    ("sunrise",    "chill"),
    ("dawn",       "chill"),
    ("morning",    "chill"),
    ("afternoon",  "journey"),
    ("evening",    "build"),
    ("late night", "wave"),
    ("midnight",   "wave"),
    ("night",      "peak"),
]

# (keyword, bpm_delta) — crowd energy adjusts BPM range up or down
CROWD_BPM_DELTA = [
    ("hyped",    +2),
    ("intense",  +3),
    ("clubbers", +2),
    ("casual",   -2),
    ("relaxed",  -3),
    ("sleepy",   -4),
]


# ---------------------------------------------------------------------------
# parse_set_intent
# ---------------------------------------------------------------------------

def parse_set_intent(
    prompt: str,
    attrs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Parse a free-form natural language set description into structured parameters.

    Uses the dynamically scanned library attributes (``attrs``) so no tag names,
    genre names, or BPM ranges are hardcoded.  The attributes are built at
    startup from the actual Rekordbox + Essentia + MIK data.

    Tag matching strategy
    ---------------------
    1. Full tag-name substring match  (e.g. prompt contains "afters" → "Afters")
    2. All-words match for multi-word tags (e.g. "high energy" → "High Energy")
    3. Tags are ranked by their library track count (most-used first) so the
       highest-signal tags appear first in the result list.

    Genre / BPM resolution
    ----------------------
    1. Match actual genre names from the library (longest first to avoid
       partial clashes), using that genre's real p25–p75 BPM range.
    2. Fall back to the venue-keyword table (VIBE_PROFILES) for BPM guidance
       while still using real library data to tighten the range when available.

    Args:
        prompt: Free-form natural language DJ set description.
        attrs:  Library attributes dict (from LibraryIndex.attributes).
                If None, falls back to heuristic-only parsing.

    Returns:
        {my_tags, genre, bpm_min, bpm_max, energy_profile, vibe_label,
         duration_minutes, reasoning}
    """
    import re as _re
    p = prompt.lower()

    # ------------------------------------------------------------------
    # 1. My Tag matching — fully dynamic from library data
    # ------------------------------------------------------------------
    my_tags: List[str] = []
    seen_tags: set = set()

    if attrs:
        tag_details = attrs.get("my_tag_details", {})
        # Sort by track count descending so the most relevant tags rank first
        ranked_tags = sorted(
            tag_details.keys(),
            key=lambda t: tag_details[t].get("count", 0),
            reverse=True,
        )
        for tag_name in ranked_tags:
            if tag_name.startswith("---"):
                continue
            tag_lower = tag_name.lower()
            # Full name present in prompt
            if tag_lower in p:
                if tag_name not in seen_tags:
                    my_tags.append(tag_name)
                    seen_tags.add(tag_name)
                continue
            # All significant words of a multi-word tag appear in prompt
            words = [
                w for w in tag_lower.split()
                if len(w) > 3 and w not in {"with", "from", "that", "this"}
            ]
            if len(words) >= 2 and all(w in p for w in words):
                if tag_name not in seen_tags:
                    my_tags.append(tag_name)
                    seen_tags.add(tag_name)

    # ------------------------------------------------------------------
    # 2. Genre + BPM — from real library genre data, then venue keywords
    # ------------------------------------------------------------------
    genre: Optional[str] = None
    bpm_min = 120.0
    bpm_max = 132.0
    vibe_label = "Custom"

    if attrs:
        genre_details = attrs.get("genre_details", {})
        # Sort longest names first to prevent "House" from shadowing "Tech House"
        for genre_name in sorted(genre_details, key=len, reverse=True):
            if genre_name.lower() in p:
                genre = genre_name
                vibe_label = genre_name
                bpm_d = genre_details[genre_name].get("bpm", {})
                if bpm_d:
                    # Use interquartile range (p25–p75) as a tight, realistic window
                    bpm_min = float(bpm_d.get("p25", bpm_d.get("min", 120)))
                    bpm_max = float(bpm_d.get("p75", bpm_d.get("max", 132)))
                break

    # Venue/vibe keyword fallback (still overrides BPM with real data if possible)
    if not genre:
        for keyword, g, mn, mx, lbl in VIBE_PROFILES:
            if keyword in p:
                genre      = g
                bpm_min    = float(mn)
                bpm_max    = float(mx)
                vibe_label = lbl
                # Try to replace hardcoded BPM with actual library data
                if attrs:
                    gd = attrs.get("genre_details", {})
                    for lib_genre, ginfo in gd.items():
                        if g.lower() in lib_genre.lower() or lib_genre.lower() in g.lower():
                            bpm_d = ginfo.get("bpm", {})
                            if bpm_d:
                                bpm_min = float(bpm_d.get("p25", bpm_min))
                                bpm_max = float(bpm_d.get("p75", bpm_max))
                            break
                break

    # ------------------------------------------------------------------
    # 3. Energy profile — from situation / time-of-day keyword tables
    #    (these express music-theory/DJ conventions, not user data)
    # ------------------------------------------------------------------
    energy_profile: Optional[str] = None
    for keyword, profile in SITUATION_TO_PROFILE:
        if keyword in p:
            energy_profile = profile
            break
    if not energy_profile:
        for keyword, profile in TIME_TO_PROFILE:
            if keyword in p:
                energy_profile = profile
                break
    if not energy_profile:
        energy_profile = "journey"

    # ------------------------------------------------------------------
    # 4. BPM nudge from crowd energy
    # ------------------------------------------------------------------
    for keyword, delta in CROWD_BPM_DELTA:
        if keyword in p:
            bpm_min += delta
            bpm_max += delta
            break

    # ------------------------------------------------------------------
    # 5. Duration hint  ("2 hour", "90 minute", "45min", "1.5h")
    # ------------------------------------------------------------------
    duration: Optional[int] = None
    m = _re.search(r'(\d+(?:\.\d+)?)\s*(?:hour|hr)s?', p)
    if m:
        duration = int(float(m.group(1)) * 60)
    else:
        m = _re.search(r'(\d+)\s*(?:minute|min)s?', p)
        if m:
            duration = int(m.group(1))

    # ------------------------------------------------------------------
    # 6. Reasoning string
    # ------------------------------------------------------------------
    parts: List[str] = []
    if my_tags:
        parts.append(f"My Tags → {', '.join(my_tags[:6])}")
    if genre:
        parts.append(f"Genre → {genre} ({vibe_label})")
    parts.append(f"BPM → {bpm_min:.0f}–{bpm_max:.0f}")
    parts.append(f"Energy arc → {energy_profile}")
    if duration:
        parts.append(f"Duration → {duration} min")
    if not attrs:
        parts.append("⚠ Library attributes unavailable — heuristic fallback used")

    return {
        "my_tags":          my_tags,
        "genre":            genre,
        "bpm_min":          bpm_min,
        "bpm_max":          bpm_max,
        "energy_profile":   energy_profile,
        "vibe_label":       vibe_label,
        "duration_minutes": duration,
        "reasoning":        " | ".join(parts),
    }


# ---------------------------------------------------------------------------
# interpret_vibe
# ---------------------------------------------------------------------------

def interpret_vibe(
    vibe: str = "",
    situation: str = "",
    venue: str = "",
    crowd_energy: str = "",
    time_of_day: str = "",
    genre_preference: str = "",
) -> Dict[str, Any]:
    """Translate freeform DJ context into SetlistRequest parameters."""
    combined = " ".join([vibe, venue, situation, crowd_energy, time_of_day]).lower()

    # 1. Genre + BPM from vibe/venue keywords
    genre: Optional[str] = None
    bpm_min = 124.0
    bpm_max = 130.0
    vibe_label = "General"

    if genre_preference:
        genre = genre_preference
        for _, g, mn, mx, lbl in VIBE_PROFILES:
            if g.lower() == genre_preference.lower():
                bpm_min, bpm_max = float(mn), float(mx)
                vibe_label = lbl
                break
    else:
        for keyword, g, mn, mx, lbl in VIBE_PROFILES:
            if keyword in combined:
                genre = g
                bpm_min, bpm_max = float(mn), float(mx)
                vibe_label = lbl
                break

    # 2. Energy profile from situation, then time of day
    energy_profile = None
    situation_text = " ".join([situation, vibe, venue]).lower()
    for keyword, profile in SITUATION_TO_PROFILE:
        if keyword in situation_text:
            energy_profile = profile
            break

    if not energy_profile:
        for keyword, profile in TIME_TO_PROFILE:
            if keyword in time_of_day.lower():
                energy_profile = profile
                break

    if not energy_profile:
        energy_profile = "journey"

    # 3. BPM nudge based on crowd energy
    bpm_delta = 0
    for keyword, delta in CROWD_BPM_DELTA:
        if keyword in crowd_energy.lower():
            bpm_delta = delta
            break
    bpm_min += bpm_delta
    bpm_max += bpm_delta

    # 4. Human-readable reasoning
    parts = []
    if genre:
        parts.append(f"Genre → {genre} ('{vibe_label}' vibe)")
    parts.append(f"BPM → {bpm_min:.0f}–{bpm_max:.0f}")
    parts.append(f"Energy profile → {energy_profile}")
    if bpm_delta != 0:
        parts.append(f"BPM shifted {bpm_delta:+d} for '{crowd_energy}' crowd")

    return {
        "genre": genre,
        "bpm_min": bpm_min,
        "bpm_max": bpm_max,
        "energy_profile": energy_profile,
        "vibe_label": vibe_label,
        "reasoning": " | ".join(parts),
    }
