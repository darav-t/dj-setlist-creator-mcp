"""
Library Index — Centralized JSONL track database for LLM grep and vector search.

Merges Rekordbox, Essentia, and Mixed In Key data into a single JSONL file at
.data/library_index.jsonl — one complete JSON object per line, one line per track.

Each record contains every field from every data source plus a ``_text`` field
that is a grep-optimized summary string.  Claude can search it with the Grep
tool; any vector-store pipeline can ingest it directly (each line = one document).

Usage:
    index = LibraryIndex()
    stats = index.build(tracks, essentia_store, mik_library)
    results = index.search("aggressive techno 130bpm")
    record  = index.get_by_id("12345")
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = _REPO_ROOT / ".data" / "library_index.jsonl"
_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _fmt_duration(length_seconds: int) -> str:
    """Convert integer seconds to 'M:SS' string."""
    if not length_seconds:
        return "0:00"
    m, s = divmod(int(length_seconds), 60)
    return f"{m}:{s:02d}"


def _build_record_from_track(
    track: Any,
    essentia_store: Any = None,
    mik_library: Any = None,
    indexed_at: str = "",
) -> dict:
    """
    Build a complete merged record dict for a single track.

    Pure function (no I/O).  Called by LibraryIndex.build() for every track.

    Args:
        track:          TrackWithEnergy instance.
        essentia_store: Optional EssentiaFeatureStore — provides audio features.
        mik_library:    Optional MixedInKeyLibrary — provides MIK energy data.
        indexed_at:     ISO-8601 UTC timestamp string to stamp on the record.

    Returns:
        Complete dict ready to be JSON-serialised as one JSONL line.
    """
    # ------------------------------------------------------------------
    # 1. Top-level fields from TrackWithEnergy
    # ------------------------------------------------------------------
    length_sec = getattr(track, "length", 0) or 0
    record: dict[str, Any] = {
        "_schema_version": _SCHEMA_VERSION,
        "id":            str(track.id),
        "title":         track.title,
        "artist":        track.artist,
        "album":         getattr(track, "album", None),
        "genre":         getattr(track, "genre", None),
        "bpm":           float(track.bpm) if track.bpm else 0.0,
        "key":           getattr(track, "key", None),
        "rating":        getattr(track, "rating", 0) or 0,
        "play_count":    getattr(track, "play_count", 0) or 0,
        "length_seconds": length_sec,
        "duration":      _fmt_duration(length_sec),
        "file_path":     getattr(track, "file_path", None),
        "date_added":    getattr(track, "date_added", None),
        "date_modified": getattr(track, "date_modified", None),
        "comments":      getattr(track, "comments", None),
        "color":         getattr(track, "color", None),
        "color_id":      getattr(track, "color_id", None),
        "my_tags":       list(getattr(track, "my_tags", []) or []),
        "energy":        getattr(track, "energy", None),
        "energy_source": getattr(track, "energy_source", "none"),
    }

    # Remove None values at the top level to keep records lean
    record = {k: v for k, v in record.items() if v is not None}

    # Restore mandatory keys even if falsy
    for key in ("id", "title", "artist", "bpm", "rating", "play_count",
                "length_seconds", "duration", "energy_source", "my_tags"):
        if key not in record:
            record[key] = [] if key == "my_tags" else ""

    # ------------------------------------------------------------------
    # 2. Essentia block
    # ------------------------------------------------------------------
    if essentia_store is not None:
        file_path = getattr(track, "file_path", None)
        if file_path:
            ess = essentia_store.get(file_path)
            if ess is not None:
                ess_dict = ess.to_cache_dict()
                # file_path is already at the top level — drop from nested block
                ess_dict.pop("file_path", None)
                if ess_dict:
                    record["essentia"] = ess_dict

    # ------------------------------------------------------------------
    # 3. MIK block
    # ------------------------------------------------------------------
    if mik_library is not None:
        title = getattr(track, "title", "")
        mik_data = mik_library.get_energy(title)
        if mik_data:
            record["mik"] = mik_data

    # ------------------------------------------------------------------
    # 4. _text grep field
    # ------------------------------------------------------------------
    record["_text"] = _build_text_field(record)

    # ------------------------------------------------------------------
    # 5. Metadata
    # ------------------------------------------------------------------
    record["_indexed_at"] = indexed_at or datetime.now(timezone.utc).isoformat()

    return record


def _build_text_field(record: dict) -> str:
    """
    Build the grep-optimized ``_text`` summary for a record.

    All searchable values are combined into a single space-separated string.
    Prefixed tokens (``energy:8``, ``my_tags:Festival``) let Claude write
    precise grep patterns that do not accidentally match JSON key names.
    """
    parts: list[str] = []

    # Core identity
    artist = record.get("artist", "")
    title  = record.get("title", "")
    if artist:
        parts.append(artist)
    if title:
        parts.append(title)
    if record.get("genre"):
        parts.append(record["genre"])
    if record.get("key"):
        parts.append(record["key"])

    # BPM and energy
    bpm = record.get("bpm", 0)
    if bpm:
        parts.append(f"{bpm:.0f}bpm")
    energy = record.get("energy")
    if energy is not None:
        parts.append(f"energy:{energy}")

    # Rating
    rating = record.get("rating", 0)
    if rating:
        parts.append(f"rating:{rating}")

    # Play count (useful for "most played" searches)
    pc = record.get("play_count", 0)
    if pc:
        parts.append(f"plays:{pc}")

    # My Tags — both joined block (for multi-word) and prefixed (for exact)
    my_tags = record.get("my_tags", [])
    if my_tags:
        parts.append("tags:" + " ".join(my_tags))
        for tag in my_tags:
            parts.append(f"my_tags:{tag}")

    # Comments
    if record.get("comments"):
        parts.append(f"comments:{record['comments']}")

    # Color label
    if record.get("color"):
        parts.append(f"color:{record['color']}")

    # Essentia fields
    ess = record.get("essentia", {})
    if ess:
        if ess.get("dominant_mood"):
            parts.append(f"mood:{ess['dominant_mood']}")
        if ess.get("dominant_genre"):
            parts.append(f"genre:{ess['dominant_genre']}")
        if ess.get("dominant_tag"):
            parts.append(f"tag:{ess['dominant_tag']}")
        if ess.get("danceability") is not None:
            parts.append(f"danceability:{ess['danceability']}")
        if ess.get("lufs") is not None:
            parts.append(f"lufs:{ess['lufs']}")
        if ess.get("key_note"):
            parts.append(ess["key_note"])
        # All Essentia genre labels for broader genre-based search
        for g_name in (ess.get("genre") or {}).keys():
            parts.append(g_name)
        # Music tag keys with prefix
        for t_name in (ess.get("tags") or {}).keys():
            parts.append(f"tag:{t_name}")

    # MIK fields
    mik = record.get("mik", {})
    if mik:
        if mik.get("collection"):
            parts.append(f"collection:{mik['collection']}")
        if mik.get("energy"):
            parts.append(f"mik_energy:{mik['energy']}")

    # Date added (year and full date for temporal searches)
    if record.get("date_added"):
        parts.append(f"date:{record['date_added']}")

    # Energy source
    if record.get("energy_source") and record["energy_source"] != "none":
        parts.append(f"energy_source:{record['energy_source']}")

    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# LibraryIndex
# ---------------------------------------------------------------------------

class LibraryIndex:
    """
    Builds and queries a JSONL index of all tracks, merging Rekordbox,
    Essentia, and Mixed In Key data into one record per track.

    The index file at ``.data/library_index.jsonl`` is designed for:

    * **LLM grep** — Claude uses the Grep tool against the ``_text`` field.
    * **Vector store ingest** — each line is a self-contained document.
    * **Programmatic lookup** — ``get_by_id()`` uses an in-memory dict.
    """

    def __init__(self, index_path: Path = INDEX_PATH) -> None:
        self._record_path: Path = index_path
        # In-memory id → record dict, populated after build() or load_from_disk()
        self._by_id: dict[str, dict] = {}
        self._built = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        tracks: list,
        essentia_store: Any = None,
        mik_library: Any = None,
    ) -> dict[str, Any]:
        """
        Build (or rebuild) the JSONL index file from scratch.

        Merges all data sources and writes one JSON line per track.
        Also populates the in-memory ``_by_id`` index for fast lookups.

        Args:
            tracks:         List of TrackWithEnergy instances.
            essentia_store: Optional EssentiaFeatureStore for audio features.
            mik_library:    Optional MixedInKeyLibrary for MIK energy data.

        Returns:
            Stats dict::

                {
                    "total":         int,   # tracks written
                    "with_essentia": int,   # records that include an essentia block
                    "with_mik":      int,   # records that include a mik block
                    "index_path":    str,
                    "built_at":      str,   # ISO-8601 UTC
                }
        """
        self._record_path.parent.mkdir(parents=True, exist_ok=True)
        self._by_id = {}

        indexed_at = datetime.now(timezone.utc).isoformat()
        total = 0
        with_essentia = 0
        with_mik = 0

        with self._record_path.open("w", encoding="utf-8") as fh:
            for track in tracks:
                try:
                    record = _build_record_from_track(
                        track,
                        essentia_store=essentia_store,
                        mik_library=mik_library,
                        indexed_at=indexed_at,
                    )
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    self._by_id[record["id"]] = record
                    total += 1
                    if "essentia" in record:
                        with_essentia += 1
                    if "mik" in record:
                        with_mik += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"LibraryIndex: skipped track {getattr(track, 'id', '?')}: {exc}")

        self._built = True

        stats = {
            "total":         total,
            "with_essentia": with_essentia,
            "with_mik":      with_mik,
            "index_path":    str(self._record_path),
            "built_at":      indexed_at,
        }
        logger.info(
            f"Library index built: {total} tracks "
            f"({with_essentia} with Essentia, {with_mik} with MIK) "
            f"→ {self._record_path}"
        )
        return stats

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str = "",
        my_tag: Optional[str] = None,
        genre: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Search the JSONL index using grep-style matching against ``_text``.

        Reads the file line-by-line (streaming — never loads the whole file).

        Args:
            query:   Free-text search against ``_text`` (case-insensitive).
            my_tag:  Filter to tracks with this Rekordbox My Tag (case-insensitive).
            genre:   Filter to tracks whose ``genre`` field contains this string.
            limit:   Maximum results to return.

        Returns:
            List of full record dicts for matching tracks.
        """
        if not self._record_path.exists():
            return []

        q_lower   = query.lower()  if query   else ""
        mt_lower  = my_tag.lower() if my_tag  else ""
        gen_lower = genre.lower()  if genre   else ""

        results: list[dict] = []

        with self._record_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Apply filters
                if q_lower and q_lower not in record.get("_text", "").lower():
                    continue

                if mt_lower:
                    tags_lower = [t.lower() for t in record.get("my_tags", [])]
                    if not any(mt_lower in t for t in tags_lower):
                        continue

                if gen_lower:
                    rec_genre = (record.get("genre") or "").lower()
                    if gen_lower not in rec_genre:
                        continue

                results.append(record)
                if len(results) >= limit:
                    break

        return results

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_by_id(self, track_id: str) -> Optional[dict]:
        """
        Return the full merged record for a track by its Rekordbox ID.

        Uses the in-memory ``_by_id`` dict when available (after build or
        load_from_disk); otherwise falls back to a file scan.
        """
        if self._built:
            return self._by_id.get(str(track_id))

        # File scan fallback
        if not self._record_path.exists():
            return None
        with self._record_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("id") == str(track_id):
                        return record
                except json.JSONDecodeError:
                    continue
        return None

    # ------------------------------------------------------------------
    # Freshness
    # ------------------------------------------------------------------

    def is_fresh(self, max_age_seconds: int = 3600) -> bool:
        """
        Return True if the index file exists and was written within
        ``max_age_seconds`` of now.

        A 1-hour default is appropriate for DJ workflows because the
        Rekordbox library is not modified while the server is running.
        """
        if not self._record_path.exists():
            return False
        mtime = self._record_path.stat().st_mtime
        age = datetime.now(timezone.utc).timestamp() - mtime
        return age < max_age_seconds

    # ------------------------------------------------------------------
    # Load without rebuild
    # ------------------------------------------------------------------

    def load_from_disk(self) -> int:
        """
        Load the existing JSONL file into the in-memory ``_by_id`` index
        without triggering a full rebuild.

        Used at startup when ``is_fresh()`` is True — avoids an expensive
        rebuild while still enabling fast ``get_by_id()`` calls.

        Returns:
            Number of records loaded, or 0 if the file does not exist.
        """
        if not self._record_path.exists():
            return 0

        count = 0
        self._by_id = {}
        with self._record_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self._by_id[record["id"]] = record
                    count += 1
                except json.JSONDecodeError:
                    continue

        self._built = True
        logger.debug(f"LibraryIndex: loaded {count} records from {self._record_path}")
        return count

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = f"{len(self._by_id)} records" if self._built else "not loaded"
        return f"LibraryIndex({status}, path={self._record_path})"
