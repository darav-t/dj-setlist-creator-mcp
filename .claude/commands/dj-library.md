---
description: Explore your DJ library — search tracks, browse tags, view stats
argument-hint: "tag name, genre, artist, track title, or 'stats' / 'tags'"
---

Explore the DJ library with this query: **$ARGUMENTS**

Determine what the user wants based on $ARGUMENTS:

## If $ARGUMENTS is empty, "stats", or "summary"

Call `mcp__mcp-dj__get_library_summary` and `mcp__mcp-dj__get_library_attributes`, then present:

**Library overview**
- Total tracks, BPM range (min/avg/max), top 5 genres with track counts
- Energy distribution: show as a small bar chart (levels 1-10 with counts)
- Top 10 MyTags by track count
- Top 10 artists by track count
- Essentia coverage (how many tracks have audio analysis)

End with: *"Use `/build-set [prompt]` to build a set, or `/dj-library [tag or genre]` to browse."*

## If $ARGUMENTS is "tags" or "mytags"

Call `mcp__mcp-dj__get_library_attributes` and display the full MyTag hierarchy:

For each group (Has / Vibes / Situation / Bangers), show a table:
| Tag | Tracks | BPM range | Avg energy | Dominant mood |
|-----|--------|-----------|------------|---------------|

## If $ARGUMENTS looks like a tag name (matches a word from the known hierarchy)

Call `mcp__mcp-dj__search_library` with `my_tag=$ARGUMENTS` and `limit=50`.
Show a compact track list: Artist – Title | BPM | Key | Energy | Duration

Then suggest: *"Use `/build-set [context] [tag name]` to build a set with these tracks."*

## If $ARGUMENTS looks like a genre, artist, or track title

Call `mcp__mcp-dj__search_library` with `query=$ARGUMENTS` and `limit=30`.
Show: Artist – Title | Genre | BPM | Key | Energy | MyTags

If a single track is found or clearly intended, also call `mcp__mcp-dj__get_track_full_info` with the title and show the full merged record including Essentia features, MIK data, and all MyTags.

## Rules
- Always show track counts so the user knows how populated each tag/genre is
- If a search returns 0 results, suggest alternative spellings or nearby tags from the library attributes
- Keep tables compact — truncate long titles to 40 chars with "…"
