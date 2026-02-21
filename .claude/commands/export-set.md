---
description: Export the last generated DJ set to a Rekordbox playlist
argument-hint: "playlist name (e.g. Sunset Set Feb 2026)"
---

Export the most recently generated setlist to Rekordbox as a playlist named: **$ARGUMENTS**

## Steps

1. Look in the current conversation for the most recent `setlist_id` returned by `build_set_from_prompt` or `generate_setlist`.

2. If a `setlist_id` is found, call `mcp__mcp-dj__export_setlist_to_rekordbox` with:
   - `setlist_name`: the name provided in $ARGUMENTS (if blank, use the set's own name from the conversation)
   - `setlist_id`: the most recent setlist_id from the conversation

3. If no `setlist_id` is in context, tell the user: "No setlist found in this conversation. Use `/build-set [prompt]` first to generate a set."

4. On success, confirm:
   > ✅ Playlist **"[name]"** created in Rekordbox with [track_count] tracks.
   > Restart Rekordbox or sync your USB to see it.

5. On error, show the error message and suggest:
   > Try restarting Rekordbox (the database may be locked) then run the export again.
