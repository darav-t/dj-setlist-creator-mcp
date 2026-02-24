"""
DJ Set Visualizer — generates a self-contained HTML flow diagram for a setlist.

Called automatically after every set build. Saves to:
  .data/dj_history/<timestamp>_<slug>_<id>.html
  .data/dj_history/<timestamp>_<slug>_<id>.json
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).parent.parent / ".data"
HISTORY_DIR = DATA_DIR / "dj_history"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_set_history(setlist: Dict[str, Any]) -> Dict[str, str]:
    """
    Persist a generated setlist to disk as JSON + HTML.

    Returns dict with keys:
        json_path  — absolute path to the saved JSON file
        html_path  — absolute path to the saved HTML file
        timestamp  — ISO timestamp used in filenames
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slugify(setlist.get("name", "set"))[:40]
    sid = setlist.get("setlist_id", "unknown")[:8]
    base = f"{ts}_{slug}_{sid}"

    json_path = HISTORY_DIR / f"{base}.json"
    html_path = HISTORY_DIR / f"{base}.html"

    # Save JSON
    json_path.write_text(json.dumps(setlist, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save HTML
    html_path.write_text(render_html(setlist), encoding="utf-8")

    return {
        "json_path": str(json_path),
        "html_path": str(html_path),
        "timestamp": ts,
    }


def render_html(setlist: Dict[str, Any]) -> str:
    """Render a fully self-contained HTML visualization of a DJ set."""
    tracks: List[Dict[str, Any]] = setlist.get("tracks", [])
    energy_arc: List[int] = setlist.get("energy_arc", [])
    genre_dist: Dict[str, int] = setlist.get("genre_distribution", {})
    intent: Dict[str, Any] = setlist.get("intent", {})

    name = setlist.get("name", "DJ Set")
    prompt = setlist.get("prompt", "")
    harmonic = setlist.get("harmonic_score", 0)
    avg_bpm = setlist.get("avg_bpm", 0)
    bpm_range = setlist.get("bpm_range", {})
    duration = setlist.get("duration_minutes", 0)
    track_count = setlist.get("track_count", len(tracks))
    sid = setlist.get("setlist_id", "")
    gen_at = datetime.now().strftime("%B %d, %Y · %H:%M")

    energy_svg = _energy_arc_svg(energy_arc)
    track_cards = "\n".join(_track_card(t, i, len(tracks)) for i, t in enumerate(tracks))
    scatter_html = _scatter_section(tracks)
    intent_html = _intent_section(intent)
    genre_pills = _genre_pills(genre_dist)
    harmonic_class = "score-good" if harmonic >= 0.75 else "score-ok" if harmonic >= 0.55 else "score-warn"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(name)} — DJ Set</title>
<style>
{_CSS}
</style>
</head>
<body>
<div class="page">

  <!-- ── HEADER ── -->
  <header>
    <div class="header-top">
      <div class="header-left">
        <div class="set-label">DJ SET</div>
        <h1>{_esc(name)}</h1>
        {f'<p class="prompt-text">"{_esc(prompt)}"</p>' if prompt else ''}
      </div>
      <div class="header-right">
        <div class="stat-grid">
          <div class="stat"><span class="stat-val">{track_count}</span><span class="stat-lbl">tracks</span></div>
          <div class="stat"><span class="stat-val">{duration}</span><span class="stat-lbl">minutes</span></div>
          <div class="stat"><span class="stat-val">{avg_bpm:.0f}</span><span class="stat-lbl">avg BPM</span></div>
          <div class="stat"><span class="stat-val {harmonic_class}">{harmonic:.0%}</span><span class="stat-lbl">harmonic</span></div>
        </div>
        {f'<div class="bpm-range">BPM range: <b>{bpm_range.get("min",0):.0f} – {bpm_range.get("max",0):.0f}</b></div>' if bpm_range else ''}
        <div class="genre-row">{genre_pills}</div>
      </div>
    </div>
    <div class="generated-at">Generated {gen_at} · ID: {sid}</div>
  </header>

  <!-- ── ENERGY ARC ── -->
  <section class="energy-section">
    <div class="section-label">ENERGY ARC</div>
    {energy_svg}
  </section>

  <!-- ── 2D TRACK MAP ── -->
  {scatter_html}

  <!-- ── FLOW DIAGRAM ── -->
  <section class="flow-section">
    <div class="section-label">SET FLOW</div>
    <div class="flow">
{track_cards}
    </div>
  </section>

  <!-- ── INTENT ── -->
  {intent_html}

</div><!-- .page -->

<script>
{_JS}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Energy arc SVG
# ---------------------------------------------------------------------------


def _energy_arc_svg(arc: List[int]) -> str:
    if not arc:
        return ""
    W, H, PAD = 900, 120, 16
    n = len(arc)
    step = (W - PAD * 2) / max(n - 1, 1)

    def x(i: int) -> float:
        return PAD + i * step

    def y(v: int) -> float:
        return H - PAD - (v / 10) * (H - PAD * 2)

    # Build smooth path
    pts = [(x(i), y(v)) for i, v in enumerate(arc)]
    path_d = f"M {pts[0][0]:.1f},{pts[0][1]:.1f}"
    for i in range(1, len(pts)):
        cx = (pts[i - 1][0] + pts[i][0]) / 2
        path_d += f" C {cx:.1f},{pts[i-1][1]:.1f} {cx:.1f},{pts[i][1]:.1f} {pts[i][0]:.1f},{pts[i][1]:.1f}"

    # Fill area under curve
    fill_d = path_d + f" L {pts[-1][0]:.1f},{H - PAD} L {pts[0][0]:.1f},{H - PAD} Z"

    # Dots
    dots = "".join(
        f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" class="arc-dot" data-pos="{i}" data-energy="{arc[i]}" />'
        for i, (px, py) in enumerate(pts)
    )

    # Grid lines (energy 2, 4, 6, 8)
    grid = "".join(
        f'<line x1="{PAD}" y1="{y(v):.1f}" x2="{W-PAD}" y2="{y(v):.1f}" class="grid-line" />'
        f'<text x="{PAD-4}" y="{y(v)+4:.1f}" class="grid-lbl">{v}</text>'
        for v in [2, 4, 6, 8]
    )

    # Track number labels at bottom
    labels = "".join(
        f'<text x="{x(i):.1f}" y="{H:.1f}" class="arc-label">{i+1}</text>'
        for i in range(n)
        if n <= 30 or i % max(1, n // 20) == 0
    )

    return f"""<div class="arc-wrap">
  <svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" class="arc-svg">
    <defs>
      <linearGradient id="arcGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#a78bfa" stop-opacity="0.5"/>
        <stop offset="100%" stop-color="#a78bfa" stop-opacity="0.05"/>
      </linearGradient>
    </defs>
    {grid}
    <path d="{fill_d}" fill="url(#arcGrad)" />
    <path d="{path_d}" stroke="#a78bfa" stroke-width="2.5" fill="none" stroke-linecap="round"/>
    {dots}
    {labels}
  </svg>
  <div class="arc-tooltip" id="arc-tip"></div>
</div>"""


# ---------------------------------------------------------------------------
# Track card
# ---------------------------------------------------------------------------

_KEY_COLORS = {
    "1A": "#e74c3c", "1B": "#e74c3c",
    "2A": "#e67e22", "2B": "#e67e22",
    "3A": "#f1c40f", "3B": "#f1c40f",
    "4A": "#2ecc71", "4B": "#2ecc71",
    "5A": "#1abc9c", "5B": "#1abc9c",
    "6A": "#3498db", "6B": "#3498db",
    "7A": "#9b59b6", "7B": "#9b59b6",
    "8A": "#e91e63", "8B": "#e91e63",
    "9A": "#ff5722", "9B": "#ff5722",
    "10A": "#cddc39", "10B": "#cddc39",
    "11A": "#00bcd4", "11B": "#00bcd4",
    "12A": "#607d8b", "12B": "#607d8b",
}

_RELATION_ICONS = {
    "same": ("⟳", "Same key"),
    "adjacent": ("→", "Adjacent"),
    "inner_outer": ("⇄", "Inner/Outer"),
    "energy_boost": ("↑", "Energy boost"),
    "energy_drop": ("↓", "Energy drop"),
    "incompatible": ("✗", "Clash"),
    "": ("·", ""),
}

_MOOD_BARS = ["happy", "aggressive", "party", "relaxed", "sad"]


def _track_card(t: Dict[str, Any], idx: int, total: int) -> str:
    pos = t.get("position", idx + 1)
    artist = t.get("artist", "")
    title = t.get("title", "")
    bpm = t.get("bpm", 0)
    key = t.get("key", "")
    energy = t.get("energy") or t.get("essentia_energy") or 0
    genre = t.get("genre", "")
    my_tags = t.get("my_tags") or []
    duration = t.get("duration", "")
    key_rel = t.get("key_relation", "")
    trans_score = t.get("transition_score", 0)
    notes = t.get("notes", "")
    dom_mood = t.get("dominant_mood", "")
    lufs = t.get("lufs")
    danceability = t.get("danceability")
    mood = t.get("mood") or {}
    top_genres = t.get("top_genres") or {}
    top_tags = t.get("top_tags") or {}

    key_color = _KEY_COLORS.get(key, "#888")
    energy_dots = _energy_dots(energy)
    mood_bars = _mood_section(mood) if mood else ""
    tag_chips = "".join(f'<span class="tag-chip">{_esc(tag)}</span>' for tag in my_tags)
    genre_top = _top_items(top_genres, 3)
    tags_top = _top_items(top_tags, 4)

    rel_icon, rel_label = _RELATION_ICONS.get(key_rel, ("·", key_rel))
    score_class = "score-good" if trans_score >= 0.8 else "score-ok" if trans_score >= 0.6 else "score-warn"
    is_first = idx == 0

    extra_meta = []
    if lufs is not None:
        extra_meta.append(f"<span class='meta-item'>Loudness: <b>{lufs:.1f} LUFS</b></span>")
    if danceability is not None:
        extra_meta.append(f"<span class='meta-item'>Dance: <b>{danceability}/10</b></span>")
    if dom_mood:
        extra_meta.append(f"<span class='meta-item'>Mood: <b>{dom_mood}</b></span>")
    extra_str = " ".join(extra_meta)

    # connector arrow (between cards)
    connector = ""
    if idx < total - 1:
        next_rel = rel_label or ""
        score_pct = f"{trans_score:.0%}" if trans_score else ""
        connector = f"""
      <div class="connector">
        <div class="conn-line"></div>
        <div class="conn-info">
          <span class="conn-rel">{_esc(rel_icon)} {_esc(next_rel)}</span>
          {f'<span class="conn-score {score_class}">{score_pct}</span>' if score_pct else ''}
          {f'<span class="conn-notes">{_esc(notes)}</span>' if notes else ''}
        </div>
        <div class="conn-arrow">▼</div>
      </div>"""

    card = f"""      <div class="track-card {'opening-track' if is_first else ''}" id="track-{pos}">
        <div class="card-header" onclick="toggleCard(this)">
          <div class="pos-badge">{pos:02d}</div>
          <div class="track-identity">
            <span class="track-artist">{_esc(artist)}</span>
            <span class="track-sep"> — </span>
            <span class="track-title">{_esc(title)}</span>
          </div>
          <div class="track-meta-row">
            <span class="bpm-badge">{bpm:.0f} BPM</span>
            <span class="key-badge" style="--kc:{key_color}">{_esc(key)}</span>
            <span class="energy-dots">{energy_dots}</span>
            {f'<span class="genre-small">{_esc(genre)}</span>' if genre else ''}
            {f'<span class="duration-small">{_esc(duration)}</span>' if duration else ''}
          </div>
          <div class="expand-icon">▸</div>
        </div>
        <div class="card-body">
          <div class="card-inner">
            {f'<div class="my-tags">{tag_chips}</div>' if tag_chips else ''}
            {f'<div class="extra-meta">{extra_str}</div>' if extra_str else ''}
            {mood_bars}
            {f'<div class="sub-section"><div class="sub-label">Top Genres</div>{genre_top}</div>' if genre_top else ''}
            {f'<div class="sub-section"><div class="sub-label">Music Tags</div>{tags_top}</div>' if tags_top else ''}
          </div>
        </div>
      </div>{connector}"""
    return card


# ---------------------------------------------------------------------------
# Intent section
# ---------------------------------------------------------------------------


def _intent_section(intent: Dict[str, Any]) -> str:
    if not intent:
        return ""

    rows = []
    if intent.get("my_tags_detected"):
        rows.append(("Tags Detected", ", ".join(intent["my_tags_detected"])))
    if intent.get("candidate_pool"):
        rows.append(("Candidate Pool", str(intent["candidate_pool"])))
    if intent.get("genre"):
        rows.append(("Genre", intent["genre"]))
    if intent.get("bpm_range"):
        rows.append(("BPM Range", intent["bpm_range"]))
    if intent.get("reasoning"):
        rows.append(("Reasoning", intent["reasoning"]))
    if intent.get("structured_params_provided"):
        rows.append(("Structured Params", "Yes — LLM-provided energy/genre/BPM curves"))

    # Tag coverage
    cov = intent.get("tag_coverage", {})
    if cov:
        cov_str = " · ".join(f"{tag}: {cnt}" for tag, cnt in cov.items())
        rows.append(("Tag Coverage", cov_str))

    # Extra structured params
    for key, label in [("energy_curve", "Energy Curve"), ("genre_phases", "Genre Phases"),
                       ("bpm_curve", "BPM Curve"), ("mood_target", "Mood Target")]:
        if intent.get(key):
            rows.append((label, json.dumps(intent[key], ensure_ascii=False)))

    rows_html = "".join(
        f'<tr><td class="ikey">{_esc(k)}</td><td class="ival">{_esc(str(v))}</td></tr>'
        for k, v in rows
    )

    return f"""<section class="intent-section">
    <div class="section-label" onclick="toggleSection(this)">SET INTENT <span class="expand-icon small">▸</span></div>
    <div class="intent-body">
      <table class="intent-table">{rows_html}</table>
    </div>
  </section>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _energy_dots(energy: int, total: int = 10) -> str:
    filled = max(0, min(int(energy), total))
    return "●" * filled + "○" * (total - filled)


def _mood_section(mood: Dict[str, float]) -> str:
    if not mood:
        return ""
    bars = ""
    for key in _MOOD_BARS:
        val = mood.get(key, 0)
        pct = val * 100
        bars += (
            f'<div class="mood-row">'
            f'<span class="mood-lbl">{key}</span>'
            f'<div class="mood-bar-bg">'
            f'<div class="mood-bar-fill" style="width:{pct:.0f}%"></div>'
            f'</div>'
            f'<span class="mood-val">{pct:.0f}%</span>'
            f'</div>'
        )
    return f'<div class="mood-section">{bars}</div>'


def _top_items(d: Dict[str, float], n: int = 4) -> str:
    if not d:
        return ""
    items = sorted(d.items(), key=lambda x: -x[1])[:n]
    chips = "".join(
        f'<span class="score-chip">{_esc(k)} <b>{v:.0%}</b></span>'
        for k, v in items
    )
    return f'<div class="chip-row">{chips}</div>'


def _genre_pills(dist: Dict[str, int]) -> str:
    if not dist:
        return ""
    total = sum(dist.values()) or 1
    top = sorted(dist.items(), key=lambda x: -x[1])[:5]
    return " ".join(
        f'<span class="genre-pill">{_esc(g)} <b>{c/total:.0%}</b></span>'
        for g, c in top
    )


def _esc(s: str) -> str:
    """HTML-escape a string."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[-\s]+", "-", text).strip("-")


# ---------------------------------------------------------------------------
# 2D track map (PCA projection)
# ---------------------------------------------------------------------------

_FEAT_NAMES = ["BPM", "Energy", "Dance", "Loud", "Happy", "Sad", "Aggr", "Relaxed", "Party"]


def _compute_2d_projection(tracks: List[Dict[str, Any]]):
    """
    Project tracks to 2D via PCA on acoustic features.

    Returns (xs, ys, feat_names, pc1_loadings, pc2_loadings, var_pc1, var_pc2).
    Falls back to a simple 2-feature projection if numpy is unavailable.
    """
    try:
        import numpy as np

        rows = []
        for t in tracks:
            bpm = float(t.get("bpm") or 125)
            energy = float(t.get("energy") or t.get("essentia_energy") or 5)
            dance = float(t.get("danceability") or 5)
            lufs = float(t.get("lufs") or -10.0)
            mood = t.get("mood") or {}
            rows.append([
                (bpm - 110) / 40,           # BPM centred ~130
                (energy - 5) / 4,           # Energy 1-10
                (dance - 5) / 4,            # Danceability 1-10
                (lufs + 12) / 8,            # LUFS — louder ↑
                float(mood.get("happy", 0.25)),
                float(mood.get("sad", 0.5)),
                float(mood.get("aggressive", 0.4)),
                float(mood.get("relaxed", 0.6)),
                float(mood.get("party", 0.2)),
            ])

        X = np.array(rows, dtype=float)
        X -= X.mean(axis=0)

        if len(X) < 2:
            return [0.0], [0.0], _FEAT_NAMES, [1] + [0] * 8, [0, 1] + [0] * 7, 0.5, 0.3

        cov = np.cov(X.T)
        vals, vecs = np.linalg.eigh(cov)
        idx = np.argsort(vals)[::-1]
        vecs = vecs[:, idx]
        vals = vals[idx]

        projected = X @ vecs[:, :2]
        total_var = float(vals.sum()) or 1.0
        var_pc1 = float(vals[0]) / total_var
        var_pc2 = float(vals[1]) / total_var

        return (
            projected[:, 0].tolist(),
            projected[:, 1].tolist(),
            _FEAT_NAMES,
            vecs[:, 0].tolist(),
            vecs[:, 1].tolist(),
            var_pc1,
            var_pc2,
        )

    except Exception:
        # Simple fallback: BPM vs (aggressive - relaxed)
        xs = [(float(t.get("bpm") or 125) - 110) / 40 for t in tracks]
        ys = [
            float((t.get("mood") or {}).get("aggressive", 0.3))
            - float((t.get("mood") or {}).get("relaxed", 0.3))
            for t in tracks
        ]
        return xs, ys, _FEAT_NAMES, [1] + [0] * 8, [0] * 6 + [1, -1, 0], 0.4, 0.2


def _axis_label(feat_names: List[str], loadings: List[float], var: float) -> str:
    """Describe a PCA axis in human-readable terms using top positive/negative loadings."""
    if not feat_names or not loadings:
        return "Component"
    pairs = sorted(zip(loadings, feat_names), key=lambda x: -abs(x[0]))
    top_pos = [n for v, n in pairs if v > 0][:2]
    top_neg = [n for v, n in pairs if v < 0][:2]
    pos_str = " + ".join(top_pos) if top_pos else "—"
    neg_str = " + ".join(top_neg) if top_neg else "—"
    return f"{neg_str}  ←→  {pos_str}   ({var:.0%} var)"


def _scatter_section(tracks: List[Dict[str, Any]]) -> str:
    """Render the 2D acoustic map section as an SVG with interactive tooltips."""
    if len(tracks) < 2:
        return ""

    result = _compute_2d_projection(tracks)
    xs, ys = result[0], result[1]
    feat_names = result[2] if len(result) > 2 else _FEAT_NAMES
    pc1_load = result[3] if len(result) > 3 else []
    pc2_load = result[4] if len(result) > 4 else []
    var1 = result[5] if len(result) > 5 else 0.0
    var2 = result[6] if len(result) > 6 else 0.0

    W, H, PAD = 860, 500, 70

    # Normalise to SVG viewport with margin
    margin = 0.12
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    xr = max(x_max - x_min, 1e-6)
    yr = max(y_max - y_min, 1e-6)
    x_min -= xr * margin; x_max += xr * margin
    y_min -= yr * margin; y_max += yr * margin
    xr = x_max - x_min
    yr = y_max - y_min

    def to_svg(xi: float, yi: float):
        sx = PAD + (xi - x_min) / xr * (W - 2 * PAD)
        sy = H - PAD - (yi - y_min) / yr * (H - 2 * PAD)
        return round(sx, 1), round(sy, 1)

    svg_pts = [to_svg(x, y) for x, y in zip(xs, ys)]

    # ── Soft grid (faint lines at 25/50/75% across each axis) ──
    grid_lines = ""
    for frac in (0.25, 0.5, 0.75):
        gx = PAD + frac * (W - 2 * PAD)
        gy = H - PAD - frac * (H - 2 * PAD)
        grid_lines += (
            f'<line x1="{gx:.1f}" y1="{PAD}" x2="{gx:.1f}" y2="{H-PAD}" class="sc-grid"/>'
            f'<line x1="{PAD}" y1="{gy:.1f}" x2="{W-PAD}" y2="{gy:.1f}" class="sc-grid"/>'
        )

    # ── Set-order path (curved polyline) ──
    if len(svg_pts) >= 2:
        path_d = f"M{svg_pts[0][0]},{svg_pts[0][1]}"
        for i in range(1, len(svg_pts)):
            cx = (svg_pts[i-1][0] + svg_pts[i][0]) / 2
            path_d += (
                f" C{cx:.1f},{svg_pts[i-1][1]:.1f}"
                f" {cx:.1f},{svg_pts[i][1]:.1f}"
                f" {svg_pts[i][0]:.1f},{svg_pts[i][1]:.1f}"
            )
        set_path = f'<path d="{path_d}" class="sc-path" fill="none"/>'
    else:
        set_path = ""

    # ── Arrowheads at midpoints ──
    arrows = ""
    for i in range(len(svg_pts) - 1):
        x1, y1 = svg_pts[i]
        x2, y2 = svg_pts[i + 1]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        arrows += (
            f'<text x="{mx:.1f}" y="{my:.1f}" class="sc-arrow"'
            f' transform="rotate({angle:.0f},{mx:.1f},{my:.1f})"'
            f' text-anchor="middle" dominant-baseline="central">›</text>'
        )

    # ── Dots + position labels ──
    dots = ""
    for i, t in enumerate(tracks):
        sx, sy = svg_pts[i]
        key = t.get("key", "")
        energy = int(t.get("energy") or t.get("essentia_energy") or 5)
        kc = _KEY_COLORS.get(key, "#6060a0")
        r = round(7 + energy * 1.1, 1)
        pos = t.get("position", i + 1)

        # Escape data attrs
        da = _esc(t.get("artist", ""))
        dt = _esc(t.get("title", ""))
        dk = _esc(key)
        db = f'{float(t.get("bpm") or 0):.0f}'
        de = str(energy)
        dd = _esc(t.get("dominant_mood", ""))
        dtags = _esc(", ".join((t.get("my_tags") or [])[:4]))
        drel = _esc(t.get("key_relation", ""))
        dscore = f'{float(t.get("transition_score") or 0):.0%}'

        dots += (
            f'<circle cx="{sx}" cy="{sy}" r="{r}" class="sc-dot"'
            f' style="fill:{kc};fill-opacity:0.82;stroke:#0a0a0f;stroke-width:2"'
            f' data-pos="{pos}" data-artist="{da}" data-title="{dt}"'
            f' data-key="{dk}" data-bpm="{db}" data-energy="{de}"'
            f' data-mood="{dd}" data-tags="{dtags}"'
            f' data-rel="{drel}" data-score="{dscore}"'
            f' onclick="scatterClick({pos})"/>'
        )
        # Position number (white text inside dot)
        dots += (
            f'<text x="{sx}" y="{sy}" class="sc-num"'
            f' text-anchor="middle" dominant-baseline="central"'
            f' style="pointer-events:none">{pos}</text>'
        )

    # ── Axis labels ──
    x_lbl = _axis_label(feat_names, pc1_load, var1)
    y_lbl = _axis_label(feat_names, pc2_load, var2)
    cx = (PAD + W - PAD) / 2
    cy = (PAD + H - PAD) / 2
    axis_markup = (
        f'<text x="{cx:.0f}" y="{H - 6}" class="sc-axis-lbl" text-anchor="middle">'
        f'PC1  {_esc(x_lbl)}</text>'
        f'<text x="12" y="{cy:.0f}" class="sc-axis-lbl" text-anchor="middle"'
        f' transform="rotate(-90,12,{cy:.0f})">PC2  {_esc(y_lbl)}</text>'
    )

    # ── Key-color legend ──
    keys_used = sorted(set(t.get("key", "") for t in tracks if t.get("key")))
    legend_items = "".join(
        f'<span class="sc-leg-item">'
        f'<svg width="12" height="12" style="display:inline-block;vertical-align:middle;margin-right:3px">'
        f'<circle cx="6" cy="6" r="5" fill="{_KEY_COLORS.get(k, "#888")}"/></svg>'
        f'{_esc(k)}</span>'
        for k in keys_used
    )

    return f"""<section class="scatter-section">
    <div class="section-label" onclick="toggleSection(this)">2D TRACK MAP <span class="expand-icon small">▸</span></div>
    <div class="scatter-body open">
      <p class="scatter-sub">Acoustic similarity projection (PCA) — dot size = energy · color = Camelot key · dashed curve = set play order<br>
      Click a dot or an energy-arc point to jump to that track card below.</p>
      <div class="sc-legend">{legend_items}</div>
      <div class="sc-wrap">
        <svg viewBox="0 0 {W} {H}" class="sc-svg" preserveAspectRatio="xMidYMid meet">
          <defs>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="blur"/>
              <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
          </defs>
          {grid_lines}
          {set_path}
          {arrows}
          {dots}
          {axis_markup}
        </svg>
        <div id="sc-tip" class="sc-tooltip"></div>
      </div>
    </div>
  </section>"""


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: #0a0a0f;
  color: #e0e0e8;
  font-family: 'SF Pro Display', 'Inter', 'Segoe UI', system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.5;
}

.page {
  max-width: 900px;
  margin: 0 auto;
  padding: 24px 20px 60px;
}

/* ── HEADER ── */
header {
  background: linear-gradient(135deg, #13131f 0%, #1a1a2e 100%);
  border: 1px solid #2a2a3a;
  border-radius: 16px;
  padding: 28px 32px 20px;
  margin-bottom: 24px;
}
.header-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; flex-wrap: wrap; }
.set-label { font-size: 10px; letter-spacing: 3px; color: #6b6b8a; font-weight: 700; margin-bottom: 4px; }
h1 { font-size: 26px; font-weight: 800; color: #f0f0ff; margin-bottom: 6px; }
.prompt-text { color: #7878a0; font-style: italic; font-size: 13px; max-width: 480px; }
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px 16px; margin-bottom: 10px; }
.stat { text-align: center; }
.stat-val { display: block; font-size: 22px; font-weight: 800; color: #a78bfa; }
.stat-lbl { font-size: 10px; letter-spacing: 1px; color: #5a5a7a; text-transform: uppercase; }
.score-good { color: #4ade80 !important; }
.score-ok   { color: #fbbf24 !important; }
.score-warn { color: #f87171 !important; }
.bpm-range  { font-size: 12px; color: #6868a0; margin-bottom: 8px; text-align: right; }
.genre-row  { display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; }
.genre-pill { background: #1e1e30; border: 1px solid #3a3a5a; border-radius: 12px; padding: 2px 10px; font-size: 11px; color: #a0a0c0; }
.genre-pill b { color: #c0b0ff; }
.generated-at { font-size: 10px; color: #3a3a5a; margin-top: 14px; border-top: 1px solid #1e1e2e; padding-top: 10px; }

/* ── SECTION LABELS ── */
.section-label {
  font-size: 10px; letter-spacing: 3px; color: #4a4a6a; font-weight: 700;
  text-transform: uppercase; margin-bottom: 14px;
  display: flex; align-items: center; gap: 8px; cursor: pointer;
}

/* ── ENERGY ARC ── */
.energy-section { margin-bottom: 28px; }
.arc-wrap { position: relative; }
.arc-svg { width: 100%; height: auto; display: block; }
.grid-line { stroke: #1e1e2e; stroke-width: 1; }
.grid-lbl { fill: #3a3a5a; font-size: 10px; text-anchor: end; }
.arc-dot {
  fill: #a78bfa; stroke: #0a0a0f; stroke-width: 2; cursor: pointer;
  transition: r 0.15s;
}
.arc-dot:hover { r: 7; fill: #c4b5fd; }
.arc-label { fill: #3a3a5a; font-size: 9px; text-anchor: middle; }
.arc-tooltip {
  display: none; position: absolute; background: #1e1e30; border: 1px solid #3a3a5a;
  border-radius: 6px; padding: 4px 10px; font-size: 12px; color: #c0c0e0;
  pointer-events: none; z-index: 10;
}

/* ── FLOW ── */
.flow-section { margin-bottom: 28px; }
.flow { display: flex; flex-direction: column; gap: 0; }

/* ── TRACK CARD ── */
.track-card {
  background: #10101c;
  border: 1px solid #1e1e2e;
  border-radius: 12px;
  overflow: hidden;
  transition: border-color 0.2s;
}
.track-card:hover { border-color: #3a3a5a; }
.track-card.opening-track { border-color: #5b4fcf; }

.card-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  cursor: pointer;
  user-select: none;
  flex-wrap: wrap;
}
.card-header:hover { background: #14141f; }

.pos-badge {
  background: #1a1a2e;
  color: #6060a0;
  border-radius: 6px;
  width: 32px; height: 32px;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 700; flex-shrink: 0;
}
.track-identity { flex: 1; min-width: 200px; }
.track-artist { color: #e0e0f0; font-weight: 600; }
.track-sep { color: #3a3a5a; }
.track-title { color: #b0b0d0; }

.track-meta-row {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  flex-shrink: 0;
}
.bpm-badge { background: #1a1a2e; color: #8080c0; border-radius: 6px; padding: 2px 8px; font-size: 12px; font-weight: 600; }
.key-badge {
  border-radius: 6px; padding: 2px 8px; font-size: 12px; font-weight: 700;
  background: color-mix(in srgb, var(--kc) 20%, #1a1a2e);
  color: var(--kc);
  border: 1px solid color-mix(in srgb, var(--kc) 50%, transparent);
}
.energy-dots { font-size: 11px; color: #a78bfa; letter-spacing: -1px; }
.genre-small { color: #5a5a8a; font-size: 11px; }
.duration-small { color: #4a4a6a; font-size: 11px; }

.expand-icon { color: #3a3a5a; font-size: 12px; margin-left: auto; transition: transform 0.2s; flex-shrink: 0; }
.expand-icon.open { transform: rotate(90deg); }

/* Card body (hidden by default) */
.card-body { display: none; border-top: 1px solid #1a1a2a; }
.card-body.open { display: block; }
.card-inner { padding: 16px 20px; display: flex; flex-direction: column; gap: 12px; }

/* Tags, meta */
.my-tags { display: flex; flex-wrap: wrap; gap: 6px; }
.tag-chip { background: #1e1e35; border: 1px solid #3030508a; border-radius: 10px; padding: 2px 10px; font-size: 11px; color: #8888c0; }
.extra-meta { display: flex; flex-wrap: wrap; gap: 12px; }
.meta-item { font-size: 12px; color: #5a5a8a; }
.meta-item b { color: #9090c0; }

/* Mood bars */
.mood-section { display: flex; flex-direction: column; gap: 5px; }
.mood-row { display: flex; align-items: center; gap: 8px; }
.mood-lbl { width: 64px; font-size: 11px; color: #5a5a8a; text-align: right; }
.mood-bar-bg { flex: 1; height: 6px; background: #1a1a28; border-radius: 3px; overflow: hidden; }
.mood-bar-fill { height: 100%; background: linear-gradient(90deg, #6d28d9, #a78bfa); border-radius: 3px; }
.mood-val { width: 32px; font-size: 10px; color: #4a4a7a; text-align: right; }

/* Sub-sections */
.sub-section { display: flex; flex-direction: column; gap: 5px; }
.sub-label { font-size: 10px; letter-spacing: 2px; color: #3a3a5a; text-transform: uppercase; font-weight: 700; }
.chip-row { display: flex; flex-wrap: wrap; gap: 5px; }
.score-chip { background: #161624; border: 1px solid #2a2a3a; border-radius: 8px; padding: 2px 8px; font-size: 11px; color: #6868a0; }
.score-chip b { color: #8888b0; }

/* ── CONNECTOR ── */
.connector {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 2px 0;
  position: relative;
}
.conn-line { width: 2px; height: 16px; background: #1e1e2e; }
.conn-info {
  display: flex; align-items: center; gap: 8px;
  background: #0d0d18; border: 1px solid #1a1a28; border-radius: 8px;
  padding: 3px 12px; font-size: 11px;
}
.conn-rel { color: #4a4a7a; }
.conn-score { font-weight: 700; }
.conn-notes { color: #3a3a5a; font-style: italic; max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.conn-arrow { color: #2a2a3a; font-size: 10px; margin-top: 2px; }

/* ── INTENT SECTION ── */
.intent-section { margin-top: 28px; }
.expand-icon.small { font-size: 10px; }
.intent-body { display: none; background: #0d0d18; border: 1px solid #1a1a28; border-radius: 10px; overflow: hidden; }
.intent-body.open { display: block; }
.intent-table { width: 100%; border-collapse: collapse; }
.intent-table tr:not(:last-child) td { border-bottom: 1px solid #1a1a28; }
.ikey { padding: 8px 16px; color: #4a4a7a; font-size: 11px; letter-spacing: 1px; text-transform: uppercase; width: 160px; vertical-align: top; }
.ival { padding: 8px 16px; color: #8080a8; font-size: 12px; word-break: break-word; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2a2a3a; border-radius: 3px; }

/* Responsive */
@media (max-width: 600px) {
  .header-top { flex-direction: column; }
  .stat-grid { grid-template-columns: repeat(2, 1fr); }
  .card-header { gap: 8px; }
  .track-meta-row { gap: 5px; }
}

/* ── 2D SCATTER ── */
.scatter-section { margin-bottom: 28px; }
.scatter-body { display: none; }
.scatter-body.open { display: block; }
.scatter-sub { font-size: 12px; color: #4a4a6a; margin-bottom: 12px; line-height: 1.6; }
.sc-legend { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
.sc-leg-item { font-size: 11px; color: #5a5a8a; display: flex; align-items: center; gap: 2px; }
.sc-wrap { position: relative; }
.sc-svg { width: 100%; height: auto; display: block; background: #0c0c18; border-radius: 10px; border: 1px solid #1a1a2a; }
.sc-grid { stroke: #161624; stroke-width: 1; }
.sc-path { stroke: #3b3b6a; stroke-width: 2; stroke-dasharray: 5 4; opacity: 0.6; }
.sc-arrow { fill: #3b3b6a; font-size: 14px; opacity: 0.7; }
.sc-dot { cursor: pointer; transition: r 0.15s, filter 0.15s; }
.sc-dot:hover { filter: url(#glow); }
.sc-num {
  fill: #fff; font-size: 9px; font-weight: 800;
  text-shadow: 0 0 3px #000;
  pointer-events: none;
}
.sc-axis-lbl { fill: #2e2e4a; font-size: 10px; }
.sc-tooltip {
  display: none; position: absolute; z-index: 20;
  background: #15152a; border: 1px solid #3030508a;
  border-radius: 10px; padding: 10px 14px; min-width: 200px;
  pointer-events: none; box-shadow: 0 4px 24px #00000088;
}
.sc-tip-pos  { font-size: 10px; color: #5050a0; font-weight: 700; letter-spacing: 2px; margin-bottom: 4px; }
.sc-tip-name { font-size: 14px; color: #e0e0f0; font-weight: 600; margin-bottom: 6px; line-height: 1.3; }
.sc-tip-row  { font-size: 11px; color: #6060a0; margin-bottom: 2px; }
.sc-tip-row b { color: #9090c0; }
.sc-tip-tags { font-size: 10px; color: #404068; margin-top: 5px; font-style: italic; }
"""

# ---------------------------------------------------------------------------
# JS
# ---------------------------------------------------------------------------

_JS = """
function toggleCard(header) {
  const body = header.nextElementSibling;
  const icon = header.querySelector('.expand-icon');
  if (!body) return;
  body.classList.toggle('open');
  icon.classList.toggle('open');
}

function toggleSection(label) {
  const body = label.nextElementSibling;
  const icon = label.querySelector('.expand-icon');
  if (!body) return;
  body.classList.toggle('open');
  if (icon) icon.classList.toggle('open');
}

// Arc dot tooltips
document.querySelectorAll('.arc-dot').forEach(dot => {
  const tip = document.getElementById('arc-tip');
  dot.addEventListener('mouseenter', (e) => {
    const pos = parseInt(dot.dataset.pos) + 1;
    const energy = dot.dataset.energy;
    tip.textContent = `Track ${pos} · Energy ${energy}`;
    tip.style.display = 'block';
  });
  dot.addEventListener('mousemove', (e) => {
    const rect = dot.closest('.arc-wrap').getBoundingClientRect();
    tip.style.left = (e.clientX - rect.left + 12) + 'px';
    tip.style.top  = (e.clientY - rect.top  - 24) + 'px';
  });
  dot.addEventListener('mouseleave', () => {
    tip.style.display = 'none';
  });
  dot.addEventListener('click', () => {
    const pos = parseInt(dot.dataset.pos) + 1;
    const card = document.getElementById('track-' + pos);
    if (card) {
      card.scrollIntoView({ behavior: 'smooth', block: 'center' });
      const header = card.querySelector('.card-header');
      const body   = card.querySelector('.card-body');
      const icon   = card.querySelector('.expand-icon');
      if (body && !body.classList.contains('open')) {
        body.classList.add('open');
        if (icon) icon.classList.add('open');
      }
    }
  });
});

// ── Scatter dot tooltips ──
const scTip = document.getElementById('sc-tip');
document.querySelectorAll('.sc-dot').forEach(dot => {
  dot.addEventListener('mouseenter', () => {
    const pos   = dot.dataset.pos;
    const key   = dot.dataset.key;
    const bpm   = dot.dataset.bpm;
    const en    = dot.dataset.energy;
    const mood  = dot.dataset.mood;
    const rel   = dot.dataset.rel;
    const score = dot.dataset.score;
    const tags  = dot.dataset.tags;
    scTip.innerHTML =
      `<div class="sc-tip-pos">TRACK ${pos}</div>` +
      `<div class="sc-tip-name">${dot.dataset.artist}<br><span style="color:#7070a0">${dot.dataset.title}</span></div>` +
      `<div class="sc-tip-row">Key <b>${key}</b> · <b>${bpm} BPM</b> · Energy <b>${en}/10</b></div>` +
      (mood  ? `<div class="sc-tip-row">Mood <b>${mood}</b></div>` : '') +
      (rel && rel !== 'start' ? `<div class="sc-tip-row">Transition <b>${rel}</b> · match <b>${score}</b></div>` : '') +
      (tags  ? `<div class="sc-tip-tags">${tags}</div>` : '');
    scTip.style.display = 'block';
  });
  dot.addEventListener('mousemove', e => {
    const wrap = dot.closest('.sc-wrap').getBoundingClientRect();
    let lx = e.clientX - wrap.left + 16;
    let ly = e.clientY - wrap.top  - 10;
    // keep inside wrap
    lx = Math.min(lx, wrap.width - 230);
    ly = Math.max(ly, 4);
    scTip.style.left = lx + 'px';
    scTip.style.top  = ly + 'px';
  });
  dot.addEventListener('mouseleave', () => {
    scTip.style.display = 'none';
  });
});

// Click scatter dot → expand track card in flow
function scatterClick(pos) {
  const card = document.getElementById('track-' + pos);
  if (!card) return;
  card.scrollIntoView({ behavior: 'smooth', block: 'center' });
  const body = card.querySelector('.card-body');
  const icon = card.querySelector('.expand-icon');
  if (body && !body.classList.contains('open')) {
    body.classList.add('open');
    if (icon) icon.classList.add('open');
  }
}
"""
