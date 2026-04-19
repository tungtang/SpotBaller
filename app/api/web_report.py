"""
HTML report pages and safe static file serving for job outputs (browser-friendly).
"""

from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse, HTMLResponse

ALLOWED_JOB_FILES = frozenset(
    {
        "annotated.mp4",
        "stats.json",
        "stats_by_track.json",
        "events.json",
        "tracks.json",
        "player_identity_map.json",
        "team_box_score.json",
        "stats.csv",
        "team_box_score_players.csv",
        "action_hints_long.csv",
        "pipeline_performance.csv",
        "pipeline.json",
        "action_hints.json",
        "videomae_aux.json",
        "job.json",
    }
)

# --- Shared design system (dark, high-contrast, responsive) ---
SHARED_CSS = """
:root {
  --bg: #0c0f14;
  --surface: #151a22;
  --surface2: #1c2330;
  --border: #2a3444;
  --text: #e8edf4;
  --muted: #8b9aaf;
  --accent: #3b82f6;
  --accent-dim: #2563eb;
  --success: #34d399;
  --warning: #fbbf24;
  --radius: 12px;
  --shadow: 0 8px 32px rgba(0,0,0,.35);
  --font: "Segoe UI", system-ui, -apple-system, sans-serif;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  font-family: var(--font);
  background: var(--bg);
  color: var(--text);
  line-height: 1.55;
  min-height: 100vh;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.wrap { max-width: 72rem; margin: 0 auto; padding: 1.25rem 1.25rem 3rem; }

/* Top bar */
.topbar {
  position: sticky; top: 0; z-index: 100;
  background: rgba(12,15,20,.92);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--border);
  padding: 0.65rem 0;
  margin: 0 -1.25rem 1.5rem;
  padding-left: 1.25rem; padding-right: 1.25rem;
}
.topbar-inner {
  max-width: 72rem; margin: 0 auto;
  display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.75rem;
}
.brand {
  font-weight: 700; font-size: 1.05rem; letter-spacing: -0.02em;
  color: var(--text) !important; text-decoration: none !important;
}
.brand span { color: var(--accent); }
.nav-links { display: flex; gap: 0.25rem; flex-wrap: wrap; }
.nav-links a {
  padding: 0.45rem 0.85rem; border-radius: 8px; font-size: 0.9rem;
  color: var(--muted) !important; text-decoration: none !important;
}
.nav-links a:hover { background: var(--surface2); color: var(--text) !important; }
.nav-links a.active { background: var(--surface2); color: var(--accent) !important; font-weight: 600; }

h1 { font-size: 1.65rem; font-weight: 700; letter-spacing: -0.03em; margin: 0 0 0.35rem; }
h2 { font-size: 1.15rem; font-weight: 600; margin: 2rem 0 0.75rem; color: #f1f5f9; }
.lede { color: var(--muted); font-size: 1rem; margin: 0 0 1.5rem; max-width: 42rem; }

/* Cards */
.grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); }
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.15rem 1.25rem;
  box-shadow: var(--shadow);
  transition: border-color .15s, transform .15s;
}
.card:hover { border-color: #3d4f66; transform: translateY(-2px); }
.card h3 { margin: 0 0 0.4rem; font-size: 1rem; font-weight: 600; }
.card p { margin: 0; font-size: 0.88rem; color: var(--muted); }
.card .cta { display: inline-block; margin-top: 0.85rem; font-weight: 600; font-size: 0.9rem; }

/* Hero */
.hero {
  padding: 1.5rem 0 0.5rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.75rem;
}
.hero-badge {
  display: inline-block; font-size: 0.72rem; text-transform: uppercase; letter-spacing: .08em;
  color: var(--accent); background: rgba(59,130,246,.12); padding: 0.25rem 0.55rem; border-radius: 6px; margin-bottom: 0.75rem;
}

/* Result rows / job list */
.section { margin-bottom: 2.25rem; }
.section-head {
  display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.75rem;
  margin-bottom: 1rem;
}
.section-head h2 { margin: 0; }
.search {
  padding: 0.5rem 0.85rem; border-radius: 8px; border: 1px solid var(--border);
  background: var(--surface); color: var(--text); font-size: 0.9rem; min-width: 200px; max-width: 100%;
}
.search::placeholder { color: var(--muted); }
.job-list { display: flex; flex-direction: column; gap: 0.5rem; }
.job-row {
  display: flex; align-items: center; justify-content: space-between; gap: 1rem;
  padding: 0.85rem 1rem; background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; flex-wrap: wrap;
}
.job-row:hover { border-color: #3d4f66; }
.job-row a.main { font-weight: 600; font-family: ui-monospace, monospace; font-size: 0.88rem; word-break: break-all; }
.job-meta { font-size: 0.8rem; color: var(--muted); }
.pill { font-size: 0.72rem; padding: 0.2rem 0.5rem; border-radius: 999px; font-weight: 600; }
.pill.ok { background: rgba(52,211,153,.15); color: var(--success); }
.pill.pending { background: rgba(251,191,36,.12); color: var(--warning); }

/* Job report */
.breadcrumb { font-size: 0.85rem; color: var(--muted); margin-bottom: 1rem; }
.breadcrumb a { color: var(--muted); }
.breadcrumb a:hover { color: var(--accent); }

.subnav {
  display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 1rem 0 1.5rem;
  padding: 0.35rem; background: var(--surface); border-radius: 10px; border: 1px solid var(--border);
}
.subnav a {
  padding: 0.45rem 0.75rem; border-radius: 7px; font-size: 0.85rem; color: var(--muted) !important;
  text-decoration: none !important;
}
.subnav a:hover { background: var(--surface2); color: var(--text) !important; }

.video-card {
  background: #000; border-radius: var(--radius); overflow: hidden; border: 1px solid var(--border);
  margin: 0.5rem 0 1rem;
}
video { width: 100%; max-height: 72vh; display: block; vertical-align: middle; }

table.data-table { border-collapse: collapse; width: 100%; font-size: 0.85rem; margin: 0.75rem 0; }
table.data-table th, table.data-table td { border: 1px solid var(--border); padding: 0.5rem 0.65rem; text-align: left; }
table.data-table th { background: var(--surface2); color: var(--muted); font-weight: 600; }
table.data-table tbody tr:nth-child(even) { background: rgba(255,255,255,.02); }
table.data-table tbody tr:hover { background: rgba(59,130,246,.06); }

code { background: var(--surface2); padding: 0.12em 0.4em; border-radius: 5px; font-size: 0.86em; }
pre.json {
  overflow: auto; max-height: 22rem; background: #0a0d12; padding: 1rem; border-radius: 10px;
  font-size: 0.78rem; border: 1px solid var(--border); line-height: 1.45;
}
details { margin: 0.6rem 0; border: 1px solid var(--border); border-radius: 10px; padding: 0.5rem 1rem; background: var(--surface); }
details summary { cursor: pointer; font-weight: 600; padding: 0.35rem 0; }
.file-links { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.75rem; }
.file-links a {
  display: inline-block; padding: 0.4rem 0.75rem; background: var(--surface2);
  border-radius: 8px; font-size: 0.85rem; border: 1px solid var(--border);
}
.file-links a:hover { border-color: var(--accent); text-decoration: none; }
.muted { color: var(--muted); font-size: 0.9rem; }
footer.page {
  margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--border);
  font-size: 0.8rem; color: var(--muted);
}
.empty { padding: 2rem; text-align: center; color: var(--muted); border: 1px dashed var(--border); border-radius: var(--radius); }
"""


def resolve_job_dir(job_root: Path, job_id: str) -> Path:
    """Ensure job directory exists and stays under job_root (no path traversal)."""
    if not job_id or ".." in job_id or "/" in job_id or "\\" in job_id:
        raise HTTPException(status_code=400, detail="Invalid job id")
    resolved = (job_root / job_id).resolve()
    root = job_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid job path") from exc
    if not resolved.is_dir():
        raise HTTPException(status_code=404, detail="Job not found")
    return resolved


def resolve_local_run_dir(runtime_root: Path, run_id: str) -> Path:
    """CLI outputs under runtime/<run_id>/ (e.g. smoke tests). Excludes jobs/ and uploads/."""
    if not run_id or ".." in run_id or "/" in run_id or "\\" in run_id:
        raise HTTPException(status_code=400, detail="Invalid run id")
    if run_id in ("jobs", "uploads"):
        raise HTTPException(status_code=404, detail="Use /results for API jobs")
    resolved = (runtime_root / run_id).resolve()
    root = runtime_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid path") from exc
    if not resolved.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    if not (resolved / "stats.json").is_file():
        raise HTTPException(status_code=404, detail="No stats in this folder")
    return resolved


def job_file_response(job_dir: Path, filename: str) -> FileResponse:
    if filename not in ALLOWED_JOB_FILES:
        raise HTTPException(status_code=404, detail="File not allowed")
    path = job_dir / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if filename.endswith(".mp4"):
        media = "video/mp4"
    elif filename.endswith(".csv"):
        media = "text/csv"
    elif filename.endswith(".json"):
        media = "application/json"
    else:
        media = "application/octet-stream"
    return FileResponse(path, media_type=media, filename=filename)


def _read_json(path: Path) -> dict | list | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt_time(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


def _html_table(rows: list[dict], preferred_cols: list[str], limit: int = 200) -> str:
    if not rows:
        return "<p class='muted'>No rows.</p>"
    cols = [c for c in preferred_cols if c in rows[0]]
    if not cols:
        cols = list(rows[0].keys())[:12]
    thead = "".join(f"<th>{escape(str(k))}</th>" for k in cols)
    body_rows = []
    for r in rows[:limit]:
        cells = "".join(f"<td>{escape(str(r.get(k, '')))}</td>" for k in cols)
        body_rows.append(f"<tr>{cells}</tr>")
    out = f'<table class="data-table"><thead><tr>{thead}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'
    if len(rows) > limit:
        out += f"<p class='muted'>Showing {limit} of {len(rows)} rows.</p>"
    return out


def _render_team_box_score(team: object) -> str:
    if not isinstance(team, list) or not team:
        return "<p class='muted'>No team box score available.</p>"
    blocks: list[str] = []
    for t in team:
        if not isinstance(t, dict):
            continue
        team_name = escape(str(t.get("team_name", "Unknown Team")))
        totals = t.get("team_totals", {}) if isinstance(t.get("team_totals"), dict) else {}
        totals_line = " · ".join(
            [
                f"PTS {int(float(totals.get('pts', 0) or 0))}",
                f"REB {int(float(totals.get('reb', 0) or 0))}",
                f"AST {int(float(totals.get('ast', 0) or 0))}",
                f"TOV {int(float(totals.get('tov', 0) or 0))}",
            ]
        )
        players = t.get("players") if isinstance(t.get("players"), list) else []
        table = _html_table(
            players, ["player_number", "player_label", "minutes_on_court", "pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga", "fg_pct"]
        )
        blocks.append(f"<h3>{team_name}</h3><p class='muted'>{totals_line}</p>{table}")
    return "".join(blocks) if blocks else "<p class='muted'>No team rows.</p>"


def _render_pipeline_summary(pipeline: object) -> str:
    if not isinstance(pipeline, dict):
        return "<p class='muted'>No pipeline metadata.</p>"
    ps = pipeline.get("pretrained_stack") if isinstance(pipeline.get("pretrained_stack"), dict) else {}
    perf = pipeline.get("performance") if isinstance(pipeline.get("performance"), dict) else {}
    stage = perf.get("stage_s") if isinstance(perf.get("stage_s"), dict) else {}
    rows = [
        {"metric": "weights", "value": pipeline.get("detection_weights", "—")},
        {"metric": "tracker_backend", "value": pipeline.get("tracker_backend", "—")},
        {
            "metric": "pretrained_loaded",
            "value": f"{sum(bool(ps.get(k)) for k in ('siglip_loaded', 'trocr_loaded', 'videomae_loaded'))}/3",
        },
        {"metric": "fps_effective", "value": perf.get("fps_effective", "—")},
    ]
    summary = _html_table(rows, ["metric", "value"], limit=20)
    if stage:
        stage_rows = [{"stage": k, "seconds": v} for k, v in sorted(stage.items(), key=lambda x: float(x[1]), reverse=True)]
        summary += "<h3>Stage timing</h3>" + _html_table(stage_rows, ["stage", "seconds"], limit=30)
    return summary


def _render_action_hints(hints: object) -> str:
    if not isinstance(hints, list) or not hints:
        return "<p class='muted'>No action hints emitted.</p>"
    rows: list[dict] = []
    for h in hints:
        if not isinstance(h, dict):
            continue
        fi = h.get("frame_index")
        scores = h.get("siglip_action_scores")
        if not isinstance(scores, dict):
            continue
        for label, prob in scores.items():
            rows.append({"frame": fi, "action": label, "probability": prob})
    if not rows:
        return "<p class='muted'>No action hint scores found.</p>"
    return _html_table(rows, ["frame", "action", "probability"], limit=300)


def build_landing_html() -> HTMLResponse:
    return HTMLResponse(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <meta name="color-scheme" content="dark"/>
  <title>SpotBaller · Basketball analytics</title>
  <style>{SHARED_CSS}</style>
</head>
<body>
  <div class="wrap">
    <header class="topbar">
      <div class="topbar-inner">
        <a class="brand" href="/">Spot<span>Baller</span></a>
        <nav class="nav-links" aria-label="Primary">
          <a href="/" class="active">Home</a>
          <a href="/results">Results</a>
          <a href="/docs">API</a>
        </nav>
      </div>
    </header>

    <div class="hero">
      <div class="hero-badge">Local-first pipeline</div>
      <h1>Basketball video analytics</h1>
      <p class="lede">Upload games, run detection and tracking, then explore annotated video, per-player stats, and exports in your browser.</p>
    </div>

    <div class="grid">
      <article class="card">
        <h3>Results hub</h3>
        <p>Browse API jobs and CLI runs (smoke tests). Watch annotated video and download JSON.</p>
        <a class="cta" href="/results">Open results →</a>
      </article>
      <article class="card">
        <h3>REST API</h3>
        <p>Upload videos, create jobs, fetch stats and team breakdowns programmatically.</p>
        <a class="cta" href="/docs">OpenAPI docs →</a>
      </article>
      <article class="card">
        <h3>Health</h3>
        <p>Quick JSON check that the server is up.</p>
        <a class="cta" href="/health">GET /health →</a>
      </article>
      <article class="card">
        <h3>Streamlit UI</h3>
        <p>Richer upload and charts (separate process). Run: <code>streamlit run app/ui/dashboard.py</code></p>
        <a class="cta" href="http://127.0.0.1:8501" target="_blank" rel="noopener">Open localhost:8501 →</a>
      </article>
    </div>

    <footer class="page">SpotBaller · FastAPI + YOLOv8 pipeline</footer>
  </div>
</body>
</html>"""
    )


def build_combined_index_html(runtime_root: Path, job_dir: Path) -> str:
    """Jobs from runtime/jobs + local runs (other dirs with stats.json)."""
    api_rows: list[str] = []
    if job_dir.is_dir():
        for p in sorted(job_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not p.is_dir():
                continue
            jid = escape(p.name)
            done = (p / "stats.json").is_file()
            ts = _fmt_time(p.stat().st_mtime)
            pill = '<span class="pill ok">Ready</span>' if done else '<span class="pill pending">In progress</span>'
            api_rows.append(
                f'<div class="job-row" data-search="{jid.lower()}">'
                f'<div><a class="main" href="/results/{jid}">{jid}</a>'
                f'<div class="job-meta">{escape(ts)} · API job</div></div><div>{pill}</div></div>'
            )

    local_rows: list[str] = []
    if runtime_root.is_dir():
        for p in sorted(runtime_root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not p.is_dir() or p.name in ("jobs", "uploads"):
                continue
            if not (p / "stats.json").is_file():
                continue
            rid = escape(p.name)
            ts = _fmt_time(p.stat().st_mtime)
            local_rows.append(
                f'<div class="job-row" data-search="{rid.lower()}">'
                f'<div><a class="main" href="/results/local/{rid}">{rid}</a>'
                f'<div class="job-meta">{escape(ts)} · CLI / local output</div></div>'
                f'<div><span class="pill ok">Ready</span></div></div>'
            )

    api_block = (
        "".join(api_rows)
        if api_rows
        else '<div class="empty">No API jobs yet. POST /videos and /jobs, or use the Streamlit UI.</div>'
    )
    local_block = (
        "".join(local_rows)
        if local_rows
        else '<div class="empty">No local runs found. Use <code>python -m app.run_local --out runtime/my_run</code>.</div>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <meta name="color-scheme" content="dark"/>
  <title>Results · SpotBaller</title>
  <style>{SHARED_CSS}</style>
</head>
<body>
  <div class="wrap">
    <header class="topbar">
      <div class="topbar-inner">
        <a class="brand" href="/">Spot<span>Baller</span></a>
        <nav class="nav-links" aria-label="Primary">
          <a href="/">Home</a>
          <a href="/results" class="active">Results</a>
          <a href="/docs">API</a>
        </nav>
      </div>
    </header>

    <div class="hero">
      <h1>Results</h1>
      <p class="lede">Open any run to view annotated video, stats tables, and JSON exports. Filter the list below.</p>
    </div>

    <label class="muted" for="q" style="display:block;margin-bottom:0.35rem;">Filter</label>
    <input type="search" id="q" class="search" placeholder="Search by id or name…" autocomplete="off"
      style="width:100%;max-width:28rem;margin-bottom:2rem;"/>

    <section class="section" aria-labelledby="api-heading">
      <div class="section-head">
        <h2 id="api-heading">API jobs</h2>
      </div>
      <div class="job-list" id="list-api">{api_block}</div>
    </section>

    <section class="section" aria-labelledby="local-heading">
      <div class="section-head">
        <h2 id="local-heading">Local &amp; CLI runs</h2>
      </div>
      <p class="muted" style="margin-top:-0.5rem;margin-bottom:1rem;">Folders under <code>runtime/</code> with <code>stats.json</code> (e.g. smoke tests).</p>
      <div class="job-list" id="list-local">{local_block}</div>
    </section>

    <footer class="page"><a href="/">← Home</a> · <a href="/docs">API docs</a></footer>
  </div>
  <script>
    (function() {{
      const q = document.getElementById('q');
      function filter() {{
        const term = (q.value || '').toLowerCase().trim();
        document.querySelectorAll('.job-row').forEach(function(row) {{
          const hay = (row.getAttribute('data-search') || '');
          row.style.display = !term || hay.includes(term) ? '' : 'none';
        }});
      }}
      q.addEventListener('input', filter);
    }})();
  </script>
</body>
</html>"""


def build_job_report_html(
    job_id: str,
    job_dir: Path,
    media_prefix: str,
    *,
    report_kind: str = "job",
    breadcrumb_href: str = "/results",
    breadcrumb_label: str = "Results",
) -> HTMLResponse:
    """Single-page report: video, stats, collapsible JSON, sticky section nav."""
    stats = _read_json(job_dir / "stats.json")
    pipeline = _read_json(job_dir / "pipeline.json")
    team = _read_json(job_dir / "team_box_score.json")
    job_meta = _read_json(job_dir / "job.json")
    hints = _read_json(job_dir / "action_hints.json")

    status = "unknown"
    if isinstance(job_meta, dict):
        status = str(job_meta.get("status", "unknown"))

    video_url = f"{media_prefix}/file/annotated.mp4"
    has_video = (job_dir / "annotated.mp4").is_file()

    table_html = ""
    if isinstance(stats, list) and stats:
        first_row = stats[0]
        if "jersey_key" in first_row or "identity_kind" in first_row:
            keys = [
                k
                for k in first_row.keys()
                if k
                in (
                    "player_label",
                    "jersey_key",
                    "team_name",
                    "merged_track_ids",
                    "merged_track_count",
                    "minutes_on_court",
                    "pts",
                    "fga",
                    "fgm",
                    "fg_pct",
                    "touches",
                )
            ]
        else:
            keys = [
                k
                for k in first_row.keys()
                if k in ("player_id", "minutes_on_court", "pts", "fga", "fgm", "fg_pct", "touches")
            ]
        if not keys:
            keys = list(first_row.keys())[:12]
        thead = "".join(f"<th>{escape(str(k))}</th>" for k in keys)
        trs = []
        for row in stats[:80]:
            cells = "".join(f"<td>{escape(str(row.get(k, '')))}</td>" for k in keys)
            trs.append(f"<tr>{cells}</tr>")
        table_html = f'<table class="data-table"><thead><tr>{thead}</tr></thead><tbody>{"".join(trs)}</tbody></table>'
        if len(stats) > 80:
            table_html += f"<p class='muted'>Showing 80 of {len(stats)} rows.</p>"

    def block(title: str, data: object) -> str:
        if data is None:
            return f"<details><summary>{escape(title)}</summary><p class='muted'>No file.</p></details>"
        raw = json.dumps(data, indent=2, ensure_ascii=False)
        return f"""<details><summary>{escape(title)}</summary>
<pre class="json">{escape(raw)}</pre></details>"""

    hints_note = ""
    if isinstance(hints, list) and hints:
        hints_note = f"<p class='muted'>{len(hints)} sampled SigLIP action frames.</p>"

    jid = escape(job_id)
    kind_label = "API job" if report_kind == "job" else "Local run"

    html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <meta name="color-scheme" content="dark"/>
  <title>{jid} · SpotBaller</title>
  <style>{SHARED_CSS}</style>
</head>
<body>
  <div class="wrap">
    <header class="topbar">
      <div class="topbar-inner">
        <a class="brand" href="/">Spot<span>Baller</span></a>
        <nav class="nav-links" aria-label="Primary">
          <a href="/">Home</a>
          <a href="/results" class="active">Results</a>
          <a href="/docs">API</a>
        </nav>
      </div>
    </header>

    <nav class="breadcrumb" aria-label="Breadcrumb">
      <a href="/">Home</a> / <a href="{escape(breadcrumb_href)}">{escape(breadcrumb_label)}</a> / <span>{jid}</span>
    </nav>

    <h1 style="margin-bottom:0.25rem;"><code>{jid}</code></h1>
    <p class="muted" style="margin-top:0;">{escape(kind_label)} · Status: <strong>{escape(status)}</strong></p>

    <nav class="subnav" aria-label="Page sections">
      <a href="#video">Video</a>
      <a href="#stats">Stats</a>
      <a href="#team">Team</a>
      <a href="#pipeline">Pipeline</a>
      <a href="#hints">Hints</a>
      <a href="#exports">Exports</a>
    </nav>

    <section id="video">
      <h2>Annotated video</h2>
      {"<div class='video-card'><video src='" + video_url + "' controls playsinline preload='metadata'></video></div>" if has_video else "<p class='muted'>No annotated.mp4 yet.</p>"}
    </section>

    <section id="stats">
      <h2>Player stats (by jersey)</h2>
      <p class="muted">Rows are merged by detected jersey number and team. Unresolved tracks stay separate.</p>
      {table_html if table_html else "<p class='muted'>No stats.json yet.</p>"}
    </section>

    <section id="team">
      <h2>Team box score</h2>
      {_render_team_box_score(team)}
      {block("team_box_score.json", team)}
    </section>

    <section id="pipeline">
      <h2>Pipeline</h2>
      {_render_pipeline_summary(pipeline)}
      {block("pipeline.json", pipeline)}
    </section>

    <section id="hints">
      <h2>Action hints</h2>
      {hints_note}
      {_render_action_hints(hints)}
      {block("action_hints.json", hints)}
    </section>

    <section id="exports">
      <h2>Download exports</h2>
      <p class="muted">Open in a new tab or save:</p>
      <div class="file-links">
        <a href="{media_prefix}/file/stats.json" target="_blank" rel="noopener">stats.json</a>
        <a href="{media_prefix}/file/stats_by_track.json" target="_blank" rel="noopener">stats_by_track.json</a>
        <a href="{media_prefix}/file/events.json" target="_blank" rel="noopener">events.json</a>
        <a href="{media_prefix}/file/tracks.json" target="_blank" rel="noopener">tracks.json</a>
        <a href="{media_prefix}/file/player_identity_map.json" target="_blank" rel="noopener">player_identity_map</a>
        <a href="{media_prefix}/file/pipeline.json" target="_blank" rel="noopener">pipeline.json</a>
        <a href="{media_prefix}/file/stats.csv" target="_blank" rel="noopener">stats.csv</a>
        <a href="{media_prefix}/file/team_box_score_players.csv" target="_blank" rel="noopener">team_box_score_players.csv</a>
        <a href="{media_prefix}/file/action_hints_long.csv" target="_blank" rel="noopener">action_hints_long.csv</a>
        <a href="{media_prefix}/file/pipeline_performance.csv" target="_blank" rel="noopener">pipeline_performance.csv</a>
        <a href="{media_prefix}/file/job.json" target="_blank" rel="noopener">job.json</a>
      </div>
    </section>

    <section>
      <h2>Job payload</h2>
      {block("job.json", job_meta)}
    </section>

    <footer class="page"><a href="{escape(breadcrumb_href)}">← {escape(breadcrumb_label)}</a></footer>
  </div>
</body>
</html>"""
    return HTMLResponse(content=html_page)


# Backwards compatibility
def landing_html() -> HTMLResponse:
    return build_landing_html()


def build_jobs_index_html(job_root: Path, results_prefix: str = "/results") -> str:
    """Deprecated single-list; use build_combined_index_html."""
    _ = results_prefix
    return build_combined_index_html(job_root.parent, job_root)
