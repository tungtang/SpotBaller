"""
HTML report pages and safe static file serving for job outputs (browser-friendly).
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse, HTMLResponse

ALLOWED_JOB_FILES = frozenset(
    {
        "annotated.mp4",
        "stats.json",
        "events.json",
        "tracks.json",
        "player_identity_map.json",
        "team_box_score.json",
        "stats.csv",
        "pipeline.json",
        "action_hints.json",
        "videomae_aux.json",
        "job.json",
    }
)


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


def build_jobs_index_html(job_root: Path, results_prefix: str = "/results") -> str:
    rows: list[str] = []
    if job_root.is_dir():
        for p in sorted(job_root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not p.is_dir():
                continue
            jid = escape(p.name)
            done = (p / "stats.json").is_file()
            badge = "✓" if done else "…"
            rows.append(
                f'<li><span class="badge">{badge}</span> '
                f'<a href="{results_prefix}/{jid}">{jid}</a></li>'
            )
    body = "\n".join(rows) if rows else "<li>No jobs yet. Upload and process a video via the API or Streamlit UI.</li>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Analysis jobs</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 56rem; margin: 2rem auto; padding: 0 1rem;
      background: #0f1419; color: #e8edf2; }}
    a {{ color: #3d9cf5; }}
    h1 {{ font-size: 1.5rem; }}
    ul {{ list-style: none; padding: 0; }}
    li {{ padding: 0.5rem 0; border-bottom: 1px solid #2d3844; }}
    .badge {{ color: #34d399; margin-right: 0.5rem; }}
    .nav {{ margin-bottom: 1.5rem; }}
    .nav a {{ margin-right: 1rem; }}
  </style>
</head>
<body>
  <nav class="nav">
    <a href="/">Home</a>
    <a href="/docs">API docs</a>
    <a href="{results_prefix}">All jobs</a>
  </nav>
  <h1>Analysis jobs</h1>
  <p>Open a job to watch the annotated video and inspect JSON outputs in the browser.</p>
  <ul>{body}</ul>
</body>
</html>"""


def build_job_report_html(job_id: str, job_dir: Path, media_prefix: str) -> HTMLResponse:
    """Single-page report: video, key metrics, collapsible JSON."""
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

    # Compact stats summary table
    table_html = ""
    if isinstance(stats, list) and stats:
        keys = [k for k in stats[0].keys() if k in ("player_id", "minutes_on_court", "pts", "fga", "fgm", "fg_pct", "touches")]
        if not keys:
            keys = list(stats[0].keys())[:12]
        thead = "".join(f"<th>{escape(str(k))}</th>" for k in keys)
        trs = []
        for row in stats[:50]:
            cells = "".join(f"<td>{escape(str(row.get(k, '')))}</td>" for k in keys)
            trs.append(f"<tr>{cells}</tr>")
        table_html = f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(trs)}</tbody></table>"
        if len(stats) > 50:
            table_html += f"<p class='muted'>Showing 50 of {len(stats)} rows. Download <code>stats.json</code> for full data.</p>"

    def block(title: str, data: object) -> str:
        if data is None:
            return f"<details><summary>{escape(title)}</summary><p class='muted'>No file.</p></details>"
        raw = json.dumps(data, indent=2, ensure_ascii=False)
        return f"""<details><summary>{escape(title)}</summary>
<pre class="json">{escape(raw)}</pre></details>"""

    hints_note = ""
    if isinstance(hints, list) and hints:
        hints_note = f"<p class='muted'>{len(hints)} sampled SigLIP action frames. See JSON below.</p>"

    jid = escape(job_id)
    html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Job {jid}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 60rem; margin: 0 auto; padding: 1rem 1.25rem 3rem;
      background: #0f1419; color: #e8edf2; line-height: 1.5; }}
    a {{ color: #3d9cf5; }}
    code {{ background: #1a222c; padding: 0.15em 0.4em; border-radius: 4px; }}
    .nav {{ margin-bottom: 1rem; font-size: 0.95rem; }}
    .nav a {{ margin-right: 1rem; }}
    h1 {{ font-size: 1.35rem; margin-bottom: 0.25rem; }}
    .status {{ color: #94a3b8; margin-bottom: 1.25rem; }}
    video {{ width: 100%; max-height: 70vh; background: #000; border-radius: 8px; }}
    .muted {{ color: #94a3b8; font-size: 0.9rem; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; margin: 1rem 0; }}
    th, td {{ border: 1px solid #2d3844; padding: 0.4rem 0.5rem; text-align: left; }}
    th {{ background: #141b24; color: #94a3b8; }}
    pre.json {{ overflow: auto; max-height: 24rem; background: #111820; padding: 1rem; border-radius: 8px;
      font-size: 0.8rem; border: 1px solid #2d3844; }}
    details {{ margin: 0.75rem 0; }}
    summary {{ cursor: pointer; font-weight: 600; }}
    .files {{ display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; margin-top: 1rem; }}
  </style>
</head>
<body>
  <nav class="nav">
    <a href="/">Home</a>
    <a href="/results">All jobs</a>
    <a href="/docs">API docs</a>
  </nav>
  <h1>Job <code>{jid}</code></h1>
  <p class="status">Status: <strong>{escape(status)}</strong></p>

  <h2>Annotated video</h2>
  {"<video src='" + video_url + "' controls playsinline></video>" if has_video else "<p class='muted'>No annotated.mp4 yet (job running or failed).</p>"}

  <h2>Player stats (preview)</h2>
  {table_html if table_html else "<p class='muted'>No stats.json yet.</p>"}

  <h2>Pipeline</h2>
  {block("pipeline.json", pipeline)}

  <h2>Team box score</h2>
  {block("team_box_score.json", team)}

  <h2>Action hints</h2>
  {hints_note}
  {block("action_hints.json", hints)}

  <h2>Raw exports</h2>
  <p class="muted">Download or open in a new tab:</p>
  <div class="files">
    <a href="{media_prefix}/file/stats.json" target="_blank">stats.json</a>
    <a href="{media_prefix}/file/events.json" target="_blank">events.json</a>
    <a href="{media_prefix}/file/tracks.json" target="_blank">tracks.json</a>
    <a href="{media_prefix}/file/player_identity_map.json" target="_blank">player_identity_map.json</a>
    <a href="{media_prefix}/file/pipeline.json" target="_blank">pipeline.json</a>
    <a href="{media_prefix}/file/stats.csv" target="_blank">stats.csv</a>
    <a href="{media_prefix}/file/job.json" target="_blank">job.json</a>
  </div>

  <h2>Full job payload</h2>
  {block("job.json", job_meta)}
</body>
</html>"""
    return HTMLResponse(content=html_page)


def landing_html() -> HTMLResponse:
    return HTMLResponse(
        """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Basketball Video Analytics</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 40rem; margin: 3rem auto; padding: 0 1rem;
      background: #0f1419; color: #e8edf2; line-height: 1.6; }
    a { color: #3d9cf5; }
    h1 { font-size: 1.6rem; }
    ul { padding-left: 1.2rem; }
  </style>
</head>
<body>
  <h1>Basketball Video Analytics</h1>
  <p>Use the API to upload videos and run analysis, then browse results here.</p>
  <ul>
    <li><a href="/results">Browse analysis results</a> — video + stats + JSON in the browser</li>
    <li><a href="/docs">OpenAPI / Swagger docs</a></li>
    <li><a href="/health">Health check</a> (JSON)</li>
  </ul>
  <p style="color:#94a3b8;font-size:0.9rem">Streamlit UI: run <code>streamlit run app/ui/dashboard.py</code></p>
</body>
</html>"""
    )
