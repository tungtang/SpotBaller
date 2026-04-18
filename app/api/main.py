from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from redis import Redis
from rq import Queue

from app.api.db import get_job_row, get_job_status_counts, init_db, upsert_job
from app.ml.model_shortlist import get_model_shortlist
from app.api.worker import process_job
from app.pipeline.pretrained_stack import pipeline_config_from_flags
from app.pipeline.video_analyzer import run_video_analysis
from app.api.web_report import (
    build_combined_index_html,
    build_job_report_html,
    build_landing_html,
    job_file_response,
    resolve_job_dir,
    resolve_local_run_dir,
)

app = FastAPI(title="Basketball Video Analytics API")

BASE_DIR = Path("runtime")
UPLOAD_DIR = BASE_DIR / "uploads"
JOB_DIR = BASE_DIR / "jobs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
JOB_DIR.mkdir(parents=True, exist_ok=True)
try:
    init_db()
except Exception:
    # Keep API usable in local/offline mode when DB isn't reachable.
    pass


def try_upsert_job(job_id: str, mode: str, status: str, video_path: str, result_path: str) -> None:
    """Best-effort DB write; local processing should still work without Postgres."""
    try:
        upsert_job(job_id, mode, status, video_path, result_path)
    except Exception:
        pass


@app.get("/", response_class=HTMLResponse)
def root_page() -> HTMLResponse:
    """Landing page with links to browse results and API docs."""
    return build_landing_html()


@app.get("/results", response_class=HTMLResponse)
def browse_results_index() -> HTMLResponse:
    """List API jobs and local CLI runs under runtime/."""
    return HTMLResponse(content=build_combined_index_html(BASE_DIR, JOB_DIR))


@app.get("/results/local/{run_id}", response_class=HTMLResponse)
def browse_local_report(run_id: str) -> HTMLResponse:
    """HTML report for outputs under runtime/<run_id>/ (e.g. app.run_local smoke tests)."""
    job_dir = resolve_local_run_dir(BASE_DIR, run_id)
    return build_job_report_html(
        run_id,
        job_dir,
        media_prefix=f"/results/local/{run_id}",
        report_kind="local",
    )


@app.get("/results/local/{run_id}/file/{filename}")
def browse_local_file(run_id: str, filename: str) -> FileResponse:
    job_dir = resolve_local_run_dir(BASE_DIR, run_id)
    return job_file_response(job_dir, filename)


@app.get("/results/{job_id}", response_class=HTMLResponse)
def browse_job_report(job_id: str) -> HTMLResponse:
    """HTML report: annotated video, stats preview, collapsible JSON."""
    job_dir = resolve_job_dir(JOB_DIR, job_id)
    return build_job_report_html(job_id, job_dir, media_prefix=f"/results/{job_id}")


@app.get("/results/{job_id}/file/{filename}")
def browse_job_file(job_id: str, filename: str) -> FileResponse:
    """Serve allowed artifacts (video, JSON, CSV) for embedding and download."""
    job_dir = resolve_job_dir(JOB_DIR, job_id)
    return job_file_response(job_dir, filename)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/models/recommendations")
def model_recommendations() -> list[dict]:
    return [m.__dict__ for m in get_model_shortlist()]


@app.get("/health/summary")
def health_summary() -> dict:
    job_dirs = [p for p in JOB_DIR.iterdir() if p.is_dir()] if JOB_DIR.exists() else []
    try:
        status_counts = get_job_status_counts()
    except Exception:
        status_counts = {}
    return {
        "ok": True,
        "jobs_on_disk": len(job_dirs),
        "job_status_counts": status_counts,
    }


@app.post("/videos")
async def upload_video(file: UploadFile = File(...)) -> dict:
    video_id = str(uuid.uuid4())
    out_path = UPLOAD_DIR / f"{video_id}_{file.filename}"
    out_path.write_bytes(await file.read())
    return {"video_id": video_id, "path": str(out_path)}


@app.post("/jobs")
def create_job(
    video_path: str = Form(...),
    mode: str = Form(default="local"),
    weights: str = Form(default="auto"),
    use_pretrained_stack: str = Form(default="true"),
    use_videomae: str = Form(default="false"),
) -> dict:
    job_id = str(uuid.uuid4())
    out_dir = JOB_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline_cfg = pipeline_config_from_flags(use_pretrained_stack, use_videomae)
    if mode == "cloud":
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        queue = Queue("basketball-jobs", connection=Redis.from_url(redis_url))
        queue.enqueue(
            process_job,
            job_id,
            video_path,
            str(JOB_DIR),
            mode,
            weights,
            use_pretrained_stack,
            use_videomae,
        )
        payload = {
            "job_id": job_id,
            "mode": mode,
            "weights": weights,
            "use_pretrained_stack": use_pretrained_stack,
            "use_videomae": use_videomae,
            "status": "queued",
        }
        try_upsert_job(job_id, mode, "queued", video_path, str(out_dir / "job.json"))
    else:
        try:
            try_upsert_job(job_id, mode, "processing", video_path, str(out_dir / "job.json"))
            result = run_video_analysis(
                Path(video_path),
                out_dir,
                weights=weights,
                pipeline_config=pipeline_cfg,
            )
            payload = {
                "job_id": job_id,
                "mode": mode,
                "weights": weights,
                "use_pretrained_stack": use_pretrained_stack,
                "use_videomae": use_videomae,
                "status": "done",
                "result": result,
            }
            try_upsert_job(job_id, mode, "done", video_path, str(out_dir / "job.json"))
        except Exception:
            payload = {
                "job_id": job_id,
                "mode": mode,
                "weights": weights,
                "use_pretrained_stack": use_pretrained_stack,
                "use_videomae": use_videomae,
                "status": "failed",
            }
            try_upsert_job(job_id, mode, "failed", video_path, str(out_dir / "job.json"))
    (out_dir / "job.json").write_text(json.dumps(payload, indent=2))
    return payload


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    job_path = JOB_DIR / job_id / "job.json"
    db_row = get_job_row(job_id)
    if db_row is None and not job_path.exists():
        return JSONResponse({"status": "not_found"}, status_code=404)
    payload = json.loads(job_path.read_text()) if job_path.exists() else {}
    if db_row is not None:
        payload["db_status"] = db_row.status
        payload["db_mode"] = db_row.mode
        payload["updated_at"] = db_row.updated_at.isoformat()
    return JSONResponse(payload)


@app.get("/jobs/{job_id}/stats")
def get_stats(job_id: str) -> JSONResponse:
    stats_path = JOB_DIR / job_id / "stats.json"
    if not stats_path.exists():
        return JSONResponse({"status": "not_found"}, status_code=404)
    return JSONResponse(json.loads(stats_path.read_text()))


@app.get("/jobs/{job_id}/team-stats")
def get_team_stats(job_id: str) -> JSONResponse:
    team_stats_path = JOB_DIR / job_id / "team_box_score.json"
    if not team_stats_path.exists():
        return JSONResponse({"status": "not_found"}, status_code=404)
    return JSONResponse(json.loads(team_stats_path.read_text()))


@app.get("/jobs/{job_id}/artifacts")
def get_artifacts(job_id: str) -> dict:
    base = JOB_DIR / job_id
    return {
        "annotated_video": str(base / "annotated.mp4"),
        "events": str(base / "events.json"),
        "tracks": str(base / "tracks.json"),
        "player_identity_map": str(base / "player_identity_map.json"),
        "team_box_score": str(base / "team_box_score.json"),
        "stats": str(base / "stats.json"),
        "stats_csv": str(base / "stats.csv"),
        "pipeline": str(base / "pipeline.json"),
        "action_hints": str(base / "action_hints.json"),
        "videomae_aux": str(base / "videomae_aux.json"),
        "job": str(base / "job.json"),
    }
