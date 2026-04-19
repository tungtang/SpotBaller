from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from redis import Redis
from rq import Queue

from app.api.db import (
    JobRecord,
    delete_job_row,
    get_job_row,
    get_job_status_counts,
    init_db,
    upsert_job,
)
from app.ml.model_shortlist import get_model_shortlist
from app.api.worker import process_job
from app.pipeline.pretrained_stack import pipeline_config_from_flags
from app.pipeline.video_analyzer import AnalysisStoppedError, run_video_analysis
from app.api.web_report import (
    build_combined_index_html,
    build_job_report_html,
    build_landing_html,
    job_file_response,
    resolve_job_dir,
    resolve_local_run_dir,
)

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


def try_get_job_row(job_id: str) -> JobRecord | None:
    """Best-effort DB read; return None if Postgres is down or row missing."""
    try:
        return get_job_row(job_id)
    except Exception:
        return None


def _lifecycle_from_status(status: str | None) -> str:
    """Stable label for UIs: complete vs stopped vs failed vs running."""
    s = (status or "").lower()
    if s == "done":
        return "complete"
    if s == "stopped":
        return "stopped"
    if s == "failed":
        return "failed"
    if s in {"queued", "processing", "retrying"}:
        return "running"
    return "unknown"


def _write_job_json(out_dir: Path, payload: dict) -> None:
    (out_dir / "job.json").write_text(json.dumps(payload, indent=2))


def reconcile_stale_local_jobs(job_root: Path) -> int:
    """Mark local jobs stuck in ``processing`` as ``stopped`` (in-process workers do not survive API restart)."""
    updated = 0
    if not job_root.is_dir():
        return 0
    for child in job_root.iterdir():
        if not child.is_dir():
            continue
        job_path = child / "job.json"
        if not job_path.is_file():
            continue
        try:
            payload = json.loads(job_path.read_text())
        except Exception:
            continue
        if payload.get("status") != "processing":
            continue
        if payload.get("mode", "local") not in ("local", "vm"):
            continue

        job_id = str(payload.get("job_id") or child.name)
        video_path = str(payload.get("video_path", ""))
        payload["status"] = "stopped"
        payload["stop_reason"] = "interrupted"
        payload["stop_detail"] = (
            "Local worker was not running when the API started (e.g. server restarted). "
            "Re-upload and analyze again to finish."
        )
        payload["stopped_at"] = datetime.now(timezone.utc).isoformat()
        _write_job_json(child, payload)
        try_upsert_job(job_id, str(payload.get("mode", "local")), "stopped", video_path, str(job_path))

        prog_path = child / "progress.json"
        if prog_path.is_file():
            try:
                prog = json.loads(prog_path.read_text())
            except Exception:
                prog = {}
            if prog.get("status") == "processing":
                prog["status"] = "stopped"
                prog["stop_reason"] = "interrupted"
                prog_path.write_text(json.dumps(prog, indent=2))
        updated += 1
    return updated


@asynccontextmanager
async def _app_lifespan(app: FastAPI):
    reconcile_stale_local_jobs(JOB_DIR)
    yield


# Local in-process workers: cooperative stop via threading.Event
_local_job_workers: dict[str, dict[str, object]] = {}


def _unregister_local_worker(job_id: str) -> None:
    _local_job_workers.pop(job_id, None)


def _run_local_job_async(
    job_id: str,
    video_path: str,
    out_dir: Path,
    weights: str,
    pipeline_cfg,
    use_pretrained_stack: str,
    use_videomae: str,
    stop_event: threading.Event,
    max_frames: int | None = None,
) -> None:
    """Background worker: run analysis and atomically replace job.json when finished."""
    mode = "local"
    payload: dict = {}
    try:
        try_upsert_job(job_id, mode, "processing", video_path, str(out_dir / "job.json"))
        result = run_video_analysis(
            Path(video_path),
            out_dir,
            weights=weights,
            pipeline_config=pipeline_cfg,
            stop_event=stop_event,
            max_frames=max_frames,
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
    except AnalysisStoppedError as exc:
        payload = {
            "job_id": job_id,
            "mode": mode,
            "weights": weights,
            "use_pretrained_stack": use_pretrained_stack,
            "use_videomae": use_videomae,
            "status": "stopped",
            "stop_reason": "user_requested",
            "stop_detail": "Analysis was stopped before completion.",
            "frames_at_stop": exc.frames_processed,
            "stopped_at": datetime.now(timezone.utc).isoformat(),
        }
        try_upsert_job(job_id, mode, "stopped", video_path, str(out_dir / "job.json"))
    except Exception as exc:
        payload = {
            "job_id": job_id,
            "mode": mode,
            "weights": weights,
            "use_pretrained_stack": use_pretrained_stack,
            "use_videomae": use_videomae,
            "status": "failed",
            "error": str(exc),
        }
        try_upsert_job(job_id, mode, "failed", video_path, str(out_dir / "job.json"))
    finally:
        _unregister_local_worker(job_id)
    if payload:
        _write_job_json(out_dir, payload)


def _spawn_local_worker(
    job_id: str,
    out_dir: Path,
    video_path: str,
    weights: str,
    pipeline_cfg,
    use_pretrained_stack: str,
    use_videomae: str,
    max_frames: int | None = None,
) -> threading.Thread:
    if job_id in _local_job_workers:
        raise HTTPException(status_code=409, detail="Job already has an active local worker.")
    stop_ev = threading.Event()
    t = threading.Thread(
        target=_run_local_job_async,
        kwargs={
            "job_id": job_id,
            "video_path": video_path,
            "out_dir": out_dir,
            "weights": weights,
            "pipeline_cfg": pipeline_cfg,
            "use_pretrained_stack": use_pretrained_stack,
            "use_videomae": use_videomae,
            "stop_event": stop_ev,
            "max_frames": max_frames,
        },
        daemon=True,
    )
    _local_job_workers[job_id] = {"stop_event": stop_ev, "thread": t, "kind": "local"}
    t.start()
    return t


def _run_vm_job_async(
    job_id: str,
    video_path: str,
    out_dir: Path,
    weights: str,
    use_pretrained_stack: str,
    use_videomae: str,
    max_frames: int | None,
) -> None:
    """Background worker: gcloud scp + remote run_local + scp performance back."""
    mode = "vm"
    payload: dict = {}
    try:
        from app.gcp.vm_runner import run_pipeline_on_vm, vm_config_from_env

        cfg = vm_config_from_env()
        if cfg is None:
            raise RuntimeError(
                "VM mode requires SPOTBALLER_GCLOUD_VM, SPOTBALLER_GCLOUD_ZONE, SPOTBALLER_GCLOUD_PROJECT"
            )
        try_upsert_job(job_id, mode, "processing", video_path, str(out_dir / "job.json"))
        use_pt = str(use_pretrained_stack).lower() in ("1", "true", "yes", "on")
        use_vm = str(use_videomae).lower() in ("1", "true", "yes", "on")
        result = run_pipeline_on_vm(
            cfg,
            job_id,
            Path(video_path),
            out_dir,
            weights=weights,
            use_pretrained_stack=use_pt,
            use_videomae=use_vm,
            max_frames=max_frames,
        )
        enriched: dict = {"vm_remote": True, **result}
        prog_path = out_dir / "progress.json"
        if prog_path.is_file():
            try:
                prog = json.loads(prog_path.read_text())
                enriched["frames_processed"] = prog.get("frames_processed")
            except Exception:
                pass
        for fn, key in (("stats.json", "stats"), ("pipeline.json", "pipeline")):
            p = out_dir / fn
            if p.is_file():
                try:
                    enriched[key] = json.loads(p.read_text())
                except Exception:
                    pass
        payload = {
            "job_id": job_id,
            "mode": mode,
            "weights": weights,
            "use_pretrained_stack": use_pretrained_stack,
            "use_videomae": use_videomae,
            "status": "done",
            "result": enriched,
        }
        try_upsert_job(job_id, mode, "done", video_path, str(out_dir / "job.json"))
    except Exception as exc:
        payload = {
            "job_id": job_id,
            "mode": mode,
            "weights": weights,
            "use_pretrained_stack": use_pretrained_stack,
            "use_videomae": use_videomae,
            "status": "failed",
            "error": str(exc),
        }
        try_upsert_job(job_id, mode, "failed", video_path, str(out_dir / "job.json"))
    finally:
        _unregister_local_worker(job_id)
    if payload:
        _write_job_json(out_dir, payload)


def _spawn_vm_worker(
    job_id: str,
    out_dir: Path,
    video_path: str,
    weights: str,
    use_pretrained_stack: str,
    use_videomae: str,
    max_frames: int | None,
) -> threading.Thread:
    if job_id in _local_job_workers:
        raise HTTPException(status_code=409, detail="Job already has an active worker.")
    t = threading.Thread(
        target=_run_vm_job_async,
        kwargs={
            "job_id": job_id,
            "video_path": video_path,
            "out_dir": out_dir,
            "weights": weights,
            "use_pretrained_stack": use_pretrained_stack,
            "use_videomae": use_videomae,
            "max_frames": max_frames,
        },
        daemon=True,
    )
    _local_job_workers[job_id] = {"stop_event": None, "thread": t, "kind": "vm"}
    t.start()
    return t


def _start_or_resume_local_job(job_id: str) -> dict:
    """Start or restart local analysis for an existing job folder (same video path, re-run from scratch)."""
    out_dir = JOB_DIR / job_id
    job_path = out_dir / "job.json"
    if not job_path.is_file():
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        payload = json.loads(job_path.read_text())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Invalid job.json: {exc}") from exc

    if payload.get("mode", "local") != "local":
        raise HTTPException(status_code=400, detail="Start/resume only applies to local jobs.")

    st = payload.get("status")
    if st == "done":
        raise HTTPException(status_code=409, detail="Job already complete.")
    if st == "queued":
        raise HTTPException(status_code=400, detail="Cloud jobs are started by the worker queue.")

    if st == "processing" and job_id not in _local_job_workers:
        payload["status"] = "stopped"
        payload["stop_reason"] = "stale"
        payload["stop_detail"] = "Stale processing state cleared before restart."
        payload["stopped_at"] = datetime.now(timezone.utc).isoformat()
        _write_job_json(out_dir, payload)
        try_upsert_job(
            job_id,
            "local",
            "stopped",
            str(payload.get("video_path", "")),
            str(job_path),
        )
        st = "stopped"

    if st not in ("stopped", "failed"):
        raise HTTPException(status_code=409, detail=f"Cannot start job from status: {st}")

    video_path = str(payload.get("video_path") or "")
    if not video_path or not Path(video_path).is_file():
        raise HTTPException(status_code=400, detail="Original video file is missing; upload again and create a new job.")

    weights = str(payload.get("weights", "auto"))
    use_pretrained_stack = str(payload.get("use_pretrained_stack", "true"))
    use_videomae = str(payload.get("use_videomae", "false"))
    mf_resume = payload.get("max_frames")
    mf_parsed = _parse_optional_max_frames(str(mf_resume)) if mf_resume is not None else None
    pipeline_cfg = pipeline_config_from_flags(use_pretrained_stack, use_videomae)

    out_payload = {
        "job_id": job_id,
        "mode": "local",
        "weights": weights,
        "use_pretrained_stack": use_pretrained_stack,
        "use_videomae": use_videomae,
        "status": "processing",
    }
    if mf_parsed is not None:
        out_payload["max_frames"] = mf_parsed
    _write_job_json(out_dir, out_payload)
    try_upsert_job(job_id, "local", "processing", video_path, str(job_path))

    _spawn_local_worker(
        job_id,
        out_dir,
        video_path,
        weights,
        pipeline_cfg,
        use_pretrained_stack,
        use_videomae,
        mf_parsed,
    )
    return out_payload


app = FastAPI(title="Basketball Video Analytics API", lifespan=_app_lifespan)


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


def _parse_optional_max_frames(raw: str) -> int | None:
    s = (raw or "").strip()
    if not s:
        return None
    try:
        return max(1, int(s))
    except ValueError:
        return None


@app.post("/jobs")
def create_job(
    video_path: str = Form(...),
    mode: str = Form(default="local"),
    weights: str = Form(default="auto"),
    use_pretrained_stack: str = Form(default="true"),
    use_videomae: str = Form(default="false"),
    max_frames: str = Form(default=""),
) -> dict:
    job_id = str(uuid.uuid4())
    out_dir = JOB_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline_cfg = pipeline_config_from_flags(use_pretrained_stack, use_videomae)
    mf = _parse_optional_max_frames(max_frames)
    if mode == "vm":
        from app.gcp.vm_runner import vm_config_from_env

        if vm_config_from_env() is None:
            raise HTTPException(
                status_code=501,
                detail="VM mode requires SPOTBALLER_GCLOUD_VM, SPOTBALLER_GCLOUD_ZONE, SPOTBALLER_GCLOUD_PROJECT "
                "and gcloud on PATH (see infra/gcp/README.md).",
            )
        try_upsert_job(job_id, mode, "processing", video_path, str(out_dir / "job.json"))
        payload = {
            "job_id": job_id,
            "mode": mode,
            "weights": weights,
            "use_pretrained_stack": use_pretrained_stack,
            "use_videomae": use_videomae,
            "status": "processing",
            "max_frames": mf,
        }
        _write_job_json(out_dir, payload)
        _spawn_vm_worker(
            job_id,
            out_dir,
            video_path,
            weights,
            use_pretrained_stack,
            use_videomae,
            mf,
        )
    elif mode == "cloud":
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
        (out_dir / "job.json").write_text(json.dumps(payload, indent=2))
    elif mode == "gcp":
        try:
            from app.gcp.enqueue import ensure_video_gcs_uri
            from app.gcp.pubsub_util import publish_video_job
            from app.gcp.sync import gcs_job_prefix
        except ImportError as exc:
            raise HTTPException(
                status_code=501,
                detail=f"GCP dependencies missing. pip install -r requirements-gcp.txt ({exc})",
            ) from exc
        try:
            video_gcs_uri = ensure_video_gcs_uri(job_id, video_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        output_prefix = gcs_job_prefix(job_id)
        msg = {
            "job_id": job_id,
            "video_gcs_uri": video_gcs_uri,
            "output_gcs_prefix": output_prefix,
            "weights": weights,
            "use_pretrained_stack": use_pretrained_stack,
            "use_videomae": use_videomae,
            "rim_fallback": True,
        }
        try:
            publish_video_job(msg)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Pub/Sub publish failed: {exc}") from exc
        payload = {
            "job_id": job_id,
            "mode": mode,
            "weights": weights,
            "use_pretrained_stack": use_pretrained_stack,
            "use_videomae": use_videomae,
            "status": "queued",
            "video_gcs_uri": video_gcs_uri,
            "output_gcs_prefix": output_prefix,
        }
        try_upsert_job(job_id, mode, "queued", video_gcs_uri, str(out_dir / "job.json"))
        _write_job_json(out_dir, payload)
    else:
        try_upsert_job(job_id, mode, "processing", video_path, str(out_dir / "job.json"))
        payload = {
            "job_id": job_id,
            "mode": mode,
            "weights": weights,
            "use_pretrained_stack": use_pretrained_stack,
            "use_videomae": use_videomae,
            "status": "processing",
        }
        if mf is not None:
            payload["max_frames"] = mf
        _write_job_json(out_dir, payload)
        _spawn_local_worker(
            job_id,
            out_dir,
            video_path,
            weights,
            pipeline_cfg,
            use_pretrained_stack,
            use_videomae,
            mf,
        )
    return payload


@app.get("/jobs/compare-performance")
def compare_job_performance(job_a: str, job_b: str) -> JSONResponse:
    """Side-by-side ``performance.json`` for two job ids (must be registered before ``/jobs/{{job_id}}``)."""
    out: dict[str, object] = {"job_a": job_a, "job_b": job_b, "rows": []}
    for label, jid in (("a", job_a), ("b", job_b)):
        p = JOB_DIR / jid / "performance.json"
        if not p.is_file():
            out[f"performance_{label}"] = None
            continue
        try:
            out[f"performance_{label}"] = json.loads(p.read_text())
        except Exception as exc:
            out[f"performance_{label}"] = {"error": str(exc)}
    pa = out.get("performance_a")
    pb = out.get("performance_b")
    if isinstance(pa, dict) and isinstance(pb, dict):
        keys = sorted(set(pa.keys()) | set(pb.keys()))
        rows = []
        for k in keys:
            rows.append({"metric": k, "job_a": pa.get(k), "job_b": pb.get(k)})
        out["rows"] = rows
    return JSONResponse(out)


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    job_path = JOB_DIR / job_id / "job.json"
    if job_path.is_file():
        try:
            from app.gcp.sync import maybe_refresh_job_from_gcs

            maybe_refresh_job_from_gcs(job_id, job_path)
        except Exception:
            pass
    db_row = try_get_job_row(job_id)
    if db_row is None and not job_path.exists():
        return JSONResponse({"status": "not_found"}, status_code=404)
    payload = json.loads(job_path.read_text()) if job_path.exists() else {}
    if db_row is not None:
        payload["db_status"] = db_row.status
        payload["db_mode"] = db_row.mode
        payload["updated_at"] = db_row.updated_at.isoformat()
    payload["lifecycle"] = _lifecycle_from_status(payload.get("status"))
    payload["worker_active"] = job_id in _local_job_workers
    return JSONResponse(payload)


@app.post("/jobs/{job_id}/start")
def start_existing_job(job_id: str) -> dict:
    """Start or restart local analysis (full pass) for a stopped/failed job using the saved video path."""
    return _start_or_resume_local_job(job_id)


@app.post("/jobs/{job_id}/resume")
def resume_job(job_id: str) -> dict:
    """Same as ``/start`` — re-run the pipeline from the beginning of the video (reuses job id and uploads)."""
    return _start_or_resume_local_job(job_id)


@app.post("/jobs/{job_id}/stop")
def stop_running_job(job_id: str) -> dict:
    """Request cooperative stop for a running local job."""
    w = _local_job_workers.get(job_id)
    if not w:
        raise HTTPException(
            status_code=404,
            detail="No active local worker for this job (not running or already finished).",
        )
    if w.get("kind") == "vm":
        raise HTTPException(
            status_code=400,
            detail="VM jobs run remotely via gcloud; stop is not supported from the API.",
        )
    stop_ev = w.get("stop_event")
    if stop_ev is not None and hasattr(stop_ev, "set"):
        stop_ev.set()  # type: ignore[union-attr]
    return {"ok": True, "job_id": job_id, "status": "stopping"}


@app.delete("/jobs/{job_id}")
def delete_job_endpoint(job_id: str) -> JSONResponse:
    """Remove job directory and DB row. Stops a running local worker first."""
    out_dir = JOB_DIR / job_id
    if not out_dir.is_dir():
        return JSONResponse({"status": "not_found"}, status_code=404)

    w = _local_job_workers.get(job_id)
    if w:
        stop_ev = w.get("stop_event")
        if stop_ev is not None and hasattr(stop_ev, "set"):
            stop_ev.set()  # type: ignore[union-attr]
        th = w.get("thread")
        if th is not None and hasattr(th, "join"):
            th.join(timeout=120.0)  # type: ignore[union-attr]

    try:
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        delete_job_row(job_id)
    except Exception:
        pass
    return JSONResponse({"ok": True, "deleted": job_id})


@app.get("/jobs/{job_id}/progress")
def get_job_progress(job_id: str) -> JSONResponse:
    """Frames processed / total_frames and percent_complete from the analyzer (poll while status is processing)."""
    progress_path = JOB_DIR / job_id / "progress.json"
    if progress_path.is_file():
        return JSONResponse(json.loads(progress_path.read_text()))
    try:
        from app.gcp.storage import default_bucket, download_blob_to_file

        bucket = default_bucket()
        uri = f"gs://{bucket}/jobs/{job_id}/progress.json"
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tf:
            tmp = Path(tf.name)
        try:
            download_blob_to_file(uri, tmp)
            return JSONResponse(json.loads(tmp.read_text()))
        finally:
            tmp.unlink(missing_ok=True)
    except Exception:
        pass
    return JSONResponse({"status": "not_found"}, status_code=404)


@app.get("/jobs/{job_id}/performance")
def get_job_performance(job_id: str) -> JSONResponse:
    """``performance.json`` timing breakdown (local or copied from VM)."""
    perf_path = JOB_DIR / job_id / "performance.json"
    if perf_path.is_file():
        return JSONResponse(json.loads(perf_path.read_text()))
    return JSONResponse({"status": "not_found"}, status_code=404)


@app.get("/jobs/{job_id}/stats")
def get_stats(job_id: str) -> JSONResponse:
    """Per-jersey aggregated stats (insight rows). Same as ``stats.json`` on disk."""
    stats_path = JOB_DIR / job_id / "stats.json"
    if not stats_path.exists():
        return JSONResponse({"status": "not_found"}, status_code=404)
    return JSONResponse(json.loads(stats_path.read_text()))


@app.get("/jobs/{job_id}/stats/by-track")
def get_stats_by_track(job_id: str) -> JSONResponse:
    """Raw per–track-id stats (debug only; not merged by jersey)."""
    path = JOB_DIR / job_id / "stats_by_track.json"
    if not path.exists():
        return JSONResponse({"status": "not_found"}, status_code=404)
    return JSONResponse(json.loads(path.read_text()))


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
        "stats_by_track": str(base / "stats_by_track.json"),
        "stats_csv": str(base / "stats.csv"),
        "team_box_score_players_csv": str(base / "team_box_score_players.csv"),
        "action_hints_long_csv": str(base / "action_hints_long.csv"),
        "pipeline_performance_csv": str(base / "pipeline_performance.csv"),
        "pipeline": str(base / "pipeline.json"),
        "action_hints": str(base / "action_hints.json"),
        "videomae_aux": str(base / "videomae_aux.json"),
        "job": str(base / "job.json"),
        "progress": str(base / "progress.json"),
    }
