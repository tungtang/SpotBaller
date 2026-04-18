from __future__ import annotations

import json
import time
from pathlib import Path

from app.api.db import upsert_job
from app.pipeline.pretrained_stack import pipeline_config_from_flags
from app.pipeline.video_analyzer import run_video_analysis


def process_job(
    job_id: str,
    video_path: str,
    output_root: str = "runtime/jobs",
    mode: str = "cloud",
    weights: str = "yolov8n.pt",
    use_pretrained_stack: str = "true",
    use_videomae: str = "false",
    max_retries: int = 2,
) -> dict:
    out_dir = Path(output_root) / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    upsert_job(job_id, mode, "processing", video_path, str(out_dir / "job.json"))
    pipeline_cfg = pipeline_config_from_flags(use_pretrained_stack, use_videomae)
    payload: dict = {"job_id": job_id, "mode": mode, "status": "failed", "error": "unknown"}
    for attempt in range(max_retries + 1):
        try:
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
                "status": "done",
                "result": result,
                "attempts": attempt + 1,
            }
            upsert_job(job_id, mode, "done", video_path, str(out_dir / "job.json"))
            break
        except FileNotFoundError as exc:
            payload = {
                "job_id": job_id,
                "mode": mode,
                "weights": weights,
                "status": "failed",
                "error_type": "file_not_found",
                "error": str(exc),
                "attempts": attempt + 1,
            }
            upsert_job(job_id, mode, "failed", video_path, str(out_dir / "job.json"))
            break
        except Exception as exc:
            if attempt < max_retries:
                upsert_job(job_id, mode, "retrying", video_path, str(out_dir / "job.json"))
                time.sleep(1.0 * (attempt + 1))
                continue
            payload = {
                "job_id": job_id,
                "mode": mode,
                "weights": weights,
                "status": "failed",
                "error_type": "processing_error",
                "error": str(exc),
                "attempts": attempt + 1,
            }
            upsert_job(job_id, mode, "failed", video_path, str(out_dir / "job.json"))
    (out_dir / "job.json").write_text(json.dumps(payload, indent=2))
    return payload
