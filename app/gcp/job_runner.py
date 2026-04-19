from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from app.gcp.storage import download_blob_to_file, upload_directory_to_prefix, upload_file_to_gs
from app.pipeline.pretrained_stack import pipeline_config_from_flags
from app.pipeline.video_analyzer import run_video_analysis


def run_remote_job(
    job_id: str,
    video_gcs_uri: str,
    output_gs_prefix: str,
    *,
    weights: str = "auto",
    use_pretrained_stack: str = "true",
    use_videomae: str = "false",
    rim_fallback: bool = True,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """
    Download ``video_gcs_uri``, run ``run_video_analysis`` in a temp dir, upload all artifacts
    to ``output_gs_prefix`` (e.g. ``gs://bucket/jobs/<job_id>/``).
    """
    pipeline_cfg = pipeline_config_from_flags(use_pretrained_stack, use_videomae)
    with tempfile.TemporaryDirectory(prefix=f"spotballer_{job_id}_") as tmp:
        work = Path(tmp)
        local_video = work / "input_video.mp4"
        out_dir = work / "out"
        download_blob_to_file(video_gcs_uri, local_video)
        result = run_video_analysis(
            local_video,
            out_dir,
            weights=weights,
            pipeline_config=pipeline_cfg,
            max_frames=max_frames,
            rim_fallback=rim_fallback,
        )
        upload_directory_to_prefix(out_dir, output_gs_prefix)
        job_payload = {
            "job_id": job_id,
            "mode": "gcp",
            "weights": weights,
            "use_pretrained_stack": use_pretrained_stack,
            "use_videomae": use_videomae,
            "status": "done",
            "result": result,
            "video_gcs_uri": video_gcs_uri,
            "output_gcs_prefix": output_gs_prefix,
        }
        job_path = work / "job.json"
        job_path.write_text(json.dumps(job_payload, indent=2))
        upload_file_to_gs(job_path, _join_gs(output_gs_prefix, "job.json"), content_type="application/json")
        return job_payload


def _join_gs(prefix: str, name: str) -> str:
    p = prefix.rstrip("/")
    return f"{p}/{name}"


def run_remote_job_from_message(data: dict[str, Any]) -> dict[str, Any]:
    """Decode Pub/Sub JSON body and run :func:`run_remote_job`."""
    job_id = str(data["job_id"])
    video_gcs_uri = str(data["video_gcs_uri"])
    output_gs_prefix = str(data["output_gcs_prefix"])
    weights = str(data.get("weights", "auto"))
    use_pretrained_stack = str(data.get("use_pretrained_stack", "true"))
    use_videomae = str(data.get("use_videomae", "false"))
    rim_fallback = bool(data.get("rim_fallback", True))
    max_frames = data.get("max_frames")
    mf = int(max_frames) if max_frames is not None else None
    return run_remote_job(
        job_id,
        video_gcs_uri,
        output_gs_prefix,
        weights=weights,
        use_pretrained_stack=use_pretrained_stack,
        use_videomae=use_videomae,
        rim_fallback=rim_fallback,
        max_frames=mf,
    )


def write_failed_job_json(
    job_id: str,
    output_gs_prefix: str,
    *,
    error: str,
    video_gcs_uri: str | None = None,
) -> None:
    if not output_gs_prefix.startswith("gs://"):
        return
    payload = {
        "job_id": job_id,
        "mode": "gcp",
        "status": "failed",
        "error": error,
        "video_gcs_uri": video_gcs_uri,
        "output_gcs_prefix": output_gs_prefix,
    }
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "job.json"
        p.write_text(json.dumps(payload, indent=2))
        upload_file_to_gs(p, _join_gs(output_gs_prefix, "job.json"), content_type="application/json")
