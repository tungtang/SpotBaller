from __future__ import annotations

from pathlib import Path

from app.gcp.storage import default_bucket, upload_file_to_gs


def ensure_video_gcs_uri(job_id: str, video_path: str) -> str:
    """Return ``gs://`` URI, uploading local files to ``uploads/`` when needed."""
    if video_path.startswith("gs://"):
        return video_path
    p = Path(video_path)
    if not p.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    bucket = default_bucket()
    dest = f"gs://{bucket}/uploads/{job_id}_{p.name}"
    upload_file_to_gs(p, dest)
    return dest
