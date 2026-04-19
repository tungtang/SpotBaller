from __future__ import annotations

import json
from pathlib import Path

from app.gcp.storage import default_bucket, download_blob_to_file


def maybe_refresh_job_from_gcs(job_id: str, job_path: Path) -> None:
    """
    If ``SPOTBALLER_GCS_BUCKET`` is set and the job is ``mode=gcp``, try to pull
    ``gs://<bucket>/jobs/<job_id>/job.json`` and replace the local file when the
    remote status is terminal (done / failed / stopped).
    """
    try:
        bucket = default_bucket()
    except RuntimeError:
        return
    if not job_path.is_file():
        return
    try:
        payload = json.loads(job_path.read_text())
    except Exception:
        return
    if payload.get("mode") != "gcp":
        return
    uri = f"gs://{bucket}/jobs/{job_id}/job.json"
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tf:
        tmp = Path(tf.name)
    try:
        download_blob_to_file(uri, tmp)
        remote = json.loads(tmp.read_text())
    except Exception:
        return
    finally:
        tmp.unlink(missing_ok=True)

    rstatus = str(remote.get("status", "")).lower()
    if rstatus in ("done", "failed", "stopped"):
        job_path.write_text(json.dumps(remote, indent=2))


def gcs_job_prefix(job_id: str) -> str:
    bucket = default_bucket()
    return f"gs://{bucket}/jobs/{job_id}/"
