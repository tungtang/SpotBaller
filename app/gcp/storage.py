from __future__ import annotations

import os
import re
from pathlib import Path

_GS_URI = re.compile(r"^gs://([^/]+)/(.+)$")


def parse_gs_uri(uri: str) -> tuple[str, str]:
    m = _GS_URI.match(uri.rstrip("/"))
    if not m:
        raise ValueError(f"Not a gs:// URI: {uri}")
    return m.group(1), m.group(2)


def download_blob_to_file(gs_uri: str, dest: Path) -> None:
    from google.cloud import storage

    bucket_name, blob_path = parse_gs_uri(gs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dest))


def upload_file_to_gs(local_path: Path, gs_uri: str, content_type: str | None = None) -> None:
    from google.cloud import storage

    bucket_name, blob_path = parse_gs_uri(gs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    if content_type:
        blob.upload_from_filename(str(local_path), content_type=content_type)
    else:
        blob.upload_from_filename(str(local_path))


def upload_directory_to_prefix(local_dir: Path, gs_prefix: str) -> None:
    """Upload every file under ``local_dir`` to ``gs://bucket/prefix/`` preserving relative paths."""
    from google.cloud import storage

    if not gs_prefix.startswith("gs://"):
        raise ValueError("gs_prefix must start with gs://")
    bucket_name, prefix = parse_gs_uri(gs_prefix if gs_prefix.endswith("/") else gs_prefix + "/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for path in local_dir.rglob("*"):
        if path.is_file():
            rel = path.relative_to(local_dir).as_posix()
            dest = f"{prefix}{rel}" if prefix.endswith("/") else f"{prefix}/{rel}"
            blob = bucket.blob(dest)
            blob.upload_from_filename(str(path))


def default_bucket() -> str:
    b = os.environ.get("SPOTBALLER_GCS_BUCKET", "").strip()
    if not b:
        raise RuntimeError("Set SPOTBALLER_GCS_BUCKET to your GCS bucket name.")
    return b
