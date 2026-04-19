from __future__ import annotations

import json
import os
from typing import Any


def _project_id() -> str:
    pid = os.environ.get("SPOTBALLER_GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not pid:
        raise RuntimeError("Set SPOTBALLER_GCP_PROJECT or GOOGLE_CLOUD_PROJECT.")
    return pid


def publish_video_job(message: dict[str, Any]) -> str:
    """Publish a JSON job to ``SPOTBALLER_PUBSUB_TOPIC`` (topic id, not full path). Returns message id."""
    from google.cloud import pubsub_v1

    topic_id = os.environ.get("SPOTBALLER_PUBSUB_TOPIC", "").strip()
    if not topic_id:
        raise RuntimeError("Set SPOTBALLER_PUBSUB_TOPIC (e.g. video-jobs).")

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(_project_id(), topic_id)
    data = json.dumps(message, separators=(",", ":")).encode("utf-8")
    future = publisher.publish(topic_path, data=data, job_id=str(message.get("job_id", "")))
    return future.result()
