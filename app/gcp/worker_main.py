"""
GPU / CPU worker entrypoint: subscribe to Pub/Sub, run :func:`run_remote_job_from_message`.

Environment:
  SPOTBALLER_GCP_PROJECT or GOOGLE_CLOUD_PROJECT
  SPOTBALLER_PUBSUB_SUBSCRIPTION  (subscription id, e.g. video-workers)

Optional:
  SPOTBALLER_PUBSUB_MAX_MESSAGES (default 1) — pull batch size for streaming pull

Ack deadline: configure the subscription in GCP (up to 600s). Jobs longer than
that need heartbeat/lease design or Cloud Tasks.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from concurrent.futures import TimeoutError as FuturesTimeoutError

from google.api_core.exceptions import GoogleAPICallError

LOG = logging.getLogger("spotballer.gcp.worker")


def _project_id() -> str:
    pid = os.environ.get("SPOTBALLER_GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not pid:
        raise RuntimeError("Set SPOTBALLER_GCP_PROJECT or GOOGLE_CLOUD_PROJECT")
    return pid


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from google.cloud import pubsub_v1

    sub_id = os.environ.get("SPOTBALLER_PUBSUB_SUBSCRIPTION", "").strip()
    if not sub_id:
        LOG.error("Set SPOTBALLER_PUBSUB_SUBSCRIPTION")
        sys.exit(1)

    subscriber = pubsub_v1.SubscriberClient()
    sub_path = subscriber.subscription_path(_project_id(), sub_id)
    max_in_flight = max(1, int(os.environ.get("SPOTBALLER_PUBSUB_MAX_MESSAGES", "1")))

    from app.gcp.job_runner import run_remote_job_from_message, write_failed_job_json

    def _handle_message(message) -> None:
        data: dict
        try:
            data = json.loads(message.data.decode("utf-8"))
        except Exception as exc:
            LOG.warning("invalid pub/sub payload: %s", exc)
            message.nack()
            return
        job_id = str(data.get("job_id", ""))
        try:
            LOG.info("job start %s", job_id)
            run_remote_job_from_message(data)
            LOG.info("job done %s", job_id)
            message.ack()
        except Exception as exc:
            LOG.exception("job failed: %s", exc)
            try:
                write_failed_job_json(
                    str(data.get("job_id", "unknown")),
                    str(data.get("output_gcs_prefix", "")),
                    error=str(exc),
                    video_gcs_uri=str(data.get("video_gcs_uri")) if data.get("video_gcs_uri") else None,
                )
            except Exception:
                pass
            message.nack()

    flow = pubsub_v1.types.FlowControl(max_messages=max_in_flight)
    future = subscriber.subscribe(sub_path, callback=_handle_message, flow_control=flow)
    LOG.info("listening on %s", sub_path)
    try:
        future.result()
    except FuturesTimeoutError:
        future.cancel()
        future.result()
    except GoogleAPICallError as exc:
        LOG.error("subscriber error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
