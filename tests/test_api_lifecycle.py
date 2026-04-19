import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import main as api_main


def test_config_gcp_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPOTBALLER_GCLOUD_VM", "spotballer-vm-2")
    monkeypatch.setenv("SPOTBALLER_GCLOUD_ZONE", "asia-southeast1-a")
    monkeypatch.setenv("SPOTBALLER_GCLOUD_PROJECT", "my-proj")
    client = TestClient(api_main.app)
    res = client.get("/config/gcp")
    assert res.status_code == 200
    data = res.json()
    assert data["vm"] == "spotballer-vm-2"
    assert data["zone"] == "asia-southeast1-a"
    assert data["project"] == "my-proj"
    assert data["vm_configured"] is True
    monkeypatch.delenv("SPOTBALLER_GCLOUD_VM", raising=False)
    res2 = client.get("/config/gcp")
    assert res2.json()["vm_configured"] is False


def test_planned_total_frames_matches_video(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from app.pipeline.video_analyzer import planned_total_frames_for_video

    mp4 = tmp_path / "clip.mp4"
    w, h = 320, 240
    vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    n = 12
    for _ in range(n):
        vw.write(np.zeros((h, w, 3), dtype=np.uint8))
    vw.release()
    cap = cv2.VideoCapture(str(mp4))
    raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    assert planned_total_frames_for_video(mp4, 5) == (5 if raw <= 0 else min(raw, 5))
    full = planned_total_frames_for_video(mp4, None)
    if raw > 0:
        assert full == raw


def test_health_summary_shape() -> None:
    client = TestClient(api_main.app)
    res = client.get("/health/summary")
    assert res.status_code == 200
    payload = res.json()
    assert payload["ok"] is True
    assert "jobs_on_disk" in payload
    assert "job_status_counts" in payload


def test_team_stats_not_found(tmp_path: Path) -> None:
    api_main.JOB_DIR = tmp_path / "jobs"
    api_main.JOB_DIR.mkdir(parents=True, exist_ok=True)
    client = TestClient(api_main.app)
    res = client.get("/jobs/missing-job/team-stats")
    assert res.status_code == 404
    assert res.json()["status"] == "not_found"


def test_progress_not_found(tmp_path: Path) -> None:
    api_main.JOB_DIR = tmp_path / "jobs"
    api_main.JOB_DIR.mkdir(parents=True, exist_ok=True)
    client = TestClient(api_main.app)
    res = client.get("/jobs/missing-job/progress")
    assert res.status_code == 404
    assert res.json()["status"] == "not_found"


def test_progress_reads_json(tmp_path: Path) -> None:
    api_main.JOB_DIR = tmp_path / "jobs"
    jid = "test-job"
    job_dir = api_main.JOB_DIR / jid
    job_dir.mkdir(parents=True)
    (job_dir / "progress.json").write_text(
        '{"status": "processing", "frames_processed": 10, "total_frames": 100, "percent_complete": 10.0}'
    )
    client = TestClient(api_main.app)
    res = client.get(f"/jobs/{jid}/progress")
    assert res.status_code == 200
    data = res.json()
    assert data["frames_processed"] == 10
    assert data["total_frames"] == 100
    assert data["percent_complete"] == 10.0


def test_get_job_includes_lifecycle(tmp_path: Path) -> None:
    api_main.JOB_DIR = tmp_path / "jobs"
    jid = "lifecycle-job"
    job_dir = api_main.JOB_DIR / jid
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps({"job_id": jid, "mode": "local", "status": "done", "result": {}})
    )
    client = TestClient(api_main.app)
    res = client.get(f"/jobs/{jid}")
    assert res.status_code == 200
    assert res.json()["lifecycle"] == "complete"


def test_reconcile_stale_processing_to_stopped(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    jid = "stale-job"
    jdir = jobs / jid
    jdir.mkdir(parents=True)
    (jdir / "job.json").write_text(
        json.dumps(
            {
                "job_id": jid,
                "mode": "local",
                "status": "processing",
                "video_path": "/tmp/x.mp4",
            }
        )
    )
    (jdir / "progress.json").write_text(
        json.dumps({"status": "processing", "frames_processed": 3, "total_frames": 100, "percent_complete": 3.0})
    )
    n = api_main.reconcile_stale_local_jobs(jobs)
    assert n == 1
    payload = json.loads((jdir / "job.json").read_text())
    assert payload["status"] == "stopped"
    assert payload.get("stop_reason") == "interrupted"
    prog = json.loads((jdir / "progress.json").read_text())
    assert prog["status"] == "stopped"


def test_delete_job_missing(tmp_path: Path) -> None:
    api_main.JOB_DIR = tmp_path / "jobs"
    api_main.JOB_DIR.mkdir(parents=True, exist_ok=True)
    client = TestClient(api_main.app)
    res = client.delete("/jobs/does-not-exist")
    assert res.status_code == 404


def test_delete_job_removes_dir(tmp_path: Path) -> None:
    api_main.JOB_DIR = tmp_path / "jobs"
    jid = "to-delete"
    d = api_main.JOB_DIR / jid
    d.mkdir(parents=True)
    (d / "job.json").write_text('{"job_id":"to-delete","mode":"local","status":"stopped"}')
    client = TestClient(api_main.app)
    res = client.delete(f"/jobs/{jid}")
    assert res.status_code == 200
    assert not d.exists()


def test_start_job_requires_existing_video(tmp_path: Path) -> None:
    api_main.JOB_DIR = tmp_path / "jobs"
    jid = "no-video"
    d = api_main.JOB_DIR / jid
    d.mkdir(parents=True)
    (d / "job.json").write_text(
        json.dumps(
            {
                "job_id": jid,
                "mode": "local",
                "status": "stopped",
                "video_path": "/nonexistent/path/video.mp4",
                "weights": "auto",
                "use_pretrained_stack": "true",
                "use_videomae": "false",
            }
        )
    )
    client = TestClient(api_main.app)
    res = client.post(f"/jobs/{jid}/start")
    assert res.status_code == 400


def test_get_job_includes_worker_active(tmp_path: Path) -> None:
    api_main.JOB_DIR = tmp_path / "jobs"
    jid = "wa-test"
    d = api_main.JOB_DIR / jid
    d.mkdir(parents=True)
    (d / "job.json").write_text('{"job_id":"wa-test","status":"done","mode":"local"}')
    client = TestClient(api_main.app)
    res = client.get(f"/jobs/{jid}")
    assert res.status_code == 200
    assert res.json().get("worker_active") is False


def test_stats_by_track_not_found(tmp_path: Path) -> None:
    api_main.JOB_DIR = tmp_path / "jobs"
    api_main.JOB_DIR.mkdir(parents=True, exist_ok=True)
    client = TestClient(api_main.app)
    res = client.get("/jobs/missing-job/stats/by-track")
    assert res.status_code == 404
    assert res.json()["status"] == "not_found"


def test_stats_by_track_reads_json(tmp_path: Path) -> None:
    api_main.JOB_DIR = tmp_path / "jobs"
    jid = "track-stats-job"
    job_dir = api_main.JOB_DIR / jid
    job_dir.mkdir(parents=True)
    raw = [{"player_id": 9, "pts": 3, "player_number": "unknown_9"}]
    (job_dir / "stats_by_track.json").write_text(json.dumps(raw))
    client = TestClient(api_main.app)
    res = client.get(f"/jobs/{jid}/stats/by-track")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert data[0]["player_id"] == 9
