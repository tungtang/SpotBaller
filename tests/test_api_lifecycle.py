from pathlib import Path

from fastapi.testclient import TestClient

from app.api import main as api_main


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
