import json
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.main import JOB_DIR, app


def test_compare_performance_endpoint(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("app.api.main.JOB_DIR", tmp_path)
    a = str(uuid.uuid4())
    b = str(uuid.uuid4())
    (tmp_path / a).mkdir(parents=True)
    (tmp_path / b).mkdir(parents=True)
    (tmp_path / a / "performance.json").write_text(
        json.dumps({"elapsed_s": 10.0, "fps_effective": 1.2, "x": 1})
    )
    (tmp_path / b / "performance.json").write_text(
        json.dumps({"elapsed_s": 20.0, "fps_effective": 0.6, "x": 2})
    )

    client = TestClient(app)
    r = client.get("/jobs/compare-performance", params={"job_a": a, "job_b": b})
    assert r.status_code == 200
    data = r.json()
    assert data["performance_a"]["elapsed_s"] == 10.0
    assert data["performance_b"]["fps_effective"] == 0.6
    assert any(row["metric"] == "elapsed_s" for row in data["rows"])
