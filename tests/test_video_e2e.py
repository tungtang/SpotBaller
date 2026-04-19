"""
End-to-end smoke: synthetic MP4 → full ``run_video_analysis`` (short, no HF pretrained stack).

Requires: opencv-python, ultralytics (YOLO weights may download on first run — use network).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from app.pipeline.pretrained_stack import pipeline_config_from_flags
from app.pipeline.video_analyzer import run_video_analysis


@pytest.mark.integration
def test_run_video_analysis_end_to_end(tmp_path: Path) -> None:
    """Run detector + tracker + stats export on a tiny synthetic clip (3 frames)."""
    mp4 = tmp_path / "synthetic.mp4"
    w, h = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(mp4), fourcc, 10.0, (w, h))
    assert vw.isOpened(), "VideoWriter failed (codec/host)"
    for _ in range(8):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (55, 55, 55)
        vw.write(frame)
    vw.release()

    out_dir = tmp_path / "job_out"
    cfg = pipeline_config_from_flags("false", "false")
    result = run_video_analysis(
        mp4,
        out_dir,
        weights="yolov8n.pt",
        pipeline_config=cfg,
        max_frames=3,
    )

    assert result["frames_processed"] == 3
    assert (out_dir / "stats.json").is_file()
    assert (out_dir / "stats_by_track.json").is_file()
    assert (out_dir / "tracks.json").is_file()
    assert (out_dir / "events.json").is_file()
    assert (out_dir / "progress.json").is_file()

    stats = json.loads((out_dir / "stats.json").read_text())
    raw = json.loads((out_dir / "stats_by_track.json").read_text())
    assert isinstance(stats, list)
    assert isinstance(raw, list)
    assert result["stats"] == stats
    assert result["stats_by_track"] == raw

    for row in stats:
        assert "jersey_key" in row
        assert "merged_track_ids" in row

    prog = json.loads((out_dir / "progress.json").read_text())
    assert prog.get("status") == "complete"
