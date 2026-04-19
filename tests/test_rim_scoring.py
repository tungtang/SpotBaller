"""Rim merge + class normalization for scoring."""

from app.pipeline.rim_scoring import RIM_TRACK_ID, RimFallbackConfig, append_rim_track
from app.pipeline.schemas import Detection, Track
from app.pipeline.video_analyzer import normalize_class_name


def test_normalize_class_name_rim_aliases() -> None:
    assert normalize_class_name("Backboard") == "rim"
    assert normalize_class_name("basketball hoop") == "rim"
    assert normalize_class_name("custom_hoop_v1") == "rim"


def test_append_rim_track_from_detection() -> None:
    dets = [
        Detection("player", 0.9, 0, 0, 10, 10),
        Detection("rim", 0.8, 100, 50, 200, 80),
    ]
    tracks = [Track(1, "player", 0, 0, 0, 10, 10)]
    out, src = append_rim_track(
        tracks,
        frame_index=0,
        detections=dets,
        frame_width=640,
        frame_height=480,
        fallback_cfg=RimFallbackConfig(enabled=True),
    )
    assert src == "detected"
    assert any(t.cls_name == "rim" and t.track_id == RIM_TRACK_ID for t in out)


def test_append_rim_skips_if_tracker_has_rim() -> None:
    dets = [Detection("rim", 0.9, 0, 0, 5, 5)]
    tracks = [Track(99, "rim", 0, 0, 0, 5, 5)]
    out, src = append_rim_track(
        tracks,
        frame_index=0,
        detections=dets,
        frame_width=640,
        frame_height=480,
        fallback_cfg=RimFallbackConfig(enabled=True),
    )
    assert src == "tracker"
    assert len(out) == 1
