from app.pipeline.event_engine_v2 import StatsEventEngineV2
from app.pipeline.schemas import Track


def test_event_engine_v2_outputs_fields() -> None:
    engine = StatsEventEngineV2(frame_width=1280)
    fps = 30.0
    tracks = [
        Track(track_id=1, cls_name="player", frame_index=0, x1=100, y1=100, x2=150, y2=220),
        Track(track_id=2, cls_name="ball", frame_index=0, x1=120, y1=120, x2=132, y2=132),
        Track(track_id=3, cls_name="rim", frame_index=0, x1=115, y1=110, x2=140, y2=130),
    ]
    engine.update(tracks, frame_index=0, fps=fps)
    rows = engine.compute_stats(fps=fps)
    assert rows
    row = rows[0]
    assert "three_pa" in row
    assert "pts" in row
    assert "efficiency" in row
