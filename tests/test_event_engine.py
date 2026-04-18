from app.pipeline.event_engine import EventEngine
from app.pipeline.video_analyzer import normalize_class_name
from app.pipeline.reporting import build_team_box_score, merge_identity_into_stats
from app.pipeline.schemas import Track


def test_presence_and_stats_compute() -> None:
    engine = EventEngine()
    fps = 30.0
    tracks = [
        Track(track_id=1, cls_name="player", frame_index=0, x1=10, y1=10, x2=20, y2=20),
        Track(track_id=2, cls_name="ball", frame_index=0, x1=12, y1=10, x2=14, y2=12),
        Track(track_id=3, cls_name="rim", frame_index=0, x1=100, y1=100, x2=140, y2=130),
    ]
    engine.update(tracks, 0, fps)
    stats = engine.compute_stats(fps)
    assert stats
    assert stats[0]["player_id"] == 1
    assert "three_pa" in stats[0]
    assert "stl" in stats[0]
    assert "efficiency" in stats[0]


def test_team_box_score_grouping() -> None:
    stats = [
        {"player_id": 1, "minutes_on_court": 1.0, "touches": 4, "fga": 3, "fgm": 2},
        {"player_id": 2, "minutes_on_court": 1.2, "touches": 3, "fga": 1, "fgm": 1},
    ]
    identity = {
        1: {"player_number": "23", "player_label": "#23", "team_id": 0, "team_name": "Team 1"},
        2: {"player_number": "8", "player_label": "#8", "team_id": 1, "team_name": "Team 2"},
    }
    merged = merge_identity_into_stats(stats, identity)
    team_box = build_team_box_score(merged)
    assert len(team_box) == 2
    assert any(team["team_name"] == "Team 1" for team in team_box)


def test_class_name_normalization() -> None:
    assert normalize_class_name("person") == "player"
    assert normalize_class_name("sports ball") == "ball"
    assert normalize_class_name("basket") == "rim"
    assert normalize_class_name("car") is None
