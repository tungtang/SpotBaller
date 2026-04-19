"""Jersey aggregation of per-track stats."""

from app.pipeline.reporting import aggregate_stats_by_jersey, jersey_group_key


def test_jersey_group_key_numeric() -> None:
    row = {
        "player_id": 99,
        "player_number": "23",
        "team_id": 0,
        "team_name": "Team 1",
    }
    assert jersey_group_key(row) == ("J", 0, "23")


def test_jersey_group_key_strips_leading_zeros() -> None:
    row = {"player_id": 1, "player_number": "07", "team_id": 1, "team_name": "Team 2"}
    assert jersey_group_key(row) == ("J", 1, "7")


def test_aggregate_merges_same_jersey() -> None:
    stats = [
        {
            "player_id": 10,
            "player_number": "23",
            "player_label": "#23",
            "team_id": 0,
            "team_name": "Team 1",
            "minutes_on_court": 1.0,
            "touches": 5,
            "poss": 3,
            "fga": 2,
            "fgm": 1,
            "fg_pct": 50.0,
            "three_pa": 1,
            "three_pm": 0,
            "three_pct": 0.0,
            "fta": 0,
            "ftm": 0,
            "ft_pct": 0.0,
            "oreb": 0,
            "dreb": 1,
            "reb": 1,
            "ast": 0,
            "stl": 0,
            "blk": 0,
            "tov": 0,
            "pf": 0,
            "pts": 2,
            "efficiency": 3.0,
        },
        {
            "player_id": 500,
            "player_number": "23",
            "player_label": "#23",
            "team_id": 0,
            "team_name": "Team 1",
            "minutes_on_court": 2.0,
            "touches": 3,
            "poss": 2,
            "fga": 1,
            "fgm": 1,
            "fg_pct": 100.0,
            "three_pa": 0,
            "three_pm": 0,
            "three_pct": 0.0,
            "fta": 0,
            "ftm": 0,
            "ft_pct": 0.0,
            "oreb": 0,
            "dreb": 0,
            "reb": 0,
            "ast": 1,
            "stl": 0,
            "blk": 0,
            "tov": 0,
            "pf": 0,
            "pts": 2,
            "efficiency": 4.0,
        },
    ]
    out = aggregate_stats_by_jersey(stats)
    assert len(out) == 1
    row = out[0]
    assert row["jersey_key"] == "23"
    assert set(row["merged_track_ids"]) == {10, 500}
    assert row["merged_track_count"] == 2
    assert row["touches"] == 8
    assert row["minutes_on_court"] == 3.0


def test_aggregate_keeps_unresolved_separate() -> None:
    stats = [
        {
            "player_id": 1,
            "player_number": "unknown_1",
            "player_label": "#unknown_1",
            "team_id": 0,
            "team_name": "Team 1",
            "minutes_on_court": 1.0,
            "touches": 1,
            "poss": 0,
            "fga": 0,
            "fgm": 0,
            "fg_pct": 0.0,
            "three_pa": 0,
            "three_pm": 0,
            "three_pct": 0.0,
            "fta": 0,
            "ftm": 0,
            "ft_pct": 0.0,
            "oreb": 0,
            "dreb": 0,
            "reb": 0,
            "ast": 0,
            "stl": 0,
            "blk": 0,
            "tov": 0,
            "pf": 0,
            "pts": 0,
            "efficiency": 0.0,
        },
        {
            "player_id": 2,
            "player_number": "unknown_2",
            "player_label": "#unknown_2",
            "team_id": 0,
            "team_name": "Team 1",
            "minutes_on_court": 1.0,
            "touches": 1,
            "poss": 0,
            "fga": 0,
            "fgm": 0,
            "fg_pct": 0.0,
            "three_pa": 0,
            "three_pm": 0,
            "three_pct": 0.0,
            "fta": 0,
            "ftm": 0,
            "ft_pct": 0.0,
            "oreb": 0,
            "dreb": 0,
            "reb": 0,
            "ast": 0,
            "stl": 0,
            "blk": 0,
            "tov": 0,
            "pf": 0,
            "pts": 0,
            "efficiency": 0.0,
        },
    ]
    out = aggregate_stats_by_jersey(stats)
    assert len(out) == 2
