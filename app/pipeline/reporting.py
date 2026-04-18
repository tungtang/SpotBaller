from __future__ import annotations

from collections import defaultdict


def merge_identity_into_stats(stats: list[dict], identity_map: dict[int, dict]) -> list[dict]:
    enriched = []
    for row in stats:
        pid = int(row["player_id"])
        ident = identity_map.get(pid, {})
        enriched.append(
            {
                **row,
                "player_number": ident.get("player_number", f"unknown_{pid}"),
                "player_label": ident.get("player_label", f"#unknown_{pid}"),
                "team_id": ident.get("team_id", -1),
                "team_name": ident.get("team_name", "Unknown Team"),
            }
        )
    return enriched


def build_team_box_score(stats: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in stats:
        grouped[row.get("team_name", "Unknown Team")].append(row)

    out = []
    for team_name, players in grouped.items():
        players_sorted = sorted(players, key=lambda x: str(x.get("player_number", "")))
        total_keys = [
            "pts",
            "reb",
            "ast",
            "stl",
            "blk",
            "tov",
            "pf",
            "fga",
            "fgm",
            "three_pa",
            "three_pm",
            "fta",
            "ftm",
            "touches",
        ]
        out.append(
            {
                "team_name": team_name,
                "players": [
                    {
                        "player_number": p["player_number"],
                        "player_label": p["player_label"],
                        "minutes_on_court": p["minutes_on_court"],
                        "touches": p["touches"],
                        "fga": p["fga"],
                        "fgm": p["fgm"],
                        "fg_pct": p.get("fg_pct", 0.0),
                        "three_pa": p.get("three_pa", 0),
                        "three_pm": p.get("three_pm", 0),
                        "three_pct": p.get("three_pct", 0.0),
                        "fta": p.get("fta", 0),
                        "ftm": p.get("ftm", 0),
                        "ft_pct": p.get("ft_pct", 0.0),
                        "reb": p.get("reb", 0),
                        "ast": p.get("ast", 0),
                        "stl": p.get("stl", 0),
                        "blk": p.get("blk", 0),
                        "tov": p.get("tov", 0),
                        "pf": p.get("pf", 0),
                        "pts": p.get("pts", 0),
                        "efficiency": p.get("efficiency", 0),
                    }
                    for p in players_sorted
                ],
                "team_totals": {k: sum(float(p.get(k, 0)) for p in players_sorted) for k in total_keys},
            }
        )
    return out
