from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


def merge_identity_into_stats(stats: list[dict], identity_map: dict[int, dict]) -> list[dict]:
    enriched = []
    for row in stats:
        pid = int(row["player_id"])
        ident = identity_map.get(pid, {})
        raw_number = str(ident.get("player_number", f"unknown_{pid}"))
        m = re.fullmatch(r"(\d{1,2})", raw_number.strip())
        jersey_number = str(int(m.group(1))) if m else None
        enriched.append(
            {
                **row,
                "player_number": raw_number,
                "jersey_number": jersey_number,
                "jersey_key": jersey_number if jersey_number is not None else f"unresolved_{pid}",
                "player_label": ident.get("player_label", f"#unknown_{pid}"),
                "player_number_confidence": float(ident.get("player_number_confidence", 0.0) or 0.0),
                "team_id": ident.get("team_id", -1),
                "team_name": ident.get("team_name", "Unknown Team"),
            }
        )
    return enriched


def jersey_group_key(row: dict) -> tuple[str, int, str]:
    """
    Group stats rows for aggregation: resolved jersey digits vs one bucket per unresolved track.

    Returns (kind, team_id, key) where kind is ``J`` (jersey) or ``U`` (unresolved track).
    """
    tid = int(row["player_id"])
    team_id = int(row.get("team_id", -1))
    pn = str(row.get("player_number", "")).strip()
    m = re.fullmatch(r"(\d{1,2})", pn)
    if m:
        return ("J", team_id, str(int(m.group(1))))
    return ("U", team_id, str(tid))


class JerseyNumberTracker:
    """
    Post-processing layer: merge per-track stats into jersey-number rows for reporting.

    The MOT ``player_id`` (ByteTrack id) is not stable across a game; this aggregates
    all track rows that map to the same detected jersey (and team) into one insight row.
    """

    @staticmethod
    def aggregate_stats(stats_with_identity: list[dict]) -> list[dict]:
        """Alias for :func:`aggregate_stats_by_jersey`."""
        return aggregate_stats_by_jersey(stats_with_identity)


def aggregate_stats_by_jersey(stats_with_identity: list[dict]) -> list[dict]:
    """
    Sum counting stats across tracks that share the same jersey number (and team).

    Rows without a numeric jersey stay separate (one row per unresolved track id).
    """
    if not stats_with_identity:
        return []

    groups: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in stats_with_identity:
        groups[jersey_group_key(row)].append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(groups.keys(), key=lambda k: (k[1], k[0], k[2])):
        rows = groups[key]
        merged = _merge_stat_rows_for_jersey(key, rows)
        out.append(merged)
    return out


def _merge_stat_rows_for_jersey(key: tuple[str, int, str], rows: list[dict]) -> dict[str, Any]:
    kind, team_id, key_part = key
    track_ids = sorted({int(r["player_id"]) for r in rows})
    team_name = str(rows[0].get("team_name", "Unknown Team"))

    sum_keys = (
        "minutes_on_court",
        "touches",
        "poss",
        "fga",
        "fgm",
        "three_pa",
        "three_pm",
        "fta",
        "ftm",
        "oreb",
        "dreb",
        "ast",
        "stl",
        "blk",
        "tov",
        "pf",
    )
    acc: dict[str, float] = {k: 0.0 for k in sum_keys}
    for r in rows:
        for k in sum_keys:
            acc[k] += float(r.get(k, 0) or 0)

    acc["reb"] = acc["oreb"] + acc["dreb"]
    fga, fgm = acc["fga"], acc["fgm"]
    tpa, tpm = acc["three_pa"], acc["three_pm"]
    fta, ftm = acc["fta"], acc["ftm"]
    acc["fg_pct"] = round((fgm / fga) * 100, 1) if fga else 0.0
    acc["three_pct"] = round((tpm / tpa) * 100, 1) if tpa else 0.0
    acc["ft_pct"] = round((ftm / fta) * 100, 1) if fta else 0.0
    pts = 2.0 * (fgm - tpm) + 3.0 * tpm + ftm
    acc["pts"] = round(pts, 1)
    eff = (
        pts
        + acc["reb"]
        + acc["ast"]
        + acc["stl"]
        + acc["blk"]
        - (fga - fgm)
        - (fta - ftm)
        - acc["tov"]
    )
    acc["efficiency"] = round(eff, 1)
    acc["minutes_on_court"] = round(acc["minutes_on_court"], 2)

    if kind == "J":
        jersey_key = key_part
        player_label = f"#{jersey_key}"
        player_number = jersey_key
        identity_kind = "jersey"
    else:
        jersey_key = f"unresolved_{track_ids[0]}"
        if len(track_ids) == 1:
            player_label = f"Unresolved (track {track_ids[0]})"
        else:
            player_label = f"Unresolved (tracks {', '.join(map(str, track_ids))})"
        player_number = f"?{track_ids[0]}"
        identity_kind = "unresolved_track"

    return {
        "identity_kind": identity_kind,
        "jersey_key": jersey_key,
        "player_number": player_number,
        "player_label": player_label,
        "team_id": team_id,
        "team_name": team_name,
        "merged_track_ids": track_ids,
        "merged_track_count": len(track_ids),
        "minutes_on_court": acc["minutes_on_court"],
        "touches": int(acc["touches"]),
        "poss": int(acc["poss"]),
        "fga": int(acc["fga"]),
        "fgm": int(acc["fgm"]),
        "fg_pct": acc["fg_pct"],
        "three_pa": int(acc["three_pa"]),
        "three_pm": int(acc["three_pm"]),
        "three_pct": acc["three_pct"],
        "fta": int(acc["fta"]),
        "ftm": int(acc["ftm"]),
        "ft_pct": acc["ft_pct"],
        "oreb": int(acc["oreb"]),
        "dreb": int(acc["dreb"]),
        "reb": int(acc["reb"]),
        "ast": int(acc["ast"]),
        "stl": int(acc["stl"]),
        "blk": int(acc["blk"]),
        "tov": int(acc["tov"]),
        "pf": int(acc["pf"]),
        "pts": int(acc["pts"]) if acc["pts"] == int(acc["pts"]) else acc["pts"],
        "efficiency": acc["efficiency"],
    }


def build_team_box_score(stats: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in stats:
        grouped[row.get("team_name", "Unknown Team")].append(row)

    out = []
    for team_name, players in grouped.items():
        players_sorted = sorted(
            players,
            key=lambda x: str(x.get("jersey_key", x.get("player_number", ""))),
        )
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
                        "player_number": p.get("player_number") or p.get("jersey_key"),
                        "player_label": p["player_label"],
                        "merged_track_ids": p.get("merged_track_ids"),
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
