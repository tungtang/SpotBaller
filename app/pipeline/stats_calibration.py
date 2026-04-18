from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_indexed_rows(path: Path, key_field: str = "player_number") -> dict[str, dict]:
    rows = json.loads(path.read_text())
    return {str(row.get(key_field, row.get("player_id"))): row for row in rows}


def compare(video_stats: dict[str, dict], official_stats: dict[str, dict]) -> dict:
    fields = ["pts", "fga", "fgm", "three_pa", "three_pm", "reb", "ast", "stl", "blk", "tov", "efficiency"]
    per_player = []
    totals = {f: 0.0 for f in fields}
    count = 0
    for pid in sorted(set(video_stats) | set(official_stats)):
        v = video_stats.get(pid, {})
        o = official_stats.get(pid, {})
        row = {"player_key": pid}
        for f in fields:
            delta = float(v.get(f, 0) or 0) - float(o.get(f, 0) or 0)
            row[f"{f}_delta"] = delta
            totals[f] += abs(delta)
        per_player.append(row)
        count += 1
    mae = {f"mae_{f}": round(totals[f] / max(count, 1), 3) for f in fields}
    return {"players_evaluated": count, "mae": mae, "per_player_deltas": per_player}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate video-estimated stats against official boxscore export.")
    parser.add_argument("--video-stats", required=True, type=Path, help="Path to estimated stats.json")
    parser.add_argument("--official-stats", required=True, type=Path, help="Path to official stats json list")
    parser.add_argument("--out", default=Path("runtime/stats_calibration.json"), type=Path)
    args = parser.parse_args()

    video_rows = load_indexed_rows(args.video_stats, key_field="player_number")
    official_rows = load_indexed_rows(args.official_stats, key_field="player_number")
    report = compare(video_rows, official_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["mae"], indent=2))


if __name__ == "__main__":
    main()
