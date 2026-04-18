from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_rows(path: Path) -> dict[int, dict]:
    rows = json.loads(path.read_text())
    return {int(row["player_id"]): row for row in rows}


def compute_deltas(pred: dict[int, dict], truth: dict[int, dict]) -> list[dict]:
    out = []
    for pid in sorted(set(pred) | set(truth)):
        p = pred.get(pid, {})
        t = truth.get(pid, {})
        touches_delta = p.get("touches", 0) - t.get("touches", 0)
        fga_delta = p.get("fga", 0) - t.get("fga", 0)
        fgm_delta = p.get("fgm", 0) - t.get("fgm", 0)
        out.append(
            {
                "player_id": pid,
                "touches_delta": touches_delta,
                "fga_delta": fga_delta,
                "fgm_delta": fgm_delta,
                "touches_abs_error": abs(touches_delta),
                "fga_abs_error": abs(fga_delta),
                "fgm_abs_error": abs(fgm_delta),
            }
        )
    return out


def summarize(rows: list[dict]) -> dict:
    n = max(1, len(rows))
    return {
        "players_evaluated": len(rows),
        "mae_touches": round(sum(r["touches_abs_error"] for r in rows) / n, 3),
        "mae_fga": round(sum(r["fga_abs_error"] for r in rows) / n, 3),
        "mae_fgm": round(sum(r["fgm_abs_error"] for r in rows) / n, 3),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "player_id",
                "touches_delta",
                "fga_delta",
                "fgm_delta",
                "touches_abs_error",
                "fga_abs_error",
                "fgm_abs_error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare predicted stats to ground truth.")
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("runtime/benchmark"))
    args = parser.parse_args()

    pred = load_rows(args.pred)
    truth = load_rows(args.truth)
    rows = compute_deltas(pred, truth)
    summary = summarize(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))
    (args.out_dir / "benchmark_deltas.json").write_text(json.dumps(rows, indent=2))
    write_csv(rows, args.out_dir / "benchmark_deltas.csv")

    for row in rows:
        print(
            f"player={row['player_id']} "
            f"touches_delta={row['touches_delta']} "
            f"fga_delta={row['fga_delta']} "
            f"fgm_delta={row['fgm_delta']}"
        )
    print(f"summary={summary}")


if __name__ == "__main__":
    main()
