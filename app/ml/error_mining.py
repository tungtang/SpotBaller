from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Select hardest stat errors for relabeling focus.")
    parser.add_argument("--calibration-report", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("runtime/hard_cases.json"))
    args = parser.parse_args()

    report = json.loads(args.calibration_report.read_text())
    rows = report.get("per_player_deltas", [])

    def score(row: dict) -> float:
        keys = ["fga_delta", "fgm_delta", "three_pa_delta", "reb_delta", "ast_delta", "stl_delta", "blk_delta"]
        return sum(abs(float(row.get(k, 0) or 0)) for k in keys)

    ranked = sorted(rows, key=score, reverse=True)[: args.top_k]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"top_hard_cases": ranked}, indent=2))
    print(f"Saved {len(ranked)} hard cases -> {args.out}")


if __name__ == "__main__":
    main()
