from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one train/eval iteration and save report.")
    parser.add_argument("--name", required=True, help="Iteration name, e.g. iter1")
    parser.add_argument("--model", default="models/e-bard/BODD_yolov8n_0001.pt")
    parser.add_argument("--data", default="app/ml/yolo_data.yaml")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--project", default="runs/training_iterations")
    args = parser.parse_args()

    project_dir = Path(args.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(project_dir),
        name=args.name,
        device="mps",
        patience=20,
    )

    best_weights = project_dir / args.name / "weights" / "best.pt"
    eval_model = YOLO(str(best_weights))
    metrics = eval_model.val(data=args.data, split="val")

    report = {
        "iteration": args.name,
        "init_model": args.model,
        "best_weights": str(best_weights),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    report_path = project_dir / args.name / "iteration_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
