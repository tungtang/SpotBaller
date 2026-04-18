from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 for basketball objects.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base checkpoint.")
    parser.add_argument("--data", default="app/ml/yolo_data.yaml", help="YOLO data yaml.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--project", default="runs/basketball")
    parser.add_argument("--name", default="players_ball_rim")
    args = parser.parse_args()

    Path(args.project).mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device="mps",
        patience=20,
    )


if __name__ == "__main__":
    main()
