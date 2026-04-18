from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model on validation set.")
    parser.add_argument("--weights", required=True, help="Path to trained weights.")
    parser.add_argument("--data", default="app/ml/yolo_data.yaml")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split="val")

    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)


if __name__ == "__main__":
    main()
