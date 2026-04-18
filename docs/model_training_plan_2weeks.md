# 2-Week Model Training Plan

## Goal

Improve basketball analytics accuracy with a repeatable train/evaluate loop on your own data.

## Target Metrics (Validation Set)

- Detection:
  - `player mAP50 >= 0.88`
  - `ball mAP50 >= 0.70`
  - `rim mAP50 >= 0.85`
- Tracking:
  - ID switch rate <= 0.12 on benchmark clips
- Stats quality (against official/manual box score):
  - `FGA MAE <= 1.5`
  - `FGM MAE <= 1.2`
  - `3PA MAE <= 1.2`
  - `REB MAE <= 2.0`
  - `AST MAE <= 1.8`

## Week 1 (Data + Baseline)

- Day 1:
  - Gather 30-60 representative clips into `data/raw/`.
  - Ensure variation in camera angle, zoom, lighting, and court.
- Day 2:
  - Label `player`, `ball`, `rim`.
  - Prioritize hard ball frames (motion blur, occlusion, long distance).
- Day 3:
  - Build train/val/test manifests using `app/ml/prepare_data.py`.
  - Verify split by game/session (no leakage).
- Day 4:
  - Train baseline using E-BARD checkpoint as init.
- Day 5:
  - Evaluate baseline and save metrics.
  - Start error mining (false negatives and unstable tracks).
- Day 6-7:
  - Relabel hard cases and run second training iteration.

## Week 2 (Iteration + Calibration)

- Day 8-9:
  - Train iteration 3 with hard-case-heavy data.
  - Tune confidence and NMS thresholds.
- Day 10:
  - Run end-to-end video analysis on benchmark clips.
- Day 11:
  - Compare estimated stats vs official/manual stats using calibration script.
- Day 12:
  - Tune event logic only where metric deltas are largest.
- Day 13:
  - Final training iteration and validation pass.
- Day 14:
  - Freeze best model and publish model card + known limitations.

## Pass / Fail Gates

- Pass if all detection metrics and at least 4/5 stat MAE targets are met.
- Fail if ball mAP50 < 0.65 or stats MAE degrades vs prior best.

## Immediate Next Command

Run this first once data is labeled:

`python -m app.ml.train_iteration --name iter1 --model models/e-bard/BODD_yolov8n_0001.pt`
