# MVP Spec

## Supported Footage

- Full-court game footage from fixed sideline/endline camera
- Training footage with half-court or drill camera
- Recommended input: 1080p, 30 FPS (minimum), MP4/H.264

## MVP Outputs

- Frame-level detections: `player`, `ball`, `rim`
- Track timelines for players and ball
- Event timeline:
  - `shot_attempt`
  - `shot_made`
  - `shot_missed`
  - `possession_change`
- Player stats:
  - Minutes on court
  - Touches
  - Field goals attempted
  - Field goals made

## Quality Targets

- Detector mAP50:
  - players >= 0.85
  - ball >= 0.60
  - rim >= 0.80
- Tracking continuity:
  - ID switch rate <= 0.15 on validation clips
- Stat accuracy vs manual labels:
  - Shots attempted error <= 15%
  - Shots made error <= 10%
  - Touches error <= 20%
