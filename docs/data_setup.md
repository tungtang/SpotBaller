# Data Pipeline Setup

## Directory structure

- `data/raw/games/` raw game videos
- `data/raw/training/` raw training videos
- `data/labels/yolo/images/` extracted frames
- `data/labels/yolo/labels/` YOLO txt labels
- `data/splits/train.txt`
- `data/splits/val.txt`
- `data/splits/test.txt`
- `data/versioned/` immutable dataset versions

## Annotation schema

Class mapping:

- `0`: player
- `1`: ball
- `2`: rim
- `3`: jersey_number (optional)

## Split policy

- Split by clip/session instead of random frames
- Default: 70% train, 15% val, 15% test
- Ensure each split has both game and training footage
- Keep camera-angle diversity in every split
