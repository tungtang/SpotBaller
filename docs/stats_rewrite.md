# Stats Rewrite (Incremental)

## Goal

Rewrite analytics output to mirror a professional player box score, with identity by jersey number and team grouping.

## Metrics Targeted

- Minutes
- PTS
- FGM/FGA/FG%
- 3PM/3PA/3P%
- FTM/FTA/FT%
- OREB/DREB/REB
- AST
- STL
- BLK
- TOV
- PF
- Efficiency proxy

## Data Source Integration

- Primary: ingest official box score exports when available.
- Development fallback: local metric schema and event-derived approximations from video pipeline.
- Production rule: keep both `estimated_from_video` and `official_boxscore` fields when official feed exists.

## Best Model Shortlist by Task

- Detection (player/ball/rim): `GabrieleGiudici/E-BARD-detection-models`
- Tracking: `ByteTrack`
- Jersey number recognition: PARSeq-style STR + tracklet voting
- Team recognition: CLIP/SigLIP embeddings + clustering
- Event verification: VideoMAE/3D CNN action model
