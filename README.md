# Basketball Video Analytics

Local-first and web-deployable basketball video analytics app built around YOLOv8.

## What is included

- MVP specification and quality targets
- Dataset layout and labeling policy
- YOLOv8 training/evaluation scripts
- Detection + tracking + event/stat extraction pipeline
- Local FastAPI app and Streamlit dashboard
- Cloud-ready API, worker, and deployment files
- Court-normalized coordinates in `tracks.json`

## Quick start (Mac)

1. Create environment and install dependencies:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Run local API:
   - `uvicorn app.api.main:app --reload`
3. Run local dashboard:
   - `streamlit run app/ui/dashboard.py`
4. Run local CLI directly:
   - `python -m app.run_local --video path/to/video.mp4`

## E-BARD model integration

- Downloaded and integrated local checkpoint:
  - `models/e-bard/BODD_yolov8n_0001.pt`
- By default, API/UI/CLI now use `weights=auto`:
  - Uses E-BARD if present.
  - Falls back to `yolov8n.pt` otherwise.

## Testing

- Install dev dependencies:
  - `pip install -r requirements-dev.txt`
- Run tests:
  - `python3 -m pytest -q`

## Benchmarking

- Compare predicted stats against ground truth:
  - `python -m app.pipeline.validate_stats --pred runtime/jobs/<id>/stats.json --truth path/to/truth.json --out-dir runtime/benchmark`
- Output files:
  - `benchmark_summary.json`
  - `benchmark_deltas.json`
  - `benchmark_deltas.csv`

## Stats calibration

- Compare estimated stats with official boxscore export:
  - `python -m app.pipeline.stats_calibration --video-stats runtime/jobs/<id>/stats.json --official-stats path/to/official_boxscore.json --out runtime/stats_calibration.json`

## Model training workflow

- Full 2-week plan:
  - `docs/model_training_plan_2weeks.md`
- Run one training iteration:
  - `python -m app.ml.train_iteration --name iter1 --model models/e-bard/BODD_yolov8n_0001.pt`
- Mine hard error cases after calibration:
  - `python -m app.ml.error_mining --calibration-report runtime/stats_calibration.json --top-k 20 --out runtime/hard_cases.json`

## API endpoints

- `POST /videos` upload
- `POST /jobs` (`mode=local` or `mode=cloud`)
  - Optional `weights` form field (default `auto`)
- `GET /jobs/{id}` status and payload
- `GET /jobs/{id}/stats` stats JSON
- `GET /jobs/{id}/team-stats` grouped team/player-number stats
- `GET /jobs/{id}/artifacts` file artifact paths
- `GET /health/summary` aggregated job status diagnostics

## Production tracking and metadata

- Tracker backend defaults to ByteTrack and falls back to centroid tracking if unavailable.
- Job metadata is persisted to PostgreSQL via SQLAlchemy (`app/api/db.py`).
- For local development, set `DATABASE_URL` before starting API/worker.
- Player identity output now includes jersey number OCR (if Tesseract is available) with fallback labels.
- Team recognition is generated from jersey color clustering and exported in `player_identity_map.json`.
- Team/player-number grouped stats are exported in `team_box_score.json`.

## Directory overview

- `docs/` planning and product requirements
- `data/` dataset placeholders and split manifests
- `app/ml/` model training and inference components
- `app/pipeline/` tracking, events, and stat engine
- `app/api/` FastAPI services and job orchestration
- `app/ui/` local dashboard
- `infra/` deployment and container assets

## Git (version control)

This repo ignores large artifacts: `runtime/`, `*.pt` weights, and `basketball_analysis/input_videos/*` (place videos locally). After `git init`, connect your GitHub account:

```bash
git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

Use SSH (`git@github.com:...`) if your SSH key is added to GitHub, or HTTPS with a personal access token. To create a new empty repo on GitHub: **New repository** (no README), then run the commands above.
