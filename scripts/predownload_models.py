#!/usr/bin/env python3
"""
Predownload all external models used by SpotBaller into the local Hugging Face cache
and optional project paths. Use after `hf auth login` or set HF_TOKEN.

The Cursor Hugging Face plugin authenticates MCP; this script uses huggingface_hub
(the same Hub API as `hf download`) for bulk cache population.

Run from repo root:
  PYTHONPATH=. python3 scripts/predownload_models.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Repo root = parent of scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Align with app/pipeline/pretrained_stack.py
SIGLIP_ID = "google/siglip-base-patch16-224"
TROCR_ID = "microsoft/trocr-base-printed"
VIDEOMAE_ID = "MCG-NJU/videomae-base"

# app/pipeline/video_analyzer.py
EBARD_LOCAL = REPO_ROOT / "models" / "e-bard" / "BODD_yolov8n_0001.pt"
EBARD_REPO = "GabrieleGiudici/E-BARD-detection-models"
EBARD_CANDIDATE_FILENAMES = (
    "BODD_yolov8n_0001.pt",
    "models/BODD_yolov8n_0001.pt",
    "BODD_yolov8n_0001/best.pt",
)


def _snapshot(repo_id: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, repo_type="model")


def _download_ebard(dest: Path) -> str | None:
    from huggingface_hub import hf_hub_download, HfApi

    api = HfApi()
    try:
        raw = api.list_repo_files(EBARD_REPO, repo_type="model")
        files = [str(f.rfilename if hasattr(f, "rfilename") else f) for f in raw]
    except Exception as exc:
        print(f"[warn] Could not list {EBARD_REPO}: {exc}")
        return None

    for name in EBARD_CANDIDATE_FILENAMES:
        if name in files:
            dest.parent.mkdir(parents=True, exist_ok=True)
            path = hf_hub_download(repo_id=EBARD_REPO, filename=name, repo_type="model")
            # Copy or symlink into project — user expects fixed path
            import shutil

            shutil.copy2(path, dest)
            return str(dest)

    # Fallback: filename substring match
    for f in files:
        if f.endswith("BODD_yolov8n_0001.pt") or f.endswith("yolov8n_0001.pt"):
            dest.parent.mkdir(parents=True, exist_ok=True)
            path = hf_hub_download(repo_id=EBARD_REPO, filename=f, repo_type="model")
            import shutil

            shutil.copy2(path, dest)
            return str(dest)

    print(f"[warn] No matching E-BARD .pt in {EBARD_REPO}; files sample: {files[:8]}")
    return None


def _yolov8n() -> None:
    from ultralytics import YOLO

    YOLO("yolov8n.pt")
    print("[ok] ultralytics yolov8n.pt cached")


def main() -> int:
    parser = argparse.ArgumentParser(description="Predownload SpotBaller Hub + YOLO weights.")
    parser.add_argument(
        "--skip-ebard",
        action="store_true",
        help="Skip E-BARD download into models/e-bard/",
    )
    parser.add_argument(
        "--skip-yolo",
        action="store_true",
        help="Skip YOLOv8n bootstrap download",
    )
    args = parser.parse_args()

    if os.environ.get("HF_TOKEN"):
        print("HF_TOKEN is set (Hub auth for gated models).")

    print("Downloading Hugging Face model snapshots (SigLIP, TrOCR, VideoMAE)...")
    for rid in (SIGLIP_ID, TROCR_ID, VIDEOMAE_ID):
        try:
            path = _snapshot(rid)
            print(f"[ok] {rid} -> {path}")
        except Exception as exc:
            print(f"[error] {rid}: {exc}")
            return 1

    if not args.skip_ebard:
        print(f"Downloading E-BARD into {EBARD_LOCAL} ...")
        out = _download_ebard(EBARD_LOCAL)
        if out:
            print(f"[ok] E-BARD -> {out}")
        else:
            print("[warn] E-BARD not placed; pipeline will fall back to yolov8n.pt until fixed.")

    if not args.skip_yolo:
        try:
            _yolov8n()
        except Exception as exc:
            print(f"[error] yolov8n: {exc}")
            return 1

    print("Done. Models are in the Hugging Face cache (~/.cache/huggingface/hub) and project paths as above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
