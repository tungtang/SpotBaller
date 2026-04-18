from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from app.pipeline.identity import IdentityResolver
from app.pipeline.calibration import CourtCalibration
from app.pipeline.event_engine_v2 import StatsEventEngineV2
from app.pipeline.pretrained_stack import PipelineConfig, build_stack
from app.pipeline.reporting import build_team_box_score, merge_identity_into_stats
from app.pipeline.schemas import Detection
from app.pipeline.tracker import ByteTrackTracker, CentroidTracker

DEFAULT_EBARD_WEIGHTS = Path("models/e-bard/BODD_yolov8n_0001.pt")


def normalize_class_name(raw_name: str) -> str | None:
    name = raw_name.strip().lower()
    aliases = {
        "player": "player",
        "person": "player",
        "ball": "ball",
        "sports ball": "ball",
        "basketball": "ball",
        "rim": "rim",
        "hoop": "rim",
        "basket": "rim",
    }
    return aliases.get(name)


def resolve_weights(weights: str) -> str:
    """Prefer local E-BARD checkpoint for basketball when available."""
    if weights and weights != "auto":
        return weights
    if DEFAULT_EBARD_WEIGHTS.exists():
        return str(DEFAULT_EBARD_WEIGHTS)
    return "yolov8n.pt"


def run_video_analysis(
    video_path: Path,
    output_dir: Path,
    weights: str = "yolov8n.pt",
    pipeline_config: PipelineConfig | None = None,
    max_frames: int | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = pipeline_config or PipelineConfig()
    stack = build_stack(cfg)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    writer = cv2.VideoWriter(
        str(output_dir / "annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    resolved_weights = resolve_weights(weights)
    model = YOLO(resolved_weights)
    calibration = CourtCalibration(frame_width=width, frame_height=height)
    try:
        tracker = ByteTrackTracker()
        tracker_backend = "bytetrack"
    except Exception:
        tracker = CentroidTracker()
        tracker_backend = "centroid_fallback"
    engine = StatsEventEngineV2(frame_width=width)
    identity_resolver = IdentityResolver(stack=stack)

    events_json = []
    tracks_json = []
    action_hints_json: list[dict] = []
    videomae_aux_json: list[dict] = []
    frame_buffer: list = []
    frame_idx = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame_buffer.append(frame.copy())
        if len(frame_buffer) > 48:
            frame_buffer.pop(0)

        result = model(frame, verbose=False)[0]
        detections = []
        for box in result.boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            raw_cls_name = str(result.names[cls])
            cls_name = normalize_class_name(raw_cls_name)
            if cls_name is None:
                continue
            detections.append(Detection(cls_name, conf, x1, y1, x2, y2))

        tracks = tracker.update(detections, frame_idx)
        identity_resolver.update(frame, tracks)
        for tr in tracks:
            cx = (tr.x1 + tr.x2) / 2
            cy = (tr.y1 + tr.y2) / 2
            court_x, court_y = calibration.to_court(cx, cy)
            tracks_json.append(
                {
                    "frame_index": tr.frame_index,
                    "track_id": tr.track_id,
                    "cls_name": tr.cls_name,
                    "bbox": [tr.x1, tr.y1, tr.x2, tr.y2],
                    "court_xy": [court_x, court_y],
                }
            )
        events = engine.update(tracks, frame_idx, fps)
        events_json.extend([e.__dict__ for e in events])

        if stack and stack.has_siglip and cfg.use_siglip_action_hints and frame_idx % 12 == 0:
            scores = stack.action_hint_scores(frame)
            if scores:
                action_hints_json.append({"frame_index": frame_idx, "siglip_action_scores": scores})

        if (
            stack
            and stack.has_videomae
            and cfg.use_videomae_clips
            and len(frame_buffer) >= 16
            and frame_idx % 24 == 0
        ):
            topk = stack.videomae_topk(frame_buffer[-16:])
            if topk:
                videomae_aux_json.append(
                    {"frame_index": frame_idx, "kinetics_topk": [{"label": a, "logit": b} for a, b in topk]}
                )

        for tr in tracks:
            cv2.rectangle(frame, (int(tr.x1), int(tr.y1)), (int(tr.x2), int(tr.y2)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{tr.cls_name}:{tr.track_id}",
                (int(tr.x1), max(0, int(tr.y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        writer.write(frame)

        frame_idx += 1
        if max_frames is not None and frame_idx >= max_frames:
            break

    cap.release()
    writer.release()

    stats = engine.compute_stats(fps)
    identity_map = identity_resolver.finalize()
    stats = merge_identity_into_stats(stats, identity_map)
    team_box_score = build_team_box_score(stats)
    pipeline_meta = {
        "detection_weights": resolved_weights,
        "tracker_backend": tracker_backend,
        "bytetrack": tracker_backend == "bytetrack",
        "pretrained_config": asdict(cfg),
        "pretrained_stack": {
            "siglip_loaded": bool(stack and stack.has_siglip),
            "trocr_loaded": bool(stack and stack.has_trocr),
            "videomae_loaded": bool(stack and stack.has_videomae),
            "load_errors": list(stack.load_errors) if stack else [],
        },
    }
    result_payload = {
        "video_path": str(video_path),
        "frames_processed": frame_idx,
        "max_frames_limit": max_frames,
        "weights": resolved_weights,
        "tracker_backend": tracker_backend,
        "pipeline": pipeline_meta,
        "events": events_json,
        "tracks": tracks_json,
        "player_identity_map": identity_map,
        "team_box_score": team_box_score,
        "stats": stats,
        "action_hints": action_hints_json,
        "videomae_aux": videomae_aux_json,
    }

    (output_dir / "events.json").write_text(json.dumps(events_json, indent=2))
    (output_dir / "tracks.json").write_text(json.dumps(tracks_json, indent=2))
    (output_dir / "player_identity_map.json").write_text(json.dumps(identity_map, indent=2))
    (output_dir / "team_box_score.json").write_text(json.dumps(team_box_score, indent=2))
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    pd.DataFrame(stats).to_csv(output_dir / "stats.csv", index=False)
    (output_dir / "pipeline.json").write_text(json.dumps(pipeline_meta, indent=2))
    (output_dir / "action_hints.json").write_text(json.dumps(action_hints_json, indent=2))
    (output_dir / "videomae_aux.json").write_text(json.dumps(videomae_aux_json, indent=2))
    return result_payload
