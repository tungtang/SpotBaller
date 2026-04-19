from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from app.pipeline.identity import IdentityResolver
from app.pipeline.calibration import CourtCalibration
from app.pipeline.event_engine_v2 import StatsEventEngineV2
from app.pipeline.cuda_perf import apply_cuda_runtime_tuning, resolve_inference_device
from app.pipeline.pretrained_stack import PipelineConfig, build_stack
from app.pipeline.reporting import aggregate_stats_by_jersey, build_team_box_score, merge_identity_into_stats
from app.pipeline.rim_scoring import RIM_TRACK_ID, RimFallbackConfig, append_rim_track
from app.pipeline.schemas import Detection
from app.pipeline.tracker import ByteTrackTracker, CentroidTracker

DEFAULT_EBARD_WEIGHTS = Path("models/e-bard/BODD_yolov8n_0001.pt")

# Write progress.json at most every N frames (plus first and last) for UI polling.
PROGRESS_WRITE_INTERVAL = 14


def planned_total_frames_for_video(video_path: Path, max_frames: int | None) -> int | None:
    """Same ``total_target`` semantics as :func:`run_video_analysis` for progress UI (local or pre-VM stub)."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        raw_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if max_frames is not None:
        return max_frames if raw_total <= 0 else min(raw_total, max_frames)
    if raw_total > 0:
        return raw_total
    return None


def _write_progress_json(
    output_dir: Path,
    frames_processed: int,
    total_frames: int | None,
    *,
    status: str = "processing",
) -> None:
    """Emit frames_processed / total_frames and percent_complete for Streamlit / API polling."""
    pct: float | None = None
    if total_frames and total_frames > 0:
        pct = min(100.0, round(100.0 * float(frames_processed) / float(total_frames), 2))
    payload = {
        "status": status,
        "frames_processed": frames_processed,
        "total_frames": total_frames,
        "percent_complete": pct,
    }
    (output_dir / "progress.json").write_text(json.dumps(payload, indent=2))


def _finalize_progress_json(output_dir: Path, frames_processed: int, planned_total: int) -> None:
    """Mark analysis complete; total_frames matches processed count when unknown from container."""
    display_total = planned_total if planned_total > 0 else frames_processed
    payload = {
        "status": "complete",
        "frames_processed": frames_processed,
        "total_frames": display_total,
        "percent_complete": 100.0,
    }
    (output_dir / "progress.json").write_text(json.dumps(payload, indent=2))


class AnalysisStoppedError(Exception):
    """Cooperative stop requested (e.g. user clicked Stop)."""

    def __init__(self, frames_processed: int) -> None:
        self.frames_processed = frames_processed


def _write_progress_user_stopped(
    output_dir: Path,
    frames_processed: int,
    total_frames: int | None,
    *,
    reason: str = "user_requested",
) -> None:
    pct: float | None = None
    if total_frames and total_frames > 0:
        pct = min(100.0, round(100.0 * float(frames_processed) / float(total_frames), 2))
    payload = {
        "status": "stopped",
        "stop_reason": reason,
        "frames_processed": frames_processed,
        "total_frames": total_frames,
        "percent_complete": pct,
    }
    (output_dir / "progress.json").write_text(json.dumps(payload, indent=2))


def normalize_class_name(raw_name: str) -> str | None:
    """Map detector class strings to pipeline roles (player / ball / rim)."""
    name = raw_name.strip().lower()
    exact = {
        "player": "player",
        "person": "player",
        "ball": "ball",
        "sports ball": "ball",
        "basketball": "ball",
        "rim": "rim",
        "hoop": "rim",
        "basket": "rim",
        "backboard": "rim",
        "net": "rim",
        "basketball hoop": "rim",
        "basket hoop": "rim",
        "basketball rim": "rim",
    }
    if name in exact:
        return exact[name]
    # Dataset-specific labels (substring)
    if any(k in name for k in ("hoop", "backboard", "back board")):
        return "rim"
    if "rim" in name:
        return "rim"
    if name in ("goal", "ring", "basket ring"):
        return "rim"
    if "ball" in name:
        return "ball"
    if "player" in name or "person" in name or "athlete" in name:
        return "player"
    return None


def resolve_weights(weights: str) -> str:
    """Prefer local E-BARD checkpoint for basketball when available."""
    if weights and weights != "auto":
        return weights
    if DEFAULT_EBARD_WEIGHTS.exists():
        return str(DEFAULT_EBARD_WEIGHTS)
    return "yolov8n.pt"


def _coerce_identity_stride(identity_stride: int | None) -> int:
    """Default 1; env ``SPOTBALLER_IDENTITY_STRIDE`` when ``identity_stride`` is None."""
    if identity_stride is not None:
        return max(1, int(identity_stride))
    raw = os.environ.get("SPOTBALLER_IDENTITY_STRIDE", "").strip()
    if not raw:
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def run_video_analysis(
    video_path: Path,
    output_dir: Path,
    weights: str = "yolov8n.pt",
    pipeline_config: PipelineConfig | None = None,
    max_frames: int | None = None,
    stop_event: object | None = None,
    prefetch_frames: int = 0,
    async_writer: bool = False,
    identity_workers: int = 0,
    rim_fallback: bool = True,
    identity_stride: int | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_cuda_runtime_tuning()
    inference_device = resolve_inference_device()
    identity_stride = _coerce_identity_stride(identity_stride)
    cfg = pipeline_config or PipelineConfig()
    stack = build_stack(cfg)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    raw_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames is not None:
        total_target = max_frames if raw_total <= 0 else min(raw_total, max_frames)
    elif raw_total > 0:
        total_target = raw_total
    else:
        total_target = 0
    _write_progress_json(
        output_dir,
        0,
        total_target if total_target > 0 else None,
    )
    writer = cv2.VideoWriter(
        str(output_dir / "annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    resolved_weights = resolve_weights(weights)
    model = YOLO(resolved_weights)
    if inference_device != "cpu":
        try:
            model.to(inference_device)
        except Exception:
            pass
    calibration = CourtCalibration(frame_width=width, frame_height=height)
    try:
        tracker = ByteTrackTracker()
        tracker_backend = "bytetrack"
    except Exception:
        tracker = CentroidTracker()
        tracker_backend = "centroid_fallback"
    engine = StatsEventEngineV2(frame_width=width, frame_height=height)
    identity_resolver = IdentityResolver(stack=stack, workers=identity_workers)
    rim_fallback_cfg = RimFallbackConfig(enabled=rim_fallback)
    rim_frame_counts: dict[str, int] = {"detected": 0, "fallback": 0, "none": 0, "tracker": 0}

    events_json = []
    tracks_json = []
    action_hints_json: list[dict] = []
    videomae_aux_json: list[dict] = []
    frame_buffer: list = []
    frame_idx = 0
    t_read = 0.0
    t_infer = 0.0
    t_track = 0.0
    t_pretrained = 0.0
    t_render = 0.0
    t_write = 0.0
    t_loop_start = time.perf_counter()

    use_prefetch = prefetch_frames > 0
    frame_q: queue.Queue[tuple[bool, object]] | None = None
    prefetch_thread: threading.Thread | None = None
    prefetch_stop = threading.Event()

    if use_prefetch:
        frame_q = queue.Queue(maxsize=prefetch_frames)

        def _prefetch_worker() -> None:
            while not prefetch_stop.is_set():
                ok, frm = cap.read()
                if not ok:
                    try:
                        frame_q.put((False, None), timeout=0.25)
                    except queue.Full:
                        pass
                    return
                while not prefetch_stop.is_set():
                    try:
                        frame_q.put((True, frm), timeout=0.25)
                        break
                    except queue.Full:
                        continue

        prefetch_thread = threading.Thread(target=_prefetch_worker, name="frame-prefetch", daemon=True)
        prefetch_thread.start()

    use_async_writer = async_writer
    write_q: queue.Queue[object] | None = None
    writer_thread: threading.Thread | None = None
    writer_stop = threading.Event()
    writer_exc: list[Exception] = []
    write_sentinel = object()

    if use_async_writer:
        write_q = queue.Queue(maxsize=32)

        def _writer_worker() -> None:
            nonlocal t_write
            while not writer_stop.is_set():
                item = write_q.get()
                if item is write_sentinel:
                    write_q.task_done()
                    return
                try:
                    tw0 = time.perf_counter()
                    writer.write(item)  # type: ignore[arg-type]
                    t_write += time.perf_counter() - tw0
                except Exception as exc:  # pragma: no cover
                    writer_exc.append(exc)
                finally:
                    write_q.task_done()

        writer_thread = threading.Thread(target=_writer_worker, name="video-writer", daemon=True)
        writer_thread.start()

    while cap.isOpened():
        if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
            prefetch_stop.set()
            writer_stop.set()
            cap.release()
            if use_async_writer and write_q is not None:
                try:
                    write_q.put_nowait(write_sentinel)
                except queue.Full:
                    pass
            writer.release()
            _write_progress_user_stopped(
                output_dir,
                frame_idx,
                total_target if total_target > 0 else None,
                reason="user_requested",
            )
            raise AnalysisStoppedError(frame_idx)

        t0 = time.perf_counter()
        if use_prefetch and frame_q is not None:
            ok, frame_obj = frame_q.get()
            frame_q.task_done()
            frame = frame_obj
        else:
            ok, frame = cap.read()
        t_read += time.perf_counter() - t0
        if not ok:
            break

        frame_buffer.append(frame.copy())
        if len(frame_buffer) > 48:
            frame_buffer.pop(0)

        t1 = time.perf_counter()
        result = model(frame, verbose=False)[0]
        t_infer += time.perf_counter() - t1
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

        t2 = time.perf_counter()
        tracks = tracker.update(detections, frame_idx)
        tracks, rim_src = append_rim_track(
            tracks,
            frame_index=frame_idx,
            detections=detections,
            frame_width=width,
            frame_height=height,
            fallback_cfg=rim_fallback_cfg,
        )
        rim_frame_counts[rim_src] = rim_frame_counts.get(rim_src, 0) + 1
        if frame_idx % identity_stride == 0:
            identity_resolver.update(frame, tracks)
        t_track += time.perf_counter() - t2
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

        tp0 = time.perf_counter()
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
        t_pretrained += time.perf_counter() - tp0

        tr0 = time.perf_counter()
        for tr in tracks:
            color = (0, 255, 200) if tr.cls_name == "rim" else (0, 255, 0)
            thick = 3 if tr.cls_name == "rim" else 2
            cv2.rectangle(frame, (int(tr.x1), int(tr.y1)), (int(tr.x2), int(tr.y2)), color, thick)
            cv2.putText(
                frame,
                f"{tr.cls_name}:{tr.track_id}",
                (int(tr.x1), max(0, int(tr.y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        t_render += time.perf_counter() - tr0
        if use_async_writer and write_q is not None:
            write_q.put(frame.copy())
        else:
            tw0 = time.perf_counter()
            writer.write(frame)
            t_write += time.perf_counter() - tw0

        frame_idx += 1
        if (
            frame_idx == 1
            or frame_idx % PROGRESS_WRITE_INTERVAL == 0
            or (total_target > 0 and frame_idx >= total_target)
        ):
            _write_progress_json(
                output_dir,
                frame_idx,
                total_target if total_target > 0 else None,
            )
        if max_frames is not None and frame_idx >= max_frames:
            break

    prefetch_stop.set()
    if prefetch_thread is not None:
        prefetch_thread.join(timeout=1.0)
    cap.release()
    if use_async_writer and write_q is not None:
        write_q.put(write_sentinel)
        write_q.join()
        writer_stop.set()
        if writer_thread is not None:
            writer_thread.join(timeout=1.0)
    if writer_exc:
        raise writer_exc[0]
    writer.release()
    _write_progress_json(
        output_dir,
        frame_idx,
        total_target if total_target > 0 else None,
        status="processing",
    )

    stats = engine.compute_stats(fps)
    identity_map = identity_resolver.finalize()
    stats_by_track = merge_identity_into_stats(stats, identity_map)
    stats = aggregate_stats_by_jersey(stats_by_track)
    team_box_score = build_team_box_score(stats)
    elapsed_s = max(0.0, time.perf_counter() - t_loop_start)
    perf = {
        "elapsed_s": round(elapsed_s, 4),
        "fps_effective": round((frame_idx / elapsed_s), 3) if elapsed_s > 0 else 0.0,
        "stage_s": {
            "frame_read": round(t_read, 4),
            "detection_infer": round(t_infer, 4),
            "tracking_identity": round(t_track, 4),
            "pretrained_aux": round(t_pretrained, 4),
            "render_overlays": round(t_render, 4),
            "video_write": round(t_write, 4),
        },
        "parallel": {
            "prefetch_frames": prefetch_frames,
            "async_writer": bool(async_writer),
            "identity_workers": int(identity_workers),
            "identity_stride": int(identity_stride),
        },
    }
    pipeline_meta = {
        "inference_device": inference_device,
        "identity_stride": identity_stride,
        "detection_weights": resolved_weights,
        "tracker_backend": tracker_backend,
        "bytetrack": tracker_backend == "bytetrack",
        "rim_scoring": {
            "rim_fallback_enabled": rim_fallback,
            "frames_by_rim_source": dict(sorted(rim_frame_counts.items())),
            "rim_track_id_constant": RIM_TRACK_ID,
        },
        "pretrained_config": asdict(cfg),
        "pretrained_stack": {
            "siglip_loaded": bool(stack and stack.has_siglip),
            "trocr_loaded": bool(stack and stack.has_trocr),
            "videomae_loaded": bool(stack and stack.has_videomae),
            "load_errors": list(stack.load_errors) if stack else [],
        },
        "performance": perf,
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
        "stats_by_track": stats_by_track,
        "action_hints": action_hints_json,
        "videomae_aux": videomae_aux_json,
        "performance": perf,
    }

    (output_dir / "events.json").write_text(json.dumps(events_json, indent=2))
    (output_dir / "tracks.json").write_text(json.dumps(tracks_json, indent=2))
    (output_dir / "player_identity_map.json").write_text(json.dumps(identity_map, indent=2))
    (output_dir / "team_box_score.json").write_text(json.dumps(team_box_score, indent=2))
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    (output_dir / "stats_by_track.json").write_text(json.dumps(stats_by_track, indent=2))
    _stats_df = pd.DataFrame(stats)
    if "merged_track_ids" in _stats_df.columns:
        _stats_df = _stats_df.copy()
        _stats_df["merged_track_ids"] = _stats_df["merged_track_ids"].apply(
            lambda x: ",".join(map(str, x)) if isinstance(x, list) else x
        )
    _stats_df.to_csv(output_dir / "stats.csv", index=False)
    (output_dir / "pipeline.json").write_text(json.dumps(pipeline_meta, indent=2))
    (output_dir / "performance.json").write_text(json.dumps(perf, indent=2))
    (output_dir / "action_hints.json").write_text(json.dumps(action_hints_json, indent=2))
    (output_dir / "videomae_aux.json").write_text(json.dumps(videomae_aux_json, indent=2))
    # Additional presentation-friendly exports.
    team_rows: list[dict] = []
    for team in team_box_score:
        players = team.get("players", []) if isinstance(team, dict) else []
        for p in players:
            team_rows.append(
                {
                    "team_name": team.get("team_name"),
                    "player_number": p.get("player_number"),
                    "player_label": p.get("player_label"),
                    "minutes_on_court": p.get("minutes_on_court"),
                    "pts": p.get("pts"),
                    "reb": p.get("reb"),
                    "ast": p.get("ast"),
                    "stl": p.get("stl"),
                    "blk": p.get("blk"),
                    "tov": p.get("tov"),
                    "fgm": p.get("fgm"),
                    "fga": p.get("fga"),
                    "fg_pct": p.get("fg_pct"),
                }
            )
    if team_rows:
        pd.DataFrame(team_rows).to_csv(output_dir / "team_box_score_players.csv", index=False)

    if action_hints_json:
        hint_rows: list[dict] = []
        for h in action_hints_json:
            for label, prob in (h.get("siglip_action_scores") or {}).items():
                hint_rows.append(
                    {
                        "frame_index": h.get("frame_index"),
                        "action_prompt": label,
                        "probability": prob,
                    }
                )
        if hint_rows:
            pd.DataFrame(hint_rows).to_csv(output_dir / "action_hints_long.csv", index=False)

    perf_rows = [{"metric": "elapsed_s", "value": perf.get("elapsed_s")}, {"metric": "fps_effective", "value": perf.get("fps_effective")}]
    for k, v in (perf.get("stage_s") or {}).items():
        perf_rows.append({"metric": f"stage_{k}", "value": v})
    pd.DataFrame(perf_rows).to_csv(output_dir / "pipeline_performance.csv", index=False)
    _finalize_progress_json(output_dir, frame_idx, total_target)
    return result_payload
