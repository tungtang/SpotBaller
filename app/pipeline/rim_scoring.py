"""
Rim / hoop handling for the analytics pipeline.

ByteTrack is only run for ``player`` and ``ball``; rim boxes must be merged into the
per-frame ``Track`` list so :class:`StatsEventEngineV2` can evaluate shot geometry.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.pipeline.schemas import Detection, Track

# Stable synthetic id for a single rim track (not a MOT player id).
RIM_TRACK_ID = 900_001


@dataclass(frozen=True)
class RimFallbackConfig:
    """When the detector never sees a hoop, use a weak prior scoring zone (broadcast-style)."""

    enabled: bool = True
    x1_frac: float = 0.38
    x2_frac: float = 0.62
    y1_frac: float = 0.02
    y2_frac: float = 0.22


def best_rim_detection(detections: list[Detection]) -> Detection | None:
    rims = [d for d in detections if d.cls_name == "rim"]
    if not rims:
        return None
    return max(rims, key=lambda d: d.conf)


def synthetic_rim_detection(width: int, height: int, cfg: RimFallbackConfig) -> Detection:
    x1 = cfg.x1_frac * width
    x2 = cfg.x2_frac * width
    y1 = cfg.y1_frac * height
    y2 = cfg.y2_frac * height
    return Detection("rim", 0.25, x1, y1, x2, y2)


def append_rim_track(
    tracks: list[Track],
    *,
    frame_index: int,
    detections: list[Detection],
    frame_width: int,
    frame_height: int,
    fallback_cfg: RimFallbackConfig,
) -> tuple[list[Track], str]:
    """
    Append exactly one rim ``Track`` when possible.

    Returns (possibly extended tracks, rim_source one of ``detected`` | ``fallback`` | ``none``).
    """
    if any(t.cls_name == "rim" for t in tracks):
        return tracks, "tracker"

    rim_det = best_rim_detection(detections)
    source = "none"
    if rim_det is not None:
        det = rim_det
        source = "detected"
    elif fallback_cfg.enabled:
        det = synthetic_rim_detection(frame_width, frame_height, fallback_cfg)
        source = "fallback"
    else:
        return tracks, source

    tracks = list(tracks)
    tracks.append(
        Track(
            track_id=RIM_TRACK_ID,
            cls_name="rim",
            frame_index=frame_index,
            x1=float(det.x1),
            y1=float(det.y1),
            x2=float(det.x2),
            y2=float(det.y2),
        )
    )
    return tracks, source
