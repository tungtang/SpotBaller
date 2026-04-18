from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import supervision as sv
from app.pipeline.schemas import Detection, Track


class CentroidTracker:
    """Simple nearest-centroid tracker to keep project runnable without external MOT setup."""

    def __init__(self, max_distance: float = 50.0) -> None:
        self.max_distance = max_distance
        self.next_id = 1
        self.last_centroids: dict[int, tuple[float, float]] = {}
        self.last_class: dict[int, str] = {}

    @staticmethod
    def _centroid(det: Detection) -> tuple[float, float]:
        return ((det.x1 + det.x2) / 2.0, (det.y1 + det.y2) / 2.0)

    def update(self, detections: list[Detection], frame_index: int) -> list[Track]:
        assigned: set[int] = set()
        tracks: list[Track] = []

        by_class = defaultdict(list)
        for det in detections:
            by_class[det.cls_name].append(det)

        for cls_name, cls_dets in by_class.items():
            for det in cls_dets:
                cx, cy = self._centroid(det)
                best_id = None
                best_dist = float("inf")
                for tid, (px, py) in self.last_centroids.items():
                    if tid in assigned or self.last_class.get(tid) != cls_name:
                        continue
                    dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                    if dist < best_dist and dist <= self.max_distance:
                        best_dist = dist
                        best_id = tid
                if best_id is None:
                    best_id = self.next_id
                    self.next_id += 1
                assigned.add(best_id)
                self.last_centroids[best_id] = (cx, cy)
                self.last_class[best_id] = cls_name
                tracks.append(
                    Track(
                        track_id=best_id,
                        cls_name=cls_name,
                        frame_index=frame_index,
                        x1=det.x1,
                        y1=det.y1,
                        x2=det.x2,
                        y2=det.y2,
                    )
                )
        return tracks


class ByteTrackTracker:
    """ByteTrack-backed tracker for player and ball classes."""

    def __init__(self) -> None:
        self._trackers: dict[str, Any] = {
            "player": sv.ByteTrack(),
            "ball": sv.ByteTrack(),
        }

    def update(self, detections: list[Detection], frame_index: int) -> list[Track]:
        tracks: list[Track] = []
        for cls_name, tracker in self._trackers.items():
            cls_dets = [d for d in detections if d.cls_name == cls_name]
            if not cls_dets:
                continue
            xyxy = np.array([[d.x1, d.y1, d.x2, d.y2] for d in cls_dets], dtype=np.float32)
            conf = np.array([d.conf for d in cls_dets], dtype=np.float32)
            class_ids = np.zeros(len(cls_dets), dtype=int)
            sv_dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_ids)
            tracked = tracker.update_with_detections(sv_dets)
            tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else []
            for idx, tid in enumerate(tracker_ids):
                if tid is None:
                    continue
                x1, y1, x2, y2 = tracked.xyxy[idx].tolist()
                tracks.append(
                    Track(
                        track_id=int(tid),
                        cls_name=cls_name,
                        frame_index=frame_index,
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                    )
                )
        return tracks
