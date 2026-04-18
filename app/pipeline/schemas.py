from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    cls_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Track:
    track_id: int
    cls_name: str
    frame_index: int
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Event:
    frame_index: int
    event_type: str
    player_id: int | None
    confidence: float
