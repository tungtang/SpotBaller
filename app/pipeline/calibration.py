from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CourtCalibration:
    """Maps image pixels to normalized court coordinates (0..1)."""

    frame_width: int
    frame_height: int

    def to_court(self, x: float, y: float) -> tuple[float, float]:
        if self.frame_width <= 0 or self.frame_height <= 0:
            return (0.0, 0.0)
        return (max(0.0, min(1.0, x / self.frame_width)), max(0.0, min(1.0, y / self.frame_height)))
