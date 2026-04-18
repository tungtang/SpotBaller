from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

import cv2
import numpy as np

from app.pipeline.schemas import Track

try:
    import pytesseract
except Exception:  # pragma: no cover - optional runtime dependency
    pytesseract = None

if TYPE_CHECKING:
    from app.pipeline.pretrained_stack import PretrainedStack


class IdentityResolver:
    """
    Builds stable player identity from per-frame crops:
    - jersey number via TrOCR (optional) then Tesseract OCR
    - team assignment via SigLIP embedding clustering (optional) else jersey color k-means
    """

    def __init__(self, stack: PretrainedStack | None = None) -> None:
        self.stack = stack
        self.number_votes: defaultdict[int, list[str]] = defaultdict(list)
        self.color_samples: defaultdict[int, list[np.ndarray]] = defaultdict(list)
        self.embedding_samples: defaultdict[int, list[np.ndarray]] = defaultdict(list)

    def update(self, frame: np.ndarray, tracks: list[Track]) -> None:
        for tr in tracks:
            if tr.cls_name != "player":
                continue
            crop = self._safe_crop(frame, tr)
            if crop.size == 0:
                continue
            number, conf = None, 0.0
            if self.stack and self.stack.has_trocr:
                number, conf = self.stack.recognize_jersey_digits(crop)
            if number is None:
                number, conf = self._extract_number(crop)
            if number is not None and conf >= 0.3:
                self.number_votes[tr.track_id].append(number)
            if self.stack and self.stack.has_siglip and self.stack.config.use_siglip_teams:
                emb = self.stack.embed_jersey_region(crop)
                if emb is not None:
                    self.embedding_samples[tr.track_id].append(emb)
            jersey_color = self._dominant_jersey_color(crop)
            self.color_samples[tr.track_id].append(jersey_color)

    def finalize(self) -> dict[int, dict]:
        player_ids = sorted(set(self.number_votes.keys()) | set(self.color_samples.keys()))
        if not player_ids:
            return {}

        emb_centroids: dict[int, np.ndarray] = {}
        for pid in player_ids:
            es = self.embedding_samples.get(pid, [])
            if es:
                emb_centroids[pid] = np.mean(np.stack(es), axis=0).astype(np.float32)

        color_centroids: dict[int, np.ndarray] = {}
        for pid in player_ids:
            samples = self.color_samples.get(pid, [])
            if samples:
                color_centroids[pid] = np.mean(np.vstack(samples), axis=0)
            else:
                color_centroids[pid] = np.array([127.0, 127.0, 127.0], dtype=np.float32)

        if len(emb_centroids) >= 2:
            team_labels = self._cluster_two_teams(emb_centroids)
        else:
            team_labels = self._cluster_two_teams(color_centroids)
        identities: dict[int, dict] = {}

        for pid in player_ids:
            votes = self.number_votes.get(pid, [])
            if votes:
                number, count = Counter(votes).most_common(1)[0]
                number_conf = count / max(len(votes), 1)
            else:
                number = f"unknown_{pid}"
                number_conf = 0.0
            team_id = team_labels.get(pid, 0)
            identities[pid] = {
                "track_id": pid,
                "player_number": str(number),
                "player_label": f"#{number}",
                "player_number_confidence": round(float(number_conf), 3),
                "team_id": int(team_id),
                "team_name": f"Team {team_id + 1}",
                "team_confidence": 0.7,
            }
        return identities

    @staticmethod
    def _safe_crop(frame: np.ndarray, tr: Track) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(tr.x1)), max(0, int(tr.y1))
        x2, y2 = min(w, int(tr.x2)), min(h, int(tr.y2))
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _dominant_jersey_color(crop: np.ndarray) -> np.ndarray:
        h = crop.shape[0]
        upper = crop[: max(1, h // 2), :]
        pixels = upper.reshape(-1, 3).astype(np.float32)
        return np.mean(pixels, axis=0) if len(pixels) else np.array([127.0, 127.0, 127.0], dtype=np.float32)

    @staticmethod
    def _extract_number(crop: np.ndarray) -> tuple[str | None, float]:
        if pytesseract is None:
            return None, 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config = "--psm 7 -c tessedit_char_whitelist=0123456789"
        try:
            text = pytesseract.image_to_string(thresh, config=config)
        except Exception:
            return None, 0.0
        match = re.search(r"\d{1,2}", text or "")
        if not match:
            return None, 0.0
        return match.group(0), 0.6

    @staticmethod
    def _cluster_two_teams(centroids: dict[int, np.ndarray]) -> dict[int, int]:
        ids = list(centroids.keys())
        data = np.vstack([centroids[i] for i in ids]).astype(np.float32)
        if len(ids) == 1:
            return {ids[0]: 0}
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _compact, labels, _centers = cv2.kmeans(data, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        return {pid: int(labels[idx][0]) for idx, pid in enumerate(ids)}
