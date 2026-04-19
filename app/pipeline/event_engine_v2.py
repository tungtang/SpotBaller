from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from app.pipeline.schemas import Event, Track


@dataclass
class PossessionState:
    player_id: int | None = None
    start_frame: int = 0
    last_frame: int = 0


class StatsEventEngineV2:
    """
    v2 event engine with explicit possession state and expanded box score fields.
    This remains heuristic for single-camera input, but keeps stat schema stable.
    """

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        shot_cooldown_frames: int = 24,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = max(1, frame_height)
        self._shot_cooldown_frames = max(0, shot_cooldown_frames)
        self._last_shot_event_frame: int | None = None
        self.possession = PossessionState()
        self.frame_presence: defaultdict[int, int] = defaultdict(int)
        self.touches: defaultdict[int, int] = defaultdict(int)
        self.poss: defaultdict[int, int] = defaultdict(int)

        self.fgm: defaultdict[int, int] = defaultdict(int)
        self.fga: defaultdict[int, int] = defaultdict(int)
        self.three_pm: defaultdict[int, int] = defaultdict(int)
        self.three_pa: defaultdict[int, int] = defaultdict(int)
        self.ftm: defaultdict[int, int] = defaultdict(int)
        self.fta: defaultdict[int, int] = defaultdict(int)
        self.oreb: defaultdict[int, int] = defaultdict(int)
        self.dreb: defaultdict[int, int] = defaultdict(int)
        self.ast: defaultdict[int, int] = defaultdict(int)
        self.stl: defaultdict[int, int] = defaultdict(int)
        self.blk: defaultdict[int, int] = defaultdict(int)
        self.tov: defaultdict[int, int] = defaultdict(int)
        self.pf: defaultdict[int, int] = defaultdict(int)

        self.last_ball_y: float | None = None
        self.last_shooter: int | None = None
        self.last_passer: int | None = None

    def update(self, tracks: list[Track], frame_index: int, fps: float) -> list[Event]:
        _ = fps
        events: list[Event] = []
        players = [t for t in tracks if t.cls_name == "player"]
        balls = [t for t in tracks if t.cls_name == "ball"]
        rims = [t for t in tracks if t.cls_name == "rim"]

        for p in players:
            self.frame_presence[p.track_id] += 1

        if not players or not balls:
            return events

        ball = balls[0]
        bx = (ball.x1 + ball.x2) / 2
        by = (ball.y1 + ball.y2) / 2
        nearest = min(
            players,
            key=lambda p: (((p.x1 + p.x2) / 2 - bx) ** 2 + ((p.y1 + p.y2) / 2 - by) ** 2),
        )

        prev_holder = self.possession.player_id
        if prev_holder != nearest.track_id:
            if prev_holder is not None:
                self.tov[prev_holder] += 1
                self.stl[nearest.track_id] += 1
            self.possession.player_id = nearest.track_id
            self.possession.start_frame = frame_index
            self.touches[nearest.track_id] += 1
            self.poss[nearest.track_id] += 1
            self.last_passer = prev_holder
            events.append(Event(frame_index, "possession_change", nearest.track_id, 0.6))
        self.possession.last_frame = frame_index

        if rims:
            rim = min(
                rims,
                key=lambda r: ((r.x1 + r.x2) / 2 - bx) ** 2 + ((r.y1 + r.y2) / 2 - by) ** 2,
            )
            rx1, ry1, rx2, ry2 = rim.x1, rim.y1, rim.x2, rim.y2
            margin_y = max(12.0, 0.04 * float(self.frame_height))
            in_rim_zone = rx1 <= bx <= rx2 and (ry1 - margin_y) <= by <= (ry2 + margin_y)
            descending = self.last_ball_y is not None and by > self.last_ball_y

            if in_rim_zone and self.possession.player_id is not None:
                cooldown_ok = (
                    self._last_shot_event_frame is None
                    or (frame_index - self._last_shot_event_frame) >= self._shot_cooldown_frames
                )
                if cooldown_ok:
                    shooter = self.possession.player_id
                    self.last_shooter = shooter
                    self.fga[shooter] += 1
                    events.append(Event(frame_index, "shot_attempt", shooter, 0.65))

                    is_three = self._is_three_point_attempt(nearest)
                    if is_three:
                        self.three_pa[shooter] += 1

                    if descending:
                        self.fgm[shooter] += 1
                        if is_three:
                            self.three_pm[shooter] += 1
                        if self.last_passer is not None and self.last_passer != shooter:
                            self.ast[self.last_passer] += 1
                        events.append(Event(frame_index, "shot_made", shooter, 0.55))
                    else:
                        self.dreb[nearest.track_id] += 1
                        if shooter != nearest.track_id:
                            self.blk[nearest.track_id] += 1
                        events.append(Event(frame_index, "shot_missed", shooter, 0.45))
                    self._last_shot_event_frame = frame_index

        self.last_ball_y = by
        return events

    def compute_stats(self, fps: float) -> list[dict]:
        player_ids = (
            set(self.frame_presence)
            | set(self.touches)
            | set(self.fga)
            | set(self.fgm)
            | set(self.three_pa)
            | set(self.three_pm)
        )
        rows: list[dict] = []
        for pid in sorted(player_ids):
            fga = self.fga[pid]
            fgm = self.fgm[pid]
            three_pa = self.three_pa[pid]
            three_pm = self.three_pm[pid]
            fta = self.fta[pid]
            ftm = self.ftm[pid]
            oreb = self.oreb[pid]
            dreb = self.dreb[pid]
            reb = oreb + dreb
            ast = self.ast[pid]
            stl = self.stl[pid]
            blk = self.blk[pid]
            tov = self.tov[pid]
            pf = self.pf[pid]
            pts = 2 * (fgm - three_pm) + 3 * three_pm + ftm
            eff = pts + reb + ast + stl + blk - (fga - fgm) - (fta - ftm) - tov
            minutes = self.frame_presence[pid] / fps / 60.0 if fps > 0 else 0.0
            rows.append(
                {
                    "player_id": pid,
                    "minutes_on_court": round(minutes, 2),
                    "touches": self.touches[pid],
                    "poss": self.poss[pid],
                    "fga": fga,
                    "fgm": fgm,
                    "fg_pct": round((fgm / fga) * 100, 1) if fga else 0.0,
                    "three_pa": three_pa,
                    "three_pm": three_pm,
                    "three_pct": round((three_pm / three_pa) * 100, 1) if three_pa else 0.0,
                    "fta": fta,
                    "ftm": ftm,
                    "ft_pct": round((ftm / fta) * 100, 1) if fta else 0.0,
                    "oreb": oreb,
                    "dreb": dreb,
                    "reb": reb,
                    "ast": ast,
                    "stl": stl,
                    "blk": blk,
                    "tov": tov,
                    "pf": pf,
                    "pts": pts,
                    "efficiency": eff,
                }
            )
        return rows

    def _is_three_point_attempt(self, player_track: Track) -> bool:
        sx = (player_track.x1 + player_track.x2) / 2
        return sx < 0.32 * self.frame_width or sx > 0.68 * self.frame_width
