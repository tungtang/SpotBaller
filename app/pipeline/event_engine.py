from __future__ import annotations

from collections import defaultdict

from app.pipeline.schemas import Event, Track


class EventEngine:
    def __init__(self) -> None:
        self.player_last_touch: int | None = None
        self.player_touches: defaultdict[int, int] = defaultdict(int)
        self.possessions: defaultdict[int, int] = defaultdict(int)
        self.fga: defaultdict[int, int] = defaultdict(int)
        self.fgm: defaultdict[int, int] = defaultdict(int)
        self.three_pa: defaultdict[int, int] = defaultdict(int)
        self.three_pm: defaultdict[int, int] = defaultdict(int)
        self.fta: defaultdict[int, int] = defaultdict(int)
        self.ftm: defaultdict[int, int] = defaultdict(int)
        self.oreb: defaultdict[int, int] = defaultdict(int)
        self.dreb: defaultdict[int, int] = defaultdict(int)
        self.ast: defaultdict[int, int] = defaultdict(int)
        self.stl: defaultdict[int, int] = defaultdict(int)
        self.blk: defaultdict[int, int] = defaultdict(int)
        self.tov: defaultdict[int, int] = defaultdict(int)
        self.pf: defaultdict[int, int] = defaultdict(int)
        self.frame_presence: defaultdict[int, int] = defaultdict(int)
        self.last_ball_y: float | None = None
        self.last_possession_player: int | None = None

    def update(self, tracks: list[Track], frame_index: int, fps: float) -> list[Event]:
        events: list[Event] = []
        players = [t for t in tracks if t.cls_name == "player"]
        balls = [t for t in tracks if t.cls_name == "ball"]
        rims = [t for t in tracks if t.cls_name == "rim"]

        for p in players:
            self.frame_presence[p.track_id] += 1

        if balls and players:
            ball = balls[0]
            bx = (ball.x1 + ball.x2) / 2
            by = (ball.y1 + ball.y2) / 2
            nearest = min(
                players,
                key=lambda p: (((p.x1 + p.x2) / 2 - bx) ** 2 + ((p.y1 + p.y2) / 2 - by) ** 2),
            )
            if self.player_last_touch != nearest.track_id:
                self.player_last_touch = nearest.track_id
                self.player_touches[nearest.track_id] += 1
                self.possessions[nearest.track_id] += 1
                if self.last_possession_player is not None and self.last_possession_player != nearest.track_id:
                    self.stl[nearest.track_id] += 1
                    self.tov[self.last_possession_player] += 1
                self.last_possession_player = nearest.track_id
                events.append(Event(frame_index, "possession_change", nearest.track_id, 0.6))

            if rims:
                rim = rims[0]
                rx1, ry1, rx2, ry2 = rim.x1, rim.y1, rim.x2, rim.y2
                in_rim_zone = rx1 <= bx <= rx2 and (ry1 - 25) <= by <= (ry2 + 25)
                descending = self.last_ball_y is not None and by > self.last_ball_y
                if in_rim_zone:
                    shooter = self.player_last_touch
                    if shooter is not None:
                        self.fga[shooter] += 1
                        events.append(Event(frame_index, "shot_attempt", shooter, 0.65))
                        # Approximate 3PA from court x-position heuristic.
                        sx = ((nearest.x1 + nearest.x2) / 2.0) if nearest is not None else 0.0
                        if sx < 0.32 * 1280 or sx > 0.68 * 1280:
                            self.three_pa[shooter] += 1
                        if descending:
                            self.fgm[shooter] += 1
                            if self.three_pa[shooter] > self.three_pm[shooter]:
                                self.three_pm[shooter] += 1
                            if self.last_possession_player is not None and self.last_possession_player != shooter:
                                self.ast[self.last_possession_player] += 1
                            events.append(Event(frame_index, "shot_made", shooter, 0.55))
                        else:
                            self.dreb[nearest.track_id] += 1
                            events.append(Event(frame_index, "shot_missed", shooter, 0.45))
            self.last_ball_y = by

        return events

    def compute_stats(self, fps: float) -> list[dict]:
        player_ids = (
            set(self.frame_presence)
            | set(self.player_touches)
            | set(self.fga)
            | set(self.fgm)
            | set(self.three_pa)
            | set(self.three_pm)
        )
        rows = []
        for pid in sorted(player_ids):
            minutes = self.frame_presence[pid] / fps / 60.0 if fps > 0 else 0.0
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
            rows.append(
                {
                    "player_id": pid,
                    "minutes_on_court": round(minutes, 2),
                    "touches": self.player_touches[pid],
                    "poss": self.possessions[pid],
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
