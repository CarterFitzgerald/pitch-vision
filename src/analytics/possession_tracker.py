import numpy as np
import supervision as sv
from collections import deque
from typing import Optional


class PossessionTracker:
    """
    Tracks ball possession per team based on proximity.
    The player with the ball closest to their feet (bottom-centre of bbox)
    within a pixel threshold is assigned possession for that frame.
    Maintains a rolling possession percentage over a configurable window.
    """

    def __init__(
        self,
        possession_radius_px: int = 80,
        smoothing_window: int = 150,   # frames — ~5 seconds at 30fps
    ):
        self.possession_radius_px = possession_radius_px
        self.smoothing_window = smoothing_window

        # Rolling window of possession: 0 = team A, 1 = team B, -1 = no possession
        self._history: deque = deque(maxlen=smoothing_window)
        self._total_frames = 0
        self._possession_frames = [0, 0]   # [team_a_frames, team_b_frames]
        self.current_possessing_team: int = -1
        self.current_possessing_player_id: Optional[int] = None

    def update(
        self,
        ball_position: Optional[tuple[float, float]],
        player_detections: sv.Detections,
        player_teams: np.ndarray,
    ) -> int:
        """
        Determine which team has possession this frame.

        Args:
            ball_position:    (cx, cy) of the ball, or None if not detected.
            player_detections: Tracked player detections with tracker_ids.
            player_teams:     Array of team indices (0 or 1) per detection.

        Returns:
            Team index (0 or 1) with possession, or -1 if no possession.
        """
        self._total_frames += 1

        if ball_position is None or len(player_detections) == 0:
            self._history.append(-1)
            self.current_possessing_team = -1
            self.current_possessing_player_id = None
            return -1

        ball_cx, ball_cy = ball_position

        # Use foot position = bottom-centre of bounding box
        best_dist = float("inf")
        best_team = -1
        best_player_id = None

        for i, bbox in enumerate(player_detections.xyxy):
            x1, y1, x2, y2 = bbox
            foot_x = (x1 + x2) / 2
            foot_y = y2   # bottom of box

            dist = np.sqrt((ball_cx - foot_x) ** 2 + (ball_cy - foot_y) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_team = int(player_teams[i])
                tid = player_detections.tracker_id
                best_player_id = int(tid[i]) if tid is not None else None

        if best_dist <= self.possession_radius_px:
            possession = best_team
            self.current_possessing_player_id = best_player_id
        else:
            possession = -1
            self.current_possessing_player_id = None

        self._history.append(possession)
        self.current_possessing_team = possession

        # Update running totals
        if possession in (0, 1):
            self._possession_frames[possession] += 1

        return possession

    def get_possession_percentage(self) -> tuple[float, float]:
        """
        Returns (team_a_pct, team_b_pct) based on the rolling window.
        Values sum to 100 (contested frames excluded from denominator).
        """
        history = list(self._history)
        team_a = history.count(0)
        team_b = history.count(1)
        total = team_a + team_b

        if total == 0:
            return (50.0, 50.0)

        return (team_a / total * 100, team_b / total * 100)

    def get_overall_possession(self) -> tuple[float, float]:
        """Overall possession percentages for the entire match so far."""
        total = sum(self._possession_frames)
        if total == 0:
            return (50.0, 50.0)
        return (
            self._possession_frames[0] / total * 100,
            self._possession_frames[1] / total * 100,
        )