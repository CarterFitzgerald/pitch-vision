import supervision as sv
import numpy as np
from trackers import ByteTrackTracker


class PlayerTracker:
    def __init__(self, fps: float = 25.0):
        self.fps = fps

        self.player_tracker = ByteTrackTracker()
        self.ball_tracker   = ByteTrackTracker()

    def update_players(self, detections: sv.Detections) -> sv.Detections:
        if len(detections) == 0:
            return detections
        return self.player_tracker.update(detections)

    def update_ball(self, detections: sv.Detections) -> sv.Detections:
        if len(detections) == 0:
            return detections
        return self.ball_tracker.update(detections)

    def reset(self):
        self.player_tracker.reset()
        self.ball_tracker.reset()