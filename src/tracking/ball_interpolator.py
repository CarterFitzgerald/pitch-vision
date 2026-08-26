"""
BallInterpolator — causal ball position tracking with velocity extrapolation.

Key fix over original:
    The original tried to interpolate using future frames not yet processed,
    so next_data was always None and the last detected position was silently
    replayed — causing false possession assignments.

This version uses:
    - Velocity-based extrapolation from past frames only
    - Confidence decay so possession tracker ignores stale positions
    - Hard None after max_gap frames so possession becomes contested
"""

import numpy as np
import supervision as sv
from collections import deque
from typing import Optional


class BallInterpolator:
    def __init__(
        self,
        buffer_size:     int = 60,
        max_gap:         int = 8,
        smooth_window:   int = 3,
        velocity_frames: int = 4,
    ):
        self.buffer_size     = buffer_size
        self.max_gap         = max_gap
        self.smooth_window   = smooth_window
        self.velocity_frames = velocity_frames

        self._history: deque        = deque(maxlen=buffer_size)
        self._smooth_buffer: deque  = deque(maxlen=smooth_window)
        self._frames_since_detection = 0
        self._last_real_frame: int   = -1
        self._current_frame: int     = 0

    def update(
        self,
        frame_idx: int,
        detections: sv.Detections,
    ) -> Optional[tuple[float, float]]:
        self._current_frame = frame_idx
        if len(detections) > 0:
            return self._handle_detection(frame_idx, detections)
        return self._handle_missing(frame_idx)

    def _handle_detection(self, frame_idx, detections):
        best = int(np.argmax(detections.confidence))
        x1, y1, x2, y2 = detections.xyxy[best]
        cx   = float((x1 + x2) / 2)
        cy   = float((y1 + y2) / 2)
        conf = float(detections.confidence[best])
        self._history.append((frame_idx, cx, cy, conf, True))
        self._smooth_buffer.append((cx, cy))
        self._frames_since_detection = 0
        self._last_real_frame        = frame_idx
        return self._smoothed(cx, cy)

    def _handle_missing(self, frame_idx):
        self._frames_since_detection += 1
        if self._frames_since_detection > self.max_gap:
            return None
        pos = self._extrapolate(frame_idx)
        if pos is None:
            return None
        cx, cy = pos
        conf = max(0.0, 1.0 - self._frames_since_detection / self.max_gap)
        self._history.append((frame_idx, cx, cy, conf, False))
        return (cx, cy)

    def _smoothed(self, raw_cx, raw_cy):
        if len(self._smooth_buffer) < 2:
            return (raw_cx, raw_cy)
        arr = np.array(list(self._smooth_buffer))
        return (float(arr[:, 0].mean()), float(arr[:, 1].mean()))

    def _extrapolate(self, frame_idx):
        real = [(f, cx, cy) for f, cx, cy, _, is_real in self._history if is_real]
        if not real:
            return None
        if len(real) < 2:
            return (real[-1][1], real[-1][2])
        recent = real[-self.velocity_frames:]
        velocities, weights = [], []
        for i in range(1, len(recent)):
            f0, x0, y0 = recent[i-1]
            f1, x1, y1 = recent[i]
            dt = f1 - f0
            if dt == 0:
                continue
            velocities.append(((x1-x0)/dt, (y1-y0)/dt))
            weights.append(float(i))
        if not velocities:
            return (real[-1][1], real[-1][2])
        w   = np.array(weights) / sum(weights)
        vx  = float(np.dot(w, [v[0] for v in velocities]))
        vy  = float(np.dot(w, [v[1] for v in velocities]))
        lf, lx, ly = real[-1]
        dt  = frame_idx - lf
        d   = 0.85 ** dt
        return (lx + vx * dt * d, ly + vy * dt * d)

    def get_trail(self, n: int = 30) -> list:
        recent = list(self._history)[-n:]
        return [(cx, cy, is_real) for _, cx, cy, _, is_real in recent]

    def get_smoothed_position(self):
        if not self._smooth_buffer:
            return None
        arr = np.array(list(self._smooth_buffer))
        return (float(arr[:, 0].mean()), float(arr[:, 1].mean()))

    @property
    def frames_since_detection(self):
        return self._frames_since_detection

    @property
    def ball_is_confident(self):
        return self._frames_since_detection <= self.max_gap // 2

    def cleanup_old_frames(self, keep_last: int = 60):
        pass  # managed by deque maxlen