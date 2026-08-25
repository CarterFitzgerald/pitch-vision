"""
TeamClassifier — jersey colour clustering with temporal confidence smoothing.

Improvements over original:
    - Per-player confidence scores that accumulate over time
    - Exponential moving average to prevent frame-to-frame flicker
    - Minimum confidence threshold before committing to a team assignment
    - Torso-only crop with grass pixel filtering
    - Handles re-identification after occlusion gracefully
"""

from __future__ import annotations
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from typing import Optional
import supervision as sv

GRASS_LOWER = np.array([35, 40, 40],  dtype=np.uint8)
GRASS_UPPER = np.array([85, 255, 255], dtype=np.uint8)

TORSO_TOP_FRAC    = 0.25
TORSO_BOTTOM_FRAC = 0.65

EMA_ALPHA       = 0.3
MIN_CONF_TO_USE = 0.55
MIN_SAMPLES     = 3


class PlayerTeamState:
    def __init__(self):
        self.team:       int   = -1
        self.confidence: float = 0.0
        self.samples:    int   = 0
        self._scores: np.ndarray = np.array([0.5, 0.5])

    def update(self, raw_team: int, raw_confidence: float):
        self.samples += 1
        obs = np.zeros(2)
        obs[raw_team] = raw_confidence
        obs[1 - raw_team] = 1.0 - raw_confidence
        self._scores = EMA_ALPHA * obs + (1 - EMA_ALPHA) * self._scores
        predicted_team = int(np.argmax(self._scores))
        predicted_conf = float(self._scores[predicted_team])
        if self.samples >= MIN_SAMPLES and predicted_conf >= MIN_CONF_TO_USE:
            self.team       = predicted_team
            self.confidence = predicted_conf
        elif self.samples < MIN_SAMPLES:
            self.team       = -1
            self.confidence = predicted_conf

    @property
    def is_committed(self) -> bool:
        return self.team != -1 and self.confidence >= MIN_CONF_TO_USE

    def __repr__(self):
        return f"PlayerTeamState(team={self.team}, conf={self.confidence:.2f}, samples={self.samples})"


class TeamClassifier:
    def __init__(self, n_clusters: int = 2):
        self.n_clusters  = n_clusters
        self.kmeans:       Optional[KMeans]     = None
        self.team_colours: Optional[np.ndarray] = None
        self._player_states: dict[int, PlayerTeamState] = defaultdict(PlayerTeamState)
        self._fitted = False

    def _extract_jersey_colour(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1
        if h < 20 or w < 10:
            return None
        cy1  = y1 + int(h * TORSO_TOP_FRAC)
        cy2  = y1 + int(h * TORSO_BOTTOM_FRAC)
        crop = frame[cy1:cy2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return None
        hsv    = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask   = cv2.inRange(hsv, GRASS_LOWER, GRASS_UPPER)
        inv    = cv2.bitwise_not(mask)
        pixels = crop[inv > 0]
        if len(pixels) < 15:
            pixels = crop.reshape(-1, 3)
        return pixels.mean(axis=0).astype(np.float32)

    def _colour_confidence(self, colour: np.ndarray) -> tuple[int, float]:
        if self.kmeans is None or self.team_colours is None:
            return 0, 0.5
        dists = np.linalg.norm(self.team_colours - colour[np.newaxis, :], axis=1)
        team  = int(np.argmin(dists))
        d0, d1 = dists[0], dists[1]
        total  = d0 + d1
        if total < 1e-6:
            return team, 0.5
        confidence = float(max(d0, d1) / total)
        return team, float(np.clip(confidence, 0.5, 1.0))

    def fit(self, frame: np.ndarray, detections: sv.Detections) -> bool:
        if len(detections) < 6:
            return False
        colours = [c for bbox in detections.xyxy
                   if (c := self._extract_jersey_colour(frame, bbox)) is not None]
        if len(colours) < 6:
            return False
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=15, max_iter=300)
        self.kmeans.fit(np.array(colours))
        self.team_colours = self.kmeans.cluster_centers_.astype(np.float32)
        self._fitted = True
        return True

    def refit(self, frame: np.ndarray, detections: sv.Detections) -> bool:
        result = self.fit(frame, detections)
        if result:
            self._player_states.clear()
        return result

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def predict(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        if not self._fitted:
            return np.zeros(len(detections), dtype=int)
        result      = np.zeros(len(detections), dtype=int)
        tracker_ids = detections.tracker_id
        for i, bbox in enumerate(detections.xyxy):
            tid    = int(tracker_ids[i]) if tracker_ids is not None else -1
            colour = self._extract_jersey_colour(frame, bbox)
            if colour is None:
                if tid != -1 and self._player_states[tid].is_committed:
                    result[i] = self._player_states[tid].team
                continue
            raw_team, raw_conf = self._colour_confidence(colour)
            if tid != -1:
                state = self._player_states[tid]
                state.update(raw_team, raw_conf)
                result[i] = state.team if state.team != -1 else raw_team
            else:
                result[i] = raw_team
        return result

    def get_team_colour_bgr(self, team_idx: int) -> tuple[int, int, int]:
        if self.team_colours is None:
            return (200, 200, 200)
        c = self.team_colours[team_idx]
        return (int(c[0]), int(c[1]), int(c[2]))

    def get_player_confidence(self, tracker_id: int) -> float:
        return self._player_states[tracker_id].confidence

    def get_uncertain_players(self, threshold: float = 0.65) -> list[int]:
        return [tid for tid, s in self._player_states.items()
                if s.samples > 0 and s.confidence < threshold]

    def debug_summary(self) -> str:
        lines = ["TeamClassifier state:"]
        for tid, s in sorted(self._player_states.items()):
            lines.append(f"  #{tid:<4} team={s.team}  conf={s.confidence:.2f}  samples={s.samples}")
        return "\n".join(lines)