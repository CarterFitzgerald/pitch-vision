import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import deque
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tracking.tracker import PlayerTracker
from src.tracking.ball_interpolator import BallInterpolator
from src.analytics.team_classifier import TeamClassifier
from src.analytics.possession_tracker import PossessionTracker

# ─── Config ───────────────────────────────────────────────────────────────────

PLAYERS_MODEL_PATH = "runs/detect/runs/players/yolo26x/weights/best.pt"
BALL_MODEL_PATH    = "runs/detect/runs/ball/yolo26x_v3/weights/best.pt"  # updated model
VIDEO_PATH         = "data/video/match.mp4"
OUTPUT_PATH        = "output/dashboard.mp4"

PLAYERS_CONF = 0.3
BALL_CONF    = 0.08    # FIX: was 0.15 — lower catches more real detections
BALL_IOU     = 0.3     # NMS threshold for ball detections

CLASS_BALL       = 0
CLASS_GOALKEEPER = 1
CLASS_PLAYER     = 2
CLASS_REFEREE    = 3

VIDEO_WIDTH      = 960
VIDEO_HEIGHT     = 540
PANEL_WIDTH      = 400
PANEL_HEIGHT     = VIDEO_HEIGHT
DASHBOARD_WIDTH  = VIDEO_WIDTH + PANEL_WIDTH
DASHBOARD_HEIGHT = VIDEO_HEIGHT

TEAM_COLOURS_BGR  = [(0, 140, 255), (255, 80, 0)]
TEAM_NAMES        = ["Team A", "Team B"]
BALL_TRAIL_LENGTH = 40   # slightly longer trail
BALL_MAP_FADE     = True

EDGE_MARGIN = 30   # pixels — reject ball detections this close to frame edge

# ─── Ball detection helpers ───────────────────────────────────────────────────

def filter_edge_detections(
    detections: sv.Detections,
    frame_w: int,
    frame_h: int,
    margin: int = EDGE_MARGIN,
) -> sv.Detections:
    """Reject ball detections within margin pixels of the frame edge."""
    if len(detections) == 0:
        return detections
    x1 = detections.xyxy[:, 0]
    y1 = detections.xyxy[:, 1]
    x2 = detections.xyxy[:, 2]
    y2 = detections.xyxy[:, 3]
    valid = (x1 > margin) & (y1 > margin) & \
            (x2 < frame_w - margin) & (y2 < frame_h - margin)
    return detections[valid]


def merge_ball_detections(
    ball_from_players: sv.Detections,
    ball_from_dedicated: sv.Detections,
) -> sv.Detections:
    """
    Ensemble both ball models using supervision's built-in NMS.
    Prefer dedicated model, fall back to players model.
    """
    if len(ball_from_players) == 0 and len(ball_from_dedicated) == 0:
        return sv.Detections.empty()

    # Both found something — merge and apply NMS
    if len(ball_from_players) > 0 and len(ball_from_dedicated) > 0:
        merged = sv.Detections.merge([ball_from_players, ball_from_dedicated])
        return merged.with_nms(threshold=0.5, class_agnostic=True)

    # Only one found something — prefer dedicated model
    return ball_from_dedicated if len(ball_from_dedicated) > 0 else ball_from_players


# ─── Drawing helpers ──────────────────────────────────────────────────────────

def make_panel(height: int, width: int) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (20, 20, 28)
    return panel


def draw_text(img, text, pos, scale=0.55, colour=(220, 220, 220), thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, colour, thickness, cv2.LINE_AA)


def draw_possession_bar(panel, pct_a, pct_b, y, colour_a, colour_b, label):
    bx, bw, bh = 20, PANEL_WIDTH - 40, 28
    draw_text(panel, label, (bx, y - 10), scale=0.45, colour=(160, 160, 160))
    a_w = max(1, int(bw * pct_a / 100))
    cv2.rectangle(panel, (bx, y), (bx + a_w, y + bh), colour_a, -1)
    cv2.rectangle(panel, (bx + a_w, y), (bx + bw, y + bh), colour_b, -1)
    cv2.rectangle(panel, (bx, y), (bx + bw, y + bh), (80, 80, 80), 1)
    draw_text(panel, f"{pct_a:.1f}%", (bx + 5, y + 20),
              scale=0.5, colour=(255, 255, 255), thickness=1)
    if bx + a_w + 5 + 45 < bx + bw:
        draw_text(panel, f"{pct_b:.1f}%", (bx + a_w + 5, y + 20),
                  scale=0.5, colour=(255, 255, 255), thickness=1)


def draw_ball_trail_map(panel, trail, y_offset, map_h):
    """
    Draw ball trail on mini pitch.
    Trail entries are (cx, cy, is_real) — real detections solid, extrapolated hollow.
    """
    mx0, mw = 20, PANEL_WIDTH - 40
    cv2.rectangle(panel, (mx0, y_offset),
                  (mx0 + mw, y_offset + map_h), (30, 90, 30), -1)
    cv2.rectangle(panel, (mx0, y_offset),
                  (mx0 + mw, y_offset + map_h), (60, 160, 60), 1)

    # Pitch markings
    cxm = mx0 + mw // 2
    cym = y_offset + map_h // 2
    cv2.circle(panel, (cxm, cym), map_h // 5, (60, 160, 60), 1)
    cv2.circle(panel, (cxm, cym), 2, (60, 160, 60), -1)
    cv2.line(panel, (cxm, y_offset), (cxm, y_offset + map_h), (60, 160, 60), 1)
    # Penalty areas (approximate)
    pa_w = int(mw * 0.15)
    pa_h = int(map_h * 0.55)
    pa_y = y_offset + (map_h - pa_h) // 2
    cv2.rectangle(panel, (mx0, pa_y), (mx0 + pa_w, pa_y + pa_h), (50, 130, 50), 1)
    cv2.rectangle(panel, (mx0 + mw - pa_w, pa_y),
                  (mx0 + mw, pa_y + pa_h), (50, 130, 50), 1)

    if not trail:
        return

    for i, entry in enumerate(trail):
        if len(entry) == 3:
            tx, ty, is_real = entry
        else:
            tx, ty = entry
            is_real = True

        px = int(np.clip(mx0 + (tx / VIDEO_WIDTH) * mw, mx0 + 1, mx0 + mw - 1))
        py = int(np.clip(y_offset + (ty / VIDEO_HEIGHT) * map_h,
                         y_offset + 1, y_offset + map_h - 1))

        alpha      = (i + 1) / max(len(trail), 1) if BALL_MAP_FADE else 1.0
        r          = max(2, int(4 * alpha))
        brightness = int(255 * alpha)

        if is_real:
            cv2.circle(panel, (px, py), r, (brightness, brightness, brightness), -1)
        else:
            dim = int(brightness * 0.45)
            cv2.circle(panel, (px, py), max(1, r - 1), (dim, dim, dim), 1)


def draw_stat_row(panel, label, value, y, colour=(200, 200, 200)):
    draw_text(panel, label, (20, y), scale=0.45, colour=(130, 130, 130))
    draw_text(panel, str(value), (220, y), scale=0.45, colour=colour)


def build_right_panel(
    frame_idx, fps, possession_tracker, ball_trail,
    player_count, team_a_count, team_b_count,
    team_colour_a, team_colour_b,
    ball_confident, frames_since_ball,
) -> np.ndarray:
    panel = make_panel(PANEL_HEIGHT, PANEL_WIDTH)

    # Header
    draw_text(panel, "PITCH VISION", (20, 35),
              scale=0.7, colour=(255, 255, 255), thickness=2)
    ts = f"{int(frame_idx/fps//60):02d}:{int(frame_idx/fps%60):02d}"
    draw_text(panel, f"Frame {frame_idx}  |  {ts}",
              (20, 58), scale=0.4, colour=(120, 120, 120))

    # Ball status indicator
    if ball_confident:
        ball_col = (0, 220, 80)
        ball_lbl = "Ball: DETECTED"
    elif frames_since_ball <= 8:
        ball_col = (0, 165, 255)
        ball_lbl = f"Ball: EST ({frames_since_ball}f)"
    else:
        ball_col = (60, 60, 180)
        ball_lbl = "Ball: LOST"
    draw_text(panel, ball_lbl, (230, 58), scale=0.36, colour=ball_col)

    cv2.line(panel, (20, 68), (PANEL_WIDTH - 20, 68), (50, 50, 60), 1)

    # Possession bars
    pct_a, pct_b = possession_tracker.get_possession_percentage()
    draw_possession_bar(panel, pct_a, pct_b, 85,
                        team_colour_a, team_colour_b, "Possession (last 12 sec)")

    ov_a, ov_b = possession_tracker.get_overall_possession()
    draw_possession_bar(panel, ov_a, ov_b, 140,
                        team_colour_a, team_colour_b, "Possession (overall)")

    cv2.line(panel, (20, 185), (PANEL_WIDTH - 20, 185), (50, 50, 60), 1)

    # Frame stats
    draw_text(panel, "FRAME STATS", (20, 205), scale=0.45, colour=(160, 160, 160))
    draw_stat_row(panel, "Players detected", str(player_count), 228)
    draw_stat_row(panel, f"{TEAM_NAMES[0]} players", str(team_a_count), 248,
                  colour=tuple(int(c) for c in team_colour_a[::-1]))
    draw_stat_row(panel, f"{TEAM_NAMES[1]} players", str(team_b_count), 268,
                  colour=tuple(int(c) for c in team_colour_b[::-1]))

    possessing  = possession_tracker.current_possessing_team
    poss_text   = TEAM_NAMES[possessing] if possessing in (0, 1) else "Contested"
    poss_colour = (team_colour_a if possessing == 0
                   else (team_colour_b if possessing == 1 else (150, 150, 150)))
    draw_stat_row(panel, "Ball with", poss_text, 290, colour=poss_colour)

    pid = possession_tracker.current_possessing_player_id
    draw_stat_row(panel, "Player ID", f"#{pid}" if pid is not None else "—", 310)

    cv2.line(panel, (20, 325), (PANEL_WIDTH - 20, 325), (50, 50, 60), 1)

    # Ball trail map
    draw_text(panel, "BALL TRAIL MAP", (20, 345),
              scale=0.45, colour=(160, 160, 160))
    draw_text(panel, "● real  ○ estimated", (195, 345),
              scale=0.32, colour=(90, 90, 90))
    draw_ball_trail_map(panel, ball_trail, y_offset=358, map_h=160)

    return panel


# ─── Annotators ───────────────────────────────────────────────────────────────

def make_annotators():
    return (
        sv.TriangleAnnotator(
            color=sv.Color.WHITE, base=20, height=18,
            position=sv.Position.TOP_CENTER,
        ),
        sv.LabelAnnotator(
            text_scale=0.35, text_thickness=1, text_padding=3,
            text_color=sv.Color.WHITE,
            text_position=sv.Position.BOTTOM_CENTER,
        ),
    )


def annotate_players(frame, player_dets, player_teams, gk_dets, ref_dets,
                     team_colour_a, team_colour_b, label_annotator):
    out = frame.copy()

    for team_idx, colour in enumerate([team_colour_a, team_colour_b]):
        mask = player_teams == team_idx
        if not any(mask):
            continue
        sv_col = sv.Color(r=colour[2], g=colour[1], b=colour[0])
        out    = sv.EllipseAnnotator(color=sv_col, thickness=2).annotate(
                    out, player_dets[mask])

    if len(gk_dets) > 0:
        out = sv.EllipseAnnotator(
            color=sv.Color.GREEN, thickness=2).annotate(out, gk_dets)

    if len(ref_dets) > 0:
        out = sv.EllipseAnnotator(
            color=sv.Color.from_hex("#FFFF00"), thickness=2).annotate(out, ref_dets)

    all_dets = sv.Detections.merge([player_dets, gk_dets, ref_dets])
    if len(all_dets) > 0 and all_dets.tracker_id is not None:
        labels = [f"#{tid}" for tid in all_dets.tracker_id]
        out    = label_annotator.annotate(out, all_dets, labels=labels)

    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading models...")
    for path, name in [
        (PLAYERS_MODEL_PATH, "Players"),
        (BALL_MODEL_PATH,    "Ball"),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {name} model not found at {path}")
            sys.exit(1)

    players_model = YOLO(PLAYERS_MODEL_PATH)
    ball_model    = YOLO(BALL_MODEL_PATH)
    print("Models loaded.\n")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video at {VIDEO_PATH}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {src_w}x{src_h} @ {fps:.1f}fps — {total_frames} frames")

    os.makedirs("output", exist_ok=True)
    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (DASHBOARD_WIDTH, DASHBOARD_HEIGHT),
    )

    # Initialise components
    tracker            = PlayerTracker(fps=fps)
    ball_interpolator  = BallInterpolator(
        buffer_size=60,
        max_gap=12,         # FIX: was 15, now 12 with velocity extrapolation
        smooth_window=3,
        velocity_frames=4,
    )
    team_classifier    = TeamClassifier()
    possession_tracker = PossessionTracker(
        possession_radius_px=50,    # FIX: was 80 — tighter
        smoothing_window=300,       # FIX: was 150 — ~12s at 25fps
    )
    triangle_annotator, label_annotator = make_annotators()

    team_colour_a     = TEAM_COLOURS_BGR[0]
    team_colour_b     = TEAM_COLOURS_BGR[1]
    classifier_fitted = False
    frame_idx         = 0

    print("Processing — press Q to close preview window...\n")

    while True:
        success, frame = cap.read()
        if not success:
            break

        fr = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

        # ── Inference ──────────────────────────────────────────────────────────────
        # Players: run on resized frame — players are large enough at any resolution
        players_result = players_model(fr, conf=PLAYERS_CONF, verbose=False)[0]

        # Ball: run on FULL resolution frame — ball is too small at 960x540
        ball_result_full = ball_model(
            frame,          # original 1920x1080, NOT the resized fr
            conf=BALL_CONF,
            iou=BALL_IOU,
            imgsz=1280,
            verbose=False,
        )[0]

        players_dets = sv.Detections.from_ultralytics(players_result)
        ball_dets    = sv.Detections.from_ultralytics(ball_result_full)

        # Scale ball coordinates from full res → dashboard res
        scale_x = VIDEO_WIDTH  / src_w   # 960/1920 = 0.5
        scale_y = VIDEO_HEIGHT / src_h   # 540/1080 = 0.5
        if len(ball_dets) > 0:
            ball_dets.xyxy[:, [0, 2]] *= scale_x
            ball_dets.xyxy[:, [1, 3]] *= scale_y

        # ── Split player classes ────────────────────────────────────────────────
        gk_dets           = players_dets[players_dets.class_id == CLASS_GOALKEEPER]
        player_dets       = players_dets[players_dets.class_id == CLASS_PLAYER]
        ref_dets          = players_dets[players_dets.class_id == CLASS_REFEREE]
        ball_from_players = players_dets[players_dets.class_id == CLASS_BALL]

        # ── Filter edge detections ─────────────────────────────────────────────
        ball_from_players = filter_edge_detections(
            ball_from_players, VIDEO_WIDTH, VIDEO_HEIGHT)
        ball_dets         = filter_edge_detections(
            ball_dets, VIDEO_WIDTH, VIDEO_HEIGHT)

        # ── Ensemble ball models ───────────────────────────────────────────────
        final_ball_dets = merge_ball_detections(ball_from_players, ball_dets)

        # ── Tracking ───────────────────────────────────────────────────────────
        tracked_players = tracker.update_players(
            sv.Detections.merge([player_dets, gk_dets, ref_dets]))
        tracked_ball    = tracker.update_ball(final_ball_dets)

        tracked_player_only = tracked_players[
            tracked_players.class_id == CLASS_PLAYER]
        tracked_gk  = tracked_players[
            tracked_players.class_id == CLASS_GOALKEEPER]
        tracked_ref = tracked_players[
            tracked_players.class_id == CLASS_REFEREE]

        # ── Team classification ────────────────────────────────────────────────
        if not classifier_fitted and len(tracked_player_only) >= 6:
            classifier_fitted = team_classifier.fit(fr, tracked_player_only)
            if classifier_fitted:
                print(f"Frame {frame_idx}: Team classifier fitted.")
                team_colour_a = team_classifier.get_team_colour_bgr(0)
                team_colour_b = team_classifier.get_team_colour_bgr(1)

        player_teams = (
            team_classifier.predict(fr, tracked_player_only)
            if classifier_fitted and len(tracked_player_only) > 0
            else np.zeros(len(tracked_player_only), dtype=int)
        )

        # ── Ball interpolation ─────────────────────────────────────────────────
        ball_position = ball_interpolator.update(frame_idx, tracked_ball)

        # ── Possession — only update when ball confidently located ─────────────
        if ball_position and ball_interpolator.ball_is_confident:
            possession_tracker.update(
                ball_position, tracked_player_only, player_teams)
        else:
            possession_tracker.update(None, tracked_player_only, player_teams)

        # ── Annotate video frame ───────────────────────────────────────────────
        annotated = annotate_players(
            fr, tracked_player_only, player_teams,
            tracked_gk, tracked_ref,
            team_colour_a, team_colour_b, label_annotator,
        )

        if len(tracked_ball) > 0:
            annotated = triangle_annotator.annotate(annotated, tracked_ball)
        elif ball_position and ball_interpolator.ball_is_confident:
            cx, cy = int(ball_position[0]), int(ball_position[1])
            cv2.circle(annotated, (cx, cy), 8, (180, 180, 180), 2)
            cv2.putText(annotated, "~", (cx - 5, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # ── Build dashboard ────────────────────────────────────────────────────
        team_a_count    = int(np.sum(player_teams == 0)) if classifier_fitted else 0
        team_b_count    = int(np.sum(player_teams == 1)) if classifier_fitted else 0
        ball_trail_data = ball_interpolator.get_trail(n=BALL_TRAIL_LENGTH)

        right_panel = build_right_panel(
            frame_idx=frame_idx,
            fps=fps,
            possession_tracker=possession_tracker,
            ball_trail=ball_trail_data,
            player_count=len(tracked_player_only),
            team_a_count=team_a_count,
            team_b_count=team_b_count,
            team_colour_a=team_colour_a,
            team_colour_b=team_colour_b,
            ball_confident=ball_interpolator.ball_is_confident,
            frames_since_ball=ball_interpolator.frames_since_detection,
        )

        dashboard = np.hstack([annotated, right_panel])
        writer.write(dashboard)

        cv2.imshow("Pitch Vision — Dashboard", dashboard)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Preview closed — continuing to write output...")
            cv2.destroyAllWindows()

        frame_idx += 1

        if frame_idx % 100 == 0:
            pct_a, pct_b = possession_tracker.get_possession_percentage()
            if ball_interpolator.ball_is_confident:
                ball_status = "DETECTED"
            elif ball_interpolator.frames_since_detection <= 12:
                ball_status = f"EST (+{ball_interpolator.frames_since_detection}f)"
            else:
                ball_status = "LOST"
            print(f"Frame {frame_idx}/{total_frames} | "
                  f"Poss: {TEAM_NAMES[0]} {pct_a:.1f}% — "
                  f"{TEAM_NAMES[1]} {pct_b:.1f}% | "
                  f"Ball: {ball_status}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\n✓ Dashboard saved to {OUTPUT_PATH}")
    ov_a, ov_b = possession_tracker.get_overall_possession()
    print(f"Final possession: {TEAM_NAMES[0]} {ov_a:.1f}% — {TEAM_NAMES[1]} {ov_b:.1f}%")