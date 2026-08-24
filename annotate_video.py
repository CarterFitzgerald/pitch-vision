import os
import sys
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm

# ─── Config ───────────────────────────────────────────────────────────────────

PLAYERS_MODEL_PATH = "runs/detect/runs/players/yolo26x/weights/best.pt"
BALL_MODEL_PATH    = "runs/detect/runs/ball/yolo26x/weights/best.pt"
VIDEO_PATH         = "data/video/match.mp4"
OUTPUT_PATH        = "output/annotated_match.mp4"

PLAYERS_CONF = 0.3
BALL_CONF    = 0.15

# Class IDs from players model
CLASS_BALL       = 0
CLASS_GOALKEEPER = 1
CLASS_PLAYER     = 2
CLASS_REFEREE    = 3

# Colours (BGR)
COLOUR_PLAYER = sv.Color.from_hex("#1A73E8")   # blue — team colours come in Phase 3
COLOUR_GK     = sv.Color.from_hex("#00FF00")   # green
COLOUR_REF    = sv.Color.from_hex("#FFFF00")   # yellow
COLOUR_BALL   = sv.Color.from_hex("#FFFFFF")   # white

# ─── Load models ──────────────────────────────────────────────────────────────

print("Loading models...")

for path, name in [(PLAYERS_MODEL_PATH, "Players"), (BALL_MODEL_PATH, "Ball")]:
    if not os.path.exists(path):
        print(f"ERROR: {name} model not found at {path}")
        sys.exit(1)

players_model = YOLO(PLAYERS_MODEL_PATH)
ball_model    = YOLO(BALL_MODEL_PATH)
print("Models loaded.\n")

# ─── Video setup ──────────────────────────────────────────────────────────────

if not os.path.exists(VIDEO_PATH):
    print(f"ERROR: Video not found at {VIDEO_PATH}")
    sys.exit(1)

cap         = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps         = cap.get(cv2.CAP_PROP_FPS)
width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {width}x{height} @ {fps:.1f}fps — {total_frames} frames total")
print(f"Estimated runtime: ~{total_frames / fps / 60:.1f} minutes of video\n")

os.makedirs("output", exist_ok=True)

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

# ─── Annotators ───────────────────────────────────────────────────────────────

player_ellipse_annotator = sv.EllipseAnnotator(color=COLOUR_PLAYER, thickness=2)
gk_ellipse_annotator     = sv.EllipseAnnotator(color=COLOUR_GK, thickness=2)
ref_ellipse_annotator    = sv.EllipseAnnotator(color=COLOUR_REF, thickness=2)

triangle_annotator = sv.TriangleAnnotator(
    color=COLOUR_BALL,
    base=20,
    height=18,
    position=sv.Position.TOP_CENTER,
)

label_annotator = sv.LabelAnnotator(
    text_scale=0.4,
    text_thickness=1,
    text_padding=3,
    text_color=sv.Color.WHITE,
    text_position=sv.Position.BOTTOM_CENTER,
)

# ─── Frame loop ───────────────────────────────────────────────────────────────

print("Processing video...")

frame_idx = 0
with tqdm(total=total_frames, unit="frame") as pbar:
    while True:
        success, frame = cap.read()
        if not success:
            break

        # ── Inference ──
        players_result = players_model(frame, conf=PLAYERS_CONF, verbose=False)[0]
        ball_result    = ball_model(frame, conf=BALL_CONF, verbose=False)[0]

        players_detections = sv.Detections.from_ultralytics(players_result)
        ball_detections    = sv.Detections.from_ultralytics(ball_result)

        # ── Split by class ──
        goalkeeper_dets = players_detections[players_detections.class_id == CLASS_GOALKEEPER]
        player_dets     = players_detections[players_detections.class_id == CLASS_PLAYER]
        referee_dets    = players_detections[players_detections.class_id == CLASS_REFEREE]
        ball_from_players = players_detections[players_detections.class_id == CLASS_BALL]

        final_ball_dets = ball_detections if len(ball_detections) > 0 else ball_from_players

        # ── Annotate ──
        annotated = frame.copy()

        if len(player_dets) > 0:
            annotated = player_ellipse_annotator.annotate(annotated, player_dets)

        if len(goalkeeper_dets) > 0:
            annotated = gk_ellipse_annotator.annotate(annotated, goalkeeper_dets)

        if len(referee_dets) > 0:
            annotated = ref_ellipse_annotator.annotate(annotated, referee_dets)

        if len(final_ball_dets) > 0:
            annotated = triangle_annotator.annotate(annotated, final_ball_dets)

        # Labels on all players
        all_player_dets = sv.Detections.merge([player_dets, goalkeeper_dets, referee_dets])
        if len(all_player_dets) > 0:
            labels = [
                f"{players_model.names[cid]} {conf:.2f}"
                for cid, conf in zip(all_player_dets.class_id, all_player_dets.confidence)
            ]
            annotated = label_annotator.annotate(annotated, all_player_dets, labels=labels)

        # Frame counter overlay
        cv2.putText(
            annotated,
            f"Frame {frame_idx}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        writer.write(annotated)
        frame_idx += 1
        pbar.update(1)

cap.release()
writer.release()

print(f"\nDone. Annotated video saved to {OUTPUT_PATH}")
print(f"Processed {frame_idx} frames.")