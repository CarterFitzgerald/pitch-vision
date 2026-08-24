import os
import sys
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# ─── Config ───────────────────────────────────────────────────────────────────

PLAYERS_MODEL_PATH = "runs/detect/runs/players/yolo26x/weights/best.pt"
BALL_MODEL_PATH    = "runs/detect/runs/ball/yolo26x/weights/best.pt"
VIDEO_PATH         = "data/video/match.mp4"   # swap to your video filename
OUTPUT_PATH        = "output/test_frame.jpg"
FRAME_INDEX        = 500                       # which frame to sample (change freely)

# Detection confidence thresholds
PLAYERS_CONF = 0.3
BALL_CONF    = 0.15   # lower — ball is small and easy to miss

# Class IDs from players model (check your data.yaml if these seem wrong)
CLASS_BALL        = 0
CLASS_GOALKEEPER  = 1
CLASS_PLAYER      = 2
CLASS_REFEREE     = 3

# Colours (BGR)
COLOUR_TEAM_A    = sv.Color.from_hex("#FF6B00")   # orange
COLOUR_TEAM_B    = sv.Color.from_hex("#1A73E8")   # blue
COLOUR_REF       = sv.Color.from_hex("#FFFF00")   # yellow
COLOUR_GK        = sv.Color.from_hex("#00FF00")   # green
COLOUR_BALL      = sv.Color.from_hex("#FFFFFF")   # white

# ─── Load models ──────────────────────────────────────────────────────────────

print("Loading models...")

if not os.path.exists(PLAYERS_MODEL_PATH):
    print(f"ERROR: Players model not found at {PLAYERS_MODEL_PATH}")
    print("Check your runs/ folder for the correct path and update PLAYERS_MODEL_PATH.")
    sys.exit(1)

if not os.path.exists(BALL_MODEL_PATH):
    print(f"ERROR: Ball model not found at {BALL_MODEL_PATH}")
    print("Check your runs/ folder for the correct path and update BALL_MODEL_PATH.")
    sys.exit(1)

players_model = YOLO(PLAYERS_MODEL_PATH)
ball_model    = YOLO(BALL_MODEL_PATH)
print("Models loaded.")

# ─── Extract frame ────────────────────────────────────────────────────────────

print(f"Extracting frame {FRAME_INDEX} from {VIDEO_PATH}...")

if not os.path.exists(VIDEO_PATH):
    print(f"ERROR: Video not found at {VIDEO_PATH}")
    print("Place a match video at data/video/match.mp4 or update VIDEO_PATH.")
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
success, frame = cap.read()
cap.release()

if not success:
    print(f"ERROR: Could not read frame {FRAME_INDEX}. Try a lower FRAME_INDEX value.")
    sys.exit(1)

print(f"Frame shape: {frame.shape}")

# ─── Run inference ────────────────────────────────────────────────────────────

print("Running inference...")

players_result = players_model(frame, conf=PLAYERS_CONF)[0]
ball_result    = ball_model(frame, conf=BALL_CONF)[0]

# Convert to supervision Detections
players_detections = sv.Detections.from_ultralytics(players_result)
ball_detections    = sv.Detections.from_ultralytics(ball_result)

print(f"Players model detections: {len(players_detections)}")
print(f"Ball model detections:    {len(ball_detections)}")

# ─── Split player detections by class ─────────────────────────────────────────

# Ball detections from the players model (class 0) — we'll merge with ball model below
ball_from_players  = players_detections[players_detections.class_id == CLASS_BALL]
goalkeeper_dets    = players_detections[players_detections.class_id == CLASS_GOALKEEPER]
player_dets        = players_detections[players_detections.class_id == CLASS_PLAYER]
referee_dets       = players_detections[players_detections.class_id == CLASS_REFEREE]

# Merge ball detections from both models (ball model is more accurate)
# Simple merge: use dedicated ball model detections, fall back to players model if none found
if len(ball_detections) > 0:
    final_ball_dets = ball_detections
    print("Using dedicated ball model detections.")
else:
    final_ball_dets = ball_from_players
    print("Ball model found nothing — falling back to players model ball detections.")

print(f"Players:     {len(player_dets)}")
print(f"Goalkeepers: {len(goalkeeper_dets)}")
print(f"Referees:    {len(referee_dets)}")
print(f"Ball:        {len(final_ball_dets)}")

# ─── Annotate ─────────────────────────────────────────────────────────────────

annotated = frame.copy()

# Ellipse annotator for players (classic football tracking style)
ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorLookup.INDEX,
    thickness=2,
)

# Triangle annotator for the ball (pointing down like a marker)
triangle_annotator = sv.TriangleAnnotator(
    color=COLOUR_BALL,
    base=20,
    height=18,
    position=sv.Position.TOP_CENTER,  # triangle above the ball
)

# Label annotator for class names + confidence
label_annotator = sv.LabelAnnotator(
    text_scale=0.4,
    text_thickness=1,
    text_padding=3,
    text_position=sv.Position.BOTTOM_CENTER,
)

# Draw players with ellipses (team colours will come in Phase 3 — for now use index colours)
if len(player_dets) > 0:
    player_ellipse_annotator = sv.EllipseAnnotator(color=sv.Color.from_hex("#1A73E8"), thickness=2)
    annotated = player_ellipse_annotator.annotate(annotated, player_dets)

if len(goalkeeper_dets) > 0:
    gk_ellipse_annotator = sv.EllipseAnnotator(color=COLOUR_GK, thickness=2)
    annotated = gk_ellipse_annotator.annotate(annotated, goalkeeper_dets)

if len(referee_dets) > 0:
    ref_ellipse_annotator = sv.EllipseAnnotator(color=COLOUR_REF, thickness=2)
    annotated = ref_ellipse_annotator.annotate(annotated, referee_dets)

# Draw ball with triangle
if len(final_ball_dets) > 0:
    annotated = triangle_annotator.annotate(annotated, final_ball_dets)

# Draw confidence labels on all player detections
all_player_dets = sv.Detections.merge([player_dets, goalkeeper_dets, referee_dets])
if len(all_player_dets) > 0:
    labels = [
        f"{players_model.names[cid]} {conf:.2f}"
        for cid, conf in zip(all_player_dets.class_id, all_player_dets.confidence)
    ]
    label_annotator_white = sv.LabelAnnotator(
        text_scale=0.4,
        text_thickness=1,
        text_padding=3,
        text_color=sv.Color.WHITE,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    annotated = label_annotator_white.annotate(annotated, all_player_dets, labels=labels)

# ─── Save output ──────────────────────────────────────────────────────────────

os.makedirs("output", exist_ok=True)
cv2.imwrite(OUTPUT_PATH, annotated)
print(f"\nSaved annotated frame to {OUTPUT_PATH}")
print("Open it in File Explorer to inspect the detections.")