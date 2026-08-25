# pitch-vision

![CI Status](https://github.com/CarterFitzgerald/pitch-vision/actions/workflows/ci.yml/badge.svg)

A football match analysis system built with computer vision and machine learning. Processes broadcast video to track players, classify teams, estimate ball possession, and calculate real-world player speeds — using YOLO26 for detection and automatic pitch keypoint detection for perspective-correct measurements.

> **Status:** In development — Phase 3 (Dashboard + Team classification + possession tracking) in progress.

---

## What it does

- **Player & ball detection** — Two dedicated YOLO26 models: one for players, goalkeepers, and referees; one for ball-only detection (4,948 training images)
- **Multi-object tracking** — Persistent player IDs across frames using ByteTrack, with ball position interpolation for occluded frames
- **Team classification** — Automatic jersey colour clustering (KMeans) to split players into Team A / Team B without manual labelling
- **Ball possession** — Proximity-based possession attribution with per-team running percentage overlay
- **Real-world speed & distance** — Perspective transform using automatically detected pitch keypoints (32-point field model) to convert pixel movement into km/h and metres
- **Annotated video output** — Full match clip with bounding boxes, team colours, possession bar, and speed labels rendered per frame

---

## Architecture

```
Raw video
    │
    ├── YOLO26 (players model)  ──┐
    │                             ├── Merged detections
    └── YOLO26 (ball model)    ──┘
                │
         ByteTrack (tracking IDs)
                │
         YOLO26-pose (field keypoints → homography)
                │
    ┌───────────┴───────────┐
    │                       │
 KMeans team          Perspective transform
 classification         (px → metres)
    │                       │
 Possession %          Speed (km/h) &
 per team              distance (m)
    │                       │
    └───────────┬───────────┘
                │
        Annotated video output
```

---

## Datasets

Three datasets from [Roboflow Universe](https://universe.roboflow.com), all CC BY 4.0:

| Dataset | Format | Images | Classes |
|---|---|---|---|
| [football-players-detection v20](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/20) | YOLO26 | 372 | ball, goalkeeper, player, referee |
| [football-ball-detection v4](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg/dataset/4) | YOLO26 | 4,948 | ball |
| [football-field-detection v18](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi/dataset/18) | YOLO26-pose | 317 | pitch (32 keypoints) |

Datasets are not included in this repo. Download from the links above and place under `data/` — see [Setup](#setup) below.

---

## Tech stack

- **Python 3.11+**
- **YOLO26** (Ultralytics) — player and ball detection
- **YOLO26-pose** (Ultralytics) — pitch keypoint detection
- **supervision** — ByteTrack multi-object tracking, video annotation
- **OpenCV** — video I/O, optical flow, perspective transform
- **scikit-learn** — KMeans jersey colour clustering
- **NumPy / Pandas** — numerical processing, ball interpolation

---

## Project structure

```
pitch-vision/
├── data/                   # Datasets (gitignored — download separately)
│   ├── players/
│   ├── ball/
│   └── field/
├── models/                 # Trained weights (gitignored)
│   ├── players.pt
│   ├── ball.pt
│   └── field.pt
├── src/
│   ├── training/
│   │   ├── train_players.py
│   │   ├── train_ball.py
│   │   └── train_field.py
│   ├── tracking/
│   │   ├── tracker.py
│   │   └── ball_interpolator.py
│   ├── analytics/
│   │   ├── team_classifier.py
│   │   ├── possession_tracker.py
│   │   └── speed_estimator.py
│   ├── field/
│   │   └── keypoint_homography.py
│   └── pipeline.py         # Main end-to-end pipeline
├── output/                 # Annotated video output (gitignored)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/CarterFitzgerald/pitch-vision.git
cd pitch-vision

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download datasets from Roboflow and extract to data/
# (see Dataset links above)
```

---

## Usage

> Training scripts and pipeline coming in Phase 1–4. This section will be updated as each phase completes.

```bash
# Train detection models (requires datasets in data/)
python src/training/train_players.py
python src/training/train_ball.py

# Run the full pipeline on a video
python src/pipeline.py --input path/to/match.mp4 --output output/annotated.mp4
```

---

## Results

> To be updated after training completes.

| Model | mAP50 | mAP50-95 |
|---|---|---|
| Players (YOLO26) | 0.910 | 0.629 |
| Ball (YOLO26) | 0.882 | 0.551 |
| Field keypoints (YOLO26-pose) | 0.970 | 0.725 |

---

## Roadmap

- [x] Dataset acquisition and inspection
- [x] Phase 1 — YOLO26 detection training (players + ball)
- [x] Phase 2 — ByteTrack multi-object tracking
- [x] Phase 3 — Team classification + possession tracking
- [ ] Phase 4 — Speed estimation + annotated video output

---

## Current Challenges

- **Ball not being detected** — Ball often not detected, especially in the air; currently this has a knock-on effect on possession tracking and ball mapping. Possible solutions include:
  - Increasing ball dataset size with more diverse images
  - Training a separate YOLO26 model specifically for aerial ball detection
  - Using optical flow to track the ball when not detected

---

## Acknowledgements

This project was inspired by the Roboflow tutorial **"Football AI Tutorial: From Basics to Advanced Stats with Python"** by [Piotr Skalski](https://github.com/SkalskiP) at Roboflow (August 2024).

- 📺 [Watch the tutorial on YouTube](https://www.youtube.com/watch?v=aBVGKoNZQUw)
- 📝 [Accompanying blog posts on the Roboflow blog](https://blog.roboflow.com/track-football-players/)
- 🔬 [roboflow/sports — open source sports CV tools](https://github.com/roboflow/sports)

This implementation extends the original tutorial with YOLO26 (in place of YOLOv8) and uses the field keypoint model to automate perspective transform rather than relying on manually selected pitch points.

All three datasets were created and published by Roboflow under CC BY 4.0:

| Dataset | Author | Link |
|---|---|---|
| football-players-detection | Roboflow | [v20](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/20) |
| football-ball-detection | Roboflow | [v4](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg/dataset/4) |
| football-field-detection | Roboflow | [v18](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi/dataset/18) |

---

## License

Datasets used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code MIT.