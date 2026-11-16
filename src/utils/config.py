"""
Global configuration and constants for Multi-Level Driver Drowsiness Detection.
Covers landmark indices, class definitions, threshold defaults, and file paths.
"""

import os
from pathlib import Path

# Silence Matplotlib cache directory warning on macOS
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs("/tmp/matplotlib", exist_ok=True)

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results"

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DATA_DIR, SYNTHETIC_DATA_DIR, SAVED_MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Drowsiness Target Classes (UTA-RLDD standard)
CLASSES = {
    0: "Alert",
    1: "Low Vigilant",
    2: "Drowsy"
}
CLASS_NAMES = ["Alert", "Low Vigilant", "Drowsy"]

# MediaPipe Face Mesh Landmark Indices
# Left Eye (6-point EAR calculation)
# p1: outer corner (33), p4: inner corner (133), p2: upper 1 (160), p3: upper 2 (158), p5: lower 2 (144), p6: lower 1 (153)
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

# Left Eye (from camera perspective / anatomical left)
# p1: inner corner (362), p4: outer corner (263), p2: upper 1 (385), p3: upper 2 (387), p5: lower 2 (373), p6: lower 1 (380)
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# Mouth Landmarks for MAR
# Corners: 61 (right), 291 (left)
# Top-Bottom pairs: (39, 181), (0, 17), (269, 405)
MOUTH_LANDMARKS = {
    "left_corner": 291,
    "right_corner": 61,
    "upper_lip_center": 0,
    "lower_lip_center": 17,
    "upper_lip_left": 269,
    "lower_lip_left": 405,
    "upper_lip_right": 39,
    "lower_lip_right": 181,
    "upper_lip_inner": 13,
    "lower_lip_inner": 14
}

# Eyebrow Landmarks
LEFT_EYEBROW_LANDMARKS = [70, 63, 105, 66, 107]
RIGHT_EYEBROW_LANDMARKS = [336, 296, 334, 293, 300]
LEFT_EYEBROW_CENTER = 105
RIGHT_EYEBROW_CENTER = 334

# 3D Facial Landmarks for Head Pose Estimation (Perspective-n-Point / PnP)
# Corresponding to standard anthropometric 3D model:
# Nose tip, Chin, Left eye corner, Right eye corner, Left mouth corner, Right mouth corner
HEAD_POSE_LANDMARKS_INDICES = [1, 152, 263, 33, 291, 61]

# Default Video / Temporal Processing Parameters
DEFAULT_FPS = 30
WINDOW_SIZE_SEC = 5.0       # 5-second sliding window for feature aggregation
WINDOW_STEP_SEC = 1.0       # 1-second slide step
WINDOW_FRAMES = int(DEFAULT_FPS * WINDOW_SIZE_SEC)  # 150 frames
STEP_FRAMES = int(DEFAULT_FPS * WINDOW_STEP_SEC)    # 30 frames

# Default Baseline Heuristics (pre-calibration)
DEFAULT_EAR_THRESHOLD = 0.22      # Below this considered eye-closure
DEFAULT_MAR_THRESHOLD = 0.55      # Above this considered yawning
DEFAULT_PITCH_NOD_THRESHOLD = 15.0 # Degrees downward tilt considered nodding

# Feature Groups for Ablation Studies
FEATURE_GROUPS = {
    "eye": [
        "ear_mean", "ear_std", "ear_min", "ear_max",
        "blink_rate", "mean_blink_duration", "max_blink_duration",
        "perclos"
    ],
    "mouth": [
        "mar_mean", "mar_std", "mar_max",
        "yawn_rate", "mean_yawn_duration", "yawn_duration_ratio"
    ],
    "head_pose": [
        "pitch_mean", "pitch_std", "pitch_min", "pitch_max",
        "yaw_mean", "yaw_std",
        "roll_mean", "roll_std",
        "nod_count", "nod_rate"
    ],
    "eyebrow": [
        "brow_dist_mean", "brow_dist_std", "brow_dist_min",
        "brow_movement_energy"
    ]
}

ALL_FEATURE_NAMES = (
    FEATURE_GROUPS["eye"] +
    FEATURE_GROUPS["mouth"] +
    FEATURE_GROUPS["head_pose"] +
    FEATURE_GROUPS["eyebrow"]
)

