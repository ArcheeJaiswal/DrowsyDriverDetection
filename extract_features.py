"""
Batch Video Preprocessing and Feature Extraction Pipeline for UTA-RLDD.
Processes raw subject videos, extracts facial landmarks via MediaPipe,
computes calibrated hybrid features, and writes real feature tables.
Author: Archee Jaiswal
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.features.calibration import SubjectCalibrator
from src.features.eye_features import compute_frame_ear
from src.features.eyebrow_features import compute_eyebrow_distance
from src.features.feature_pipeline import HybridFeatureExtractor
from src.features.head_pose import HeadPoseEstimator
from src.features.mouth_features import compute_mar
from src.preprocessing.dataset_manager import UTARLDDManager
from src.preprocessing.landmark_extractor import LandmarkExtractor
from src.utils.config import (
    ALL_FEATURE_NAMES,
    CLASSES,
    DEFAULT_FPS,
    FEATURES_DATA_DIR,
    RAW_DATA_DIR,
    WINDOW_SIZE_SEC,
    WINDOW_STEP_SEC,
)


def process_video_file(
    video_path: str,
    subject_id: str,
    label: int,
    landmark_extractor: LandmarkExtractor,
    pose_estimator: HeadPoseEstimator,
    calibrator: SubjectCalibrator,
    feature_extractor: HybridFeatureExtractor,
    target_fps: int = DEFAULT_FPS,
    window_sec: float = WINDOW_SIZE_SEC,
    step_sec: float = WINDOW_STEP_SEC,
    max_segments_per_video: Optional[int] = 12,
) -> List[Dict[str, float]]:
    """
    Processes a single video:
    - Reads frames at standardized fps
    - Extracts 2D/3D facial landmarks
    - Calibrates neutral baseline on initial frames
    - Aggregates sliding window features
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [Warning] Could not open video: {video_path}")
        return []

    source_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    frame_interval = max(1, round(source_fps / target_fps))
    frames_per_window = int(target_fps * window_sec)
    frames_per_step = int(target_fps * step_sec)

    video_name = Path(video_path).stem
    video_id = f"{subject_id}_{video_name}"

    buf_ear, buf_mar, buf_brow = [], [], []
    buf_pitch, buf_yaw, buf_roll = [], [], []

    raw_frame_idx = 0
    segment_idx = 0
    records = []

    # Calibration buffer (first 100 valid frames)
    calib_samples = {"ear": [], "mar": [], "brow": [], "pitch": [], "yaw": [], "roll": []}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if raw_frame_idx % frame_interval == 0:
            lm_data = landmark_extractor.extract_landmarks(frame)

            if lm_data is not None:
                _, _, ear = compute_frame_ear(lm_data["left_eye"], lm_data["right_eye"])
                mar = compute_mar(lm_data["mouth"])
                brow = compute_eyebrow_distance(
                    lm_data["left_eyebrow"],
                    lm_data["right_eyebrow"],
                    lm_data["left_eye"],
                    lm_data["right_eye"],
                )
                pitch, yaw, roll = pose_estimator.estimate_pose(
                    lm_data["head_pose_points"],
                    frame_width=lm_data["frame_width"],
                    frame_height=lm_data["frame_height"],
                )

                # Accumulate calibration if neutral or if calibrator not yet calibrated
                if len(calib_samples["ear"]) < 90 and (label == 0 or not calibrator.is_calibrated):
                    calib_samples["ear"].append(ear)
                    calib_samples["mar"].append(mar)
                    calib_samples["brow"].append(brow)
                    calib_samples["pitch"].append(pitch)
                    calib_samples["yaw"].append(yaw)
                    calib_samples["roll"].append(roll)

                buf_ear.append(ear)
                buf_mar.append(mar)
                buf_brow.append(brow)
                buf_pitch.append(pitch)
                buf_yaw.append(yaw)
                buf_roll.append(roll)

                # Once sliding window is filled
                if len(buf_ear) >= frames_per_window:
                    # Calibrate if not yet calibrated
                    if not calibrator.is_calibrated and len(calib_samples["ear"]) >= 30:
                        calibrator.calibrate(
                            ear_samples=calib_samples["ear"],
                            mar_samples=calib_samples["mar"],
                            brow_samples=calib_samples["brow"],
                            pitch_samples=calib_samples["pitch"],
                            yaw_samples=calib_samples["yaw"],
                            roll_samples=calib_samples["roll"],
                        )

                    feats = feature_extractor.process_window(
                        buf_ear, buf_mar, buf_pitch, buf_yaw, buf_roll, buf_brow
                    )
                    feats["subject_id"] = subject_id
                    feats["video_id"] = video_id
                    feats["clip_id"] = f"{video_id}_seg_{segment_idx:03d}"
                    feats["label"] = label
                    feats["class_name"] = CLASSES.get(label, "Unknown")
                    records.append(feats)

                    segment_idx += 1
                    if max_segments_per_video and segment_idx >= max_segments_per_video:
                        break

                    # Slide window
                    buf_ear = buf_ear[frames_per_step:]
                    buf_mar = buf_mar[frames_per_step:]
                    buf_brow = buf_brow[frames_per_step:]
                    buf_pitch = buf_pitch[frames_per_step:]
                    buf_yaw = buf_yaw[frames_per_step:]
                    buf_roll = buf_roll[frames_per_step:]

        raw_frame_idx += 1

    cap.release()
    return records


def extract_features_from_dataset(
    raw_dir: Path = RAW_DATA_DIR,
    output_csv: Optional[Path] = None,
    max_segments_per_video: int = 25,
    target_fps: int = 10,
    window_sec: float = WINDOW_SIZE_SEC,
    step_sec: float = 2.0,
) -> pd.DataFrame:
    """
    Scans raw directory and extracts features from all available subject videos.
    Calibrates neutral baselines per subject on Alert recordings (label 0).
    """
    mgr = UTARLDDManager(raw_dir=raw_dir)
    df_files = mgr.scan_dataset()

    if df_files.empty:
        print(f"[Error] No videos found in {raw_dir}!")
        print("  Expected structure:")
        print("    data/raw/31/0.mp4   (Alert)")
        print("    data/raw/31/5.mp4   (Low-vigilant)")
        print("    data/raw/31/10.mp4  (Drowsy)")
        return pd.DataFrame()

    # Sort so label 0 (Alert) is processed first for each subject
    df_files["sort_order"] = df_files["label"].apply(lambda l: 0 if l == 0 else (1 if l == 1 else 2))
    df_files = df_files.sort_values(by=["subject_id", "sort_order"]).reset_index(drop=True)

    print(f"\n[Feature Extraction] Discovered {len(df_files)} videos across {df_files['subject_id'].nunique()} subjects.")
    print(df_files[["subject_id", "filename", "class_name"]].to_string())

    lm_extractor = LandmarkExtractor(max_num_faces=1, refine_landmarks=True)
    pose_estimator = HeadPoseEstimator()

    all_records = []
    current_subject = None
    calibrator = None
    feat_extractor = None

    for _, row in tqdm(df_files.iterrows(), total=len(df_files), desc="Processing Videos"):
        v_path = row["video_path"]
        s_id = str(row["subject_id"])
        lbl = int(row["label"])

        if lbl not in [0, 1, 2]:
            continue

        # Reset calibrator when moving to a new subject
        if s_id != current_subject or calibrator is None:
            current_subject = s_id
            calibrator = SubjectCalibrator()
            feat_extractor = HybridFeatureExtractor(fps=target_fps, calibrator=calibrator)

        recs = process_video_file(
            video_path=v_path,
            subject_id=s_id,
            label=lbl,
            landmark_extractor=lm_extractor,
            pose_estimator=pose_estimator,
            calibrator=calibrator,
            feature_extractor=feat_extractor,
            target_fps=target_fps,
            window_sec=window_sec,
            step_sec=step_sec,
            max_segments_per_video=max_segments_per_video,
        )
        all_records.extend(recs)

    lm_extractor.close()

    if not all_records:
        print("[Warning] No valid segments extracted.")
        return pd.DataFrame()

    df_features = pd.DataFrame(all_records)

    if output_csv is None:
        output_csv = FEATURES_DATA_DIR / "utarldd_real_features.csv"

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_csv, index=False)

    print(f"\n[Feature Extraction Complete] Saved {len(df_features)} extracted feature vectors to:")
    print(f"  -> {output_csv}")
    print("\nClass distribution:")
    print(df_features["class_name"].value_counts())
    print(f"Unique subjects: {df_features['subject_id'].nunique()}")
    return df_features


def main():
    parser = argparse.ArgumentParser(description="Extract Features from Raw UTA-RLDD Videos")
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DATA_DIR), help="Path to raw videos directory")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--max-segments", type=int, default=25, help="Max segments per video (default: 25)")
    parser.add_argument("--target-fps", type=int, default=10, help="Standardized processing FPS (default: 10)")
    args = parser.parse_args()

    extract_features_from_dataset(
        raw_dir=Path(args.raw_dir),
        output_csv=Path(args.output) if args.output else None,
        max_segments_per_video=args.max_segments,
        target_fps=args.target_fps,
    )


if __name__ == "__main__":
    main()

