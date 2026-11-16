"""
Realistic Benchmark Data Generator for UTA-RLDD Simulation.
Generates multi-subject synthetic physiological features matching published literature distributions
so the entire training, evaluation, ablation, and demo pipeline runs immediately out of the box.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from src.utils.config import ALL_FEATURE_NAMES, FEATURES_DATA_DIR


def generate_utarldd_benchmark_dataset(
    num_subjects: int = 60,
    clips_per_class: int = 6,
    random_seed: int = 42,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generates realistic synthetic dataset matching the 60-subject UTA-RLDD dataset structure.
    Classes:
        0: Alert
        1: Low Vigilant
        2: Drowsy
    Features:
        Full 22-dimensional hybrid feature set (Eye, Mouth, Head Pose, Eyebrow)
    """
    np.random.seed(random_seed)
    records = []

    for subj_idx in range(1, num_subjects + 1):
        subj_id = f"subject_{subj_idx:02d}"

        # Per-subject anatomical physiological bias (pre-calibration variance)
        subj_eye_factor = np.random.uniform(0.92, 1.08)
        subj_mouth_factor = np.random.uniform(0.90, 1.10)
        subj_pose_instability = np.random.uniform(0.85, 1.15)
        subj_brow_factor = np.random.uniform(0.90, 1.10)

        for label in [0, 1, 2]:
            video_id = f"{subj_id}_class_{label}"

            for clip_idx in range(clips_per_class):
                rec = {
                    "subject_id": subj_id,
                    "video_id": video_id,
                    "clip_id": f"{video_id}_clip_{clip_idx:02d}",
                    "label": label,
                }

                if label == 0:  # Alert
                    # Eye
                    rec["ear_mean"] = np.random.normal(0.98, 0.04) * subj_eye_factor
                    rec["ear_std"] = np.random.normal(0.04, 0.01)
                    rec["ear_min"] = max(0.55, np.random.normal(0.70, 0.05))
                    rec["ear_max"] = np.random.normal(1.05, 0.04)
                    rec["blink_rate"] = max(8.0, np.random.normal(16.0, 3.5))
                    rec["mean_blink_duration"] = np.random.normal(0.20, 0.03)
                    rec["max_blink_duration"] = np.random.normal(0.28, 0.05)
                    rec["perclos"] = max(0.01, np.random.normal(0.04, 0.015))

                    # Mouth
                    rec["mar_mean"] = np.random.normal(0.85, 0.05) * subj_mouth_factor
                    rec["mar_std"] = np.random.normal(0.04, 0.01)
                    rec["mar_max"] = np.random.normal(1.05, 0.06)
                    rec["yawn_rate"] = 0.0
                    rec["mean_yawn_duration"] = 0.0
                    rec["yawn_duration_ratio"] = 0.0

                    # Head Pose
                    rec["pitch_mean"] = np.random.normal(0.0, 1.5)
                    rec["pitch_std"] = np.random.normal(1.8, 0.4) * subj_pose_instability
                    rec["pitch_min"] = -np.random.uniform(2.0, 5.0)
                    rec["pitch_max"] = np.random.uniform(2.0, 5.0)
                    rec["yaw_mean"] = np.random.normal(0.0, 2.0)
                    rec["yaw_std"] = np.random.normal(2.2, 0.5)
                    rec["roll_mean"] = np.random.normal(0.0, 1.2)
                    rec["roll_std"] = np.random.normal(1.5, 0.3)
                    rec["nod_count"] = 0.0
                    rec["nod_rate"] = 0.0

                    # Eyebrow
                    rec["brow_dist_mean"] = np.random.normal(1.00, 0.03) * subj_brow_factor
                    rec["brow_dist_std"] = np.random.normal(0.03, 0.01)
                    rec["brow_dist_min"] = np.random.normal(0.92, 0.03)
                    rec["brow_movement_energy"] = np.random.normal(1.5, 0.3)

                elif label == 1:  # Low Vigilant
                    # Eye: early signs of fatigue (elevated blink rate, slightly lower EAR, mild PERCLOS)
                    rec["ear_mean"] = np.random.normal(0.88, 0.05) * subj_eye_factor
                    rec["ear_std"] = np.random.normal(0.07, 0.015)
                    rec["ear_min"] = max(0.40, np.random.normal(0.52, 0.06))
                    rec["ear_max"] = np.random.normal(0.98, 0.04)
                    rec["blink_rate"] = max(18.0, np.random.normal(28.0, 5.0))
                    rec["mean_blink_duration"] = np.random.normal(0.32, 0.05)
                    rec["max_blink_duration"] = np.random.normal(0.55, 0.08)
                    rec["perclos"] = min(0.24, max(0.08, np.random.normal(0.14, 0.03)))

                    # Mouth: occasional yawning, lip stretches
                    is_yawn_clip = np.random.rand() > 0.65
                    rec["mar_mean"] = np.random.normal(0.98, 0.08) * subj_mouth_factor
                    rec["mar_std"] = np.random.normal(0.09, 0.03)
                    rec["mar_max"] = np.random.normal(1.35, 0.15) if is_yawn_clip else np.random.normal(1.10, 0.08)
                    rec["yawn_rate"] = np.random.uniform(0.8, 2.0) if is_yawn_clip else 0.0
                    rec["mean_yawn_duration"] = np.random.normal(1.8, 0.3) if is_yawn_clip else 0.0
                    rec["yawn_duration_ratio"] = np.random.uniform(0.05, 0.14) if is_yawn_clip else 0.0

                    # Head Pose: mild posture wobble
                    rec["pitch_mean"] = np.random.normal(-2.5, 2.0)
                    rec["pitch_std"] = np.random.normal(3.8, 0.8) * subj_pose_instability
                    rec["pitch_min"] = -np.random.uniform(6.0, 12.0)
                    rec["pitch_max"] = np.random.uniform(3.0, 8.0)
                    rec["yaw_mean"] = np.random.normal(1.0, 3.0)
                    rec["yaw_std"] = np.random.normal(3.8, 0.8)
                    rec["roll_mean"] = np.random.normal(0.5, 2.0)
                    rec["roll_std"] = np.random.normal(2.5, 0.5)
                    rec["nod_count"] = float(np.random.choice([0, 1], p=[0.7, 0.3]))
                    rec["nod_rate"] = rec["nod_count"] * 12.0

                    # Eyebrow: tension, brow furrowing as driver fights sleep
                    rec["brow_dist_mean"] = np.random.normal(0.92, 0.04) * subj_brow_factor
                    rec["brow_dist_std"] = np.random.normal(0.06, 0.02)
                    rec["brow_dist_min"] = np.random.normal(0.80, 0.04)
                    rec["brow_movement_energy"] = np.random.normal(2.8, 0.6)

                else:  # label == 2: Drowsy
                    # Eye: severe eye closures, slow blinks, high PERCLOS
                    rec["ear_mean"] = np.random.normal(0.68, 0.06) * subj_eye_factor
                    rec["ear_std"] = np.random.normal(0.12, 0.02)
                    rec["ear_min"] = max(0.15, np.random.normal(0.28, 0.05))
                    rec["ear_max"] = np.random.normal(0.88, 0.05)
                    rec["blink_rate"] = max(10.0, np.random.normal(20.0, 6.0))  # Can drop due to long closures
                    rec["mean_blink_duration"] = np.random.normal(0.58, 0.12)
                    rec["max_blink_duration"] = np.random.normal(1.20, 0.25)
                    rec["perclos"] = min(0.75, max(0.25, np.random.normal(0.38, 0.08)))

                    # Mouth: frequent deep yawning
                    is_yawn_clip = np.random.rand() > 0.30
                    rec["mar_mean"] = np.random.normal(1.25, 0.15) * subj_mouth_factor
                    rec["mar_std"] = np.random.normal(0.18, 0.04)
                    rec["mar_max"] = np.random.normal(1.75, 0.20) if is_yawn_clip else np.random.normal(1.25, 0.10)
                    rec["yawn_rate"] = np.random.uniform(2.0, 5.0) if is_yawn_clip else 0.0
                    rec["mean_yawn_duration"] = np.random.normal(3.2, 0.6) if is_yawn_clip else 0.0
                    rec["yawn_duration_ratio"] = np.random.uniform(0.18, 0.40) if is_yawn_clip else 0.0

                    # Head Pose: head nodding episodes, micro-sleep drops
                    nod_episodes = np.random.choice([1, 2, 3], p=[0.35, 0.45, 0.20])
                    rec["pitch_mean"] = np.random.normal(-7.5, 3.5)
                    rec["pitch_std"] = np.random.normal(7.2, 1.5) * subj_pose_instability
                    rec["pitch_min"] = -np.random.uniform(18.0, 32.0)
                    rec["pitch_max"] = np.random.uniform(5.0, 12.0)
                    rec["yaw_mean"] = np.random.normal(-2.0, 5.0)
                    rec["yaw_std"] = np.random.normal(6.5, 1.2)
                    rec["roll_mean"] = np.random.normal(-3.0, 3.5)
                    rec["roll_std"] = np.random.normal(4.8, 0.9)
                    rec["nod_count"] = float(nod_episodes)
                    rec["nod_rate"] = float(nod_episodes * 12.0)

                    # Eyebrow: drooping, muscle fatigue
                    rec["brow_dist_mean"] = np.random.normal(0.82, 0.05) * subj_brow_factor
                    rec["brow_dist_std"] = np.random.normal(0.08, 0.02)
                    rec["brow_dist_min"] = np.random.normal(0.68, 0.05)
                    rec["brow_movement_energy"] = np.random.normal(3.8, 0.8)

                records.append(rec)

    df = pd.DataFrame(records)

    # Ensure all defined features exist
    for f in ALL_FEATURE_NAMES:
        if f not in df.columns:
            df[f] = 0.0

    if output_path is None:
        output_path = FEATURES_DATA_DIR / "utarldd_benchmark_features.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Benchmark Generator] Successfully generated {len(df)} samples across {num_subjects} subjects.")
    print(f"  Saved to: {output_path}")
    return df

