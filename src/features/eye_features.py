"""
Eye feature extraction: EAR, Blink Rate, Blink Duration, PERCLOS.
Author: Archee Jaiswal
"""

from typing import Dict, List, Tuple
import numpy as np


def compute_ear(eye_landmarks: np.ndarray) -> float:
    """
    Computes Eye Aspect Ratio (EAR) from 6 landmark points:
    p1: outer corner, p4: inner corner
    p2, p3: upper eyelid points
    p5, p6: lower eyelid points

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2.0 * ||p1 - p4||)
    """
    if eye_landmarks is None or len(eye_landmarks) < 6:
        return 0.0

    # Vertical eye distances
    d_v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    d_v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])

    # Horizontal eye distance
    d_h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

    if d_h < 1e-6:
        return 0.0

    ear = (d_v1 + d_v2) / (2.0 * d_h)
    return float(ear)


def compute_frame_ear(
    left_eye_pts: np.ndarray, right_eye_pts: np.ndarray
) -> Tuple[float, float, float]:
    """
    Returns (left_ear, right_ear, average_ear) for a single frame.
    """
    left_ear = compute_ear(left_eye_pts)
    right_ear = compute_ear(right_eye_pts)
    avg_ear = (left_ear + right_ear) / 2.0
    return left_ear, right_ear, avg_ear


def extract_eye_temporal_features(
    ear_series: List[float],
    fps: float,
    ear_threshold: float = 0.22,
    min_blink_frames: int = 2,
    max_blink_frames: int = 15,
) -> Dict[str, float]:
    """
    Computes temporal eye metrics over a sequence of frame EAR values:
    - ear_mean, ear_std, ear_min, ear_max
    - blink_count, blink_rate (blinks per minute)
    - mean_blink_duration (sec), max_blink_duration (sec)
    - perclos (fraction of time EAR < ear_threshold)
    """
    if not ear_series:
        return {
            "ear_mean": 0.0,
            "ear_std": 0.0,
            "ear_min": 0.0,
            "ear_max": 0.0,
            "blink_rate": 0.0,
            "mean_blink_duration": 0.0,
            "max_blink_duration": 0.0,
            "perclos": 0.0,
        }

    ears = np.array(ear_series, dtype=np.float32)
    total_frames = len(ears)
    duration_sec = total_frames / max(fps, 1.0)

    # Statistical properties
    ear_mean = float(np.mean(ears))
    ear_std = float(np.std(ears))
    ear_min = float(np.min(ears))
    ear_max = float(np.max(ears))

    # PERCLOS (P80 metric): proportion of frames where eye is closed
    closed_frames = np.sum(ears < ear_threshold)
    perclos = float(closed_frames / total_frames) if total_frames > 0 else 0.0

    # Blink detection via state transitions
    blink_durations = []
    current_closed_count = 0

    for ear in ears:
        if ear < ear_threshold:
            current_closed_count += 1
        else:
            if min_blink_frames <= current_closed_count <= max_blink_frames:
                # Blink ended
                duration_in_sec = current_closed_count / fps
                blink_durations.append(duration_in_sec)
            current_closed_count = 0

    # Trailing check
    if min_blink_frames <= current_closed_count <= max_blink_frames:
        blink_durations.append(current_closed_count / fps)

    blink_count = len(blink_durations)
    # Blink rate scaled to blinks per minute (standard clinical unit)
    blink_rate = (blink_count / duration_sec) * 60.0 if duration_sec > 0 else 0.0

    mean_blink_dur = float(np.mean(blink_durations)) if blink_durations else 0.0
    max_blink_dur = float(np.max(blink_durations)) if blink_durations else 0.0

    return {
        "ear_mean": ear_mean,
        "ear_std": ear_std,
        "ear_min": ear_min,
        "ear_max": ear_max,
        "blink_rate": float(blink_rate),
        "mean_blink_duration": mean_blink_dur,
        "max_blink_duration": max_blink_dur,
        "perclos": perclos,
    }

