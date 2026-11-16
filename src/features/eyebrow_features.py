"""
Eyebrow feature extraction: Eyebrow-to-Eye Distance and Movement Dynamics.
Author: Archee Jaiswal
"""

from typing import Dict, List
import numpy as np


def compute_eyebrow_distance(
    left_eyebrow_pts: np.ndarray,
    right_eyebrow_pts: np.ndarray,
    left_eye_pts: np.ndarray,
    right_eye_pts: np.ndarray,
) -> float:
    """
    Computes normalized distance between eyebrows and eyes:
    - Measures vertical distance between upper eyelid and eyebrow center.
    - Normalizes by inter-ocular distance (distance between outer eye corners)
      to maintain scale invariance across different camera distances.
    """
    if (
        left_eyebrow_pts is None
        or right_eyebrow_pts is None
        or left_eye_pts is None
        or right_eye_pts is None
    ):
        return 0.0

    # Centers of eyebrows
    left_brow_center = np.mean(left_eyebrow_pts, axis=0)
    right_brow_center = np.mean(right_eyebrow_pts, axis=0)

    # Centers of eyes
    left_eye_center = np.mean(left_eye_pts, axis=0)
    right_eye_center = np.mean(right_eye_pts, axis=0)

    # Vertical distance
    d_left = np.abs(left_brow_center[1] - left_eye_center[1])
    d_right = np.abs(right_brow_center[1] - right_eye_center[1])
    avg_vert_dist = (d_left + d_right) / 2.0

    # Scale normalization: inter-ocular distance
    # Outer corners: right_eye[0] and left_eye[3] in our index mapping
    iod = np.linalg.norm(left_eye_center - right_eye_center)
    if iod < 1e-6:
        return 0.0

    normalized_brow_dist = avg_vert_dist / iod
    return float(normalized_brow_dist)


def extract_eyebrow_temporal_features(
    brow_dist_series: List[float],
    fps: float,
) -> Dict[str, float]:
    """
    Computes statistical features of eyebrow distance series:
    - brow_dist_mean
    - brow_dist_std
    - brow_dist_min
    - brow_movement_energy (sum of absolute frame-to-frame changes)
    """
    if not brow_dist_series:
        return {
            "brow_dist_mean": 0.0,
            "brow_dist_std": 0.0,
            "brow_dist_min": 0.0,
            "brow_movement_energy": 0.0,
        }

    series = np.array(brow_dist_series, dtype=np.float32)

    mean_dist = float(np.mean(series))
    std_dist = float(np.std(series))
    min_dist = float(np.min(series))

    # Frame-to-frame velocity / movement energy
    diffs = np.diff(series)
    movement_energy = float(np.sum(np.abs(diffs))) if len(diffs) > 0 else 0.0

    return {
        "brow_dist_mean": mean_dist,
        "brow_dist_std": std_dist,
        "brow_dist_min": min_dist,
        "brow_movement_energy": movement_energy,
    }

