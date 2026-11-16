"""
Mouth feature extraction: Mouth Aspect Ratio (MAR), Yawn Detection, Duration and Frequency.
Author: Archee Jaiswal
"""

from typing import Dict, List
import numpy as np


def compute_mar(mouth_dict: Dict[str, np.ndarray]) -> float:
    """
    Computes Mouth Aspect Ratio (MAR):
    Vertical opening: average of distances between upper and lower lip points
    Horizontal opening: distance between mouth corners (left and right)

    MAR = (|upper_center - lower_center| + |upper_l - lower_l| + |upper_r - lower_r|) / (2.0 * |corner_l - corner_r|)
    """
    if not mouth_dict:
        return 0.0

    corner_left = mouth_dict.get("left_corner")
    corner_right = mouth_dict.get("right_corner")
    lip_c_up = mouth_dict.get("upper_lip_center")
    lip_c_down = mouth_dict.get("lower_lip_center")
    lip_l_up = mouth_dict.get("upper_lip_left")
    lip_l_down = mouth_dict.get("lower_lip_left")
    lip_r_up = mouth_dict.get("upper_lip_right")
    lip_r_down = mouth_dict.get("lower_lip_right")

    if any(p is None for p in [corner_left, corner_right, lip_c_up, lip_c_down]):
        return 0.0

    d_horizontal = np.linalg.norm(corner_left - corner_right)
    if d_horizontal < 1e-6:
        return 0.0

    d_v_center = np.linalg.norm(lip_c_up - lip_c_down)

    d_v_left = (
        np.linalg.norm(lip_l_up - lip_l_down)
        if (lip_l_up is not None and lip_l_down is not None)
        else d_v_center
    )
    d_v_right = (
        np.linalg.norm(lip_r_up - lip_r_down)
        if (lip_r_up is not None and lip_r_down is not None)
        else d_v_center
    )

    mar = (d_v_center + d_v_left + d_v_right) / (2.0 * d_horizontal)
    return float(mar)


def extract_mouth_temporal_features(
    mar_series: List[float],
    fps: float,
    mar_threshold: float = 0.55,
    min_yawn_seconds: float = 1.5,
) -> Dict[str, float]:
    """
    Computes temporal mouth features over a sequence of MAR frames:
    - mar_mean, mar_std, mar_max
    - yawn_rate (yawns per minute)
    - mean_yawn_duration (sec)
    - yawn_duration_ratio (fraction of time spent yawning)
    """
    if not mar_series:
        return {
            "mar_mean": 0.0,
            "mar_std": 0.0,
            "mar_max": 0.0,
            "yawn_rate": 0.0,
            "mean_yawn_duration": 0.0,
            "yawn_duration_ratio": 0.0,
        }

    mars = np.array(mar_series, dtype=np.float32)
    total_frames = len(mars)
    duration_sec = total_frames / max(fps, 1.0)

    mar_mean = float(np.mean(mars))
    mar_std = float(np.std(mars))
    mar_max = float(np.max(mars))

    min_yawn_frames = int(min_yawn_seconds * fps)

    yawn_durations = []
    consecutive_yawn_frames = 0
    total_yawn_frames = 0

    for m in mars:
        if m > mar_threshold:
            consecutive_yawn_frames += 1
            total_yawn_frames += 1
        else:
            if consecutive_yawn_frames >= min_yawn_frames:
                yawn_durations.append(consecutive_yawn_frames / fps)
            consecutive_yawn_frames = 0

    if consecutive_yawn_frames >= min_yawn_frames:
        yawn_durations.append(consecutive_yawn_frames / fps)

    yawn_count = len(yawn_durations)
    yawn_rate = (yawn_count / duration_sec) * 60.0 if duration_sec > 0 else 0.0
    mean_yawn_dur = float(np.mean(yawn_durations)) if yawn_durations else 0.0
    yawn_duration_ratio = (
        float(total_yawn_frames / total_frames) if total_frames > 0 else 0.0
    )

    return {
        "mar_mean": mar_mean,
        "mar_std": mar_std,
        "mar_max": mar_max,
        "yawn_rate": float(yawn_rate),
        "mean_yawn_duration": mean_yawn_dur,
        "yawn_duration_ratio": yawn_duration_ratio,
    }

