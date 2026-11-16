"""
Head pose estimation (Pitch, Yaw, Roll) using 2D-3D correspondence and solvePnP.
Detects head nodding and postural instability associated with drowsiness.
Author: Archee Jaiswal
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class HeadPoseEstimator:
    """
    Estimates 3D head orientation (Pitch, Yaw, Roll) via Perspective-n-Point (solvePnP).
    """

    # Generic 3D facial model points (in mm) from literature
    # 1. Nose tip: (0.0, 0.0, 0.0)
    # 2. Chin: (0.0, -330.0, -65.0)
    # 3. Left eye outer corner: (-225.0, 170.0, -135.0)
    # 4. Right eye outer corner: (225.0, 170.0, -135.0)
    # 5. Left mouth corner: (-150.0, -150.0, -125.0)
    # 6. Right mouth corner: (150.0, -150.0, -125.0)
    MODEL_POINTS_3D = np.array(
        [
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye outer corner
            (225.0, 170.0, -135.0),      # Right eye outer corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0),     # Right mouth corner
        ],
        dtype=np.float64,
    )

    def estimate_pose(
        self,
        image_points_2d: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[float, float, float]:
        """
        Estimates Euler angles (Pitch, Yaw, Roll) in degrees.

        image_points_2d must be an array of shape (6, 2):
        [nose_tip, chin, left_eye_corner, right_eye_corner, left_mouth, right_mouth]
        """
        if image_points_2d is None or len(image_points_2d) < 6:
            return 0.0, 0.0, 0.0

        pts_2d = np.array(image_points_2d, dtype=np.float64)

        # Approximate camera matrix based on focal length
        focal_length = frame_width
        center = (frame_width / 2.0, frame_height / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # Assuming no lens distortion

        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS_3D,
            pts_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return 0.0, 0.0, 0.0

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Decompose rotation matrix into Euler angles
        # Using RQDecomp3x3 for robust Euler angle extraction
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        # angles: [pitch, yaw, roll] in degrees
        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        return pitch, yaw, roll


def extract_head_pose_temporal_features(
    pitch_series: List[float],
    yaw_series: List[float],
    roll_series: List[float],
    fps: float,
    nod_pitch_threshold: float = 12.0,
    min_nod_frames: int = 4,
) -> Dict[str, float]:
    """
    Computes summary features from temporal pitch, yaw, and roll series.
    Detects head nods (characteristic micro-sleep downward head dropping).
    """
    if not pitch_series:
        return {
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "pitch_min": 0.0,
            "pitch_max": 0.0,
            "yaw_mean": 0.0,
            "yaw_std": 0.0,
            "roll_mean": 0.0,
            "roll_std": 0.0,
            "nod_count": 0.0,
            "nod_rate": 0.0,
        }

    pitches = np.array(pitch_series, dtype=np.float32)
    yaws = np.array(yaw_series, dtype=np.float32)
    rolls = np.array(roll_series, dtype=np.float32)

    total_frames = len(pitches)
    duration_sec = total_frames / max(fps, 1.0)

    # Detect nodding: Pitch exceeds nod threshold downwards for min_nod_frames
    nod_count = 0
    in_nod = False
    consecutive_nod_frames = 0

    for p in pitches:
        if abs(p) > nod_pitch_threshold:
            consecutive_nod_frames += 1
            if consecutive_nod_frames >= min_nod_frames and not in_nod:
                nod_count += 1
                in_nod = True
        else:
            in_nod = False
            consecutive_nod_frames = 0

    nod_rate = (nod_count / duration_sec) * 60.0 if duration_sec > 0 else 0.0

    return {
        "pitch_mean": float(np.mean(pitches)),
        "pitch_std": float(np.std(pitches)),
        "pitch_min": float(np.min(pitches)),
        "pitch_max": float(np.max(pitches)),
        "yaw_mean": float(np.mean(yaws)),
        "yaw_std": float(np.std(yaws)),
        "roll_mean": float(np.mean(rolls)),
        "roll_std": float(np.std(rolls)),
        "nod_count": float(nod_count),
        "nod_rate": float(nod_rate),
    }

