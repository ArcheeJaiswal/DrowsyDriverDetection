"""
Adaptive Per-Subject Calibration and Threshold Normalization.
Normalizes facial metrics against each individual's resting baseline to eliminate
anatomical biases (e.g. naturally narrow eyes, resting mouth position).
Author: Archee Jaiswal
"""

from typing import Dict, List, Optional
import numpy as np


class SubjectCalibrator:
    """
    Computes per-subject baseline parameters from neutral/alert frames
    and adaptively normalizes features and thresholds.
    """

    def __init__(
        self,
        ear_drop_ratio: float = 0.68,
        mar_elevation_factor: float = 1.75,
        default_ear_base: float = 0.26,
        default_mar_base: float = 0.25,
        default_brow_base: float = 0.35,
    ):
        self.ear_drop_ratio = ear_drop_ratio
        self.mar_elevation_factor = mar_elevation_factor

        self.ear_baseline: float = default_ear_base
        self.mar_baseline: float = default_mar_base
        self.brow_baseline: float = default_brow_base
        self.pitch_baseline: float = 0.0
        self.yaw_baseline: float = 0.0
        self.roll_baseline: float = 0.0

        self.is_calibrated: bool = False

    def calibrate(
        self,
        ear_samples: List[float],
        mar_samples: List[float],
        brow_samples: Optional[List[float]] = None,
        pitch_samples: Optional[List[float]] = None,
        yaw_samples: Optional[List[float]] = None,
        roll_samples: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Calibrates baselines using neutral/alert reference samples
        (e.g. first 3-5 seconds of recording).
        """
        if ear_samples and len(ear_samples) > 5:
            # Exclude blink dips (< 0.15) so we get genuine resting open-eye baseline
            open_eye_samples = [e for e in ear_samples if e > 0.15]
            if len(open_eye_samples) >= 5:
                self.ear_baseline = float(np.median(open_eye_samples))
            else:
                self.ear_baseline = float(np.median(ear_samples))

        if mar_samples and len(mar_samples) > 5:
            # Exclude yawning / speaking (> 0.45) for resting mouth baseline
            closed_mouth_samples = [m for m in mar_samples if m < 0.45]
            if len(closed_mouth_samples) >= 5:
                self.mar_baseline = float(np.median(closed_mouth_samples))
            else:
                self.mar_baseline = float(np.median(mar_samples))

        if brow_samples and len(brow_samples) > 5:
            self.brow_baseline = float(np.median(brow_samples))

        if pitch_samples and len(pitch_samples) > 5:
            self.pitch_baseline = float(np.median(pitch_samples))

        if yaw_samples and len(yaw_samples) > 5:
            self.yaw_baseline = float(np.median(yaw_samples))

        if roll_samples and len(roll_samples) > 5:
            self.roll_baseline = float(np.median(roll_samples))

        self.is_calibrated = True

        return self.get_calibration_summary()

    @property
    def adaptive_ear_threshold(self) -> float:
        """Personalized EAR threshold for eye-closure."""
        return max(0.12, self.ear_baseline * self.ear_drop_ratio)

    @property
    def adaptive_mar_threshold(self) -> float:
        """Personalized MAR threshold for yawning."""
        return max(0.40, self.mar_baseline * self.mar_elevation_factor)

    def normalize_frame(
        self,
        ear: float,
        mar: float,
        brow_dist: float,
        pitch: float,
        yaw: float,
        roll: float,
    ) -> Dict[str, float]:
        """
        Normalizes a single frame's raw metrics against the subject's baseline.
        """
        norm_ear = ear / max(self.ear_baseline, 1e-4)
        norm_mar = mar / max(self.mar_baseline, 1e-4)
        norm_brow = brow_dist / max(self.brow_baseline, 1e-4)
        delta_pitch = pitch - self.pitch_baseline
        delta_yaw = yaw - self.yaw_baseline
        delta_roll = roll - self.roll_baseline

        return {
            "norm_ear": float(norm_ear),
            "norm_mar": float(norm_mar),
            "norm_brow": float(norm_brow),
            "delta_pitch": float(delta_pitch),
            "delta_yaw": float(delta_yaw),
            "delta_roll": float(delta_roll),
        }

    def get_calibration_summary(self) -> Dict[str, float]:
        """Returns baseline parameters and active adaptive thresholds."""
        return {
            "ear_baseline": self.ear_baseline,
            "mar_baseline": self.mar_baseline,
            "brow_baseline": self.brow_baseline,
            "pitch_baseline": self.pitch_baseline,
            "yaw_baseline": self.yaw_baseline,
            "roll_baseline": self.roll_baseline,
            "adaptive_ear_threshold": self.adaptive_ear_threshold,
            "adaptive_mar_threshold": self.adaptive_mar_threshold,
        }

