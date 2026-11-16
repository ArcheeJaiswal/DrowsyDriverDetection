"""
Unified Feature Extraction Pipeline.
Aggregates temporal frame sequences into multi-modal hybrid feature vectors.
Author: Archee Jaiswal
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.features.calibration import SubjectCalibrator
from src.features.eye_features import extract_eye_temporal_features
from src.features.eyebrow_features import extract_eyebrow_temporal_features
from src.features.head_pose import extract_head_pose_temporal_features
from src.features.mouth_features import extract_mouth_temporal_features
from src.utils.config import ALL_FEATURE_NAMES, DEFAULT_FPS


class HybridFeatureExtractor:
    """
    Computes calibrated, multi-modal temporal features across video windows.
    Outputs the standardized feature vector for downstream classifiers.
    """

    def __init__(
        self,
        fps: float = DEFAULT_FPS,
        calibrator: Optional[SubjectCalibrator] = None,
    ):
        self.fps = fps
        self.calibrator = calibrator or SubjectCalibrator()

    def process_window(
        self,
        ear_series: List[float],
        mar_series: List[float],
        pitch_series: List[float],
        yaw_series: List[float],
        roll_series: List[float],
        brow_series: List[float],
    ) -> Dict[str, float]:
        """
        Takes raw metric series for a temporal segment (e.g. 5 seconds)
        and extracts the full set of normalized hybrid features.
        """
        ear_thresh = self.calibrator.adaptive_ear_threshold
        mar_thresh = self.calibrator.adaptive_mar_threshold

        # 1. Eye features
        eye_feats = extract_eye_temporal_features(
            ear_series, fps=self.fps, ear_threshold=ear_thresh
        )

        # Normalize EAR metrics by baseline (calibrated or default 0.28)
        base = max(self.calibrator.ear_baseline, 1e-4)
        eye_feats["ear_mean"] /= base
        eye_feats["ear_min"] /= base
        eye_feats["ear_max"] /= base

        # 2. Mouth features
        mouth_feats = extract_mouth_temporal_features(
            mar_series, fps=self.fps, mar_threshold=mar_thresh
        )

        # Normalize MAR metrics by baseline (calibrated or default 0.25)
        base_m = max(self.calibrator.mar_baseline, 1e-4)
        mouth_feats["mar_mean"] /= base_m
        mouth_feats["mar_max"] /= base_m

        # 3. Head pose features (relative to resting pose)
        pitch_adj = [p - self.calibrator.pitch_baseline for p in pitch_series]
        yaw_adj = [y - self.calibrator.yaw_baseline for y in yaw_series]
        roll_adj = [r - self.calibrator.roll_baseline for r in roll_series]

        pose_feats = extract_head_pose_temporal_features(
            pitch_adj, yaw_adj, roll_adj, fps=self.fps
        )

        # 4. Eyebrow features
        brow_feats = extract_eyebrow_temporal_features(
            brow_series, fps=self.fps
        )

        base_b = max(self.calibrator.brow_baseline, 1e-4)
        brow_feats["brow_dist_mean"] /= base_b
        brow_feats["brow_dist_min"] /= base_b

        # Combine all features
        features = {}
        features.update(eye_feats)
        features.update(mouth_feats)
        features.update(pose_feats)
        features.update(brow_feats)

        # Ensure all registered features exist
        for k in ALL_FEATURE_NAMES:
            if k not in features:
                features[k] = 0.0

        return features

    def to_dataframe(self, records: List[Dict[str, float]]) -> pd.DataFrame:
        """Converts list of feature dicts into a clean DataFrame aligned to ALL_FEATURE_NAMES."""
        df = pd.DataFrame(records)
        cols = [c for c in ALL_FEATURE_NAMES if c in df.columns]
        other_cols = [c for c in df.columns if c not in ALL_FEATURE_NAMES]
        return df[other_cols + cols]

