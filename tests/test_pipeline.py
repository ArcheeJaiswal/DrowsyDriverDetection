"""
Unit and integration tests for Multi-Level Driver Drowsiness Detection System.
Tests landmark feature calculations, adaptive calibration, classifier ensemble, and evaluation metrics.
Author: Archee Jaiswal
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.calibration import SubjectCalibrator
from src.features.eye_features import compute_ear, compute_frame_ear, extract_eye_temporal_features
from src.features.eyebrow_features import compute_eyebrow_distance, extract_eyebrow_temporal_features
from src.features.feature_pipeline import HybridFeatureExtractor
from src.features.head_pose import HeadPoseEstimator, extract_head_pose_temporal_features
from src.features.mouth_features import compute_mar, extract_mouth_temporal_features
from src.models.classifiers import get_default_models
from src.models.ensemble import DrowsinessEnsemble
from src.evaluation.metrics import compute_metrics, compute_video_level_accuracy
from src.utils.config import ALL_FEATURE_NAMES, CLASSES
from src.utils.synthetic_data import generate_utarldd_benchmark_dataset


class TestEyeFeatures(unittest.TestCase):
    def test_compute_ear_open_eye(self):
        # Synthetic open eye landmarks (6 points)
        landmarks = np.array([
            [30.0, 50.0],  # p1: outer corner
            [40.0, 42.0],  # p2: upper 1
            [50.0, 42.0],  # p3: upper 2
            [60.0, 50.0],  # p4: inner corner
            [50.0, 58.0],  # p5: lower 2
            [40.0, 58.0],  # p6: lower 1
        ], dtype=np.float32)
        ear = compute_ear(landmarks)
        self.assertGreater(ear, 0.25)
        self.assertLess(ear, 0.60)

    def test_compute_ear_closed_eye(self):
        # Synthetic closed eye landmarks
        landmarks = np.array([
            [30.0, 50.0],
            [40.0, 49.5],
            [50.0, 49.5],
            [60.0, 50.0],
            [50.0, 50.5],
            [40.0, 50.5],
        ], dtype=np.float32)
        ear = compute_ear(landmarks)
        self.assertLess(ear, 0.10)

    def test_extract_eye_temporal_features(self):
        fps = 30.0
        ear_series = [0.32] * 20 + [0.12] * 5 + [0.32] * 25  # 1 blink
        feats = extract_eye_temporal_features(ear_series, fps=fps, ear_threshold=0.22)
        self.assertIn("ear_mean", feats)
        self.assertIn("blink_rate", feats)
        self.assertIn("perclos", feats)
        self.assertGreater(feats["perclos"], 0.05)


class TestMouthFeatures(unittest.TestCase):
    def test_compute_mar_neutral(self):
        mouth_dict = {
            "left_corner": np.array([80.0, 80.0]),
            "right_corner": np.array([20.0, 80.0]),
            "upper_lip_center": np.array([50.0, 72.0]),
            "lower_lip_center": np.array([50.0, 88.0]),
            "upper_lip_left": np.array([65.0, 74.0]),
            "lower_lip_left": np.array([65.0, 86.0]),
            "upper_lip_right": np.array([35.0, 74.0]),
            "lower_lip_right": np.array([35.0, 86.0]),
        }
        mar = compute_mar(mouth_dict)
        self.assertGreater(mar, 0.15)
        self.assertLess(mar, 0.45)

    def test_compute_mar_yawn(self):
        mouth_dict = {
            "left_corner": np.array([75.0, 80.0]),
            "right_corner": np.array([25.0, 80.0]),
            "upper_lip_center": np.array([50.0, 55.0]),
            "lower_lip_center": np.array([50.0, 105.0]),
            "upper_lip_left": np.array([65.0, 60.0]),
            "lower_lip_left": np.array([65.0, 100.0]),
            "upper_lip_right": np.array([35.0, 60.0]),
            "lower_lip_right": np.array([35.0, 100.0]),
        }
        mar = compute_mar(mouth_dict)
        self.assertGreater(mar, 0.60)


class TestHeadPoseAndEyebrows(unittest.TestCase):
    def test_head_pose_estimation(self):
        estimator = HeadPoseEstimator()
        # Mock 2D projection
        pts_2d = np.array([
            [320.0, 240.0],  # Nose tip
            [320.0, 360.0],  # Chin
            [260.0, 190.0],  # Left eye corner
            [380.0, 190.0],  # Right eye corner
            [280.0, 300.0],  # Left mouth corner
            [360.0, 300.0],  # Right mouth corner
        ], dtype=np.float64)
        pitch, yaw, roll = estimator.estimate_pose(pts_2d, frame_width=640, frame_height=480)
        self.assertIsInstance(pitch, float)
        self.assertIsInstance(yaw, float)
        self.assertIsInstance(roll, float)

    def test_eyebrow_distance(self):
        left_brow = np.array([[60.0, 30.0], [70.0, 28.0], [80.0, 30.0]])
        right_brow = np.array([[20.0, 30.0], [30.0, 28.0], [40.0, 30.0]])
        left_eye = np.array([[60.0, 50.0], [70.0, 50.0], [80.0, 50.0]])
        right_eye = np.array([[20.0, 50.0], [30.0, 50.0], [40.0, 50.0]])
        dist = compute_eyebrow_distance(left_brow, right_brow, left_eye, right_eye)
        self.assertGreater(dist, 0.1)


class TestCalibrationAndPipeline(unittest.TestCase):
    def test_subject_calibrator(self):
        calibrator = SubjectCalibrator()
        ear_samples = [0.30, 0.31, 0.29, 0.32, 0.30, 0.10, 0.31]  # with one blink
        mar_samples = [0.22, 0.23, 0.21, 0.22, 0.24]
        summary = calibrator.calibrate(ear_samples=ear_samples, mar_samples=mar_samples)
        self.assertTrue(calibrator.is_calibrated)
        self.assertAlmostEqual(summary["ear_baseline"], 0.305, places=2)
        self.assertGreater(summary["adaptive_ear_threshold"], 0.15)
        self.assertLess(summary["adaptive_ear_threshold"], 0.30)

    def test_hybrid_feature_extractor(self):
        calibrator = SubjectCalibrator()
        extractor = HybridFeatureExtractor(fps=30.0, calibrator=calibrator)
        ear_series = [0.30] * 60
        mar_series = [0.22] * 60
        pitch_series = [0.0] * 60
        yaw_series = [0.0] * 60
        roll_series = [0.0] * 60
        brow_series = [0.35] * 60
        feats = extractor.process_window(ear_series, mar_series, pitch_series, yaw_series, roll_series, brow_series)
        for col in ALL_FEATURE_NAMES:
            self.assertIn(col, feats, f"Missing expected feature: {col}")


class TestModelEnsemble(unittest.TestCase):
    def test_ensemble_fit_and_predict(self):
        np.random.seed(42)
        n_samples = 60
        n_features = len(ALL_FEATURE_NAMES)
        X = np.random.randn(n_samples, n_features)
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)

        models = get_default_models(random_state=42)
        rf = models["Random Forest"].fit(X, y)
        xgb_clf = models["XGBoost"].fit(X, y)
        mlp = models["MLP"].fit(X, y)

        ensemble = DrowsinessEnsemble(
            rf_model=rf,
            xgb_model=xgb_clf,
            mlp_model=mlp,
            scale_inputs=True,
        )
        ensemble.fit(X, y)

        preds = ensemble.predict(X)
        self.assertEqual(len(preds), n_samples)

        probs = ensemble.predict_proba(X)
        self.assertEqual(probs.shape, (n_samples, 3))
        np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

        single_pred, single_probs, single_name = ensemble.predict_single(X[0])
        self.assertIn(single_pred, [0, 1, 2])
        self.assertEqual(single_name, CLASSES[single_pred])


class TestMetrics(unittest.TestCase):
    def test_compute_metrics(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 2, 2, 2]
        res = compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(res["accuracy"], 5 / 6, places=3)
        self.assertIn("f1_macro", res)
        self.assertIn("precision_macro", res)
        self.assertIn("recall_macro", res)


class TestRealtimeDetector(unittest.TestCase):
    def test_detector_initialization(self):
        from app.realtime_detector import RealtimeDrowsinessDetector
        detector = RealtimeDrowsinessDetector()
        self.assertIsNotNone(detector.model)
        self.assertIsNotNone(detector.landmark_extractor)
        self.assertEqual(detector.blink_count, 0)
        self.assertEqual(detector.nod_count, 0)


if __name__ == "__main__":
    unittest.main()
