"""
Real-Time Driver Drowsiness Detection System.
Processes live webcam feed or offline video with MediaPipe Face Mesh, adaptive calibration,
live hybrid feature extraction, and multi-class soft-voting ensemble prediction.
Includes audio-visual alert mechanism for driver safety.
Author: Archee Jaiswal
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple
import cv2
import joblib
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.features.calibration import SubjectCalibrator
from src.features.eye_features import compute_frame_ear
from src.features.eyebrow_features import compute_eyebrow_distance
from src.features.feature_pipeline import HybridFeatureExtractor
from src.features.head_pose import HeadPoseEstimator
from src.features.mouth_features import compute_mar
from src.models.ensemble import DrowsinessEnsemble
from src.preprocessing.landmark_extractor import LandmarkExtractor
from src.utils.config import (
    ALL_FEATURE_NAMES,
    CLASSES,
    DEFAULT_FPS,
    SAVED_MODELS_DIR,
    WINDOW_SIZE_SEC,
)


class RealtimeDrowsinessDetector:
    """
    Real-time vision & ML pipeline for monitoring driver drowsiness.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        camera_id: int = 0,
        fps: int = DEFAULT_FPS,
        window_size_sec: float = WINDOW_SIZE_SEC,
    ):
        self.camera_id = camera_id
        self.fps = fps
        self.window_frames = int(fps * window_size_sec)

        # Initialize vision extractors
        self.landmark_extractor = LandmarkExtractor(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose_estimator = HeadPoseEstimator()
        self.calibrator = SubjectCalibrator()
        self.feature_extractor = HybridFeatureExtractor(fps=fps, calibrator=self.calibrator)

        # Load trained ensemble model
        if model_path is None:
            model_path = SAVED_MODELS_DIR / "drowsiness_ensemble.joblib"

        self.model: Optional[DrowsinessEnsemble] = None
        if Path(model_path).exists():
            print(f"[Detector] Loading trained ensemble from {model_path}...")
            self.model = joblib.load(model_path)
        else:
            print(f"[Detector Warning] Model file not found at {model_path}. Will run in metric monitor mode.")

        # Temporal Rolling Buffers
        self.buf_ear: Deque[float] = deque(maxlen=self.window_frames)
        self.buf_mar: Deque[float] = deque(maxlen=self.window_frames)
        self.buf_pitch: Deque[float] = deque(maxlen=self.window_frames)
        self.buf_yaw: Deque[float] = deque(maxlen=self.window_frames)
        self.buf_roll: Deque[float] = deque(maxlen=self.window_frames)
        self.buf_brow: Deque[float] = deque(maxlen=self.window_frames)

        # Eye closure & Blink tracking
        self.consecutive_closed_frames = 0
        self.consecutive_yawn_frames = 0
        self.blink_count = 0
        self.nod_count = 0

        # Calibration state
        self.calibrating = False
        self.calibration_samples = {"ear": [], "mar": [], "brow": [], "pitch": [], "yaw": [], "roll": []}
        self.calib_start_time = 0.0
        self.calib_duration_sec = 3.0

        # Alert / Beep state
        self.last_alert_time = 0.0

    def start_calibration(self):
        """Initiates per-subject baseline capture."""
        self.calibrating = True
        self.calibration_samples = {"ear": [], "mar": [], "brow": [], "pitch": [], "yaw": [], "roll": []}
        self.consecutive_closed_frames = 0
        self.consecutive_yawn_frames = 0
        self.calib_start_time = time.time()
        print("[Calibration] Capturing neutral baseline for 3 seconds... Keep face neutral.")

    def update_calibration(self, ear: float, mar: float, brow: float, pitch: float, yaw: float, roll: float):
        """Accumulates samples during calibration window."""
        if ear > 0.15:  # Only capture when eyes are naturally open (avoid blinks)
            self.calibration_samples["ear"].append(ear)
            self.calibration_samples["mar"].append(mar)
            self.calibration_samples["brow"].append(brow)
            self.calibration_samples["pitch"].append(pitch)
            self.calibration_samples["yaw"].append(yaw)
            self.calibration_samples["roll"].append(roll)

        elapsed = time.time() - self.calib_start_time
        num_samples = len(self.calibration_samples["ear"])
        if (elapsed >= self.calib_duration_sec or num_samples >= int(self.fps * 2.5)) and num_samples >= 20:
            self.calibrator.calibrate(
                ear_samples=self.calibration_samples["ear"],
                mar_samples=self.calibration_samples["mar"],
                brow_samples=self.calibration_samples["brow"],
                pitch_samples=self.calibration_samples["pitch"],
                yaw_samples=self.calibration_samples["yaw"],
                roll_samples=self.calibration_samples["roll"],
            )
            self.calibrating = False
            self.consecutive_closed_frames = 0
            self.consecutive_yawn_frames = 0
            print("[Calibration] Complete! Baselines and adaptive thresholds active:")
            summary = self.calibrator.get_calibration_summary()
            print(f"  EAR Baseline: {summary['ear_baseline']:.3f} | Eye Closure Threshold: {summary['adaptive_ear_threshold']:.3f}")
            print(f"  MAR Baseline: {summary['mar_baseline']:.3f} | Yawn Threshold: {summary['adaptive_mar_threshold']:.3f}")

    def trigger_drowsiness_alert(self):
        """Issues audio warning chime upon persistent drowsiness."""
        curr_time = time.time()
        if curr_time - self.last_alert_time > 2.0:
            sys.stdout.write("\a")  # ASCII Bell
            sys.stdout.flush()
            # Cross-platform sound notification
            if sys.platform == "darwin":
                os.system("afplay /System/Library/Sounds/Ping.aiff 2>/dev/null &")
            elif sys.platform == "win32":
                try:
                    import winsound
                    winsound.Beep(1000, 400)
                except Exception:
                    pass
            self.last_alert_time = curr_time

    def draw_hud(
        self,
        frame: np.ndarray,
        pred_class: int,
        probs: np.ndarray,
        ear: float,
        mar: float,
        pitch: float,
        perclos: float,
        fps_display: float,
    ) -> np.ndarray:
        """Renders rich HUD dashboard over video frame."""
        h, w, _ = frame.shape
        overlay = frame.copy()

        # Top banner background
        cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
        # Left telemetry panel
        cv2.rectangle(overlay, (15, 95), (280, 320), (15, 15, 15), -1)
        # Probability panel
        cv2.rectangle(overlay, (w - 260, 95), (w - 15, 255), (15, 15, 15), -1)

        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Class Color coding
        colors = {
            0: (0, 220, 0),     # Green - Alert
            1: (0, 165, 255),   # Orange - Low Vigilant
            2: (0, 0, 255),     # Red - Drowsy
        }
        status_color = colors.get(pred_class, (200, 200, 200))
        status_text = CLASSES.get(pred_class, "ANALYZING...")

        if self.calibrating:
            rem = max(0.0, self.calib_duration_sec - (time.time() - self.calib_start_time))
            status_text = f"CALIBRATING... ({rem:.1f}s)"
            status_color = (255, 255, 0)

        # Title & Status badge
        cv2.putText(frame, "DRIVER DROWSINESS MONITOR", (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"STATUS: {status_text}", (20, 68), cv2.FONT_HERSHEY_DUPLEX, 0.9, status_color, 2)
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (w - 120, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # Left Telemetry Metrics
        cv2.putText(frame, "PHYSIOLOGICAL TELEMETRY", (25, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

        ear_thresh = self.calibrator.adaptive_ear_threshold
        mar_thresh = self.calibrator.adaptive_mar_threshold

        ear_col = (0, 255, 0) if ear >= ear_thresh else (0, 0, 255)
        mar_col = (0, 0, 255) if mar >= mar_thresh else (0, 255, 0)
        perclos_col = (0, 0, 255) if perclos > 0.20 else (0, 255, 0)

        cv2.putText(frame, f"EAR: {ear:.3f} (th: {ear_thresh:.2f})", (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.52, ear_col, 1)
        cv2.putText(frame, f"MAR: {mar:.3f} (th: {mar_thresh:.2f})", (25, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.52, mar_col, 1)
        cv2.putText(frame, f"Pitch: {pitch:.1f} deg", (25, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
        cv2.putText(frame, f"PERCLOS: {perclos*100:.1f}%", (25, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.52, perclos_col, 1)
        cv2.putText(frame, f"Blinks: {self.blink_count}", (25, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1)
        cv2.putText(frame, f"Calibrated: {'YES' if self.calibrator.is_calibrated else 'NO (Press C)'}", (25, 298), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        # Soft Voting Probabilities
        cv2.putText(frame, "ENSEMBLE CONFIDENCE", (w - 245, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1)
        for i, name in enumerate(["Alert", "Low Vig.", "Drowsy"]):
            prob_val = probs[i] if probs is not None and len(probs) > i else 0.0
            y_bar = 150 + i * 35

            cv2.putText(frame, f"{name}: {prob_val*100:.0f}%", (w - 245, y_bar), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)
            # Bar graph
            cv2.rectangle(frame, (w - 130, y_bar - 12), (w - 130 + int(100 * prob_val), y_bar), colors[i], -1)
            cv2.rectangle(frame, (w - 130, y_bar - 12), (w - 30, y_bar), (80, 80, 80), 1)

        # Bottom Instructions
        cv2.putText(frame, "[C] Calibrate Neutral Baseline | [Q] Quit", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Drowsy warning banner if class is 2
        if pred_class == 2:
            cv2.rectangle(frame, (0, h - 70), (w, h - 35), (0, 0, 220), -1)
            cv2.putText(frame, "WARNING: DROWSINESS DETECTED! PULL OVER SAFELY!", (w // 2 - 270, h - 46), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)

        return frame

    def run(
        self,
        video_source=None,
        display: bool = True,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ):
        """Runs the real-time processing loop."""
        src = video_source if video_source is not None else self.camera_id
        cap = cv2.VideoCapture(src)

        if not cap.isOpened() and video_source is None:
            # Try camera index 1 (common for Mac internal vs external/continuity camera)
            print(f"[Detector] Could not open camera {src}, attempting camera index 1...")
            cap = cv2.VideoCapture(1)
            if cap.isOpened():
                src = 1

        if not cap.isOpened():
            print(f"[Error] Could not open video source: {src}")
            print("  TIP: Ensure camera permission is enabled in System Settings -> Privacy & Security -> Camera for Terminal/IDE.")
            print("  Or test with an offline video: python app/realtime_detector.py --video path/to/sample.mp4")
            return

        # Prepare video writer if output_path is provided
        writer = None
        if output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            source_fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, source_fps, (width, height))
            print(f"[Detector] Saving annotated video output to {output_path}")

        print("\n=== Driver Drowsiness Detection Active ===")
        print("  Controls: Press 'c' to calibrate neutral baseline | Press 'q' to exit")

        # Auto-calibrate on start
        self.start_calibration()

        prev_time = time.time()
        fps_display = self.fps
        pred_class = 0
        probs = np.array([0.9, 0.08, 0.02])
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps_display = 0.9 * fps_display + 0.1 * (1.0 / dt)

            # 1. Extract landmarks
            lm_data = self.landmark_extractor.extract_landmarks(frame)

            ear, mar, pitch, yaw, roll, brow = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            if lm_data is not None:
                # Compute instantaneous metrics
                _, _, ear = compute_frame_ear(lm_data["left_eye"], lm_data["right_eye"])
                mar = compute_mar(lm_data["mouth"])
                brow = compute_eyebrow_distance(
                    lm_data["left_eyebrow"],
                    lm_data["right_eyebrow"],
                    lm_data["left_eye"],
                    lm_data["right_eye"],
                )
                pitch, yaw, roll = self.pose_estimator.estimate_pose(
                    lm_data["head_pose_points"],
                    frame_width=lm_data["frame_width"],
                    frame_height=lm_data["frame_height"],
                )

                if self.calibrating:
                    self.update_calibration(ear, mar, brow, pitch, yaw, roll)
                else:
                    ear_thresh = self.calibrator.adaptive_ear_threshold
                    mar_thresh = self.calibrator.adaptive_mar_threshold

                    # Track eye closure duration in frames
                    if ear < ear_thresh:
                        self.consecutive_closed_frames += 1
                    else:
                        # Detect completed normal blink (duration between 2 and max ~12 frames / ~400ms)
                        if 2 <= self.consecutive_closed_frames <= int(self.fps * 0.40):
                            self.blink_count += 1
                        self.consecutive_closed_frames = 0

                    # Track yawn duration in frames
                    if mar >= mar_thresh:
                        self.consecutive_yawn_frames += 1
                    else:
                        self.consecutive_yawn_frames = 0

                    # Store in rolling buffers
                    self.buf_ear.append(ear)
                    self.buf_mar.append(mar)
                    self.buf_pitch.append(pitch)
                    self.buf_yaw.append(yaw)
                    self.buf_roll.append(roll)
                    self.buf_brow.append(brow)

                    # Perform classification once buffer has at least 2.0 seconds of frames
                    if self.model is not None and len(self.buf_ear) >= int(self.fps * 2.0):
                        feat_dict = self.feature_extractor.process_window(
                            list(self.buf_ear),
                            list(self.buf_mar),
                            list(self.buf_pitch),
                            list(self.buf_yaw),
                            list(self.buf_roll),
                            list(self.buf_brow),
                        )
                        feat_vec = np.array([feat_dict[k] for k in ALL_FEATURE_NAMES])
                        raw_class, raw_probs, _ = self.model.predict_single(feat_vec)
                        raw_probs = np.array(raw_probs, dtype=np.float32)

                        current_perclos = float(np.mean([e < ear_thresh for e in self.buf_ear]))
                        ear_norm = ear / max(self.calibrator.ear_baseline, 1e-4)

                        # Physiological condition flags:
                        # Micro-sleep: eyes continuously shut for >= 2.0 seconds
                        is_microsleep = self.consecutive_closed_frames >= int(self.fps * 2.0)
                        # Sluggish eye closure: eyes closed between 0.5s and 2.0s (slow blink / heavy fatigue)
                        is_sluggish_closure = int(self.fps * 0.50) <= self.consecutive_closed_frames < int(self.fps * 2.0)
                        # Normal quick blink (100-400ms)
                        is_momentary_blink = 0 < self.consecutive_closed_frames < int(self.fps * 0.50)
                        # Yawning: mouth opened wide for >= 1.2 seconds continuously
                        is_sustained_yawn = self.consecutive_yawn_frames >= int(self.fps * 1.2)
                        # Droopy / heavy eyelids (visibly half-closed while not in a blink)
                        is_droopy_eyes = (ear_norm < 0.75 and self.consecutive_closed_frames == 0)

                        # Three-tier classification logic:
                        if is_microsleep or current_perclos >= 0.32:
                            # Level 2 (Drowsy): Micro-sleep (>= 2.0s closed) or severe PERCLOS
                            raw_probs[2] = max(raw_probs[2], 0.85)
                            raw_probs[0] = min(raw_probs[0], 0.05)
                        elif is_sustained_yawn or is_sluggish_closure or is_droopy_eyes or (0.12 <= current_perclos < 0.30):
                            # Level 1 (Low Vigilant): Heavy eyes, sluggish closure (0.5-2.0s), yawning, or elevated PERCLOS
                            raw_probs[1] = max(raw_probs[1], 0.65)
                            raw_probs[2] = min(raw_probs[2], 0.15)
                        elif is_momentary_blink:
                            # Normal quick blink: maintain Alert state, prevent false Drowsy
                            if current_perclos < 0.10:
                                raw_probs[0] = max(raw_probs[0], 0.75)
                                raw_probs[2] = min(raw_probs[2], 0.05)
                        elif ear_norm >= 0.78 and current_perclos < 0.10 and not is_sustained_yawn:
                            # Level 0 (Alert): Normal open eyes, awake, looking at screen
                            raw_probs[0] = max(raw_probs[0], 0.80)
                            raw_probs[1] = min(raw_probs[1], 0.15)
                            raw_probs[2] = min(raw_probs[2], 0.05)

                        # Normalize to valid probability distribution
                        raw_probs = np.maximum(raw_probs, 0.01)
                        raw_probs /= np.sum(raw_probs)

                        # Smooth probabilities using Exponential Moving Average
                        probs = 0.75 * probs + 0.25 * raw_probs
                        pred_class = int(np.argmax(probs))

                        # Audio alarm triggers ONLY for Level 2 severe drowsiness (micro-sleep >= 2.0s or high-confidence drowsy)
                        if is_microsleep or (pred_class == 2 and probs[2] > 0.65):
                            self.trigger_drowsiness_alert()

            # PERCLOS for HUD
            thresh = self.calibrator.adaptive_ear_threshold
            perclos = float(np.mean([e < thresh for e in self.buf_ear])) if self.buf_ear else 0.0

            # Render HUD
            annotated = self.draw_hud(frame, pred_class, probs, ear, mar, pitch, perclos, fps_display)

            if writer is not None:
                writer.write(annotated)

            if display:
                cv2.imshow("Multi-Level Driver Drowsiness Monitor", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key == ord("c"):
                    self.start_calibration()

        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        self.landmark_extractor.close()
        print(f"[Detector] Finished processing {frame_idx} frames successfully.")


def main():
    parser = argparse.ArgumentParser(description="Multi-Level Driver Drowsiness Detection Live Demo")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file for offline testing")
    parser.add_argument("--model", type=str, default=None, help="Path to trained .joblib ensemble model")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated output video (.mp4)")
    parser.add_argument("--no-display", action="store_true", help="Run without opening GUI display window")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process before exiting")
    args = parser.parse_args()

    detector = RealtimeDrowsinessDetector(model_path=args.model, camera_id=args.camera)
    detector.run(
        video_source=args.video,
        display=not args.no_display,
        output_path=args.output,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()

