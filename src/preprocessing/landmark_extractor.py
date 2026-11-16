"""
Facial Landmark Extractor using MediaPipe Face Mesh.
Extracts 468/478 3D landmarks for eyes, mouth, eyebrows, and head pose anchors.
Author: Archee Jaiswal
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    import mediapipe as mp
    try:
        from mediapipe.python.solutions import face_mesh as mp_face_mesh
    except (ImportError, AttributeError):
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            mp_face_mesh = mp.solutions.face_mesh
        else:
            mp_face_mesh = None
    MEDIAPIPE_AVAILABLE = mp_face_mesh is not None
except Exception:
    MEDIAPIPE_AVAILABLE = False
    mp_face_mesh = None

from src.utils.config import (
    RIGHT_EYE_LANDMARKS,
    LEFT_EYE_LANDMARKS,
    MOUTH_LANDMARKS,
    LEFT_EYEBROW_LANDMARKS,
    RIGHT_EYEBROW_LANDMARKS,
    HEAD_POSE_LANDMARKS_INDICES,
)


class LandmarkExtractor:
    """Extracts facial landmarks from image frames using MediaPipe Face Mesh."""

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if not MEDIAPIPE_AVAILABLE or mp_face_mesh is None:
            raise ImportError(
                "MediaPipe Face Mesh is not available. Please ensure mediapipe>=0.10.9,<0.10.15 is installed."
            )

        self.mp_face_mesh = mp_face_mesh
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._init_face_mesh()

    def _init_face_mesh(self):
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def extract_landmarks(
        self, frame: np.ndarray
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Process a single BGR frame and return landmark coordinates.
        """
        if self.face_mesh is None:
            self._init_face_mesh()

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = self.face_mesh.process(rgb_frame)
        except Exception:
            self._init_face_mesh()
            results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        n_landmarks = len(face_landmarks.landmark)

        landmarks_2d = np.zeros((n_landmarks, 2), dtype=np.float32)
        landmarks_3d = np.zeros((n_landmarks, 3), dtype=np.float32)

        for i, lm in enumerate(face_landmarks.landmark):
            landmarks_2d[i] = [lm.x * w, lm.y * h]
            landmarks_3d[i] = [lm.x, lm.y, lm.z]

        left_eye_pts = landmarks_2d[LEFT_EYE_LANDMARKS]
        right_eye_pts = landmarks_2d[RIGHT_EYE_LANDMARKS]
        left_brow_pts = landmarks_2d[LEFT_EYEBROW_LANDMARKS]
        right_brow_pts = landmarks_2d[RIGHT_EYEBROW_LANDMARKS]
        head_pose_pts = landmarks_2d[HEAD_POSE_LANDMARKS_INDICES]

        mouth_pts = {
            k: landmarks_2d[idx] for k, idx in MOUTH_LANDMARKS.items()
        }

        return {
            "all_landmarks_2d": landmarks_2d,
            "all_landmarks_3d": landmarks_3d,
            "left_eye": left_eye_pts,
            "right_eye": right_eye_pts,
            "mouth": mouth_pts,
            "left_eyebrow": left_brow_pts,
            "right_eyebrow": right_brow_pts,
            "head_pose_points": head_pose_pts,
            "frame_width": w,
            "frame_height": h,
        }

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, "face_mesh") and self.face_mesh:
            self.face_mesh.close()

