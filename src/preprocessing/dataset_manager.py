"""
Dataset Manager for UTA-RLDD (University of Texas at Arlington Real-Life Drowsiness Dataset).
Handles video discovery, clip segmentation, frame rate standardization, and manifest creation.
Author: Archee Jaiswal
"""

from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.config import (
    CLASSES,
    DEFAULT_FPS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    WINDOW_SIZE_SEC,
    WINDOW_STEP_SEC,
)


class UTARLDDManager:
    """
    Manages UTA-RLDD video dataset ingestion and clip segmentation.
    UTA-RLDD standard format:
        raw/
          ├── 01/
          │    ├── 0.mp4   (Alert - 0)
          │    ├── 5.mp4   (Low-vigilant - 1)
          │    └── 10.mp4  (Drowsy - 2)
          ├── 02/
          ...
    Also supports common variations (e.g., alert.mp4, drowsy.mp4, etc.).
    """

    @staticmethod
    def parse_label(name: str) -> Optional[int]:
        s = name.lower()
        # Check '10' / drowsy first (to avoid '0' substring false match)
        if s == "10" or s.startswith("10_") or s.startswith("10.") or "drowsy" in s or "sleepy" in s:
            return 2
        if s == "5" or s.startswith("5_") or s.startswith("5.") or "low_vigilant" in s or "lowvigilant" in s or "low" in s:
            return 1
        if s == "0" or s.startswith("0_") or s.startswith("0.") or "alert" in s or "normal" in s:
            return 0
        return None

    def __init__(self, raw_dir: Path = RAW_DATA_DIR, processed_dir: Path = PROCESSED_DATA_DIR):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def scan_dataset(self) -> pd.DataFrame:
        """
        Scans raw directory and builds an index of available subject recordings.
        """
        records = []
        video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

        for p in sorted(self.raw_dir.rglob("*")):
            if p.suffix.lower() in video_extensions:
                stem_lower = p.stem.lower()
                parent_name = p.parent.name

                label = self.parse_label(p.name) or self.parse_label(stem_lower)
                if label is None:
                    label = self.parse_label(parent_name)

                subject_id = parent_name if parent_name != self.raw_dir.name else p.stem.split("_")[0]

                records.append({
                    "subject_id": str(subject_id),
                    "video_path": str(p.resolve()),
                    "filename": p.name,
                    "label": label if label is not None else -1,
                    "class_name": CLASSES.get(label, "Unknown")
                })

        df = pd.DataFrame(records)
        return df

    def get_video_properties(self, video_path: str) -> Dict[str, float]:
        """Reads metadata of a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"fps": 0, "frame_count": 0, "duration_sec": 0, "width": 0, "height": 0}

        fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": duration,
            "width": width,
            "height": height
        }

    def extract_temporal_segments(
        self,
        video_path: str,
        target_fps: int = DEFAULT_FPS,
        segment_duration_sec: float = WINDOW_SIZE_SEC,
        step_duration_sec: float = WINDOW_STEP_SEC,
    ) -> Generator[Tuple[int, List[np.ndarray]], None, None]:
        """
        Reads a video, normalizes frame rate, and yields sliding window frame chunks.

        Yields:
            (segment_index, list_of_frames_in_segment)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0 or np.isnan(source_fps):
            source_fps = target_fps

        frame_interval = max(1, round(source_fps / target_fps))
        frames_per_window = int(target_fps * segment_duration_sec)
        frames_per_step = int(target_fps * step_duration_sec)

        buffer: List[np.ndarray] = []
        raw_idx = 0
        segment_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if raw_idx % frame_interval == 0:
                buffer.append(frame)

                if len(buffer) == frames_per_window:
                    yield segment_idx, list(buffer)
                    segment_idx += 1
                    # Slide buffer by step
                    buffer = buffer[frames_per_step:]

            raw_idx += 1

        cap.release()

