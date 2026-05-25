from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from .config import SELECTED_CONNECTIONS, SELECTED_POINTS, ThresholdConfig
from .geometry import to_global_point

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def build_pose_options(model_path: str, thresholds: ThresholdConfig) -> PoseLandmarkerOptions:
    return PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=thresholds.pose_det_conf,
        min_pose_presence_confidence=thresholds.pose_presence_conf,
        min_tracking_confidence=thresholds.pose_track_conf,
        output_segmentation_masks=False,
    )


class PoseEstimator:
    def __init__(self, pose_options: PoseLandmarkerOptions) -> None:
        self.pose_options = pose_options
        self.landmarker = None

    def __enter__(self):
        self.landmarker = PoseLandmarker.create_from_options(self.pose_options)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.landmarker is not None:
            self.landmarker.close()
        self.landmarker = None

    def detect_on_roi(self, roi_bgr: np.ndarray, timestamp_ms: int):
        if self.landmarker is None:
            raise RuntimeError("PoseEstimator 尚未進入 with 區塊")

        rgb_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_roi)
        return self.landmarker.detect_for_video(mp_image, timestamp_ms)


def draw_selected_pose(frame: np.ndarray, landmarks, roi_box: tuple[int, int, int, int]) -> dict[int, tuple[int, int]]:
    x1, y1, x2, y2 = roi_box
    roi_w = max(1, x2 - x1)
    roi_h = max(1, y2 - y1)

    pts: dict[int, tuple[int, int]] = {}
    for idx in SELECTED_POINTS:
        landmark = landmarks[idx]
        pts[idx] = to_global_point(landmark, x1, y1, roi_w, roi_h)

    for start, end in SELECTED_CONNECTIONS:
        if start in pts and end in pts:
            cv2.line(frame, pts[start], pts[end], (0, 255, 0), 2)

    for idx, (px, py) in pts.items():
        radius = 6 if idx in (15, 16) else 5
        color = (0, 0, 255) if idx in (15, 16) else (255, 255, 0)
        cv2.circle(frame, (px, py), radius, color, -1)
        cv2.putText(
            frame,
            SELECTED_POINTS[idx],
            (px + 4, py - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )
    return pts
