from __future__ import annotations

from pathlib import Path

import cv2

from .config import GrabConfig, ProjectPaths, ThresholdConfig
from .geometry import clamp_box
from .grab_logic import evaluate_grab, is_stable_grab, update_person_state
from .models import DetectionBox, PersonState
from .pose_service import PoseEstimator, build_pose_options, draw_selected_pose
from .renderer import draw_grab_status, draw_object_boxes, draw_person_box, draw_top_status
from .yolo_service import YoloDetector, load_yolo_model


def validate_paths(paths: ProjectPaths) -> None:
    if not paths.yolo_model_path.exists():
        raise FileNotFoundError(f"找不到 YOLO 模型: {paths.yolo_model_path}")
    if not paths.pose_model_path.exists():
        raise FileNotFoundError(f"找不到 MediaPipe 模型: {paths.pose_model_path}")
    if not paths.video_path.exists():
        raise FileNotFoundError(f"找不到影片: {paths.video_path}")


class GrabVideoPipeline:
    def __init__(
        self,
        paths: ProjectPaths | None = None,
        thresholds: ThresholdConfig | None = None,
        grab_cfg: GrabConfig | None = None,
    ) -> None:
        self.paths = paths or ProjectPaths()
        self.thresholds = thresholds or ThresholdConfig()
        self.grab_cfg = grab_cfg or GrabConfig()

    def process(self) -> Path:
        validate_paths(self.paths)

        print("載入 YOLO 模型:", self.paths.yolo_model_path)
        print("載入 MediaPipe 模型:", self.paths.pose_model_path)
        print("輸入影片:", self.paths.video_path)
        print("輸出影片:", self.paths.output_path)

        yolo_model = load_yolo_model(str(self.paths.yolo_model_path))
        yolo_detector = YoloDetector(yolo_model, self.thresholds, self.grab_cfg)
        pose_options = build_pose_options(str(self.paths.pose_model_path), self.thresholds)

        cap = cv2.VideoCapture(str(self.paths.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟影片: {self.paths.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.paths.output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(self.paths.output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        print("FPS:", fps)
        print("解析度:", width, "x", height)
        print("總幀數:", frame_count)

        frame_idx = 0
        cached_persons: list[DetectionBox] = []
        cached_objects: list[DetectionBox] = []
        person_states = [PersonState() for _ in range(self.grab_cfg.max_persons)]

        try:
            with PoseEstimator(pose_options) as pose_estimator:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1

                    if frame_idx == 1 or frame_idx % self.grab_cfg.yolo_every_n_frames == 0:
                        cached_persons, cached_objects = yolo_detector.predict(frame.copy())

                    draw_object_boxes(frame, cached_objects)
                    grab_total = 0

                    for person_idx, person in enumerate(cached_persons):
                        if person_idx >= len(person_states):
                            break

                        x1, y1, x2, y2 = clamp_box(*person.xyxy, width, height)
                        roi = frame[y1:y2, x1:x2]
                        if roi.size == 0:
                            continue

                        draw_person_box(frame, person)
                        timestamp_ms = int((frame_idx / fps) * 1000)
                        pose_result = pose_estimator.detect_on_roi(roi, timestamp_ms)

                        if not pose_result.pose_landmarks:
                            person_states[person_idx].grab_frames = 0
                            person_states[person_idx].last_label = "no_pose"
                            continue

                        pts = draw_selected_pose(frame, pose_result.pose_landmarks[0], (x1, y1, x2, y2))
                        decision = evaluate_grab(pts, cached_objects, frame.shape, self.grab_cfg)
                        update_person_state(person_states[person_idx], decision)

                        stable = is_stable_grab(person_states[person_idx], self.grab_cfg)
                        if stable:
                            grab_total += 1

                        draw_grab_status(frame, (x1, y1, x2, y2), person_states[person_idx], decision, stable)

                    draw_top_status(frame, frame_idx, frame_count, len(cached_persons), len(cached_objects), grab_total)
                    writer.write(frame)

                    if frame_idx % 30 == 0:
                        print(f"已處理 {frame_idx}/{frame_count} 幀")
        finally:
            cap.release()
            writer.release()

        print("處理完成")
        print("輸出影片:", self.paths.output_path)
        return self.paths.output_path
