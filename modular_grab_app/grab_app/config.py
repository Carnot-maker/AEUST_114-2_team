from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ProjectPaths:
    root: Path = Path(r"E:\Project_Collection\L3_YOLO_media")

    @property
    def yolo_model_path(self) -> Path:
        return self.root / "workspace" / "models" / "yolov8n.pt"

    @property
    def pose_model_path(self) -> Path:
        return self.root / "workspace" / "models" / "pose_landmarker_lite.task"

    @property
    def video_path(self) -> Path:
        return self.root / "workspace" / "data" / "videos" / "test.mp4"

    @property
    def output_path(self) -> Path:
        return self.root / "workspace" / "data" / "output" / "yolo_mp_grab_test.mp4"


@dataclass(slots=True)
class ThresholdConfig:
    person_conf: float = 0.40
    object_conf: float = 0.25
    pose_det_conf: float = 0.35
    pose_presence_conf: float = 0.35
    pose_track_conf: float = 0.35


@dataclass(slots=True)
class GrabConfig:
    interaction_box_scale: float = 1.20
    grab_elbow_angle_deg: float = 135.0
    stable_frames_for_grab: int = 3
    yolo_every_n_frames: int = 2
    max_persons: int = 8
    yolo_imgsz: int = 640
    person_class_id: int = 0
    target_object_names: set[str] = field(
        default_factory=lambda: {
            "book",
            "bottle",
            "cup",
            "cell phone",
            "remote",
            "mouse",
        }
    )


SELECTED_POINTS: dict[int, str] = {
    11: "L_shoulder",
    12: "R_shoulder",
    13: "L_elbow",
    14: "R_elbow",
    15: "L_wrist",
    16: "R_wrist",
    23: "L_hip",
    24: "R_hip",
}

SELECTED_CONNECTIONS: list[tuple[int, int]] = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
]
