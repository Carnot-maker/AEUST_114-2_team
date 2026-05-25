from __future__ import annotations

from typing import Iterable

from ultralytics import YOLO

from .config import GrabConfig, ThresholdConfig
from .models import DetectionBox


def load_yolo_model(model_path: str) -> YOLO:
    return YOLO(model_path)


class YoloDetector:
    def __init__(self, model: YOLO, thresholds: ThresholdConfig, grab_cfg: GrabConfig) -> None:
        self.model = model
        self.thresholds = thresholds
        self.grab_cfg = grab_cfg

    def predict(self, frame) -> tuple[list[DetectionBox], list[DetectionBox]]:
        result = self.model.predict(
            source=frame,
            verbose=False,
            conf=min(self.thresholds.person_conf, self.thresholds.object_conf),
            imgsz=self.grab_cfg.yolo_imgsz,
        )[0]
        return self._parse_result(result, self.model.names)

    def _parse_result(self, yolo_result, names: dict | Iterable) -> tuple[list[DetectionBox], list[DetectionBox]]:
        persons: list[DetectionBox] = []
        objects: list[DetectionBox] = []

        if yolo_result.boxes is None:
            return persons, objects

        for box in yolo_result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_name = str(names.get(cls_id, cls_id))

            det = DetectionBox((x1, y1, x2, y2), cls_id, cls_name, conf)

            if cls_id == self.grab_cfg.person_class_id and conf >= self.thresholds.person_conf:
                persons.append(det)
            elif cls_name in self.grab_cfg.target_object_names and conf >= self.thresholds.object_conf:
                objects.append(det)

        return persons, objects
