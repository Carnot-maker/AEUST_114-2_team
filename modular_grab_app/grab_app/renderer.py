from __future__ import annotations

import cv2
import numpy as np

from .models import DetectionBox, GrabDecision, PersonState


def draw_object_boxes(frame: np.ndarray, objects: list[DetectionBox]) -> None:
    for obj in objects:
        x1, y1, x2, y2 = obj.xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(
            frame,
            f"{obj.cls_name} {obj.conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 180, 0),
            2,
        )


def draw_person_box(frame: np.ndarray, person_box: DetectionBox) -> None:
    x1, y1, x2, y2 = person_box.xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 255, 0), 2)
    cv2.putText(
        frame,
        f"person {person_box.conf:.2f}",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 255, 0),
        2,
    )


def draw_grab_status(
    frame: np.ndarray,
    person_xyxy: tuple[int, int, int, int],
    person_state: PersonState,
    decision: GrabDecision,
    stable: bool,
) -> None:
    x1, y1, x2, y2 = person_xyxy
    height = frame.shape[0]
    text_y = min(height - 20, y2 + 22)

    if stable:
        cv2.putText(
            frame,
            f"GRAB: {person_state.last_label}",
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
        )
        if decision.object_box is not None:
            ox1, oy1, ox2, oy2 = decision.object_box
            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 0, 255), 3)
        if decision.wrist_point is not None:
            cv2.circle(frame, decision.wrist_point, 10, (0, 0, 255), 2)
    else:
        cv2.putText(
            frame,
            person_state.last_label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (50, 220, 255),
            2,
        )


def draw_top_status(frame: np.ndarray, frame_idx: int, frame_count: int, person_count: int, object_count: int, grab_total: int) -> None:
    cv2.putText(
        frame,
        f"Frame: {frame_idx}/{frame_count}  Person: {person_count}  Obj: {object_count}  Grab: {grab_total}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
    )
