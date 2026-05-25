from __future__ import annotations

import numpy as np

from .config import GrabConfig
from .geometry import calc_angle, expand_box, point_in_box
from .models import DetectionBox, GrabDecision, PersonState


def evaluate_grab(
    pts: dict[int, tuple[int, int]],
    object_boxes: list[DetectionBox],
    frame_shape: tuple[int, int, int],
    grab_cfg: GrabConfig,
) -> GrabDecision:
    height, width = frame_shape[:2]

    for side in ("left", "right"):
        if side == "left":
            shoulder_id, elbow_id, wrist_id = 11, 13, 15
        else:
            shoulder_id, elbow_id, wrist_id = 12, 14, 16

        if not all(idx in pts for idx in (shoulder_id, elbow_id, wrist_id)):
            continue

        shoulder = np.array(pts[shoulder_id], dtype=np.float32)
        elbow = np.array(pts[elbow_id], dtype=np.float32)
        wrist = np.array(pts[wrist_id], dtype=np.float32)

        elbow_angle = calc_angle(shoulder, elbow, wrist)
        wrist_pt = tuple(map(int, wrist))

        for obj in object_boxes:
            check_box = expand_box(obj.xyxy, grab_cfg.interaction_box_scale, width, height)
            if point_in_box(wrist_pt, check_box) and elbow_angle <= grab_cfg.grab_elbow_angle_deg:
                return GrabDecision(
                    is_grab=True,
                    label=f"{side}_hand -> {obj.cls_name}",
                    object_box=obj.xyxy,
                    wrist_point=wrist_pt,
                )

    return GrabDecision(is_grab=False)


def update_person_state(state: PersonState, decision: GrabDecision) -> bool:
    if decision.is_grab:
        state.grab_frames += 1
        state.last_label = decision.label or "grab"
    else:
        state.grab_frames = 0
        state.last_label = "no_grab"
    return state.grab_frames > 0


def is_stable_grab(state: PersonState, grab_cfg: GrabConfig) -> bool:
    return state.grab_frames >= grab_cfg.stable_frames_for_grab
