from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class DetectionBox:
    xyxy: tuple[int, int, int, int]
    cls_id: int
    cls_name: str
    conf: float


@dataclass(slots=True)
class PersonState:
    grab_frames: int = 0
    last_label: str = "no_grab"


@dataclass(slots=True)
class GrabDecision:
    is_grab: bool
    label: Optional[str] = None
    object_box: Optional[tuple[int, int, int, int]] = None
    wrist_point: Optional[tuple[int, int]] = None
