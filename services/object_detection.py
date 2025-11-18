"""Lightweight YOLO-based object detection helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
from ultralytics import YOLO


@dataclass
class Obstacle:
    label: str
    confidence: float
    position: str
    proximity: str


@dataclass
class DetectionInsight:
    summary: str
    movement_hint: str
    obstacles: List[Obstacle]


FrameArray = npt.NDArray[np.uint8]


class YOLODetector:
    """Wraps Ultralytics YOLO for quick obstacle insights."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.35,
    ) -> None:
        self.model_name = model_name
        self.confidence = confidence
        self._model: Optional[YOLO] = None

    def _ensure_model(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(self.model_name)
        return self._model

    def analyze(self, frame: FrameArray) -> Optional[DetectionInsight]:
        model = self._ensure_model()
        results = model.predict(
            source=frame,
            conf=self.confidence,
            device="cpu",
            verbose=False,
        )
        if not results:
            return None

        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.shape[0] == 0:
            return None

        width = int(frame.shape[1])
        height = int(frame.shape[0])
        obstacles: List[Obstacle] = []
        zone_weights: Dict[str, float] = {"left": 0.0, "center": 0.0, "right": 0.0}

        xyxy = _to_numpy(boxes.xyxy)
        class_ids = _to_numpy(boxes.cls)
        scores = _to_numpy(boxes.conf)

        for coords, cls_idx, conf in zip(xyxy, class_ids, scores):
            label = result.names.get(int(cls_idx), "object")
            x1, y1, x2, y2 = [float(val) for val in coords.tolist()]
            center_ratio = ((x1 + x2) / 2) / width
            height_ratio = (y2 - y1) / height
            position = _position_from_ratio(center_ratio)
            proximity = _proximity_from_ratio(height_ratio)
            obstacles.append(
                Obstacle(
                    label=label,
                    confidence=float(conf),
                    position=position,
                    proximity=proximity,
                )
            )
            zone_weights[position] += _zone_weight(proximity)

        if not obstacles:
            return None

        summary_parts = [
            f"{ob.label.title()} {ob.position} ({ob.proximity}, {ob.confidence:.0%})"
            for ob in obstacles[:5]
        ]
        summary = ", ".join(summary_parts)
        safe_zone = min(zone_weights, key=lambda zone: zone_weights[zone])
        movement_hint = _hint_from_zone(safe_zone)

        return DetectionInsight(
            summary=summary,
            movement_hint=movement_hint,
            obstacles=obstacles,
        )


def _position_from_ratio(center_ratio: float) -> str:
    if center_ratio < 0.33:
        return "left"
    if center_ratio > 0.66:
        return "right"
    return "center"


def _proximity_from_ratio(height_ratio: float) -> str:
    if height_ratio >= 0.55:
        return "very close"
    if height_ratio >= 0.35:
        return "close"
    if height_ratio >= 0.2:
        return "mid-range"
    return "far"


def _zone_weight(proximity: str) -> float:
    mapping = {
        "very close": 3.0,
        "close": 2.0,
        "mid-range": 1.0,
        "far": 0.5,
    }
    return mapping.get(proximity, 1.0)


def _hint_from_zone(zone: str) -> str:
    return {
        "left": "Move cautiously to the right.",
        "center": "Path ahead looks clearest; proceed straight.",
        "right": "Move cautiously to the left.",
    }.get(zone, "Proceed carefully and reassess.")


def _to_numpy(value: Any) -> npt.NDArray[np.float32]:
    if hasattr(value, "cpu") and callable(getattr(value, "cpu")):
        tensor = getattr(value, "cpu")()
        if hasattr(tensor, "numpy"):
            return tensor.numpy()
    return np.asarray(value)
