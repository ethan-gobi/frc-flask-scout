from __future__ import annotations

from dataclasses import dataclass
from typing import List

try:
    from ultralytics import YOLO
except Exception:  # noqa: BLE001
    YOLO = None


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str


class RobotDetector:
    """YOLO-based detector with lightweight fallback behavior if YOLO is unavailable."""

    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        self.model_name = model_name
        self.model = None

        if YOLO is not None:
            try:
                self.model = YOLO(model_name)
            except Exception:  # noqa: BLE001
                self.model = None

    def detect(self, frame) -> List[Detection]:
        if self.model is None:
            return []

        results = self.model.predict(source=frame, verbose=False)
        detections: List[Detection] = []

        for result in results:
            boxes = result.boxes
            names = result.names
            if boxes is None:
                continue

            for box in boxes:
                cls_idx = int(box.cls[0])
                class_name = names.get(cls_idx, "unknown")
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(v) for v in coords]
                confidence = float(box.conf[0])
                detections.append(Detection(x1, y1, x2, y2, confidence, class_name))

        return detections


class VideoStream:
    def __init__(self) -> None:
        self.capture = None

    def open(self, stream_url: str) -> None:
        import cv2

        self.capture = cv2.VideoCapture(stream_url)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open stream: {stream_url}")

    def read(self):
        if self.capture is None:
            return False, None
        return self.capture.read()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
