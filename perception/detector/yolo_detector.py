import torch
from ultralytics import YOLO
import numpy as np


class PersonDetector:
    """YOLOv8n person detector with M1 MPS or CPU fallback."""

    def __init__(self, model_path="yolov8n.pt", conf=0.4, device=None):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf = conf
        print(f"[Detector] YOLOv8n on {device}")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns Nx5 array: [x1, y1, x2, y2, conf] for person class only.
        """
        results = self.model(frame, conf=self.conf, classes=[0], verbose=False)[0]
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 5), dtype=np.float32)
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)
        return np.hstack([xyxy, conf]).astype(np.float32)
