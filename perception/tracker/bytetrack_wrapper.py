import numpy as np
import cv2
from ultralytics import YOLO


class PersonTracker:
    """
    Combined YOLOv8 detector + ByteTrack via ultralytics model.track().
    Replaces separate detector + boxmot tracker — avoids Python 3.13 compat issues.
    """

    def __init__(self, model_path="yolov8n.pt", conf=0.4, device=None):
        import torch
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        self.model = YOLO(model_path)
        self.conf = conf
        self._tracker_config = "bytetrack.yaml"
        print(f"[Tracker] YOLOv8n + ByteTrack on {device}")

    def update(self, frame: np.ndarray) -> np.ndarray:
        """
        Run detection + tracking on frame.
        Returns Nx6 array: [x1, y1, x2, y2, track_id, conf]
        """
        results = self.model.track(
            frame,
            conf=self.conf,
            classes=[0],  # person only
            tracker=self._tracker_config,
            persist=True,
            verbose=False,
            device=self.device,
        )[0]

        boxes = results.boxes
        if boxes is None or boxes.id is None:
            return np.empty((0, 6), dtype=np.float32)

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().reshape(-1, 1)
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)
        return np.hstack([xyxy, ids, conf]).astype(np.float32)

    def reset(self):
        self.model.predictor = None  # clears tracker state
