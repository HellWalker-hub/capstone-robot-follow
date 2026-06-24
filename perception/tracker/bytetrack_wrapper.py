import numpy as np
import yaml
import os
from ultralytics import YOLO


class PersonTracker:
    """
    Combined YOLOv8 detector + ByteTrack via ultralytics model.track().
    track_buffer controls how many frames a track survives without detection —
    higher values keep occluder IDs alive longer so they remain in the
    occluder exclusion set and can't bypass it by getting a new ID.
    """

    def __init__(self, model_path="yolov8n.pt", conf=0.4, device=None,
                 track_buffer=90):
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
        self._tracker_config = self._make_tracker_config(track_buffer)
        print(f"[Tracker] YOLOv8n + ByteTrack on {device} (track_buffer={track_buffer})")

    def _make_tracker_config(self, track_buffer: int) -> str:
        """Write a custom bytetrack config with the requested track_buffer."""
        config = {
            "tracker_type": "bytetrack",
            "track_high_thresh": 0.5,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.6,
            "track_buffer": track_buffer,
            "match_thresh": 0.8,
            "fuse_score": True,
        }
        path = os.path.join(os.path.dirname(__file__), "_bytetrack_custom.yaml")
        with open(path, "w") as f:
            yaml.dump(config, f)
        return path

    def update(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns Nx6 array: [x1, y1, x2, y2, track_id, conf]
        """
        results = self.model.track(
            frame,
            conf=self.conf,
            classes=[0],
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
        self.model.predictor = None
