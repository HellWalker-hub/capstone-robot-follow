"""
Drop-in PersonTracker: Hailo YOLO + standalone PyPI bytetracker.
Same .update(frame) -> Nx6 [x1,y1,x2,y2,id,conf] interface.
"""
import numpy as np
import torch
from bytetracker import BYTETracker

from perception.detector.hailo_yolo import HailoYoloDetector


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    return inter / (aa + bb - inter + 1e-6)


class PersonTracker:
    def __init__(self, model_path=None, conf=0.4, device=None,
                 track_buffer=90,
                 hef_path="/home/capstone/hailo_demo/models/yolov11n.hef"):
        self.detector = HailoYoloDetector(hef_path=hef_path, conf=conf)
        self._track_buffer = track_buffer
        self.tracker = BYTETracker(
            track_thresh=0.5,
            track_buffer=track_buffer,
            match_thresh=0.8,
            frame_rate=30,
        )
        print(f"[Tracker] Hailo YOLOv11n + ByteTracker (track_buffer={track_buffer})")

    def update(self, frame: np.ndarray) -> np.ndarray:
        dets = self.detector.detect(frame)
        if len(dets) == 0:
            try:
                self.tracker.update(torch.zeros((0, 6), dtype=torch.float32), None)
            except Exception:
                pass
            return np.empty((0, 6), dtype=np.float32)

        bt_in = np.zeros((len(dets), 6), dtype=np.float32)
        bt_in[:, :4] = dets[:, :4]
        bt_in[:, 4]  = dets[:, 4]
        bt_in[:, 5]  = 0

        online = self.tracker.update(torch.from_numpy(bt_in), None)
        if online is None or len(online) == 0:
            return np.empty((0, 6), dtype=np.float32)

        online = np.asarray(online, dtype=np.float32)
        out = np.zeros((len(online), 6), dtype=np.float32)
        out[:, 0:4] = online[:, 0:4]
        out[:, 4]   = online[:, 4]
        for i, trk in enumerate(out):
            best_iou, best_conf = 0.0, 1.0
            for d in dets:
                iou = _iou(trk[:4], d[:4])
                if iou > best_iou:
                    best_iou, best_conf = iou, float(d[4])
            out[i, 5] = best_conf
        return out

    def reset(self):
        self.tracker = BYTETracker(
            track_thresh=0.5,
            track_buffer=self._track_buffer,
            match_thresh=0.8,
            frame_rate=30,
        )
