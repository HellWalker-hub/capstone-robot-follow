import numpy as np
from boxmot import ByteTrack


class PersonTracker:
    """ByteTrack wrapper for multi-person tracking."""

    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        self.tracker = ByteTrack(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
        )

    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        Args:
            detections: Nx5 [x1, y1, x2, y2, conf]
            frame: BGR image (H, W, 3)
        Returns:
            Nx6 [x1, y1, x2, y2, track_id, conf]
        """
        if len(detections) == 0:
            return np.empty((0, 6), dtype=np.float32)
        tracks = self.tracker.update(detections, frame)
        if tracks is None or len(tracks) == 0:
            return np.empty((0, 6), dtype=np.float32)
        return tracks.astype(np.float32)

    def reset(self):
        self.tracker.reset()
