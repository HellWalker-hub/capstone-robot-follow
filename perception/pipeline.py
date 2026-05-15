import numpy as np
import cv2
from enum import Enum

from perception.detector.yolo_detector import PersonDetector
from perception.tracker.bytetrack_wrapper import PersonTracker
from perception.reid.osnet_reid import OSNetReID
from perception.occlusion.cmoh import CMOH


class RPFState(Enum):
    IDLE = "idle"
    IDENTIFICATION = "identification"
    FOLLOWING = "following"
    SUSPENDED = "suspended"
    REIDENTIFICATION = "reidentification"


class FollowPipeline:
    """
    Full perception pipeline: detect → track → ReID → occlusion recovery.
    State machine: IDLE → IDENTIFICATION → FOLLOWING ↔ SUSPENDED/REIDENTIFICATION
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.detector = PersonDetector(
            conf=cfg.get("det_conf", 0.4),
        )
        self.tracker = PersonTracker(
            track_thresh=cfg.get("track_thresh", 0.5),
            track_buffer=cfg.get("track_buffer", 30),
        )
        self.reid = OSNetReID(
            model_name=cfg.get("reid_model", "osnet_x0_25"),
        )
        self.cmoh = CMOH(
            k=cfg.get("cmoh_k", 10),
            sim_threshold=cfg.get("reid_threshold", 0.55),
        )

        self.state = RPFState.IDLE
        self.target_id: int | None = None
        self.target_bbox: np.ndarray | None = None

        # frames without target before suspend
        self._lost_frames = 0
        self._lost_threshold = cfg.get("lost_threshold", 15)

        # confirmation frames for re-id
        self._reid_confirm_count = 0
        self._reid_confirm_needed = cfg.get("reid_confirm_frames", 5)
        self._reid_candidate_id: int | None = None

    def register_target(self, frame: np.ndarray, bbox: np.ndarray):
        """Call once to set the person to follow (click or auto-select)."""
        embedding = self.reid.extract(frame, bbox)
        # use a synthetic track_id=0 for registration
        self.cmoh.update(0, embedding)
        self.target_id = None  # will be assigned on first match
        self._initial_embedding = embedding
        self.state = RPFState.IDENTIFICATION
        print("[Pipeline] Target registered. Waiting for tracker assignment.")

    def process(self, frame: np.ndarray) -> dict:
        """
        Run one frame through the pipeline.
        Returns result dict with state, target_bbox, all_tracks, debug_info.
        """
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame)

        result = {
            "state": self.state,
            "target_id": self.target_id,
            "target_bbox": None,
            "all_tracks": tracks,
        }

        if self.state == RPFState.IDLE:
            return result

        if len(tracks) == 0:
            self._handle_no_tracks()
            result["state"] = self.state
            return result

        # extract embeddings for all current tracks
        track_embeddings = {}
        for track in tracks:
            tid = int(track[4])
            emb = self.reid.extract(frame, track[:4])
            track_embeddings[tid] = emb

        if self.state == RPFState.IDENTIFICATION:
            self._identify(frame, tracks, track_embeddings)

        elif self.state == RPFState.FOLLOWING:
            self._follow(frame, tracks, track_embeddings)

        elif self.state in (RPFState.SUSPENDED, RPFState.REIDENTIFICATION):
            self._reidentify(frame, tracks, track_embeddings)

        # update result
        result["state"] = self.state
        result["target_id"] = self.target_id
        if self.target_id is not None:
            for track in tracks:
                if int(track[4]) == self.target_id:
                    result["target_bbox"] = track[:4].copy()
                    break

        return result

    def _identify(self, frame, tracks, embeddings):
        """Match initial embedding to a track to assign target_id."""
        candidate_ids = [int(t[4]) for t in tracks]
        for tid, emb in embeddings.items():
            sim = float(np.dot(emb, self._initial_embedding))
            if sim >= self.cmoh.sim_threshold:
                self.target_id = tid
                self.cmoh.register(tid)
                self.cmoh.update(tid, emb)
                self.state = RPFState.FOLLOWING
                self._lost_frames = 0
                print(f"[Pipeline] Target assigned to track {tid} (sim={sim:.2f})")
                return

    def _follow(self, frame, tracks, embeddings):
        """Update target embedding; detect if target lost."""
        current_ids = {int(t[4]) for t in tracks}
        if self.target_id in current_ids:
            self._lost_frames = 0
            self.cmoh.update(self.target_id, embeddings[self.target_id])
        else:
            self._lost_frames += 1
            if self._lost_frames >= self._lost_threshold:
                print(f"[Pipeline] Target lost for {self._lost_frames} frames → SUSPENDED")
                self.state = RPFState.SUSPENDED

    def _reidentify(self, frame, tracks, embeddings):
        """Try to match any current track to stored target memory."""
        candidate_ids = [int(t[4]) for t in tracks]
        best_id, best_sim = self.cmoh.match(
            self._get_target_embedding(), candidate_ids
        )
        if best_id is not None:
            if best_id == self._reid_candidate_id:
                self._reid_confirm_count += 1
            else:
                self._reid_candidate_id = best_id
                self._reid_confirm_count = 1

            if self._reid_confirm_count >= self._reid_confirm_needed:
                self.target_id = best_id
                self.state = RPFState.FOLLOWING
                self._lost_frames = 0
                self._reid_confirm_count = 0
                print(f"[Pipeline] Re-identified as track {best_id} (sim={best_sim:.2f})")
        else:
            self._reid_confirm_count = 0
            self._reid_candidate_id = None
            if self.state == RPFState.SUSPENDED:
                self.state = RPFState.REIDENTIFICATION

    def _handle_no_tracks(self):
        if self.state == RPFState.FOLLOWING:
            self._lost_frames += 1
            if self._lost_frames >= self._lost_threshold:
                self.state = RPFState.SUSPENDED

    def _get_target_embedding(self) -> np.ndarray:
        emb = self.cmoh.get_mean_embedding(self.target_id)
        if emb is None:
            return self._initial_embedding
        return emb
