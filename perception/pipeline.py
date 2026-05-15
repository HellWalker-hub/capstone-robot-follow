import numpy as np
import cv2
from enum import Enum

from perception.tracker.bytetrack_wrapper import PersonTracker
from perception.reid.osnet_reid import OSNetReID
from perception.occlusion.cmoh import CMOH


class RPFState(Enum):
    IDLE = "idle"
    REGISTERING = "registering"   # collecting multi-frame appearance profile
    IDENTIFICATION = "identification"
    FOLLOWING = "following"
    SUSPENDED = "suspended"
    REIDENTIFICATION = "reidentification"


class FollowPipeline:
    """
    Perception pipeline: YOLOv8+ByteTrack → ReID → CMOH occlusion memory.
    State machine: IDLE → IDENTIFICATION → FOLLOWING ↔ SUSPENDED/REIDENTIFICATION

    Occluder exclusion: track IDs continuously visible since target was lost
    are treated as occluders and excluded from re-id candidates. The real
    target can only re-appear as a new or briefly-absent track.
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.tracker = PersonTracker(
            conf=cfg.get("det_conf", 0.4),
        )
        self.reid = OSNetReID(
            model_name=cfg.get("reid_model", "mobilenet_v3_small"),
        )
        self.cmoh = CMOH(
            k=cfg.get("cmoh_k", 10),
            sim_threshold=cfg.get("reid_threshold", 0.60),
        )

        self.state = RPFState.IDLE
        self.target_id: int | None = None
        self._initial_embedding: np.ndarray | None = None

        self._lost_frames = 0
        self._lost_threshold = cfg.get("lost_threshold", 15)

        self._reid_confirm_count = 0
        self._reid_confirm_needed = cfg.get("reid_confirm_frames", 3)
        self._reid_candidate_id: int | None = None

        # occluder exclusion: IDs visible in every frame since suspension
        self._continuously_visible: set = set()

        # registration phase: accumulate N frames before following
        self._reg_frames_needed = cfg.get("registration_frames", 45)
        self._reg_frames_collected = 0
        self._reg_bbox: np.ndarray | None = None
        self.registration_progress: float = 0.0  # 0.0 → 1.0, exposed for UI

    def register_target(self, frame: np.ndarray, bbox: np.ndarray):
        """Begin multi-frame registration from clicked bounding box."""
        self._reg_bbox = bbox.copy()
        self._reg_frames_collected = 0
        self.registration_progress = 0.0
        self.cmoh.clear()
        self._initial_embedding = None
        self.target_id = None
        self._continuously_visible.clear()
        self.state = RPFState.REGISTERING
        print(f"[Pipeline] Registration started — hold pose for {self._reg_frames_needed} frames.")

    def process(self, frame: np.ndarray) -> dict:
        """Run one frame. Returns state, target_bbox, all_tracks, occluder_ids."""
        tracks = self.tracker.update(frame)

        result = {
            "state": self.state,
            "target_id": self.target_id,
            "target_bbox": None,
            "all_tracks": tracks,
            "occluder_ids": set(self._continuously_visible),
        }

        if self.state == RPFState.IDLE:
            return result

        if len(tracks) == 0:
            self._handle_no_tracks()
            # no tracks during suspension — clear occluder set so a returning
            # target isn't blocked by a stale occluder ID
            if self.state in (RPFState.SUSPENDED, RPFState.REIDENTIFICATION):
                self._continuously_visible.clear()
            result["state"] = self.state
            return result

        track_embeddings = {}
        for track in tracks:
            tid = int(track[4])
            emb = self.reid.extract(frame, track[:4])
            track_embeddings[tid] = emb

        if self.state == RPFState.REGISTERING:
            self._register(frame, tracks)
        elif self.state == RPFState.IDENTIFICATION:
            self._identify(track_embeddings)
        elif self.state == RPFState.FOLLOWING:
            self._follow(track_embeddings)
        elif self.state in (RPFState.SUSPENDED, RPFState.REIDENTIFICATION):
            self._update_occluders(track_embeddings)
            self._reidentify(track_embeddings)

        result["state"] = self.state
        result["target_id"] = self.target_id
        result["occluder_ids"] = set(self._continuously_visible)
        if self.target_id is not None:
            for track in tracks:
                if int(track[4]) == self.target_id:
                    result["target_bbox"] = track[:4].copy()
                    break

        return result

    def _register(self, frame: np.ndarray, tracks: np.ndarray):
        """
        Collect embeddings from the registered bbox each frame.
        Tracks the closest detected person to the clicked bbox centroid.
        After reg_frames_needed frames, builds mean embedding and transitions
        to IDENTIFICATION to lock onto the matching track ID.
        """
        if len(tracks) == 0:
            return

        # find track closest to original click bbox centre
        cx = (self._reg_bbox[0] + self._reg_bbox[2]) / 2
        cy = (self._reg_bbox[1] + self._reg_bbox[3]) / 2
        best_track, best_dist = None, float("inf")
        for track in tracks:
            tx = (track[0] + track[2]) / 2
            ty = (track[1] + track[3]) / 2
            dist = (tx - cx) ** 2 + (ty - cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_track = track

        if best_track is None:
            return

        emb = self.reid.extract(frame, best_track[:4])
        self.cmoh.update(0, emb)  # accumulate into temp id=0
        self._reg_frames_collected += 1
        self.registration_progress = self._reg_frames_collected / self._reg_frames_needed

        if self._reg_frames_collected >= self._reg_frames_needed:
            # build mean embedding as the identity reference
            self._initial_embedding = self.cmoh.get_mean_embedding(0)
            self.state = RPFState.IDENTIFICATION
            print(f"[Pipeline] Registration complete ({self._reg_frames_needed} frames). Locking onto target...")

    def _identify(self, embeddings: dict):
        for tid, emb in embeddings.items():
            sim = float(np.dot(emb, self._initial_embedding))
            if sim >= self.cmoh.sim_threshold:
                self.target_id = tid
                self.cmoh.register(tid)
                self.cmoh.update(tid, emb)
                self.state = RPFState.FOLLOWING
                self._lost_frames = 0
                print(f"[Pipeline] Tracking target as ID {tid} (sim={sim:.2f})")
                return

    def _follow(self, embeddings: dict):
        if self.target_id in embeddings:
            self._lost_frames = 0
            self.cmoh.update(self.target_id, embeddings[self.target_id])
        else:
            self._lost_frames += 1
            if self._lost_frames >= self._lost_threshold:
                # entering suspension — seed occluder set with everyone visible now
                self._continuously_visible = set(embeddings.keys())
                self._reid_confirm_count = 0
                self._reid_candidate_id = None
                print("[Pipeline] Target lost → SUSPENDED")
                self.state = RPFState.SUSPENDED

    def _update_occluders(self, embeddings: dict):
        """
        Intersect continuously_visible with current track IDs each frame.
        A track that disappears even briefly is no longer considered an occluder —
        it could be the target re-emerging from behind them.
        """
        current_ids = set(embeddings.keys())
        self._continuously_visible &= current_ids

    def _reidentify(self, embeddings: dict):
        """
        Match each non-occluder track against stored target embedding.
        Occluders (continuously visible since suspension) are excluded.
        """
        target_emb = self._get_target_embedding()

        # only consider tracks that weren't continuously present since the target was lost
        candidates = {
            tid: emb for tid, emb in embeddings.items()
            if tid not in self._continuously_visible
        }

        if not candidates:
            # only occluders visible — wait for target to emerge
            if self.state == RPFState.SUSPENDED:
                self.state = RPFState.REIDENTIFICATION
            return

        best_id, best_sim = None, 0.0
        for tid, emb in candidates.items():
            sim = float(np.dot(emb, target_emb))
            if sim > best_sim:
                best_sim = sim
                best_id = tid

        if best_id is not None and best_sim >= self.cmoh.sim_threshold:
            if best_id == self._reid_candidate_id:
                self._reid_confirm_count += 1
            else:
                self._reid_candidate_id = best_id
                self._reid_confirm_count = 1

            if self._reid_confirm_count >= self._reid_confirm_needed:
                self.target_id = best_id
                self.cmoh.register(best_id)
                self.cmoh.update(best_id, embeddings[best_id])
                self._continuously_visible.clear()
                self.state = RPFState.FOLLOWING
                self._lost_frames = 0
                self._reid_confirm_count = 0
                print(f"[Pipeline] Re-identified as ID {best_id} (sim={best_sim:.2f})")
        else:
            self._reid_confirm_count = 0
            self._reid_candidate_id = None
            if self.state == RPFState.SUSPENDED:
                self.state = RPFState.REIDENTIFICATION

    def _handle_no_tracks(self):
        if self.state == RPFState.FOLLOWING:
            self._lost_frames += 1
            if self._lost_frames >= self._lost_threshold:
                self._continuously_visible.clear()
                self._reid_confirm_count = 0
                self._reid_candidate_id = None
                self.state = RPFState.SUSPENDED

    def _get_target_embedding(self) -> np.ndarray:
        emb = self.cmoh.get_mean_embedding(self.target_id)
        return emb if emb is not None else self._initial_embedding
