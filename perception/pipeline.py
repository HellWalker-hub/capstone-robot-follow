import numpy as np
from enum import Enum

from perception.tracker.bytetrack_wrapper import PersonTracker
from perception.reid.osnet_reid import OSNetReID
from perception.occlusion.cmoh import CMOH


class RPFState(Enum):
    IDLE = "idle"
    REGISTERING = "registering"
    IDENTIFICATION = "identification"
    FOLLOWING = "following"
    SUSPENDED = "suspended"
    REIDENTIFICATION = "reidentification"


class FollowPipeline:
    """
    Perception pipeline: YOLOv8+ByteTrack → ReID → CMOH occlusion memory.

    Registration uses diversity-based keyframe selection: a frame is only
    kept if its embedding differs enough from the current mean (cosine
    distance > diversity_threshold). This ensures the identity profile
    contains genuinely different viewpoints rather than redundant similar
    frames, regardless of how long the user stands in front of the camera.

    Occluder exclusion: IDs continuously visible since target was lost are
    excluded from re-id candidates — the real target re-appears as a new
    or briefly-absent track.
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.tracker = PersonTracker(conf=cfg.get("det_conf", 0.4))
        self.reid = OSNetReID(model_name=cfg.get("reid_model", "mobilenet_v3_small"))
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
        self._continuously_visible: set = set()

        # --- diversity-based registration ---
        # keep a frame only if cosine distance from current mean > this value
        self._diversity_threshold = cfg.get("diversity_threshold", 0.12)
        # stop registration once this many diverse frames are collected
        self._reg_target_frames = cfg.get("reg_target_frames", 20)
        # minimum before "early complete" is allowed (need at least some coverage)
        self._reg_min_frames = cfg.get("reg_min_frames", 8)
        # hard timeout: give up waiting for diversity after this many frames seen
        self._reg_timeout = cfg.get("reg_timeout_frames", 300)  # ~20s at 15fps

        # runtime registration state (reset on each register_target call)
        self._reg_diverse_embeddings: list = []
        self._reg_frames_seen = 0        # total frames processed during registration
        self._reg_bbox: np.ndarray | None = None
        self._reg_current_mean: np.ndarray | None = None

        # exposed to UI: 0.0→1.0 based on diverse frames collected
        self.registration_progress: float = 0.0
        self.registration_ready: bool = False  # True once min_frames reached

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_target(self, frame: np.ndarray, bbox: np.ndarray):
        """Begin diversity-based registration from a clicked bounding box."""
        self._reg_bbox = bbox.copy()
        self._reg_diverse_embeddings = []
        self._reg_frames_seen = 0
        self._reg_current_mean = None
        self.registration_progress = 0.0
        self.registration_ready = False
        self.cmoh.clear()
        self._initial_embedding = None
        self.target_id = None
        self._continuously_visible.clear()
        self.state = RPFState.REGISTERING
        print("[Pipeline] Registration started — turn slowly for best coverage.")

    def process(self, frame: np.ndarray) -> dict:
        tracks = self.tracker.update(frame)

        result = {
            "state": self.state,
            "target_id": self.target_id,
            "target_bbox": None,
            "all_tracks": tracks,
            "occluder_ids": set(self._continuously_visible),
            "reg_diverse_count": len(self._reg_diverse_embeddings),
            "reg_target": self._reg_target_frames,
        }

        if self.state == RPFState.IDLE:
            return result

        if len(tracks) == 0:
            self._handle_no_tracks()
            if self.state in (RPFState.SUSPENDED, RPFState.REIDENTIFICATION):
                self._continuously_visible.clear()
            result["state"] = self.state
            return result

        track_embeddings = {}
        for track in tracks:
            tid = int(track[4])
            track_embeddings[tid] = self.reid.extract(frame, track[:4])

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
        result["reg_diverse_count"] = len(self._reg_diverse_embeddings)
        if self.target_id is not None:
            for track in tracks:
                if int(track[4]) == self.target_id:
                    result["target_bbox"] = track[:4].copy()
                    break

        return result

    # ------------------------------------------------------------------
    # Internal state handlers
    # ------------------------------------------------------------------

    def _register(self, frame: np.ndarray, tracks: np.ndarray):
        """
        Diversity-based keyframe registration.

        Each frame: extract embedding from the person closest to the click.
        Accept it only if cosine distance from current mean > diversity_threshold.
        Complete when reg_target_frames diverse frames collected, or on timeout
        (using however many diverse frames were gathered so far).
        """
        if len(tracks) == 0:
            return

        self._reg_frames_seen += 1

        # find track whose centre is closest to original click
        cx = (self._reg_bbox[0] + self._reg_bbox[2]) / 2
        cy = (self._reg_bbox[1] + self._reg_bbox[3]) / 2
        best_track = min(
            tracks,
            key=lambda t: (((t[0]+t[2])/2 - cx)**2 + ((t[1]+t[3])/2 - cy)**2)
        )
        emb = self.reid.extract(frame, best_track[:4])

        # accept frame if it adds new information
        is_diverse = self._is_diverse(emb)
        if is_diverse:
            self._reg_diverse_embeddings.append(emb)
            self._reg_current_mean = self._compute_mean(self._reg_diverse_embeddings)

        n = len(self._reg_diverse_embeddings)
        self.registration_progress = min(n / self._reg_target_frames, 1.0)
        self.registration_ready = n >= self._reg_min_frames

        enough = n >= self._reg_target_frames
        timed_out = self._reg_frames_seen >= self._reg_timeout

        if enough or (timed_out and n >= self._reg_min_frames):
            self._finalise_registration(n, timed_out and not enough)
        elif timed_out:
            # timed out without enough diversity — lower bar and accept what we have
            print(f"[Pipeline] Registration timeout — only {n} diverse frames. Proceeding anyway.")
            if n > 0:
                self._finalise_registration(n, forced=True)
            else:
                # nothing collected at all — reset to IDLE
                self.state = RPFState.IDLE
                print("[Pipeline] Registration failed — no person detected. Click again.")

    def _finalise_registration(self, n_frames: int, forced: bool = False):
        self._initial_embedding = self._compute_mean(self._reg_diverse_embeddings)
        # seed CMOH with all diverse frames so re-id has rich history from the start
        self.cmoh.clear()
        for emb in self._reg_diverse_embeddings:
            self.cmoh.update(0, emb)
        self.state = RPFState.IDENTIFICATION
        tag = " (timeout)" if forced else ""
        print(f"[Pipeline] Registration complete{tag} — {n_frames} diverse frames. Locking on...")

    def _is_diverse(self, emb: np.ndarray) -> bool:
        """True if emb is sufficiently different from current registration mean."""
        if self._reg_current_mean is None:
            return True  # first frame always accepted
        cos_sim = float(np.dot(emb, self._reg_current_mean))
        cos_dist = 1.0 - cos_sim
        return cos_dist > self._diversity_threshold

    @staticmethod
    def _compute_mean(embeddings: list) -> np.ndarray:
        mean = np.stack(embeddings).mean(axis=0)
        return (mean / (np.linalg.norm(mean) + 1e-6)).astype(np.float32)

    def _identify(self, embeddings: dict):
        for tid, emb in embeddings.items():
            sim = float(np.dot(emb, self._initial_embedding))
            if sim >= self.cmoh.sim_threshold:
                self.target_id = tid
                self.cmoh.register(tid)
                self.cmoh.update(tid, emb)
                self.state = RPFState.FOLLOWING
                self._lost_frames = 0
                print(f"[Pipeline] Following target ID {tid} (sim={sim:.2f})")
                return

    def _follow(self, embeddings: dict):
        if self.target_id in embeddings:
            self._lost_frames = 0
            self.cmoh.update(self.target_id, embeddings[self.target_id])
        else:
            self._lost_frames += 1
            if self._lost_frames >= self._lost_threshold:
                self._continuously_visible = set(embeddings.keys())
                self._reid_confirm_count = 0
                self._reid_candidate_id = None
                print("[Pipeline] Target lost → SUSPENDED")
                self.state = RPFState.SUSPENDED

    def _update_occluders(self, embeddings: dict):
        self._continuously_visible &= set(embeddings.keys())

    def _reidentify(self, embeddings: dict):
        target_emb = self._get_target_embedding()
        candidates = {
            tid: emb for tid, emb in embeddings.items()
            if tid not in self._continuously_visible
        }

        if not candidates:
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
