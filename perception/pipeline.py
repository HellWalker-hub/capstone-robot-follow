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

    Key design decisions:
    - Re-id ALWAYS compares against _initial_embedding (diverse registration mean),
      never the runtime CMOH. This prevents identity drift: wrong re-ids during
      following cannot corrupt future re-id decisions.
    - CMOH is used only for appearance adaptation DURING correct following
      (updates gated behind a high-confidence threshold).
    - Occluder exclusion: IDs continuously visible since loss are excluded.
      ByteTrack track_buffer=90 keeps occluder IDs alive ~6s so they can't
      escape the exclusion by getting a new ID.
    - Registration is locked once started — multiple clicks are ignored.
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.tracker = PersonTracker(
            conf=cfg.get("det_conf", 0.4),
            track_buffer=cfg.get("track_buffer", 90),
        )
        self.reid = OSNetReID(model_name=cfg.get("reid_model", "mobilenet_v3_small"))
        self.cmoh = CMOH(
            k=cfg.get("cmoh_k", 10),
            sim_threshold=cfg.get("reid_threshold", 0.65),
        )

        self.state = RPFState.IDLE
        self.target_id: int | None = None
        self._initial_embedding: np.ndarray | None = None  # never overwritten after registration

        self._lost_frames = 0
        self._lost_threshold = cfg.get("lost_threshold", 20)

        # re-id requires N consecutive frames above threshold before confirming
        self._reid_confirm_count = 0
        self._reid_confirm_needed = cfg.get("reid_confirm_frames", 5)
        self._reid_candidate_id: int | None = None

        # occluder exclusion
        self._continuously_visible: set = set()

        # diversity-based registration
        self._diversity_threshold = cfg.get("diversity_threshold", 0.12)
        self._reg_target_frames = cfg.get("reg_target_frames", 20)
        self._reg_min_frames = cfg.get("reg_min_frames", 8)
        self._reg_timeout = cfg.get("reg_timeout_frames", 300)
        self._reg_diverse_embeddings: list = []
        self._reg_frames_seen = 0
        self._reg_bbox: np.ndarray | None = None
        self._reg_current_mean: np.ndarray | None = None

        # CMOH update gate: only update during following if sim to initial is high
        self._cmoh_update_threshold = cfg.get("cmoh_update_threshold", 0.70)

        # UI
        self.registration_progress: float = 0.0
        self.registration_ready: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_target(self, frame: np.ndarray, bbox: np.ndarray):
        """Begin diversity-based registration. Ignored if already registering."""
        if self.state == RPFState.REGISTERING:
            return  # lock out re-clicks during registration

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
        self._reid_confirm_count = 0
        self._reid_candidate_id = None
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
    # State handlers
    # ------------------------------------------------------------------

    def _register(self, frame: np.ndarray, tracks: np.ndarray):
        if len(tracks) == 0:
            return

        self._reg_frames_seen += 1

        cx = (self._reg_bbox[0] + self._reg_bbox[2]) / 2
        cy = (self._reg_bbox[1] + self._reg_bbox[3]) / 2
        best_track = min(
            tracks,
            key=lambda t: (((t[0]+t[2])/2 - cx)**2 + ((t[1]+t[3])/2 - cy)**2)
        )
        emb = self.reid.extract(frame, best_track[:4])

        if self._is_diverse(emb):
            self._reg_diverse_embeddings.append(emb)
            self._reg_current_mean = self._compute_mean(self._reg_diverse_embeddings)

        n = len(self._reg_diverse_embeddings)
        self.registration_progress = min(n / self._reg_target_frames, 1.0)
        self.registration_ready = n >= self._reg_min_frames

        if n >= self._reg_target_frames:
            self._finalise_registration(n, forced=False)
        elif self._reg_frames_seen >= self._reg_timeout:
            if n >= self._reg_min_frames:
                self._finalise_registration(n, forced=True)
            else:
                print("[Pipeline] Registration failed — not enough diversity. Click again.")
                self.state = RPFState.IDLE

    def _finalise_registration(self, n_frames: int, forced: bool):
        self._initial_embedding = self._compute_mean(self._reg_diverse_embeddings)
        self.cmoh.clear()
        for emb in self._reg_diverse_embeddings:
            self.cmoh.update(0, emb)
        tag = " (timeout)" if forced else ""
        print(f"[Pipeline] Registration complete{tag} — {n_frames} diverse frames.")
        self.state = RPFState.IDENTIFICATION

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
            # only update CMOH if embedding is confidently the right person
            # prevents drift when target is partially occluded or blurry
            emb = embeddings[self.target_id]
            sim_to_initial = float(np.dot(emb, self._initial_embedding))
            if sim_to_initial >= self._cmoh_update_threshold:
                self.cmoh.update(self.target_id, emb)
        else:
            self._lost_frames += 1
            if self._lost_frames >= self._lost_threshold:
                # defensive: exclude target's own ID in case of brief flicker
                self._continuously_visible = (
                    set(embeddings.keys()) - {self.target_id}
                )
                self._reid_confirm_count = 0
                self._reid_candidate_id = None
                print("[Pipeline] Target lost → SUSPENDED")
                self.state = RPFState.SUSPENDED

    def _update_occluders(self, embeddings: dict):
        self._continuously_visible &= set(embeddings.keys())

    def _reidentify(self, embeddings: dict):
        """
        Compare non-occluder candidates against _initial_embedding directly.
        Never uses runtime CMOH mean — prevents identity drift from bad re-ids
        poisoning future re-id decisions.
        """
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
            # always compare against clean registration embedding, not drifted runtime mean
            sim = float(np.dot(emb, self._initial_embedding))
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
                self._continuously_visible.discard(best_id)  # safety clear
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_target_embedding(self) -> np.ndarray | None:
        """Return the clean registration embedding used for all re-id decisions."""
        return self._initial_embedding

    def _is_diverse(self, emb: np.ndarray) -> bool:
        if self._reg_current_mean is None:
            return True
        return (1.0 - float(np.dot(emb, self._reg_current_mean))) > self._diversity_threshold

    @staticmethod
    def _compute_mean(embeddings: list) -> np.ndarray:
        mean = np.stack(embeddings).mean(axis=0)
        return (mean / (np.linalg.norm(mean) + 1e-6)).astype(np.float32)
