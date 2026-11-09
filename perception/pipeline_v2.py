import numpy as np
from enum import Enum

import os
if os.environ.get("USE_HAILO", "0") == "1":
    from perception.tracker.hailo_yolo_tracker import PersonTracker
else:
    from perception.tracker.bytetrack_wrapper import PersonTracker
if os.environ.get("USE_HAILO", "0") == "1":
    from perception.reid.hailo_osnet import OSNetReID
    from perception.reid.hailo_attributes import HailoAttributes
else:
    from perception.reid.osnet_reid import OSNetReID
from perception.occlusion.cmoh import CMOH
from perception.cap_detector import CapDetector, CAP_THRESHOLD


class RPFState(Enum):
    IDLE = "idle"
    REGISTERING = "registering"
    IDENTIFICATION = "identification"
    FOLLOWING = "following"
    SUSPENDED = "suspended"
    REIDENTIFICATION = "reidentification"
    BLIND_CHASE = "blind_chase"


ATTRIBUTE_THRESHOLD = 0.85   # cosine sim floor for PETA attribute vectors
ATTR_WEIGHT = 0.4            # combined = 0.6*appearance + 0.4*attributes
APPEARANCE_WEIGHT = 0.6

# --- Cap identity gate ---
# When the marker cap is visible on a track, only cap-wearing tracks may become or
# regain the target (hard veto on everyone else — kills kandoora look-alike switches).
# Because the cap is strong identity proof, we relax the appearance floor for the
# cap-wearer and confirm faster. When NO cap is visible, the pipeline falls back to
# the normal appearance+attribute ReID (the "bridge").
CAP_REID_THRESHOLD = 0.40    # relaxed appearance floor when a candidate is cap-gated
CAP_CONFIRM_FRAMES = 2       # consecutive cap-gated frames before locking (anti-flicker)


class FollowPipeline:
    """
    Perception pipeline: detector + ByteTrack -> ReID (+ PETA attributes) -> cap
    identity gate -> CMOH occlusion memory. With USE_HAILO=1 the detector is YOLOv11n
    on the Hailo-8L; otherwise it is the Ultralytics CPU/MPS path.

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
        self.reid = OSNetReID(model_name=cfg.get("reid_model", "osnet_x1_0"))
        self.attr = HailoAttributes()
        self.cap = CapDetector()
        self.cmoh = CMOH(
            k=cfg.get("cmoh_k", 10),
            sim_threshold=cfg.get("reid_threshold", 0.65),
        )

        self.state = RPFState.IDLE
        self.target_id: int | None = None
        self._initial_attr_vec = None
        self._reg_attr_samples = []
        self._initial_embedding: np.ndarray | None = None  # never overwritten after registration

        self._lost_frames = 0
        self._lost_threshold = cfg.get("lost_threshold", 20)

        # FPS optimisation: only run ReID every N frames during FOLLOWING
        # tracker handles identity by bbox overlap in between
        self._reid_every_n = cfg.get("reid_every_n", 3)
        self._frame_counter = 0

        # re-id requires N consecutive frames above threshold before confirming
        self._reid_confirm_count = 0
        self._reid_confirm_needed = cfg.get("reid_confirm_frames", 5)
        self._reid_candidate_id: int | None = None

        # cap-gate hysteresis (separate from appearance confirm)
        self._cap_confirm_count = 0
        self._cap_candidate_id: int | None = None

        # cap-correction hysteresis (authoritative cap check during FOLLOWING)
        self._cap_correct_count = 0
        self._cap_correct_id: int | None = None

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

        # blind-chase: keep moving briefly after losing target
        self._blind_chase_frames = 0
        self._blind_chase_max = cfg.get("blind_chase_frames", 180)
        # Stage thresholds (frames into blind-chase)
        self._bc_grace_end   = cfg.get("bc_grace_end",   30)   # 0–30: glide, no search
        self._bc_lidar_end   = cfg.get("bc_lidar_end",   60)   # 30–60: lidar only
        self._bc_gated_end   = cfg.get("bc_gated_end",  120)   # 60–120: camera re-ID, narrow gate
        # 120–blind_chase_max: gate expanding
        # Gate sizes
        self._gate_px_init   = cfg.get("gate_px_init", 200)    # ±px around last bbox center
        self._gate_px_max    = cfg.get("gate_px_max",  400)
        self._gate_rad_init  = cfg.get("gate_rad_init", 0.26)  # ±15° in radians
        self._gate_rad_max   = cfg.get("gate_rad_max",  0.52)  # ±30°
        # Last known target anchor (set on FOLLOWING → BLIND_CHASE)
        self._last_target_px_x = None
        self._last_target_lidar_ang = None
        self._last_target_lidar_dist = None
        # Lidar blob result (set during stage 1, read by runner)
        self.lidar_blob_angle = None
        self.lidar_blob_dist  = None
        # Max bbox-center jump per frame allowed in FOLLOWING (px)
        # If bbox jumps further than this, treat as a hijack -> lose target
        self._max_bbox_jump_px = cfg.get("max_bbox_jump_px", 150)
        self._last_bbox_cx = None
        self._last_bbox_cy = None

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
        self._initial_attr_vec = None
        self._reg_attr_samples = []
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
            "caps": {},
        }

        if self.state == RPFState.IDLE:
            return result

        if len(tracks) == 0:
            self._handle_no_tracks()
            if self.state in (RPFState.SUSPENDED, RPFState.REIDENTIFICATION):
                self._continuously_visible.clear()
            result["state"] = self.state
            return result

        self._frame_counter += 1
        run_reid = (self._frame_counter % self._reid_every_n == 0)

        # Marker-cap scores per track — cheap HSV, computed every frame for the
        # identity gate and for the overlay.
        track_caps = self._detect_caps(frame, tracks)
        result["caps"] = track_caps

        if self.state == RPFState.REGISTERING:
            self._register(frame, tracks)
        elif self.state == RPFState.IDENTIFICATION:
            # always run ReID during identification
            track_embeddings = self._extract_all(frame, tracks)
            track_attrs = self._extract_all_attrs(frame, tracks)
            self._identify(track_embeddings, track_attrs, track_caps)
        elif self.state == RPFState.FOLLOWING:
            # Authoritative cap check first: if the cap is on someone other than the
            # current target, hand the lock back to the cap-wearer (fixes occluder steals).
            self._cap_correction(track_caps)
            if run_reid:
                # single full-body crop only during stable following — fastest path
                track_embeddings = self._extract_all(frame, tracks, head_crop=False)
                self._follow(track_embeddings)
            else:
                # tracker-only frame: just check target is still present
                self._follow_trackonly(tracks)
        elif self.state == RPFState.BLIND_CHASE:
            self._blind_chase_frames += 1
            if self._blind_chase_frames >= self._blind_chase_max:
                print("[Pipeline] Blind-chase expired -> SUSPENDED")
                self.state = RPFState.SUSPENDED
            else:
                track_embeddings = self._extract_all(frame, tracks, head_crop=True)
                track_attrs = self._extract_all_attrs(frame, tracks)
                self._reidentify(track_embeddings, track_attrs, track_caps)
        elif self.state in (RPFState.SUSPENDED, RPFState.REIDENTIFICATION):
            # always run full dual-crop ReID when searching for target
            track_embeddings = self._extract_all(frame, tracks, head_crop=True)
            track_attrs = self._extract_all_attrs(frame, tracks)
            self._update_occluders(track_embeddings)
            self._reidentify(track_embeddings, track_attrs, track_caps)

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

    def _extract_all_attrs(self, frame, tracks):
        attrs = {}
        for track in tracks:
            tid = int(track[4])
            attrs[tid] = self.attr.extract(frame, track[:4])
        return attrs

    def _attr_similarity(self, candidate_attr):
        if self._initial_attr_vec is None or candidate_attr is None:
            return 0.0
        return HailoAttributes.similarity(self._initial_attr_vec, candidate_attr)

    def _detect_caps(self, frame, tracks):
        """Per-track marker-cap score {track_id: score in [0,1]}. Mirrors _extract_all_attrs."""
        caps = {}
        for track in tracks:
            tid = int(track[4])
            caps[tid] = self.cap.detect(frame, track[:4])
        # DEBUG: log raw cap scores periodically so the threshold can be tuned to reality
        self._cap_dbg = getattr(self, "_cap_dbg", 0) + 1
        if self._cap_dbg % 30 == 0 and caps:
            try:
                with open("/tmp/cap_debug.log", "a") as f:
                    f.write("caps " + " ".join(f"{k}:{v:.3f}" for k, v in caps.items())
                            + f"  target={self.target_id}\n")
            except Exception:
                pass
        return caps

    @staticmethod
    def _cap_wearers(caps):
        """Track ids whose cap score clears the threshold (i.e. visibly wearing the cap)."""
        if not caps:
            return []
        return [tid for tid, s in caps.items() if s >= CAP_THRESHOLD]

    def _cap_correction(self, caps):
        """Authoritative cap check run every FOLLOWING frame.

        If the marker cap is visible on a track that is NOT the current target while
        the current target has no cap, we are locked on the wrong person (e.g. an
        occluder stole the lock during a pass-through). The cap is unique, so hand the
        target straight back to the cap-wearer (brief hysteresis guards against flukes).
        """
        cap_ids = self._cap_wearers(caps)
        # no cap visible, or the target legitimately wears it -> nothing to correct
        if not cap_ids or self.target_id in cap_ids:
            self._cap_correct_count = 0
            self._cap_correct_id = None
            return
        new_id = cap_ids[0]          # single marked target assumed
        if new_id == self._cap_correct_id:
            self._cap_correct_count += 1
        else:
            self._cap_correct_id = new_id
            self._cap_correct_count = 1
        if self._cap_correct_count >= CAP_CONFIRM_FRAMES:
            print(f"[Pipeline] CAP CORRECTION: target {self.target_id} has no cap; "
                  f"cap is on {new_id} -> switching back to cap-wearer")
            self.target_id = new_id
            self.cmoh.register(new_id)
            self._lost_frames = 0
            self._cap_correct_count = 0
            self._cap_correct_id = None

    def _extract_all(self, frame: np.ndarray, tracks: np.ndarray,
                     head_crop: bool = True) -> dict:
        """Extract embeddings for all tracks. head_crop=False skips dual-crop (faster)."""
        embeddings = {}
        orig_hw = self.reid.head_weight
        orig_bw = self.reid.body_weight
        if not head_crop:
            # temporarily disable head weighting — single full-body forward pass
            self.reid.head_weight = 0.0
            self.reid.body_weight = 1.0
        for track in tracks:
            tid = int(track[4])
            embeddings[tid] = self.reid.extract(frame, track[:4])
        if not head_crop:
            self.reid.head_weight = orig_hw
            self.reid.body_weight = orig_bw
        return embeddings

    def _follow_trackonly(self, tracks: np.ndarray):
        """Tracker-only frame: check target presence without ReID inference."""
        current_ids = {int(t[4]) for t in tracks}
        if self.target_id in current_ids:
            self._lost_frames = 0
        else:
            self._lost_frames += 1
            if self._lost_frames >= self._lost_threshold:
                self._continuously_visible = set()
                self._reid_confirm_count = 0
                self._reid_candidate_id = None
                self._blind_chase_frames = 0
                print("[Pipeline] Target lost → BLIND_CHASE")
                self.state = RPFState.BLIND_CHASE

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
        attr_vec = self.attr.extract(frame, best_track[:4])
        if attr_vec is not None and attr_vec.size > 0:
            self._reg_attr_samples.append(attr_vec)

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
        # Average attribute samples
        if self._reg_attr_samples:
            import numpy as _np
            self._initial_attr_vec = _np.mean(_np.stack(self._reg_attr_samples, axis=0), axis=0).astype('float32')
            top5 = _np.argsort(self._initial_attr_vec)[-5:][::-1].tolist()
            print(f"[Pipeline-v2] Attribute vector captured ({len(self._reg_attr_samples)} samples). Top-5 dim indices: {top5}")
        else:
            self._initial_attr_vec = None
            print("[Pipeline-v2] No attribute samples — attribute gate disabled")
        self.state = RPFState.IDENTIFICATION

    def _identify(self, embeddings: dict, attrs: dict = None, caps: dict = None):
        cap_ids = self._cap_wearers(caps)
        cap_gated = bool(cap_ids)
        # CAP GATE: when the cap is visible, only cap-wearing tracks are eligible.
        eligible = cap_ids if cap_gated else list(embeddings.keys())

        best_id, best_sim, best_app = None, -1.0, 0.0
        for tid in eligible:
            emb = embeddings.get(tid)
            if emb is None:
                continue
            app_sim = float(np.dot(emb, self._initial_embedding))
            if attrs and self._initial_attr_vec is not None and tid in attrs:
                sim = APPEARANCE_WEIGHT * app_sim + ATTR_WEIGHT * self._attr_similarity(attrs[tid])
            else:
                sim = app_sim
            if sim > best_sim:
                best_sim, best_id, best_app = sim, tid, app_sim

        # cap is strong identity proof → relaxed floor; else the normal appearance bar
        thr = CAP_REID_THRESHOLD if cap_gated else self.cmoh.sim_threshold
        if best_id is None or best_sim < thr:
            self._cap_confirm_count = 0
            self._cap_candidate_id = None
            return

        # anti-flicker: a cap-gated lock must persist a couple of frames
        if cap_gated:
            if best_id == self._cap_candidate_id:
                self._cap_confirm_count += 1
            else:
                self._cap_candidate_id = best_id
                self._cap_confirm_count = 1
            if self._cap_confirm_count < CAP_CONFIRM_FRAMES:
                return

        self.target_id = best_id
        self.cmoh.register(best_id)
        self.cmoh.update(best_id, embeddings[best_id])
        self.state = RPFState.FOLLOWING
        self._lost_frames = 0
        self._cap_confirm_count = 0
        self._cap_candidate_id = None
        tag = "CAP" if cap_gated else "appearance"
        print(f"[Pipeline] Following target ID {best_id} via {tag} "
              f"(app={best_app:.2f} combined={best_sim:.2f})")

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
                self._continuously_visible = (
                    set(embeddings.keys()) - {self.target_id}
                )
                self._reid_confirm_count = 0
                self._reid_candidate_id = None
                self._blind_chase_frames = 0
                # anchor is already set by _follow recording — keep it
                print(f"[Pipeline] Target lost → BLIND_CHASE (anchor px_x={self._last_target_px_x})")
                self.state = RPFState.BLIND_CHASE

    def _update_occluders(self, embeddings: dict):
        self._continuously_visible &= set(embeddings.keys())

    def _reidentify(self, embeddings: dict, attrs: dict = None, caps: dict = None):
        """
        Compare non-occluder candidates against _initial_embedding directly.
        Never uses runtime CMOH mean — prevents identity drift from bad re-ids
        poisoning future re-id decisions.

        CAP GATE: when the marker cap is visible, candidates are restricted to
        cap-wearing tracks — a look-alike occluder without the cap can never be
        re-identified as the target. When no cap is visible, falls back to the
        normal appearance+attribute search (the bridge).
        """
        candidates = {
            tid: emb for tid, emb in embeddings.items()
            if tid not in self._continuously_visible
        }

        cap_ids = self._cap_wearers(caps)
        cap_gated = bool(cap_ids)
        if cap_gated:
            candidates = {tid: emb for tid, emb in candidates.items() if tid in cap_ids}

        if not candidates:
            if self.state == RPFState.SUSPENDED:
                self.state = RPFState.REIDENTIFICATION
            return

        best_id, best_sim = None, 0.0
        for tid, emb in candidates.items():
            app_sim = float(np.dot(emb, self._initial_embedding))
            if attrs and self._initial_attr_vec is not None and tid in attrs:
                attr_sim = self._attr_similarity(attrs[tid])
                sim = APPEARANCE_WEIGHT * app_sim + ATTR_WEIGHT * attr_sim
            else:
                sim = app_sim
            if sim > best_sim:
                best_sim = sim
                best_id = tid

        # cap-gated: relaxed floor + fast confirm (the cap already proves identity)
        thr = CAP_REID_THRESHOLD if cap_gated else self.cmoh.sim_threshold
        need = CAP_CONFIRM_FRAMES if cap_gated else self._reid_confirm_needed

        if best_id is not None and best_sim >= thr:
            if best_id == self._reid_candidate_id:
                self._reid_confirm_count += 1
            else:
                self._reid_candidate_id = best_id
                self._reid_confirm_count = 1

            if self._reid_confirm_count >= need:
                self.target_id = best_id
                self.cmoh.register(best_id)
                self.cmoh.update(best_id, embeddings[best_id])
                self._continuously_visible.discard(best_id)  # safety clear
                self._continuously_visible.clear()
                self.state = RPFState.FOLLOWING
                self._lost_frames = 0
                self._reid_confirm_count = 0
                tag = "CAP" if cap_gated else "appearance"
                print(f"[Pipeline] Re-identified as ID {best_id} via {tag} (sim={best_sim:.2f})")
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
                self._blind_chase_frames = 0
                self.state = RPFState.BLIND_CHASE
        elif self.state == RPFState.BLIND_CHASE:
            self._blind_chase_frames += 1
            if self._blind_chase_frames >= self._blind_chase_max:
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
