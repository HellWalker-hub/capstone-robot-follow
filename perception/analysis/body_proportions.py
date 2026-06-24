"""
Body proportion + real-world height signature for kandoora-robust ReID.

Two view-invariant signals derived from pose keypoints:
1) Body proportion ratios (shoulder/hip, torso/leg, etc.) — same from any angle
2) Real-world height in meters (uses lidar distance for pixel->meter conversion)

Used in pipeline_v2.py to gate ReID candidates BEFORE appearance similarity.
If a candidate's proportions or height don't match the registered target,
reject without even running appearance ReID.

COCO-17 keypoint indices used here:
   5=L_shoulder, 6=R_shoulder, 11=L_hip, 12=R_hip,
  13=L_knee, 14=R_knee, 15=L_ankle, 16=R_ankle, 0=nose
"""
import numpy as np


# Keypoint indices (COCO-17 standard)
KP_NOSE = 0
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_HIP = 11
KP_R_HIP = 12
KP_L_KNEE = 13
KP_R_KNEE = 14
KP_L_ANKLE = 15
KP_R_ANKLE = 16

# Minimum confidence for a keypoint to be trusted
KP_CONF_THRESHOLD = 0.4


def _kp_ok(kp_row):
    """kp_row is (3,) [x, y, conf]. Returns True if usable."""
    return kp_row[2] >= KP_CONF_THRESHOLD


def _midpoint(a, b):
    """Average of two (x, y) points."""
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def _dist(p1, p2):
    return float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))


def compute_proportions(keypoints: np.ndarray) -> dict | None:
    """
    Compute view-invariant body proportion ratios from one frame of keypoints.

    Args:
        keypoints: (17, 3) [x, y, conf] in image coords.

    Returns:
        Dict of ratios, or None if too many critical keypoints are missing.

        Keys:
          shoulder_to_hip_width: shoulder width / hip width
          torso_to_leg:          torso length / avg leg length
          torso_to_total:        torso length / (torso + leg) length
          head_to_torso:         (nose -> shoulder midpoint) / torso length
        All values are dimensionless ratios (view-invariant up to perspective).
    """
    if keypoints is None or keypoints.shape != (17, 3):
        return None

    ls = keypoints[KP_L_SHOULDER]
    rs = keypoints[KP_R_SHOULDER]
    lh = keypoints[KP_L_HIP]
    rh = keypoints[KP_R_HIP]
    lk = keypoints[KP_L_KNEE]
    rk = keypoints[KP_R_KNEE]
    la = keypoints[KP_L_ANKLE]
    ra = keypoints[KP_R_ANKLE]
    nose = keypoints[KP_NOSE]

    # Need shoulders + hips at minimum
    if not (_kp_ok(ls) and _kp_ok(rs) and _kp_ok(lh) and _kp_ok(rh)):
        return None

    shoulder_width = _dist(ls[:2], rs[:2])
    hip_width = _dist(lh[:2], rh[:2])
    if shoulder_width < 5 or hip_width < 5:
        return None  # degenerate

    shoulder_mid = _midpoint(ls[:2], rs[:2])
    hip_mid = _midpoint(lh[:2], rh[:2])
    torso_length = _dist(shoulder_mid, hip_mid)
    if torso_length < 10:
        return None

    # Leg length: hip -> ankle, average of legs that have visible knee+ankle
    leg_lengths = []
    if _kp_ok(lk) and _kp_ok(la):
        leg_lengths.append(_dist(lh[:2], la[:2]))
    if _kp_ok(rk) and _kp_ok(ra):
        leg_lengths.append(_dist(rh[:2], ra[:2]))
    avg_leg = float(np.mean(leg_lengths)) if leg_lengths else None

    ratios = {
        "shoulder_to_hip_width": float(shoulder_width / hip_width),
    }
    if avg_leg is not None and avg_leg > 10:
        ratios["torso_to_leg"] = float(torso_length / avg_leg)
        ratios["torso_to_total"] = float(torso_length / (torso_length + avg_leg))
    if _kp_ok(nose):
        head_offset = _dist(nose[:2], shoulder_mid)
        if head_offset > 5:
            ratios["head_to_torso"] = float(head_offset / torso_length)

    return ratios


def proportion_similarity(p1: dict, p2: dict) -> float:
    """
    Compare two proportion signatures.
    Returns similarity in [0, 1] where 1 = identical, 0 = wildly different.

    Uses only ratio keys present in BOTH signatures.
    Each ratio compared with relative error converted to similarity:
      sim_i = max(0, 1 - |p1 - p2| / max(|p1|, |p2|, 0.01))
    Final similarity = mean of per-ratio similarities.
    """
    if not p1 or not p2:
        return 0.0
    common_keys = set(p1.keys()) & set(p2.keys())
    if not common_keys:
        return 0.0
    sims = []
    for k in common_keys:
        a, b = p1[k], p2[k]
        denom = max(abs(a), abs(b), 0.01)
        rel_err = abs(a - b) / denom
        sims.append(max(0.0, 1.0 - rel_err))
    return float(np.mean(sims))


def compute_pixel_height(keypoints: np.ndarray, bbox: np.ndarray) -> float | None:
    """
    Compute the person's pixel height from pose if possible, fall back to bbox.

    Args:
        keypoints: (17, 3) keypoints
        bbox: [x1, y1, x2, y2] fallback

    Returns:
        Pixel height (float) or None if uncertain.
    """
    if keypoints is None or keypoints.shape != (17, 3):
        # bbox fallback
        return float(bbox[3] - bbox[1])

    # Try nose-to-ankle (most accurate height proxy)
    nose = keypoints[KP_NOSE]
    la = keypoints[KP_L_ANKLE]
    ra = keypoints[KP_R_ANKLE]
    if _kp_ok(nose) and (_kp_ok(la) or _kp_ok(ra)):
        ankle_y_candidates = []
        if _kp_ok(la): ankle_y_candidates.append(la[1])
        if _kp_ok(ra): ankle_y_candidates.append(ra[1])
        ankle_y = max(ankle_y_candidates)  # bottom-most ankle
        head_y = nose[1]
        if ankle_y > head_y:
            return float(ankle_y - head_y)

    # Fallback to bbox
    return float(bbox[3] - bbox[1])


def compute_real_height(pixel_height: float,
                        lidar_distance_m: float,
                        camera_focal_px: float) -> float | None:
    """
    Compute real-world height in meters using pinhole geometry:
        real_h = pixel_h * distance / focal

    Args:
        pixel_height: person's pixel height in image
        lidar_distance_m: target distance from lidar (meters)
        camera_focal_px: camera focal length in pixels
                         = (image_width / 2) / tan(FOV_horizontal / 2)

    Returns:
        Real-world height in meters, or None on bad input.
    """
    if pixel_height is None or pixel_height <= 0:
        return None
    if lidar_distance_m is None or lidar_distance_m <= 0.1:
        return None
    if camera_focal_px <= 0:
        return None
    return float(pixel_height * lidar_distance_m / camera_focal_px)


def height_similarity(h1_m: float, h2_m: float, tolerance_m: float = 0.10) -> float:
    """
    Compare two real-world heights.
    Returns similarity in [0, 1].
    Within tolerance => 1.0. Falls off linearly to 0 at 3*tolerance.

    Default tolerance 0.10m = 10cm — accounts for posture variation, lidar noise.
    """
    if h1_m is None or h2_m is None:
        return 0.5  # neutral if unknown — don't penalize missing data
    diff = abs(h1_m - h2_m)
    if diff <= tolerance_m:
        return 1.0
    if diff >= 3 * tolerance_m:
        return 0.0
    return float(1.0 - (diff - tolerance_m) / (2 * tolerance_m))


def combined_signature_similarity(sig1: dict, sig2: dict) -> dict:
    """
    Compare two full body signatures.
    Each sig has keys: 'proportions' (dict) and 'real_height_m' (float or None).

    Returns dict:
      'proportion_sim': similarity of proportions (0..1)
      'height_sim':     similarity of real heights (0..1)
      'combined':       weighted combo (0..1)
    """
    p_sim = proportion_similarity(sig1.get("proportions"),
                                   sig2.get("proportions"))
    h_sim = height_similarity(sig1.get("real_height_m"),
                               sig2.get("real_height_m"))
    # Weight: height is harder to fake than proportions, weight slightly more
    combined = 0.4 * p_sim + 0.6 * h_sim
    return {
        "proportion_sim": p_sim,
        "height_sim": h_sim,
        "combined": combined,
    }
