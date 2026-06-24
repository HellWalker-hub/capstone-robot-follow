#!/usr/bin/env python3
"""
cap_detector.py — bright marker-cap detector for the RPF target person.

WHY THIS EXISTS
  ReID (OSNet + attributes) struggles to separate visually-similar people (e.g. two
  men in white kandooras). This module adds a deliberate, unmistakable visual marker:
  a bright-orange cap with a white number panel. Detecting that marker on a person
  track gives a hard, appearance-proof identity signal used to GATE the ReID decision
  in pipeline_v2 (only a cap-wearing track may become / regain the target).

WHAT IT DOES
  detect(frame, bbox) -> float in [0,1]
    Looks at the head region (top slice of the person bbox), HSV-thresholds for the
    cap's orange, and boosts the score when a bright white panel is also present
    inside that region. Returns a confidence-like score; pipeline treats
    score >= CAP_THRESHOLD as "this track is wearing the cap".

  No OCR: with a single marked target we only need "is the cap here", not the digit.

CALIBRATION
  The orange/white HSV ranges below MUST be tuned to the real cap under real lighting.
  Run this file standalone (see __main__) to dump mask images for tuning.
"""
import cv2
import numpy as np

# --- region of the person bbox to inspect (cap sits on the head = top slice) ---
# The cap occupies only the very top of a full-body box, so keep this tight or the
# orange fraction gets diluted by shoulders/background below the head.
HEAD_TOP_FRAC = 0.00     # start at top of bbox
HEAD_BOT_FRAC = 0.16     # ...down to 16% of bbox height (≈ the head)
SIDE_PAD_FRAC = 0.10     # ignore 10% on each side (background bleed)

# --- HSV thresholds (OpenCV: H 0-179, S 0-255, V 0-255). TUNE THESE LIVE. ---
# Bright orange of the cap body. Tuned live against the real cap (full coverage,
# no skin/background false positives). Lowering S/V further starts catching skin.
ORANGE_LO = np.array([  3,  90,  90])
ORANGE_HI = np.array([ 26, 255, 255])
# White number panel (low saturation, high value).
WHITE_LO  = np.array([  0,   0, 170])
WHITE_HI  = np.array([179,  60, 255])

# --- scoring ---
# Tuned live: cap-wearer's head-region orange fraction measured ~0.03-0.08 (distance-
# invariant), background ~0. Threshold sits comfortably between them.
CAP_THRESHOLD   = 0.025  # pipeline treats score >= this as "wearing the cap"
# (legacy knobs, no longer used by detect() — kept for reference)
MIN_ORANGE_FRAC = 0.12
MIN_WHITE_FRAC  = 0.03
WHITE_BOOST     = 0.30


class CapDetector:
    def __init__(self,
                 orange_lo=ORANGE_LO, orange_hi=ORANGE_HI,
                 white_lo=WHITE_LO, white_hi=WHITE_HI):
        self.orange_lo, self.orange_hi = orange_lo, orange_hi
        self.white_lo, self.white_hi = white_lo, white_hi

    def _head_region(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h))
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1 or bh <= 1:
            return None
        ry1 = y1 + int(bh * HEAD_TOP_FRAC)
        ry2 = y1 + int(bh * HEAD_BOT_FRAC)
        sx = int(bw * SIDE_PAD_FRAC)
        rx1, rx2 = x1 + sx, x2 - sx
        if rx2 - rx1 <= 1 or ry2 - ry1 <= 1:
            return None
        return frame[ry1:ry2, rx1:rx2]

    def _fractions(self, region_bgr):
        hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
        orange = cv2.inRange(hsv, self.orange_lo, self.orange_hi)
        white  = cv2.inRange(hsv, self.white_lo, self.white_hi)
        return orange.mean() / 255.0, white.mean() / 255.0, orange, white

    def detect(self, frame, bbox) -> float:
        """Cap score in [0,1] = fraction of the head region that is cap-orange.

        White-panel boost is intentionally NOT used: in white-robe / white-room scenes
        the white cue is everywhere and unreliable. Orange alone is the discriminator.
        The pipeline's CAP_THRESHOLD decides 'wearing the cap' from this raw fraction.
        """
        region = self._head_region(frame, bbox)
        if region is None or region.size == 0:
            return 0.0
        orange_frac, _, _, _ = self._fractions(region)
        return float(orange_frac)


# --------------------------------------------------------------------------
# Standalone calibration: grab a frame (camera or image file), dump masks to
# /tmp so we can tune the HSV ranges against the real cap. Usage:
#   python cap_detector.py                 # grab one frame from the webcam
#   python cap_detector.py /path/img.jpg   # use an image file
# Outputs: /tmp/cap_frame.jpg, /tmp/cap_orange.jpg, /tmp/cap_white.jpg
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    CAM = "/dev/v4l/by-id/usb-UGREEN_Camera_UGREEN_Camera_SN0001-video-index0"
    if len(sys.argv) > 1:
        frame = cv2.imread(sys.argv[1])
        if frame is None:
            print(f"could not read {sys.argv[1]}"); sys.exit(1)
    else:
        cap = cv2.VideoCapture(CAM, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        for _ in range(5):           # let auto-exposure settle
            ok, frame = cap.read()
        cap.release()
        if not ok:
            print("camera read failed"); sys.exit(1)

    det = CapDetector()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, det.orange_lo, det.orange_hi)
    white  = cv2.inRange(hsv, det.white_lo, det.white_hi)
    cv2.imwrite("/tmp/cap_frame.jpg", frame)
    cv2.imwrite("/tmp/cap_orange.jpg", orange)
    cv2.imwrite("/tmp/cap_white.jpg", white)
    print(f"frame {frame.shape}, orange_frac={orange.mean()/255.0:.3f}, "
          f"white_frac={white.mean()/255.0:.3f}")
    print("wrote /tmp/cap_frame.jpg /tmp/cap_orange.jpg /tmp/cap_white.jpg")
