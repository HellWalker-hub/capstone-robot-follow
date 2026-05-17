#!/usr/bin/env python3
"""
UKF Spatial Tracker Node — adapted from ukf_tracker_node.py.

Changes from original:
  - Removed: YOLO model, /image_raw subscription, run_yolo(), image_callback()
  - Added:   /rpf/tracks subscription, tracks_callback(), _process_tracks()
  - Added:   bytetrack_id field on Track for controller association
  - Simplified: track association now uses ByteTrack IDs directly (dict keyed
    by bytetrack_id) — no Hungarian algorithm needed since ByteTrack already
    handles data association upstream.

Unchanged: pixel_to_ground(), make_ukf(), UKF math, /tracked_persons output.

Camera intrinsics (FX, FY, CX, CY) must match the actual webcam.
Run camera calibration and update these constants before deployment.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import numpy as np
import json
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# ---------------------------------------------------------------------------
# Camera intrinsics — update after calibration
# Default values assume 320x240 resolution
# ---------------------------------------------------------------------------
CAM_HEIGHT  = 0.32    # camera height above ground (metres)
TILT_DEG    = -2.0    # camera tilt (negative = tilted down)
TILT_RAD    = np.radians(TILT_DEG)
FX          = 600.0
FY          = 600.0
CX          = 160.0   # principal point x (frame_width  / 2 for 320x240)
CY          = 120.0   # principal point y (frame_height / 2 for 320x240)

MAX_MISSED      = 15
MIN_HEIGHT_M    = 1.4
MAX_HEIGHT_M    = 2.2
DT              = 1.0 / 30


# ---------------------------------------------------------------------------
# Geometry helpers (unchanged from original)
# ---------------------------------------------------------------------------

def pixel_to_ground(u, v):
    """Project image point (u, v) to robot-frame ground coordinates (x_fwd, y_left)."""
    xn = (u - CX) / FX
    yn = (v - CY) / FY
    ray_cam = np.array([xn, yn, 1.0])
    cos_t = np.cos(TILT_RAD)
    sin_t = np.sin(TILT_RAD)
    R = np.array([
        [1,     0,      0    ],
        [0,  cos_t, -sin_t   ],
        [0,  sin_t,  cos_t   ],
    ])
    ray_robot = R @ ray_cam
    if abs(ray_robot[1]) < 1e-6:
        return None
    t = CAM_HEIGHT / ray_robot[1]
    if t < 0:
        return None
    return float(t * ray_robot[2]), float(t * ray_robot[0])  # (x_fwd, y_left)


def make_ukf(x0, y0, height0):
    """Create a 5-state UKF: [x, y, vx, vy, height]."""
    points = MerweScaledSigmaPoints(n=5, alpha=0.1, beta=2.0, kappa=0.0)

    def fx(state, dt):
        x, y, vx, vy, h = state
        return np.array([x + vx * dt, y + vy * dt, vx, vy, h])

    def hx(state):
        return np.array([state[0], state[1], state[4]])

    ukf = UKF(dim_x=5, dim_z=3, fx=fx, hx=hx, dt=DT, points=points)
    ukf.x = np.array([x0, y0, 0.0, 0.0, height0])
    ukf.Q = np.diag([0.05, 0.05, 0.5, 0.5, 0.01])
    ukf.R = np.diag([0.1,  0.1,  0.1])
    ukf.P = np.diag([0.5,  0.5,  1.0, 1.0, 0.5])
    return ukf


# ---------------------------------------------------------------------------
# Track — keyed by ByteTrack ID for direct association
# ---------------------------------------------------------------------------

class Track:

    def __init__(self, x, y, height, bbox, bytetrack_id):
        self.bytetrack_id = bytetrack_id
        self.ukf          = make_ukf(x, y, height)
        self.missed       = 0
        self.hits         = 1
        self.bbox         = bbox

    def predict(self):
        self.ukf.predict(dt=DT)

    def update(self, x, y, height, bbox):
        self.ukf.update(np.array([x, y, height]))
        self.missed  = 0
        self.hits   += 1
        self.bbox    = bbox

    @property
    def pos(self):
        return self.ukf.x[0], self.ukf.x[1]

    @property
    def height(self):
        return self.ukf.x[4]

    @property
    def velocity(self):
        return self.ukf.x[2], self.ukf.x[3]


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class UKFTrackerNode(Node):

    def __init__(self):
        super().__init__('ukf_tracker')

        # keyed by bytetrack_id — no Hungarian algorithm needed
        self._tracks: dict[int, Track] = {}

        self.sub = self.create_subscription(
            String, '/rpf/tracks', self.tracks_callback, 10)
        self.pub_tracked = self.create_publisher(
            String, '/tracked_persons', 10)

        self.get_logger().info(
            'UKF Tracker node started. Consuming /rpf/tracks.')

    # ------------------------------------------------------------------

    def tracks_callback(self, msg: String):
        track_list = json.loads(msg.data)
        detections = self._process_tracks(track_list)
        self._update_tracks(detections)
        self._publish_tracks()

    def _process_tracks(self, track_list: list) -> list:
        """
        Convert /rpf/tracks JSON into metric detections using pixel_to_ground.
        Uses ankle keypoints when available, falls back to bbox bottom-centre.
        """
        detections = []
        for t in track_list:
            bbox         = t['bbox']
            bytetrack_id = t['id']
            kps_xy       = t.get('kps_xy')
            kps_conf     = t.get('kps_conf')
            x1, y1, x2, y2 = bbox

            ground_pos = None

            # -- ankle keypoints (COCO indices 15=left, 16=right ankle) ----
            if kps_xy is not None and kps_conf is not None:
                kps_xy   = np.array(kps_xy,   dtype=np.float32)
                kps_conf = np.array(kps_conf, dtype=np.float32)
                ankle_l, ankle_r       = kps_xy[15],   kps_xy[16]
                conf_l,  conf_r        = kps_conf[15], kps_conf[16]

                if conf_l > 0.3 and conf_r > 0.3:
                    u = (ankle_l[0] + ankle_r[0]) / 2
                    v = (ankle_l[1] + ankle_r[1]) / 2
                    ground_pos = pixel_to_ground(u, v)
                elif conf_l > 0.3:
                    ground_pos = pixel_to_ground(ankle_l[0], ankle_l[1])
                elif conf_r > 0.3:
                    ground_pos = pixel_to_ground(ankle_r[0], ankle_r[1])

            # -- fallback: bottom-centre of bounding box -------------------
            if ground_pos is None:
                ground_pos = pixel_to_ground((x1 + x2) / 2, y2)

            if ground_pos is None:
                continue

            x_fwd, y_left = ground_pos

            # -- height estimate (nose keypoint, COCO index 0) -------------
            height_est = 1.7
            if kps_xy is not None and kps_conf is not None:
                nose_conf = float(kps_conf[0])
                if nose_conf > 0.3:
                    dist = np.sqrt(x_fwd ** 2 + y_left ** 2)
                    nose_v     = float(kps_xy[0][1])
                    h          = (CY - nose_v) / FY * dist
                    height_est = float(np.clip(h, MIN_HEIGHT_M, MAX_HEIGHT_M))

            detections.append({
                'bytetrack_id': bytetrack_id,
                'x':            x_fwd,
                'y':            y_left,
                'height':       height_est,
                'bbox':         bbox,
            })

        return detections

    def _update_tracks(self, detections: list):
        # predict all existing tracks forward
        for track in self._tracks.values():
            track.predict()

        seen_ids = set()
        for det in detections:
            bt_id = det['bytetrack_id']
            seen_ids.add(bt_id)

            if bt_id in self._tracks:
                self._tracks[bt_id].update(
                    det['x'], det['y'], det['height'], det['bbox'])
            else:
                self._tracks[bt_id] = Track(
                    det['x'], det['y'], det['height'], det['bbox'], bt_id)

        # age out tracks not seen this frame
        for bt_id in list(self._tracks.keys()):
            if bt_id not in seen_ids:
                self._tracks[bt_id].missed += 1
                if self._tracks[bt_id].missed >= MAX_MISSED:
                    del self._tracks[bt_id]

    def _publish_tracks(self):
        output = []
        for track in self._tracks.values():
            x, y   = track.pos
            vx, vy = track.velocity
            output.append({
                'bytetrack_id': track.bytetrack_id,
                'x':      round(x,  3),
                'y':      round(y,  3),
                'vx':     round(vx, 3),
                'vy':     round(vy, 3),
                'height': round(track.height, 3),
                'missed': track.missed,
                'hits':   track.hits,
                'bbox':   [round(v, 1) for v in track.bbox],
            })

        msg      = String()
        msg.data = json.dumps(output)
        self.pub_tracked.publish(msg)


# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = UKFTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
