#!/usr/bin/env python3
"""
RPF Perception Node — runs on the Mac (offboard).

Wraps FollowPipeline (detector + tracker + ReID + state machine) and
publishes two topics every frame:

  /rpf/target  (std_msgs/String, JSON)
      { state, target_id, target_bbox }

  /rpf/tracks  (std_msgs/String, JSON)
      [ { id, bbox, conf, kps_xy, kps_conf }, ... ]   — all tracked persons

Click a person in the OpenCV window to register the follow target.
Press 'r' to reset, 'q' to quit.
"""

import sys
import os
# make the repo root importable so 'from perception.pipeline import ...' works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))

import json
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from perception.pipeline import FollowPipeline, RPFState

STATE_COLORS = {
    RPFState.IDLE:             (128, 128, 128),
    RPFState.REGISTERING:      (255, 200,   0),
    RPFState.IDENTIFICATION:   (  0, 255, 255),
    RPFState.FOLLOWING:        (  0, 255,   0),
    RPFState.SUSPENDED:        (  0, 165, 255),
    RPFState.REIDENTIFICATION: (  0,   0, 255),
}


class PerceptionNode(Node):

    def __init__(self):
        super().__init__('rpf_perception')

        self.declare_parameter('camera_index',    0)
        self.declare_parameter('reid_threshold',  0.55)
        self.declare_parameter('reid_every_n',    3)

        cam_idx      = self.get_parameter('camera_index').value
        reid_thresh  = self.get_parameter('reid_threshold').value
        reid_every_n = self.get_parameter('reid_every_n').value

        config = {
            'reid_model':     'osnet_x0_25',
            'reid_threshold': reid_thresh,
            'reid_every_n':   reid_every_n,
        }
        self.pipeline = FollowPipeline(config)

        self.pub_target = self.create_publisher(String, '/rpf/target', 10)
        self.pub_tracks = self.create_publisher(String, '/rpf/tracks', 10)

        self.cap = cv2.VideoCapture(cam_idx)
        if not self.cap.isOpened():
            self.get_logger().error(f'Cannot open camera index {cam_idx}')
            raise RuntimeError(f'Cannot open camera {cam_idx}')

        self._clicked = None
        self._thumbnail = None
        cv2.namedWindow('RPF Perception')
        cv2.setMouseCallback('RPF Perception', self._mouse_cb)

        self.get_logger().info('Perception node ready — click a person to follow.')

    # ------------------------------------------------------------------

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._clicked = (x, y)

    def process_frame(self):
        """Read one frame, run pipeline, publish, update window. Call each loop tick."""
        ret, frame = self.cap.read()
        if not ret:
            return

        result = self.pipeline.process(frame)

        # handle click → register target
        if self._clicked is not None:
            bbox = self._bbox_at(result['all_tracks'], self._clicked)
            if bbox is not None:
                self.pipeline.register_target(frame, bbox)
                x1, y1, x2, y2 = map(int, bbox)
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size > 0:
                    self._thumbnail = cv2.resize(crop, (80, 160))
            else:
                self.get_logger().info('No tracked person at clicked location.')
            self._clicked = None

        # publish /rpf/target
        target_msg = String()
        target_msg.data = json.dumps({
            'state':       result['state'].value,
            'target_id':   result['target_id'],
            'target_bbox': result['target_bbox'].tolist()
                           if result['target_bbox'] is not None else None,
        })
        self.pub_target.publish(target_msg)

        # publish /rpf/tracks (bboxes + keypoints for UKF node)
        tracks_msg = String()
        tracks_msg.data = json.dumps(self._serialize_tracks(result['all_tracks']))
        self.pub_tracks.publish(tracks_msg)

        self._draw_overlay(frame, result)
        cv2.imshow('RPF Perception', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rclpy.shutdown()
        elif key == ord('r'):
            self.pipeline.state = RPFState.IDLE
            self.pipeline.target_id = None
            self.pipeline.tracker.reset()
            self.pipeline.cmoh.clear()
            self.get_logger().info('Reset.')

    # ------------------------------------------------------------------

    def _serialize_tracks(self, tracks: np.ndarray) -> list:
        """Pack tracked bboxes + pose keypoints into a JSON-serialisable list."""
        keypoints = self.pipeline.tracker.get_last_keypoints()
        out = []
        for i, track in enumerate(tracks):
            entry = {
                'id':       int(track[4]),
                'bbox':     track[:4].tolist(),
                'conf':     float(track[5]),
                'kps_xy':   None,
                'kps_conf': None,
            }
            if keypoints is not None and i < len(keypoints.xy):
                entry['kps_xy'] = keypoints.xy[i].cpu().numpy().tolist()
                if keypoints.conf is not None:
                    entry['kps_conf'] = keypoints.conf[i].cpu().numpy().tolist()
            out.append(entry)
        return out

    def _bbox_at(self, tracks: np.ndarray, point: tuple):
        px, py = point
        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            if x1 <= px <= x2 and y1 <= py <= y2:
                return track[:4]
        return None

    def _draw_overlay(self, frame: np.ndarray, result: dict):
        state  = result['state']
        color  = STATE_COLORS.get(state, (255, 255, 255))
        occluders = result.get('occluder_ids', set())

        for track in result['all_tracks']:
            x1, y1, x2, y2 = map(int, track[:4])
            tid = int(track[4])
            if tid == result['target_id']:
                c, th = (0, 255, 0), 3
            elif tid in occluders:
                c, th = (0, 128, 255), 2
            else:
                c, th = (180, 180, 180), 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, th)
            cv2.putText(frame, f'ID:{tid}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

        cv2.rectangle(frame, (0, 0), (300, 30), (0, 0, 0), -1)
        cv2.putText(frame, f'State: {state.value.upper()}', (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if state == RPFState.REGISTERING and self.pipeline is not None:
            n      = len(self.pipeline._reg_diverse_embeddings)
            target = self.pipeline._reg_target_frames
            prog   = self.pipeline.registration_progress
            filled = int(300 * prog)
            bar_c  = (0, 220, 80) if self.pipeline.registration_ready else (255, 200, 0)
            cv2.rectangle(frame, (5, 40), (305, 58), (60, 60, 60), -1)
            cv2.rectangle(frame, (5, 40), (5 + filled, 58), bar_c, -1)
            cv2.putText(frame, f'Turn slowly... {n}/{target} views',
                        (5, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.48, bar_c, 1)

        if self._thumbnail is not None:
            h, w = frame.shape[:2]
            th_, tw_ = self._thumbnail.shape[:2]
            y_off, x_off = h - th_ - 8, 8
            frame[y_off:y_off + th_, x_off:x_off + tw_] = self._thumbnail
            cv2.putText(frame, 'TARGET', (x_off, y_off - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # ------------------------------------------------------------------

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0)
            node.process_frame()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
