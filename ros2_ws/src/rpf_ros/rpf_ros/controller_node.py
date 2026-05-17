#!/usr/bin/env python3
"""
RPF Controller Node — runs on the Mac (offboard).

Subscribes:
  /rpf/target       (std_msgs/String, JSON) — pipeline state + target ByteTrack ID
  /tracked_persons  (std_msgs/String, JSON) — UKF metric positions (bytetrack_id keyed)

Publishes:
  /cmd_vel          (geometry_msgs/Twist)   — TurtleBot3 velocity commands

Control law:
  angular_z = Kp_ang * y_left          (turn toward target, metres lateral offset)
  linear_x  = Kp_lin * (x_fwd - FOLLOW_DISTANCE)   (close/back-off distance error)

Robot stops (zero Twist) when:
  - pipeline state != 'following'
  - target not yet in UKF tracks
  - no /rpf/target update received within SAFETY_TIMEOUT seconds
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import json
import numpy as np

# ---------------------------------------------------------------------------
# Tuning parameters
# ---------------------------------------------------------------------------
FOLLOW_DISTANCE  = 1.0    # metres — desired gap between robot and target
STOP_DISTANCE    = 0.5    # metres — back off if closer than this
Kp_ang           = 0.8    # rad/s per metre of lateral offset
Kp_lin           = 0.4    # m/s   per metre of distance error
MAX_LINEAR       = 0.22   # TurtleBot3 Burger hardware limit (m/s)
MAX_ANGULAR      = 2.84   # TurtleBot3 Burger hardware limit (rad/s)
SAFETY_TIMEOUT   = 0.5    # seconds — publish zero if no update received


class ControllerNode(Node):

    def __init__(self):
        super().__init__('rpf_controller')

        self.sub_target = self.create_subscription(
            String, '/rpf/target',      self._target_cb,  10)
        self.sub_ukf    = self.create_subscription(
            String, '/tracked_persons', self._ukf_cb,     10)
        self.pub_cmd    = self.create_publisher(
            Twist,  '/cmd_vel',                           10)

        self._state      = 'idle'
        self._target_id  = None
        self._ukf_by_bt  = {}     # bytetrack_id → UKF track dict
        self._last_target_time = self.get_clock().now()

        # safety watchdog — stop if perception goes silent
        self.create_timer(SAFETY_TIMEOUT, self._watchdog)

        self.get_logger().info(
            f'Controller ready. Follow dist={FOLLOW_DISTANCE}m '
            f'stop dist={STOP_DISTANCE}m')

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def _target_cb(self, msg: String):
        data = json.loads(msg.data)
        self._state     = data.get('state', 'idle')
        self._target_id = data.get('target_id')
        self._last_target_time = self.get_clock().now()
        self._compute_and_publish()

    def _ukf_cb(self, msg: String):
        tracks = json.loads(msg.data)
        self._ukf_by_bt = {t['bytetrack_id']: t for t in tracks}

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def _compute_and_publish(self):
        cmd = Twist()

        if self._state != 'following' or self._target_id is None:
            self.pub_cmd.publish(cmd)
            return

        track = self._ukf_by_bt.get(self._target_id)
        if track is None:
            # target not yet in UKF (first few frames after lock-on)
            self.pub_cmd.publish(cmd)
            return

        x_fwd  = track['x']   # metres ahead  (positive = in front of robot)
        y_left = track['y']   # metres left   (positive = target is to the left)

        # angular: positive y_left → turn left → positive angular_z (ROS convention)
        angular_z = float(np.clip(Kp_ang * y_left, -MAX_ANGULAR, MAX_ANGULAR))

        # linear: approach target if too far, back off if too close
        dist_error = x_fwd - FOLLOW_DISTANCE
        if x_fwd < STOP_DISTANCE:
            # only allow backing up, not forward
            linear_x = float(np.clip(Kp_lin * dist_error, -MAX_LINEAR, 0.0))
        else:
            linear_x = float(np.clip(Kp_lin * dist_error, -MAX_LINEAR, MAX_LINEAR))

        cmd.linear.x  = linear_x
        cmd.angular.z = angular_z
        self.pub_cmd.publish(cmd)

        self.get_logger().debug(
            f'target={self._target_id} '
            f'x={x_fwd:.2f}m y={y_left:.2f}m → '
            f'lin={linear_x:.3f} ang={angular_z:.3f}')

    def _watchdog(self):
        """Stop the robot if perception node has gone silent."""
        elapsed = (self.get_clock().now() - self._last_target_time).nanoseconds / 1e9
        if elapsed > SAFETY_TIMEOUT:
            self.pub_cmd.publish(Twist())


# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
