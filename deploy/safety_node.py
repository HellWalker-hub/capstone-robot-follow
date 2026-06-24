"""
Obstacle-avoidance safety filter.
  In:  /cmd_vel_raw (TwistStamped) + /scan (LaserScan)
  Out: /cmd_vel (TwistStamped) with forward velocity reduced if obstacles ahead
"""
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
from rclpy.qos import qos_profile_sensor_data

FRONT_HALF_ARC_RAD = math.radians(30.0)   # ±30°
SLOWDOWN_NEAR      = 0.30                 # below this → lx = 0
SLOWDOWN_FAR       = 0.60                 # above this → no slowdown
LOOP_HZ            = 30.0

class SafetyNode(Node):
    def __init__(self):
        super().__init__('safety_node')
        self.latest_scan = None
        self.latest_cmd  = None
        self.create_subscription(LaserScan,    '/scan',         self._scan_cb, qos_profile_sensor_data)
        self.create_subscription(TwistStamped, '/cmd_vel_raw',  self._cmd_cb,  10)
        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_timer(1.0/LOOP_HZ, self._tick)
        self.get_logger().info(f'safety: front ±{math.degrees(FRONT_HALF_ARC_RAD):.0f}°, '
                               f'slowdown {SLOWDOWN_NEAR:.2f}–{SLOWDOWN_FAR:.2f} m')

    def _scan_cb(self, msg): self.latest_scan = msg
    def _cmd_cb(self, msg):  self.latest_cmd  = msg

    def _min_front_distance(self):
        s = self.latest_scan
        if s is None:
            return float('inf')
        amin, ainc = s.angle_min, s.angle_increment
        rmin, rmax = max(s.range_min, 0.05), min(s.range_max, 10.0)
        n = len(s.ranges)
        best = float('inf')
        half = FRONT_HALF_ARC_RAD
        two_pi = 2.0 * math.pi
        for i, r in enumerate(s.ranges):
            if r is None or r != r or r < rmin or r > rmax:  # NaN / inf / out of range
                continue
            ang = amin + i * ainc
            # wrap to -pi..pi centered on 0 (front)
            while ang >  math.pi: ang -= two_pi
            while ang < -math.pi: ang += two_pi
            if -half <= ang <= half and r < best:
                best = r
        return best

    def _tick(self):
        if self.latest_cmd is None:
            return
        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "base_link"
        out.twist.linear.x  = self.latest_cmd.twist.linear.x
        out.twist.angular.z = self.latest_cmd.twist.angular.z

        # only override forward motion; allow backing up and turning regardless
        if out.twist.linear.x > 0.0:
            d = self._min_front_distance()
            if d < SLOWDOWN_NEAR:
                out.twist.linear.x = 0.0
            elif d < SLOWDOWN_FAR:
                scale = (d - SLOWDOWN_NEAR) / (SLOWDOWN_FAR - SLOWDOWN_NEAR)
                out.twist.linear.x *= scale
        self.pub.publish(out)

def main():
    rclpy.init()
    n = SafetyNode()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    finally:
        n.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
