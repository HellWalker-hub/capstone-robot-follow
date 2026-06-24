"""
Broadcast latest /scan over UDP 127.0.0.1:9998 so perception (Python 3.11) can read it.
Wire format per packet:
  <Iff{N}f  → uint32 N, float angle_min, float angle_increment, then N floats (ranges)
"""
import socket, struct
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data

UDP_ADDR = ('127.0.0.1', 9998)

class LidarBridge(Node):
    def __init__(self):
        super().__init__('lidar_bridge')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, qos_profile_sensor_data)
        self.get_logger().info(f'lidar_bridge: forwarding /scan to UDP {UDP_ADDR}')

    def _scan_cb(self, msg):
        ranges = [(r if (r is not None and r == r) else 0.0) for r in msg.ranges]
        n = len(ranges)
        # cap to keep one UDP packet under ~64 KB (4 bytes * N <= 60_000)
        if n > 1500: n, ranges = 1500, ranges[:1500]
        hdr = struct.pack('<Iff', n, float(msg.angle_min), float(msg.angle_increment))
        body = struct.pack(f'<{n}f', *ranges)
        try: self.sock.sendto(hdr + body, UDP_ADDR)
        except Exception as e: self.get_logger().warn(f'send fail: {e}')

def main():
    rclpy.init()
    n = LidarBridge()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    finally:
        n.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
