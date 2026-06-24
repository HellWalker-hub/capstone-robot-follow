"""
Receives (lx, az) doubles on UDP 127.0.0.1:9999 and republishes as TwistStamped on /cmd_vel.
Jazzy TB3 uses TwistStamped, not Twist.
"""
import socket, struct, time
import rclpy
from geometry_msgs.msg import TwistStamped

UDP_PORT = 9999

rclpy.init()
node = rclpy.create_node('cmd_vel_relay')
pub  = node.create_publisher(TwistStamped, '/cmd_vel_raw', 10)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('127.0.0.1', UDP_PORT))
sock.settimeout(0.5)
print(f"[Relay] Listening on UDP {UDP_PORT}, republishing as TwistStamped to /cmd_vel")

last_msg_time = time.time()

def make_msg(lx, az):
    m = TwistStamped()
    m.header.stamp = node.get_clock().now().to_msg()
    m.header.frame_id = "base_link"
    m.twist.linear.x  = float(lx)
    m.twist.angular.z = float(az)
    return m

try:
    while rclpy.ok():
        try:
            data, _ = sock.recvfrom(64)
            if len(data) >= 16:
                lx, az = struct.unpack('<dd', data[:16])
                pub.publish(make_msg(lx, az))
                last_msg_time = time.time()
        except socket.timeout:
            if time.time() - last_msg_time > 0.5:
                pub.publish(make_msg(0.0, 0.0))
                last_msg_time = time.time()
except KeyboardInterrupt:
    pass
finally:
    pub.publish(make_msg(0.0, 0.0))
    pub.publish(make_msg(0.0, 0.0))
    node.destroy_node(); rclpy.shutdown()
    print("[Relay] Stopped")
