#!/usr/bin/env python3.12
"""
lidar_serial_scan.py — LDROBOT 0x54-protocol serial -> /scan publisher.

WHY THIS EXISTS
  The lidar on this robot transmits the standard LDROBOT 0x54/0x2C 47-byte frame
  protocol but at 115200 baud (LD14-class), while the stock turtlebot3 ld08_driver
  opens the port at 230400 and reads only noise -> /scan never publishes. This node
  reads /dev/ttyUSB0 at 115200, parses frames, accumulates one full revolution, and
  publishes a standard ROS LaserScan (REP-103: angle 0 = front, CCW positive),
  frame_id=base_scan -- a drop-in replacement for ld08_driver's /scan.

  Verified empirically: at 115200 the stream decodes into clean 47-byte frames with
  sane speeds (~3800 deg/s) and distances; at 230400 it is pure noise.

ORIENTATION
  LDROBOT units spin clockwise (viewed from top); ROS wants CCW. INVERT_DIR maps
  sensor degrees -> ROS degrees. ANGLE_OFFSET_DEG shifts where "front" sits. Both are
  meant to be calibrated live against the camera overlay if points look mirrored or
  rotated -- change the two constants below, nothing else.
"""
import math
import time
import serial
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

# ---- config (the only things you should ever need to tweak) ----
PORT             = "/dev/ttyUSB0"
BAUD             = 115200
NBINS            = 360            # 1-degree bins, like the stock TB3 scan
FRAME_ID         = "base_scan"
RANGE_MIN        = 0.10
RANGE_MAX        = 8.0
INVERT_DIR       = False          # ros_deg = +sensor_deg (verified live: True mirrored left/right)
ANGLE_OFFSET_DEG = 0.0            # added after inversion; tweak so an object dead-ahead lands at bin 0

# ---- LDROBOT 0x54 frame format ----
HDR         = 0x54
VERLEN      = 0x2C
FRAME_LEN   = 47
N_POINTS    = 12
MAX_CENTIDEG = 36000          # angles are centi-degrees in [0, 36000)

# NOTE: we deliberately do NOT validate the LDROBOT CRC8. This unit's CRC does not
# match the canonical LDLiDAR table (measured ~4% pass on known-good frames), so CRC
# would throw away nearly every valid frame. Instead we sync on the 0x54/0x2C header
# and sanity-check the angle fields; any rare corrupt frame just gets overwritten on
# the next revolution and is filtered by range_min/range_max downstream.


def u16(buf, i):
    return buf[i] | (buf[i + 1] << 8)


class LidarSerialScan(Node):
    def __init__(self):
        super().__init__("lidar_serial_scan")
        self.pub = self.create_publisher(LaserScan, "/scan", qos_profile_sensor_data)
        self.ser = None
        self._open_serial()

        self.scan = [0.0] * NBINS
        self.inten = [0.0] * NBINS
        self.prev_start = None
        self.last_pub = time.time()
        self.buf = bytearray()
        self.frames = 0
        self.revs = 0

        self.get_logger().info(
            f"lidar_serial_scan: {PORT}@{BAUD} -> /scan ({NBINS} bins, frame={FRAME_ID}, "
            f"invert={INVERT_DIR}, offset={ANGLE_OFFSET_DEG}deg)")
        # poll the serial port fast; parsing is cheap
        self.create_timer(0.005, self._tick)

    def _open_serial(self):
        # retry so we tolerate ld08_driver still releasing the port at startup
        for attempt in range(40):
            try:
                self.ser = serial.Serial(PORT, BAUD, timeout=0.05)
                self.ser.reset_input_buffer()
                return
            except Exception as e:
                if attempt == 0:
                    self.get_logger().warn(f"waiting for {PORT}: {e}")
                time.sleep(0.25)
        raise RuntimeError(f"could not open {PORT} after retries")

    def _tick(self):
        try:
            data = self.ser.read(2048)
        except Exception as e:
            self.get_logger().warn(f"serial read error: {e}")
            return
        if data:
            self.buf.extend(data)
        self._parse()

    def _parse(self):
        buf = self.buf
        i = 0
        n = len(buf)
        while i + FRAME_LEN <= n:
            if buf[i] != HDR or buf[i + 1] != VERLEN:
                i += 1
                continue
            frame = buf[i:i + FRAME_LEN]
            # structural sanity in place of CRC: start/end angles must be valid
            if u16(frame, 4) >= MAX_CENTIDEG or u16(frame, 42) >= MAX_CENTIDEG:
                i += 1                       # misaligned/garbage -> resync from next byte
                continue
            self._consume(frame)
            i += FRAME_LEN
        # keep the unparsed tail (partial frame at the end)
        del buf[:i]

    def _consume(self, f):
        sa = u16(f, 4) / 100.0               # start angle, deg
        ea = u16(f, 42) / 100.0              # end angle, deg
        span = (ea - sa) % 360.0
        step = span / (N_POINTS - 1)
        for k in range(N_POINTS):
            dist_mm = u16(f, 6 + k * 3)
            inten = f[8 + k * 3]
            ang = (sa + step * k) % 360.0
            ros = (-ang if INVERT_DIR else ang) + ANGLE_OFFSET_DEG
            b = int(round(ros)) % NBINS
            if dist_mm == 0:
                continue                     # no return -> leave 0.0 (consumers skip it)
            self.scan[b] = dist_mm / 1000.0
            self.inten[b] = float(inten)

        # full revolution when the start angle wraps past 360 -> 0
        if self.prev_start is not None and sa + 180.0 < self.prev_start:
            self._publish()
            self.scan = [0.0] * NBINS
            self.inten = [0.0] * NBINS
        self.prev_start = sa
        self.frames += 1

    def _publish(self):
        now = time.time()
        dt = now - self.last_pub
        self.last_pub = now
        self.revs += 1

        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = FRAME_ID
        msg.angle_min = 0.0
        msg.angle_increment = 2.0 * math.pi / NBINS
        msg.angle_max = msg.angle_min + (NBINS - 1) * msg.angle_increment
        msg.range_min = RANGE_MIN
        msg.range_max = RANGE_MAX
        msg.scan_time = float(dt) if 0.0 < dt < 1.0 else 0.1
        msg.time_increment = msg.scan_time / NBINS
        msg.ranges = [float(r) for r in self.scan]
        msg.intensities = [float(v) for v in self.inten]
        self.pub.publish(msg)

        if self.revs % 20 == 0:
            hz = 1.0 / dt if dt > 0 else 0.0
            valid = sum(1 for r in self.scan if r > 0.0)
            self.get_logger().info(f"/scan rev {self.revs}: ~{hz:.1f} Hz, {valid}/{NBINS} valid bins")


def main():
    rclpy.init()
    node = LidarSerialScan()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node.ser:
                node.ser.close()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
