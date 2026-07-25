# Cap-Marker Identity System (v3)

This is the current, final identity system used by the robot: a **marker-cap identity
gate** layered on top of the appearance-based ReID baseline described in the main
[`README.md`](README.md). It was developed on a separate branch (`cap-marker-v3`, tag
`cap-system-v3`, preserved in git history) and has since been merged into `main` as the
submitted implementation. See the main README for the full project overview.

## What it adds on top of the ReID baseline
- `perception/cap_detector.py` — HSV orange-cap detector (head-region orange fraction).
- `perception/pipeline_v2.py` — identity gate wired into `_identify` / `_reidentify`, plus a
  continuous `_cap_correction()` during FOLLOWING (cap = hard veto when visible; appearance
  ReID bridges when the cap is hidden).
- `scripts/run_perception_hailo_v3.py` — v3 entry point (cap overlay badge, 2.0–2.3 m follow band).
- Hailo on-device inference: `perception/detector/hailo_backend.py`, `reid/hailo_osnet.py`,
  `reid/hailo_attributes.py`, trackers, etc.

## deploy/ — host-level ROS2 nodes & launchers (run on the Pi, outside the python package)
- `robot_v3.sh` — launches the full v3 stack; `robot_v2.sh` / `robot.sh` — older launchers.
- `lidar_serial_scan.py` — **LiDAR fix**: the unit speaks the LDROBOT 0x54 protocol at
  **115200** (not 230400); this node reads it and publishes `/scan`.
- `safety_node.py`, `cmd_vel_relay.py`, `lidar_bridge.py` — safety governor, UDP→ROS velocity
  relay, and `/scan`→UDP bridge for the perception overlay.

## Excluded from git (see .gitignore)
Model binaries (HEFs, `weights/*.onnx`) — kept on the Pi at `~/hailo_demo/models/`.
