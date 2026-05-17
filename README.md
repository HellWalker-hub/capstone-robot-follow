# capstone-robot-follow

Occlusion-resistant person re-identification and following for TurtleBot3 Burger (ROS2 Humble).
Designed for UAE environments with high kandoora (white thobe) prevalence — follows from behind, no face visible.

## Architecture

```
Webcam → YOLO26n (MPS) → ByteTrack → YoutuReID ONNX (768-d) → CMOH Occlusion Memory
                                              ↓
                                 Runtime Diversity Buffer (max 100 frames)
                                              ↓
                                    ROS2 /cmd_vel (offboard Mac → RPi4)
```

## Stack

| Component | Detail |
|---|---|
| Detection | YOLO26n (Ultralytics) — NMS-free, 2× faster CPU than YOLOv8n |
| Tracking | ByteTrack, track_buffer=90 (~6s occluder ID stability) |
| ReID | YoutuReID ONNX 768-d — part-based: 60% head / 40% body |
| Occlusion memory | CMOH (K=10 rolling buffer) + occluder exclusion via `_continuously_visible` |
| Runtime adaptation | Diversity buffer (max 100 frames) grown during FOLLOWING, feeds CMOH mean |
| Re-id anchor | `_initial_embedding` — frozen mean of 20 diverse registration frames, never updated |
| State machine | IDLE → REGISTERING → IDENTIFICATION → FOLLOWING ↔ SUSPENDED/REIDENTIFICATION |
| Robot | TurtleBot3 Burger, ROS2 Humble, offboard Mac → RPi4 via WiFi |

## Key Design Decisions

- **`_initial_embedding` is sacred** — set once at registration, used for ALL re-id comparisons, never drifted
- **Runtime buffer enriches CMOH** — diverse high-confidence frames collected during FOLLOWING adapt appearance memory to new lighting/distances without touching the re-id anchor
- **CMOH updates gated at sim ≥ 0.70** — prevents bad frames from polluting appearance memory
- **Occluder exclusion** — IDs continuously visible since target loss are excluded from re-id candidates; ByteTrack buffer=90 keeps their IDs stable so they can't escape by getting reassigned
- **5-frame re-id confirmation** — prevents false positives from brief appearance matches
- **ReID every 3 frames during FOLLOWING** — tracker handles identity by overlap in between; full dual-crop only during SUSPENDED/REIDENTIFICATION

## Setup

```bash
python scripts/download_weights.py   # downloads YoutuReID ONNX (106MB, one-time)
pip install -r requirements.txt
```

## Run

```bash
# Laptop prototype (webcam, click to register)
python scripts/run_perception.py

# Custom source or threshold
python scripts/run_perception.py --source 1 --reid-threshold 0.55
```

Click a person in the window to register. Turn slowly (include back-view) during registration. Press `r` to reset, `q` to quit.

## ROS2 Integration (offboard Mac → RPi4)

```bash
source /opt/ros/humble/setup.bash
cd ros2_ws && colcon build --symlink-install && source install/setup.bash
ros2 launch rpf_ros follow.launch.py
```

Three nodes launch: `perception_node` (pipeline + webcam), `ukf_node` (metric distance via UKF), `controller_node` (publishes `/cmd_vel`).

## Project Structure

```
perception/
├── pipeline.py          — RPF state machine
├── detector/            — YOLO26n wrapper
├── tracker/             — ByteTrack wrapper (yolo26n.pt)
├── reid/                — YoutuReID ONNX + part-based extraction
└── occlusion/           — CMOH rolling memory
scripts/
├── run_perception.py    — webcam entry point
└── download_weights.py  — HuggingFace weight download
ros2_ws/src/rpf_ros/     — ROS2 package (perception, UKF, controller nodes)
weights/                 — YoutuReID ONNX (gitignored, download separately)
```

## Hardware

| Role | Hardware |
|---|---|
| Prototype compute | MacBook M1 Air (MPS, offboard) |
| Robot | TurtleBot3 Burger — OpenCR + RPi 4, ROS2 Humble |
| Camera | USB webcam |
| Future upgrade | NVIDIA Jetson Orin Nano (onboard, JetPack 6, ROS2 Humble native) |

## Safe Revert Tag

`stable-youtureid-baseline` — pre-optimization baseline on GitHub.
