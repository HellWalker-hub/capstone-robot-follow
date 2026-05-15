# capstone-robot-follow

Occlusion-resistant person re-identification and following for TurtleBot3 Burger (ROS2 Humble).

## Architecture

```
Webcam → YOLOv8n (MPS/CPU) → ByteTrack → OSNet ReID → CMOH Occlusion Memory
                                                ↓
                                     ROS2 /target_pose topic
                                                ↓
                                    TurtleBot3 /cmd_vel
```

## Stack

| Component | Library | Notes |
|-----------|---------|-------|
| Detection | YOLOv8n (Ultralytics) | MPS on M1, export ONNX for RPi |
| Tracking | ByteTrack (BoxMOT) | Dual-association for occlusion |
| ReID | OSNet (torchreid) | Pretrained MSMT17 |
| Occlusion | CMOH + visibility gating | K-frame history |
| Robot | ROS2 Humble | Offboard compute → /cmd_vel |

## Setup

```bash
pip install -r requirements.txt
```

## Run (laptop prototype)

```bash
python scripts/run_perception.py --source 0  # webcam
```

## Hardware

- **Prototype:** MacBook M1 Air (offboard compute)
- **Robot:** TurtleBot3 Burger + RPi 4 (ROS2 Humble)
- **Future:** NVIDIA Jetson Nano (onboard compute)
