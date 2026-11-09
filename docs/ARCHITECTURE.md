# Architecture notes

Design notes for the robot build. The README has the overview; this file has the details
someone would need to read or change the code.

## Processes on the robot

The robot runs two kinds of process that cannot share an interpreter. Perception runs in the
Hailo virtual environment (Python 3.11, HailoRT 4.20.0). The ROS 2 Jazzy nodes run on the
system Python 3.12. They talk over UDP on localhost.

```
lidar_serial_scan.py  -> /scan (LaserScan, best-effort QoS)
                          |-> lidar_bridge.py  -> UDP 127.0.0.1:9998 -> perception
                          '-> safety_node.py
perception (run_perception_hailo_v3.py)
                         -> UDP 127.0.0.1:9999 (lx, az) -> cmd_vel_relay.py
cmd_vel_relay.py      -> /cmd_vel_raw (TwistStamped)
safety_node.py        -> /cmd_vel     (TwistStamped) -> turtlebot3_bringup -> OpenCR
```

`deploy/robot_v3.sh` starts these in order (bringup, relay, safety node, LiDAR driver, LiDAR
bridge, perception) and stops all of them, sending zero velocity, when its terminal closes.
Run it inside tmux.

## Perception pipeline (`perception/pipeline_v2.py`)

With `USE_HAILO=1` the pipeline imports the Hailo backends:

| Stage | Module | Model file |
|---|---|---|
| Detection + tracking | `tracker/hailo_yolo_tracker.py` | `yolov11n.hef` |
| Appearance embedding | `reid/hailo_osnet.py` | `osnet_x1_0.hef` (512-d) |
| Attributes | `reid/hailo_attributes.py` | `person_attr_resnet_v1_18.hef` (35 PETA attributes) |
| Cap gate | `cap_detector.py` | none, HSV colour test on the CPU |

Without `USE_HAILO`, it falls back to the Ultralytics tracker and the YoutuReID ONNX model
(`reid/osnet_reid.py`, which keeps the class name `OSNetReID` for interface reasons), and the
attribute model is skipped.

### Registration

The user is selected by clicking them in the web UI. During registration the pipeline only
accepts a frame's embedding if it differs from the running mean by more than a diversity
threshold, so the reference covers several views instead of many copies of one. The robot
build uses `reg_min_frames=150` and `reg_target_frames=300`. The mean of the accepted
embeddings becomes `_initial_embedding`, which is never updated afterwards. Re-acquisition
always compares against it, so a wrong match during following cannot drift the reference.

### Matching

During following the pipeline scores candidates as `0.6 x appearance + 0.4 x attributes`
(cosine similarity for both). The robot build uses `reid_threshold=0.75`. Re-identification
runs every third frame while following; in between, the tracker's IDs carry identity.

### Cap gate

`CapDetector.detect()` crops the top 16% of a person's box (skipping 10% at each side to avoid
background) and returns the fraction of pixels inside an orange HSV range
(H 3 to 26, S 90 to 255, V 90 to 255). A score of 0.025 or more counts as wearing the cap.

The gate is applied in two places:

1. **Re-acquisition** (`_identify`, `_reidentify`): if anyone in view is wearing the cap,
   only cap wearers are candidates, and the appearance floor drops to 0.40 because the cap is
   strong evidence. Two consecutive frames are needed before locking.
2. **Every frame while following** (`_cap_correction`): if the cap is on a person who is not
   the current target, the target moves to the cap wearer after two consecutive frames. This
   is what fixes a look-alike taking the lock during a crossing.

If nobody in view is wearing the cap, both places fall back to appearance and attributes.

### Losing the user

When the target disappears, the pipeline records everyone visible at that moment as likely
occluders and excludes them from re-acquisition (ByteTrack's 90-frame track buffer keeps
their IDs stable long enough for this to work). The state machine then moves through blind
chase (a short coast on decaying velocity), suspended, and re-identification.

## Follow controller (`scripts/run_perception_hailo_v3.py`)

- Angular: proportional on the horizontal offset of the target's box, with a dead zone.
- Linear: holds 2.0 to 2.3 m. Distance comes from the LiDAR, using the minimum range within
  ±5° of the target's bearing. Speed is capped at 0.20 m/s in software.
- Both outputs are smoothed (EMA) and rate-limited per frame.

Known issue: the camera bearing is computed with left positive, but the LiDAR driver keeps
the sensor's clockwise convention (left negative), and the bearing is passed to the LiDAR
lookup without negation. For a user off to one side, distance is read from the mirrored side.

## LiDAR driver (`deploy/lidar_serial_scan.py`)

The stock TurtleBot3 driver opens the port at 230400 baud. This unit speaks the LDROBOT `0x54`
protocol at 115200, so the stock driver produced no scans and no error. Our driver reads
47-byte packets of 12 points each, fills 360 one-degree bins, and publishes `/scan` in frame
`base_scan` (range 0.10 to 8.0 m). The standard CRC8 rejected about 96% of valid packets on
this unit, so the driver checks packet structure and angle ranges instead.

## Safety governor (`deploy/safety_node.py`)

Subscribes to `/cmd_vel_raw` and `/scan`, publishes `/cmd_vel` at 30 Hz. It only ever reduces
forward speed: the nearest return within ±30° of straight ahead scales speed linearly between
0.60 m and 0.30 m, and stops it below 0.30 m. Turning and reversing pass through unchanged.

Known issue: it republishes the most recent command with no timeout, so if the relay stops
sending, the last velocity persists until the node is stopped.
