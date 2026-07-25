# The Carry My Bag Robot

**Capstone Project — Abu Dhabi University, College of Engineering, B.Sc. Computer Engineering**

An autonomous mobile robot that follows one specific, registered person through a room —
carrying their belongings hands-free — and keeps following the **right** person even when
someone else crosses in front, or when bystanders are dressed almost identically to the
target. This repository is the final submission of the project.

**Team:** Muaadh Al Haddad · Haroon Siddiqui · Mohsin Al Hashmi · Saif Almashjari · Talal Almahri
**Supervisors:** Prof. Mohammed Ghazal · Eng. Yaser Othman

---

## Table of Contents
1. [The problem](#1-the-problem)
2. [What the robot does](#2-what-the-robot-does)
3. [System architecture](#3-system-architecture)
4. [The cap-marker identity gate](#4-the-cap-marker-identity-gate--the-core-contribution)
5. [Hardware](#5-hardware)
6. [Repository structure](#6-repository-structure)
7. [Setup](#7-setup)
8. [Running the robot — full walkthrough](#8-running-the-robot--full-walkthrough)
9. [Results](#9-results)
10. [Known limitations & future work](#10-known-limitations--future-work)
11. [Project history / legacy prototype](#11-project-history--legacy-prototype)

---

## 1. The Problem

Detecting *a* person in a video frame is a solved problem. The hard problem for a
follow-me robot is deciding **which** person to follow, and never losing track of that
decision — especially when:

- the target is briefly **occluded** by someone walking in front of the camera, or
- bystanders are dressed **almost identically** to the target (in our test environment,
  multiple people in plain white thobes/kandooras).

Appearance-based re-identification alone struggles here: two people in the same outfit
produce nearly indistinguishable visual embeddings. This project's central engineering
contribution is a lightweight, appearance-proof solution to that exact failure mode.

## 2. What the Robot Does

1. An operator opens a web UI streaming the robot's camera feed and **clicks the person**
   to follow.
2. The robot captures several diverse views of that person (turn slowly) to build a
   robust identity reference.
3. It then **follows that person at a fixed distance (~2.0–2.3 m)**, turning to keep them
   centred, using LiDAR to maintain distance and avoid collisions.
4. If the target is briefly hidden or a look-alike walks past, the robot **does not
   switch targets** — it holds its lock (or self-corrects onto the right person) and
   resumes following once the target reappears.

## 3. System Architecture

```
Camera → YOLOv11n (Hailo-8L) → ByteTrack ─┬─ OSNet ReID embedding (512-d)
                                            ├─ PETA attribute vector (35-d)
                                            └─ Marker-cap detector (HSV, CPU)
                                                            │
                                        Identity decision + state machine
                                                            │  target_id
                                                 Control law (bearing + distance)
                                                            │  (linear, angular vel)
     LiDAR (custom 115200 serial node) → /scan → Safety governor → /cmd_vel → OpenCR → motors
```

- **Detection & tracking:** YOLOv11n (nano) + ByteTrack, running on-device on the
  **Hailo-8L** AI accelerator (13 TOPS) — no tethered laptop required.
- **Re-identification:** an OSNet x1.0 appearance embedding (compiled to a custom Hailo
  HEF — no pre-built one existed for this accelerator) fused with a 35-attribute
  pedestrian model (age/gender/clothing-type/carried-items), combined as
  `0.6 × appearance + 0.4 × attributes`.
- **Occlusion memory:** track IDs seen continuously since the target was lost are
  excluded as re-acquisition candidates, and the appearance reference is only updated
  during high-confidence following — preventing identity drift.
- **Marker-cap identity gate:** see [§4](#4-the-cap-marker-identity-gate--the-core-contribution).
- **Control & safety:** proportional control on bearing + LiDAR distance, with an
  independent LiDAR-based safety governor that overrides any command that would drive
  the robot into an obstacle.
- **State machine:** `IDLE → REGISTERING → IDENTIFICATION → FOLLOWING → BLIND_CHASE →
  SUSPENDED/REIDENTIFICATION → FOLLOWING`. The robot only drives during `FOLLOWING`.

## 4. The Cap-Marker Identity Gate — the core contribution

Appearance and attributes alone cannot reliably separate two people wearing the same
outfit. To solve this without adding extra hardware, the target optionally wears a
**bright-orange marker cap** (with a white number panel) on the back of their head —
exactly where the robot's camera sees it while following from behind.

- A lightweight HSV colour detector measures the fraction of orange in the head region
  of every detected person, every frame.
- **When the cap is visible, only the cap-wearing track is eligible to be — or remain —
  the target.** A look-alike bystander without the cap can never be selected, no matter
  how similar their appearance embedding is.
- This check runs **continuously during FOLLOWING**, not only when re-acquiring a lost
  target: if the robot is ever mistakenly locked onto the wrong person, the gate
  self-corrects back onto the cap-wearer within a couple of frames.
- When the cap is not visible (e.g. the target turns to face the camera), the system
  **falls back to appearance + attribute ReID** as a bridge — so the robot still keeps
  following, just without the extra veto.

See [`CAP_SYSTEM.md`](CAP_SYSTEM.md) for the file-by-file breakdown of this feature.

## 5. Hardware

| Component | Role |
|---|---|
| TurtleBot3 Burger + OpenCR | Mobile base, motor control, IMU/odometry |
| Raspberry Pi 5 (Ubuntu 24.04) | Main computer — runs ROS2 + all perception |
| Hailo-8L AI HAT (13 TOPS) | On-device inference for YOLOv11n, OSNet, attribute model |
| USB webcam (UGREEN) | RGB perception input |
| LDROBOT 360° LiDAR | Distance keeping + obstacle safety |
| Marker cap (orange, printed number) | Appearance-proof target identity marker |

All perception, decision-making, and control run **on the robot itself** — there is no
tethered laptop in the loop.

## 6. Repository Structure

```
perception/
├── pipeline_v2.py            — CURRENT state machine + cap identity gate (use this)
├── pipeline.py                — legacy pipeline (pre-cap-gate)
├── cap_detector.py            — the marker-cap HSV detector
├── detector/                  — YOLO wrappers (hailo_yolo.py = current on-device detector)
├── reid/                      — OSNet + attribute extractors (hailo_osnet.py, hailo_attributes.py)
├── tracker/                   — ByteTrack wrappers (hailo_yolo_tracker.py = current)
├── occlusion/                 — CMOH occlusion-memory module
└── analysis/                  — body-proportion helpers

scripts/
├── run_perception_hailo_v3.py — CURRENT entry point (cap system, on-device inference)
├── run_perception_hailo.py    — earlier on-device version (pre-cap-gate)
└── run_perception*.py, test_original.py, run_robot.py — earlier/legacy entry points

deploy/                        — scripts that run directly on the Pi (outside the python package)
├── robot_v3.sh                 — CURRENT launcher: starts the full cap-system stack
├── robot_v2.sh, robot.sh       — earlier launchers, kept for reference/rollback
├── lidar_serial_scan.py       — custom LiDAR driver (see note below — this is a real fix)
├── safety_node.py              — LiDAR-based obstacle safety governor
├── cmd_vel_relay.py            — UDP → ROS2 velocity bridge
└── lidar_bridge.py             — /scan → UDP bridge for the perception overlay

ros2_ws/                        — an EARLIER, offboard-processing prototype (see §11) — not
                                   part of the current on-board system, kept for history

CAP_SYSTEM.md                   — detailed breakdown of the cap-marker feature
requirements.txt                — Python deps for the legacy laptop-side prototype (§11)
```

> **Note on the LiDAR driver:** the LiDAR unit on this robot speaks the LDROBOT `0x54`
> protocol at **115200 baud**, not the 230400 baud the stock TurtleBot3 `ld08_driver`
> assumes. `deploy/lidar_serial_scan.py` is a custom node that reads the sensor at the
> correct baud rate and publishes a standard ROS2 `LaserScan`; `robot_v3.sh` disables the
> stock driver and starts this one instead. Without this fix, `/scan` never publishes and
> the robot loses its LiDAR-based safety distance entirely.

## 7. Setup

The current system runs entirely on the Raspberry Pi 5 inside a Hailo-enabled Python
environment (referred to as `hailo_env` on the robot), which includes the Hailo Runtime
(HailoRT) and Python bindings provided by the Hailo SDK, plus:
```
opencv-python
numpy
```
Model files (HEFs for YOLOv11n, OSNet, and the attribute model) are **not** committed to
this repository (see `.gitignore`) — they are binary artifacts stored on the robot itself.
The OSNet HEF in particular was custom-compiled for the Hailo-8L via the Hailo Dataflow
Compiler, since no pre-built version exists for this accelerator.

For the legacy laptop-side prototype (§11), install:
```bash
pip install -r requirements.txt
python scripts/download_weights.py   # one-time ReID weight download
```

## 8. Running the Robot — Full Walkthrough

1. **Connect to the same network as the robot**, then SSH in:
   ```bash
   ssh capstone@turtlebot1.local
   ```
2. **Launch the full stack** (bringup, velocity relay, safety node, LiDAR, perception):
   ```bash
   bash ~/robot_v3.sh
   ```
   Keep this terminal open — closing it stops all robot services. The on-screen board
   lets you `s` (stop), `r` (restart), `l` (tail the perception log), or `q` (quit).
3. **Open the web UI** in a browser on the same network:
   ```
   http://turtlebot1.local:5000
   ```
4. **Register the target:** have them wear the marker cap (number panel facing away from
   them — it sits on the back of the head) and **click on them** in the video feed. The
   robot enters `REGISTERING`; have the target turn slowly for a few seconds to capture
   diverse views, then it automatically proceeds to `IDENTIFICATION` and locks on.
5. **Walk around.** The robot should keep ~2 m distance, turn to stay centred on the
   target, and ignore bystanders — including a look-alike wearing similar clothing but
   not the marker cap.
6. To stop, use the `s` control in the launch terminal, or `Ctrl+C`.

## 9. Results

Measured from real recorded runs of the deployed system:
- **~17–18 FPS** end-to-end (detection + tracking + ReID + attributes + cap detection),
  stable across 1–4 people in frame.
- **Cap-marker separation:** the registered target's cap-detector score consistently
  reads **0.06–0.18**; bystanders read **0.00–0.02** (decision threshold: 0.025) — a
  clean margin that holds across different clothing colours and lighting.
- **Occlusion recovery:** in a multi-person test run, the robot recovered onto the
  correct target roughly **9 seconds** after a full occlusion loss.
- Across recorded multi-person test runs, the system held its lock on the correct,
  registered target with **no confirmed wrong-person switches**.
- A controlled **side-by-side comparison** (appearance-ReID-only vs. the full cap-gated
  system, run on identical footage) shows the ReID-only baseline losing its lock onto a
  bystander after an occlusion/crossing event, while the cap-gated system holds — and
  recovers — the correct target for the remainder of the clip.

## 10. Known Limitations & Future Work

- The cap gate currently detects **the marker's colour**, not a cap-specific pattern —
  a second object of the same colour in frame would also register. A future version
  could add a unique visual code (e.g. reading the printed number, or an ArUco-style
  tag) for true per-marker specificity.
- The robot currently decelerates abruptly the instant a target is lost, rather than
  coasting through a brief occlusion before entering its search behaviour — a smoother
  "coast through occlusion" transition is planned.
- An RF-based identity tag (e.g. UWB ranging) was investigated as an alternative,
  appearance-independent identity signal that would work even when the visual marker is
  hidden. It was deferred in favour of the zero-extra-hardware visual marker for this
  submission, but remains a promising direction for future hardware iterations.
- The system is designed for indoor, single-target following; multi-target and outdoor
  operation are out of scope for this submission.

## 11. Project History / Legacy Prototype

Earlier in development, perception ran **offboard** on a laptop (streaming frames to the
robot over WiFi) using a different detector/ReID stack (`ros2_ws/`, `requirements.txt`,
`scripts/run_perception.py`). This was superseded by the current fully on-board
Hailo-8L pipeline described above, which removes the laptop dependency entirely and adds
the marker-cap identity gate. The legacy code is kept in the repository for historical
reference and is not part of the final submitted system.
