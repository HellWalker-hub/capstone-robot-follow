"""
Pi5 + Hailo perception demo — WebSocket binary stream, smooth as butter.
"""
import sys, os, time, threading
from collections import deque
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import atexit
from flask import Flask, request, jsonify
from flask_sock import Sock

import socket, struct

from perception.pipeline_v2 import FollowPipeline, RPFState

# === Motion control constants (matched to run_robot.py) ===
DEAD_ZONE_X        = 30
ANG_GAIN           = 0.004
MAX_ANG            = 0.5
TARGET_WIDTH_RATIO = 0.35
DEAD_ZONE_RATIO    = 0.04
MAX_LIN            = 0.20
LIN_GAIN           = 1.8

# === Velocity smoothing ===
EMA_ALPHA          = 0.25     # 0..1, lower = smoother but slower response
MAX_DAZ_PER_FRAME  = 0.05     # rad/s, hard cap on angular change per frame
MAX_DLX_PER_FRAME  = 0.03     # m/s, hard cap on linear change per frame

# === Blind-chase ===
BLIND_DECAY        = 0.97     # multiply smoothed vel by this each blind frame
BLIND_VEL_FACTOR   = 0.60     # clamp blind-chase velocity to 60% of normal max

# === Lidar-based follow distance ===
FOLLOW_DIST_MIN    = 2.0
FOLLOW_DIST_MAX    = 2.3
CAMERA_FOV_DEG     = 70.0     # UGREEN camera horizontal FOV (approx)
LIDAR_HALF_ARC_DEG = 5.0      # ± angle window around target for averaging
LIDAR_UDP_PORT     = 9998

# === Camera–lidar geometry (for camera overlay) ===
# Tweak these if the projected dots are off in the image.
CAM_FORWARD_M      = 0.095    # camera is THIS far in front of lidar (m)
CAM_VERTICAL_M     = 0.140    # camera is THIS far above lidar scan plane (m)
CAM_LATERAL_M      = -0.050    # camera offset to one side from lidar centreline (m).
                              # Flip sign if dots are shifted the wrong way.
CAM_PITCH_DEG      = 20.0     # camera tilt: + = lens points UP
CAM_YAW_DEG        = 0.5      # camera yaw vs lidar 0° (should be 0)
OVERLAY_MAX_R      = 6.0      # max lidar distance to draw on image (m)

# Stable webcam path (survives /dev/videoN renumbering on reboot).
# Resolves to the UGREEN USB webcam regardless of enumeration order.
CAM_INDEX = "/dev/v4l/by-id/usb-UGREEN_Camera_UGREEN_Camera_SN0001-video-index0"
HTTP_PORT = 5000

def project_lidar_to_image(ranges, amin, ainc, image_w, image_h):
    """Project lidar scan points onto camera image.
    Lidar frame:    x=forward, y=left,  z=up (scan plane at z=0).
    Camera origin:  CAM_FORWARD_M ahead, CAM_VERTICAL_M above, CAM_LATERAL_M to the side.
    Camera frame:   x=right, y=down, z=forward (standard CV convention).
    Camera looks UP by CAM_PITCH_DEG (positive pitch = lens tilted up).
    """
    if ranges is None or ainc <= 0:
        return []
    pitch = np.radians(CAM_PITCH_DEG)
    yaw   = np.radians(CAM_YAW_DEG)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cyw, syw = np.cos(yaw), np.sin(yaw)
    focal = (image_w / 2.0) / np.tan(np.radians(CAMERA_FOV_DEG / 2.0))
    cx_img = image_w / 2.0
    cy_img = image_h / 2.0
    out = []
    n = len(ranges)
    for i in range(n):
        r = ranges[i]
        if r is None or not (0.05 < r < OVERLAY_MAX_R):
            continue
        ang = amin + i * ainc
        # Lidar point in lidar/robot frame
        Xl = r * np.cos(ang)
        Yl = r * np.sin(ang)
        Zl = 0.0
        # Translate so camera is origin (camera sits forward/above/lateral of lidar)
        Xr = Xl - CAM_FORWARD_M
        Yr = Yl - CAM_LATERAL_M
        Zr = Zl - CAM_VERTICAL_M     # lidar is BELOW camera → Zr is negative
        # Yaw around vertical (Z)
        Xy = cyw * Xr + syw * Yr
        Yy = -syw * Xr + cyw * Yr
        Zy = Zr
        # Pitch around the LEFT axis (Y), positive pitch = camera tilts UP.
        # That means a point that was in front (high X) and below (negative Z)
        # rotates INTO the camera view (Z component decreases, etc.)
        Xp = cp * Xy + sp * Zy
        Yp = Yy
        Zp = -sp * Xy + cp * Zy
        # Convert robot axes (X=fwd, Y=left, Z=up) to CV camera axes (x=right, y=down, z=fwd)
        x_cam = -Yp
        y_cam = -Zp
        z_cam =  Xp
        if z_cam <= 0.1:
            continue
        px = focal * x_cam / z_cam + cx_img
        py = focal * y_cam / z_cam + cy_img
        if 0 <= px < image_w and 0 <= py < image_h:
            out.append((int(px), int(py), float(r)))
    return out

def color_by_distance(r):
    """BGR color, green (close) → yellow → red (far)."""
    t = min(r / OVERLAY_MAX_R, 1.0)
    if t < 0.5:
        return (0, 255, int(t * 2 * 255))
    return (0, int((1 - (t - 0.5) * 2) * 255), 255)

STATE_COLORS = {
    RPFState.IDLE:             (128, 128, 128),
    RPFState.REGISTERING:      (255, 200,   0),
    RPFState.IDENTIFICATION:   (  0, 255, 255),
    RPFState.FOLLOWING:        (  0, 255,   0),
    RPFState.SUSPENDED:        (  0, 165, 255),
    RPFState.REIDENTIFICATION: (  0,   0, 255),
}

latest_jpeg      = None
latest_id        = 0
frame_lock       = threading.Lock()
latest_lidar     = None    # bytes blob for browser
lidar_lock       = threading.Lock()
target_angle_deg = None    # angle of followed target relative to robot front, degrees
target_angle_lock= threading.Lock()
click_q          = []
click_lock       = threading.Lock()
reset_request    = False
target_thumbnail = None


def find_bbox_at_click(tracks, point):
    px, py = point
    for tr in tracks:
        x1, y1, x2, y2 = map(int, tr[:4])
        if x1 <= px <= x2 and y1 <= py <= y2:
            return tr[:4]
    return None


def draw_overlay(frame, result, pipeline):
    global target_thumbnail
    state = result["state"]
    color = STATE_COLORS.get(state, (255, 255, 255))
    target_emb = (pipeline._get_target_embedding()
                  if pipeline._initial_embedding is not None else None)
    occluder_ids = result.get("occluder_ids", set())
    caps = result.get("caps", {})

    for tr in result["all_tracks"]:
        x1, y1, x2, y2 = int(tr[0]), int(tr[1]), int(tr[2]), int(tr[3])
        tid = int(tr[4])
        is_target   = tid == result["target_id"]
        is_occluder = tid in occluder_ids
        if   is_target:   c, t = (0, 255, 0), 3
        elif is_occluder: c, t = (0, 128, 255), 2
        else:             c, t = (180, 180, 180), 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, t)
        label = f"ID:{tid}"
        if is_occluder: label += " [OCC]"
        elif target_emb is not None and state in (RPFState.SUSPENDED, RPFState.REIDENTIFICATION):
            emb = pipeline.reid.extract(frame, tr[:4])
            sim = float(np.dot(emb, target_emb))
            label += f" {sim:.2f}"
        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        cap_score = caps.get(tid, 0.0)
        cap_col = (0,140,255) if cap_score >= 0.025 else (120,120,120)
        cap_txt = ("CAP " if cap_score >= 0.025 else "cap ") + f"{cap_score:.2f}"
        cv2.putText(frame, cap_txt, (x1+4, y1+24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cap_col, 2)

    cv2.rectangle(frame, (0,0), (300,30), (0,0,0), -1)
    cv2.putText(frame, f"State: {state.value.upper()}", (5,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if state == RPFState.IDLE:
        cv2.putText(frame, "Click a person to follow", (5,55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 1)
    elif state == RPFState.REGISTERING:
        n      = result.get("reg_diverse_count", 0)
        target = result.get("reg_target", 20)
        prog   = pipeline.registration_progress
        bar_w  = 300
        filled = int(bar_w * prog)
        bar_color = (0, 220, 80) if pipeline.registration_ready else (255, 200, 0)
        cv2.rectangle(frame, (5,40), (5+bar_w,58), (60,60,60), -1)
        cv2.rectangle(frame, (5,40), (5+filled,58), bar_color, -1)
        if pipeline.registration_ready:
            msg = f"Good! {n}/{target} views — keep turning for more coverage"
        else:
            msg = f"Turn slowly... {n}/{target} diverse views captured"
        cv2.putText(frame, msg, (5,75), cv2.FONT_HERSHEY_SIMPLEX, 0.48, bar_color, 1)
    elif state in (RPFState.SUSPENDED, RPFState.REIDENTIFICATION):
        thresh = pipeline.cmoh.sim_threshold
        cv2.putText(frame, f"Searching... threshold={thresh:.2f}", (5,55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)

    if target_thumbnail is not None:
        h, w = frame.shape[:2]
        th, tw = target_thumbnail.shape[:2]
        pad = 8
        y_off, x_off = h-th-pad, pad
        cv2.rectangle(frame, (x_off-2, y_off-18), (x_off+tw+2, y_off+th+2),
                      (50,50,50), -1)
        frame[y_off:y_off+th, x_off:x_off+tw] = target_thumbnail
        cv2.putText(frame, "TARGET", (x_off, y_off-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)


def inference_loop():
    global latest_jpeg, latest_id, reset_request, target_thumbnail

    # --- Lidar UDP receiver (background thread, parses packets from lidar_bridge.py) ---
    lidar_state = {"ranges": None, "amin": 0.0, "ainc": 0.0}
    def _lidar_listen():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("127.0.0.1", LIDAR_UDP_PORT))
        while True:
            try:
                data, _ = s.recvfrom(65535)
                if len(data) < 12: continue
                n, amin, ainc = struct.unpack("<Iff", data[:12])
                if len(data) < 12 + 4*n: continue
                ranges = struct.unpack(f"<{n}f", data[12:12+4*n])
                lidar_state["ranges"] = ranges
                lidar_state["amin"] = amin
                lidar_state["ainc"] = ainc
                global latest_lidar
                with lidar_lock:
                    latest_lidar = data    # raw packet, browser parses same format
            except Exception:
                pass
    threading.Thread(target=_lidar_listen, daemon=True).start()
    print("[Init] Lidar UDP receiver bound on 127.0.0.1:9998")

    def _lidar_distance_at_angle(target_angle_rad, half_window_rad):
        """Return min lidar range within ±half_window around target_angle, or None."""
        ranges = lidar_state["ranges"]
        if ranges is None: return None
        amin = lidar_state["amin"]
        ainc = lidar_state["ainc"]
        if ainc <= 0: return None
        two_pi = 2.0 * np.pi
        ta = target_angle_rad
        while ta >  np.pi: ta -= two_pi
        while ta < -np.pi: ta += two_pi
        best = float("inf")
        for i, r in enumerate(ranges):
            if r is None or r != r or r <= 0.05 or r > 8.0: continue
            ang = amin + i * ainc
            while ang >  np.pi: ang -= two_pi
            while ang < -np.pi: ang += two_pi
            if abs(ang - ta) <= half_window_rad and r < best:
                best = r
        return None if best == float("inf") else best

    print("[Init] UDP cmd_vel sender → 127.0.0.1:9999 (relay must be running)")
    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    CMD_ADDR = ('127.0.0.1', 9999)

    def _stop_motors():
        try:
            zero = struct.pack('<dd', 0.0, 0.0)
            cmd_sock.sendto(zero, CMD_ADDR)
            cmd_sock.sendto(zero, CMD_ADDR)
            print("[Shutdown] sent zero cmd")
        except Exception as e:
            print(f"[Shutdown] {e}")
    atexit.register(_stop_motors)

    print("[Init] Building pipeline...")
    pipeline = FollowPipeline({
        "diversity_threshold":     0.025,
        "reid_threshold":          0.75,
        "cmoh_update_threshold":   0.90,
        "reg_target_frames":       300,
        "reg_min_frames":          150,
        "reg_timeout_frames":      3600,
        "blind_chase_frames":      60,
    })

    # smoothed velocity state (persists across frames)
    smoothed_lx = 0.0
    smoothed_az = 0.0
    print(f"[Init] state={pipeline.state}")

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAM_INDEX}")
    print(f"[Cam] {CAM_INDEX} opened {int(cap.get(3))}x{int(cap.get(4))}")

    frame_times = deque(maxlen=30)
    fps_smooth  = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01); continue

        with click_lock:
            click = click_q.pop(0) if click_q else None
        if reset_request:
            pipeline.state = RPFState.IDLE
            pipeline.target_id = None
            pipeline.tracker.reset()
            pipeline.cmoh.clear()
            pipeline._initial_embedding = None
            target_thumbnail = None
            reset_request = False
            print("[Reset]")

        result = pipeline.process(frame)

        # === Lidar overlay on camera frame ===
        try:
            _r = lidar_state.get("ranges")
            if not hasattr(inference_loop, "_dbg"):
                inference_loop._dbg = 0
            inference_loop._dbg += 1
            if inference_loop._dbg % 30 == 0:
                with open("/tmp/overlay_debug.log", "a") as f:
                    f.write(f"frame {inference_loop._dbg}: ranges is None? {_r is None}, ")
                    f.write(f"len(ranges)={len(_r) if _r else 0}, ")
                    f.write(f"amin={lidar_state.get('amin')}, ainc={lidar_state.get('ainc')}\n")
            if _r is not None:
                pts = project_lidar_to_image(
                    _r, lidar_state["amin"], lidar_state["ainc"],
                    frame.shape[1], frame.shape[0])
                if inference_loop._dbg % 30 == 0:
                    with open("/tmp/overlay_debug.log", "a") as f:
                        f.write(f"  projected pts: {len(pts)}\n")
                for px, py, dist in pts:
                    cv2.circle(frame, (px, py), 5, color_by_distance(dist), -1)
        except Exception as e:
            with open("/tmp/overlay_debug.log", "a") as f:
                f.write(f"OVERLAY ERROR: {e}\n")

        # === Compute raw cmd_vel from target bbox ===
        raw_lx, raw_az = 0.0, 0.0
        target_id = result.get("target_id")
        if pipeline.state == RPFState.FOLLOWING and target_id is not None:
            for tr in result["all_tracks"]:
                if int(tr[4]) == target_id:
                    x1, y1, x2, y2 = tr[:4]
                    fh, fw = frame.shape[:2]
                    box_cx   = (x1 + x2) / 2.0
                    frame_cx = fw / 2.0
                    err_x = box_cx - frame_cx
                    if abs(err_x) > DEAD_ZONE_X:
                        raw_az = float(np.clip(-err_x * ANG_GAIN, -MAX_ANG, MAX_ANG))

                    # === Lidar-based distance (preferred), else bbox-ratio fallback ===
                    norm_x = (box_cx - frame_cx) / (fw / 2.0)   # -1 (left edge) .. +1 (right edge)
                    target_angle = -norm_x * np.radians(CAMERA_FOV_DEG / 2.0)
                    global target_angle_deg
                    with target_angle_lock:
                        target_angle_deg = float(np.degrees(target_angle))
                    half_window = np.radians(LIDAR_HALF_ARC_DEG)
                    dist = _lidar_distance_at_angle(target_angle, half_window)
                    if dist is not None:
                        if dist < FOLLOW_DIST_MIN:
                            raw_lx = 0.0                                            # too close — stop
                        elif dist > FOLLOW_DIST_MAX:
                            err_d = dist - FOLLOW_DIST_MAX
                            raw_lx = float(np.clip(err_d * 0.6,  0.0, MAX_LIN))     # chase
                        # in [MIN, MAX] → raw_lx stays 0 (comfort range)
                    else:
                        ratio = (x2 - x1) / fw
                        err_r = ratio - TARGET_WIDTH_RATIO
                        if abs(err_r) > DEAD_ZONE_RATIO:
                            raw_lx = float(np.clip(-err_r * LIN_GAIN, -MAX_LIN, MAX_LIN))
                    break

        # clear target angle if not actively following
        if pipeline.state != RPFState.FOLLOWING:
            with target_angle_lock:
                target_angle_deg = None

        # === Smoothing + blind-chase ===
        if pipeline.state == RPFState.BLIND_CHASE:
            smoothed_lx *= BLIND_DECAY
            smoothed_az *= BLIND_DECAY
            smoothed_lx = float(np.clip(smoothed_lx, -MAX_LIN * BLIND_VEL_FACTOR,  MAX_LIN * BLIND_VEL_FACTOR))
            smoothed_az = float(np.clip(smoothed_az, -MAX_ANG * BLIND_VEL_FACTOR,  MAX_ANG * BLIND_VEL_FACTOR))
        elif pipeline.state == RPFState.FOLLOWING:
            # EMA toward raw
            target_lx = EMA_ALPHA * raw_lx + (1.0 - EMA_ALPHA) * smoothed_lx
            target_az = EMA_ALPHA * raw_az + (1.0 - EMA_ALPHA) * smoothed_az
            # slew rate limit
            dlx = float(np.clip(target_lx - smoothed_lx, -MAX_DLX_PER_FRAME, MAX_DLX_PER_FRAME))
            daz = float(np.clip(target_az - smoothed_az, -MAX_DAZ_PER_FRAME, MAX_DAZ_PER_FRAME))
            smoothed_lx += dlx
            smoothed_az += daz
        else:
            # IDLE / REGISTERING / IDENTIFICATION / SUSPENDED / REIDENTIFICATION → halt
            smoothed_lx = 0.0
            smoothed_az = 0.0

        lx, az = smoothed_lx, smoothed_az
        cmd_sock.sendto(struct.pack('<dd', lx, az), CMD_ADDR)

        if click is not None:
            bbox = find_bbox_at_click(result["all_tracks"], click)
            if bbox is not None:
                pipeline.register_target(frame, bbox)
                x1, y1, x2, y2 = map(int, bbox[:4])
                crop = frame[max(0,y1):y2, max(0,x1):x2]
                if crop.size > 0:
                    target_thumbnail = cv2.resize(crop, (80, 160))
                print(f"[Click] Registered target at {click}")
            else:
                print(f"[Click] No person at {click}")

        draw_overlay(frame, result, pipeline)

        now = time.time()
        frame_times.append(now)
        if len(frame_times) >= 2:
            fps_smooth = (len(frame_times)-1) / max(1e-6, frame_times[-1] - frame_times[0])
        cv2.putText(frame, f"FPS:{fps_smooth:5.1f}",
                    (frame.shape[1]-130, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        # cmd_vel readout
        cv2.rectangle(frame, (0, frame.shape[0]-26), (260, frame.shape[0]),
                      (0,0,0), -1)
        cmd_color = (0,255,0) if (lx != 0 or az != 0) else (160,160,160)
        cv2.putText(frame, f"lx={lx:+.2f}  az={az:+.2f}",
                    (8, frame.shape[0]-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cmd_color, 2)

        ok2, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok2:
            with frame_lock:
                latest_jpeg = buf.tobytes()
                latest_id  += 1


app  = Flask(__name__)
sock = Sock(app)

INDEX_HTML = """
<!doctype html><html><body style="margin:0;background:#111;color:#eee;
  font-family:system-ui;text-align:center;overflow:hidden">
  <div style="padding:8px">
    <button onclick="fetch('/reset',{method:'POST'})"
            style="padding:6px 14px;font-size:14px">RESET</button>
    <span id="status" style="margin-left:16px;color:#888">
      connecting...
    </span>
  </div>
  <div style="display:flex;gap:8px;justify-content:center;align-items:flex-start;
              flex-wrap:wrap;padding:0 8px">
    <canvas id="cv" style="cursor:crosshair;width:70vw;height:auto;
      max-height:78vh;object-fit:contain;background:#000"></canvas>
    <canvas id="lidar" width="360" height="360"
      style="background:#000;border:1px solid #333"></canvas>
  </div>
  <script>
    const canvas = document.getElementById('cv');
    const ctx    = canvas.getContext('2d', {alpha:false});
    const status = document.getElementById('status');
    let frameCount = 0, lastT = performance.now();

    function connect() {
      const ws = new WebSocket(`ws://${location.host}/ws`);
      ws.binaryType = 'blob';
      ws.onopen    = () => status.textContent = 'connected — click a person to follow';
      ws.onclose   = () => { status.textContent = 'disconnected, retrying...';
                              setTimeout(connect, 1000); };
      ws.onerror   = () => status.textContent = 'error';
      ws.onmessage = async (e) => {
        try {
          const bitmap = await createImageBitmap(e.data);
          if (canvas.width !== bitmap.width) {
            canvas.width  = bitmap.width;
            canvas.height = bitmap.height;
          }
          ctx.drawImage(bitmap, 0, 0);
          bitmap.close();
          frameCount++;
          const now = performance.now();
          if (now - lastT > 1000) {
            const fps = (frameCount * 1000 / (now - lastT)).toFixed(1);
            status.textContent = `connected — browser ${fps} fps`;
            frameCount = 0; lastT = now;
          }
        } catch (err) { /* ignore decode glitches */ }
      };
    }
    connect();

    // ===== Lidar live view =====
    const lcv = document.getElementById('lidar');
    const lctx = lcv.getContext('2d');
    function drawLidar(buffer) {
      const dv = new DataView(buffer);
      const n = dv.getUint32(0, true);
      const amin = dv.getFloat32(4, true);
      const ainc = dv.getFloat32(8, true);
      const W = lcv.width, H = lcv.height;
      const cx = W/2, cy = H/2;
      const maxR = 4.0;
      const scale = (Math.min(W,H)/2 - 8) / maxR;

      lctx.fillStyle = '#000';
      lctx.fillRect(0,0,W,H);
      lctx.strokeStyle = '#222';
      lctx.lineWidth = 1;
      for (let r=1; r<=4; r++) {
        lctx.beginPath();
        lctx.arc(cx, cy, r*scale, 0, Math.PI*2);
        lctx.stroke();
      }
      lctx.strokeStyle = '#333';
      lctx.beginPath(); lctx.moveTo(0,cy); lctx.lineTo(W,cy); lctx.stroke();
      lctx.beginPath(); lctx.moveTo(cx,0); lctx.lineTo(cx,H); lctx.stroke();
      lctx.fillStyle = '#5a5'; lctx.font = '11px system-ui';
      lctx.fillText('FRONT', cx+4, 14);

      lctx.fillStyle = '#7af';
      for (let i=0; i<n; i++) {
        const r = dv.getFloat32(12 + i*4, true);
        if (!isFinite(r) || r <= 0.05 || r > maxR) continue;
        let ang = amin + i*ainc;
        const x = cx + Math.sin(ang) * r * scale;
        const y = cy - Math.cos(ang) * r * scale;
        lctx.fillRect(x-1, y-1, 3, 3);
      }
      lctx.fillStyle = '#fc4';
      lctx.beginPath(); lctx.arc(cx, cy, 4, 0, Math.PI*2); lctx.fill();
    }

    function connectLidar() {
      const lws = new WebSocket(`ws://${location.host}/ws_lidar`);
      lws.binaryType = 'arraybuffer';
      lws.onmessage = (e) => drawLidar(e.data);
      lws.onclose = () => setTimeout(connectLidar, 1000);
    }
    connectLidar();

    canvas.addEventListener('click', (e) => {
      const r = canvas.getBoundingClientRect();
      const x = Math.round((e.clientX - r.left) * (canvas.width  / r.width));
      const y = Math.round((e.clientY - r.top ) * (canvas.height / r.height));
      fetch(`/click?x=${x}&y=${y}`, {method:'POST'});
    });
  </script></body></html>
"""


@app.route('/')
def index(): return INDEX_HTML


@sock.route('/ws_lidar')
def ws_lidar_handler(ws):
    last = None
    while True:
        with lidar_lock:
            data = latest_lidar
        if data is not None and data is not last:
            try:
                ws.send(data); last = data
            except Exception:
                break
        else:
            time.sleep(0.05)

@sock.route('/ws')
def ws_handler(ws):
    last_sent_id = -1
    while True:
        with frame_lock:
            jpg = latest_jpeg
            fid = latest_id
        if jpg is not None and fid != last_sent_id:
            try:
                ws.send(jpg)
                last_sent_id = fid
            except Exception:
                break
        else:
            time.sleep(0.003)


@app.route('/click', methods=['POST'])
def click():
    try:
        x = int(request.args.get('x','0'))
        y = int(request.args.get('y','0'))
    except Exception:
        return jsonify(ok=False), 400
    with click_lock:
        click_q.append((x, y))
    return jsonify(ok=True)


@app.route('/reset', methods=['POST'])
def reset():
    global reset_request
    reset_request = True
    return jsonify(ok=True)


if __name__ == "__main__":
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()
    print(f"\n[Server] Open http://192.168.1.20:{HTTP_PORT}/ on your Mac\n")
    app.run(host='0.0.0.0', port=HTTP_PORT, threaded=True, use_reloader=False)
