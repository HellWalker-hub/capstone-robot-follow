"""
Laptop-only person tracker demo — no robot, no zenoh, just webcam.
Click a person to register as target. Tracker + ReID handle re-identification.
The "distance feedback" overlay shows what the robot version WOULD do,
but no commands are sent anywhere.

Run:  python scripts/run_laptop_demo.py
      python scripts/run_laptop_demo.py --source 0      # webcam
      python scripts/run_laptop_demo.py --source video.mp4
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2, argparse
import numpy as np
from perception.pipeline import FollowPipeline, RPFState

# --- Same params as the robot version (for identical visual feel) ---
DEAD_ZONE_X        = 30
ANG_GAIN           = 0.004
MAX_ANG            = 0.5

TARGET_WIDTH_RATIO = 0.35
DEAD_ZONE_RATIO    = 0.04

MAX_LIN            = 0.20
LIN_GAIN           = 1.8

DEFAULT_REID_MODEL     = "osnet_x0_25"
DEFAULT_REID_THRESHOLD = 0.65

STATE_COLORS = {
    RPFState.IDLE:             (128,128,128),
    RPFState.REGISTERING:      (255,200,0),
    RPFState.IDENTIFICATION:   (0,255,255),
    RPFState.FOLLOWING:        (0,255,0),
    RPFState.SUSPENDED:        (0,165,255),
    RPFState.REIDENTIFICATION: (0,0,255),
}

def compute_virtual_cmd(result, frame_w, frame_h):
    """Same math as the robot — but values are only displayed, not sent."""
    if result["state"] != RPFState.FOLLOWING:
        return 0.0, 0.0
    bbox = result.get("target_bbox")
    if bbox is None:
        return 0.0, 0.0
    x1, y1, x2, y2 = bbox

    err_x = ((x1+x2)/2.0) - (frame_w/2.0)
    az = float(np.clip(-err_x*ANG_GAIN, -MAX_ANG, MAX_ANG)) if abs(err_x) > DEAD_ZONE_X else 0.0

    bbox_w_px = max(0, x2 - x1)
    ratio     = bbox_w_px / float(frame_w)
    err_r     = ratio - TARGET_WIDTH_RATIO
    if abs(err_r) > DEAD_ZONE_RATIO:
        lx = float(np.clip(-err_r * LIN_GAIN, -MAX_LIN, MAX_LIN))
    else:
        lx = 0.0
    return lx, az

def draw_overlay(frame, result, pipeline, lx, az):
    state = result["state"]
    color = STATE_COLORS.get(state,(255,255,255))
    fh, fw = frame.shape[:2]
    occluder_ids = result.get("occluder_ids", set())

    for track in result["all_tracks"]:
        x1,y1,x2,y2 = int(track[0]),int(track[1]),int(track[2]),int(track[3])
        tid = int(track[4])
        is_target   = tid==result["target_id"]
        is_occluder = tid in occluder_ids
        c = (0,255,0) if is_target else (0,128,255) if is_occluder else (180,180,180)
        t = 3 if is_target else 2 if is_occluder else 1
        cv2.rectangle(frame,(x1,y1),(x2,y2),c,t)
        cv2.putText(frame,f"ID:{tid}"+(" [OCC]" if is_occluder else ""),
                    (x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,c,1)

    if state == RPFState.FOLLOWING:
        bbox = result.get("target_bbox")
        cx = int((bbox[0]+bbox[2])/2) if bbox is not None else fw//2
        ideal_w = int(TARGET_WIDTH_RATIO * fw)
        glx1 = max(0, cx - ideal_w//2)
        glx2 = min(fw-1, cx + ideal_w//2)
        cv2.line(frame, (glx1, 0), (glx1, fh), (0,255,255), 2)
        cv2.line(frame, (glx2, 0), (glx2, fh), (0,255,255), 2)
        cv2.putText(frame, f"ideal ({TARGET_WIDTH_RATIO:.2f})",
                    (glx1+4, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        if bbox is not None:
            cur = (bbox[2]-bbox[0])/float(fw)
            diff = cur - TARGET_WIDTH_RATIO
            if   diff >  DEAD_ZONE_RATIO: msg, mc = f"TOO CLOSE  cur={cur:.2f} tgt={TARGET_WIDTH_RATIO:.2f}", (0,100,255)
            elif diff < -DEAD_ZONE_RATIO: msg, mc = f"TOO FAR    cur={cur:.2f} tgt={TARGET_WIDTH_RATIO:.2f}", (0,200,255)
            else:                         msg, mc = f"OK         cur={cur:.2f} tgt={TARGET_WIDTH_RATIO:.2f}", (0,255,0)
            cv2.rectangle(frame, (0, fh-28), (fw, fh), (0,0,0), -1)
            cv2.putText(frame, msg, (5, fh-9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, mc, 1)

    cv2.rectangle(frame,(0,0),(320,30),(0,0,0),-1)
    cv2.putText(frame,f"State: {state.value.upper()}",(5,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    cv2.rectangle(frame,(fw-330,0),(fw,30),(0,0,0),-1)
    cv2.putText(frame,f"[demo] Lin:{lx:+.2f}  Ang:{az:+.2f}",(fw-325,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,200),1)

    if state==RPFState.IDLE:
        cv2.putText(frame,"Click a person to follow",(5,55),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,0),1)
    elif state==RPFState.REGISTERING and pipeline:
        n=result.get("reg_diverse_count",0); tgt=result.get("reg_target",20)
        filled=int(300*pipeline.registration_progress)
        bc=(0,220,80) if pipeline.registration_ready else (255,200,0)
        cv2.rectangle(frame,(5,40),(305,58),(60,60,60),-1)
        cv2.rectangle(frame,(5,40),(5+filled,58),bc,-1)
        cv2.putText(frame,f"Turn slowly... {n}/{tgt} views",(5,75),
                    cv2.FONT_HERSHEY_SIMPLEX,0.48,bc,1)
    elif state in (RPFState.SUSPENDED,RPFState.REIDENTIFICATION):
        cv2.putText(frame,"Searching for target...",(5,55),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,165,255),1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",         default="0",
                        help="Camera index (0,1,...) or path to video file")
    parser.add_argument("--reid-model",     default=DEFAULT_REID_MODEL)
    parser.add_argument("--reid-threshold", type=float, default=DEFAULT_REID_THRESHOLD)
    args = parser.parse_args()

    print(f"[Demo] TARGET_WIDTH_RATIO = {TARGET_WIDTH_RATIO:.3f}")
    print(f"[Demo] reid_model        = {args.reid_model}")
    print(f"[Demo] reid_threshold    = {args.reid_threshold:.2f}")
    print(f"[Demo] source            = {args.source}")

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Error] Cannot open source: {args.source}")
        return

    pipeline = FollowPipeline({
        "reid_model":     args.reid_model,
        "reid_threshold": args.reid_threshold,
    })

    clicked_point    = [None]
    target_thumbnail = [None]
    display_w        = [0]

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point[0] = (x, y)

    cv2.namedWindow("Person Tracker Demo", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Person Tracker Demo", mouse_cb)
    print("Click person to follow. q=quit r=reset")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Demo] Source ended or read failed.")
                break

            fh, fw = frame.shape[:2]

            if display_w[0] == 0:
                display_w[0] = 1280
                display_h = int(fh * 1280 / fw)
                cv2.resizeWindow("Person Tracker Demo", 1280, display_h)

            result = pipeline.process(frame)

            if clicked_point[0] is not None:
                dh = int(fh * 1280 / fw)
                px = int(clicked_point[0][0] * fw / 1280)
                py = int(clicked_point[0][1] * fh / dh)
                for track in result["all_tracks"]:
                    x1,y1,x2,y2 = map(int,track[:4])
                    if x1<=px<=x2 and y1<=py<=y2:
                        pipeline.register_target(frame, track[:4])
                        crop = frame[max(0,y1):y2, max(0,x1):x2]
                        if crop.size > 0:
                            target_thumbnail[0] = cv2.resize(crop,(80,160))
                        break
                clicked_point[0] = None

            lx, az = compute_virtual_cmd(result, fw, fh)

            draw_overlay(frame, result, pipeline, lx, az)

            if target_thumbnail[0] is not None:
                th,tw = target_thumbnail[0].shape[:2]
                yo = fh - th - 38
                xo = fw - tw - 8
                try:
                    frame[yo:yo+th, xo:xo+tw] = target_thumbnail[0]
                    cv2.putText(frame,"TARGET",(xo, yo-4),
                                cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1)
                except:
                    pass

            dh = int(fh * 1280 / fw)
            display = cv2.resize(frame,(1280, dh))
            cv2.imshow("Person Tracker Demo", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                pipeline.state=RPFState.IDLE; pipeline.target_id=None
                pipeline.tracker.reset(); pipeline.cmoh.clear()
                target_thumbnail[0] = None
                print("[Demo] Reset.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
