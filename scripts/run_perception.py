"""
Laptop prototype — person following with occlusion recovery.
Click on a person to register as follow target.
Press 'q' to quit, 'r' to reset.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import argparse
from perception.pipeline import FollowPipeline, RPFState

STATE_COLORS = {
    RPFState.IDLE: (128, 128, 128),
    RPFState.IDENTIFICATION: (0, 255, 255),
    RPFState.FOLLOWING: (0, 255, 0),
    RPFState.SUSPENDED: (0, 165, 255),
    RPFState.REIDENTIFICATION: (0, 0, 255),
}

clicked_point = None
target_thumbnail = None  # stores last known crop of registered target


def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)


def find_bbox_at_click(tracks, point):
    px, py = point
    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        if x1 <= px <= x2 and y1 <= py <= y2:
            return track[:4]
    return None


def draw_overlay(frame, result, pipeline=None):
    state = result["state"]
    color = STATE_COLORS.get(state, (255, 255, 255))
    target_emb = None
    if pipeline is not None and pipeline._initial_embedding is not None:
        target_emb = pipeline._get_target_embedding()

    for track in result["all_tracks"]:
        x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
        tid = int(track[4])
        is_target = tid == result["target_id"]
        c = (0, 255, 0) if is_target else (180, 180, 180)
        thickness = 3 if is_target else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, thickness)

        label = f"ID:{tid}"
        # show live similarity score for all tracks during re-id
        if target_emb is not None and state in (RPFState.SUSPENDED, RPFState.REIDENTIFICATION):
            import numpy as _np
            emb = pipeline.reid.extract(frame, track[:4])
            sim = float(_np.dot(emb, target_emb))
            label += f" {sim:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

    cv2.rectangle(frame, (0, 0), (300, 30), (0, 0, 0), -1)
    cv2.putText(frame, f"State: {state.value.upper()}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if state == RPFState.IDLE:
        cv2.putText(frame, "Click a person to follow", (5, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
    elif state in (RPFState.SUSPENDED, RPFState.REIDENTIFICATION):
        thresh = pipeline.cmoh.sim_threshold if pipeline else 0.60
        cv2.putText(frame, f"Searching... threshold={thresh:.2f}", (5, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    # target thumbnail — bottom-left corner
    if target_thumbnail is not None:
        h, w = frame.shape[:2]
        th, tw = target_thumbnail.shape[:2]
        pad = 8
        y_off, x_off = h - th - pad, pad
        cv2.rectangle(frame, (x_off - 2, y_off - 18), (x_off + tw + 2, y_off + th + 2), (50, 50, 50), -1)
        frame[y_off:y_off+th, x_off:x_off+tw] = target_thumbnail
        cv2.putText(frame, "TARGET", (x_off, y_off - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def main():
    global clicked_point

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0, help="Camera index or video path")
    parser.add_argument("--reid-model", default="osnet_x0_25")
    parser.add_argument("--reid-threshold", type=float, default=0.55)
    args = parser.parse_args()

    config = {
        "reid_model": args.reid_model,
        "reid_threshold": args.reid_threshold,
    }

    pipeline = FollowPipeline(config)

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Error] Cannot open source: {args.source}")
        return

    cv2.namedWindow("Robot Follow")
    cv2.setMouseCallback("Robot Follow", mouse_callback)
    print("Click on a person to follow. 'q' quit, 'r' reset.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = pipeline.process(frame)

        if clicked_point is not None:
            bbox = find_bbox_at_click(result["all_tracks"], clicked_point)
            if bbox is not None:
                pipeline.register_target(frame, bbox)
                # save thumbnail of registered target
                global target_thumbnail
                x1, y1, x2, y2 = map(int, bbox[:4])
                crop = frame[max(0,y1):y2, max(0,x1):x2]
                if crop.size > 0:
                    target_thumbnail = cv2.resize(crop, (80, 160))
            else:
                print("[Main] No tracked person at that location.")
            clicked_point = None

        draw_overlay(frame, result, pipeline)
        cv2.imshow("Robot Follow", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            pipeline.state = RPFState.IDLE
            pipeline.target_id = None
            pipeline.tracker.reset()
            pipeline.cmoh.clear()
            print("[Main] Reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
