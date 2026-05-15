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


def draw_overlay(frame, result):
    state = result["state"]
    color = STATE_COLORS.get(state, (255, 255, 255))

    for track in result["all_tracks"]:
        x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
        tid = int(track[4])
        is_target = tid == result["target_id"]
        c = (0, 255, 0) if is_target else (180, 180, 180)
        thickness = 3 if is_target else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, thickness)
        cv2.putText(frame, f"ID:{tid}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

    cv2.rectangle(frame, (0, 0), (300, 30), (0, 0, 0), -1)
    cv2.putText(frame, f"State: {state.value.upper()}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if state == RPFState.IDLE:
        cv2.putText(frame, "Click a person to follow", (5, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)


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
            else:
                print("[Main] No tracked person at that location.")
            clicked_point = None

        draw_overlay(frame, result)
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
