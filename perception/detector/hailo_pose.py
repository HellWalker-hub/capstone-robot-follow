"""
Hailo YOLOv8s-Pose detector (OPTIMIZED — vectorized decode).
Same API: detect_with_keypoints() returns {'boxes': (N,5), 'keypoints': (N,17,3)}.
"""
import numpy as np
import cv2
from hailo_platform import (
    InputVStreamParams, OutputVStreamParams,
    InferVStreams, FormatType,
)
from perception.detector.hailo_backend import HailoBackend


NUM_KEYPOINTS = 17
NUM_CLASSES = 1
REG_MAX = 16
INPUT_SIZE = 640
STRIDES = [8, 16, 32]
GRID_SIZES = [80, 40, 20]


def _make_grids():
    grids = []
    for gs in GRID_SIZES:
        yv, xv = np.meshgrid(np.arange(gs), np.arange(gs), indexing='ij')
        grids.append(np.stack([xv, yv], axis=-1).astype(np.float32))
    return grids


_GRIDS = _make_grids()
_DFL_IDX = np.arange(REG_MAX, dtype=np.float32)


def _nms_vectorized(boxes, scores, iou_thresh=0.45):
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[rest] - inter
        iou = inter / (union + 1e-9)
        order = rest[iou <= iou_thresh]
    return np.array(keep, dtype=np.int64)


class HailoPoseDetector:
    PERSON_CLASS = 0

    OUT_NAMES_BY_SCALE = [
        ("yolov8s_pose/conv43", "yolov8s_pose/conv44", "yolov8s_pose/conv45"),
        ("yolov8s_pose/conv57", "yolov8s_pose/conv58", "yolov8s_pose/conv59"),
        ("yolov8s_pose/conv70", "yolov8s_pose/conv71", "yolov8s_pose/conv72"),
    ]

    def __init__(self, hef_path="/home/capstone/hailo_demo/models/yolov8s_pose.hef",
                 conf=0.4, iou_thresh=0.45):
        self.conf = conf
        self.iou_thresh = iou_thresh

        backend = HailoBackend()
        self._hef, self._ngrp = backend.configure(hef_path, "yolov8s_pose")

        in_info = self._hef.get_input_vstream_infos()[0]
        self._in_h, self._in_w, _ = in_info.shape
        self._in_name = in_info.name

        self._in_params  = InputVStreamParams.make(self._ngrp,  format_type=FormatType.UINT8)
        self._out_params = OutputVStreamParams.make(self._ngrp, format_type=FormatType.FLOAT32)

        self._pipe = InferVStreams(self._ngrp, self._in_params, self._out_params)
        self._pipe_ctx = self._pipe.__enter__()

        self._conf_logit = self.conf  # outputs are already sigmoid

        print(f"[Detector-v2] HailoPoseDetector ready ({self._in_w}x{self._in_h}, "
              f"17 keypoints, vectorized decode)")

    def detect_with_keypoints(self, frame: np.ndarray) -> dict:
        fh, fw = frame.shape[:2]

        scale = min(self._in_w / fw, self._in_h / fh)
        new_w, new_h = int(fw * scale), int(fh * scale)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (new_w, new_h))
        pad_x = (self._in_w - new_w) // 2
        pad_y = (self._in_h - new_h) // 2
        padded = np.full((self._in_h, self._in_w, 3), 114, dtype=np.uint8)
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        tensor = np.ascontiguousarray(np.expand_dims(padded, 0))
        results = self._pipe_ctx.infer({self._in_name: tensor})

        all_boxes = []
        all_scores = []
        all_kpts = []

        for scale_idx, (box_n, cls_n, kpt_n) in enumerate(self.OUT_NAMES_BY_SCALE):
            stride = STRIDES[scale_idx]
            gs = GRID_SIZES[scale_idx]
            grid = _GRIDS[scale_idx]

            cls_raw = np.asarray(results[cls_n])[0, ..., 0]
            mask = cls_raw >= self._conf_logit
            if not mask.any():
                continue

            ys, xs = np.where(mask)
            n_keep = len(ys)
            cell_scores_logit = cls_raw[ys, xs]
            cell_scores = cell_scores_logit

            box_raw_full = np.asarray(results[box_n])[0]
            box_kept = box_raw_full[ys, xs].astype(np.float32)
            box_kept = box_kept.reshape(n_keep, 4, REG_MAX)
            box_kept -= box_kept.max(axis=-1, keepdims=True)
            np.exp(box_kept, out=box_kept)
            box_kept /= box_kept.sum(axis=-1, keepdims=True) + 1e-9
            distances = (box_kept * _DFL_IDX).sum(axis=-1)

            anchor_x = grid[ys, xs, 0] + 0.5
            anchor_y = grid[ys, xs, 1] + 0.5
            x1_b = (anchor_x - distances[:, 0]) * stride
            y1_b = (anchor_y - distances[:, 1]) * stride
            x2_b = (anchor_x + distances[:, 2]) * stride
            y2_b = (anchor_y + distances[:, 3]) * stride

            kpt_raw_full = np.asarray(results[kpt_n])[0]
            kpt_kept = kpt_raw_full[ys, xs].astype(np.float32)
            kpt_kept = kpt_kept.reshape(n_keep, NUM_KEYPOINTS, 3)
            kx = (kpt_kept[:, :, 0] * 2.0 - 0.5 + anchor_x[:, None]) * stride
            ky = (kpt_kept[:, :, 1] * 2.0 - 0.5 + anchor_y[:, None]) * stride
            kc = 1.0 / (1.0 + np.exp(-kpt_kept[:, :, 2]))

            all_boxes.append(np.stack([x1_b, y1_b, x2_b, y2_b], axis=-1))
            all_scores.append(cell_scores.astype(np.float32))
            all_kpts.append(np.stack([kx, ky, kc], axis=-1).astype(np.float32))

        if not all_boxes:
            return {"boxes": np.empty((0, 5), dtype=np.float32),
                    "keypoints": np.empty((0, NUM_KEYPOINTS, 3), dtype=np.float32)}

        boxes = np.concatenate(all_boxes, axis=0).astype(np.float32)
        scores = np.concatenate(all_scores, axis=0)
        kpts = np.concatenate(all_kpts, axis=0)

        keep_idx = _nms_vectorized(boxes, scores, self.iou_thresh)
        if len(keep_idx) == 0:
            return {"boxes": np.empty((0, 5), dtype=np.float32),
                    "keypoints": np.empty((0, NUM_KEYPOINTS, 3), dtype=np.float32)}
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        kpts = kpts[keep_idx]

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        np.clip(boxes[:, 0], 0, fw - 1, out=boxes[:, 0])
        np.clip(boxes[:, 1], 0, fh - 1, out=boxes[:, 1])
        np.clip(boxes[:, 2], 0, fw - 1, out=boxes[:, 2])
        np.clip(boxes[:, 3], 0, fh - 1, out=boxes[:, 3])

        kpts[..., 0] = (kpts[..., 0] - pad_x) / scale
        kpts[..., 1] = (kpts[..., 1] - pad_y) / scale

        out_boxes = np.concatenate([boxes, scores[:, None]], axis=1).astype(np.float32)
        return {"boxes": out_boxes, "keypoints": kpts}

    def detect(self, frame: np.ndarray) -> np.ndarray:
        return self.detect_with_keypoints(frame)["boxes"]

    def close(self):
        try: self._pipe.__exit__(None, None, None)
        except Exception: pass

    def __del__(self):
        self.close()
