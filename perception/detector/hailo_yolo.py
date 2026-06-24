"""
Hailo YOLOv11n person detector — uses shared HailoBackend.
"""
import numpy as np
import cv2
from hailo_platform import (
    InputVStreamParams, OutputVStreamParams,
    InferVStreams, FormatType,
)

from perception.detector.hailo_backend import HailoBackend


class HailoYoloDetector:
    PERSON_CLASS = 0

    def __init__(self, hef_path="/home/capstone/hailo_demo/models/yolov11n.hef",
                 conf=0.4):
        self.conf = conf
        backend = HailoBackend()
        self._hef, self._ngrp = backend.configure(hef_path, "yolov11n")

        in_info  = self._hef.get_input_vstream_infos()[0]
        self._in_h, self._in_w, _ = in_info.shape
        self._in_name = in_info.name

        self._in_params  = InputVStreamParams.make(self._ngrp,  format_type=FormatType.UINT8)
        self._out_params = OutputVStreamParams.make(self._ngrp, format_type=FormatType.FLOAT32)

        # Scheduler mode: no explicit .activate() needed
        self._pipe = InferVStreams(self._ngrp, self._in_params, self._out_params)
        self._pipe_ctx = self._pipe.__enter__()

        print(f"[Detector] HailoYoloDetector ready ({self._in_w}x{self._in_h})")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        fh, fw = frame.shape[:2]
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._in_w, self._in_h))
        tensor  = np.ascontiguousarray(np.expand_dims(resized, 0)).astype(np.uint8)

        results = self._pipe_ctx.infer({self._in_name: tensor})
        out = results[next(iter(results.keys()))]
        batch0 = out[0] if isinstance(out, list) else out

        dets = []
        if isinstance(batch0, list):
            cls_dets = batch0[self.PERSON_CLASS] if len(batch0) > self.PERSON_CLASS else None
            if cls_dets is not None and len(cls_dets) > 0:
                for row in cls_dets:
                    score = float(row[4])
                    if score < self.conf: continue
                    y1, x1, y2, x2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    dets.append([x1*fw, y1*fh, x2*fw, y2*fh, score])
        else:
            for row in batch0[self.PERSON_CLASS]:
                score = float(row[4])
                if score < self.conf: continue
                y1, x1, y2, x2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                dets.append([x1*fw, y1*fh, x2*fw, y2*fh, score])

        if not dets:
            return np.empty((0, 5), dtype=np.float32)
        return np.array(dets, dtype=np.float32)

    def close(self):
        try: self._pipe.__exit__(None, None, None)
        except Exception: pass

    def __del__(self):
        self.close()
