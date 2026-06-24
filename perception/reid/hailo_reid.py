"""
Hailo person ReID using repvgg_a0_person_reid_512.
Same OSNetReID interface so pipeline.py needs no changes.

Note: Hailo model takes a single full-body crop and outputs a 512-d embedding.
The head/body part-weighting from YoutuReID is dropped — the Market-1501
training produces strong full-body embeddings on its own.
"""
import numpy as np
import cv2
from hailo_platform import (
    InputVStreamParams, OutputVStreamParams,
    InferVStreams, FormatType,
)

from perception.detector.hailo_backend import HailoBackend

EMBED_DIM = 512
HEF_PATH  = "/home/capstone/hailo_demo/models/repvgg_a0_person_reid_512.hef"


class OSNetReID:
    """Hailo ReID — name kept to match pipeline.py imports."""

    def __init__(self, model_name="repvgg_a0_512", device=None,
                 head_weight=0.6, body_weight=0.4, head_fraction=0.35):
        # Kept as attributes for interface compatibility (pipeline reads them)
        self.head_weight   = head_weight
        self.body_weight   = body_weight
        self.head_fraction = head_fraction
        self.model_name    = model_name
        backend = HailoBackend()
        self._hef, self._ngrp = backend.configure(HEF_PATH, "reid_repvgg_512")

        in_info  = self._hef.get_input_vstream_infos()[0]
        out_info = self._hef.get_output_vstream_infos()[0]
        self._in_h, self._in_w, _ = in_info.shape   # 256, 128, 3
        self._in_name = in_info.name

        self._in_params  = InputVStreamParams.make(self._ngrp,  format_type=FormatType.UINT8)
        # Output is UINT8 from the chip — we'll get raw bytes and dequantize
        self._out_params = OutputVStreamParams.make(self._ngrp, format_type=FormatType.FLOAT32)

        self._pipe = InferVStreams(self._ngrp, self._in_params, self._out_params)
        self._pipe_ctx = self._pipe.__enter__()

        print(f"[ReID] Hailo repvgg_a0_person_reid_512 ({EMBED_DIM}-d) | "
              f"input {self._in_w}x{self._in_h}")

    def extract(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox[:4])
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros(EMBED_DIM, dtype=np.float32)

        crop    = frame[y1:y2, x1:x2]
        rgb     = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._in_w, self._in_h))   # (256, 128, 3)
        tensor  = np.ascontiguousarray(np.expand_dims(resized, 0)).astype(np.uint8)

        results = self._pipe_ctx.infer({self._in_name: tensor})
        raw = results[next(iter(results.keys()))]
        # raw is (1, 512) UINT8 — convert to float and L2-normalize
        emb = np.asarray(raw).reshape(-1).astype(np.float32)
        n = np.linalg.norm(emb)
        return (emb / (n + 1e-6)).astype(np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def close(self):
        try: self._pipe.__exit__(None, None, None)
        except Exception: pass

    def __del__(self):
        self.close()
