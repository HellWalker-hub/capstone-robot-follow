"""
Hailo person-attribute extractor: person_attr_resnet_v1_18 from the Hailo Model Zoo
(ResNet-18 backbone, trained on PETA, 35 attributes).
Extracts a 35-dim attribute probability vector per person crop.

Use for ReID disambiguation: two people with different clothing/accessories
will have very different attribute vectors even when OSNet appearance scores
are close (e.g. similar overall color statistics).

Cosine similarity between attribute vectors is the recommended comparison.
"""
import numpy as np
import cv2
from hailo_platform import (
    InputVStreamParams, OutputVStreamParams,
    InferVStreams, FormatType,
)
from perception.detector.hailo_backend import HailoBackend


# The 35 outputs follow PETA's standard 35-attribute protocol. The pipeline only
# compares whole vectors with cosine similarity, so per-dimension labels are not needed.


class HailoAttributes:
    def __init__(self,
                 hef_path="/home/capstone/hailo_demo/models/person_attr_resnet_v1_18.hef"):
        backend = HailoBackend()
        self._hef, self._ngrp = backend.configure(hef_path, "person_attr_resnet_v1_18")

        in_info = self._hef.get_input_vstream_infos()[0]
        out_info = self._hef.get_output_vstream_infos()[0]
        self._in_h, self._in_w, _ = in_info.shape
        self._in_name = in_info.name
        self._out_name = out_info.name
        self._dim = out_info.shape[0]

        self._in_params  = InputVStreamParams.make(self._ngrp,  format_type=FormatType.UINT8)
        self._out_params = OutputVStreamParams.make(self._ngrp, format_type=FormatType.FLOAT32)

        self._pipe = InferVStreams(self._ngrp, self._in_params, self._out_params)
        self._pipe_ctx = self._pipe.__enter__()

        print(f"[Attr-v2] Hailo person_attr_resnet_v1_18 ({self._dim} attrs) | input {self._in_w}x{self._in_h}")

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def extract(self, frame, bbox):
        """Extract attribute vector for the person inside bbox.
        
        frame: HxWx3 BGR uint8
        bbox: array-like [x1, y1, x2, y2]
        
        Returns: (35,) float32 vector of attribute probabilities in [0,1].
        """
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1)); x2 = max(x1 + 1, min(w, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(y1 + 1, min(h, y2))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(self._dim, dtype=np.float32)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._in_w, self._in_h))
        tensor = np.ascontiguousarray(np.expand_dims(resized, 0))
        out = self._pipe_ctx.infer({self._in_name: tensor})
        raw = np.asarray(out[self._out_name]).flatten().astype(np.float32)
        return self._sigmoid(raw)

    @staticmethod
    def similarity(a, b):
        """Cosine similarity between two attribute vectors."""
        if a is None or b is None:
            return 0.0
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def close(self):
        try: self._pipe.__exit__(None, None, None)
        except Exception: pass

    def __del__(self):
        self.close()
