"""
Person ReID using OpenCV YoutuReID ONNX model.
Proper ReID training (not ImageNet) — discriminates identity not just clothing.

Part-based weighting for kandoora/thobe environments:
- Body embedding dominated by white clothing → low discriminability
- Head region (face, hair, glasses, headwear) → high discriminability
- Combined: head_weight * head_emb + body_weight * full_body_emb
"""
import os
import numpy as np
import cv2
import onnxruntime as ort

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "weights",
                            "person_reid_youtu_2021nov.onnx")
WEIGHTS_PATH = os.path.normpath(WEIGHTS_PATH)

EMBED_DIM = 768  # YoutuReID output dimension


class OSNetReID:
    """
    YoutuReID ONNX inference with part-based head/body weighting.
    Class kept as OSNetReID so pipeline.py needs no changes.
    """

    def __init__(self, model_name="youtureid", device=None,
                 head_weight=0.6, body_weight=0.4,
                 head_fraction=0.35):
        """
        Args:
            head_weight:   contribution of head-region embedding
            body_weight:   contribution of full-body embedding
            head_fraction: top fraction of bbox height treated as head
        """
        self.head_weight = head_weight
        self.body_weight = body_weight
        self.head_fraction = head_fraction

        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(
                f"ReID weights not found: {WEIGHTS_PATH}\n"
                "Run: python scripts/download_weights.py"
            )

        # onnxruntime CPU — fast enough on M1, same binary runs on RPi4/Jetson
        self.session = ort.InferenceSession(
            WEIGHTS_PATH,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # input size from model: (batch, 3, 256, 128)
        self._h, self._w = 256, 128

        print(f"[ReID] YoutuReID ONNX ({EMBED_DIM}-d) | "
              f"head={head_weight:.0%} body={body_weight:.0%}")

    def extract(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Extract part-weighted embedding from a person bounding box.

        Args:
            frame: BGR image (H, W, 3)
            bbox:  [x1, y1, x2, y2]
        Returns:
            EMBED_DIM-d L2-normalized embedding
        """
        x1, y1, x2, y2 = map(int, bbox[:4])
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return np.zeros(EMBED_DIM, dtype=np.float32)

        body_emb = self._embed(frame[y1:y2, x1:x2])

        head_h = max(int((y2 - y1) * self.head_fraction), 20)
        head_emb = self._embed(frame[y1:y1 + head_h, x1:x2])

        combined = self.head_weight * head_emb + self.body_weight * body_emb
        norm = np.linalg.norm(combined)
        return (combined / (norm + 1e-6)).astype(np.float32)

    def _embed(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return np.zeros(EMBED_DIM, dtype=np.float32)

        # preprocess: BGR→RGB, resize, normalize (ImageNet stats)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._w, self._h))
        tensor = resized.astype(np.float32) / 255.0
        tensor -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
        tensor /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = tensor.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

        feat = self.session.run([self.output_name], {self.input_name: tensor})[0]
        feat = feat.flatten()
        norm = np.linalg.norm(feat)
        return (feat / (norm + 1e-6)).astype(np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))
