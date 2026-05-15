import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms, models


class OSNetReID:
    """
    MobileNetV3-Small ReID embedding extractor.
    Uses torchvision pretrained weights — no external dependency.
    Outputs 576-d L2-normalized embeddings. Fast on M1 MPS.
    """

    def __init__(self, model_name="mobilenet_v3_small", device=None):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        # strip classifier — use avgpool output as embedding
        self.model = nn.Sequential(*list(backbone.children())[:-1])
        self.model.eval()
        self.model.to(device)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        print(f"[ReID] MobileNetV3-Small on {device}")

    @torch.no_grad()
    def extract(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Extract embedding for a person crop.
        Args:
            frame: BGR image (H, W, 3)
            bbox: [x1, y1, x2, y2]
        Returns:
            L2-normalized embedding vector
        """
        x1, y1, x2, y2 = map(int, bbox[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(576, dtype=np.float32)

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        feat = self.model(tensor)
        feat = feat.squeeze().cpu().numpy().flatten()
        norm = np.linalg.norm(feat)
        return (feat / (norm + 1e-6)).astype(np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # both L2-normalized
