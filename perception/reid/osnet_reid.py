import torch
import numpy as np
import cv2
from torchvision import transforms

try:
    import torchreid
    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False
    print("[ReID] torchreid not found — install with: pip install torchreid")


class OSNetReID:
    """
    OSNet-based ReID embedding extractor.
    Pretrained on MSMT17. Outputs 512-d L2-normalized embeddings.
    """

    def __init__(self, model_name="osnet_x0_25", device=None):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        if not TORCHREID_AVAILABLE:
            raise RuntimeError("torchreid required. pip install torchreid")

        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1041,  # MSMT17 class count
            pretrained=True,
        )
        self.model.eval()
        self.model.to(device)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        print(f"[ReID] OSNet {model_name} on {device}")

    @torch.no_grad()
    def extract(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Extract embedding for a single person crop.
        Args:
            frame: BGR image
            bbox: [x1, y1, x2, y2]
        Returns:
            512-d normalized embedding
        """
        x1, y1, x2, y2 = map(int, bbox[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(512, dtype=np.float32)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        feat = self.model(tensor)
        feat = feat.squeeze().cpu().numpy()
        norm = np.linalg.norm(feat)
        return feat / (norm + 1e-6)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # both L2-normalized already
