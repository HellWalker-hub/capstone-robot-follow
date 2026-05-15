"""
Download pretrained YoutuReID ONNX weights from HuggingFace.
Run once: python scripts/download_weights.py
Weights saved to: weights/person_reid_youtu_2021nov.onnx

Model: OpenCV YoutuReID — trained for person re-identification (not ImageNet).
Source: https://huggingface.co/opencv/person_reid_youtureid
"""
import os
import sys

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")
WEIGHTS_FILE = "person_reid_youtu_2021nov.onnx"
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, WEIGHTS_FILE)


def download():
    if os.path.exists(WEIGHTS_PATH):
        size_mb = os.path.getsize(WEIGHTS_PATH) / 1e6
        print(f"Weights already exist: {WEIGHTS_PATH} ({size_mb:.1f} MB)")
        return WEIGHTS_PATH

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub -q")
        from huggingface_hub import hf_hub_download

    print(f"Downloading YoutuReID ONNX from HuggingFace → {WEIGHTS_PATH}")
    path = hf_hub_download(
        repo_id="opencv/person_reid_youtureid",
        filename=WEIGHTS_FILE,
        local_dir=WEIGHTS_DIR,
    )
    size_mb = os.path.getsize(path) / 1e6
    print(f"Done. ({size_mb:.1f} MB) — properly ReID-trained, 768-d embeddings.")
    return path


if __name__ == "__main__":
    download()
