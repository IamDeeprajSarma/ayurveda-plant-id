"""Demo predictor (no Django). Same idea as backend/identify/predictor.py.

PREDICTOR=auto (default): use the trained model if models/efficientnet_b0_best.pt exists, else dummy.
PREDICTOR=torch forces the real model, PREDICTOR=dummy forces placeholder answers.
MODEL_PATH overrides the checkpoint location.
"""
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
TOP_K = 3
DEFAULT_CLASSES = ["tulsi", "neem", "centella", "eclipta", "mentha", "jamun", "aloe_vera", "justicia",
                   "andrographis", "aegle", "kalanchoe", "boerhavia", "hibiscus", "catharanthus",
                   "houttuynia", "clitoria", "hygrophila", "leucas", "guava", "betel", "nyctanthes",
                   "bacopa", "tinospora", "vitex", "gymnema", "clerodendrum", "haritaki", "eryngium",
                   "euphorbia", "cissus"]


def open_image(file_obj):
    """Open an upload, apply the phone's rotation metadata, return an RGB PIL image."""
    return ImageOps.exif_transpose(Image.open(file_obj)).convert("RGB")


def status_for(top3, threshold):
    return "confident" if top3[0][1] >= threshold else "not_sure"


class DummyPredictor:
    """Placeholder answers. The page shows a visible 'demo mode' badge when this is active."""
    kind = "dummy"

    def __init__(self, classes):
        self.classes = classes

    def predict(self, image):
        # vary the answer a little with the image so the demo is not one fixed card
        k = (image.getpixel((image.width // 2, image.height // 2))[1]) % max(len(self.classes) - 2, 1)
        c = self.classes
        return [(c[k], 0.88), (c[k + 1], 0.07), (c[k + 2], 0.03)]


class TorchPredictor:
    kind = "torch"

    def __init__(self, checkpoint_path):
        import torch
        sys.path.insert(0, str(ROOT / "ml"))
        import common  # ml/common.py: build_model and the SAME transforms used in training
        ck = torch.load(checkpoint_path, map_location="cpu")
        self.classes = ck["classes"]
        model, _ = common.build_model(ck["arch"], len(self.classes), pretrained=False)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        self.torch, self.model, self.tf, self.arch = torch, model, common.eval_tf(ck["img"]), ck["arch"]

    def predict(self, image):
        with self.torch.no_grad():
            x = self.tf(image).unsqueeze(0)
            probs = self.torch.softmax(self.model(x), dim=1)[0]
            p, idx = probs.topk(min(TOP_K, len(self.classes)))
        return [(self.classes[i], float(v)) for v, i in zip(p.tolist(), idx.tolist())]


def load():
    choice = os.environ.get("PREDICTOR", "auto")
    path = Path(os.environ.get("MODEL_PATH", ROOT / "models" / "efficientnet_b0_best.pt"))
    if choice == "torch" or (choice == "auto" and path.exists()):
        return TorchPredictor(path)
    cj = ROOT / "models" / "classes.json"
    classes = json.load(open(cj, encoding="utf-8")) if cj.exists() else DEFAULT_CLASSES
    return DummyPredictor(classes)
