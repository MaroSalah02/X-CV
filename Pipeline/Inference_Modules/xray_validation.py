from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms


def to_rgb(img: Image.Image) -> Image.Image:
    """Any PIL image -> 8-bit RGB. 16-bit radiographs are windowed (0.5-99.5 percentile) first,
    exactly as during training (Model Fine-tuning/Models Workspaces/X-Ray Validation Gate)."""
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        a = np.asarray(img, dtype=np.float32)
        lo, hi = np.percentile(a, [0.5, 99.5])
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        return Image.fromarray(a.astype(np.uint8), "L").convert("RGB")
    if img.mode in ("P", "LA", "RGBA", "PA"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


class PadToSquare:
    """Pad the shorter side with black so the whole image survives resizing."""

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = max(w, h)
        if w == h:
            return img
        left, top = (s - w) // 2, (s - h) // 2
        return ImageOps.expand(img, border=(left, top, s - w - left, s - h - top), fill=0)


class XrayValidationModel:
    """
    Binary gate that checks whether an upload is a plain radiograph (X-ray image).
    The pipeline runs it first and stops for anything else (photos, people, documents,
    CT / MRI / ultrasound, ...), because the region and fracture models always return
    an answer, even for images they were never meant to see.
    """

    def __init__(self, model_weights):
        self.model_weights = Path(model_weights)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.meta = {}
        self.model = self.load_model()
        self.threshold = float(self.meta.get("threshold", 0.5))
        size = int(self.meta.get("img_size", 224))
        self.transform = transforms.Compose([
            PadToSquare(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.meta.get("mean", [0.485, 0.456, 0.406]),
                                 std=self.meta.get("std", [0.229, 0.224, 0.225])),
        ])

    def load_model(self):
        if not self.model_weights.exists():
            print(f"Warning: X-ray validation weights not found at {self.model_weights}; "
                  "every upload will be treated as an X-ray")
            return None
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier[1].in_features, 1))
        checkpoint = torch.load(self.model_weights, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.meta = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
        model = model.to(self.device)
        model.eval()
        return model

    def predict(self, image: Image.Image) -> dict:
        if self.model is None:
            return {"is_xray": True, "xray_probability": None}
        x = self.transform(to_rgb(image)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(x)).item()
        return {"is_xray": prob >= self.threshold, "xray_probability": round(prob, 4)}
