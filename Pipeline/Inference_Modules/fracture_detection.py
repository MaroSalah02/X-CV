import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from pathlib import Path

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def to_grayscale(img: Image.Image) -> Image.Image:
    """Convert any PIL image (16-bit, palette, RGBA, RGB) to 8-bit grayscale 'L'.
    Mirrors data_preprocessing.to_grayscale in the training workspace."""
    if img.mode in ('I;16', 'I;16B', 'I;16L', 'I'):
        a = np.asarray(img, dtype=np.float32)
        lo, hi = np.percentile(a, [0.5, 99.5])
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        return Image.fromarray(a.astype(np.uint8), 'L')
    if img.mode in ('P', 'LA', 'RGBA', 'PA'):
        img = img.convert('RGBA')
        bg = Image.new('RGBA', img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert('L')


class PadToSquare:
    """Pad the shorter side with black so the whole radiograph survives resizing."""

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = max(w, h)
        if w == h:
            return img
        left, top = (s - w) // 2, (s - h) // 2
        return ImageOps.expand(img, border=(left, top, s - w - left, s - h - top), fill=0)


class FractureDetectionModel:
    # Used for checkpoints saved before preprocessing settings were stored in them
    LEGACY_THRESHOLD = 0.6

    def __init__(self, model_weights):
        self.model_weights = Path(model_weights)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.meta = {}
        self.model = self.load_model()
        # Newer checkpoints carry the preprocessing, threshold and TTA setting they were tuned with
        self.pad_to_square = self.meta.get("preprocessing") == "pad_to_square"
        self.threshold = float(self.meta.get("threshold", self.LEGACY_THRESHOLD))
        self.tta = bool(self.meta.get("tta", False))
        self.transform = self.get_transforms()

    def build_densenet121(self, num_classes: int = 2, dropout_rate: float = 0.5) -> nn.Module:
        model = models.densenet121(weights=None)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        return model

    def load_model(self):
        model = self.build_densenet121()
        model_path = self.model_weights

        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                self.meta = {k: checkpoint[k] for k in ("img_size", "preprocessing", "threshold", "tta")
                             if k in checkpoint}
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            print(f"Warning: Model weights not found at {model_path}")

        model = model.to(self.device)
        model.eval()
        return model

    def get_transforms(self):
        if self.pad_to_square:
            size = int(self.meta.get("img_size", 384))
            return transforms.Compose([
                PadToSquare(),
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def predict(self, image: Image.Image) -> bool:
        if self.pad_to_square:
            image = to_grayscale(image).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(img_tensor), dim=1)
            if self.tta:
                probs = 0.5 * (probs + torch.softmax(self.model(torch.flip(img_tensor, dims=[3])), dim=1))
            # Legacy checkpoints require 60% confidence to minimise false positives;
            # newer ones use the threshold tuned on the validation set
            is_fractured = probs[0, 1].item() > self.threshold

        return is_fractured
