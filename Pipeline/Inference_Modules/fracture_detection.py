import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

class FractureDetectionModel:
    def __init__(self, model_weights):
        self.model_weights = Path(model_weights)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
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
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image: Image.Image) -> bool:
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)
            # Require 60% confidence to classify as fractured to minimize false positives
            is_fractured = probs[0, 1].item() > 0.6
            
        return is_fractured
