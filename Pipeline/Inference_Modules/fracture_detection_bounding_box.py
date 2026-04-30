from PIL import Image
from torch import cuda
from ultralytics import RTDETR
from pathlib import Path


class FractureDetectionBoundingBoxModel:
    def __init__(self, model_path, conf_threshold=0.25):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold #  Confidence threshold for detections (default: 0.25)
        self.model = self.load_model()

    def load_model(self):
        model = RTDETR(str(self.model_path))
        return model

    def predict(self, image) -> dict:
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image object")

        # Run inference
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            imgsz=640,
            verbose=False
        )

        detections = [] # list of detection dicts with confidence, bbox_xyxy, bbox_xywhn
        for r in results:
            for box in r.boxes:
                detection = {
                    "confidence": float(box.conf),
                    "bbox_xyxy": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    "bbox_xywhn": box.xywhn[0].tolist(),  # [x_center, y_center, width, height] normalized
                }
                detections.append(detection)

        return {
            "has_fracture": len(detections) > 0, # bool indicating if fractures were detected
            "detections": detections
        }
