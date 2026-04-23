from PIL import Image, ImageOps
from torch import cuda
from transformers import AutoModelForImageClassification, AutoImageProcessor, MobileNetV2Config
from pathlib import Path

class RegionClassifierModel():
    def __init__(self, model_dir, padding_transformation=True):
        self.output_dir = Path(model_dir)
        self.model = self.load_model()
        self.image_preprocessor_object = self.load_image_preprocessor()
        self.padding_transformation = padding_transformation

    def load_model(self):
        config = MobileNetV2Config.from_pretrained(self.output_dir)
        model = AutoModelForImageClassification.from_pretrained(self.output_dir, config=config)
        model = model.to("cuda" if cuda.is_available() else "cpu")
        return model

    def load_image_preprocessor(self):
        image_processor = AutoImageProcessor.from_pretrained(self.output_dir)
        return image_processor
    
    def predict(self, image) -> str:
        image = self.preprocess_image(image)
        inputs = self.image_preprocessor_object(images=image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.device)
        outputs = self.model(**inputs)
        predicted_class_idx = outputs.logits.argmax(-1).item()
        predicted_class_label = self.model.config.id2label[predicted_class_idx]
        return predicted_class_label

    def preprocess_image(self, image):
        target_size = (self.model.config.image_size, self.model.config.image_size)

        if image.mode != "RGB":
            image = image.convert("RGB")

        if not self.padding_transformation:
            image = image.resize(target_size, Image.BILINEAR)
        
        # If padding transformation, do the following

        ## Get aspect ratio of the image and the target image's aspect ratio
        image_aspect_ratio = image.width / image.height
        target_aspect_ratio = target_size[0] / target_size[1]
        if image_aspect_ratio == target_aspect_ratio:
            image = image.resize(target_size, Image.BILINEAR)
            return image
        
        longer_dimension = 0 if image.width > image.height else 1
        ## Scale the image up/down so that it's longer dimension is the same as the target size in that dimension, maintaining aspect ratio
        if longer_dimension == 0:
            new_width = target_size[0]
            new_height = int(image.height * (new_width / image.width))
        else:
            new_height = target_size[1]
            new_width = int(image.width * (new_height / image.height))
        image = image.resize((new_width, new_height), Image.BILINEAR)
        ## Pad the shorter dimension so that the final image is the target size, select padding from the image edges, pad evenly left and right/ top and bottom
        padding_left = (target_size[0] - new_width) // 2
        padding_right = target_size[0] - new_width - padding_left
        padding_top = (target_size[1] - new_height) // 2
        padding_bottom = target_size[1] - new_height - padding_top
        image = ImageOps.expand(image, border=(padding_left, padding_top, padding_right, padding_bottom), fill=0)
        return image