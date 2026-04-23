from  Inference_Modules.region_classification import RegionClassifierModel
from pathlib import Path
from PIL import Image
import os

# Comment if not using an AMD RDNA3 GPU
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"


pipeline_dir = Path(__file__).parent
model_dir = pipeline_dir / "Inference_Modules/models/region_classification"
region_classifier_model = RegionClassifierModel(model_dir=model_dir)

# You should handle extracting the image from the request and passing it to the pipeline
# The return should be a dict containing the LLMs response in addition to the image with the bounding box drawn on it
def run_pipeline(image) -> dict:
    # Region Classification
    # Send result via websocket to frontend

    # Fracture Detection
    # Send result via websocket to frontend

    # if fracture
        # Bounding Box (if fracture)
        # Send result to frontend via websocket

    
    # LLM Api Call
    # Send result to frontend via websocket

    #close socket connection
    pass

if __name__ == "__main__":
    # Have an api endpoint triggering the method "run_pipeline", waiting for call
    image_filepath = "/home/omarriad/Documents/10th Sem/DMET 1067 - Deep Learning in Computer Vision/Project Milestone 2/Model Fine-tuning/Datasets - Unsplit/X-Ray Region Classification/Tibia_and_Fibula/fracture_00003.png"
    output = region_classifier_model.predict(Image.open(image_filepath))
    print(output)