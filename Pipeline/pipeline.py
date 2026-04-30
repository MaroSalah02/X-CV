from  Inference_Modules.region_classification import RegionClassifierModel
from Pipeline.Inference_Modules.fracture_detection_bounding_box import FractureDetectionBoundingBoxModel
from pathlib import Path
from PIL import Image
import os
from fastapi import FastAPI, WebSocket
from PIL import Image
import io

# Comment if not using an AMD RDNA3 GPU
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"


pipeline_dir = Path(__file__).parent
model_dir = pipeline_dir / "Inference_Modules/models/region_classification"
region_classifier_model = RegionClassifierModel(model_dir=model_dir)

# Initialize fracture detection model with best.pt
fracture_model_path = pipeline_dir / "Inference_Modules/models/fracture_detection/best.pt"
fracture_detection_model = FractureDetectionBoundingBoxModel(model_path=fracture_model_path)
app = FastAPI()

async def run_pipeline(image: str):
    region_result = region_classifier_model.predict(image)
    yield {"step": "region_classification", "result": region_result}

    # Run fracture detection
    fracture_result = fracture_detection_model.predict(image)
    yield {"step": "fracture_detection", "result": fracture_result}

    if fracture_result["has_fracture"]:
        # Detections are already included in fracture_result
        bbox_result = fracture_result["detections"]
        yield {"step": "bounding_box", "result": bbox_result}

    # same goes here now you get it
    llm_output = None
    yield {"step": "llm", "result": {"text": llm_output}}


@app.websocket("/scan")
async def scan(websocket: WebSocket):
    await websocket.accept()
    try:
        image_bytes = await websocket.receive_bytes()
        image = Image.open(io.BytesIO(image_bytes))  # PIL Image, ready to use

        async for update in run_pipeline(image):
            await websocket.send_json(update)

        await websocket.send_json({"step": "pipeline", "status": "complete"})

    except Exception as e:
        await websocket.send_json({"step": "error", "detail": str(e)})
    finally:
        await websocket.close()
        

# if __name__ == "__main__":
#     # Have an api endpoint triggering the method "run_pipeline", waiting for call
#     image_filepath = "/home/omarriad/Documents/10th Sem/DMET 1067 - Deep Learning in Computer Vision/Project Milestone 2/Model Fine-tuning/Datasets - Unsplit/X-Ray Region Classification/Tibia_and_Fibula/fracture_00003.png"
#     output = region_classifier_model.predict(Image.open(image_filepath))
#     print(output)