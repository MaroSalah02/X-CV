from  Inference_Modules.region_classification import RegionClassifierModel
from  Inference_Modules.fracture_detection import FractureDetectionModel
from Inference_Modules.fracture_detection_bounding_box import FractureDetectionBoundingBoxModel
from pathlib import Path
from PIL import Image
import os
from fastapi import FastAPI, WebSocket
from PIL import Image
import io
from LLM_Module import generate_llm_response

# Comment if not using an AMD RDNA3 GPU
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"


pipeline_dir = Path(__file__).parent
model_dir = pipeline_dir / "Inference_Modules/models/region_classification"
region_classifier_model = RegionClassifierModel(model_dir=model_dir)

# Initialize fracture detection model (DenseNet121 for binary classification)
fracture_detection_model_dir = pipeline_dir / "Inference_Modules/models/fracture_detection"
fracture_detection_model = FractureDetectionModel(model_dir=fracture_detection_model_dir)

# Initialize fracture bounding box detection model (RTDETR for bounding boxes)
fracture_bbox_model_path = pipeline_dir / "Inference_Modules/models/fracture_detection_bounding_box/best.pt"
fracture_detection_bouding_box_model = FractureDetectionBoundingBoxModel(model_path=fracture_bbox_model_path)

app = FastAPI()

async def run_pipeline(image: str):
    region_result = region_classifier_model.predict(image)
    yield {"step": "region_classification", "result": region_result}

    # Run fracture detection (binary classification)
    fracture_result = fracture_detection_model.predict(image)
    yield {"step": "fracture_detection", "result": fracture_result}

    # If fracture detected, get bounding boxes
    if fracture_result:
        bbox_result = fracture_detection_bouding_box_model.predict(image)
        yield {"step": "bounding_box", "result": bbox_result}

    # LLM processing
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