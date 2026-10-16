from  Inference_Modules.region_classification import RegionClassifierModel
from  Inference_Modules.fracture_detection import FractureDetectionModel
from Inference_Modules.fracture_detection_bounding_box import FractureDetectionBoundingBoxModel
from Inference_Modules.xray_validation import XrayValidationModel
from pathlib import Path
from PIL import Image
import os
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, UnidentifiedImageError
import io
import base64
from LLM_Module import generate_llm_response
import asyncio

# Uncomment if using an AMD RDNA3 GPU locally
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"


pipeline_dir = Path(__file__).parent

# Gate that rejects uploads which are not X-rays before any other model runs
xray_validation_weights = pipeline_dir / "Inference_Modules/models/xray_validation/xray_gate.pt"
xray_validation_model = XrayValidationModel(model_weights=xray_validation_weights)
NOT_XRAY_MESSAGE = ("This image does not look like an X-ray that X-CV can analyze, so it was not analyzed. "
                    "Please upload an X-ray of a bone or the chest.")

model_dir = pipeline_dir / "Inference_Modules/models/region_classification"
region_classifier_model = RegionClassifierModel(model_dir=model_dir)

fracture_model_weights = pipeline_dir / "Inference_Modules/models/fracture_detection/checkpoint_best.pt"
fracture_detection_model = FractureDetectionModel(model_weights=fracture_model_weights)

# Initialize fracture bounding box detection model (RTDETR for bounding boxes)
fracture_bbox_model_path = pipeline_dir / "Inference_Modules/models/fracture_detection_bounding_box/best.pt"
fracture_detection_bouding_box_model = FractureDetectionBoundingBoxModel(model_path=fracture_bbox_model_path)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def run_pipeline(image: Image.Image):
    xray_check = xray_validation_model.predict(image)
    yield {"step": "xray_check", "result": xray_check}
    if not xray_check["is_xray"]:
        # The models below always return an answer, even for photos or documents, so stop here
        return

    region_result = region_classifier_model.predict(image)
    yield {"step": "region_classification", "result": region_result}

    # Run fracture detection 
    fracture_result = fracture_detection_model.predict(image)
    yield {"step": "fracture_detection", "result": fracture_result}

    image_base64 = None

    # If fracture detected, get bounding boxes
    if fracture_result:
        bbox_result = fracture_detection_bouding_box_model.predict(image)
        
        if bbox_result.get("has_fracture"):
            draw = ImageDraw.Draw(image)
            for det in bbox_result.get("detections", []):
                x1, y1, x2, y2 = det["bbox_xyxy"]
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        bbox_result["image_base64"] = image_base64
        yield {"step": "bounding_box", "result": bbox_result}

    # LLM processing (run in thread to avoid blocking the event loop)
    llm_prompt = (
        f"Region classification result: {region_result}. "
        f"Fracture detected: {'yes' if fracture_result else 'no'}. "
        "Generate a short, clear clinical summary for the user."
    )
    try:
        llm_output = await asyncio.to_thread(
            generate_llm_response,
            llm_prompt,
            is_fracture=fracture_result,
            fracture_bone=region_result,
            image_base64=image_base64,
        )
    except Exception as e:
        llm_output = f"LLM error: {e}"

    yield {"step": "llm", "result": {"text": llm_output}}


@app.websocket("/scan")
async def scan(websocket: WebSocket):
    await websocket.accept()
    try:
        image_bytes = await websocket.receive_bytes()
        try:
            image = Image.open(io.BytesIO(image_bytes))  # PIL Image, ready to use
            image.load()
        except (UnidentifiedImageError, OSError):
            await websocket.send_json({"step": "error", "detail": "The uploaded file is not a readable image. "
                                                                  "Please upload an X-ray as PNG or JPEG."})
            return

        rejected = False
        async for update in run_pipeline(image):
            await websocket.send_json(update)
            if update["step"] == "xray_check" and not update["result"]["is_xray"]:
                rejected = True

        if rejected:
            await websocket.send_json({"step": "pipeline", "status": "rejected", "detail": NOT_XRAY_MESSAGE})
        else:
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