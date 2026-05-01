from  Inference_Modules.region_classification import RegionClassifierModel
from  Inference_Modules.fracture_detection import FractureDetectionModel
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

fracture_model_weights = pipeline_dir / "Inference_Modules/models/fracture_detection/checkpoint_best.pt"
fracture_detection_model = FractureDetectionModel(model_weights=fracture_model_weights)

app = FastAPI()

async def run_pipeline(image: str):
    region_result = region_classifier_model.predict(image)
    yield {"step": "region_classification", "result": region_result}

    # Getting the output of the fracture detection model
    fracture_result = fracture_detection_model.predict(image)

    yield {"step": "fracture_detection", "result": fracture_result}

    if fracture_result:
        #should be the output of the bounding box model, but for testing we set it to None
        bbox_result = None
        yield {"step": "bounding_box", "result": bbox_result}

    llm_prompt = (
        f"Region classification result: {region_result}. "
        f"Fracture detected: {'yes' if fracture_result else 'no'}. "
        "Generate a short, clear clinical summary for the user."
    )
    llm_output = generate_llm_response(
        llm_prompt,
        is_fracture=fracture_result,
        fracture_bone=region_result,
    )
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