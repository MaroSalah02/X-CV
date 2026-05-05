import os
from langchain_google_genai import ChatGoogleGenerativeAI
import dotenv
import base64
from langchain_core.messages import HumanMessage, SystemMessage

dotenv.load_dotenv()


def get_model():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=api_key,
    )


def build_multimodal_message(text, is_fracture, fracture_bone, image_base64=None):
    content = [{"type": "text", "text": text}]
    content.append({
            "type": "text",
            "text": f"Fracture detected in {fracture_bone}" if is_fracture else "No fracture detected"
        })
    if image_base64 and is_fracture:
        content.append({
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{image_base64}",
        })
    
    return HumanMessage(content=content)


def generate_llm_response(user_input, is_fracture=False, fracture_bone=None, image_base64=None):

    messages = [
        SystemMessage(content="""
            You are a medical imaging assistant specialized in interpreting X-ray fracture analysis results.

            You are given:
            - A prediction indicating whether a fracture is present or not
            - The anatomical location (bone/organ) of the fracture if detected
            - An X-ray image with a highlighted bounding box around the suspected fracture region

            Your task is to:
            1. Clearly explain the result in simple, understandable language
            2. Describe what is visible in the image, especially the highlighted region
            3. If a fracture is detected, explain its location and what it might imply
            4. If no fracture is detected, reassure the user while mentioning that subtle issues may still require professional review
            5. Avoid making definitive medical diagnoses or giving treatment plans
            6. Encourage consulting a qualified healthcare professional for confirmation

            Keep the tone:
            - Clear and informative
            - Calm and professional
            - Reassuring, not alarming

            Avoid:
            - Sarcasm or humor
            - Overly technical jargon unless explained simply
            - Definitive claims like "you definitely have" or "you are fine"

            Always prioritize safety and clarity.
        """)
    ]
    
    model = get_model()

    user_message = build_multimodal_message(user_input, is_fracture, fracture_bone, image_base64)
    messages.append(user_message)
    
    response = model.invoke(messages)

    print("AI:", response.content) 
    return response.content 