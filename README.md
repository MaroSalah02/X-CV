# X-CV: X-Ray Computer Vision Pipeline

![X-CV](https://img.shields.io/badge/Project-X--CV-blue)

**X-CV** is an end-to-end computer vision and web application pipeline designed for automated X-Ray analysis. It integrates multi-stage inference modules to classify X-Ray regions, localize potential issues using bounding boxes, and accurately detect fractures.

**[View the Project Demo here!](https://marosalah02.github.io/X-CV/)**

## Project Structure

- **`Frontend/`**: Contains the React/Vite-based user interface (`xcv-app`) where users can interact with the system, upload X-rays, and view inference results.
- **`Model Fine-tuning/`**: The research and training workspace. Contains datasets, Python scripts, and Jupyter notebooks for:
  - **X-Ray Region Classification**
  - **X-Ray Bounding Box Extraction** (Includes models like Faster R-CNN, RTDETR, and YOLO)
  - **X-Ray Fracture Detection** 
- **`Pipeline/`**: The core Python backend pipeline. It orchestrates the model inference, integrates LLM capabilities (`LLM_Module.py`), and coordinates the end-to-end processing. It also includes a `Dockerfile` for easy containerization.

## Key Features

- **Region Classification**: Automatically identifies the body part/region shown in the X-Ray.
- **Fracture Bounding Box Extraction**: Localizes potential fractures within the image using state-of-the-art object detection models.
- **Fracture Detection & Classification**: Confirms the presence of fractures in the localized regions.
- **LLM Integration**: Enhances the analytical output or generates readable summaries.
- **Modern Web UI**: An intuitive frontend built with React and Vite for seamless user interaction.

## Getting Started

### Prerequisites
- Node.js & npm (for Frontend)
- Python 3.9+ (for Pipeline/Backend)
- Docker (optional, for containerized deployment)

### 1. Pipeline / Backend Setup
1. Navigate to the pipeline directory:
   ```bash
   cd Pipeline
   ```
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main pipeline service:
   ```bash
   python main.py
   ```
   *(Alternatively, use the provided `Dockerfile` to build and run the backend container).*

### 2. Frontend Setup
1. Navigate to the frontend app:
   ```bash
   cd Frontend/xcv-app
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
