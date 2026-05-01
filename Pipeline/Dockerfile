FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV (needed by ultralytics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and models
COPY . .

ENV PIPELINE_HOST=0.0.0.0
ENV PIPELINE_PORT=8000

EXPOSE 8000

CMD ["python", "main.py"]
