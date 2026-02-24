FROM python:3.11-slim

# System deps for OpenCV and FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install VidLens with all extras (no CLIP by default for smaller image)
RUN pip install --no-cache-dir -e ".[yolo,ui]"

# Pre-download YOLOv8 nano weights
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Expose Gradio UI port
EXPOSE 7860

ENTRYPOINT ["vidlens"]
CMD ["--help"]
