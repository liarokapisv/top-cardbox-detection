# CUDA-ready training/inference image for top-cardbox-detection.
# GPU:  docker run --gpus all -v $PWD/dataset:/app/dataset ... 
# CPU:  works unchanged (torch falls back automatically).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
COPY cutouts/ cutouts/
COPY models/ models/
COPY results/ results/
COPY dataset/intrinsics.json dataset/

# dataset/two_lights is mounted at runtime:
#   docker run --gpus all -v /path/to/two_lights:/app/dataset/two_lights ...
CMD ["python3", "-c", "import torch; print('cuda:', torch.cuda.is_available())"]
