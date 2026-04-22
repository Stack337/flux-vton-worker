FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/app/hf_cache
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app

# System deps for PIL/OpenCV + git for pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git && \
    rm -rf /var/lib/apt/lists/*

# PyTorch 2.7+ with CUDA 12.6 — supports Ada, Hopper AND Blackwell GPUs
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126

# diffusers from main branch (required for Flux2KleinPipeline)
# requests for URL image loading
RUN pip install --no-cache-dir \
    "numpy<2" \
    git+https://github.com/huggingface/diffusers.git \
    transformers \
    accelerate \
    safetensors \
    sentencepiece \
    protobuf \
    huggingface_hub \
    pillow \
    requests \
    runpod

ARG CACHE_BUST=4
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
