FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/app/hf_cache

WORKDIR /app

# System deps for PIL/OpenCV + git for pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git && \
    rm -rf /var/lib/apt/lists/*

# PyTorch 2.2.2 + CUDA 12.1 (supports RTX 3090/4090)
RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

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

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
