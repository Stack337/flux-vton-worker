FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git && \
    rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA 12.1 (bundles its own CUDA runtime)
RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Make torch's CUDA libs visible to onnxruntime-gpu
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/torch/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}

# fashn-vton + runpod
RUN pip install --no-cache-dir \
    runpod \
    git+https://github.com/fashn-AI/fashn-vton-1.5.git

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
