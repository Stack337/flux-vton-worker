FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/app/hf_cache

WORKDIR /app

# System deps for DWPose/OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git && \
    rm -rf /var/lib/apt/lists/*

# PyTorch 2.2.2 + CUDA 12.1 (supports RTX 4090 sm_89)
RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Expose torch's bundled CUDA libs for onnxruntime-gpu
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/torch/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.11/site-packages/nvidia/cusparse/lib:/usr/local/lib/python3.11/site-packages/nvidia/nccl/lib:/usr/local/lib/python3.11/site-packages/nvidia/nvtx/lib:${LD_LIBRARY_PATH}

# fashn-vton + runpod + numpy<2 fix
RUN pip install --no-cache-dir \
    "numpy<2" \
    runpod \
    git+https://github.com/fashn-AI/fashn-vton-1.5.git

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
