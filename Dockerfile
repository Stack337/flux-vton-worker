FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/app/hf_cache

WORKDIR /app

# Python 3.10 (default in Ubuntu 22.04) + deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libgl1 libglib2.0-0 git && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# PyTorch CUDA 12.1
RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

# fashn-vton + runpod
RUN pip install --no-cache-dir \
    runpod \
    git+https://github.com/fashn-AI/fashn-vton-1.5.git

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
