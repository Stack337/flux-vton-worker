FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System deps for opencv + git for pip install from github
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 git && \
    rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA 12.1 (pip wheel bundles CUDA runtime)
RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

# fashn-vton + runpod SDK
RUN pip install --no-cache-dir \
    runpod \
    git+https://github.com/fashn-AI/fashn-vton-1.5.git

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
