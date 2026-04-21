FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System deps for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 git && \
    rm -rf /var/lib/apt/lists/*

# Install fashn-vton from public GitHub + runpod SDK
RUN pip install --no-cache-dir \
    runpod \
    git+https://github.com/fashn-AI/fashn-vton-1.5.git

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
