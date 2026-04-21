FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/hf_cache

WORKDIR /app

RUN pip install --no-cache-dir \
    runpod>=1.7.0 \
    diffusers>=0.35.0 \
    transformers>=4.50.0 \
    accelerate>=1.5.0 \
    safetensors \
    sentencepiece \
    protobuf \
    peft>=0.14.0 \
    Pillow \
    huggingface_hub

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
