FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir \
    runpod \
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
