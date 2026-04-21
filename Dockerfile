FROM runpod/base:0.6.2-cuda12.2.0

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir runpod

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
