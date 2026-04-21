FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir runpod

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
