FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY audio_engine/requirements-service.txt /tmp/requirements-service.txt
RUN pip install --no-cache-dir -r /tmp/requirements-service.txt

COPY audio_engine/main.py audio_engine/preprocessing.py ./audio_engine/
COPY audio_engine/cardio_model.onnx ./audio_engine/cardio_model.onnx
COPY data/processed/scaler.joblib ./data/processed/scaler.joblib

ENV MODEL_PATH=/app/audio_engine/cardio_model.onnx \
    SCALER_PATH=/app/data/processed/scaler.joblib \
    MODEL_INPUT=mel

EXPOSE 8000

CMD ["uvicorn", "audio_engine.main:app", "--host", "0.0.0.0", "--port", "8000"]
