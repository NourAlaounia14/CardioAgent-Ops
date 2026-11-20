FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY rag_engine/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY rag_engine/ ./rag_engine/

EXPOSE 8001

CMD ["uvicorn", "rag_engine.api:app", "--host", "0.0.0.0", "--port", "8001"]