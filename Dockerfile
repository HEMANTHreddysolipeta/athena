# ---- 1. build the web UI -------------------------------------------------
FROM node:22.18-alpine AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- 2. the API, serving the UI ------------------------------------------
FROM python:3.13-slim

# libgl/glib: OpenCV (used by the OCR engine) needs them even headless.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend
# CPU-only PyTorch: the GPU build is ~2 GB larger and Athena never uses CUDA.
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Bake the embedding and reranker models into the image, so the container
# works offline and the first question isn't slowed by a download.
ENV HF_HOME=/app/hf-cache
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY backend/ ./
COPY demo_corpus/ /app/demo_corpus/
COPY --from=ui /ui/dist /app/frontend/dist

ENV ATHENA_DATA_DIR=/app/data \
    ATHENA_CHROMA_DIR=/app/index \
    ATHENA_DB_PATH=/app/index/athena.db
EXPOSE 8000
# 0.0.0.0 inside the container; docker-compose publishes it on 127.0.0.1 only.
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
