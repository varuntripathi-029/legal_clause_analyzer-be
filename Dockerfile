# ==============================================================================
# Production Dockerfile for Legal Contract Analyzer RAG Backend
# ==============================================================================
FROM python:3.11-slim

# Set environment variables for Python, caching, and application defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    PADDLE_HOME=/app/.cache/paddle \
    PORT=8000 \
    REDIS_URL=redis://localhost:6379/0

WORKDIR /app

# Install critical system libraries required by OpenCV/PaddleOCR, FAISS, ONNX runtime, and healthchecks,
# plus a local Redis server for multi-worker chat session state (loopback-only, no persistence)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Create non-root system user and cache directories for security & permissions
RUN useradd -u 10001 -m -s /bin/bash appuser && \
    mkdir -p /app/.cache/fastembed /app/.cache/paddle /app/logs && \
    chown -R appuser:appuser /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download fastembed embedding model at build time for instant offline-ready startup
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-base-en-v1.5')"

# Copy application source code and data files
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser data ./data
COPY --chown=appuser:appuser ingestion ./ingestion
COPY --chown=appuser:appuser main.py .
COPY --chown=appuser:appuser .env.example ./.env.example
COPY --chown=appuser:appuser start.sh .
RUN chmod +x start.sh

# Ensure proper ownership of pre-downloaded model cache and all app files
RUN chown -R appuser:appuser /app

# Switch to non-root user for enhanced container security
USER appuser

EXPOSE 8000

# Healthcheck probe using the application's liveness endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Start Redis (loopback sidecar) then the application server with proxy header support
CMD ["./start.sh"]
