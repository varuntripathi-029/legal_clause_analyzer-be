FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
# Install backend dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY app ./app
COPY data ./data
COPY main.py .
COPY .env.example ./.env.example

EXPOSE 7860

# Chat sessions are backed by Redis, so we can safely run multiple workers!
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
