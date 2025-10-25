# syntax=docker/dockerfile:1

# Base image: slim Python for smaller footprint.
FROM python:3.10-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000

# System deps for pdfplumber (requires poppler utils via libjpeg/zlib/freetype/cairo)
# and for building some wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libjpeg62-turbo-dev \
       zlib1g-dev \
       libfreetype6-dev \
       libpng-dev \
       libcairo2 \
       libglib2.0-0 \
       ghostscript \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest first for better layer caching
COPY requirements.txt ./

# Torch is included in requirements; allow CPU-only install by default.
# Users needing CUDA can override at build with appropriate index URL/extra packages.
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy application
COPY main.py ./

# Expose FastAPI port
EXPOSE 8000

# By default, run the API with uvicorn
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

