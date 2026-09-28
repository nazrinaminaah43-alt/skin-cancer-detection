# Use slim Python 3.11 image for optimal size and pre-built ML wheels
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

# Install system dependencies required for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies first for efficient layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project source code, models, and static assets
COPY . .

# Ensure upload and runtime directories exist
RUN mkdir -p data/uploads data/samples models

# Expose default web ports (5000 for standard Flask/Docker, 7860 for Hugging Face Spaces)
EXPOSE 5000 7860

# Start production Gunicorn server, automatically adapting to $PORT
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --threads 2 --timeout 120 run:app"]
