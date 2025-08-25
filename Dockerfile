# Multi-arch Dockerfile for Photo Frame Slideshow Server
FROM python:3.11-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies - minimal set that works on both amd64 and arm64
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    libglib2.0-0 \
    libgomp1 \
    libjpeg-dev \
    libpng-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements-server.txt .

# Install Python packages with increased timeout and no cache for better compatibility
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout=1000 -r requirements-server.txt

# Copy existing photo processing modules
COPY face_crop_tool.py .
COPY slideshow_generator.py .

# Copy new server application
COPY app/ ./app/

# Create necessary directories
RUN mkdir -p uploads slideshows db temp

# Set permissions
RUN chmod -R 755 /app

# Create non-root user for security
RUN useradd -r -s /bin/false slideshow && \
    chown -R slideshow:slideshow /app
USER slideshow

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "app.app"]