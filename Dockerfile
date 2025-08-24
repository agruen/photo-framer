# Multi-arch Dockerfile for Photo Frame Slideshow Server
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

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