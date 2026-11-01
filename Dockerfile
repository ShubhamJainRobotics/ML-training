# ==============================================================================
# Production Dockerfile for Jaanch ML Inference Service
# Lightweight, secure, non-root user execution
# ==============================================================================

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Install system dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create a non-privileged system user for running the service
RUN groupadd -g 10001 jaanchgroup && \
    useradd -u 10001 -g jaanchgroup -s /bin/false -m jaanchuser

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy service source code and model artifacts
COPY app/ ./app/
COPY models/ ./models/

# Set ownership of all files to the non-root user
RUN chown -R jaanchuser:jaanchgroup /app

# Switch to the non-privileged user
USER jaanchuser

# Expose service port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start the uvicorn production server
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 2"]

