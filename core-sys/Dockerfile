# AI Hedge Fund — Part 9: Cloud Production
# ==========================================
# Dockerfile — Multi-stage production container
#
# Stage 1 (builder): Install all dependencies
# Stage 2 (production): Minimal runtime image
#
# Multi-stage rationale:
#   Development dependencies (gcc, build tools) add ~800MB to the image.
#   The production stage only copies installed packages, not build tools.
#   Result: ~400MB production image vs ~1.2GB naive single-stage.
#
# Security:
#   Runs as non-root user (hedgefund:hedgefund, uid=1000)
#   No SSH keys or credentials baked in — all via environment variables
#   Read-only filesystem where possible
#
# Usage:
#   Build:   docker build -t hedgefund-api:latest .
#   Run:     docker run -p 8000:8000 --env-file .env hedgefund-api:latest
#   Shell:   docker run -it --env-file .env hedgefund-api:latest /bin/bash

# ─── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Build dependencies only needed at install time
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements first (layer caching — rebuilds only when deps change)
COPY requirements.txt .

# Install to a local directory for copying to production stage
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ─── Stage 2: Production ─────────────────────────────────────────────────────
FROM python:3.11-slim AS production

# Metadata labels
LABEL maintainer="AI Hedge Fund <ops@hedgefund.internal>"
LABEL version="1.0.0"
LABEL description="AI Hedge Fund — Production API and Strategy Engine"

# Runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user and group
RUN groupadd -g 1000 hedgefund && \
    useradd -u 1000 -g hedgefund -m -s /bin/bash hedgefund

# Create required directories with correct ownership
RUN mkdir -p /app /app/db /app/logs /app/reports /app/data && \
    chown -R hedgefund:hedgefund /app

WORKDIR /app

# Copy application code
COPY --chown=hedgefund:hedgefund . .

# Switch to non-root user
USER hedgefund

# Environment defaults (override via --env-file or -e flags)
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    ENV=production \
    LOG_LEVEL=INFO \
    WORKERS=2

# Expose API port
EXPOSE 8000

# Health check — polls /health every 30 seconds
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/health || exit 1

# Default command: production API server
CMD ["python", "-m", "uvicorn", "src.api.api_server:create_app", \
     "--factory", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "warning", \
     "--access-log", \
     "--proxy-headers"]
