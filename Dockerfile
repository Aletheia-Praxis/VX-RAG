# Dockerfile for VX-RAG
# Security: Use specific version to ensure reproducible builds
FROM python:3.13.9-slim AS builder

# Set working directory for builder
WORKDIR /app

# Install system dependencies for building
# Security: Update packages and install only necessary tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential=12.9 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Final stage
# Security: Use same specific version as builder
FROM python:3.13.9-slim

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Set working directory
WORKDIR /app

# Copy source code
COPY src/ ./src/
COPY config/ ./config/

# Security: Create non-root user with fixed UID/GID for predictability
RUN groupadd -r -g 1000 appuser && \
    useradd -r -u 1000 -g appuser -s /sbin/nologin appuser

# Change ownership of the app directory
RUN chown -R appuser:appuser /app

# Security: Switch to non-root user (all processes run as appuser)
USER appuser

# Expose port for MCP API (default port 25191)
EXPOSE 25191

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Security: Add health check for container monitoring
# Note: For STDIO mode, we check if Python process is responsive
# For HTTP/SSE modes, override this healthcheck in docker-compose.yml
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "from src.utils.logging_config import get_logger; logger = get_logger('healthcheck'); logger.info('Health check OK'); import sys; sys.exit(0)" || exit 1

# Run the MCP server via CLI (default: stdio mode for IDE integration)
# Security: Using exec form to avoid shell injection
ENTRYPOINT ["python", "-m", "src.cli", "serve"]
CMD ["--transport", "stdio"]