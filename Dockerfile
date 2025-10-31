# Dockerfile for VX-RAG
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/

# Create data directories
RUN mkdir -p data/raw/md data/raw/pdf data/raw/txt data/processed data/index

# Expose port for MCP API
EXPOSE 5000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the MCP server
CMD ["python", "src/mcp/server.py"]