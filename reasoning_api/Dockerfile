# Dockerfile.api - Main Reasoning Agent API
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml ./
COPY uv.lock ./

# Install Python dependencies
RUN uv sync --group api --no-dev

# Copy application code
COPY reasoning_api/reasoning_api/ ./reasoning_api/
COPY reasoning_api/prompts/ ./prompts/
COPY config/ ./config/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["uv", "run", "uvicorn", "reasoning_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
