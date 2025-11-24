# Tools API Dockerfile
FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    tree \
    lynx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./

# Install dependencies (including api group for FastAPI)
RUN uv sync --frozen --no-dev --group api

# Copy tools-api module
COPY tools_api/ ./tools_api/

# Expose port
EXPOSE 8001

# Run the application
# Set PYTHONPATH to the parent directory so tools_api package is found correctly
ENV PYTHONPATH=/app/tools_api
CMD ["uv", "run", "uvicorn", "tools_api.main:app", "--host", "0.0.0.0", "--port", "8001"]
