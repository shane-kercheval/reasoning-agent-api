# Tools API Dockerfile
FROM python:3.13-slim

# Install system dependencies including GitHub CLI
RUN apt-get update && apt-get install -y \
    git \
    curl \
    tree \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh \
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
COPY tools_api/tools_api/ ./tools_api/

# Expose port
EXPOSE 8001

# Run the application
CMD ["uv", "run", "uvicorn", "tools_api.main:app", "--host", "0.0.0.0", "--port", "8001"]
