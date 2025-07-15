# Docker Compose Setup Guide

This document explains how to run the Reasoning Agent API project using Docker Compose, making it easy to deploy all services together.

## 🚀 Quick Start

1. **Copy the environment file**:
   ```bash
   cp .env.dev.example .env
   ```

2. **Configure your OpenAI API key**:
   ```bash
   # Edit .env file
   OPENAI_API_KEY=your-openai-api-key-here
   ```

3. **Start all services**:

   ```bash
   # Starts all services with Docker Compose (dev mode with hot reload)...
   make docker_up
   ```

4. **Access the services**:
   - **Web Interface**: http://localhost:8080
   - **API Documentation**: http://localhost:8000/docs
   - **API Health Check**: http://localhost:8000/health
   - **MCP Server**: http://localhost:8001/mcp/

## 📋 Services Overview

The Docker Compose setup includes three main services:

### 1. Reasoning API (`reasoning-api`)

- **Port**: 8000
- **Description**: Main API service with OpenAI compatibility
- **Health Check**: http://localhost:8000/health
- **Documentation**: http://localhost:8000/docs

### 2. Web Client (`web-client`)

- **Port**: 8080
- **Description**: MonsterUI web interface
- **Health Check**: http://localhost:8080/health
- **Interface**: http://localhost:8080

### 3. Fake MCP Server (`fake-mcp-server`)

- **Port**: 8001
- **Description**: Demo MCP server with fake tools
- **Health Check**: http://localhost:8001/
- **MCP Endpoint**: http://localhost:8001/mcp/

## 🔧 Configuration

### Environment Variables

The main configuration is in the `.env` file (unified for all services):

```bash
# Required
OPENAI_API_KEY=your-openai-api-key-here
API_TOKENS=web-client-dev-token,admin-dev-token,mobile-dev-token
WEB_CLIENT_TOKEN=web-client-dev-token
REASONING_API_URL=http://localhost:8000
REQUIRE_AUTH=false

# Optional (defaults provided)
WEB_CLIENT_PORT=8080
HTTP_CONNECT_TIMEOUT=5.0
HTTP_READ_TIMEOUT=30.0
HTTP_WRITE_TIMEOUT=10.0
```

### MCP Configuration

The MCP servers are configured in `config/mcp_servers.docker.yaml`. This file uses Docker service names for internal networking:

```yaml
servers:
  - name: fake-mcp-server
    url: http://fake-mcp-server:8001/mcp/
    enabled: true
```

## 🛠️ Development Workflow

### Building and Running

```bash
# Build all services
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Individual Service Management

```bash
# Start only specific services
docker-compose up -d reasoning-api fake-mcp-server

# Rebuild a specific service
docker-compose build reasoning-api

# View logs for specific service
docker-compose logs -f web-client
```

### Development Mode

For development with code changes:

```bash
# Stop the services
docker-compose down

# Rebuild after code changes
docker-compose build

# Start with logs visible
docker-compose up
```

## 🔍 Testing the Setup

### 1. Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Check web client health
curl http://localhost:8080/health

# Check MCP server health
curl http://localhost:8001/
```

### 2. API Testing

```bash
# Test the API directly
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer web-client-dev-token" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is the weather like?"}],
    "stream": false
  }'
```

### 3. Web Interface Testing

1. Open http://localhost:8080 in your browser
2. Try asking: "What's the weather like in Paris?"
3. The system should use the fake MCP server to provide weather data

## 🔧 Troubleshooting

### Common Issues

1. **Port conflicts**:
   ```bash
   # Check if ports are in use
   lsof -i :8000 :8080 :8001
   
   # Kill processes if needed
   make cleanup
   ```

2. **Permission issues**:
   ```bash
   # Fix Docker permissions
   sudo chown -R $USER:$USER .
   ```

3. **Service not starting**:
   ```bash
   # Check logs
   docker-compose logs service-name
   
   # Check health
   docker-compose ps
   ```

### Debug Mode

```bash
# Run with verbose logging
docker-compose up --build

# Check specific service logs
docker-compose logs -f reasoning-api
```

## 🔄 Updates and Maintenance

### Updating Dependencies

```bash
# Update base images
docker-compose pull

# Rebuild with latest changes
docker-compose build --no-cache

# Restart services
docker-compose down && docker-compose up -d
```

### Adding New MCP Servers

1. **Create a new Dockerfile** (e.g., `Dockerfile.new-mcp`)
2. **Add service to docker-compose.yml**:
   ```yaml
   new-mcp-server:
     build:
       context: .
       dockerfile: Dockerfile.new-mcp
     ports:
       - "8002:8002"
     networks:
       - reasoning-network
   ```
3. **Update MCP configuration** in `config/mcp_servers.docker.yaml`
4. **Rebuild and restart**

## 📊 Monitoring

### Service Status

```bash
# Check all services
docker-compose ps

# Check resource usage
docker stats

# Check logs
docker-compose logs -f --tail=100
```

### Health Monitoring

All services include health checks that can be monitored:

```bash
# Check health status
docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
```

## 💡 Development vs Docker

### Local Development (Current)
```bash
# Terminal 1: Start API
make api

# Terminal 2: Start MCP server
make demo_mcp_server  

# Terminal 3: Start web client
make web_client
```

### Docker Development
```bash
# Single command starts everything
docker-compose up -d

# View all logs
docker-compose logs -f
```

Both approaches work great - Docker is better for deployment and consistency, while local development is better for rapid iteration and debugging.

This setup provides a robust, scalable foundation for your reasoning agent project that can grow with your needs!
