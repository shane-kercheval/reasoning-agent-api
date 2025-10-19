# Phoenix Arize Integration Guide

This document explains how Phoenix Arize is integrated into the Reasoning Agent API for LLM observability, tracing, and performance monitoring.

## 🔭 What is Phoenix?

Phoenix is an open-source LLM observability platform that helps you:
- **Trace LLM Applications**: Visualize the flow of requests through your AI system
- **Monitor Performance**: Track latency, token usage, and error rates
- **Debug Issues**: Understand where and why failures occur
- **Evaluate Quality**: Assess reasoning steps and tool usage effectiveness

## 🚀 Quick Start

### Running Phoenix with Docker

```bash
# 1. Phoenix is automatically started with Docker Compose
make docker_up

# 2. Access Phoenix UI
open http://localhost:6006

# 3. View traces as you use the API
# Traces appear automatically when you make API calls (coming in Milestone 4)
```

### Data Management

Phoenix stores trace data in PostgreSQL for persistence and scalability. We provide simple commands to manage this data:

```bash
# View database size
make phoenix_backup_data        # Create a backup
make phoenix_reset_data         # Clear all trace data
make phoenix_restore_data       # Restore from backup
```

## 📊 Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐     ┌─────────────┐
│   Browser   │────▶│  Reasoning API   │────▶│   LiteLLM    │────▶│   OpenAI    │
│             │     │   (Port 8000)    │     │   Proxy      │     │     API     │
└─────────────┘     └────────┬─────────┘     │  (Port 4000) │     └─────────────┘
                             │                └──────┬───────┘
                             │                       │
                             │ OTEL Traces           │ OTEL Traces
                             │ (traceparent)         │ (callbacks)
                             │                       │
                             └───────────────────────┘
                                         │
                                         ▼
                              ┌──────────────────┐     ┌──────────────────┐
                              │     Phoenix      │────▶│   PostgreSQL     │
                              │   (Port 6006)    │     │ (postgres-phoenix│
                              │   OTEL Collector │     │   Port 5432)     │
                              │   (Port 4317)    │     │                  │
                              └──────────────────┘     └──────────────────┘
```

**Unified Observability**:
- Reasoning API creates OTEL spans for request handling
- LiteLLM proxy creates OTEL spans for LLM API calls
- Trace context propagated via `traceparent` header
- All traces collected in Phoenix with unified parent-child relationships
- Component metadata enables filtering (reasoning-agent, passthrough, classifier)

## 🔧 Configuration

### Environment Variables

Phoenix configuration is managed through environment variables in `.env`:

```bash
# PostgreSQL Database for Phoenix
PHOENIX_POSTGRES_PASSWORD=phoenix_dev_password123
# POSTGRES_USER=phoenix_user (default in docker-compose.yml)
# PHOENIX_POSTGRES_DB=phoenix (default in docker-compose.yml)

# Phoenix Service (configured in docker-compose.yml)
# PHOENIX_PORT=6006
# PHOENIX_GRPC_PORT=4317
# PHOENIX_SQL_DATABASE_URL=postgresql://phoenix_user:${PHOENIX_POSTGRES_PASSWORD}@postgres-phoenix:5432/phoenix
```

**Note**: Most Phoenix configuration is handled in `docker-compose.yml`. Only `PHOENIX_POSTGRES_PASSWORD` needs to be set in `.env`.

### Docker Services

Phoenix runs as part of the unified observability stack:

1. **PostgreSQL Database** (`postgres-phoenix`)
   - Stores all trace data persistently
   - Port: 5432 (external/internal)
   - Uses volume `phoenix_postgres_data`
   - Automatically initialized by Phoenix
   - Separate from LiteLLM database (postgres-litellm on port 5433)

2. **Phoenix Service** (`phoenix`)
   - Web UI on port 6006
   - OTLP gRPC collector on port 4317
   - Connected to postgres-phoenix for data persistence
   - Receives traces from both Reasoning API and LiteLLM proxy

## 📈 What Gets Traced?

Phoenix captures comprehensive traces of your AI application through unified OTEL integration:

### API Level
- Request/response payloads
- Authentication details
- Total request latency
- Error rates and types
- Routing decisions (passthrough, reasoning, orchestration)

### LiteLLM Proxy Level
- All LLM API calls (reasoning-agent, passthrough, classifier)
- Model parameters (temperature, max_tokens, etc.)
- Token usage and costs
- Virtual key tracking (dev, eval, prod)
- Retry attempts and failures

### Request Routing
- Header-based routing (`X-Routing-Mode`)
- Auto-routing classifier decisions
- Tier 1 passthrough rule evaluation
- Routing latency overhead

### Reasoning Steps
- Individual reasoning step content
- Step-by-step latency breakdown
- Reasoning quality metrics
- Tool identification and selection

### Tool Usage (MCP)
- MCP tool calls and responses
- Tool execution time
- Success/failure rates
- Input/output data

### Component Metadata
All LLM calls are tagged with component metadata:
- `reasoning-agent`: Step generation and synthesis
- `passthrough`: Direct API forwarding
- `routing-classifier`: Auto-routing classification

This enables filtering and analysis by component in Phoenix dashboard.

## 🛠️ Data Management

### Viewing Database Size

```bash
# Using the Makefile
make phoenix_backup_data

# Using the script directly
./scripts/phoenix_data_management.sh size
```

### Creating Backups

```bash
# Creates timestamped backup in ./backups/
make phoenix_backup_data

# List available backups
./scripts/phoenix_data_management.sh list
```

### Resetting Data

```bash
# WARNING: Deletes all trace data
make phoenix_reset_data

# Confirmation required before deletion
```

### Restoring from Backup

```bash
# Show available backups
make phoenix_restore_data

# Restore specific backup
cat ./backups/phoenix_backup_20240120_143022.sql | docker compose exec -T postgres psql -U phoenix_user phoenix
```

## 🔍 Using Phoenix UI

Once Phoenix is running, you can:

1. **View Traces**: See all API calls and their details
2. **Analyze Performance**: Identify slow operations
3. **Debug Errors**: Understand failure patterns
4. **Monitor Usage**: Track token consumption and costs
5. **Test with Playground**: Interactive chat interface for testing

### Phoenix Playground Configuration

To use Phoenix's built-in playground to test your local reasoning API:

1. **Open Phoenix UI**: http://localhost:6006
2. **Navigate to Playground**: Click on the "Playground" tab
3. **Configure API Base URL**: 
   - Set "Base URL" to: `http://reasoning-api:8000/v1`
4. **Test your API**: Send messages and see reasoning steps in action

**Important**: Use `http://reasoning-api:8000/v1` (not `localhost`) when running in Docker, as this is the internal Docker network address for the reasoning API service.

### Trace Filtering

Phoenix provides powerful filtering capabilities:
- Filter by time range
- Search by trace content
- Filter by error status
- Group by operation type

### Performance Analysis

Key metrics available in Phoenix:
- P50/P95/P99 latencies
- Token usage distribution
- Error rate trends
- Tool usage patterns

## 🧪 Development Workflow

### Local Development with Phoenix

```bash
# 1. Start PostgreSQL
docker compose up -d postgres

# 2. Start Phoenix (coming in Milestone 2)
docker compose up -d phoenix

# 3. Run your API locally
make api

# 4. View traces at http://localhost:6006
```

### Testing with Phoenix

Phoenix is invaluable for testing:
- Verify reasoning steps are captured
- Check tool usage is logged correctly
- Ensure error handling works
- Monitor performance during load tests

## 🚨 Troubleshooting

### Common Issues

**Phoenix UI not loading**:
- Check PostgreSQL is running: `docker compose ps postgres`
- Verify Phoenix is healthy: `docker compose ps phoenix`
- Check logs: `docker compose logs phoenix`

**No traces appearing**:
- Ensure API is configured with Phoenix endpoint
- Check OTEL configuration in Python code
- Verify network connectivity between services

**Database connection errors**:
- Check PostgreSQL credentials in `.env`
- Ensure database is initialized
- Verify Docker network is created

### Debug Commands

```bash
# Check Phoenix logs
docker compose logs -f phoenix

# Test PostgreSQL connection
docker compose exec postgres psql -U phoenix_user -d phoenix -c '\dt'

# Verify Phoenix health
curl http://localhost:6006/health
```

## 📚 Additional Resources

- [Phoenix Documentation](https://arize.com/docs/phoenix)
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)
- [Phoenix Tracing Guide](https://arize.com/docs/phoenix/tracing/llm-traces)

## 🎯 Coming Next

### ✅ Milestone 2: Phoenix Service Integration - COMPLETE
- Phoenix container deployment
- Web UI access at port 6006
- Ready for trace collection (Milestone 4)

### Milestone 3: Python Integration
- OTEL instrumentation in API code
- Automatic span creation
- Custom attributes and metrics

### Milestone 4: Advanced Features
- Custom evaluations
- Performance benchmarking
- Alert configuration