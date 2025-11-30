# Personal Usage Guide

This guide covers running the Reasoning Agent API for personal use alongside development.

## Two Environments, Running Simultaneously

| Environment | Config | Ports | Purpose |
|-------------|--------|-------|---------|
| **Development** | `.env` | 8000, 8001, 4000, 6006 | Testing, coding, experiments |
| **Personal** | `.env.personal` | 18000, 18001, 14000, 16006 | Actual use, data you want to keep |

**Key differences:**
- **Separate ports**: Both can run at the same time
- **Separate config**: Different API keys, passwords, settings
- **Separate volumes**: Personal uses external volumes that survive `docker compose down -v`

---

## Quick Start

### 1. Create Personal Config

```bash
cp .env.personal.example .env.personal
```

Edit `.env.personal` and fill in:

```bash
# Required - your OpenAI key (can be same as dev)
OPENAI_API_KEY=your-key-here

# Generate and set these (run each command, copy output):
#   uv run python -c "import secrets; print('sk-' + secrets.token_urlsafe(32))"
LITELLM_MASTER_KEY=sk-generated-value-here

#   uv run python -c "import secrets; print(secrets.token_urlsafe(16))"
LITELLM_POSTGRES_PASSWORD=generated-value-here
PHOENIX_POSTGRES_PASSWORD=generated-value-here
REASONING_POSTGRES_PASSWORD=generated-value-here
```

**Optional:**
- `PROMPTS_HOST_PATH` — Path to your prompts directory. Defaults to `./prompts` if not set.

Note: Auth is disabled by default (`REQUIRE_AUTH=false`) for local personal use.

### 2. Configure Filesystem Access (Optional)

The tools-api needs volume mounts to access your filesystem (read files, search code, etc.). Both dev and personal environments share the same override file since they run on the same machine.

If you haven't already set up the override file for dev:

```bash
cp docker-compose.override.yml.example docker-compose.override.yml
```

Edit `docker-compose.override.yml` to add your paths:

```yaml
services:
  tools-api:
    volumes:
      # Read-write (code repos the agent can edit)
      - /Users/yourname/repos:/mnt/read_write/Users/yourname/repos:rw

      # Read-only (reference directories)
      - /Users/yourname/Downloads:/mnt/read_only/downloads:ro
      - /Users/yourname/Documents:/mnt/read_only/documents:ro
```

Skip this step if you don't need filesystem access from the tools-api.

### 3. Setup and Start

```bash
# One-time: create protected volumes
make personal_setup
# Start personal environment
make personal_up
# Generate LiteLLM API keys (first time only)
make personal_litellm_setup
# Copy the generated LITELLM_API_KEY to .env.personal

# Restart to apply the new key
make personal_down && make personal_up
# Run database migrations
make personal_migrate
```

### 4. Start the Client

```bash
# Connect to personal environment (port 18000)
make client_personal

# Or connect to dev environment (port 8000)
make client
```

---

## Port Reference

| Service | Dev Port | Personal Port |
|---------|----------|---------------|
| Reasoning API | 8000 | 18000 |
| Tools API | 8001 | 18001 |
| LiteLLM Dashboard | 4000 | 14000 |
| Phoenix UI | 6006 | 16006 |
| postgres-phoenix | 5432 | 15432 |
| postgres-litellm | 5433 | 15433 |
| postgres-reasoning | 5434 | 15434 |

---

## Backup & Restore

### Backup

```bash
make personal_backup
```

Backups go to `./backups/personal/` by default.

### Backup to Cloud Storage

Set `PERSONAL_BACKUP_DIR` in `.env.personal`:

```bash
# iCloud
PERSONAL_BACKUP_DIR=~/Library/Mobile Documents/com~apple~CloudDocs/backups/reasoning

# Google Drive
PERSONAL_BACKUP_DIR=~/Google Drive/backups/reasoning
```

### Restore

```bash
# List available backups
make personal_restore

# Restore a specific database (follow the printed instructions)
gunzip -c ./backups/personal/reasoning_YYYYMMDD_HHMMSS.sql.gz | \
  docker compose --env-file .env.personal -p personal \
  -f docker-compose.yml -f docker-compose.personal.yml \
  exec -T postgres-reasoning psql -U reasoning_user reasoning
```

### Automated Daily Backups

Add to crontab (`crontab -e`):

```bash
0 2 * * * cd ~/repos/reasoning-agent-api && make personal_backup 2>&1 | logger -t reasoning-backup
```

---

## How Protection Works

### External Volumes

Personal uses **external Docker volumes** that you create manually:

```bash
make personal_setup
# Creates: personal_reasoning_postgres, personal_litellm_postgres, personal_phoenix_postgres
```

These volumes are **not deleted** by `docker compose down -v`. Docker will skip them:

```
Volume personal_reasoning_postgres is external, skipping
```

To actually delete them (careful!):
```bash
docker volume rm personal_reasoning_postgres
```

### What's Stored Where

| Database | Volume | Contents |
|----------|--------|----------|
| postgres-reasoning | `personal_reasoning_postgres` | Your conversations and messages |
| postgres-litellm | `personal_litellm_postgres` | API keys, usage tracking |
| postgres-phoenix | `personal_phoenix_postgres` | LLM traces and observability |

---

## Commands Reference

### Personal Environment

| Command | Description |
|---------|-------------|
| `make personal_setup` | Create external volumes (run once) |
| `make personal_up` | Start all services |
| `make personal_down` | Stop services (data preserved) |
| `make personal_logs` | View service logs |
| `make personal_restart` | Restart all services |
| `make personal_status` | Show containers and volume sizes |
| `make personal_litellm_setup` | Generate LiteLLM API keys |
| `make personal_migrate` | Run database migrations |

### Backup

| Command | Description |
|---------|-------------|
| `make personal_backup` | Backup all databases |
| `make personal_restore` | Show restore instructions |
| `make personal_backup_clean` | Delete backups older than 30 days |

### Client

| Command | Description |
|---------|-------------|
| `make client` | Start client → dev (localhost:8000) |
| `make client_personal` | Start client → personal (localhost:18000) |

---

## Typical Workflow

```bash
# Terminal 1: Personal environment (always running)
make personal_up

# Terminal 2: Development environment (when coding)
make docker_up

# Terminal 3: Client for personal use
make client_personal

# Terminal 4: Client for testing dev changes
make client
```

---

## Disaster Recovery

### Corrupted database

```bash
# 1. Stop services
make personal_down

# 2. Delete the corrupted volume
docker volume rm personal_reasoning_postgres

# 3. Recreate it
docker volume create personal_reasoning_postgres

# 4. Start services
make personal_up

# 5. Restore from backup (see restore instructions above)
```

### New machine

1. Install Docker
2. Clone repo
3. Copy `.env.personal` from old machine (or create new)
4. Copy backup files
5. Run:
   ```bash
   make personal_setup
   make personal_up
   # Restore databases from backups
   ```

---

## Tips

1. **Keep environments separate**: Dev for experiments, personal for real data.

2. **Backup before updates**: Run `make personal_backup` before pulling new code or running migrations.

3. **Different API keys**: Consider using separate OpenAI API keys for dev vs personal to track costs separately.

4. **Monitor Phoenix traces**: Personal Phoenix (port 16006) can grow large. Clean periodically or set up retention.
