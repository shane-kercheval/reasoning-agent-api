#!/bin/bash
# Phoenix Data Management Script
# This script provides utilities for managing Phoenix database data

set -e

# Load environment variables if .env exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Default values
POSTGRES_USER=${POSTGRES_USER:-phoenix_user}
PHOENIX_POSTGRES_DB=${PHOENIX_POSTGRES_DB:-phoenix}
BACKUP_DIR="./backups"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if docker compose is running
check_docker() {
    if ! docker compose ps 2>/dev/null | grep -E "postgres.*Up.*healthy|postgres.*running" > /dev/null; then
        echo -e "${RED}Error: PostgreSQL container is not running.${NC}"
        echo "Please run 'make docker_up' first."
        exit 1
    fi
}

# Function to reset Phoenix data
reset_data() {
    echo -e "${YELLOW}WARNING: This will delete all Phoenix trace data!${NC}"
    echo "Press Ctrl+C to cancel, or any other key to continue..."
    read -n 1 -s
    
    check_docker
    
    echo -e "${GREEN}Resetting Phoenix data...${NC}"
    docker compose exec -T postgres psql -U $POSTGRES_USER -d $PHOENIX_POSTGRES_DB -c "DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;" 2>/dev/null || {
        echo -e "${RED}Failed to reset data. Database may not have been initialized yet.${NC}"
        exit 1
    }
    
    echo -e "${GREEN}Phoenix data reset complete.${NC}"
}

# Function to backup Phoenix data
backup_data() {
    check_docker
    
    mkdir -p $BACKUP_DIR
    
    BACKUP_FILE="$BACKUP_DIR/phoenix_backup_$(date +%Y%m%d_%H%M%S).sql"
    
    echo -e "${GREEN}Backing up Phoenix database...${NC}"
    
    if docker compose exec -T postgres pg_dump -U $POSTGRES_USER $PHOENIX_POSTGRES_DB > $BACKUP_FILE; then
        echo -e "${GREEN}Backup saved to: $BACKUP_FILE${NC}"
        echo "Backup size: $(du -h $BACKUP_FILE | cut -f1)"
    else
        echo -e "${RED}Backup failed!${NC}"
        rm -f $BACKUP_FILE
        exit 1
    fi
}

# Function to list backups
list_backups() {
    echo -e "${GREEN}Available backups:${NC}"
    if ls $BACKUP_DIR/phoenix_backup_*.sql 2>/dev/null | head -n 20; then
        echo ""
        echo "Total backups: $(ls $BACKUP_DIR/phoenix_backup_*.sql 2>/dev/null | wc -l)"
    else
        echo -e "${YELLOW}No backups found in $BACKUP_DIR/${NC}"
    fi
}

# Function to restore from backup
restore_data() {
    list_backups
    
    if [ -z "$1" ]; then
        echo ""
        echo "Usage: $0 restore <backup_file>"
        echo "Example: $0 restore ./backups/phoenix_backup_20240120_143022.sql"
        exit 1
    fi
    
    if [ ! -f "$1" ]; then
        echo -e "${RED}Error: Backup file not found: $1${NC}"
        exit 1
    fi
    
    check_docker
    
    echo -e "${YELLOW}WARNING: This will replace all current Phoenix data with the backup!${NC}"
    echo "Backup file: $1"
    echo "Press Ctrl+C to cancel, or any other key to continue..."
    read -n 1 -s
    
    echo -e "${GREEN}Restoring Phoenix database...${NC}"
    
    # First reset the database
    docker compose exec -T postgres psql -U $POSTGRES_USER -d $PHOENIX_POSTGRES_DB -c "DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;" 2>/dev/null
    
    # Then restore from backup
    if cat "$1" | docker compose exec -T postgres psql -U $POSTGRES_USER $PHOENIX_POSTGRES_DB; then
        echo -e "${GREEN}Database restored successfully from: $1${NC}"
    else
        echo -e "${RED}Restore failed!${NC}"
        exit 1
    fi
}

# Function to show database size
show_size() {
    check_docker
    
    echo -e "${GREEN}Phoenix Database Size:${NC}"
    docker compose exec -T postgres psql -U $POSTGRES_USER -d $PHOENIX_POSTGRES_DB -c "
        SELECT 
            pg_database.datname as database,
            pg_size_pretty(pg_database_size(pg_database.datname)) as size
        FROM pg_database
        WHERE datname = '$PHOENIX_POSTGRES_DB';"
}

# Main script logic
case "$1" in
    reset)
        reset_data
        ;;
    backup)
        backup_data
        ;;
    list)
        list_backups
        ;;
    restore)
        restore_data "$2"
        ;;
    size)
        show_size
        ;;
    *)
        echo "Phoenix Data Management Tool"
        echo ""
        echo "Usage: $0 {reset|backup|list|restore|size}"
        echo ""
        echo "Commands:"
        echo "  reset    - Delete all Phoenix trace data (preserves database)"
        echo "  backup   - Create a backup of the Phoenix database"
        echo "  list     - List available backups"
        echo "  restore  - Restore from a specific backup file"
        echo "  size     - Show database size"
        echo ""
        echo "Examples:"
        echo "  $0 reset"
        echo "  $0 backup"
        echo "  $0 restore ./backups/phoenix_backup_20240120_143022.sql"
        exit 1
        ;;
esac