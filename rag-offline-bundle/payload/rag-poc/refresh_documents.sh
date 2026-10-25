#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATABASE_DIR="$PROJECT_DIR/rag_db"
DOCUMENTS_DIR="$PROJECT_DIR/documents"
BACKUP_DIR="$PROJECT_DIR/backups"

BACKUP_FILE=""

echo "=== RAG DOCUMENT REFRESH ==="

# Verify the required project files.
if [[ ! -f "$PROJECT_DIR/ingest_documents.py" ]]; then
    echo "ERROR: ingest_documents.py not found."
    exit 1
fi

if [[ ! -d "$DOCUMENTS_DIR" ]]; then
    echo "ERROR: Documents directory not found."
    exit 1
fi

source "$PROJECT_DIR/ragvenv/bin/activate"

DOC_COUNT=$(find "$DOCUMENTS_DIR" -type f -name '*.txt' | wc -l)
DOC_COUNT="${DOC_COUNT//[[:space:]]/}"

if [[ "$DOC_COUNT" -eq 0 ]]; then
    echo "ERROR: No .txt files found. Database was not changed."
    exit 1
fi

echo "Source text files: $DOC_COUNT"

# Check that the existing local database can be opened.
# This also detects the usual case where the RAG application is
# still running and holding the database lock.
if [[ -d "$DATABASE_DIR" ]]; then
    echo "Checking database availability..."

    if ! python - "$DATABASE_DIR" <<'PY'
import sys
from qdrant_client import QdrantClient

client = QdrantClient(path=sys.argv[1])
try:
    client.get_collections()
finally:
    client.close()

print("Database is available.")
PY
    then
        echo "ERROR: Could not open the database."
        echo "Exit the RAG application and try again."
        exit 1
    fi

    # Preserve the existing database before rebuilding it.
    mkdir -p "$BACKUP_DIR"
    BACKUP_FILE="$BACKUP_DIR/rag_db_pre_refresh_$(date +%Y%m%d_%H%M%S)_$$.tar.gz"

    echo "Creating database backup..."
    tar -czf "$BACKUP_FILE" -C "$PROJECT_DIR" rag_db

    echo "Backup saved: $BACKUP_FILE"
fi

restore_database() {
    echo "Restoring the previous database..."

    if [[ -n "$BACKUP_FILE" && -f "$BACKUP_FILE" ]]; then
        rm -rf -- "$DATABASE_DIR"
        tar -xzf "$BACKUP_FILE" -C "$PROJECT_DIR"
        echo "Previous database restored."
    else
        rm -rf -- "$DATABASE_DIR"
        echo "No previous database existed; removed incomplete database."
    fi
}

echo
echo "Rebuilding the documentation database..."

if ! python "$PROJECT_DIR/ingest_documents.py"; then
    echo "ERROR: Document ingestion failed."
    restore_database
    exit 1
fi

# Independently verify that the new collection contains data.
if ! python - "$DATABASE_DIR" <<'PY'
import sys
from qdrant_client import QdrantClient

client = QdrantClient(path=sys.argv[1])

try:
    collection = "text_chunks"

    if not client.collection_exists(collection):
        raise RuntimeError(f"Missing collection: {collection}")

    count = client.count(
        collection_name=collection,
        exact=True,
    ).count

    if count <= 0:
        raise RuntimeError("The new collection is empty.")

    print(f"Verified stored vectors: {count}")
finally:
    client.close()
PY
then
    echo "ERROR: Database verification failed."
    restore_database
    exit 1
fi

# Keep the two newest pre-refresh database backups.
if [[ -d "$BACKUP_DIR" ]]; then
    mapfile -t BACKUPS < <(
        find "$BACKUP_DIR" -maxdepth 1 -type f \
            -name 'rag_db_pre_refresh_*.tar.gz' \
            -printf '%T@ %p\n' |
        sort -nr |
        cut -d' ' -f2-
    )

    for OLD_BACKUP in "${BACKUPS[@]:2}"; do
        rm -f -- "$OLD_BACKUP"
    done
fi

echo
echo "=== DOCUMENT REFRESH SUCCESSFUL ==="
echo "Source files: $DOC_COUNT"
echo "The new database passed verification."
