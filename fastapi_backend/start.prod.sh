#!/bin/bash
# Production entrypoint for Cloud Run / any platform that injects $PORT.
# Honors PORT (default 8080 on Cloud Run) and skips the dev hot-reload loop.

set -euo pipefail

PORT="${PORT:-8000}"

# Run any pending migrations before starting the API. This keeps the
# service self-contained for a single-instance deploy. For multi-instance
# Cloud Run revisions, run `alembic upgrade head` from a separate Cloud
# Build step instead.
alembic -c alembic.ini upgrade head || echo "alembic upgrade failed (continuing)"

exec fastapi run app/main.py --host 0.0.0.0 --port "$PORT"
