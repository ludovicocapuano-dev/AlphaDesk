#!/usr/bin/env bash
# AlphaDesk Dashboard — Startup Script
# Usage: bash dashboard/run.sh [port]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PORT="${1:-5000}"

export ALPHADESK_DB="${PROJECT_DIR}/alphadesk.db"
export DASHBOARD_PORT="$PORT"

# Activate venv if it exists
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

# Install Flask if missing
python -c "import flask" 2>/dev/null || pip install flask

echo "============================================"
echo "  AlphaDesk Dashboard"
echo "  http://localhost:${PORT}"
echo "  DB: ${ALPHADESK_DB}"
echo "============================================"

python "${SCRIPT_DIR}/app.py"
