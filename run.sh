#!/usr/bin/env bash
# Start PHANTOMLINK: FastAPI serves the API and the built frontend on http://127.0.0.1:8000
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-.venv/bin/python}
if [ ! -f data/processed/domains.csv ]; then "$PY" scripts/prepare_data.py --download; fi
if [ ! -f frontend/dist/index.html ]; then
  (cd frontend && npm install && npm run build)
fi
exec "$PY" -m uvicorn phantomlink.api:app --app-dir backend --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
