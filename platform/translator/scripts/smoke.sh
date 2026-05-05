#!/usr/bin/env bash
# Smoke test runner. Reads .env, installs deps in a venv, runs the pipeline.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

if [ -f .env ]; then
    set -a; . ./.env; set +a
fi

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
. .venv/bin/activate
pip install -q -e . >/dev/null
pip install -q pytest pytest-asyncio >/dev/null

echo "--- unit tests ---"
PYTHONPATH=src python -m pytest tests/ -q

PDF="${1:-}"
if [ -z "$PDF" ]; then
    echo "Usage: $0 <pdf_path>" >&2
    exit 2
fi

echo "--- pipeline ---"
python -m sohn_translator "$PDF" --output-dir ./out --log-level INFO
