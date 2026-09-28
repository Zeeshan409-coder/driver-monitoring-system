#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  . .venv/bin/activate
  pip install -r requirements.txt
  python scripts/download_samples.py
else
  . .venv/bin/activate
fi
echo "open http://localhost:8000"
python -m uvicorn server.app:app --port 8000
