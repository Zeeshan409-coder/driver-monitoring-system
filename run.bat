@echo off
cd /d "%~dp0"
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv
  call .venv\Scripts\activate.bat
  pip install -r requirements.txt
  python scripts\download_samples.py
) else (
  call .venv\Scripts\activate.bat
)
start "" http://localhost:8000
python -m uvicorn server.app:app --port 8000
