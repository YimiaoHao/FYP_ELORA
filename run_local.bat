@echo off
setlocal

cd /d %~dp0

REM --- Create venv if not exists ---
if not exist venv\Scripts\python.exe (
  echo [ELORA] Creating virtual environment...
  python -m venv venv
)

REM --- Activate venv ---
call venv\Scripts\activate.bat

REM --- Install deps ---
echo [ELORA] Installing requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM --- Run server ---
echo [ELORA] Starting FastAPI (http://127.0.0.1:8000) ...
uvicorn app.main:app --reload

endlocal