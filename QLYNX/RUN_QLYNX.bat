@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title QLYNX - Anti-Money Laundering, Fraud and Player Protection

echo ========================================
echo    QLYNX Investigator Dashboard
echo ========================================
echo.
if not exist "main.py" (
  echo ERROR: main.py is missing. Extract the complete ZIP first.
  goto :failed
)
if not exist "models\neural_v5\seed_17\quantum\quantum_best.pt" (
  echo ERROR: Model weights are missing. Extract the complete ZIP first.
  goto :failed
)
if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating a Python environment...
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3.11 -m venv .venv 2>nul
    if not exist ".venv\Scripts\python.exe" py -3 -m venv .venv 2>nul
  )
  if not exist ".venv\Scripts\python.exe" python -m venv .venv
  if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Python could not be found. Install Python 3.11 or 3.12 and retry.
    goto :failed
  )
)
set "PYTHON=.venv\Scripts\python.exe"
echo [2/3] Checking dependencies...
"%PYTHON%" -c "import streamlit, pandas, numpy, pyarrow, lightgbm, sklearn, torch, plotly, requests, networkx" >nul 2>nul
if errorlevel 1 (
  echo Installing dependencies. First launch may take several minutes...
  "%PYTHON%" -m pip install --disable-pip-version-check -r requirements-dashboard.txt
  if errorlevel 1 (
    echo ERROR: Dependency installation failed. See pip errors above.
    goto :failed
  )
)
echo [3/3] Starting QLYNX...
echo Streamlit URL: http://localhost:8501
echo Press Ctrl+C to stop the server.
"%PYTHON%" -m streamlit run main.py --server.address 127.0.0.1 --server.port 8501
if errorlevel 1 goto :failed
exit /b 0
:failed
echo.
echo QLYNX could not start. Please keep this window open and review the error.
pause
exit /b 1
