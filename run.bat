@echo off
cd /d %~dp0
if not exist venv\Scripts\python.exe (
  echo Virtual environment not found.
  echo Run install.bat first.
  pause
  exit /b
)
venv\Scripts\python.exe app.py
pause
