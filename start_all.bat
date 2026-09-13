@echo off
echo Starting AI Autonomous Data Scientist Agent...
start "Backend (FastAPI + SQL Server)" cmd /k "run_backend.bat"
start "Frontend (Angular 17)" cmd /k "run_frontend.bat"
echo Services launched! Backend at http://localhost:8000 and Frontend at http://localhost:4200
pause
