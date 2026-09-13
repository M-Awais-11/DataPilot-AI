@echo off
title Autonomous Data Scientist Agent - Backend
echo ===================================================
echo Starting AI Autonomous Data Scientist Backend
echo Database: AWAIS\SQLEXPRESS / AI_Autonomous_Data_Scientist_Agent
echo LLM: OpenRouter (openai/gpt-4o-mini)
echo ===================================================

cd backend
call venv\Scripts\activate

echo Initializing database tables in SQL Server...
python init_db.py

echo Starting FastAPI server at http://localhost:8000 ...
uvicorn app.main:app --reload --port 8000 --host 0.0.0.0
pause
