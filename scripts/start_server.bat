@echo off
echo Starting Legal Chatbot API Server...
echo.
cd /d %~dp0..
python backend/api/server.py
pause

