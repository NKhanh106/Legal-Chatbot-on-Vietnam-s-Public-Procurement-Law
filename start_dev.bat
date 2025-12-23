@echo off
REM Script để khởi động cả backend và frontend cùng lúc trên Windows
REM Sử dụng start để chạy mỗi process trong cửa sổ riêng

echo ============================================================
echo   LEGAL CHATBOT - DEVELOPMENT SERVER
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python.
    pause
    exit /b 1
)

REM Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found! Please install Node.js.
    pause
    exit /b 1
)

echo [INFO] Starting Backend Server...
echo [INFO] Backend URL: http://localhost:5000
echo.

REM Start backend in new window
start "Legal Chatbot - Backend" cmd /k "cd /d %~dp0backend\api && python server.py"

REM Wait a bit for backend to start
timeout /t 3 /nobreak >nul

echo [INFO] Starting Frontend Dev Server...
echo [INFO] Frontend URL: http://localhost:5173
echo.

REM Check if node_modules exists
if not exist "frontend\node_modules" (
    echo [INFO] Installing frontend dependencies...
    cd frontend
    call npm install
    cd ..
)

REM Start frontend in new window
start "Legal Chatbot - Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ============================================================
echo   Both servers are starting...
echo ============================================================
echo.
echo Backend:  http://localhost:5000
echo Frontend: http://localhost:5173
echo.
echo [INFO] Two windows will open - one for backend, one for frontend
echo [INFO] Close both windows to stop the servers
echo.
pause

