#!/bin/bash
# Script để khởi động cả backend và frontend cùng lúc trên Linux/Mac

echo "============================================================"
echo "  LEGAL CHATBOT - DEVELOPMENT SERVER"
echo "============================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found! Please install Python3."
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "[ERROR] Node.js not found! Please install Node.js."
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "[INFO] Starting Backend Server..."
echo "[INFO] Backend URL: http://localhost:5000"
echo ""

# Start backend in background
cd backend/api
python3 server.py &
BACKEND_PID=$!
cd "$SCRIPT_DIR"

# Wait a bit for backend to start
sleep 3

echo "[INFO] Starting Frontend Dev Server..."
echo "[INFO] Frontend URL: http://localhost:5173"
echo ""

# Check if node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    echo "[INFO] Installing frontend dependencies..."
    cd frontend
    npm install
    cd "$SCRIPT_DIR"
fi

# Start frontend in background
cd frontend
npm run dev &
FRONTEND_PID=$!
cd "$SCRIPT_DIR"

echo ""
echo "============================================================"
echo "  Both servers are starting..."
echo "============================================================"
echo ""
echo "Backend:  http://localhost:5000"
echo "Frontend: http://localhost:5173"
echo ""
echo "[INFO] Press Ctrl+C to stop both servers"
echo ""

# Handle Ctrl+C
trap "echo ''; echo '[INFO] Shutting down servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo '[INFO] Servers stopped. Goodbye!'; exit" INT TERM

# Wait for processes
wait

