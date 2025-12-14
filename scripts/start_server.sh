#!/bin/bash
echo "Starting Legal Chatbot API Server..."
echo ""
cd "$(dirname "$0")/.."
python backend/api/server.py

