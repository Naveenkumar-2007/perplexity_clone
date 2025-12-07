#!/bin/bash
set -e

echo "Starting FastAPI backend on port 8000..."
uvicorn app.api:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Waiting for backend to start..."
sleep 3
echo "Backend started successfully (PID: $BACKEND_PID)"

echo "Starting Streamlit frontend on port 10000..."
exec streamlit run streamlit_app.py --server.port 10000 --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false
