#!/bin/bash
set -e

# Start FastAPI backend on port 8000 (internal only)
echo "Starting FastAPI backend on port 8000..."
uvicorn app.api:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

echo "Waiting for backend to initialize..."
sleep 5
echo "Backend started (PID: $BACKEND_PID)"

# Start Streamlit on port 10000 (Render's expected port)
echo "Starting Streamlit on port 10000..."
exec streamlit run streamlit_app.py \
    --server.port=10000 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
