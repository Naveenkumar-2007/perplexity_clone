#!/bin/bash
set -e

echo "Starting FastAPI backend on port 8000..."
uvicorn app.api:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend to start..."
sleep 10

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "ERROR: Backend failed to start"
    exit 1
fi

echo "Backend started successfully (PID: $BACKEND_PID)"
echo "Starting Streamlit frontend on port 8501..."

# Start Streamlit frontend (this will keep the container running)
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
