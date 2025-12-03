#!/bin/bash

# Start FastAPI backend on port 8000 in background
uvicorn app.api:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend on port 8501 (Azure will use this)
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
