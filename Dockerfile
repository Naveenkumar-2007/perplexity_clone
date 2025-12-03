FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data persistence
RUN mkdir -p /app/workspace_data /app/chroma_db

# Expose ports
EXPOSE 8000 8501

# Create startup script to run both services
RUN echo '#!/bin/bash\n\
uvicorn app.api:app --host 0.0.0.0 --port 8000 &\n\
sleep 5\n\
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true\n\
' > /app/start.sh && chmod +x /app/start.sh

# Run both services
CMD ["/bin/bash", "/app/start.sh"]
