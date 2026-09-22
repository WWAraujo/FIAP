FROM python:3.12-slim

# Define working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p data models artifacts logs

# Add src directory to PYTHONPATH
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose port for Streamlit
EXPOSE 8501

# Default command - run Streamlit app
CMD ["streamlit", "run", "src/medical_assistant/interfaces/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
