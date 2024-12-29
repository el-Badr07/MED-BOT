# Use Python base image
FROM python:3.11.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies including PDF tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV BASE_URL=http://host.docker.internal:11434

#ENV STREAMLIT_SERVER_PORT=8501

# Create directories for persistent data
RUN mkdir -p /app/data/chroma_db

# Expose Streamlit port
EXPOSE 8501
EXPOSE 11434

# Create and switch to non-root user
RUN useradd -m appuser
USER appuser

# Run Streamlit
CMD ["streamlit", "run", "--server.port=8501","LLM.py"]

#docker run -e BASE_URL=http://host.docker.internal:11434 -p 8501:8501 -p 11434:11434 new


