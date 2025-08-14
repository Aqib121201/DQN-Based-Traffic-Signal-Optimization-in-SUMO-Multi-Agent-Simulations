# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg2 \
    software-properties-common \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install SUMO
RUN wget -O - https://sumo.dlr.de/releases/1.18.0/sumo-1.18.0.tar.gz | tar -xz \
    && cd sumo-1.18.0 \
    && ./configure --prefix=/usr/local \
    && make \
    && make install \
    && cd .. \
    && rm -rf sumo-1.18.0

# Set SUMO environment variables
ENV SUMO_HOME=/usr/local/share/sumo
ENV PATH=$PATH:/usr/local/bin
ENV PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.9/site-packages

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/external \
    models visualizations logs tests report app

# Set permissions
RUN chmod +x run_pipeline.py

# Expose port for Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
