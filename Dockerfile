# Multi-platform lightweight Python base image
FROM python:3.9-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and setup.py first for caching layers
COPY requirements.txt setup.py ./

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and artifacts
COPY . .

# Expose target port
EXPOSE 8080

# Run Flask application with Gunicorn WSGI server
CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:8080", "--timeout=120", "app:app"]
