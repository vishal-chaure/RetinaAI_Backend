# Use Python 3.9 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create models directory
RUN mkdir -p models

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} app:app"] 