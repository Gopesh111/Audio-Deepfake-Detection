# Use a slim Python base image for efficiency
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the modular code folders
COPY api/ ./api/
COPY core/ ./core/
COPY main.py .

# Expose the port Cloud Run expects (usually 8080 or 8000)
EXPOSE 8080

# Command to run the scalable REST API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]