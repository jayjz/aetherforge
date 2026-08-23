# Use the official lightweight Python 3.12 image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies first (leverage Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create the logs directory with wide permissions so both containers can write/read
RUN mkdir -p /app/logs && chmod 777 /app/logs

# The default command (can be overridden by docker-compose)
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]