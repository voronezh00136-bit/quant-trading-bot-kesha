FROM python:3.12-slim

LABEL maintainer="Kesha Trading System"
LABEL description="Quantitative Trading Automation System using OpenClaw framework"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY config/ ./config/
COPY src/ ./src/

# Run as non-root user for security
RUN useradd -m kesha
USER kesha

CMD ["python", "-m", "src.main"]
