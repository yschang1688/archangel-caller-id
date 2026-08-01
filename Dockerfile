FROM python:3.10-slim

LABEL maintainer="Archangel Intelligence System"
LABEL description="Anti-Fraud Data Pipeline — FastAPI Detection Service"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY label_1000_dataset.csv .

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run FastAPI with uvicorn
CMD ["uvicorn", "src.api.detection_api:app", "--host", "0.0.0.0", "--port", "8000"]
