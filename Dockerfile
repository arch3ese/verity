FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy package metadata and requirements
COPY requirements.txt pyproject.toml README.md ./

# Install python dependencies and the verity package
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# Copy application source and assets
COPY src/ ./src/
COPY data/ ./data/
COPY examples/ ./examples/

# Set environment defaults
ENV PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000 7860

# Launch Verity Studio (dynamically binding to assigned PORT)
CMD ["sh", "-c", "uvicorn verity.web.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
