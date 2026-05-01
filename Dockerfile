# ── Build Stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System deps for PyMuPDF and faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create data directories
RUN mkdir -p data/uploads data/faiss_index

# ── Runtime ────────────────────────────────────────────────────────────────────
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV API_RELOAD=false

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
