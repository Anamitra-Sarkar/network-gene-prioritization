# syntax=docker/dockerfile:1

# ---- builder ----
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for building some wheels (scipy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a virtualenv-like layer
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- runtime ----
FROM python:3.11-slim AS runtime

WORKDIR /app

# Minimal runtime system deps (curl for healthcheck, libgomp for scipy/sklearn/torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed site-packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY backend/ ./backend/
COPY data_pipeline/ ./data_pipeline/
COPY pyproject.toml ./pyproject.toml

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Fail-closed release gate: OFF by default (documented safe default).
# Do NOT set MODEL_RELEASE_APPROVED=true here — production must explicitly opt in.
ENV MODEL_RELEASE_APPROVED="false" \
    APPROVED_ARTIFACT_REVISION="" \
    ARTIFACT_DIR="artifacts" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Healthcheck hits the public /healthz endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8000/api/v1/healthz || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
